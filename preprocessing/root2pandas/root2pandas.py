import numpy as np
import pandas as pd
import uproot as root
import re
import glob
import os
import shutil
import matplotlib as mpl
#if os.environ.get('DISPLAY','') == '':
#    print('no display found. Using non-interactive Agg backend')
mpl.use('Agg')
import matplotlib.pyplot as plt
import base64
import ROOT as r00t

import preprocessing_utils as pputils

class EventCategories:
    def __init__(self):
        self.categories = {}

    def addCategory(self, name, selection = None):
        self.categories[name] = selection

    def getCategorySelections(self):
        selections = []
        for cat in self.categories:
            if self.categories[cat]:
                selections.append(self.categories[cat])
        return selections

class Sample:
    def __init__(self, sampleName, ntuples, categories, selections = None, MEMs = None, ownVars = [], even_odd = False):
        self.sampleName = sampleName
        self.ntuples    = ntuples
        self.selections = selections
        self.categories = categories
        self.MEMs       = MEMs
        self.ownVars    = ownVars
        self.even_odd   = even_odd
        self.evenOddSplitting()

    def printInfo(self):
        print("\nHANDLING SAMPLE {}\n".format(self.sampleName))
        print("\tntuples: {}".format(self.ntuples))
        print("\tselections: {}".format(self.selections))

    def evenOddSplitting(self):
        if self.even_odd:
            if self.selections:
                self.selections += " and (Evt_Odd == 1)"
            else:
                self.selections = "(Evt_Odd == 1)"

class ImageConfig():
    ''' configuration of image/2dhist data written to h5 file for CNNs '''
    def __init__(self, channels, xRange, yRange, imageSize, rotation=None, logNorm=False, x="Eta", y="Phi", ):
        # image size with [x,y,z] size
        self.x          = x
        self.y          = y
        self.z          = channels # a lists of strings like ["Jet_Pt", "Electron_Pt"]
        self.imageSize  = imageSize # array like [12,34]
        self.xRange     = xRange
        self.yRange     = yRange
        self.rotation   = rotation

        self.logNorm = logNorm

        # check if rotation mode is supported
        if self.rotation != None and self.rotation != "MaxJetPt" and self.rotation != "ttbar_toplep" and self.rotation != "sphericity_ev1" and self.rotation != "sphericity_ev2" and self.rotation != "sphericity_ev3":
            print("ImageConfig: the rotation-mode "+str(self.rotation)+" is not supported.")
            exit()


        # generates a list of the form [["Jet_Eta", "Jet_Phi", "Jet_Pt"], ["Electron_Eta", "Electron_Phi", "Electron_Pt"]]
        self.images = []
        for channel in channels:
            m=re.match("(\w+)\[(\d+)\-(\d+)\]", channel)
            if m==None:
                ch=re.split("_", channel)
                if len(ch)!=2:
                    print("The string '"+channel+"' has more then one _ symbol. Exit.")
                    exit()
                self.images.append([ch[0]+"_"+x,ch[0]+"_"+y,channel])
            else:
                ch=re.split("_", m.group(1))
                if len(ch)!=2:
                    print("The string '"+channel+"' has more then one _ symbol. Exit.")
                    exit()
                indexstart=m.group(2)
                indexend=m.group(3)
                j=channels.index(channel)
                self.images.append([])
                for i in range(int(indexstart),int(indexend)+1):
                    self.images[j].append([ch[0]+"_"+x+"["+str(i)+"]",ch[0]+"_"+y+"["+str(i)+"]",m.group(1)+"["+str(i)+"]"])

        # flattens a list of lists of lists
        self.variables = []
        for image in self.images:
            for v in image:
                if isinstance(v,list):
                    for v2 in v:
                        self.variables.append(v2)
                else:
                    self.variables.append(v)

        if self.rotation == "ttbar_toplep":
            self.variables.append("Reco_ttbar_toplep_phi")

        if self.rotation != "sphericity_ev1" or self.rotation != "sphericity_ev2" or self.rotation != "sphericity_ev3":
            LeptonVars=["TightLepton_Pt", "TightLepton_Eta", "TightLepton_Phi", "TightLepton_M"]
            METVars=["Evt_MET_Phi", "Evt_MET"] #"Evt_MET_Pt" might also be here, but it will be included via baseselection. ommitong it here for temporariy bugfix, so it won't be droped from dataframe before baseselection
            JetVars=[]
            for i in range(int(indexstart),int(indexend)+1):
                JetVars.append("Jet_M["+str(i)+"]")
            for v in LeptonVars+METVars+JetVars:
                self.variables.append(v)



class Dataset:
    def __init__(self, outputdir, tree='MVATree', naming='', addMEM=False, maxEntries=50000, varName_Run='Evt_Run', varName_LumiBlock='Evt_Lumi', varName_Event='Evt_ID'):
        # settings for paths
        self.outputdir = outputdir
        self.naming = naming
        self.tree = tree
        self.varName_Run = varName_Run
        self.varName_LumiBlock = varName_LumiBlock
        self.varName_Event = varName_Event
        
        self.timer_general=r00t.TStopwatch()
        self.timer_yan=r00t.TStopwatch()
        self.timer_hist=r00t.TStopwatch()
        self.timer_update=r00t.TStopwatch()



        # generating output dir
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

        # settings for dataset
        self.addMEM     = addMEM
        self.maxEntries = int(maxEntries)

        # default values for some configs
        self.baseSelection  = None
        self.samples        = {}
        self.variables      = []


    def addBaseSelection(self, selection):
        self.baseSelection = selection

    def addSample(self, **kwargs):
        print("adding sample: "+str(kwargs["sampleName"]))
        self.samples[kwargs["sampleName"]] = Sample(**kwargs)


    # ====================================================================
    # variable handling
    def addVariables(self, variables):
        print("adding {} variables.".format(len(variables)))
        self.variables += variables
        self.variables = list(set(self.variables))

        # mem variable is not in ntuples so remove it from list and add it via mem dataframes
        if "memDBp" in self.variables: self.variables.remove("memDBp")

    def addAllVariablesNoIndex(self):
        ''' open up a root file and figure out variables automatically
            dont consider indices separately, write them as subentry '''
        test_sample = self.samples[list(self.samples.keys())[0]]
        test_file = list(glob.glob(test_sample.ntuples))[0]
        print("using test file {} to figure out variables.".format(test_file))
        with root.open(test_file) as f:
            tree = f[self.tree]
            df = tree.pandas.df()
            variables = list(df.columns)

        self.addVariables(variables)

    def removeVariables(self, variables):
        n_removed = 0
        for v in variables:
            if v in self.variables:
                self.variables.remove(v)
                n_removed += 1
        print("removed {} variables from list.".format(n_removed))

    def gatherTriggerVariables(self):
        # search for all trigger strings
        self.trigger = []

        # search in base selection string
        if self.baseSelection:
            self.trigger.append(self.baseSelection)

        for key in self.samples:
            # collect variables for specific samples
            own_variables = []

            # search in additional selection strings
            if self.samples[key].selections:
                own_variables += self.searchVariablesInTriggerString( self.samples[key].selections )
            # search in category selections
            categorySelections = self.samples[key].categories.getCategorySelections()
            for selection in categorySelections:
                own_variables += self.searchVariablesInTriggerString( selection )
            # save list of variables
            self.samples[key].ownVars = [v for v in list(set(own_variables)) if not v in self.variables]

        # list of triggers
        self.trigger = list(set(self.trigger))

        # scan trigger strings for variable names
        self.triggerVariables = []
        for triggerstring in self.trigger:
            self.triggerVariables += self.searchVariablesInTriggerString( triggerstring )

        self.triggerVariables = list(set(self.triggerVariables))

        # select variables that only appear in triggerVariables to remove them before saving the final dataframes
        self.removedVariables = [v for v in self.triggerVariables if not v in self.variables]

        # add trigger variables to variable list
        self.addVariables(self.triggerVariables)

    def searchVariablesInTriggerString(self, string):
        # split trigger string into smaller bits
        splitters = [")", "(", "==", ">=", ">=", ">", "<", "="]

        candidates = string.split(" ")
        for splt in splitters:
            candidates = [item for c in candidates for item in c.split(splt)]

        # remove some entries
        remove_entries = ["", "and", "or", "abs"]
        for entry in remove_entries:
            candidates = [c for c in candidates if not c == entry]

        # remove numbers
        candidates = [c for c in candidates if not c.replace(".","",1).isdigit()]

        # the remaining candidates should be variables
        return candidates

    def searchVectorVariables(self):
        # list for variables
        variables = []
        # dictionary for vector variables
        vector_variables = {}

        # loop over variables in list
        for var in self.variables:
            # search for index in name (dummyvar[index])
            found_vector_variable = re.search("\[\d+?\]$", var)
            # append variable to list if not a vector variable
            if not found_vector_variable:
                variables.append(var)
                continue

            # handle vector variable
            index = found_vector_variable.group(0)
            var_name = var[:-len(index)]
            var_index = int(index[1:-1])

            # add variable with index to vector_variables dictionary
            if var_name in vector_variables:
                vector_variables[var_name].append( var_index )
            else:
                vector_variables[var_name] = [var_index]

        self.variables = variables
        self.vector_variables = vector_variables

    # ========================== CNN SPECIFIC STUFF ===============================

    def yan_2dhist(self, ImageConfig, image, df, draw=False, phi0=0):
        ''' create 2d histogram of event '''
        varname_x      = image[0]
        varname_y      = image[1]
        varname_weight = image[2]

        #print("Within 2dhist(): Bins: "+str(ImageConfig.imageSize))
        #print("Range: "+str([ImageConfig.xRange, ImageConfig.yRange]))
        #print(df[varname_weight])
        #print("processing "+varname_weight+"-layer to become a 2d histogram")

        H, _, _ = np.histogram2d(
            x       = np.array(df[varname_x]),
            y       = self.phi_rotation(np.array(df[varname_y]), phi0), 
            bins    = ImageConfig.imageSize,
            range   = [ImageConfig.xRange, ImageConfig.yRange],
            weights = np.array(df[varname_weight]))
        
        #print(H.shape, H.size)

        if ImageConfig.logNorm:
            H = np.where(H > 1, np.log(H), 0)

        if draw:
            plt.figure(figsize=(8,10))
            #need to transpose H and set origin="lower" (default is "upper") in imshow() for the drawing of the 2d array to be the same/correct orientation as with plt.hist2d()
            plt.imshow(H.T, extent=[ImageConfig.xRange[0], ImageConfig.xRange[1], ImageConfig.yRange[0], ImageConfig.yRange[1]], 
                aspect = 'equal', interpolation="none", origin="lower", cmap="Blues")
            plt.xlabel(varname_x)
            plt.ylabel(varname_y)
            plt.title(varname_weight)
            plt.tight_layout()
            #plt.savefig(self.outputdir+"/"+varname_weight+"_100nominal_tree_Blues.pdf")

            #plt.figure(figsize=(8,10))
            #h,_,_,_=plt.hist2d(
            #    x       = df[varname_x],
            #    y       = df[varname_y],
            #    weights = df[varname_weight],
            #    range   = [Image_Config.xRange, Image_Config.yRange],
            #    bins    = Image_Config.imageSize)
            #plt.xlabel(varname_x)
            #plt.ylabel(varname_y)
            #print(h)
            #print(h.size)
            #print(h.shape)
            
            #plt.show()
            exit()
        #print(H)
        return H

    def phi_rotation(self, phi_array, phi0):
        phi_array_rotated=[]
        for phi in phi_array:
            phi = phi - phi0
            if phi < -np.pi:
                phi =  2*np.pi+phi
            if phi >= np.pi:
                phi = -2*np.pi+phi
            phi_array_rotated.append(phi)
        return phi_array_rotated

    def getSp(self, df):
        nJets = df["N_Jets"]
       
        lepton = r00t.TLorentzVector()
        met = r00t.TLorentzVector()

        lepton.SetPtEtaPhiM(df["TightLepton_Pt"], df["TightLepton_Eta"], df["TightLepton_Phi"], df["TightLepton_M"])
        met.SetPtEtaPhiM(df["Evt_MET_Pt"], 0, df["Evt_MET_Phi"], df["Evt_MET_Pt"])
        jets=[]
        for i in range(nJets):
            jeti=r00t.TLorentzVector()
            jeti.SetPtEtaPhiM(df["Jet_Pt["+str(i)+"]"], df["Jet_Eta["+str(i)+"]"], df["Jet_Phi["+str(i)+"]"], df["Jet_M["+str(i)+"]"])
            jets.append(jeti)

        mxx = lepton.Px()*lepton.Px() + met.Px()*met.Px()
        myy = lepton.Py()*lepton.Py() + met.Py()*met.Py()
        mzz = lepton.Pz()*lepton.Pz() + met.Pz()*met.Pz()
        mxy = lepton.Px()*lepton.Py() + met.Px()*met.Py()
        mxz = lepton.Px()*lepton.Pz() + met.Px()*met.Pz()
        myz = lepton.Py()*lepton.Pz() + met.Py()*met.Pz()

        for jet in jets:
            mxx += jet.Px()*jet.Px()
            myy += jet.Py()*jet.Py()
            mzz += jet.Pz()*jet.Pz()
            mxy += jet.Px()*jet.Py()
            mxz += jet.Px()*jet.Pz()
            myz += jet.Py()*jet.Pz()       
        
        summe = mxx + myy + mzz
        mxx /= summe
        myy /= summe
        mzz /= summe
        mxy /= summe
        mxz /= summe
        myz /= summe

        tensor=r00t.TMatrix(3,3)
        tensor[0][0] = mxx
        tensor[1][1] = myy
        tensor[2][2] = mzz
        tensor[0][1] = mxy
        tensor[1][0] = mxy
        tensor[0][2] = mxz
        tensor[2][0] = mxz
        tensor[1][2] = myz
        tensor[2][1] = myz
        eigenval=r00t.TVector(3)
        eigenvec=tensor.EigenVectors(eigenval)

        sphericity  = 3.0*(eigenval(1)+eigenval(2))/2.0
        aplanarity  = 3.0*eigenval(2)/2.0
        tsphericity = 2.0*eigenval(1)/(eigenval(1)+eigenval(0))
        
        #tensor.Print()
        #eigenval.Print()
        #eigenvec.Print()
        ev31=eigenvec[0][2]
        ev32=eigenvec[1][2]
        ev33=eigenvec[2][2]
        #print(ev31,ev32,ev33)
        
        #print(aplanarity)
        #print(np.float(df["Evt_aplanarity"]))
        #exit()
        if(eigenvec[0][0]==0):
            ev1_phi=np.sign(eigenvec[1][0])*0.5*np.pi
        else:
            ev1_phi=np.arctan(eigenvec[1][0]/eigenvec[0][0])
        
        if(eigenvec[0][1]==0):
            ev2_phi=np.sign(eigenvec[1][1])*0.5*np.pi
        else:
            ev2_phi=np.arctan(eigenvec[1][1]/eigenvec[0][1])
        
        if(eigenvec[0][2]==0):
            ev3_phi=np.sign(eigenvec[1][2])*0.5*np.pi
        else:
            ev3_phi=np.arctan(eigenvec[1][2]/eigenvec[0][2])

        #print(ev1_phi, ev2_phi, ev3_phi)
        #exit()
        return([ev1_phi, ev2_phi, ev3_phi])

        

    # ====================================================================

    def runPreprocessing(self, Image_Config=None):
        # add variables as configured in the Image_Config
        if Image_Config is not None:
            self.addVariables(Image_Config.variables)

        # add variables for triggering and event category selection
        self.gatherTriggerVariables()

        # search for vector variables in list of variables and handle them separately
        self.searchVectorVariables()

        print("LOADING {} VARIABLES IN TOTAL.".format(len(self.variables)))
        # remove old files
        #self.removeOldFiles()
        self.renameOldFiles()

        if self.addMEM:
            # generate MEM path
            self.memPath = self.outputdir + "/MEM/"
            # remove old mem files
            old_mem_files = glob.glob(self.memPath+"/*.h5")
            for f in old_mem_files:
                os.remove(f)
            if not os.path.exists(self.memPath):
                os.makedirs(self.memPath)

        sampleList = []

        # start loop over all samples to preprocess them
        for key in self.samples:
            # include own variables of the sample
            self.addVariables( self.samples[key].ownVars )

            # process the sample
            self.processSample(Image_Config=Image_Config,

              sample = self.samples[key],

              varName_Run       = self.varName_Run,
              varName_LumiBlock = self.varName_LumiBlock,
              varName_Event     = self.varName_Event,
            )

            # remove the own variables
            self.removeVariables( self.samples[key].ownVars )
            pputils.createSampleList(sampleList, self.samples[key])
            print("done.")
        # write file with preprocessed samples
        pputils.createSampleFile(self.outputdir, sampleList)

        # handle old files
        self.handleOldFiles()

    def processSample(self, sample, varName_Run, varName_LumiBlock, varName_Event, Image_Config=None):
        # print sample info
        sample.printInfo()
        timer=r00t.TStopwatch()
        timer.Start(False)

        # collect ntuple files
        ntuple_files = sorted(glob.glob(sample.ntuples))

        # collect mem files
        if self.addMEM:
            mem_files = glob.glob(sample.MEMs)
            mem_df = self.generateMEMdf(mem_files, sample.sampleName)

        # initialize loop over ntuple files
        n_entries = 0
        concat_df = pd.DataFrame()
        n_files = len(ntuple_files)
        self.timer_general.Start(False)
        # loop over files
        for iFile, f in enumerate(ntuple_files):
            print("({}/{}) loading file {}".format(iFile+1,n_files,f))
            #if(iFile+1==4):#yan makefast testing
            #    break

            # open root file
            with root.open(f) as rf:
                # get TTree
                try:
                    tree = rf[self.tree]
                except:
                    print("could not open "+str(self.tree)+" in ROOT file")
                    continue

            if tree.numentries == 0:
               print(str(self.tree)+" has no entries - skipping file")
               continue

            # convert to dataframe
            df = tree.pandas.df(self.variables)

            # delete subentry index
            try: df = df.reset_index(1, drop = True)
            except: None

            # handle vector variables, loop over them
            for vecvar in self.vector_variables:

                # load dataframe with vector variable
                vec_df = tree.pandas.df(vecvar)

                # loop over inices in vecvar list
                for idx in self.vector_variables[vecvar]:

                    # slice the index
                    idx_df = vec_df.loc[ (slice(None), slice(idx,idx)), :]
                    idx_df = idx_df.reset_index(1, drop = True)

                    # define name for column in df
                    col_name = str(vecvar)+"["+str(idx)+"]"

                    # initialize column in original dataframe
                    df[col_name] = 0.
                    # append column to original dataframe
                    df.update( idx_df[vecvar].rename(col_name) )

            # apply event selection
            df = self.applySelections(df, sample.selections)
            #print(df)
            self.timer_general.Stop()
            self.timer_yan.Start(False)
            # generate 2d histogram if ImageConfig was passed
            # ===============================================
            if Image_Config is not None:
                print("*"*50)
                print("processing data from file for use with CNN")

                #initialize colums for the matrices/histograms
                #for layer_name in Image_Config.z:
                #    df[layer_name+"_Hist"] = 'AAAAAAAAAAA='

                #loop over the event ids
                evtids=df["Evt_ID"]
                #evt=evtids
                #print(len(evtids))
                #print(df.count)
                #print(len(df["Evt_MET_Pt"]))
                #print(df)
                #for x in df.columns: print x

                H_List_Dict={z:pd.Series() for z in Image_Config.z}
                #print(Image_Config.images)

                for ievt, evt in enumerate(evtids):
                    #if ievt>5:
                    #    break
                    #print(ievt)

                    df_tmp=df.loc[df["Evt_ID"]==evt]
                    entry=df_tmp.index[0]
                    #for x in df_tmp.columns: print x
                    #print(df_tmp)
                    #exit()

                    #calculate phi0, which is used to shift every entry in histogramm
                    phi0=0
                    if Image_Config.rotation=="MaxJetPt":
                        image = Image_Config.images[0] #assuming JetPt is at the first color channel
                        img = image[0] #first in the list is greatest jet
                        phi0 =np.float(np.array(df_tmp[img[1]])) #second entry is phi value of jet
                    if Image_Config.rotation=="ttbar_toplep":
                        phi0 = np.float(np.array(df_tmp["Reco_ttbar_toplep_phi"]))
                    if Image_Config.rotation== "sphericity_ev1" or Image_Config.rotation== "sphericity_ev2" or Image_Config.rotation== "sphericity_ev3":
                        ev1_phi,ev2_phi,ev3_phi=self.getSp(df_tmp)
                        if Image_Config.rotation== "sphericity_ev1":
                            phi0 = ev1_phi
                        if Image_Config.rotation== "sphericity_ev2":
                            phi0 = ev2_phi
                        if Image_Config.rotation== "sphericity_ev3":
                            phi0 = ev3_phi

                    #loop over the differet "color" channels of the image
                    for image in Image_Config.images:

                        #print("index: "+str(pd.Index(evtids).get_loc(evt)))
                        self.timer_hist.Start(False)
                        # for sth like "Jet_Pt[0-8]" this if-statement is true, while its not for "Jet_Pt"
                        if isinstance(image[0], list):
                            H=np.zeros(Image_Config.imageSize)
                            #df_tmp2=pd.DataFrame(columns=["x","y","z"])
                            for img in image:
                                #df_tmp2=df_tmp2.append({'x':float(df_tmp[img[0]]) , 'y':float(df_tmp[img[1]]), 'z':float(df_tmp[img[2]])} , ignore_index=True)
                                H+=self.yan_2dhist(Image_Config, img, df_tmp, phi0=phi0)
                            #H=self.yan_2dhist(Image_Config, ["x","y","z"], df_tmp2, phi0=phi0)   
                        else: 
                            H=self.yan_2dhist(Image_Config, image, df_tmp, phi0=phi0)
                        
                        self.timer_hist.Stop()

                        #col_name=Image_Config.z[Image_Config.images.index(image)]+"_Hist"
                        H=base64.b64encode(np.ascontiguousarray(H))
                        self.timer_update.Start(False)
                        #df.update(pd.DataFrame({col_name:[H]}, index=[pd.Index(evtids).get_loc(evt)]))
                        z=Image_Config.z[Image_Config.images.index(image)]
                        H=pd.Series(H,index=[entry])
                        H_List_Dict[z]=H_List_Dict[z].append(H)
                        self.timer_update.Stop()
                    #break #for debugging, eg to see if saving to file works
                
                #print(H_List)
                #add matrices/histograms to df
                #for layer_name in Image_Config.z:
                #    df[layer_name+"_Hist"] = 'AAAAAAAAAAA='
                #print(len(df), len(H_List))
                self.timer_update.Start(False)
                for key in H_List_Dict:
                    df[key+"_Hist"]=H_List_Dict[key]
                self.timer_update.Stop()
                #print(df)
                #exit()
                    
                #print(df["Jet_Pt[0-16]_Hist"])
                df=df.drop(columns=Image_Config.variables)
                #print(df)

            self.timer_yan.Stop()
            self.timer_general.Start(False) 

            # add to list of dataframes
            if concat_df.empty: concat_df = df
            else: concat_df = concat_df.append(df)

            # count entries so far
            n_entries += df.shape[0]
            #print("n_entries: "+str(n_entries))

            # if number of entries exceeds max threshold, add labels and mem and save dataframe
            if (n_entries > self.maxEntries or f == ntuple_files[-1]):
                print("*"*50)
                print("max entries reached ...")

                # add class labels
                concat_df = self.addClassLabels(concat_df, sample.categories.categories) #WARNING: Try using .loc[row_indexer,col_indexer] = value instead

                # add indexing
                concat_df.set_index([varName_Run, varName_LumiBlock, varName_Event], inplace=True, drop=True)

                # add MEM variables
                if self.addMEM:
                   concat_df = self.addMEMVariable(concat_df, mem_df)

                # remove trigger variables
                concat_df = self.removeTriggerVariables(concat_df)

                # write data to file
                #print("Lets Save...")
                #for x in df.columns: print x
                #print(df)
                #print(df["Jet_Pt[0-2]_Hist"])
                self.createDatasets(concat_df, sample.categories.categories, Image_Config)
                print("*"*50)

                # reset counters
                n_entries = 0
                concat_df = pd.DataFrame()

            #yan debugging
            #print(df)
            
            self.timer_general.Stop()
            print("time spent in general processing: "+str(np.round(self.timer_general.RealTime(),2)))
            print("time spent in picture processing: "+str(np.round(self.timer_yan.RealTime(),2)))
            print("        -> in 2D hist processing: "+str(np.round(self.timer_hist.RealTime(),2)))
            print("        -> in updating processing: "+str(np.round(self.timer_update.RealTime(),2)))

            #break #only run for one file
            
            
        print("time processing sample "+str(round(timer.RealTime(),4)))

    # ====================================================================

    def generateMEMdf(self, files, sampleName):
        ''' generate and load mem lookuptable '''
        memVariables = ["event", "lumi", "run", "mem_p"]
        outputFile = self.memPath+"/"+sampleName+"_MEM.h5"
        print("-"*50)
        for f in files:
            print("loading mem file "+str(f))
            # open root file
            with root.open(f) as rf:
                # get tree
                tree = rf["tree"]

                # convert tree to df but only extract the variables needed
                df = tree.pandas.df(memVariables)

                # set index
                df.set_index(["run", "lumi", "event"], inplace = True, drop = True)

                # save data
                with pd.HDFStore(outputFile, "a") as store:
                    store.append("MEM_data", df, index = False)
                del df

        # load the generated MEM file
        with pd.HDFStore(outputFile, "r") as store:
            df = store.select("MEM_data")
        print("-"*50)

        return df

    def applySelections(self, df, sampleSelection):
        if self.baseSelection:
            df = df.query(self.baseSelection)
        if sampleSelection:
            df = df.query(sampleSelection)

        return df

    def addClassLabels(self, df, categories):
        print("adding class labels to df ...")
        split_dfs = []
        for key in categories:
            if categories[key]:
                tmp_df = df.query(categories[key])
            else:
                tmp_df = df
            tmp_df["class_label"] = pd.Series([key]*tmp_df.shape[0], index = tmp_df.index)
            split_dfs.append(tmp_df)

        # concatenate the split dataframes again
        df = pd.concat(split_dfs)
        return df

    def addMEMVariable(self, df, memdf):
        print("adding MEM to dataframe ...")
        # create variable with default value
        df["memDBp"] = pd.Series([-1]*df.shape[0], index = df.index)

        # add mem variable
        df.update( memdf["mem_p"].rename("memDBp") )

        # check if some mems could not be set
        if not df.query("memDBp == -1").empty:
            print("ATTENTION: SOME ENTRIES COULD NOT FIND A MATCHING MEM - SET TO -1")
            entries_before = df.shape[0]
            df = df.query("memDBp != -1")
            entries_after = df.shape[0]
            print("    lost {}/{} events".format(entries_before-entries_after, entries_before))
            print("    we will only save events with mem...")
        return df

    def removeTriggerVariables(self, df):
        df.drop(self.removedVariables, axis = 1, inplace = True)
        return df

    def createDatasets(self, df, categories, Image_Config=None):
        for key in categories:
            outFile = self.outputdir+"/"+key+"_"+self.naming+".h5"

            # create dataframe for category
            cat_df = df.query("(class_label == \""+str(key)+"\")")
            n_events = cat_df.shape[0]
            print("creating dataset for class label {} with {} entries".format(key, n_events))

            with pd.HDFStore(outFile, "a") as store:
                store.append("data", cat_df, index = False)

            if Image_Config is not None:
                meta_info_dict = {"input_shape": Image_Config.imageSize, "n_events": n_events}
                meta_info_df = pd.DataFrame.from_dict( meta_info_dict )
                meta_info_df.to_hdf(outFile, key = "meta_info", mode = "a")

    def removeOldFiles(self):
        for key in self.samples:
            sample = self.samples[key]
            for cat in sample.categories.categories:
                outFile = self.outputdir+"/"+cat+"_"+self.naming+".h5"
                if os.path.exists(outFile):
                    print("removing file {}".format(outFile))
                    os.remove(outFile)

    def renameOldFiles(self):
        for key in self.samples:
            sample = self.samples[key]
            for cat in sample.categories.categories:
                outFile = self.outputdir+"/"+cat+"_"+self.naming+".h5"
                if os.path.exists(outFile):
                    print("renaming file {}".format(outFile))
                    os.rename(outFile,outFile+".old")

    # deletes old files that were created new and rerenames old files by removing ".old", if no new files were created
    def handleOldFiles(self):
        old = []
        actual = []
        rerename = []
        remo = []
        for filename in os.listdir(self.outputdir):
            if filename.endswith(".old"):
                old.append(filename.split(".")[0])
            else:
                actual.append(filename.split(".")[0])
        for name in old:
            if name in actual:
                remo.append(name)
            else:
                rerename.append(name)
        for filename in os.listdir(self.outputdir):
            if filename.endswith(".old") and filename.split(".")[0] in remo:
                print("removing file {}".format(filename))
                os.remove(self.outputdir+"/"+filename)
            if filename.endswith(".old") and filename.split(".")[0] in rerename:
                print("re-renaming file {}".format(filename))
                os.rename(self.outputdir+"/"+filename,self.outputdir+"/"+filename[:-4])
