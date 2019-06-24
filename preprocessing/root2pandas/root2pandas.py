import numpy as np
import pandas as pd
import uproot as root
import re
import glob
import os
import shutil
import matplotlib.pyplot as plt
import ROOT
from math import sin, cos
from random import randrange

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
    def __init__(self, sampleName, ntuples, categories, selections = None, MEMs = None, ownVars=[], even_odd = False):
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
                self.selections += "(Evt_Odd == 1)"
            else:
                self.selections = "(Evt_Odd == 1)"



class Dataset:
    def __init__(self, outputdir, naming = "", addMEM = False, maxEntries = 50000):
        # settings for paths
        self.outputdir  = outputdir
        self.naming     = naming

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
            tree = f["MVATree"]
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

    # ====================================================================

    def runPreprocessing(self):
        # add variables for triggering and event category selection
        self.gatherTriggerVariables()

        # search for vector variables in list of variables and handle them separately
        self.searchVectorVariables()

        print("LOADING {} VARIABLES IN TOTAL.".format(len(self.variables)))
        # remove old files
        self.removeOldFiles()

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
            self.processSample(self.samples[key])

            # remove the own variables
            self.removeVariables( self.samples[key].ownVars )
            createSampleList(sampleList, self.samples[key])
            print("done.")
        # write file with preprocessed samples
        createSampleFile(self.outputdir, sampleList)

    def processSample(self, sample):
        # print sample info
        sample.printInfo()

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

        # loop over files
        for iFile, f in enumerate(ntuple_files):
            print("({}/{}) loading file {}".format(iFile+1,n_files,f))

            # open root file
            with root.open(f) as rf:
                # get MVATree
                try:
                    tree = rf["MVATree"]
                except:
                    print("could not open MVATree in ROOT file")
                    continue

            if tree.numentries == 0:
                print("MVATree has no entries - skipping file")
                continue

            # convert to dataframe
            df = tree.pandas.df(self.variables)

            # delete subentry index
            df = df.reset_index(1, drop = True)

            # handle vector variables, loop over them
            for vecvar in self.vector_variables:

                # load dataframe with vector variable
                vec_df = tree.pandas.df(vecvar)

                # loop over inices in vecvar list
                for idx in self.vector_variables[vecvar]:

                    # slice the indexdf
                    idx_df = vec_df.loc[ (slice(None), slice(idx,idx)), :]
                    idx_df = idx_df.reset_index(1, drop = True)

                    # define name for column in df
                    col_name = str(vecvar)+"["+str(idx)+"]"

                    # initialize column in original dataframe
                    df[col_name] = 0.

                    # append column to original dataframe
                    df.update( idx_df[vecvar].rename(col_name) )



            #in Bearbeitung
            ttbar_df = self.findbest(df)
            df = ttbar_df
            # apply event selection
            df = self.applySelections(df, sample.selections)



            sample.categories.categories["ttbar"] = "(Evt_is_ttbar==1)"
            sample.categories.categories["bkg"] = "(Evt_is_ttbar==0)"

            # print(ttbar_df)
            # plt.figure()
            # df["deltar"].hist(bins=20)
            # plt.show()
            # add to list of dataframes
            if concat_df.empty: concat_df = df
            else: concat_df = concat_df.append(df)

            # count entries so far
            n_entries += df.shape[0]

            # if number of entries exceeds max threshold, add labels and mem and save dataframe
            if (n_entries > self.maxEntries or f == ntuple_files[-1]):
                print("*"*50)
                print("max entries reached ...")

                # add class labels
                concat_df = self.addClassLabels(concat_df, sample.categories.categories)

                # add indexing
                concat_df.set_index(["Evt_Run", "Evt_Lumi", "Evt_ID"], inplace = True, drop = True)

                # add MEM variables
                if self.addMEM:
                    concat_df = self.addMEMVariable(concat_df, mem_df)


                # remove trigger variables
                concat_df = self.removeTriggerVariables(concat_df)

                # write data to file
                self.createDatasets(concat_df, sample.categories.categories)
                # self.createDatasets_for_ttbar(concat_df)

                print("*"*50)
                n_entries = 0
                concat_df = pd.DataFrame()


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

    #########################################################################################
    def addClassLabels_for_ttbar(seld,df):
        print("adding class labels to df ...")
        split_dfs=[]
        tmp_df = df.query("Evt_is_ttbar == 1")
        tmp_df["class_label"] = pd.Series()



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

    def createDatasets(self, df, categories):
        for key in categories:
            outFile = self.outputdir+"/"+key+"_"+self.naming+".h5"

            # create dataframe for category
            cat_df = df.query("(class_label == \""+str(key)+"\")")
            print("creating dataset for class label {} with {} entries".format(key, cat_df.shape[0]))

            with pd.HDFStore(outFile, "a") as store:
                store.append("data", cat_df, index = False)

########################################################################################
    def createDatasets_for_ttbar(self, df):
        outFile = self.outputdir + "/" + "ttbar" + "_"+ self.naming + ".h5"
        cat_df = df.query("Evt_is_ttbar == 1")
        print("creating dataset for class label {} with {} entries".format("ttbar", cat_df.shape[0]))
        with pd.HDFStore(outFile, "a") as store:
            store.append("data", cat_df, index = False)

        outFile = self.outputdir + "/" + "bkg" + "_" + self.naming + ".h5"
        cat_df = df.query("Evt_is_ttbar==0")
        print("creating dataset for class label {} with {} entries".format("bkg", cat_df.shape[0]))
        with pd.HDFStore(outFile, "a") as store:
            store.append("data", cat_df, index = False)




    def removeOldFiles(self):
        for key in self.samples:
            sample = self.samples[key]
            for cat in sample.categories.categories:
                outFile = self.outputdir+"/"+cat+"_"+self.naming+".h5"
                if os.path.exists(outFile):
                    print("removing file {}".format(outFile))
                    os.remove(outFile)

    def findbest(self, df):

        total_nr = df["Jet_Pt[0]"].size

        njets_vec       = df["N_Jets"].as_matrix()
        # n_tightmuons    = df["N_TightMuons"].as_matrix()
        # muon_pt         = df["Muon_Pt[0]"].as_matrix()
        # muon_eta        = df["Muon_Eta[0]"].as_matrix()
        # muon_phi        = df["Muon_Phi[0]"].as_matrix()
        # muon_m          = df["Muon_M[0]"].as_matrix()
        # electron_pt     = df["Electron_Pt[0]"].as_matrix()
        # electron_eta    = df["Electron_Eta[0]"].as_matrix()
        # electron_phi    = df["Electron_Phi[0]"].as_matrix()
        # electron_m      = df["Electron_M[0]"].as_matrix()
        # pt_met          = df["Evt_MET_Pt"].as_matrix()
        # phi_met         = df["Evt_MET_Phi"].as_matrix()
        # gentoplep_eta   = df["GenTopLep_Eta"].as_matrix()
        # gentoplep_phi   = df["GenTopLep_Phi"].as_matrix()
        # gentophad_eta   = df["GenTopHad_Eta"].as_matrix()
        # gentophad_phi   = df["GenTopHad_Phi"].as_matrix()

        gentophad_b_phi = df["GenTopHad_B_Phi"].as_matrix()
        gentophad_b_eta = df["GenTopHad_B_Eta"].as_matrix()
        gentoplep_b_phi = df["GenTopLep_B_Phi"].as_matrix()
        gentoplep_b_eta = df["GenTopLep_B_Eta"].as_matrix()

        gentophad_q1_phi= df["GenTopHad_Q1_Phi"].as_matrix()
        gentophad_q1_eta= df["GenTopHad_Q1_Eta"].as_matrix()
        gentophad_q2_phi= df["GenTopHad_Q2_Phi"].as_matrix()
        gentophad_q2_eta= df["GenTopHad_Q2_Eta"].as_matrix()

        numb = np.zeros(40)

        pepec = ["Jet_Pt", "Jet_Eta", "Jet_Phi", "Jet_E", "Jet_CSV"]
        jets  = ["TopHad_B", "TopLep_B", "TopHad_Q1", "TopHad_Q2"]

        for index in jets:
            for index2 in pepec:
                globals()[index + "_" + index2] = np.zeros(total_nr)

        is_ttbar = np.zeros(total_nr)
        cntr     = 0
        # idx_bhad, idx_blep, idx_q1, idx_q2, delta_r = np.zeros(total_nr),np.zeros(total_nr),np.zeros(total_nr),np.zeros(total_nr), np.zeros(total_nr)

        # loop over all events
        for i in range(total_nr):

            njets = min(njets_vec[i],11)
            for index in pepec:
                globals()[index] = np.array([])
            # jet_pts  = np.array([])
            # jet_etas = np.array([])
            # jet_phis = np.array([])
            # jet_es   = np.array([])
            # jet_ms   = np.array([])
            # jet_csv  = np.array([])

            #loop over all jets to write them into arrays with len: njets
            for j in range(njets):
                for index2 in pepec:
                    globals()[index2] = np.append(globals()[index2], df[str(index2) + "[" + str(j) + "]"].as_matrix()[i])
                # jet_pts     = np.append(jet_pts, df["Jet_Pt[" + str(j) +"]"].as_matrix()[i])
                # jet_etas    = np.append(jet_etas, df["Jet_Eta[" + str(j) +"]"].as_matrix()[i])
                # jet_phis    = np.append(jet_phis, df["Jet_Phi[" + str(j) +"]"].as_matrix()[i])
                # jet_es      = np.append(jet_es, df["Jet_E[" + str(j) +"]"].as_matrix()[i])
                # jet_ms      = np.append(jet_ms, df["Jet_M[" + str(j) +"]"].as_matrix()[i])
                # jet_csv     = np.append(jet_csv, df["Jet_CSV[" + str(j) +"]"].as_matrix()[i])


            # Pt_MET     = pt_met[i]
            # Phi_MET    = phi_met[i]

            #print(event.Evt_Phi_MET)


            # if(n_tightmuons[i]==1):
            #     lepton4=ROOT.TLorentzVector()
            #     lepton4.SetPtEtaPhiM(muon_pt[i], muon_eta[i], muon_phi[i], muon_m[i])
            # if(n_tightmuons[i]==0):
            #     lepton4=ROOT.TLorentzVector()
            #     lepton4.SetPtEtaPhiM(electron_pt[i], electron_eta[i], electron_phi[i], electron_m[i])

            minr=100000.
            # minrHad=100000.
            best_combi = np.zeros(4)
            # best_combiHad = np.zeros(4)
            # thadbest=ROOT.TLorentzVector()
            # tlepbest=ROOT.TLorentzVector()
            #Loop over all combinations of 4 jets
            #loop for TopHad_B
            for j in range(njets):
                #loop for TopLep_B
                for k in range(njets):
                    #loop for TopHad_Q1
                    for l in range(njets):
                        #loop for TopHad_Q2
                        for m in range(njets):
                            if(j==k or j==l or j==m or k==l or k==m or l==m):
                                continue
                            if(Jet_CSV[j]<0.227 or Jet_CSV[k]<0.227):
                                continue

                            delta_phi_bhad  = Jet_Phi[j] - gentophad_b_phi[i]
                            delta_phi_blep  = Jet_Phi[k] - gentoplep_b_phi[i]
                            delta_phi_q1    = Jet_Phi[l] - gentophad_q1_phi[i]
                            delta_phi_q2    = Jet_Phi[m] - gentophad_q2_phi[i]

                            deltar_bhad = ((Jet_Eta[j]-gentophad_b_eta[i])**2  + (correct_phi(delta_phi_bhad))**2)**0.5
                            deltar_blep = ((Jet_Eta[k]-gentoplep_b_eta[i])**2  + (correct_phi(delta_phi_blep))**2)**0.5
                            deltar_q1   = ((Jet_Eta[l]-gentophad_q1_eta[i])**2 + (correct_phi(delta_phi_q1))**2)**0.5
                            deltar_q2   = ((Jet_Eta[m]-gentophad_q2_eta[i])**2 + (correct_phi(delta_phi_q2))**2)**0.5


                            total_deltar = deltar_bhad + deltar_blep + deltar_q1 + deltar_q2
                            if(total_deltar<minr):
                                minr = total_deltar # macht quadratische Summation unter der Wurzel mehr Sinn?
                                best_comb = [j,k,l,m]
                                minr_bhad   = deltar_bhad
                                minr_blep   = deltar_blep
                                minr_q1     = deltar_q1
                                minr_q2     = deltar_q2

            n=0
            if(minr_bhad<0.3 and minr_blep<0.3 and minr_q1<0.3 and minr_q2<0.3):
                for index in jets:
                    for index2 in pepec:
                        globals()[index + "_" + index2][i] = df[str(index2) + "[" + str(best_comb[n]) + "]"].as_matrix()[i]
                    n+=1
                is_ttbar[i]=1

            # idx_bhad[i], idx_blep[i], idx_q1[i], idx_q2[i], delta_r[i] = best_comb[0], best_comb[1],best_comb[2],best_comb[3], minr
            if(i%500 == 0):
                print minr, minr_bhad, minr_blep, minr_q1, minr_q2

            n=0
            false_comb = np.zeros(4)
            j1,k1,l1,m1 = 0,0,0,0
            if is_ttbar[i]==1:
                df=df.append(df.iloc[[i]])
                while(j1 == k1 or j1==l1 or j1==m1 or k1==l1 or k1==m1 or  l1==m1 or (j1==best_comb[0] and k1 == best_comb[1] and l1==best_comb[2] and m1 ==best_comb[3])):
                    j1 = randrange(njets)
                    k1 = randrange(njets)
                    l1 = randrange(njets)
                    m1 = randrange(njets)
                false_comb = [j1,k1,l1,m1]
                for index in jets:
                    for index2 in pepec:
                        globals()[index + "_" + index2] = np.append(globals()[index + "_" + index2],(df[str(index2) + "[" + str(false_comb[n]) + "]"].as_matrix()[i]))
                    n+=1
                is_ttbar = np.append(is_ttbar,0)
                cntr +=1

            #Anteil Daten, die mitgenommen werden in Abhaengigkeit der hoehe des delta r cuttes auf die einzelnen jets
            # for u in range(40):
            #     if(minr_bhad<u/100. and minr_blep<u/100. and minr_q1<u/100. and minr_q2<u/100.):
            #         numb[u]+=1

                # if(i==total_nr-1 and u%10 == 0):
                #     print "Anteil verwendeter Daten bei delta r cut ", u/100.," : ", numb[u]/total_nr

        # x = np.arange(0,0.4,0.01)
        # plt.plot(x, numb/total_nr)
        # plt.title("ratio between number of tolerated events and total number")
        # plt.xlabel("$\Delta$r cut")
        # plt.show()
        df_new = pd.DataFrame()
        df_new["N_Jets"] = df["N_Jets"].as_matrix()
        df_new["N_BTagsM"]   = df["N_BTagsM"].as_matrix()
        df_new["Evt_Run"]   = df["Evt_Run"].as_matrix()
        df_new["Evt_Lumi"]   = df["Evt_Lumi"].as_matrix()
        df_new["Evt_ID"]   = df["Evt_ID"].as_matrix()
        df_new["Evt_MET_Pt"]   = df["Evt_MET_Pt"].as_matrix()
        df_new["Weight_GEN_nom"]   = df["Weight_GEN_nom"].as_matrix()
        df_new["Weight_XS"]   = df["Weight_XS"].as_matrix()
        df_new["Weight_CSV"]   = df["Weight_CSV"].as_matrix()
        df_new["N_LooseElectrons"]   = df["N_LooseElectrons"].as_matrix()
        df_new["N_TightMuons"]   = df["N_TightMuons"].as_matrix()
        df_new["Muon_Pt"]   = df["Muon_Pt"].as_matrix()
        df_new["N_LooseMuons"]   = df["N_LooseMuons"].as_matrix()
        df_new["N_TightElectrons"]   = df["N_TightElectrons"].as_matrix()






        #print df.shape, len(globals()[index + "_" + index2]), len(is_ttbar)
        entr=0
        for index in jets:
            for index2 in pepec:
                df[index + "_" + index2] = globals()[index + "_" + index2] # hier scheint irgendwas mit den Indices nicht zu passen. Vermutung: array zu lang, muss falsche events noch rausschmeissen
                entr+=1
        df["Evt_is_ttbar"]  = is_ttbar
        #print df,entr, df.columns[0:-entr]
        df.drop(df.columns[:-(entr+1)],inplace = True, axis = 1)
        # # print df
        df["N_Jets"]= df_new["N_Jets"].as_matrix()
        df["N_BTagsM"]   = df_new["N_BTagsM"].as_matrix()
        df["Evt_Run"]   = df_new["Evt_Run"].as_matrix()
        df["Evt_Lumi"]   = df_new["Evt_Lumi"].as_matrix()
        df["Evt_ID"]   = df_new["Evt_ID"].as_matrix()
        df["Evt_MET_Pt"]   = df_new["Evt_MET_Pt"].as_matrix()
        df["Weight_GEN_nom"]   = df_new["Weight_GEN_nom"].as_matrix()
        df["Weight_XS"]   = df_new["Weight_XS"].as_matrix()
        df["Weight_CSV"]   = df_new["Weight_CSV"].as_matrix()
        df["N_LooseElectrons"]   = df_new["N_LooseElectrons"].as_matrix()
        df["N_TightMuons"]   = df_new["N_TightMuons"].as_matrix()
        df["Muon_Pt"]   = df_new["Muon_Pt"].as_matrix()
        df["N_LooseMuons"]   = df_new["N_LooseMuons"].as_matrix()
        df["N_TightElectrons"]   = df_new["N_TightElectrons"].as_matrix()







        i,j=0,0
        while(i<total_nr):
            if df["Evt_is_ttbar"].as_matrix()[i]==0:
                df=df.drop(df.index[i],axis=0)
                total_nr -= 1
                j+=1
            else:
                i+=1
                j+=1
        print df.shape, df.columns
        return df


# function to append a list with sample, label and normalization_weight to a list samples
def createSampleList(sList, sample, label = None, nWeight = 1):
    """ takes a List a sample and appends a list with category, label and weight. Checks if even/odd splitting was made and therefore adjusts the normalization weight """
    if sample.even_odd: nWeight*=2.
    for cat in sample.categories.categories:
        if label==None:
            sList.append([cat, cat, nWeight])
        else:
            sList.append([cat, label, nWeight])
    return sList

# function to create a file with all preprocessed samples
def createSampleFile(outPath, sampleList):
    # create file
    processedSamples=""
    # write samplenames in file
    for sample in sampleList:
        processedSamples+=str(sample[0])+" "+str(sample[1])+" "+str(sample[2])+"\n"
    with open(outPath+"/sampleFile.dat","w") as sampleFile:
        sampleFile.write(processedSamples)

# function to read Samplefile
def readSampleFile(inPath):
    """ reads file and returns a list with all samples, that are not uncommented. Uncomment samples by adding a '#' in front of the line"""
    sampleList = []
    with open(outPath+"/sampleFile.dat","w") as sampleFile:
        for row in sampleFile:
            if row[0] != "#":
                sample = row.split()
                sampleList.append(dict( sample=sample[0], label=sample[1], normWeight=sample[2]) )
    return sampleList

# function to add samples to InputSamples
def addToInputSamples(inputSamples, samples, naming="_dnn.h5"):
    """ adds each sample in samples to the inputSample """
    for sample in samples:
        inputSamples.addSample(sample["sample"]+naming, label=sample["label"], normalization_weight = sample["normWeight"])

def correct_phi(phi):
    if(phi  <=  -np.pi):
        phi += 2*np.pi
    if(phi  >    np.pi):
        phi -= 2*np.pi
    return phi
    # ====================================================================
