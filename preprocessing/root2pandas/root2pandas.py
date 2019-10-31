import numpy as np
import pandas as pd
import uproot as root
import re
import glob
import os
import shutil
import matplotlib.pyplot as plt
import ROOT
from math import sin, cos, log
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
		self.ntuples	= ntuples
		self.selections = selections
		self.categories = categories
		self.MEMs	   = MEMs
		self.ownVars	= ownVars
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
	def __init__(self, outputdir, naming = "", addMEM = False, maxEntries = 50000, Scale=False):
		# settings for paths
		self.outputdir  = outputdir
		self.naming	 = naming

		# generating output dir
		if not os.path.exists(self.outputdir):
			os.makedirs(self.outputdir)

		# settings for dataset
		self.addMEM	 = addMEM
		self.maxEntries = int(maxEntries)
		self.Scale	  = Scale

		# default values for some configs
		self.baseSelection  = None
		self.samples		= {}
		self.variables	  = []

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
			#ttbar_df = self.findbest(df)
			#ttbar_df = self.findbestwithHiggs(df)
			ttbar_df = self.findbestHiggs(df)
			df = ttbar_df
			# apply event selection
			df = self.applySelections(df, sample.selections)



			#sample.categories.categories["ttbar"] = "(Evt_is_ttbar==1)"
			sample.categories.categories["bkg"] = "(Evt_is_Higgs==0)"
			sample.categories.categories["ttH"]  = "(Evt_is_Higgs==1)"

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
			print("	lost {}/{} events".format(entries_before-entries_after, entries_before))
			print("	we will only save events with mem...")
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

#########################################################################################
# function to get dataframe with all variables relevant for ttbar-reconstruction
	def findbest(self, df):

		# (1) total number of events in current df
		total_nr  = df["N_Jets"].size
		#print("-"*10+"total_nr"+"-"*10)
		#print(total_nr)
	
		# array containing the number of jets per event
		njets_vec = df["N_Jets"].values
		#print("-"*10+"njets_vec"+"-"*10)
		#print(njets_vec,len(njets_vec))
		#print("_"*20)

		# (2) just for plot that shows amount of events accepted as ttbar: x-axis: acceptance-niveau of delta r between jets and gen blep/bhad/q1/q2 (); r_max: max of x-axis
		r_max = 0.8
		numb = np.zeros(int(r_max*100))

		# (3) get gen-info of quarks needed for calculation of delta r at
		GenTopHad_B_Phi = df["GenTopHad_B_Phi"].values
		GenTopHad_B_Eta = df["GenTopHad_B_Eta"].values
		GenTopLep_B_Phi = df["GenTopLep_B_Phi"].values
		GenTopLep_B_Eta = df["GenTopLep_B_Eta"].values

		GenTophad_Q1_Phi= df["GenTopHad_Q1_Phi"].values
		GenTophad_Q1_Eta= df["GenTopHad_Q1_Eta"].values
		Gentophad_Q2_Phi= df["GenTopHad_Q2_Phi"].values
		GenTophad_Q2_Eta= df["GenTopHad_Q2_Eta"].values

		# (4) arrays for better clarity in (5),...; jets and reconstructed particles
		jets  = ["TopHad_B", "TopLep_B", "TopHad_Q1", "TopHad_Q2"]
		pepec = ["Pt", "Eta", "Phi", "E", "CSV"]

		recos = ["TopHad", "TopLep", "WHad", "WLep"]
		pepm  = ["Pt", "Eta", "Phi", "M", "logM"]

		# (5) create arrays: TopHad_B_Pt etc. for all jets and reconstructed particles get filled in (13); information about whether an event is accepted as ttbar (is_ttbar); scaling-factor if all wrong combinations shall be considered as background if only one event: remains 1(bkg_scale)
		for index in jets:
			for index2 in pepec:
				globals()[index + "_" + index2] = np.zeros(total_nr)

		for index in recos:
			for index2 in pepm:
				globals()["reco_" + index + "_" + index2] = np.zeros(total_nr)

		is_ttbar = np.zeros(total_nr)
		bkg_scale = np.zeros(total_nr)+1

		ttbar_phi = np.zeros(total_nr)
		ttbar_Pt_div_Ht_p_Met = np.zeros(total_nr)
		# cntr	 = 0
	#for i in range(total_nr):
	#	print(njets_vec[i])
		# (6) loop over all events
		for i in range(total_nr):

			# (7) get number of jets in current event; create array for bkg scaling, shape(2,4) is to make np.append with new rows possible later (12). get deleted again in (12); minr and array for indices of best combination
		#if not isinstance(njets_vec[i],float):
		#if not type(njets_vec[i]) is float:
		#	print("NaN, continue")
		#	continue
			njets = min(int(njets_vec[i]),12)
			#print(njets)
			bkg = np.zeros((2,4))
			minr=100000.
			best_comb = np.zeros(4)


			# (8) create arrays for entries of pepec and loop over all jets: df to arrays with len(njets), i.e. Jet_Pt (contains Pt info of all jets of this event)
			for index in pepec:
				globals()[index] = np.array([])

			for j in range(njets):
				for index2 in pepec:
					globals()[index2] = np.append(globals()[index2], df["Jet_" + str(index2) + "[" + str(j) + "]"].values[i])

			minr_bhad = 10
			minr_blep = 10
			minr_q1   = 10
			minr_q2   = 10
			# (9) Loop over all combinations of 4 jets: j for TopHad_B ; k for TopLep_B ; l for TopHad_Q1 ; m for TopHad_Q2
			for j in range(njets):
				for k in range(njets):
					for l in range(njets):
						for m in range(njets):
							# (10) exclude combinations with two identical indices or without B-tags for b-quarks (medium working point)
							if(j==k or j==l or j==m or k==l or k==m or l==m):
								continue
							if(CSV[j]<0.277 or CSV[k]<0.277):
								continue


							# (11) look for combination with smallest delta r between jets and genjets, if scale: add all relevant combinations except for right combination to bkg
							deltar_bhad = ((Eta[j]-GenTopHad_B_Eta[i])**2  + (correct_phi(Phi[j] - GenTopHad_B_Phi[i]))**2)**0.5
							deltar_blep = ((Eta[k]-GenTopLep_B_Eta[i])**2  + (correct_phi(Phi[k] - GenTopLep_B_Phi[i]))**2)**0.5
							deltar_q1   = ((Eta[l]-GenTophad_Q1_Eta[i])**2 + (correct_phi(Phi[l] - GenTophad_Q1_Phi[i]))**2)**0.5
							deltar_q2   = ((Eta[m]-GenTophad_Q2_Eta[i])**2 + (correct_phi(Phi[m] - Gentophad_Q2_Phi[i]))**2)**0.5

							total_deltar = deltar_bhad + deltar_blep + deltar_q1 + deltar_q2

							if(total_deltar<minr):
								if self.Scale:
									bkg = np.append(bkg, [[best_comb[0],best_comb[1],best_comb[2],best_comb[3]]], axis=0)
								minr = total_deltar
								best_comb = [j,k,l,m]
								minr_bhad   = deltar_bhad
								minr_blep   = deltar_blep
								minr_q1	 = deltar_q1
								minr_q2	 = deltar_q2

							if (total_deltar >= minr and self.Scale):
								bkg = np.append(bkg, [[j,k,l,m]], axis = 0)

			# (12) delete first two rows
			bkg = np.delete(bkg, (0,1), axis=0)

			# (13) only combinations with smallest total_deltar and each delta r <0.3 are ttbar-events (~50% of all events pass this); reconstruct further particles, quark jets not to be b-tagged
			n=0
			# if(minr_bhad<0.3 and minr_blep<0.3 and minr_q1<0.3 and minr_q2<0.3 and (CSV[best_comb[2]]<0.277 or CSV[best_comb[3]]<0.277)):
			if(minr_bhad<0.1 and minr_blep<0.1 and minr_q1<0.1 and minr_q2<0.1):
				for index in jets:
					for index2 in pepec:
						globals()[index + "_" + index2][i] = df["Jet_" + str(index2) + "[" + str(best_comb[n]) + "]"].values[i]
					n+=1
				is_ttbar[i]=1

				reco_TopHad_4vec, reco_TopLep_4vec, reco_WHad_4vec, reco_WLep_4vec, reco_lepton_4vec, reco_Neutrino_4vec = reconstruct_ttbar(df, i,best_comb[0],best_comb[1],best_comb[2],best_comb[3])

				for index in recos:
					globals()["reco_" + index + "_Pt"][i]   =	 locals()["reco_" + index + "_4vec"].Pt()
					globals()["reco_" + index + "_Eta"][i]  =	 locals()["reco_" + index + "_4vec"].Eta()
					globals()["reco_" + index + "_Phi"][i]  =	 locals()["reco_" + index + "_4vec"].Phi()
					globals()["reco_" + index + "_M"][i]	=	 locals()["reco_" + index + "_4vec"].M()
					globals()["reco_" + index + "_logM"][i] = log(locals()["reco_" + index + "_4vec"].M())

				ttbar_phi[i] = correct_phi(reco_TopHad_4vec.Phi() - reco_TopLep_4vec.Phi())
				ttbar_Pt_div_Ht_p_Met[i] = (reco_TopHad_4vec.Pt() + reco_TopLep_4vec.Pt())/(df["Evt_HT"].values[i] + df["Evt_MET_Pt"].values[i] + reco_lepton_4vec.Pt())

			# (14) short print against long boring time
			if(i%5000 == 0):
				print minr, minr_bhad, minr_blep, minr_q1, minr_q2


			# (14) for ttbar events: append wrong combination(s) of jets to arrays and df
			if is_ttbar[i]==1:
				if self.Scale:
					for nr in range(len(bkg)):
						df=df.append(df.iloc[[i]])
						false_comb = [int(bkg[nr][0]),int(bkg[nr][1]),int(bkg[nr][2]),int(bkg[nr][3])]

						n=0
						for index in jets:
							for index2 in pepec:
								globals()[index + "_" + index2] = np.append(globals()[index + "_" + index2],(df["Jet_" +str(index2) + "[" + str(false_comb[n]) + "]"].values[i]))
							n+=1

						reco_TopHad_4vec, reco_TopLep_4vec, reco_WHad_4vec, reco_WLep_4vec, reco_lepton_4vec, reco_neutrino_4vec = reconstruct_ttbar(df, i,false_comb[0],false_comb[1],false_comb[2],false_comb[3])

						for index in recos:
							globals()["reco_" + index + "_Pt"]   = np.append(globals()["reco_" + index + "_Pt"],	   locals()["reco_" + index + "_4vec"].Pt())
							globals()["reco_" + index + "_Eta"]  = np.append(globals()["reco_" + index + "_Eta"],	  locals()["reco_" + index + "_4vec"].Eta())
							globals()["reco_" + index + "_Phi"]  = np.append(globals()["reco_" + index + "_Phi"],	  locals()["reco_" + index + "_4vec"].Phi())
							globals()["reco_" + index + "_M"]	= np.append(globals()["reco_" + index + "_M"],		locals()["reco_" + index + "_4vec"].M())
							globals()["reco_" + index + "_logM"] = np.append(globals()["reco_" + index + "_logM"], log(locals()["reco_" + index + "_4vec"].M()))

						ttbar_phi = np.append(ttbar_phi, correct_phi(reco_TopHad_4vec.Phi() - reco_TopLep_4vec.Phi()))
						ttbar_Pt_div_Ht_p_Met = np.append(ttbar_Pt_div_Ht_p_Met,(reco_TopHad_4vec.Pt() + reco_TopLep_4vec.Pt())/(df["Evt_HT"].values[i] + df["Evt_MET_Pt"].values[i] + reco_lepton_4vec.Pt()))

						is_ttbar  = np.append(is_ttbar,0)
						bkg_scale = np.append(bkg_scale, 1./len(bkg))



				if self.Scale==0:
					false_comb = np.zeros(4)
					j1,k1,l1,m1 = 0,0,0,0
					df=df.append(df.iloc[[i]])
					#kombis suchen, die sich von bester kombi unterscheiden (beachte jets aus dem w duerfen auch nicht vertauschen) und selbst den anforderungen genuegen
					while(CSV[j1]<0.277 or CSV[k1]<0.277 or j1 == k1 or j1==l1 or j1==m1 or k1==l1 or k1==m1 or  l1==m1 or (j1==best_comb[0] and k1 == best_comb[1] and l1==best_comb[2] and m1 ==best_comb[3]) or (j1==best_comb[0] and k1 == best_comb[1] and l1==best_comb[3] and m1 ==best_comb[2])):
						j1 = randrange(njets)
						k1 = randrange(njets)
						l1 = randrange(njets)
						m1 = randrange(njets)
					false_comb = [j1,k1,l1,m1]

					n=0
					for index in jets:
						for index2 in pepec:
							globals()[index + "_" + index2] = np.append(globals()[index + "_" + index2],(df["Jet_" +str(index2) + "[" + str(false_comb[n]) + "]"].values[i]))
						n+=1

					reco_TopHad_4vec, reco_TopLep_4vec, reco_WHad_4vec, reco_WLep_4vec, reco_lepton_4vec, reco_neutrino_4vec = reconstruct_ttbar(df, i,false_comb[0],false_comb[1],false_comb[2],false_comb[3])

					for index in recos:
						globals()["reco_" + index + "_Pt"]   = np.append(globals()["reco_" + index + "_Pt"],	   locals()["reco_" + index + "_4vec"].Pt())
						globals()["reco_" + index + "_Eta"]  = np.append(globals()["reco_" + index + "_Eta"],	  locals()["reco_" + index + "_4vec"].Eta())
						globals()["reco_" + index + "_Phi"]  = np.append(globals()["reco_" + index + "_Phi"],	  locals()["reco_" + index + "_4vec"].Phi())
						globals()["reco_" + index + "_M"]	= np.append(globals()["reco_" + index + "_M"],		locals()["reco_" + index + "_4vec"].M())
						globals()["reco_" + index + "_logM"] = np.append(globals()["reco_" + index + "_logM"], log(locals()["reco_" + index + "_4vec"].M()))

					ttbar_phi = np.append(ttbar_phi, correct_phi(reco_TopHad_4vec.Phi() - reco_TopLep_4vec.Phi()))
					ttbar_Pt_div_Ht_p_Met = np.append(ttbar_Pt_div_Ht_p_Met,(reco_TopHad_4vec.Pt() + reco_TopLep_4vec.Pt())/(df["Evt_HT"].values[i] + df["Evt_MET_Pt"].values[i] + reco_lepton_4vec.Pt()))

					is_ttbar = np.append(is_ttbar,0)
					bkg_scale = np.append(bkg_scale, 1)



			# (15) Anteil Daten, die mitgenommen werden in Abhaengigkeit der hoehe des delta r cuttes auf die einzelnen jets
			for u in range(int(r_max*100)):
				if(minr_bhad<u/100. and minr_blep<u/100. and minr_q1<u/100. and minr_q2<u/100.):
					numb[u]+=1

				if(i==total_nr-1 and u%10 == 0):
					print "Anteil verwendeter Daten bei delta R cut ", u/100.," : ", numb[u]/total_nr

		x = np.arange(0,r_max,0.01)
		plt.plot(x, numb/total_nr)
		plt.title("Anteil der in der Aufbereitung beruecksichtigten Daten")
		plt.xlabel("akzeptiertes maximales $\Delta$R")
		plt.ylabel("Verhaeltnis akzeptierter Ereignisse zur Gesamtzahl an Ereignissen")
		plt.grid()
		plt.savefig("ratio_neu_btag.pdf")

		# (16) delete all variables except for:
		df_new = pd.DataFrame()
		variables_toadd  = ["N_Jets", "N_BTagsM", "Evt_Run", "Evt_Lumi", "Evt_ID", "Evt_MET_Pt", "Evt_MET_Phi", "Weight_GEN_nom", "Weight_XS", "Weight_CSV", "N_LooseElectrons", "N_TightMuons",
							"Muon_Pt[0]", "Muon_Eta[0]", "Muon_Phi[0]","Muon_E[0]", "Electron_Pt[0]","Electron_Eta[0]","Electron_Phi[0]","Electron_E[0]","N_LooseMuons", "N_TightElectrons", "Evt_Odd"]

		for ind in variables_toadd:
			df_new[ind] = df[ind].values

		# (17) another short print for control
		#print df.shape, len(globals()["TopHad_B_" + index2]), len(is_ttbar)

		# (18) add correct and false combinations with their tags to df
		entr=0
		for index in jets:
			for index2 in pepec:
				df[index + "_" + index2] = globals()[index + "_" + index2]
				entr+=1


		for index in recos:
			for index2 in pepm:
				df["reco_" + index + "_" + index2]   = globals()["reco_" + index + "_" + index2]
				entr+=1

		print is_ttbar
		print ttbar_phi
		print df

		df["bkg_scale"] = bkg_scale
		df["Evt_is_ttbar"]  = is_ttbar
		df["ttbar_phi"] = ttbar_phi
		df["ttbar_pt_div_ht_p_met"] = ttbar_Pt_div_Ht_p_Met

		# (19) drop all columns except for those added just before and add variables_toadd again (easier than prooving every column if it is in variables_toadd before dropping)
		df.drop(df.columns[:-(entr+4)],inplace = True, axis = 1)

		for ind in variables_toadd:
			df[ind] = df_new[ind].values


		# (20) delete combinations of events without certain ttbar event
		i,j=0,0
		while(i<total_nr):
			if df["Evt_is_ttbar"].values[i]==0:
				df=df.drop(df.index[i],axis=0)
				total_nr -= 1
				j+=1
			else:
				i+=1
				j+=1
		print df.shape, df.columns

		return df

###########################################################################
# function to get dataframe with all variables relevant for ttbar-reconstruction
	def findbestwithHiggs(self, df):

		# (1) total number of events in current df
		total_nr  = df["N_Jets"].size
	
		# array containing the number of jets per event
		njets_vec = df["N_Jets"].values
		#print("-"*10+"njets_vec"+"-"*10)
		#print(njets_vec,len(njets_vec))
		#print("_"*20)

		# (2) just for plot that shows amount of events accepted as ttbar: x-axis: acceptance-niveau of delta r between jets and gen blep/bhad/q1/q2 (); r_max: max of x-axis
		r_max = 0.8
		numb = np.zeros(int(r_max*100))

		# (3) get gen-info of quarks needed for calculation of delta r at
		GenTopHad_B_Phi = df["GenTopHad_B_Phi"].values
		GenTopHad_B_Eta = df["GenTopHad_B_Eta"].values
		GenTopLep_B_Phi = df["GenTopLep_B_Phi"].values
		GenTopLep_B_Eta = df["GenTopLep_B_Eta"].values

		GenTophad_Q1_Phi= df["GenTopHad_Q1_Phi"].values
		GenTophad_Q1_Eta= df["GenTopHad_Q1_Eta"].values
		Gentophad_Q2_Phi= df["GenTopHad_Q2_Phi"].values
		GenTophad_Q2_Eta= df["GenTopHad_Q2_Eta"].values

		GenHiggs_B1_Phi = df["GenHiggs_B1_Phi"].values
		GenHiggs_B1_Eta = df["GenHiggs_B1_Eta"].values
		GenHiggs_B2_Phi = df["GenHiggs_B2_Phi"].values
		GenHiggs_B2_Eta = df["GenHiggs_B2_Eta"].values


		# (4) arrays for better clarity in (5),...; jets and reconstructed particles
		jets  = ["TopHad_B", "TopLep_B", "TopHad_Q1", "TopHad_Q2"]#,"Higgs_B1","Higgs_B2"]
		pepec = ["Pt", "Eta", "Phi", "E", "CSV"]

		recos = ["TopHad", "TopLep", "WHad", "WLep"]#,"Higgs"]
		pepm  = ["Pt", "Eta", "Phi", "M", "logM"]

		# (5) create arrays: TopHad_B_Pt etc. for all jets and reconstructed particles get filled in (13); information about whether an event is accepted as ttbar (is_ttbar); scaling-factor if all wrong combinations shall be considered as background if only one event: remains 1(bkg_scale)
		for index in jets:
			for index2 in pepec:
				globals()[index + "_" + index2] = np.zeros(total_nr)

		for index in recos:
			for index2 in pepm:
				globals()["reco_" + index + "_" + index2] = np.zeros(total_nr)

		is_ttbar = np.zeros(total_nr)
		bkg_scale = np.zeros(total_nr)+1

		ttbar_phi = np.zeros(total_nr)
		ttbar_Pt_div_Ht_p_Met = np.zeros(total_nr)
		# cntr	 = 0
	#for i in range(total_nr):
	#	print(njets_vec[i])

		jkcounter = 0
		jnkcounter = 0
		counter = 0
		counter3 = 0
		counter2 = 0
		counter1 = 0
		Rb1 = np.zeros(total_nr)
		Rb2 = np.zeros(total_nr)
		Rb12 = np.zeros(total_nr)

		# (6) loop over all events
		for i in range(total_nr):

			# (7) get number of jets in current event; create array for bkg scaling, shape(2,4) is to make np.append with new rows possible later (12). get deleted again in (12); minr and array for indices of best combination

			njets = min(int(njets_vec[i]),12)
			#print(njets)
			bkg = np.zeros((2,4))
			minr = 100000.
			minrH = 10000.
			best_comb = np.zeros(4)


			# (8) create arrays for entries of pepec and loop over all jets: df to arrays with len(njets), i.e. Jet_Pt (contains Pt info of all jets of this event)
			for index in pepec:
				globals()[index] = np.array([])

			for j in range(njets):
				for index2 in pepec:
					globals()[index2] = np.append(globals()[index2], df["Jet_" + str(index2) + "[" + str(j) + "]"].values[i])

			minr_bhad = 10
			minr_blep = 10
			minr_q1   = 10
			minr_q2   = 10
#get Higgs Bjets---------------------

			minr_b1 = 10
			minr_b2 = 10
			minr_bhiggs = 10
			gibtesueberhauptbeidebjets = 0


			# Loop over all combinations of 2 b-jets: j for b1, k for b2
			for j in range(njets):
				for k in range(njets):

						# look for combination with smallest delta r between jets and genjets, if scale: add all relevant combinations except for right combination to bkg
					deltar_b1 = ((Eta[j]-GenHiggs_B1_Eta[i])**2  + (correct_phi(Phi[j] - GenHiggs_B1_Phi[i]))**2)**0.5
					deltar_b2 = ((Eta[k]-GenHiggs_B2_Eta[i])**2  + (correct_phi(Phi[k] - GenHiggs_B2_Phi[i]))**2)**0.5
	 		#if deltar_b1 < 0.3 and deltar_b2 < 0.3:	
					total_deltarH = deltar_b1 + deltar_b2
					if(total_deltarH < minrH):
						minrH = total_deltarH
						minr_b1   = deltar_b1
						minr_b2   = deltar_b2
						J = j
						K = k
						gibtesueberhauptbeidebjets = 1

			# if there are no 2 matching bjets (each delta R < 0.3) at all, ignore event
			if gibtesueberhauptbeidebjets!=1:
				continue
			else:
				counter3=counter3+1
			gibtesueberhauptbeidebjets = 0

			if deltar_b1 < 0.7 and deltar_b2 < 0.7:
				counter1=counter1+1
			if deltar_b1 < 0.5 and deltar_b2 < 0.5:
				counter2=counter2+1
			if deltar_b1 < 0.3 and deltar_b2 < 0.3:
				counter=counter+1
	


#---------------------------

			# (9) Loop over all combinations of 5 jets: j for TopHad_B ; k for TopLep_B ; l for TopHad_Q1 ; m for TopHad_Q2
			for j in range(njets):
				for k in range(njets):
					for l in range(njets):
						for m in range(njets):
				#for n in range(njets):
							# (10) exclude combinations with two identical indices or without B-tags for b-quarks (medium working point)
							if(j==k or j==l or j==m or k==l or k==m or l==m):# or n==j or n==k or n==l or n==m):
								continue
							if(CSV[j]<0.277 or CSV[k]<0.277):
								continue

							# (11) look for combination with smallest delta r between jets and genjets, if scale: add all relevant combinations except for right combination to bkg
				# ! ANNAHME: higgsb1 jet = higgsb2jet----------------------------------------------

							deltar_bhad = ((Eta[j]-GenTopHad_B_Eta[i])**2  + (correct_phi(Phi[j] - GenTopHad_B_Phi[i]))**2)**0.5
							deltar_blep = ((Eta[k]-GenTopLep_B_Eta[i])**2  + (correct_phi(Phi[k] - GenTopLep_B_Phi[i]))**2)**0.5
							deltar_q1   = ((Eta[l]-GenTophad_Q1_Eta[i])**2 + (correct_phi(Phi[l] - GenTophad_Q1_Phi[i]))**2)**0.5
							deltar_q2   = ((Eta[m]-GenTophad_Q2_Eta[i])**2 + (correct_phi(Phi[m] - Gentophad_Q2_Phi[i]))**2)**0.5
						#deltar_b12higgs = ((Eta[n]-GenHiggs_B1_Eta[i])**2 + (correct_phi(Phi[n] - GenHiggs_B1_Phi[i]))**2 + (Eta[n]-GenHiggs_B2_Eta[i])**2 + (correct_phi(Phi[n] - GenHiggs_B2_Phi[i]))**2 )**0.5

							total_deltar = deltar_bhad + deltar_blep + deltar_q1 + deltar_q2 #+ deltar_b12higgs

							if(total_deltar<minr):
								if self.Scale:
									bkg = np.append(bkg, [[best_comb[0],best_comb[1],best_comb[2],best_comb[3]]], axis=0)
								minr = total_deltar
								best_comb = [j,k,l,m]#,n]
								minr_bhad   = deltar_bhad
								minr_blep   = deltar_blep
								minr_q1	 = deltar_q1
								minr_q2	 = deltar_q2
				#minr_bhiggs = deltar_b12higgs

							if (total_deltar >= minr and self.Scale):
								bkg = np.append(bkg, [[j,k,l,m,n]], axis = 0)

			# (12) delete first two rows
			bkg = np.delete(bkg, (0,1), axis=0)

				# (13) only combinations with smallest total_deltar and each delta r <0.3 are ttbar-events (~50% of all events pass this); reconstruct further particles, quark jets not to be b-tagged
			n=0
				# if(minr_bhad<0.3 and minr_blep<0.3 and minr_q1<0.3 and minr_q2<0.3 and (CSV[best_comb[2]]<0.277 or CSV[best_comb[3]]<0.277)):
			if(minr_bhad<0.1 and minr_blep<0.1 and minr_q1<0.1 and minr_q2<0.1):# and minr_bhiggs<0.3):
				for index in jets:
					for index2 in pepec:
						globals()[index + "_" + index2][i] = df["Jet_" + str(index2) + "[" + str(best_comb[n]) + "]"].values[i]
					n+=1
				is_ttbar[i]=1

				reco_TopHad_4vec, reco_TopLep_4vec, reco_WHad_4vec, reco_WLep_4vec, reco_lepton_4vec, reco_Neutrino_4vec = reconstruct_ttbar(df, i,best_comb[0],best_comb[1],best_comb[2],best_comb[3])

				for index in recos:
					globals()["reco_" + index + "_Pt"][i]   =	 locals()["reco_" + index + "_4vec"].Pt()
					globals()["reco_" + index + "_Eta"][i]  =	 locals()["reco_" + index + "_4vec"].Eta()
					globals()["reco_" + index + "_Phi"][i]  =	 locals()["reco_" + index + "_4vec"].Phi()
					globals()["reco_" + index + "_M"][i]	=	 locals()["reco_" + index + "_4vec"].M()
					globals()["reco_" + index + "_logM"][i] = log(locals()["reco_" + index + "_4vec"].M())

				ttbar_phi[i] = correct_phi(reco_TopHad_4vec.Phi() - reco_TopLep_4vec.Phi())
				ttbar_Pt_div_Ht_p_Met[i] = (reco_TopHad_4vec.Pt() + reco_TopLep_4vec.Pt())/(df["Evt_HT"].values[i] + df["Evt_MET_Pt"].values[i] + reco_lepton_4vec.Pt())

			# (14) short print against long boring time
			if(i%5000 == 0):
				print " minR:",minr," minRbHad:", minr_bhad," minRbLep:", minr_blep," minRQ1:", minr_q1," minRQ2:", minr_q2#," minRbHiggs:", minr_bhiggs,"minRb1:",minr_b1,"minRb2:",minr_b2


			# (14) for ttbar events: append wrong combination(s) of jets to arrays and df
			if is_ttbar[i]==1:
				if self.Scale:
					for nr in range(len(bkg)):
						df=df.append(df.iloc[[i]])
						false_comb = [int(bkg[nr][0]),int(bkg[nr][1]),int(bkg[nr][2]),int(bkg[nr][3])]

						n=0
						for index in jets:
							for index2 in pepec:
								globals()[index + "_" + index2] = np.append(globals()[index + "_" + index2],(df["Jet_" +str(index2) + "[" + str(false_comb[n]) + "]"].values[i]))
							n+=1
	
						reco_TopHad_4vec, reco_TopLep_4vec, reco_WHad_4vec, reco_WLep_4vec, reco_lepton_4vec, reco_neutrino_4vec = reconstruct_ttbar(df, i,false_comb[0],false_comb[1],false_comb[2],false_comb[3])

						for index in recos:
							globals()["reco_" + index + "_Pt"]   = np.append(globals()["reco_" + index + "_Pt"],	   locals()["reco_" + index + "_4vec"].Pt())
							globals()["reco_" + index + "_Eta"]  = np.append(globals()["reco_" + index + "_Eta"],	  locals()["reco_" + index + "_4vec"].Eta())
							globals()["reco_" + index + "_Phi"]  = np.append(globals()["reco_" + index + "_Phi"],	  locals()["reco_" + index + "_4vec"].Phi())
							globals()["reco_" + index + "_M"]	= np.append(globals()["reco_" + index + "_M"],		locals()["reco_" + index + "_4vec"].M())
							globals()["reco_" + index + "_logM"] = np.append(globals()["reco_" + index + "_logM"], log(locals()["reco_" + index + "_4vec"].M()))

						ttbar_phi = np.append(ttbar_phi, correct_phi(reco_TopHad_4vec.Phi() - reco_TopLep_4vec.Phi()))
						ttbar_Pt_div_Ht_p_Met = np.append(ttbar_Pt_div_Ht_p_Met,(reco_TopHad_4vec.Pt() + reco_TopLep_4vec.Pt())/(df["Evt_HT"].values[i] + df["Evt_MET_Pt"].values[i] + reco_lepton_4vec.Pt()))

						is_ttbar  = np.append(is_ttbar,0)
						bkg_scale = np.append(bkg_scale, 1./len(bkg))



				if self.Scale==0:
					false_comb = np.zeros(4)
					j1,k1,l1,m1 = 0,0,0,0
					df=df.append(df.iloc[[i]])
					#kombis suchen, die sich von bester kombi unterscheiden (beachte jets aus dem w duerfen auch nicht vertauschen) und selbst den anforderungen genuegen
					while(CSV[j1]<0.277 or CSV[k1]<0.277 or j1 == k1 or j1==l1 or j1==m1 or k1==l1 or k1==m1 or  l1==m1 or (j1==best_comb[0] and k1 == best_comb[1] and l1==best_comb[2] and m1 ==best_comb[3]) or (j1==best_comb[0] and k1 == best_comb[1] and l1==best_comb[3] and m1 ==best_comb[2])):
						j1 = randrange(njets)
						k1 = randrange(njets)
						l1 = randrange(njets)
						m1 = randrange(njets)
					false_comb = [j1,k1,l1,m1]

					n=0
					for index in jets:
						for index2 in pepec:
							globals()[index + "_" + index2] = np.append(globals()[index + "_" + index2],(df["Jet_" +str(index2) + "[" + str(false_comb[n]) + "]"].values[i]))
						n+=1

					reco_TopHad_4vec, reco_TopLep_4vec, reco_WHad_4vec, reco_WLep_4vec, reco_lepton_4vec, reco_neutrino_4vec = reconstruct_ttbar(df, i,false_comb[0],false_comb[1],false_comb[2],false_comb[3])

					for index in recos:
						globals()["reco_" + index + "_Pt"]   = np.append(globals()["reco_" + index + "_Pt"],	   locals()["reco_" + index + "_4vec"].Pt())
						globals()["reco_" + index + "_Eta"]  = np.append(globals()["reco_" + index + "_Eta"],	  locals()["reco_" + index + "_4vec"].Eta())
						globals()["reco_" + index + "_Phi"]  = np.append(globals()["reco_" + index + "_Phi"],	  locals()["reco_" + index + "_4vec"].Phi())
						globals()["reco_" + index + "_M"]	= np.append(globals()["reco_" + index + "_M"],		locals()["reco_" + index + "_4vec"].M())
						globals()["reco_" + index + "_logM"] = np.append(globals()["reco_" + index + "_logM"], log(locals()["reco_" + index + "_4vec"].M()))
	
					ttbar_phi = np.append(ttbar_phi, correct_phi(reco_TopHad_4vec.Phi() - reco_TopLep_4vec.Phi()))
					ttbar_Pt_div_Ht_p_Met = np.append(ttbar_Pt_div_Ht_p_Met,(reco_TopHad_4vec.Pt() + reco_TopLep_4vec.Pt())/(df["Evt_HT"].values[i] + df["Evt_MET_Pt"].values[i] + reco_lepton_4vec.Pt()))

					is_ttbar = np.append(is_ttbar,0)
					bkg_scale = np.append(bkg_scale, 1)



			# (15) Anteil Daten, die mitgenommen werden in Abhaengigkeit der hoehe des delta r cuttes auf die einzelnen jets
			for u in range(int(r_max*100)):
				if(minr_bhad<u/100. and minr_blep<u/100. and minr_q1<u/100. and minr_q2<u/100.):
					numb[u]+=1

				if(i==total_nr-1 and u%10 == 0):
					print "Anteil verwendeter Daten bei delta R cut ", u/100.," : ", numb[u]/total_nr
			Rb1[i]=minr_b1
			Rb2[i]=minr_b2
			Rb12[i]=minrH

		# counting how often b1-jet==b2-jet and b1 != b2 jet
		if(J==K):
			jkcounter = jkcounter+1
		else:
			jnkcounter = jnkcounter+1
	
		print("Higgs Events mit 1 Jet:")
		print jkcounter, "von", total_nr, "also", round(jkcounter/float(total_nr)*100,1),"%"
		print("Higgs Events mit 2 Jets:")
		print jnkcounter, "von", total_nr, "also", round(jnkcounter/float(total_nr)*100,1),"%"
		#print counter1,"mal delta R von b1 und b2 unter 0,7"
		#print counter2,"mal delta R von b1 und b2 unter 0,5"
		#print counter,"mal delta R von b1 und b2 unter 0,3"
		print"b1 und b2 unter 0.3:", counter3, "mal, also",round(counter3/float(total_nr)*100,1),"%"


		#---Histogramme Higgs delta R
		#plt.hist(Rb1,bins=200,range=(0,3))	#minR bis hoechstens 0.3
		#plt.show()
#		plt.hist(Rb2,bins=200,range=(0,3))
#		plt.show()
#		plt.savefig("DeltaR2bjets.pdf")
#		plt.hist(Rb12,bins=200,range=(0,3))
#		plt.show()
#		plt.savefig("DeltaRgesamtHbjets.pdf")

		x = np.arange(0,r_max,0.01)
		plt.plot(x, numb/total_nr)
		plt.title("Anteil der in der Aufbereitung beruecksichtigten Daten")
		plt.xlabel("akzeptiertes maximales $\Delta$R")
		plt.ylabel("Verhaeltnis akzeptierter Ereignisse zur Gesamtzahl an Ereignissen")
		plt.grid()
		plt.savefig("ratio_neu_btag.pdf")

		# (16) delete all variables except for:
		df_new = pd.DataFrame()
		variables_toadd  = ["N_Jets", "N_BTagsM", "Evt_Run", "Evt_Lumi", "Evt_ID", "Evt_MET_Pt", "Evt_MET_Phi", "Weight_GEN_nom", "Weight_XS", "Weight_CSV", "N_LooseElectrons", "N_TightMuons",
"Muon_Pt[0]", "Muon_Eta[0]", "Muon_Phi[0]","Muon_E[0]", "Electron_Pt[0]","Electron_Eta[0]","Electron_Phi[0]","Electron_E[0]","N_LooseMuons", "N_TightElectrons", "Evt_Odd"]

		for ind in variables_toadd:
			df_new[ind] = df[ind].values
	
		# (17) another short print for control
		#print df.shape, len(globals()["TopHad_B_" + index2]), len(is_ttbar)

		# (18) add correct and false combinations with their tags to df
		entr=0
		for index in jets:
			for index2 in pepec:
				df[index + "_" + index2] = globals()[index + "_" + index2]
				entr+=1
	
	
		for index in recos:
			for index2 in pepm:
				df["reco_" + index + "_" + index2]   = globals()["reco_" + index + "_" + index2]
				entr+=1
	
		print is_ttbar
		print ttbar_phi
		print df
	
		df["bkg_scale"] = bkg_scale
		df["Evt_is_ttbar"]  = is_ttbar
		df["ttbar_phi"] = ttbar_phi
		df["ttbar_pt_div_ht_p_met"] = ttbar_Pt_div_Ht_p_Met
	
			# (19) drop all columns except for those added just before and add variables_toadd again (easier than prooving every column if it is in variables_toadd before dropping)
		df.drop(df.columns[:-(entr+4)],inplace = True, axis = 1)
	
		for ind in variables_toadd:
			df[ind] = df_new[ind].values
	
	
			# (20) delete combinations of events without certain ttbar event
		i,j=0,0
		while(i<total_nr):
			if df["Evt_is_ttbar"].values[i]==0:
				df=df.drop(df.index[i],axis=0)
				total_nr -= 1
				j+=1
			else:
				i+=1
				j+=1
		print df.shape, df.columns

		return df

###########################################################################

	def findbestHiggs(self,df):
		

		# (1) total number of events in current df
		total_nr  = df["N_Jets"].size
		print total_nr, "Events in file."

		# array containing the number of jets per event
		njets_vec = df["N_Jets"].values

		# (2) just for plot that shows amount of events accepted as ttbar: x-axis: acceptance-niveau of delta r between jets and gen blep/bhad/q1/q2 (); r_max: max of x-axis
		r_max = 1.1
		numb = np.zeros(int(r_max*100))
		numbbkg = np.zeros(int(r_max*100))

		# (3) get gen-info of bjets needed for calculation of delta r at
		GenHiggs_B1_Phi = df["GenHiggs_B1_Phi"].values
		GenHiggs_B1_Eta = df["GenHiggs_B1_Eta"].values
		GenHiggs_B2_Phi = df["GenHiggs_B2_Phi"].values
		GenHiggs_B2_Eta = df["GenHiggs_B2_Eta"].values


		# (4) arrays for better clarity in (5),...; jets and reconstructed particles
		jets  = ["Higgs_B1","Higgs_B2"]
		pepec = ["Pt", "Eta", "Phi", "E", "CSV"]

		recos = ["Higgs"]
		pepm  = ["Pt", "Eta", "Phi", "M", "logM"]

		# (5) create arrays: TopHad_B_Pt etc. for all jets and reconstructed particles get filled in (13); information about whether an event is accepted as ttbar (is_ttbar); scaling-factor 			if all wrong combinations shall be considered as background if only one event: remains 1(bkg_scale)
		for index in jets:
			for index2 in pepec:
				globals()[index + "_" + index2] = np.zeros(total_nr)

		for index in recos:
			for index2 in pepm:
				globals()["reco_" + index + "_" + index2] = np.zeros(total_nr)


		is_Higgs = np.zeros(total_nr)
		bkg_scale = np.zeros(total_nr)+1
#		Rb1 = np.zeros(total_nr)
#		Rb2 = np.zeros(total_nr)
#		Rb12 = np.zeros(total_nr)
		DeltaR = np.zeros(total_nr)

		jkcounter = 0
		jnkcounter = 0


#		Ptbkg = np.zeros(total_nr)
#		Pthiggs = np.zeros(total_nr)
#		GenPt = np.zeros(total_nr)
#
#		Etabkg = np.full(total_nr,10.)
#		Etahiggs = np.full(total_nr,10.)
#		GenEta = np.full(total_nr,10.)
#
#		Pt1bkg = np.zeros(total_nr)
#		Pt1higgs = np.zeros(total_nr)
#		GenPt1 = np.zeros(total_nr)
#
#		Eta1bkg = np.full(total_nr,10.)
#		Eta1higgs = np.full(total_nr,10.)
#		GenEta1 = np.full(total_nr,10.)

		# (6) loop over all events
		for i in range(total_nr):	

			if df["Evt_ID"].values[i]%10 != 0:
				continue

			# (7) get number of jets in current event; create array for bkg scaling, shape(2,2) is to make np.append with new rows possible later (12). get deleted again in (12); minr and 			array for indices of best combination

			njets = min(int(njets_vec[i]),12)
#			minrH = 10000.

			if njets < 4:
				continue

			# (8) create arrays for entries of pepec and loop over all jets: df to arrays with len(njets), i.e. Jet_Pt (contains Pt info of all jets of this event)
			for index in pepec:
				globals()[index] = np.array([])

			for j in range(njets):
				for index2 in pepec:
					globals()[index2] = np.append(globals()[index2], df["Jet_" + str(index2) + "[" + str(j) + "]"].values[i])

#get Higgs Bjets---------------------

			minr_b1 = 10
			minr_b2 = 10
			best_comb = np.zeros(2)


			# Loop over all combinations of 2 b-jets: j for b1, k for b2
			for j in range(njets):
				# look for combination with smallest delta r between jets and genjets, if scale: add all relevant combinations except for right combination to bkg
				deltar_b1 = ((Eta[j]-GenHiggs_B1_Eta[i])**2  + (correct_phi(Phi[j] - GenHiggs_B1_Phi[i]))**2)**0.5
				if(deltar_b1 < minr_b1):
					minr_b1   = deltar_b1
					J = j
					best_comb[0]= j
			for k in range(njets):
				deltar_b2 = ((Eta[k]-GenHiggs_B2_Eta[i])**2  + (correct_phi(Phi[k] - GenHiggs_B2_Phi[i]))**2)**0.5
				if(deltar_b2 < minr_b2):
					minr_b2   = deltar_b2
					K = k
					best_comb[1]= k
			DeltaR[i] = ((Eta[J]-Eta[K])**2  + (correct_phi(Phi[J] - Phi[K]))**2)**0.5

#			minrH = minr_b1 + minr_b2

#			Rb1[i] = minr_b1
#			Rb2[i] = minr_b2
#			Rb12[i] = minrH

			# if there are no 2 matching bjets (each delta R < 0.3) at all, ignore event
			#if gibtesueberhauptbeidebjets!=1:
			#	continue
			#else:
			#	counter3=counter3+1
			#gibtesueberhauptbeidebjets = 0


			# (13) short print against long boring time
			if(i%1000 == 0):
				print "Event:",i," minr_b1:", minr_b1," minr_b2:", minr_b2


			n = 0
			# (14) only combinations with smallest total_deltar and each delta r <0.3 are ttbar-events (~50% of all events pass this); reconstruct further particles, quark jets not to be 					b-tagged
	 		if minr_b1 < 0.4 and minr_b2 < 0.4:	
				for index in jets:
					for index2 in pepec:
						globals()[index + "_" + index2][i] = df["Jet_" + str(index2) + "[" + str(int(best_comb[n])) + "]"].values[i]
					n+=1
				is_Higgs[i]=1

				reco_Higgs_4vec = reconstruct_Higgs(df, i,best_comb[0],best_comb[1])
	
				for index in recos:
					globals()["reco_" + index + "_Pt"][i]   =	 locals()["reco_" + index + "_4vec"].Pt()
					globals()["reco_" + index + "_Eta"][i]  =	 locals()["reco_" + index + "_4vec"].Eta()
					globals()["reco_" + index + "_Phi"][i]  =	 locals()["reco_" + index + "_4vec"].Phi()
					globals()["reco_" + index + "_M"][i]	=	 locals()["reco_" + index + "_4vec"].M()
					globals()["reco_" + index + "_logM"][i] = log(locals()["reco_" + index + "_4vec"].M())


#				Pthiggs[i] = globals()["reco_Higgs_Pt"][i]
#				GenPt[i] = df["GenHiggs_Pt"].values[i]
#				Etahiggs[i] = globals()["reco_Higgs_Eta"][i]
#				GenEta[i] = df["GenHiggs_Eta"].values[i]
#				Eta1higgs[i] = df["Jet_Eta["+str(J)+"]"].values[i]
#				GenEta1[i] = df["GenHiggs_B1_Eta"].values[i]
#				Pt1higgs[i] = df["Jet_Pt["+str(J)+"]"].values[i]
#				GenPt1[i] = df["GenHiggs_B1_Pt"].values[i]


				# counting how often b1-jet==b2-jet and b1 != b2 jet
				if(J==K):
					jkcounter=jkcounter+1
				else:
					jnkcounter=jnkcounter+1
#---------------------------


#				if self.Scale:
#					for nr in range(len(bkg)):
#						df=df.append(df.iloc[[i]])
#						false_comb = [int(bkg[nr][0]),int(bkg[nr][1])]
#
#						n=0
#						for index in jets:
#							for index2 in pepec:
#								globals()[index + "_" + index2] = np.append(globals()[index + "_" + index2],(df["Jet_" +str(index2) + "[" + str(false_comb[n]) + "]"].values[i]))
#							n+=1
#	
#						reco_Higgs_4vec = reconstruct_Higgs(df, i,false_comb[0],false_comb[1])
#
#						for index in recos:
#							globals()["reco_" + index + "_Pt"]   = np.append(globals()["reco_" + index + "_Pt"],	   locals()["reco_" + index + "_4vec"].Pt())
#							globals()["reco_" + index + "_Eta"]  = np.append(globals()["reco_" + index + "_Eta"],	  locals()["reco_" + index + "_4vec"].Eta())
#							globals()["reco_" + index + "_Phi"]  = np.append(globals()["reco_" + index + "_Phi"],	  locals()["reco_" + index + "_4vec"].Phi())
#							globals()["reco_" + index + "_M"]	= np.append(globals()["reco_" + index + "_M"],		locals()["reco_" + index + "_4vec"].M())
#							globals()["reco_" + index + "_logM"] = np.append(globals()["reco_" + index + "_logM"], log(locals()["reco_" + index + "_4vec"].M()))
#
#
#						is_Higgs  = np.append(is_Higgs,0)
#						bkg_scale = np.append(bkg_scale, 1./len(bkg))



				if self.Scale==0:
					false_comb = np.zeros(2)
					j1,k1 = J,K
					df=df.append(df.iloc[[i]])
	

					#kombis suchen, die sich von bester kombi unterscheiden (beachte jets aus dem w duerfen auch nicht vertauschen) und selbst den anforderungen genuegen

					while(j1 == best_comb[0] or j1 == best_comb[1]):
						j1 = randrange(njets)

					while(k1 == best_comb[0] or k1 == best_comb[1] or k1 == j1):
						k1 = randrange(njets)


					false_comb = [j1,k1]
					n=0
					for index in jets:
						for index2 in pepec:
							globals()[index + "_" + index2] = np.append(globals()[index + "_" + index2],(df["Jet_" +str(index2) + "[" + str(false_comb[n]) + "]"].values[i]))
						n+=1

					reco_Higgs_4vec = reconstruct_Higgs(df, i,false_comb[0],false_comb[1])

					for index in recos:
						globals()["reco_" + index + "_Pt"]   = np.append(globals()["reco_" + index + "_Pt"],	   locals()["reco_" + index + "_4vec"].Pt())
						globals()["reco_" + index + "_Eta"]  = np.append(globals()["reco_" + index + "_Eta"],	  locals()["reco_" + index + "_4vec"].Eta())
						globals()["reco_" + index + "_Phi"]  = np.append(globals()["reco_" + index + "_Phi"],	  locals()["reco_" + index + "_4vec"].Phi())
						globals()["reco_" + index + "_M"]	= np.append(globals()["reco_" + index + "_M"],		locals()["reco_" + index + "_4vec"].M())
						globals()["reco_" + index + "_logM"] = np.append(globals()["reco_" + index + "_logM"], log(locals()["reco_" + index + "_4vec"].M()))

					is_Higgs = np.append(is_Higgs,0)
					bkg_scale = np.append(bkg_scale, 1)

#					Ptbkg[i] = globals()["reco_Higgs_Pt"][total_nr+jkcounter+jnkcounter-1]
#					Etabkg[i] = globals()["reco_Higgs_Eta"][total_nr+jkcounter+jnkcounter-1]
#					Pt1bkg[i] = df["Jet_Pt["+str(j1)+"]"].values[jkcounter+jnkcounter-1]
#					Eta1bkg[i] = df["Jet_Eta["+str(j1)+"]"].values[jkcounter+jnkcounter-1]
					false_dr1 = ((Eta[j1]-GenHiggs_B1_Eta[i])**2  + (correct_phi(Phi[j1] - GenHiggs_B1_Phi[i]))**2)**0.5
					false_dr2 = ((Eta[k1]-GenHiggs_B1_Eta[i])**2  + (correct_phi(Phi[k1] - GenHiggs_B1_Phi[i]))**2)**0.5
					DeltaR = np.append(DeltaR,((Eta[j1]-Eta[k1])**2  + (correct_phi(Phi[j1] - Phi[k1]))**2)**0.5)
#					Rb1 = np.append(Rb1,0)
#					Rb2 = np.append(Rb2,0)
#					Rb12 = np.append(Rb12,0)



			# (15) Anteil Daten, die mitgenommen werden in Abhaengigkeit der hoehe des delta r cuttes auf die einzelnen jets
			for u in range(int(r_max*100)):
				if(minr_b1<u/100. and minr_b2<u/100.):
					numb[u]+=1
#				if(false_dr1<u/100. and false_dr2<u/100.):
#					numbbkg[u]+=1
				
				if(i==total_nr-1 and u%10 == 0):
					print "Anteil verwendeter Daten bei Delta R cut ", u/100.," : ", numb[u]/total_nr*100, "%"

		print("Higgs Events mit 1 Jet (nach 0.4 Krit):")
		print jkcounter, "von", total_nr, "also", round(jkcounter/float(total_nr)*100,3),"%"
		print("Higgs Events mit 2 Jets (nach 0.4 Krit):")
		print jnkcounter, "von", total_nr, "also", round(jnkcounter/float(total_nr)*100,2),"%"


		x = np.arange(0,r_max,0.01)
		plt.plot(x, numb/total_nr)
		#plt.plot(x, numbbkg/total_nr)
		plt.title("Anteil der in der Aufbereitung beruecksichtigten Daten")
		plt.xlabel("akzeptiertes maximales $\Delta$R")
		plt.ylabel("Verhaeltnis akzeptierter Ereignisse zur Gesamtzahl an Ereignissen")
		plt.grid()
		plt.savefig("ratio_neu_btag1.pdf")

		#---Histogramme Higgs delta R
		#plt.hist(Rb1,bins=200,range=(0,3))	
		#plt.show()
		#plt.hist(Rb2,bins=200,range=(0,3))
		#plt.ylabel("Anzahl Events")
		#plt.xlabel("Delta R von b2")
		#plt.show()
		#plt.savefig("DeltaR2bjets.pdf")
		#plt.hist(Rb12,bins=200,range=(0,3))
		#plt.ylabel("Anzahl Events")
		#plt.xlabel("Delta R von b1+b2")
		#plt.show()
		#plt.savefig("DeltaRgesamtHbjets.pdf")


		#Bins=100
		#b,e = np.histogram(Ptbkg,bins=Bins,range=(20,1000))
		#X=np.arange(20,1000,(1000-20)/float(Bins))
		#plt.plot(X,b,  'b-')
		#b,e = np.histogram(Pthiggs,bins=Bins,range=(20,1000))
		#plt.plot(X,b, 'g-')
		#b,e = np.histogram(GenPt,bins=Bins,range=(20,1000))
		#plt.plot(X,b, 'r-')
		#plt.ylabel("Anzahl Events")
		#plt.xlabel("Higgs Pt")
		#plt.ylim((0,total_nr/10))
		#plt.xlim((20,300))
		#plt.show()
		#plt.savefig("Pt.pdf")

#		b,e = np.histogram(Pt1bkg,bins=Bins,range=(20,1000))
#		X=np.arange(20,1000,(1000-20)/float(Bins))
#		plt.plot(X,b,  'b-')
#		b,e = np.histogram(Pt1higgs,bins=Bins,range=(20,1000))
#		plt.plot(X,b, 'g-')
#		b,e = np.histogram(GenPt1,bins=Bins,range=(20,1000))
#		plt.plot(X,b, 'r-')
#		plt.ylabel("Anzahl Events")
#		plt.xlabel("erster Higgs b-jet Pt")
#		plt.ylim((0,total_nr/5))
#		plt.xlim((20,300))
#		#plt.show()
#		plt.savefig("Pt1.pdf")


#		Bins=30
#		b1,e = np.histogram(Etabkg,bins=Bins,range=(-2.4,2.4))
#		X=np.arange(-2.4,2.4,(2.4+2.4)/float(Bins))
#		plt.plot(X,b1,'b-')
#		b1,e = np.histogram(Etahiggs,bins=Bins,range=(-2.4,2.4))
#		plt.plot(X,b1,'g-')
#		b1,e = np.histogram(GenEta,bins=Bins,range=(-2.4,2.4))
#		plt.plot(X,b1,'r-')
#		plt.ylabel("Anzahl Events")
#		plt.xlabel("Higgs Eta")
#		#plt.ylim((0,100))
#		#plt.show()
#		plt.savefig("Eta.pdf")

#		b1,e = np.histogram(Eta1bkg,bins=Bins,range=(-2.4,2.4))
#		X=np.arange(-2.4,2.4,(2.4+2.4)/float(Bins))
#		plt.plot(X,b1,'b-')
#		b1,e = np.histogram(Eta1higgs,bins=Bins,range=(-2.4,2.4))
#		plt.plot(X,b1,'g-')
#		b1,e = np.histogram(GenEta1,bins=Bins,range=(-2.4,2.4))
#		plt.plot(X,b1,'r-')
#		plt.ylabel("Anzahl Events")
#		plt.xlabel("erster Higgs b-jet Eta")
#		#plt.ylim((0,100))
#		#plt.show()
#		plt.savefig("Eta1.pdf")


		# (16) delete all variables except for:
		df_new = pd.DataFrame()
		variables_toadd  = ["GenHiggs_B1_Pt","GenHiggs_B1_E","GenHiggs_B1_Phi","GenHiggs_B1_Eta","GenHiggs_B2_Pt","GenHiggs_B2_E","GenHiggs_B2_Phi","GenHiggs_B2_Eta","N_Jets", "N_BTagsM", "Evt_Run","Evt_Lumi", "Evt_ID", "Evt_MET_Pt", "Evt_MET_Phi", "Weight_GEN_nom", "Weight_XS", "Weight_CSV", "N_LooseElectrons", "N_TightMuons","Muon_Pt[0]", "Muon_Eta[0]", "Muon_Phi[0]","Muon_E[0]","Electron_Pt[0]","Electron_Eta[0]","Electron_Phi[0]","Electron_E[0]","N_LooseMuons", "N_TightElectrons", "Evt_Odd"]

		for ind in variables_toadd:
			df_new[ind] = df[ind].values
		#print "df shape",df.shape,"df columns", df.columns,len(globals()["Higgs_B1_Pt"])
		# (18) add correct and false combinations with their tags to df
		entr=0
		for index in jets:
			for index2 in pepec:
				df[index + "_" + index2] = globals()[index + "_" + index2]
				entr+=1

		for index in recos:
			for index2 in pepm:
				df["reco_" + index + "_" + index2]   = globals()["reco_" + index + "_" + index2]
				entr+=1

#		df["DeltaR_B1"] = Rb1
#		df["DeltaR_B2"] = Rb2
#		df["DeltaR_B1_B2"] = Rb12
		df["DeltaR"] = DeltaR
		df["bkg_scale"] = bkg_scale
		df["Evt_is_Higgs"]  = is_Higgs
		#df["ttbar_phi"] = ttbar_phi
		#df["ttbar_pt_div_ht_p_met"] = ttbar_Pt_div_Ht_p_Met
		entr+=3
	
		# (19) drop all columns except for those added just before and add variables_toadd again (easier than prooving every column if it is in variables_toadd before dropping)
		df.drop(df.columns[:-(entr)],inplace = True, axis = 1)

		for ind in variables_toadd:
			df[ind] = df_new[ind].values


		# (20) delete combinations of events without certain ttH event
		i,j=0,0
		while(i<total_nr):
			if df["Evt_is_Higgs"].values[i]==0:
				df=df.drop(df.index[i],axis=0)
				total_nr -= 1
				j+=1
			else:
				i+=1
				j+=1
		print "df shape",df.shape,"df columns", df.columns

		return df
###########################################################################

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

#function to correct a difference of two angulars phi which is in [-2pi,2pi] to the correct interval [-pi,pi]
def correct_phi(phi):
	if(phi  <=  -np.pi):
		phi += 2*np.pi
	if(phi  >	np.pi):
		phi -= 2*np.pi
	return phi


# funtion to get 4-momenta of reconstuctable particles
def reconstruct_ttbar(df,i,j,k,l,m):
	# reconstruction of lepton-4vec
	if df["N_TightMuons"].values[i]:
		lepton_4vec=ROOT.TLorentzVector()
		lepton_4vec.SetPtEtaPhiE(df["Muon_Pt[0]"].values[i],df["Muon_Eta[0]"].values[i], df["Muon_Phi[0]"].values[i], df["Muon_E[0]"].values[i])
	if df["N_TightMuons"].values[i]==0 :
		lepton_4vec=ROOT.TLorentzVector()
		lepton_4vec.SetPtEtaPhiE(df["Electron_Pt[0]"].values[i], df["Electron_Eta[0]"].values[i], df["Electron_Phi[0]"].values[i], df["Electron_E[0]"].values[i])

	Pt_MET = df["Evt_MET_Pt"].values[i]
	Phi_MET = df["Evt_MET_Phi"].values[i]
	mW = 80.4

	# reconstruction  of neutrino_4vec
	neutrino_4vec = ROOT.TLorentzVector(Pt_MET*cos(Phi_MET),Pt_MET*sin(Phi_MET),0.,Pt_MET)
	mu = ((mW*mW)/2) + lepton_4vec.Px()*neutrino_4vec.Px()+ lepton_4vec.Py()*neutrino_4vec.Py()
	a = (mu*lepton_4vec.Pz())/(lepton_4vec.Pt()**2)
	a2 = a**2
	b = (lepton_4vec.E()**2*neutrino_4vec.Pt()**2 - mu**2)/(lepton_4vec.Pt()**2)
	if(a2<b):
		neutrino_4vec.SetPz(a)
	else:
		pz1=a+(a2-b)**0.5
		pz2=a-(a2-b)**0.5

		if(abs(pz1) <= abs(pz2)):
			neutrino_4vec.SetPz(pz1)
		else:
			neutrino_4vec.SetPz(pz2)
	neutrino_4vec.SetE(neutrino_4vec.P())

	combi = [j,k,l,m]
	ind = 0
	for index in ["TopHad_B", "TopLep_B", "TopHad_Q1", "TopHad_Q2"]:
		globals()[index + "_4vec"] = ROOT.TLorentzVector()
		globals()[index + "_4vec"].SetPtEtaPhiE(df["Jet_Pt[" + str(combi[ind])+"]"].values[i],df["Jet_Eta[" + str(combi[ind])+"]"].values[i],
		df["Jet_Phi[" + str(combi[ind])+"]"].values[i],df["Jet_E[" + str(combi[ind])+"]"].values[i])
		ind+=1

	#reconstructions
	whad_4vec = TopHad_Q1_4vec + TopHad_Q2_4vec
	wlep_4vec = neutrino_4vec + lepton_4vec
	thad_4vec = whad_4vec + TopHad_B_4vec
	tlep_4vec = wlep_4vec + TopLep_B_4vec

	return thad_4vec, tlep_4vec, whad_4vec, wlep_4vec, lepton_4vec, neutrino_4vec

# funtion to get 4-momenta of reconstuctable particles
def reconstruct_Higgs(df,i,j,k):

	combi = [int(j),int(k)]
	ind = 0
	for index in ["Higgs_B1", "Higgs_B2"]:
		globals()[index + "_4vec"] = ROOT.TLorentzVector()
		globals()[index + "_4vec"].SetPtEtaPhiE(df["Jet_Pt[" + str(combi[ind])+"]"].values[i],df["Jet_Eta[" + str(combi[ind])+"]"].values[i],
		df["Jet_Phi[" + str(combi[ind])+"]"].values[i],df["Jet_E[" + str(combi[ind])+"]"].values[i])
		ind+=1

	#reconstructions
	if j!=k:
		higgs_4vec = Higgs_B1_4vec + Higgs_B2_4vec
	else:
		higgs_4vec = Higgs_B1_4vec

	return higgs_4vec
	# ====================================================================

