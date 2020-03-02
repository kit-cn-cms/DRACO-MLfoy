import numpy as np
import pandas as pd
import uproot as root
import re
import glob
import os
from multiprocessing import Pool

import Reco_eval as Reco



# multi processing magic
def processChunk(info):
	info["self"].processChunk(info["sample"], info["chunk"], info["chunkNumber"])

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
	def __init__(self, outputdir, naming = "", addMEM = False, maxEntries = 50000, ZReco = False, HiggsReco = False, ncores = 1):
		# settings for paths
		self.outputdir  = outputdir
		self.naming	 = naming

		# generating output dir
		if not os.path.exists(self.outputdir):
			os.makedirs(self.outputdir)

		# settings for dataset
		self.addMEM	 = addMEM
		self.maxEntries = int(maxEntries)
		self.ZReco  = ZReco
		self.HiggsReco= HiggsReco

		# default values for some configs
		self.baseSelection  = None
		self.samples		= {}
		self.variables	  = []

		self.ncores		 = int(ncores)
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
		n_files = len(ntuple_files)
		print("number of files: {}".format(n_files))

		while not len(ntuple_files)%self.ncores == 0:
			ntuple_files.append("")
		ntuple_files = np.array(ntuple_files).reshape(self.ncores, -1)
	
		pool = Pool(self.ncores)
		chunks = [{"chunk": c, "sample": sample, "chunkNumber": i+1, "self": self} for i,c in enumerate(ntuple_files)]
		pool.map(processChunk, chunks)

		# concatenate single thread files
		self.mergeFiles(sample.categories.categories)

	def processChunk(self, sample, files, chunkNumber):
		files = [f for f in files if not f == ""]

		n_entries = 0
		concat_df = pd.DataFrame()
		n_files = len(files)
		for iF, f in enumerate(files):
			print("chunk #{}: starting file ({}/{}): {}".format(chunkNumber, iF+1, n_files, f))

			# open root file
			with root.open(f) as rf:
				# get MVATree
				try:
					tree = rf["MVATree"]
				except:
					print("could not open MVATree in ROOT file")
					return

			if tree.numentries == 0:
				print("MVATree has no entries - skipping file")
				return

			# convert to dataframe
			df = tree.pandas.df(self.variables)

			# delete subentry index
			df = df.reset_index( drop = True)

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



			# apply event selection
			df = self.applySelections(df, sample.selections)

			# perform ttZ or ttH reconstruction
			if self.ZReco:

				df = Reco.Eval(df, self.variables, "Z")
				sample.categories.categories["eval_allCombs"] = "(Event_Nr > 0)"

			if self.HiggsReco:

				df = Reco.Eval(df, self.variables, "Higgs")
				sample.categories.categories["eval_allCombs"] = "(Event_Nr > 0)"





			if concat_df.empty: concat_df = df
			else: concat_df = concat_df.append(df)			
			n_entries += df.shape[0]

			if (n_entries > self.maxEntries or f == files[-1]):
				print("*"*50)
				print("chunk #{}: max entries reached ...".format(chunkNumber))
				concat_df = self.addClassLabels(concat_df, sample.categories.categories)

				# add indexing
				concat_df.set_index(["Evt_Run", "Evt_Lumi", "Evt_ID"], inplace = True, drop = True)

				# add MEM variables
				if self.addMEM:
					concat_df = self.addMEMVariable(concat_df, mem_df)

				# remove trigger variables
				concat_df = self.removeTriggerVariables(concat_df)

				# write data to file
				self.createDatasets(concat_df, sample.categories.categories, chunkNumber)
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

	def createDatasets(self, df, categories, chunkNumber = None):
		for key in categories:
			if chunkNumber is None:
				outFile = self.outputdir+"/"+key+"_"+self.naming+".h5"
			else:
				outFile = self.outputdir+"/"+key+"_"+self.naming+"_"+str(chunkNumber)+".h5"

			# create dataframe for category
			cat_df = df.query("(class_label == \""+str(key)+"\")")
			print("creating dataset for class label {} with {} entries".format(key, cat_df.shape[0]))

			with pd.HDFStore(outFile, "a") as store:
				store.append("data", cat_df, index = False)
			
	def mergeFiles(self, categories):
		print("="*50)
		print("merging multicore threading files ...")
		for key in categories:
			print("category: {}".format(key))
			threadFiles = [ self.outputdir+"/"+key+"_"+self.naming+"_"+str(chunkNumber+1)+".h5" for chunkNumber in range(self.ncores) ]
			outFile = self.outputdir+"/"+key+"_"+self.naming+".h5"
			with pd.HDFStore(outFile, "a") as store:
				for f in threadFiles:
					store.append("data", pd.read_hdf(f), index = False)
					os.remove(f)
				print("number of events: {}".format(store.get_storer("data").nrows))
		print("="*50)

	def removeOldFiles(self):
		for key in self.samples:
			sample = self.samples[key]
			for cat in sample.categories.categories:
				outFile = self.outputdir+"/"+cat+"_"+self.naming+".h5"
				if os.path.exists(outFile):
					print("removing file {}".format(outFile))
					os.remove(outFile)

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



