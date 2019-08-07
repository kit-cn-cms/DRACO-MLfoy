import os
import sys
import optparse
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(os.path.dirname(filedir))
sys.path.append(basedir)

import root2pandas

"""
USE: python preprocessing.py --outputdirectory=DIR --variableSelection=FILE --maxentries=INT --MEM=BOOL
"""
usage="usage=%prog [options] \n"
usage+="USE: python preprocessing.py --outputdirectory=DIR --variableselection=FILE --maxentries=INT --MEM=BOOL --name=STR\n"
usage+="OR: python preprocessing.py -o DIR -v FILE -e INT -m BOOL -n STR"

parser = optparse.OptionParser(usage=usage)

parser.add_option("-o", "--outputdirectory", dest="outputDir",default="InputFeatures",
        help="DIR for output", metavar="outputDir")

parser.add_option("-v", "--variableselection", dest="variableSelection",default="example_variables",
        help="FILE for variables used to train DNNs", metavar="variableSelection")

parser.add_option("-e", "--maxentries", dest="maxEntries", default=50000,
        help="INT used for maximal number of entries for each batch (to restrict memory usage)", metavar="maxEntries")

parser.add_option("-m", "--MEM", dest="MEM", action = "store_true", default=False,
        help="BOOL to use MEM or not", metavar="MEM")

parser.add_option("-n", "--name", dest="Name", default="dnn",
        help="STR of the output file name", metavar="Name")

parser.add_option("--odd",dest="even_odd_splitting",default=False,action="store_true",
        help="BOOL to activate even odd splitting (only process events with 'Evt_Odd==1'). default is FALSE (all events are processed)")

(options, args) = parser.parse_args()

if not os.path.isabs(options.variableSelection):
    sys.path.append(basedir+"/variable_sets/")
    variable_set = __import__(options.variableSelection)
elif os.path.exists(options.variableSelection):
    variable_set = __import__(options.variableSelection)
else:
    sys.exit("ERROR: Variable Selection File does not exist!")

if not os.path.isabs(options.outputDir):
    outputdir = basedir+"/workdir/"+options.outputDir
elif os.path.exists(options.outputDir) or os.path.exists(os.path.dirname(options.outputDir)):
    outputdir=options.outputDir
else:
    sys.exit("ERROR: Output Directory does not exist!")

# define a base event selection which is applied for all Samples
# select only events with GEN weight > 0 because training with negative weights is weird
base = "(N_Jets >= 4 and N_BTagsM >= 3 and Evt_MET_Pt > 20. and Weight_GEN_nom > 0.)"

# single lepton selections
single_mu_sel = "(N_LooseElectrons == 0 and N_TightMuons == 1 and Triggered_HLT_IsoMu24_vX == 1)"
single_el_sel = "(N_LooseMuons == 0 and N_TightElectrons == 1 and Triggered_HLT_Ele32_WPTight_Gsf_vX == 1)"

base_selection = "("+base+" and ("+single_mu_sel+" or "+single_el_sel+"))"

# define output classes
ttH_categories = root2pandas.EventCategories()
ttH_categories.addCategory("ttH", selection = None)

ttbar_categories = root2pandas.EventCategories()
ttbar_categories.addCategory("tthf",        selection = "(GenEvt_I_TTPlusBB >= 1 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories.addCategory("ttmb",        selection = "((GenEvt_I_TTPlusBB == 1 or GenEvt_I_TTPlusBB == 3) and GenEvt_I_TTPlusCC == 0)")
ttbar_categories.addCategory("ttbb",        selection = "(GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories.addCategory("tt2b",        selection = "(GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories.addCategory("ttb",         selection = "(GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories.addCategory("ttlf",        selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories.addCategory("ttcc",        selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 1)")

# initialize dataset class
dataset = root2pandas.Dataset(
    outputdir   = outputdir,
    naming      = options.Name,
    addMEM      = options.MEM,
    maxEntries  = options.maxEntries)

# add base event selection
dataset.addBaseSelection(base_selection)



ntuplesPath2017 = "/nfs/dust/cms/user/swieland/ttH_legacy/ntuple/2017"
ntuplesPath2018 = "/nfs/dust/cms/user/swieland/ttH_legacy/ntuple/2018"

# add samples to dataset
dataset.addSample(
    sampleName  = "ttH2017",
    ntuples     = ntuplesPath2017+"/ttH*/*nominal*.root",
    categories  = ttH_categories,
    dataera     = 2017,
    even_odd    = options.even_odd_splitting,
   ) 

dataset.addSample(
    sampleName  = "ttH2018",
    ntuples     = ntuplesPath2018+"/ttH*/*nominal*.root",
    categories  = ttH_categories,
    dataera     = 2018,
    even_odd    = options.even_odd_splitting,
   ) 

dataset.addSample(
    sampleName  = "TTTo2017",
    ntuples     = ntuplesPath2017+"/TTTo*/*nominal*.root",
    categories  = ttbar_categories,
    dataera     = 2017,
    even_odd    = options.even_odd_splitting,
      )

dataset.addSample(
    sampleName  = "TTTo2018",
    ntuples     = ntuplesPath2018+"/TTTo*/*nominal*.root",
    categories  = ttbar_categories,
    dataera     = 2018,
    even_odd    = options.even_odd_splitting,
      )


# initialize variable list 
dataset.addVariables(variable_set.all_variables)

# define an additional variable list
additional_variables = [
    "Evt_Odd",
    "N_Jets",
    "N_BTagsL",
    "N_BTagsM",
    "N_BTagsT",
    "Weight_XS",
    "Weight_CSV",
    "Weight_GEN_nom",
    "Evt_ID", 
    "Evt_Run", 
    "Evt_Lumi"]

# add these variables to the variable list
dataset.addVariables(additional_variables)

# run the preprocessing
dataset.runPreprocessing()
