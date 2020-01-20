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

parser.add_option("-o", "--outputdirectory", dest="outputDir",default="/nfs/dust/cms/user/vdlinden/ttZAnalysis/DNNInputs",
        help="DIR for output", metavar="outputDir")

parser.add_option("-v", "--variableselection", dest="variableSelection",default="allVariables",
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
single_el_sel = "(N_LooseMuons == 0 and N_TightElectrons == 1 and (Triggered_HLT_Ele32_WPTight_Gsf_vX == 1 or Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX == 1))"

base_selection = "("+base+" and ("+single_mu_sel+" or "+single_el_sel+"))"

# define output classes
ttH_categories = root2pandas.EventCategories()
ttH_categories.addCategory("ttH", selection = None)

'''
ttbar_categories_bb = root2pandas.EventCategories()
ttbar_categories_bb.addCategory("ttbb",        selection = "(GenEvt_I_TTPlusBB >= 1 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories_else = root2pandas.EventCategories()
ttbar_categories_else.addCategory("ttlf",        selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories_else.addCategory("ttcc",        selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 1)")

ttZ_categories = root2pandas.EventCategories()
ttZ_categories.addCategory("ttZ", selection = None)
'''

# initialize dataset class
dataset = root2pandas.Dataset(
    outputdir   = outputdir,
    naming      = options.Name,
    addMEM      = options.MEM,
    maxEntries  = options.maxEntries)

# add base event selection
dataset.addBaseSelection(base_selection)



ntuplesPath = "/nfs/dust/cms/user/lbosch/ntuple_production/ntuple_v5/"

# add samples to dataset

dataset.addSample(
    sampleName  = "ttH",
    ntuples     = ntuplesPath+"/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
    categories  = ttH_categories,
    even_odd    = options.even_odd_splitting,
    selections  = "(Evt_Odd==1)",     ## split
    ) 

'''
dataset.addSample(	#for 2017 samples the files seem to be splitted
    sampleName  = "ttbar_bb",
    ntuples     = ntuplesPath+"/TTbb_Powheg_Openloops_new_pmx/*nominal*.root",
    categories  = ttbar_categories_bb,
    even_odd    = options.even_odd_splitting,#    selections  = "(Evt_Odd==1)"
    selections  = "(Evt_Odd==1)"   
    )

dataset.addSample(	#for 2017 samples the files seem to be splitted
    sampleName  = "ttbar_else",
    ntuples     = ntuplesPath+"/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
    categories  = ttbar_categories_else,
    even_odd    = options.even_odd_splitting,
    selections  = None , ## no splitting
    )


dataset.addSample(
    sampleName  = "ttZ",
    ntuples     = ntuplesPath+"/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/*nominal*.root",
    categories  = ttZ_categories,
    even_odd    = options.even_odd_splitting,
    selections  = "(Evt_Odd==1)"
    )
'''

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
