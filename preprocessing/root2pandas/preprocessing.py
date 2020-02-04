import os
import sys
import optparse
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(os.path.dirname(filedir))
sys.path.append(basedir)

import root2pandas as root2pandas

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

parser.add_option("--ZReco", dest="ZReco", action= "store_true", default=True,
        help="activate preprocessing for Z reconstruction", metavar="ZReco")

parser.add_option("--cores", dest="ncores", default = 1,
        help="number of cores for parallel multiprocessing")


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
# N<13 could be removed, because in the given dataset there is no event with N>12
base = "(N_Jets >= 4 and N_BTagsM >= 2 and Weight_GEN_nom > 0.)"


# single lepton selections
single_mu_sel = "(N_LooseElectrons == 0 and N_TightMuons == 1)"

single_el_sel = "(N_LooseMuons == 0 and N_TightElectrons == 1)"

base_selection = "("+base+" and ("+single_mu_sel+" or "+single_el_sel+"))"

ttZ_selection = "(Evt_Odd == 1)"

ttZ_categories = root2pandas.EventCategories()
ttZ_categories.addCategory("ttZ")
ttZ_categories.addCategory("bkg")
#ttZ_categories.addCategory("TopBkg")
#ttbar_categories.addCategory("ttbar", selection = "Evt_Odd==1")

# initialize dataset class
dataset = root2pandas.Dataset(
    outputdir   = outputdir,
    naming      = options.Name,
    addMEM      = options.MEM,
    maxEntries  = options.maxEntries,
    ZReco   = options.ZReco,
    ncores      = options.ncores)

# add base event selection
dataset.addBaseSelection(base)
#ntuplesPath = "/nfs/dust/cms/user/swieland/ttH_legacy/ntupleHadded_2017/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root"
#ntuplesPath = "/nfs/dust/cms/user/swieland/ttH_legacy/ntupleHadded_2017/ttHTobb_ttToSemiLep_M125_TuneCP5_13TeV-powheg-pythia8/*nominal*.root"
ntuplesPath = "/nfs/dust/cms/user/lbosch/ntuple_production/ntuple_v5/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8/*nominal*.root"


dataset.addSample(
    sampleName  = "TTToSL",
    ntuples     = ntuplesPath,
    categories  = ttZ_categories,
    selections  = ttZ_selection
)
# initialize variable list
dataset.addVariables(variable_set.all_variables)

# define an additional variable list
additional_variables = [
    "N_Jets",
    "N_BTagsM",
    "Weight_XS",
    "Weight_CSV",
    "Weight_GEN_nom",
    "Evt_ID",
    "Evt_Run",
    "Evt_Lumi"
    ]
# add these variables to the variable list
dataset.addVariables(additional_variables)

# run the preprocessing
dataset.runPreprocessing()

