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

parser.add_option("--ttbarReco", dest="ttbarReco", action= "store_true", default=False,
        help="activate preprocessing for ttbar reconstruction", metavar="ttbarReco")

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
base = "(N_Jets >= 4 and N_BTagsM >= 3 and RecoX_matchable > 0.)"

base_selection = "("+base+")"

# define output classes
sig_Z_categories = root2pandas.EventCategories()
sig_Z_categories.addCategory("Zbb", selection = None)

sig_Higgs_categories = root2pandas.EventCategories()
sig_Higgs_categories.addCategory("Hbb", selection = None)

sig_cc_categories = root2pandas.EventCategories()
sig_cc_categories.addCategory("cc", selection = None)

sig_bb_categories = root2pandas.EventCategories()
sig_bb_categories.addCategory("bb", selection = None)

sig_bbfromttbar_categories = root2pandas.EventCategories()
sig_bbfromttbar_categories.addCategory("ttTobb", selection = None)

bkg_categories = root2pandas.EventCategories()
bkg_categories.addCategory("bkg", selection = None)


# initialize dataset class
dataset = root2pandas.Dataset(
    outputdir   = outputdir,
    naming      = options.Name,
    addMEM      = options.MEM,
    maxEntries  = options.maxEntries,
    ttbarReco   = options.ttbarReco,
    ncores      = options.ncores)

# add base event selection
dataset.addBaseSelection(base_selection)


ntuplesPath_Higgs = "/nfs/dust/cms/user/larmbrus/combined_ttZ_ttH/ntuples/2017/new_ntuples/multiclassJAN/matchX/cTag_infos/v1/match_Higgs_as_X/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8_new_pmx"
ntuplesPath_Z = "/nfs/dust/cms/user/larmbrus/combined_ttZ_ttH/ntuples/2017/new_ntuples/multiclassJAN/matchX/cTag_infos/v1/match_Z_as_X/TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8"
ntuplesPath_cc = "/nfs/dust/cms/user/larmbrus/combined_ttZ_ttH/ntuples/2017/new_ntuples/multiclassJAN/matchX/cTag_infos/v1/match_cc_as_X/*"
ntuplesPath_bb = "/nfs/dust/cms/user/larmbrus/combined_ttZ_ttH/ntuples/2017/new_ntuples/multiclassJAN/matchX/cTag_infos/v1/match_bb_as_X/TTbb_Powheg_Openloops_new_pmx"
ntuplesPath_bbfromttbar = "/nfs/dust/cms/user/larmbrus/combined_ttZ_ttH/ntuples/2017/new_ntuples/multiclassJAN/matchX/cTag_infos/v1/match_bbfromttbar/TTbb_Powheg_Openloops_new_pmx"

# add samples to dataset
dataset.addSample(
    sampleName  = "Hbb",
    ntuples     = ntuplesPath_Higgs+"/*Tree.root",
    categories  = sig_Higgs_categories,
    lumiWeight  = 41.5,
    )

dataset.addSample(
    sampleName  = "bkg_Higgs",
    ntuples     = ntuplesPath_Higgs+"/*bkg.root",
    categories  = bkg_categories,
    lumiWeight  = 41.5,
    )

dataset.addSample(
    sampleName  = "Zbb",
    ntuples     = ntuplesPath_Z+"/*Tree.root",
    categories  = sig_Z_categories,
    lumiWeight  = 41.5,
    )

dataset.addSample(
    sampleName  = "bkg_Z",
    ntuples     = ntuplesPath_Z+"/*bkg.root",
    categories  = bkg_categories,
    lumiWeight  = 41.5,
    )

dataset.addSample(
    sampleName  = "cc",
    ntuples     = ntuplesPath_cc+"/*Tree.root",
    categories  = sig_cc_categories,
    lumiWeight  = 41.5,
    )

dataset.addSample(
    sampleName  = "bkg_cc",
    ntuples     = ntuplesPath_cc+"/*bkg.root",
    categories  = bkg_categories,
    lumiWeight  = 41.5,
    )

dataset.addSample(
    sampleName  = "bb",
    ntuples     = ntuplesPath_bb+"/*Tree.root",
    categories  = sig_bb_categories,
    lumiWeight  = 41.5,
    )

dataset.addSample(
    sampleName  = "bkg_bb",
    ntuples     = ntuplesPath_bb+"/*bkg.root",
    categories  = bkg_categories,
    lumiWeight  = 41.5,
    )

dataset.addSample(
    sampleName  = "ttTobb",
    ntuples     = ntuplesPath_bbfromttbar+"/*Tree.root",
    categories  = sig_bbfromttbar_categories,
    lumiWeight  = 41.5,
    )

dataset.addSample(
    sampleName  = "bkg_bbfromttbar",
    ntuples     = ntuplesPath_bbfromttbar+"/*bkg.root",
    categories  = bkg_categories,
    lumiWeight  = 41.5,
    )

# initialize variable list 
dataset.addVariables(variable_set.all_variables)

# define an additional variable list
additional_variables = [
    "Evt_Odd",
    "N_Jets",
    "N_BTagsM",
    "Weight_XS",
    "Weight_btagSF",
    "Weight_GEN_nom",
    "Evt_ID", 
    "Evt_Run", 
    "Evt_Lumi"]

# add these variables to the variable list
dataset.addVariables(additional_variables)

# run the preprocessing
dataset.runPreprocessing()
