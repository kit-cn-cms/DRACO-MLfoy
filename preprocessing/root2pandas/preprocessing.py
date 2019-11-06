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

parser.add_option("-s", "--scale", dest="Scale", action= "store_true", default=False,
        help="BOOl to use every wrong bkg event (1) or just one (0)", metavar="Scale")


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
base = "(N_Jets<14 and N_Jets >= 4 and N_BTagsM >= 2 and Weight_GEN_nom > 0.)"


# single lepton selections
single_mu_sel = "(N_LooseElectrons == 0 and N_TightMuons == 1)"

single_el_sel = "(N_LooseMuons == 0 and N_TightElectrons == 1)"

base_selection = "("+base+" and ("+single_mu_sel+" or "+single_el_sel+"))"

ttbar_selection = "(Evt_Odd == 1)"

ttbar_categories = root2pandas.EventCategories()
ttbar_categories.addCategory("ttH")
#ttbar_categories.addCategory("ttbar")
ttbar_categories.addCategory("bkg")

# initialize dataset class
dataset = root2pandas.Dataset(
    outputdir   = outputdir,
    naming      = options.Name,
    addMEM      = options.MEM,
    maxEntries  = options.maxEntries,
    Scale       = options.Scale)

# add base event selection
dataset.addBaseSelection(base_selection)



#ntuplesPath = "/nfs/dust/cms/user/mwassmer/ttH_2019/ntuples_2018/"
# ntuplesPath = "/nfs/dust/cms/user/vdlinden/legacyTTH/ntuples/legacy_2018_ttZ/"
# ntuplesPath = "/nfs/dust/cms/user/vdlinden/legacyTTH/ntuples/legacy_2018_ttZ_v2/"
ntuplesPath = "/nfs/dust/cms/user/swieland/ttH_legacy/ntuple/2017/"



dataset.addSample(
    sampleName  = "TTHbb",
    #ntuples     = "/ceph/mheim/ntuples/ttbar.root",
    #ntuples     = "/ceph/mheim/ntuples/ttH.root",
    ntuples     = ntuplesPath+"ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/*nominal*.root",
    categories  = ttbar_categories,
    selections  = ttbar_selection
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
import time
start_time = time.time()
# your script
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
# add these variables to the variable list
dataset.addVariables(additional_variables)

# run the preprocessing
dataset.runPreprocessing()
