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
base = "(N_Jets >= 4 and N_BTagsM >= 3 and Evt_MET_Pt > 20. and Weight_GEN_nom > 0.)"# and matchH_ft_RecoHiggs_matchable < 100. and matchZ_ft_RecoZ_matchable < 100.)"

# single lepton selections
single_mu_sel = "(N_LooseElectrons == 0 and N_TightMuons == 1 and Triggered_HLT_IsoMu24_vX == 1)"
single_el_sel = "(N_LooseMuons == 0 and N_TightElectrons == 1 and (Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX == 1 or ( Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX == 1 and Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX == 1 )))"

base_selection = "("+base+" and ("+single_mu_sel+" or "+single_el_sel+"))"

# define output classes
ttH_categories = root2pandas.EventCategories()
ttH_categories.addCategory("ttH",       selection = None)
ttH_categories.addCategory("ttHbb",     selection = "(matchH_ft_RecoX_matchable > 0.)")
ttH_categories.addCategory("ttHnonbb",  selection = "(matchH_ft_RecoX_matchable <= 0.)")
#ttH_categories.addCategory("ttX",       selection = "(matchH_ft_RecoHiggs_matchable < 100. and matchZ_ft_RecoZ_matchable < 100.)")
#ttH_categories.addCategory("ttXbb",     selection = "(matchH_ft_RecoHiggs_matchable > 0.)")
#ttH_categories.addCategory("ttXnonbb",  selection = "(matchH_ft_RecoHiggs_matchable <= 0.)")

ttZ_categories = root2pandas.EventCategories()
ttZ_categories.addCategory("ttZ",       selection = None)
ttZ_categories.addCategory("ttZbb",     selection = "(matchZ_ft_RecoX_matchable > 0.)")
ttZ_categories.addCategory("ttZnonbb",  selection = "(matchZ_ft_RecoX_matchable <= 0)")
#ttZ_categories.addCategory("ttX",       selection = "(matchH_ft_RecoHiggs_matchable < 100. and matchZ_ft_RecoZ_matchable < 100.)")
#ttZ_categories.addCategory("ttXbb",     selection = "(matchZ_ft_RecoZ_matchable > 0.)")
#ttZ_categories.addCategory("ttXnonbb",  selection = "(matchZ_ft_RecoZ_matchable <= 0.)")

ttbar_categories = root2pandas.EventCategories()
#ttbar_categories.addCategory("ttbar",   selection = None)
#ttbar_categories.addCategory("ttnonbb", selection = "(GenEvt_I_TTPlusBB == 0)")
ttbar_categories.addCategory("ttlf",    selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 0)")
#ttbar_categories.addCategory("ttcc",   selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 1)")
ttbar_categories.addCategory("ttcc",    selection = "(GenEvt_I_TTPlusCC >= 1 and GenEvt_I_TTPlusBB == 0)")
#ttbar_categories.addCategory("ttbb5FS", selection = "(GenEvt_I_TTPlusBB >= 1 and GenEvt_I_TTPlusCC == 0)")
#ttbar_categories.addCategory("ttTobb",  selection = "(GenEvt_I_TTPlusBB >= 1 and GenEvt_I_TTPlusCC == 0)")

ttbb_categories = root2pandas.EventCategories()
ttbb_categories.addCategory("ttbb",     selection = "(GenEvt_I_TTPlusBB >= 1 and GenEvt_I_TTPlusCC == 0)")

#ttcc_categories = root2pandas.EventCategories()
#ttcc_categories.addCategory("ttcc",     selection = "(GenEvt_I_TTPlusCC >= 1 and GenEvt_I_TTPlusBB == 0)")

#ttTobb_categories = root2pandas.EventCategories()
#ttTobb_categories.addCategory("ttTobb",     selection = "(GenEvt_I_TTPlusBB >= 1 and GenEvt_I_TTPlusCC == 0)")



ntuplespath = "/nfs/dust/cms/user/vdlinden/legacyTTZ/ntuples/2017"
ftpath = "/nfs/dust/cms/user/larmbrus/combined_ttZ_ttH/ntuples/2017/new_ntuples/multiclassJAN"
friendTrees = {
    "dnnZ": ftpath+"/recoX/bkg_merging/cTag_infos/v1/recoZ",
    "dnnH": ftpath+"/recoX/bkg_merging/cTag_infos/v1/recoHiggs",
    "dnnbb": ftpath+"/recoX/bkg_merging/cTag_infos/v1/recobb",
    "dnncc": ftpath+"/recoX/bkg_merging/cTag_infos/v1/recocc",
    "dnnttTobb": ftpath+"/recoX/bkg_merging/cTag_infos/v1/reco_bbfromttbar",
    #"dnnaddC": ftpath+"/recoX/bkg_merging/cTag_infos/v1/reco",
    "matchZ": ftpath+"/matchX/cTag_infos/v1/match_Z_as_X",
    "matchH": ftpath+"/matchX/cTag_infos/v1/match_Higgs_as_X",
    "chi2Z": ftpath+"/recoX/chi2_test/v3/reco_Z_chi2",
    "chi2H": ftpath+"/recoX/chi2_test/v3/reco_Higgs_chi2",
    }

# initialize dataset class
dataset = root2pandas.Dataset(
    outputdir   = outputdir,
    naming      = options.Name,
    addMEM      = options.MEM,
    maxEntries  = options.maxEntries,
    ttbarReco   = options.ttbarReco,
    friendTrees = friendTrees,
    ncores      = options.ncores)

# add base event selection
dataset.addBaseSelection(base_selection)



# add samples to dataset
dataset.addSample(
    sampleName  = "ttH",
    ntuples     = ntuplespath+"/ttHTobb*/*nominal*.root",
    categories  = ttH_categories,
    selections   = "(Evt_Odd == 1)",
    lumiWeight  = 41.5,
    )
dataset.addSample(
    sampleName  = "ttZ",
    ntuples     = ntuplespath+"/TTZToQQ*/*nominal*.root",
    categories  = ttZ_categories,
    selections   = "(Evt_Odd == 1)",
    lumiWeight  = 41.5,
    )

dataset.addSample(
    sampleName  = "ttbar",
    ntuples     = ntuplespath+"/TTToSemiLeptonic_TuneCP5_*/*nominal*.root",
    categories  = ttbar_categories,
    lumiWeight  = 41.5,
    )

dataset.addSample(
    sampleName  = "ttbb",
    ntuples     = ntuplespath+"/TTbb*/*nominal*.root",
    categories  = ttbb_categories,
    selections   = "(Evt_Odd == 1)",
    lumiWeight  = 41.5,
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
    "Weight_btagSF",
    "Weight_GEN_nom",
    "Evt_ID", 
    "Evt_Run", 
    "Evt_Lumi"]

# add these variables to the variable list
dataset.addVariables(additional_variables)

# run the preprocessing
dataset.runPreprocessing()
