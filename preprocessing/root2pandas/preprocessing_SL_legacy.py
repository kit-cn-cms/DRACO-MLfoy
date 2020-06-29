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
base = "(N_Jets >= 4 and N_BTagsM >= 3 and Evt_MET_Pt > 20. and Weight_GEN_nom > 0.)"
base_selection = "("+base+")"

# single lepton selections # me = IMPORTANT; hier vordefiniert for 3 years
single_mu_sel_16 = "(N_LooseElectrons == 0 and N_TightMuons == 1 and (Triggered_HLT_IsoMu24_vX == 1 or Triggered_HLT_IsoTkMu24_vX == 1))"
single_el_sel_16 = "(N_LooseMuons == 0 and N_TightElectrons == 1 and Triggered_HLT_Ele27_WPTight_Gsf_vX == 1)"
trigger_sel_16 = "("+single_mu_sel_16+" or "+single_el_sel_16+")"

single_mu_sel_17 = "(N_LooseElectrons == 0 and N_TightMuons == 1 and Triggered_HLT_IsoMu27_vX == 1)"
single_el_sel_17 = "(N_LooseMuons == 0 and N_TightElectrons == 1 and (Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX == 1 or ( Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX == 1 and Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX == 1 )))"
trigger_sel_17 = "("+single_mu_sel_17+" or "+single_el_sel_17+")"

single_mu_sel_18 = "(N_LooseElectrons == 0 and N_TightMuons == 1 and Triggered_HLT_IsoMu24_vX == 1)"
single_el_sel_18 = "(N_LooseMuons == 0 and N_TightElectrons == 1 and (Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX == 1 or Triggered_HLT_Ele32_WPTight_Gsf_vX == 1))"
trigger_sel_18 = "("+single_mu_sel_18+" or "+single_el_sel_18+")"

# define output classes
STXS_Stage0 = "abs(GenHiggs_Eta) < 2.5"

# define output classes
ttH_categories = root2pandas.EventCategories()
ttH_categories.addCategory("ttH", selection = None)
# me ttH_categories.addCategory("ttH_STXS_0", selection = STXS_Stage0 + " and GenHiggs_Pt >= 0 and GenHiggs_Pt < 60")
# me ttH_categories.addCategory("ttH_STXS_1", selection = STXS_Stage0 + " and GenHiggs_Pt >= 60 and GenHiggs_Pt < 120")
# me ttH_categories.addCategory("ttH_STXS_2", selection = STXS_Stage0 + " and GenHiggs_Pt >= 120 and GenHiggs_Pt < 200")
# me ttH_categories.addCategory("ttH_STXS_3", selection = STXS_Stage0 + " and GenHiggs_Pt >= 200 and GenHiggs_Pt < 300")
# me ttH_categories.addCategory("ttH_STXS_4", selection = STXS_Stage0 + " and GenHiggs_Pt >= 300 ")

#unimportant for me
# tHW_categories = root2pandas.EventCategories()
# tHW_categories.addCategory("tHW", selection = None)
# 
# tHq_categories = root2pandas.EventCategories()
# tHq_categories.addCategory("tHq", selection = None)

ttbar_categories = root2pandas.EventCategories()
ttbar_categories.addCategory("ttlf",        selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 0)")
ttbar_categories.addCategory("ttcc",        selection = "(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 1)")

ttbb_categories = root2pandas.EventCategories()
ttbb_categories.addCategory("tthf",        selection = "(GenEvt_I_TTPlusBB >= 1 and GenEvt_I_TTPlusCC == 0)")
# ttbb_categories.addCategory("ttmb",        selection = "((GenEvt_I_TTPlusBB == 1 or GenEvt_I_TTPlusBB == 3) and GenEvt_I_TTPlusCC == 0)")
#ttbar_categories.addCategory("ttbb",        selection = "(GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0)")
#ttbb_categories.addCategory("ttb",         selection = "(GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0)")
# ttbb_categories.addCategory("tt2b",        selection = "(GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0)")

#ttZ_categories = root2pandas.EventCategories()
#ttZ_categories.addCategory("ttZ", selection = None)

#tH_categories = root2pandas.EventCategories()
#tH_categories.addCategory("tH", selection = None)

#ST_categories = root2pandas.EventCategories()
#ST_categories.addCategory("ST", selection = None)

friendTrees_2018 = {
    "matchH": "/nfs/dust/cms/user/vdlinden/legacyTTH/ntuples/friendTrees/matchHiggs_2018_v2/",
    }

friendTrees_2017 = {
    "matchH": "/nfs/dust/cms/user/swieland/ttH_legacy/friendTrees/matchHiggs/2017/",
    }

friendTrees_2016 = {
    "matchH": "/nfs/dust/cms/user/swieland/ttH_legacy/friendTrees/matchHiggs/2016/",
    }

# initialize dataset class
dataset = root2pandas.Dataset(
    outputdir   = outputdir,
    naming      = options.Name,
    addMEM      = options.MEM,
    maxEntries  = options.maxEntries,
    ttbarReco   = options.ttbarReco,
    # friendTrees = friendTrees,
    ncores      = options.ncores)

# add base event selection
dataset.addBaseSelection(base_selection)



#ntuplesPath2018 = "/nfs/dust/cms/user/vdlinden/legacyTTH/ntuples/legacy_2018_ttH_newJEC/"
#nur 2017
ntuplesPath2017 = "/nfs/dust/cms/user/swieland/ttH_legacy/ntupleHadded_2017_JECgroups_newGT/"
#ntuplesPath2016 = "/nfs/dust/cms/user/vdlinden/legacyTTH/ntuples/legacy_2016_ttH_newJEC_v3/"
memPath = "/nfs/dust/cms/user/swieland/ttH_legacy/MEMdatabase/mem_final/"

# add samples to dataset
# dataset.addSample(
#     sampleName  = "ttH_18",
#     ntuples     = ntuplesPath2018+"/ttHTobb_ttToSemiLep_M125_TuneCP5_13TeV-powheg-pythia8/*nominal*.root",
#     categories  = ttH_categories,
#     lumiWeight  = 59.7,
#     selections  = trigger_sel_18,
#     #selections  = "(Evt_Odd == 1)",
#     MEMs        = memPath+"2018/ttHTobb_ttToSemiLep_M125_TuneCP5_13TeV-powheg-pythia8.root",
#     ) 

dataset.addSample(
    sampleName  = "ttH_17",
    ntuples     = ntuplesPath2017+"/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
    categories  = ttH_categories,
    lumiWeight  = 41.5,
    selections  = "(Evt_Odd == 1) and " + trigger_sel_17,
    # selections  = "(Evt_Odd == 1)",
    MEMs        = memPath+"2017/ttHTobb_ttToSemiLep_M125_TuneCP5_13TeV-powheg-pythia8.root",
    )

# dataset.addSample(
#     sampleName  = "ttH_16",
#     ntuples     = ntuplesPath2016+"/ttHTobb_ttToSemiLep_M125_TuneCP5_13TeV-powheg-pythia8/*nominal*.root",
#     categories  = ttH_categories,
#     lumiWeight  = 35.9,
#     selections  = trigger_sel_16,
#     #selections  = "(Evt_Odd == 1)",
#     MEMs        = memPath+"2016/ttHTobb_ttToSemiLep_M125_TuneCP5_13TeV-powheg-pythia8.root",
#     )


# dataset.addSample(
#     sampleName  = "TTToSL_18",
#     ntuples     = ntuplesPath2018+"/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/*nominal*.root",
#     categories  = ttbar_categories,
#     lumiWeight  = 59.7,
#     selections   = "(Evt_Odd == 1) and " + trigger_sel_18 ,
#     MEMs        = memPath+"2018/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.root",
#     )

dataset.addSample(
    sampleName  = "ttHnonbb_17",
    ntuples     = ntuplesPath2017+"/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8_new_pmx/*nominal*.root",
    categories  = ttHnonbb_categories,
    lumiWeight  = 41.5,
    selections  = trigger_sel_17,
    #selections   = "(Evt_Odd == 1)",
    MEMs        = memPath+"2017/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8.root",
    )

# dataset.addSample(
#     sampleName  = "TTToSL_16",
#     ntuples     = ntuplesPath2016+"/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/*nominal*.root",
#     categories  = ttbar_categories,
#     lumiWeight  = 35.9,
#     selections   = "(Evt_Odd == 1) and " + trigger_sel_16 ,
#     MEMs        = memPath+"2016/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8.root",
#     )


# dataset.addSample(
#     sampleName  = "TTbb_18",
#     ntuples     = ntuplesPath2018+"/TTbb_4f_TTToSemiLeptonic_TuneCP5-Powheg-Openloops-Pythia8/*nominal*.root",
#     categories  = ttbb_categories,
#     lumiWeight  = 59.7,
#     selections   = "(Evt_Odd == 1) and " + trigger_sel_18 ,
#     # selections   = "(Evt_Odd == 1)",
#     MEMs        = memPath+"2018/TTbb_4f_TTToSemiLeptonic_TuneCP5-Powheg-Openloops-Pythia8.root",
#     )

dataset.addSample(
    sampleName  = "TTbb_17",
    ntuples     = ntuplesPath2017+"/TTbb_Powheg_Openloops_new_pmx/*nominal*.root",
    categories  = ttbb_categories,
    lumiWeight  = 41.5,
    selections   = "(Evt_Odd == 1) and " + trigger_sel_17 ,
    # selections   = "(Evt_Odd == 1)", #me variable in ntupels , later bei plotskript framework ... angeben/ ausreichend die unnötigen variablen auskommentieren
    MEMs        = memPath+"2017/TTbb_Powheg_Openloops_SL.root",
    )

# dataset.addSample(
#     sampleName  = "TTbb_16",
#     ntuples     = ntuplesPath2016+"/TTbb_4f_TTToSemiLeptonic_TuneCP5-Powheg-Openloops-Pythia8/*nominal*.root",
#     categories  = ttbb_categories,
#     lumiWeight  = 35.9,
#     selections   = "(Evt_Odd == 1) and " + trigger_sel_16 ,
#     # selections   = "(Evt_Odd == 1)",
#     MEMs        = memPath+"2016/TTbb_4f_TTToSemiLeptonic_TuneCP5-Powheg-Openloops-Pythia8.root",
#     )



# dataset.addSample(
#     sampleName  = "tHq_18",
#     ntuples     = ntuplesPath2018+"/THQ_4f_Hincl_13TeV_madgraph_pythia8/*nominal*.root",
#     categories  = tHq_categories,
#     lumiWeight  = 59.7,
#     selections  = trigger_sel_18,
#     #selections  = "(Evt_Odd == 1)",
#     MEMs        = memPath+"2018/THQ_4f_Hincl_13TeV_madgraph_pythia8.root",
#     )

# dataset.addSample(
#     sampleName  = "tHq_17",
#     ntuples     = ntuplesPath2017+"/THQ_4f_Hincl_13TeV_madgraph_pythia8/*nominal*.root",
#     categories  = tHq_categories,
#     lumiWeight  = 41.5,
#     selections  = trigger_sel_17,
#     #selections  = "(Evt_Odd == 1)",
#     MEMs        = memPath+"2017/THQ_4f_Hincl_13TeV_madgraph_pythia8.root",
#     )

# dataset.addSample(
#     sampleName  = "tHq_16",
#     ntuples     = ntuplesPath2016+"/THQ_Hincl_13TeV-madgraph-pythia8_TuneCUETP8M1/*nominal*.root",
#     categories  = tHq_categories,
#     lumiWeight  = 35.9,
#     selections  = trigger_sel_16,
#     #selections  = "(Evt_Odd == 1)",
#     MEMs        = memPath+"2016/THQ_Hincl_13TeV-madgraph-pythia8_TuneCUETP8M1.root",
#     )


# dataset.addSample(
#     sampleName  = "tHW_18",
#     ntuples     = ntuplesPath2018+"/THW_5f_Hincl_13TeV_madgraph_pythia8/*nominal*.root",
#     categories  = tHW_categories,
#     lumiWeight  = 59.7,
#     selections  = trigger_sel_18,
#     #selections  = "(Evt_Odd == 1)",
#     MEMs        = memPath+"2018/THW_5f_Hincl_13TeV_madgraph_pythia8.root",
#     )

# dataset.addSample(
#     sampleName  = "tHW_17",
#     ntuples     = ntuplesPath2017+"/THW_5f_Hincl_13TeV_madgraph_pythia8/*nominal*.root",
#     categories  = tHW_categories,
#     lumiWeight  = 41.5,
#     selections  = trigger_sel_17,
#     #selections  = "(Evt_Odd == 1)",
#     MEMs        = memPath+"2017/THW_5f_Hincl_13TeV_madgraph_pythia8.root",
#     )

# dataset.addSample(
#     sampleName  = "tHW_16",
#     ntuples     = ntuplesPath2016+"/THW_Hincl_13TeV-madgraph-pythia8_TuneCUETP8M1/*nominal*.root",
#     categories  = tHW_categories,
#     lumiWeight  = 35.9,
#     selections  = trigger_sel_16,
#     #selections  = "(Evt_Odd == 1)",
#     MEMs        = memPath+"2016/THW_Hincl_13TeV-madgraph-pythia8_TuneCUETP8M1.root",
#     )

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
    "Evt_Lumi",
    "Triggered_HLT_IsoMu24_vX",
    "Triggered_HLT_IsoTkMu24_vX",
    "Triggered_HLT_Ele27_WPTight_Gsf_vX",
    "Triggered_HLT_IsoMu27_vX",
    "Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX",
    "Triggered_HLT_Ele32_WPTight_Gsf_L1DoubleEG_vX",
    "Triggered_HLT_Ele32_WPTight_Gsf_vX",
    "Triggered_HLT_Ele32_WPTight_Gsf_2017SeedsX"
    ]

# add these variables to the variable list
dataset.addVariables(additional_variables)

# run the preprocessing
dataset.runPreprocessing()
