import os
import sys
import optparse
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(os.path.dirname(filedir))
sys.path.append(basedir)

import root2pandas

"""
USE: python preprocessing.py --outputdirectory=DIR --variableSelection=FILE --treeName=STR --maxentries=INT --MEM=BOOL
"""
usage="usage=%prog [options] \n"
usage+="USE: python preprocessing.py --outputdirectory=DIR --variableselection=FILE --treeName=STR --maxentries=INT --MEM=BOOL --name=STR\n"
usage+="OR: python preprocessing.py -o DIR -v FILE -t STR-e INT -m BOOL -n STR"

parser = optparse.OptionParser(usage=usage)

parser.add_option("-o", "--outputdirectory", dest="outputDir",default="InputFeatures",
        help="DIR for output", metavar="outputDir")

parser.add_option("-v", "--variableselection", dest="variableSelection",default="variables_ttHbb_DL",
        help="FILE for variables used to train DNNs", metavar="variableSelection")

parser.add_option("-t", "--treeName",action='append', default=[],
        help="Name of the tree corresponding to the right category", metavar="treeName")

parser.add_option("-e", "--maxentries", dest="maxEntries", default=100000,
        help="INT used for maximal number of entries for each batch (to restrict memory usage)", metavar="maxEntries")

parser.add_option("-m", "--MEM", dest="MEM", action = "store_true", default=False,
        help="BOOL to use MEM or not", metavar="MEM")

parser.add_option("-n", "--name", dest="Name", default="dnn",
        help="STR of the output file name", metavar="Name")


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

base_selection = "(N_allJets_corr_nom >= 2 and N_taggedJets_corr_nom >= 1 and nPairs_corr ==1)"
ttH_selection = None #"(Evt_Odd == 1)"
Signal_selection = "(Flag_decay_channel == 1)"


# initialize dataset class
dataset = root2pandas.Dataset(

  outputdir  = outputdir,
  naming     = options.Name,
  addMEM     = options.MEM,
  maxEntries = options.maxEntries,
  tree = options.treeName,
  varName_Run = "run",
  varName_LumiBlock = "lumi",
  varName_Event = "event",
)

# add base event selection
dataset.addBaseSelection(base_selection)

ntuplesPath = "/nfs/dust/cms/user/mmarz/tuple/snape/NMSSM_28_6/2018/"

Sig_MH600_hSBB_categories = root2pandas.EventCategories()
Sig_MH600_hSBB_categories.addCategory("Sig_MH600_hSBB", selection = Signal_selection)

Sig_MH700_hSBB_categories = root2pandas.EventCategories()
Sig_MH700_hSBB_categories.addCategory("Sig_MH700_hSBB", selection = Signal_selection)

Sig_MH800_hSBB_categories = root2pandas.EventCategories()
Sig_MH800_hSBB_categories.addCategory("Sig_MH800_hSBB", selection = Signal_selection)

Sig_MH900_hSBB_categories = root2pandas.EventCategories()
Sig_MH900_hSBB_categories.addCategory("Sig_MH900_hSBB", selection = Signal_selection)

Sig_MH1000_hSBB_categories = root2pandas.EventCategories()
Sig_MH1000_hSBB_categories.addCategory("Sig_MH1000_hSBB", selection = Signal_selection)

Sig_MH1200_hSBB_categories = root2pandas.EventCategories()
Sig_MH1200_hSBB_categories.addCategory("Sig_MH1200_hSBB", selection = Signal_selection)

Sig_MH600_hSTauTau_categories = root2pandas.EventCategories()
Sig_MH600_hSTauTau_categories.addCategory("Sig_MH600_hSTauTau", selection = Signal_selection)

Sig_MH700_hSTauTau_categories = root2pandas.EventCategories()
Sig_MH700_hSTauTau_categories.addCategory("Sig_MH700_hSTauTau", selection = Signal_selection)

Sig_MH800_hSTauTau_categories = root2pandas.EventCategories()
Sig_MH800_hSTauTau_categories.addCategory("Sig_MH800_hSTauTau", selection = Signal_selection)

Sig_MH900_hSTauTau_categories = root2pandas.EventCategories()
Sig_MH900_hSTauTau_categories.addCategory("Sig_MH900_hSTauTau", selection = Signal_selection)

Sig_MH1000_hSTauTau_categories = root2pandas.EventCategories()
Sig_MH1000_hSTauTau_categories.addCategory("Sig_MH1000_hSTauTau", selection = Signal_selection)

Sig_MH1200_hSTauTau_categories = root2pandas.EventCategories()
Sig_MH1200_hSTauTau_categories.addCategory("Sig_MH1200_hSTauTau", selection = Signal_selection)

ttbar_categories = root2pandas.EventCategories()
ttbar_categories.addCategory("ttbar")

Zll_categories = root2pandas.EventCategories()
Zll_categories.addCategory("Zll")

Wjet_categories = root2pandas.EventCategories()
Wjet_categories.addCategory("Wjet")

misc_categories = root2pandas.EventCategories()
misc_categories.addCategory("misc")

# friend trees
friendTrees = {
    "gen": "/nfs/dust/cms/user/mmarz/NMSSM_friends/UL/friends/2018/14_7/friend_genWeights",
    "bSF": "/nfs/dust/cms/user/mmarz/NMSSM_friends/UL/friends/2018/14_7/friend_bTagWeights",
    "lepSF": "/nfs/dust/cms/user/mmarz/NMSSM_friends/UL/friends/2018/14_7/friend_lepWeights",
    }

# Signal
for i in range(1,5):
    dataset.addSample(
      sampleName = "Sig_MH600_hSBB_"+str(i),
      ntuples    = ntuplesPath+"Signal_nanoaodv9_UL_MH600_MhS60_hSToBB_hToTauTau"+"/tree_"+str(i)+".root",
      categories = Sig_MH600_hSBB_categories,
      selections = None,
    )

for i in range(1,5):
    dataset.addSample(
      sampleName = "Sig_MH600_hSTauTau_"+str(i),
      ntuples    = ntuplesPath+"Signal_nanoaodv9_UL_MH600_MhS60_hSToTauTau_hToBB"+"/tree_"+str(i)+".root",
      categories = Sig_MH600_hSTauTau_categories,
      selections = None,
    )

for i in range(1,5):
    dataset.addSample(
      sampleName = "Sig_MH700_hSBB_"+str(i),
      ntuples    = ntuplesPath+"Signal_nanoaodv9_UL_MH700_MhS60_hSToBB_hToTauTau"+"/tree_"+str(i)+".root",
      categories = Sig_MH700_hSBB_categories,
      selections = None,
    )

for i in range(1,5):
    dataset.addSample(
      sampleName = "Sig_MH700_hSTauTau_"+str(i),
      ntuples    = ntuplesPath+"Signal_nanoaodv9_UL_MH700_MhS60_hSToTauTau_hToBB"+"/tree_"+str(i)+".root",
      categories = Sig_MH700_hSTauTau_categories,
      selections = None,
    )

for i in range(1,5):
    dataset.addSample(
      sampleName = "Sig_MH800_hSBB_"+str(i),
      ntuples    = ntuplesPath+"Signal_nanoaodv9_UL_MH800_MhS60_hSToBB_hToTauTau"+"/tree_"+str(i)+".root",
      categories = Sig_MH800_hSBB_categories,
      selections = None,
    )

for i in range(1,5):
    dataset.addSample(
      sampleName = "Sig_MH800_hSTauTau_"+str(i),
      ntuples    = ntuplesPath+"Signal_nanoaodv9_UL_MH800_MhS60_hSToTauTau_hToBB"+"/tree_"+str(i)+".root",
      categories = Sig_MH800_hSTauTau_categories,
      selections = None,
    )

for i in range(1,5):
    dataset.addSample(
      sampleName = "Sig_MH900_hSBB_"+str(i),
      ntuples    = ntuplesPath+"Signal_nanoaodv9_UL_MH900_MhS60_hSToBB_hToTauTau"+"/tree_"+str(i)+".root",
      categories = Sig_MH900_hSBB_categories,
      selections = None,
    )

for i in range(1,5):
    dataset.addSample(
      sampleName = "Sig_MH900_hSTauTau_"+str(i),
      ntuples    = ntuplesPath+"Signal_nanoaodv9_UL_MH900_MhS60_hSToTauTau_hToBB"+"/tree_"+str(i)+".root",
      categories = Sig_MH900_hSTauTau_categories,
      selections = None,
    )

for i in range(1,5):
    dataset.addSample(
      sampleName = "Sig_MH1000_hSBB_"+str(i),
      ntuples    = ntuplesPath+"Signal_nanoaodv9_UL_MH1000_MhS60_hSToBB_hToTauTau"+"/tree_"+str(i)+".root",
      categories = Sig_MH1000_hSBB_categories,
      selections = None,
    )

for i in range(1,5):
    dataset.addSample(
      sampleName = "Sig_MH1000_hSTauTau_"+str(i),
      ntuples    = ntuplesPath+"Signal_nanoaodv9_UL_MH1000_MhS60_hSToTauTau_hToBB"+"/tree_"+str(i)+".root",
      categories = Sig_MH1000_hSTauTau_categories,
      selections = None,
    )

for i in range(1,5):
    dataset.addSample(
      sampleName = "Sig_MH1200_hSBB_"+str(i),
      ntuples    = ntuplesPath+"Signal_nanoaodv9_UL_MH1200_MhS60_hSToBB_hToTauTau"+"/tree_"+str(i)+".root",
      categories = Sig_MH1200_hSBB_categories,
      selections = None,
    )

for i in range(1,5):
    dataset.addSample(
      sampleName = "Sig_MH1200_hSTauTau_"+str(i),
      ntuples    = ntuplesPath+"Signal_nanoaodv9_UL_MH1200_MhS60_hSToTauTau_hToBB"+"/tree_"+str(i)+".root",
      categories = Sig_MH1200_hSTauTau_categories,
      selections = None,
    )

# ttbar samples
for i in range(1,53):
    dataset.addSample(
      sampleName = "TTTo2L2Nu_"+str(i),
      ntuples    = ntuplesPath+"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"+"/tree_"+str(i)+".root",
      categories = ttbar_categories,
      selections = None,
    )

for i in range(1,15):
    dataset.addSample(
      sampleName = "TTToSemi_"+str(i),
      ntuples    = ntuplesPath+"TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8"+"/tree_"+str(i)+".root",
      categories = ttbar_categories,
      selections = None,
    )

dataset.addSample(
  sampleName = "TTToHadronic",
  ntuples    = ntuplesPath+"TTToHadronic_TuneCP5_13TeV-powheg-pythia8/tree_1.root",
  categories = ttbar_categories,
  selections = None,
)

# Zll samples
dataset.addSample(
  sampleName = "EWKZ_ZToll",
  ntuples    = ntuplesPath+"EWKZ2Jets_ZToLL_M-50_TuneCP5_withDipoleRecoil_13TeV-madgraph-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "DYJets_lowpT",
  ntuples    = ntuplesPath+"DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/tree_1.root",
  categories = Zll_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "DYJets_70to100",
  ntuples    = ntuplesPath+"DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/tree_1.root",
  categories = Zll_categories,
  selections = None,
)

for i in range(1,3):
    dataset.addSample(
      sampleName = "DYJets_100to200_"+str(i),
      ntuples    = ntuplesPath+"DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8"+"/tree_"+str(i)+".root",
      categories = Zll_categories,
      selections = None,
    )

for i in range(1,3):
    dataset.addSample(
      sampleName = "DYJets_200to400_"+str(i),
      ntuples    = ntuplesPath+"DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8"+"/tree_"+str(i)+".root",
      categories = Zll_categories,
      selections = None,
    )

dataset.addSample(
  sampleName = "DYJets_400to600",
  ntuples    = ntuplesPath+"DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/tree_1.root",
  categories = Zll_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "DYJets_600to800",
  ntuples    = ntuplesPath+"DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/tree_1.root",
  categories = Zll_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "DYJets_800to1200",
  ntuples    = ntuplesPath+"DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/tree_1.root",
  categories = Zll_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "DYJets_1200to2500",
  ntuples    = ntuplesPath+"DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/tree_1.root",
  categories = Zll_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "DYJets_2500toInf",
  ntuples    = ntuplesPath+"DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/tree_1.root",
  categories = Zll_categories,
  selections = None,
)

# W+jets samples
dataset.addSample(
  sampleName = "Wjet_70to100",
  ntuples    = ntuplesPath+"WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/tree_1.root",
  categories = Wjet_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "Wjet_100to200",
  ntuples    = ntuplesPath+"WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/tree_1.root",
  categories = Wjet_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "Wjet_200to400",
  ntuples    = ntuplesPath+"WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/tree_1.root",
  categories = Wjet_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "Wjet_400to600",
  ntuples    = ntuplesPath+"WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/tree_1.root",
  categories = Wjet_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "Wjet_600to800",
  ntuples    = ntuplesPath+"WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/tree_1.root",
  categories = Wjet_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "Wjet_800to1200",
  ntuples    = ntuplesPath+"WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/tree_1.root",
  categories = Wjet_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "Wjet_1200to2500",
  ntuples    = ntuplesPath+"WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/tree_1.root",
  categories = Wjet_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "Wjet_2500toInf",
  ntuples    = ntuplesPath+"WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/tree_1.root",
  categories = Wjet_categories,
  selections = None,
)

# other
dataset.addSample(
  sampleName = "EWKWMinus",
  ntuples    = ntuplesPath+"EWKWMinus2Jets_WToLNu_M-50_TuneCP5_withDipoleRecoil_13TeV-madgraph-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "EWKWPlus",
  ntuples    = ntuplesPath+"EWKWPlus2Jets_WToLNu_M-50_TuneCP5_withDipoleRecoil_13TeV-madgraph-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

# dataset.addSample(
#   sampleName = "EWKZ_ZToNuNu",
#   ntuples    = ntuplesPath+"EWKZ2Jets_ZToNuNu_M-50_TuneCP5_withDipoleRecoil_13TeV-madgraph-pythia8/tree_1.root",
#   categories = misc_categories,
#   selections = None,
# )

dataset.addSample(
  sampleName = "ggHToBB",
  ntuples    = ntuplesPath+"GluGluHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

for i in range(1,5):
    dataset.addSample(
      sampleName = "ggHToTauTau_"+str(i),
      ntuples    = ntuplesPath+"GluGluHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8"+"/tree_"+str(i)+".root",
      categories = misc_categories,
      selections = None,
    )

dataset.addSample(
  sampleName = "QCD_15to30",
  ntuples    = ntuplesPath+"QCD_Pt_15to30_TuneCP5_13TeV_pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "QCD_30to50",
  ntuples    = ntuplesPath+"QCD_Pt_30to50_TuneCP5_13TeV_pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

# dataset.addSample(
#   sampleName = "QCD_50to80",
#   ntuples    = ntuplesPath+"QCD_Pt_50to80_TuneCP5_13TeV_pythia8/tree_1.root",
#   categories = misc_categories,
#   selections = None,
# )

dataset.addSample(
  sampleName = "QCD_80to120",
  ntuples    = ntuplesPath+"QCD_Pt_80to120_TuneCP5_13TeV_pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "QCD_120to170",
  ntuples    = ntuplesPath+"QCD_Pt_120to170_TuneCP5_13TeV_pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "QCD_170to300",
  ntuples    = ntuplesPath+"QCD_Pt_170to300_TuneCP5_13TeV_pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "QCD_300to470",
  ntuples    = ntuplesPath+"QCD_Pt_300to470_TuneCP5_13TeV_pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "QCD_470to600",
  ntuples    = ntuplesPath+"QCD_Pt_470to600_TuneCP5_13TeV_pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "QCD_600to800",
  ntuples    = ntuplesPath+"QCD_Pt_600to800_TuneCP5_13TeV_pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "QCD_800to1000",
  ntuples    = ntuplesPath+"QCD_Pt_800to1000_TuneCP5_13TeV_pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "QCD_1000to1400",
  ntuples    = ntuplesPath+"QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "QCD_1400to1800",
  ntuples    = ntuplesPath+"QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "QCD_1800to2400",
  ntuples    = ntuplesPath+"QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "QCD_2400to3200",
  ntuples    = ntuplesPath+"QCD_Pt_2400to3200_TuneCP5_13TeV_pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "QCD_3200toInf",
  ntuples    = ntuplesPath+"QCD_Pt_3200toInf_TuneCP5_13TeV_pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "ST_sc_4f_lep",
  ntuples    = ntuplesPath+"ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "ST_tc_4f_inc",
  ntuples    = ntuplesPath+"ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "ST_tc_4f_inc",
  ntuples    = ntuplesPath+"ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "ST_tW_5f_inc",
  ntuples    = ntuplesPath+"ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "ST_tW_5f_inc",
  ntuples    = ntuplesPath+"ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "ttHTobb",
  ntuples    = ntuplesPath+"ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "ttHToTauTau",
  ntuples    = ntuplesPath+"ttHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "VBFHTobb",
  ntuples    = ntuplesPath+"VBFHToBB_M-125_TuneCP5_13TeV-powheg-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

for i in range(1,3):
    dataset.addSample(
      sampleName = "VBFHToTauTau_"+str(i),
      ntuples    = ntuplesPath+"VBFHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8"+"/tree_"+str(i)+".root",
      categories = misc_categories,
      selections = None,
    )

for i in range(1,4):
    dataset.addSample(
      sampleName = "WminHToTauTau_"+str(i),
      ntuples    = ntuplesPath+"WminusHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8"+"/tree_"+str(i)+".root",
      categories = misc_categories,
      selections = None,
    )

for i in range(1,4):
    dataset.addSample(
      sampleName = "WplHToTauTau_"+str(i),
      ntuples    = ntuplesPath+"WplusHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8"+"/tree_"+str(i)+".root",
      categories = misc_categories,
      selections = None,
    )

dataset.addSample(
  sampleName = "WplHTobb",
  ntuples    = ntuplesPath+"WplusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "WW",
  ntuples    = ntuplesPath+"WW_TuneCP5_13TeV-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "WZ",
  ntuples    = ntuplesPath+"WZ_TuneCP5_13TeV-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

dataset.addSample(
  sampleName = "ZZ",
  ntuples    = ntuplesPath+"ZZ_TuneCP5_13TeV-pythia8/tree_1.root",
  categories = misc_categories,
  selections = None,
)

for i in range(1,4):
    dataset.addSample(
      sampleName = "ZHToTauTau_"+str(i),
      ntuples    = ntuplesPath+"ZHToTauTau_M125_CP5_13TeV-powheg-pythia8"+"/tree_"+str(i)+".root",
      categories = misc_categories,
      selections = None,
    )

# initialize variable list
dataset.addVariables(variable_set.all_variables)

# define an additional variable list
additional_variables = [
  "run",
  "lumi",
  "event",
  "",
]

# add these variables to the variable list
dataset.addVariables(additional_variables)

# run the preprocessing
dataset.runPreprocessing()