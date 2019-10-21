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

parser.add_option("-t", "--treeName",action='append', default=["liteTreeTTH_step7_cate8"],
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

base_selection = "(N_jets >= 3 and N_btags >= 2)"
ttH_selection = None #"(Evt_Odd == 1)"


# initialize dataset class
dataset = root2pandas.Dataset(

  outputdir  = outputdir,
  naming     = options.Name,
  addMEM     = options.MEM,
  maxEntries = options.maxEntries,
  tree = options.treeName,
  varName_Run = "runNumber",
  varName_LumiBlock = "lumiBlock",
  varName_Event = "eventNumber",
)

# add base event selection
dataset.addBaseSelection(base_selection)

ntuplesPath = "/nfs/dust/cms/user/missirol/sandbox/ttHbb/output_190914_DeepJet/2017/exe1/selectionRoot_reco_liteTreeTTH/Nominal"

ttH_categories = root2pandas.EventCategories()
ttH_categories.addCategory("ttH", selection = None)

ttbar_categories = root2pandas.EventCategories()
ttbar_categories.addCategory("ttbar")

ttbar_bb = root2pandas.EventCategories()
ttbar_bb.addCategory("ttbb")

ttbar_b = root2pandas.EventCategories()
ttbar_b.addCategory("ttb")

ttbar_2b = root2pandas.EventCategories()
ttbar_2b.addCategory("tt2b")

ttbar_cc = root2pandas.EventCategories()
ttbar_cc.addCategory("ttcc")

ttbar_lf = root2pandas.EventCategories()
ttbar_lf.addCategory("ttlf")

#ttHbb samples
for i in range(15):
    dataset.addSample(
      sampleName = "ttH"+str(i),
      ntuples    = ntuplesPath+"/*/*_ttbarH125tobbbar_2L_"+str(i)+".root",
      categories = ttH_categories,
      selections = None,
    )

#ttbar_bb samples
for i in range(5):
    dataset.addSample(
        sampleName  = "ttbarNotau_bb"+str(i),
        ntuples     = ntuplesPath+"/*/*_ttbarDileptonNotauBbbar_fromDilepton_PSweights_"+str(i)+".root",
        categories  = ttbar_bb,
        selections  = None
    )
    dataset.addSample(
        sampleName  = "ttbarOnlytau_bb"+str(i),
        ntuples     = ntuplesPath+"/*/*_ttbarDileptonOnlytauBbbar_fromDilepton_PSweights_"+str(i)+".root",
        categories  = ttbar_bb,
        selections  = None
    )
#dataset.addSample(
#    sampleName  = "ttbarNotau_bb",
#    ntuples     = ntuplesPath+"/*/*_ttbarDileptonNotauBbbar_fromDilepton_ttbbPowheg.root",
#    categories  = ttbar_bb,
#    selections  = None
#)
#dataset.addSample(
#    sampleName  = "ttbarOnlytau_bb",
#    ntuples     = ntuplesPath+"/*/*_ttbarDileptonOnlytauBbbar_fromDilepton_ttbbPowheg.root",
#    categories  = ttbar_bb,
#    selections  = None
#)

#ttbar_b samples
for i in range(5):
    dataset.addSample(
        sampleName  = "ttbarNotau_b"+str(i),
        ntuples     = ntuplesPath+"/*/*_ttbarDileptonNotauB_fromDilepton_PSweights_"+str(i)+".root",
        categories  = ttbar_b,
        selections  = None
    )
    dataset.addSample(
        sampleName  = "ttbarOnlytau_b"+str(i),
        ntuples     = ntuplesPath+"/*/*_ttbarDileptonOnlytauB_fromDilepton_PSweights_"+str(i)+".root",
        categories  = ttbar_b,
        selections  = None
    )
#dataset.addSample(
#    sampleName  = "ttbarNotau_b",
#    ntuples     = ntuplesPath+"/*/*_ttbarDileptonNotauB_fromDilepton_ttbbPowheg.root",
#    categories  = ttbar_b,
#    selections  = None
#)
#dataset.addSample(
#    sampleName  = "ttbarOnlytau_b",
#    ntuples     = ntuplesPath+"/*/*_ttbarDileptonOnlytauB_fromDilepton_ttbbPowheg.root",
#    categories  = ttbar_b,
#    selections  = None
#)

#ttbar_2b samples
for i in range(5):
    dataset.addSample(
        sampleName  = "ttbarNotau_2b"+str(i),
        ntuples     = ntuplesPath+"/*/*_ttbarDileptonNotau2b_fromDilepton_PSweights_"+str(i)+".root",
        categories  = ttbar_2b,
        selections  = None
    )
    dataset.addSample(
        sampleName  = "ttbarOnlytau_2b"+str(i),
        ntuples     = ntuplesPath+"/*/*_ttbarDileptonOnlytau2b_fromDilepton_PSweights_"+str(i)+".root",
        categories  = ttbar_2b,
        selections  = None
    )
#dataset.addSample(
#    sampleName  = "ttbarNotau_2b",
#    ntuples     = ntuplesPath+"/*/*_ttbarDileptonNotauB_fromDilepton_ttbbPowheg.root",
#    categories  = ttbar_2b,
#    selections  = None
#)
#dataset.addSample(
#    sampleName  = "ttbarOnlytau_2b",
#    ntuples     = ntuplesPath+"/*/*_ttbarDileptonOnlytau2b_fromDilepton_ttbbPowheg.root",
#    categories  = ttbar_2b,
#    selections  = None
#)

#ttbar_cc samples
for i in range(5):
    dataset.addSample(
        sampleName  = "ttbar_cc"+str(i),
        ntuples     = ntuplesPath+"/*/*_ttbarDileptonPlustauCcbar_fromDilepton_PSweights_"+str(i)+".root",
        categories  = ttbar_cc,
        selections  = None
    )

#dataset.addSample(
#    sampleName  = "ttbarPowheg_cc",
#    ntuples     = ntuplesPath+"/*/*_ttbarDileptonPlustauCcbar_fromDilepton_ttbbPowheg.root",
#    categories  = ttbar_cc,
#    selections  = None
#)

#ttbar_lf samples
for i in range(5):
    dataset.addSample(
        sampleName  = "ttbar_lf"+str(i),
        ntuples     = ntuplesPath+"/*/*_ttbarDileptonPlustauOther_fromDilepton_PSweights_"+str(i)+".root",
        categories  = ttbar_lf,
        selections  = None
    )

#dataset.addSample(
#    sampleName  = "ttbarPowheg_lf",
#    ntuples     = ntuplesPath+"/*/*_ttbarDileptonPlustauOther_fromDilepton_ttbbPowheg.root",
#    categories  = ttbar_lf,
#    selections  = None
#)


# initialize variable list
dataset.addVariables(variable_set.all_variables)

# define an additional variable list
additional_variables = [
  "N_jets",
  "N_btags",
  "runNumber",
  "lumiBlock",
  "eventNumber",
  "weight",
]

# add these variables to the variable list
dataset.addVariables(additional_variables)

# run the preprocessing
dataset.runPreprocessing()
