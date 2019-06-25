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

parser.add_option("-v", "--variableselection", dest="variableSelection",default="DL_variables",
        help="FILE for variables used to train DNNs", metavar="variableSelection")

parser.add_option("-t", "--treeName", dest="treeName",default="liteTreeTTH_step7_cate8",
        help="Name of the tree corresponding to the right category", metavar="treeName")

parser.add_option("-e", "--maxentries", dest="maxEntries", default=50000,
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

# define output classes
ttH_categories = root2pandas.EventCategories()
ttH_categories.addCategory("ttHbb", selection = None)

#ttbar_categories = root2pandas.EventCategories()
#ttbar_categories.addCategory("ttbar")

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

# initialize dataset class
dataset = root2pandas.Dataset(

  outputdir  = outputdir,
  naming     = options.Name,
  tree       = options.treeName,
  addMEM     = options.MEM,
  maxEntries = options.maxEntries,
)

# add base event selection
dataset.addBaseSelection(base_selection)

ntuplesPath = "/nfs/dust/cms/user/angirald/sandbox/ttHbb/InputFeatures/6nodes/cate9"
#ntuplesPath = '/nfs/dust/cms/user/angirald/sandbox/ttHbb/InputFeatures/binary/cate9'

# add samples to dataset
dataset.addSample(

  sampleName = "ttHbb",
  ntuples    = ntuplesPath+"/ttHbb_2L_cate9.root",
  categories = ttH_categories,
  selections = None,
)

#!! dataset.addSample(
#!! 
#!!   sampleName = "ttbar",
#!!   ntuples    = ntuplesPath+"/background_cate9.root",
#!!   categories = ttbar_categories,
#!!   selections = None,
#!! )

dataset.addSample(

    sampleName  = "ttbar_bb",
    ntuples     = ntuplesPath+"/ttbar_bb_cate9.root",
    categories  = ttbar_bb,
    selections  = None
)

dataset.addSample(

    sampleName  = "ttbar_b",
    ntuples     = ntuplesPath+"/ttbar_b_cate9.root",
    categories  = ttbar_b,
    selections  = ttH_selection
)

dataset.addSample(

    sampleName  = "ttbar_2b",
    ntuples     = ntuplesPath+"/ttbar_2b_cate9.root",
    categories  = ttbar_2b,
    selections  = ttH_selection
)

dataset.addSample(

    sampleName  = "ttbar_cc",
    ntuples     = ntuplesPath+"/ttbar_cc_cate9.root",
    categories  = ttbar_cc,
    selections  = ttH_selection
)

dataset.addSample(

    sampleName  = "ttbar_lf",
    ntuples     = ntuplesPath+"/ttbar_lf_cate9.root",
    categories  = ttbar_lf,
    selections  = ttH_selection
)

# initialize variable list
dataset.addVariables(variable_set.all_variables)

# define an additional variable list
additional_variables = [

  "N_jets",
  "N_btags",

  "runNumber",
  "lumiBlock",
  "eventNumber",

#  "weight_GEN",
  "weight",
]

# add these variables to the variable list
dataset.addVariables(additional_variables)

# run the preprocessing
dataset.runPreprocessing()
