import optparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
studiesdir = os.path.dirname(filedir)
basedir = os.path.dirname(studiesdir)
sys.path.append(basedir)


def print_query(data, query, name):
    ''' print query line '''
    if type(query) == list:
        query = " and ".join(query)
    querydf = data.query(query)
    nQuery = querydf.shape[0]
    print("{}: {} events (= {}%)".format(
        name, nQuery, 100.*nQuery/data.shape[0]))
    return querydf


# parser
parser = optparse.OptionParser(usage="%prog [options] file")
parser.add_option("-s","--sample",dest="samplename",default="ttZJets",metavar="SAMPLENAME",
    help="name of sample in ./output/SAMPLENAME.h5")
(opts,args)=parser.parse_args()

# get sample
sample_path = filedir+"/output/"+opts.samplename+".h5"
if not os.path.exists(sample_path):
    print("sample {} does not exist")
    exit()

# load data
data = pd.read_hdf(sample_path, "data")
print("total events: {}".format(data.shape[0]))



# define queries
Zbb = "ZToBB == 1 and ZToQQ == 0 and ZToLL == 0 and ZToNN == 0"
Zqq = "ZToBB == 0 and ZToQQ == 1 and ZToLL == 0 and ZToNN == 0"
Zll = "ZToBB == 0 and ZToQQ == 0 and ZToLL == 1 and ZToNN == 0"
Znn = "ZToBB == 0 and ZToQQ == 0 and ZToLL == 0 and ZToNN == 1"
Zother = "ZToBB == 0 and ZToQQ == 0 and ZToLL == 0 and ZToNN == 0"
# top decay
ttLep = "leptonicTops == 2"
ttHad = "leptonicTops == 0"
ttSL  = "leptonicTops == 1"
# jet cuts
jt_4j_ge3t = "N_Jets == 4 and N_BTagsM >= 3"
jt_5j_ge3t = "N_Jets == 5 and N_BTagsM >= 3"
jt_ge6j_ge3t = "N_Jets >= 6 and N_BTagsM >= 3"


# printouts
print_query(data, Zbb, "Z->bb")
print_query(data, Zqq, "Z->qq")
print_query(data, Zll, "Z->ll")
print_query(data, Znn, "Z->nunu")
print_query(data, Zother, "Z->other")
print("="*50)

print_query(data, ttLep, "full leptonic ttbar")
print_query(data, ttHad, "full hadronic ttbar")
print_query(data, ttSL,  "semi leptonic ttbar")

print("="*50)

ttZqq = print_query(data, [ttSL, Zbb], "ttZ(bb) SL")

print("="*50)

print_query(data, [ttSL, Zbb, jt_4j_ge3t], "ttZ(bb) SL 4j_ge3t")
print_query(data, [ttSL, Zbb, jt_5j_ge3t], "ttZ(bb) SL 5j_ge3t")
print_query(data, [ttSL, Zbb, jt_ge6j_ge3t], "ttZ(bb) SL ge6j_ge3t")


# histograms
ttZqq["N_Jets"].hist(bins = 10, range = [0.5,10.5], histtype = "stepfilled")
plt.xlabel("nJets semilep ttZ(bb)")
name = "output/{}_nJets.pdf".format(opts.samplename)
plt.savefig(name)
print("saved plot of nJets at {}".format(name))

plt.clf()
ttZqq["N_BTagsM"].hist(bins = 6, range = [0.5,6.5], histtype = "stepfilled")
plt.xlabel("nBTagsM semilep ttZ(bb)")
name = "output/{}_nTagsM.pdf".format(opts.samplename)
plt.savefig(name)
print("saved plot of nTagsM at {}".format(name))


