import os
import sys

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

filedir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(filedir)

import set5node_1718_ge4j_3t_variableset as set17183t
import set5node_1718_ge4j_ge4t_variableset as set17184t
import set5node_2016_ge4j_3t_variableset as set20163t
import set5node_2016_ge4j_ge4t_variableset as set20164t
import set5node_2017_ge4j_3t_variableset as set20173t
import set5node_2017_ge4j_ge4t_variableset as set20174t
import set5node_2018_ge4j_3t_variableset as set20183t
import set5node_2018_ge4j_ge4t_variableset as set20184t
import set5node_combined_ge4j_3t_variableset as setcombined3t
import set5node_combined_ge4j_ge4t_variableset as setcombined4t

def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 

listge4j_ge4t = []
listge4j_3t = []

listge4j_ge4t.extend(set17184t.all_variables)
listge4j_ge4t.extend(set20164t.all_variables)
listge4j_ge4t.extend(set20174t.all_variables)
listge4j_ge4t.extend(set20184t.all_variables)
listge4j_ge4t.extend(setcombined4t.all_variables)

listge4j_3t.extend(set17183t.all_variables)
listge4j_3t.extend(set20163t.all_variables)
listge4j_3t.extend(set20173t.all_variables)
listge4j_3t.extend(set20183t.all_variables)
listge4j_3t.extend(setcombined3t.all_variables)

print Diff(set20164t.all_variables,setcombined4t.all_variables)
print Diff(setcombined4t.all_variables,set20164t.all_variables)
print "-"*12
print Diff(set20174t.all_variables,setcombined4t.all_variables)
print Diff(setcombined4t.all_variables,set20174t.all_variables)
print "-"*12
print Diff(set20184t.all_variables,setcombined4t.all_variables)
print Diff(setcombined4t.all_variables,set20184t.all_variables)
print "-"*12
print Diff(set17184t.all_variables,setcombined4t.all_variables)
print Diff(setcombined4t.all_variables,set17184t.all_variables)

labels, values = zip(*Counter(listge4j_ge4t).items())

indexes = np.arange(len(labels))
width = 1

plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels,rotation='vertical')
plt.margins(0.2)
plt.subplots_adjust(bottom=0.4)
plt.title("Top variables used for DNN training")
#plt.xticks(fontsize=10)
plt.tight_layout()
plt.xlabel("Variables")
plt.ylabel("Number of times used")
plt.ylim([0,6])
#plt.show()
plt.savefig("plotge4j_ge4t.pdf")

labels, values = zip(*Counter(listge4j_3t).items())

indexes = np.arange(len(labels))
width = 1

plt.clf()
plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels,rotation='vertical')
plt.margins(0.2)
plt.ylim([0,6])
plt.tight_layout()
plt.subplots_adjust(bottom=0.4)
plt.title("Top variables used for DNN training")
plt.xlabel("Variables")
plt.ylabel("Number of times used")
#plt.show()
plt.savefig("plotge4j_3t.pdf")
