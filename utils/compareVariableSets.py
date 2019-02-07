# global imports
import os
import sys

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# specify the compared variable sets
set1_name = "new set of good variables"
import variable_sets.goodVariables as set1
#import variable_sets.topVariables_L as set1
#set2_name = "pre christmas tight cut variables"
#import variable_sets.topVariables_T as set2
set2_name = "pre christmas tight cut variables"
import variable_sets.topVariables_T as set2

vars1 = set1.variables
vars2 = set2.variables

for key in vars1:
    print("\n\n\n"+"-"*40)
    print("CATEGORY: {}".format(key))
    
    variables = list(set(vars1[key] + vars2[key]))
    common = []
    only_set1 = []
    only_set2 = []

    for v in variables:
        if v not in vars2[key]:
            only_set1.append(v)
        elif v not in vars1[key]:
            only_set2.append(v)
        else:
            common.append(v)

    print("\n\tvariables only in {}".format(set1_name))
    for v in only_set1: print(v)

    print("\n\tvariables only in {}".format(set2_name))
    for v in only_set2: print(v)

    print("\n\tcommon variables")
    for v in common: print(v)

