# global imports
import numpy as np
import os
import sys

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

import DRACO_Frameworks.DNN_Aachen.DNN_Aachen as DNN_Aachen
import DRACO_Frameworks.DNN_Aachen.variable_info as variable_info

categories_dic = {"(N_Jets == 6 and N_BTagsM >= 3)": variable_info.variables_4j_3b,
    			  "(N_Jets == 5 and N_BTagsM >= 3)": variable_info.variables_5j_3b,
    			  "(N_Jets == 4 and N_BTagsM >= 3)": variable_info.variables_6j_3b,} 

event_category = ["(N_Jets == 6 and N_BTagsM >= 3)",
				  "(N_Jets == 5 and N_BTagsM >= 3)",
				  "(N_Jets == 4 and N_BTagsM >= 3)"]


inPath = "/storage/c/vanderlinden/DRACO-MLfoy/workdir/aachen_data"
outpath = "/storage/9/jschindler/Aachen_DNN"
event_classes = ["ttHbb", "ttbb", "ttb", "tt2b", "ttcc", "ttlf"]
category = event_category[0]



dnn_aachen = DNN_Aachen.DNN(in_path=inPath, 
							save_path=outpath,
                			event_classes=event_classes, 
                			event_category=category,
                			train_variables=categories_dic[category], 
                			prenet_targets=["N_Jets", "N_BTagsM"]) # placeholders )

dnn_aachen.build_model()
dnn_aachen.train_model()


