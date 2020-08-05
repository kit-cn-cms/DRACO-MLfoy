input=("training_modified_multiANNs/modified_MultiANN_QT_training_*_JTSTRING" "training_modified_multiANNs/modified_MultiANN_training_*_JTSTRING" "training_ANN_150N/ANN_150N_training_*_JTSTRING" "training_ANN_150N/QT_ANN_150N_training_*_JTSTRING" "Flipout_BNN_training_50_JTSTRING" "Flipout_QT_BNN_training_50_JTSTRING" "Var_QT_BNN_training_50_JTSTRING" "Var_BNN_training_50_JTSTRING" "modified_Flipout_V1_BNN_training_50_JTSTRING" "modified_Flipout_V1_QT_BNN_training_50_JTSTRING" "modified_Flipout_V3_BNN_training_50_JTSTRING" "modified_Flipout_V3_QT_BNN_training_50_JTSTRING" "modified_Var_V1_BNN_training_50_JTSTRING" "modified_Var_V1_QT_BNN_training_50_JTSTRING")
output=("multiANN_QT" "multiANN" "ANN_150N" "QT_ANN_150N" "Flip" "QT_Flip" "QT_Var" "Var" "modified_Flip_V1" "QT_modified_Flip_V1" "modified_Flip_V3" "QT_modified_Flip_V3" "modified_Var_V1" "QT_modified_Var_V1")
len=${#output[@]}

for ((i=0; i<$len; i++)); do
     python getTopVariables.py -w /home/ycung/Desktop/DRACO-MLfoy_thesis/workdir -i ${input[$i]} -o /home/ycung/Desktop/DRACO-MLfoy_thesis/workdir/variable_ranking_"${output[$i]}" -p --nplot 25 --std -v -t first_layer ge4j_ge3t
done


