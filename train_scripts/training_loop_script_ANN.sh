#ANN
name=binary_crossentropy_Adam_v2
output1=QT_ANN_v2_training_
output2=ANN_v2_training_
epochs=4000
iteration=100

for ((i=0; i<$iteration; i+=1)); do
    python train_template.py -o $output1"$i" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e "$epochs" -q
    python train_template.py -o $output2"$i" -i /local/scratch/ssd/nshadskiy/2017_nominal -c ge4j_ge3t -v allVariables_2017_bnn -n "$name" -p --printroc --binary --signal ttH -e "$epochs"
done