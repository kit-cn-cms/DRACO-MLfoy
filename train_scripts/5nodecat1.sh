
python 5node.py -n dnnstudies -v 5node_2017_ge4j_3t_variableset -i /ceph/lreuter/DNNstudies/DNNstudies4/ -o VP5nodefinal -c ge4j_3t --plot -s -1 -S ttH --odd --printroc --dataera=2017
python 5node.py -n dnnstudies -v 5node_2017_ge4j_ge4t_variableset -i /ceph/lreuter/DNNstudies/DNNstudies4/ -o VP5nodefinal -c ge4j_ge4t --plot -s -1 -S ttH --odd --printroc --dataera=2017

python 5node.py -n dnnstudies -v 5node_2018_ge4j_3t_variableset -i /ceph/lreuter/DNNstudies/DNNstudies4/ -o VP5nodefinal -c ge4j_3t --plot -s -1 -S ttH --odd --printroc --dataera=2018
python 5node.py -n dnnstudies -v 5node_2018_ge4j_ge4t_variableset -i /ceph/lreuter/DNNstudies/DNNstudies4/ -o VP5nodefinal -c ge4j_ge4t --plot -s -1 -S ttH --odd --printroc --dataera=2018

python 5node.py -n dnnstudies -v 5node_combined_ge4j_3t_variableset -i /ceph/lreuter/DNNstudies/DNNstudies4/ -o VP5nodefinal -c ge4j_3t --plot -s -1 -S ttH --odd --printroc
python 5node.py -n dnnstudies -v 5node_combined_ge4j_ge4t_variableset -i /ceph/lreuter/DNNstudies/DNNstudies4/ -o VP5nodefinal -c ge4j_ge4t --plot -s -1 -S ttH --odd --printroc

python 5node.py -n dnnstudies -v 5node_1718_ge4j_3t_variableset -i /ceph/lreuter/DNNstudies/DNNstudies4/ -o VP5nodefinal -c ge4j_3t --plot -s -1 -S ttH --odd --printroc --dataera=2017,2018
python 5node.py -n dnnstudies -v 5node_1718_ge4j_ge4t_variableset -i /ceph/lreuter/DNNstudies/DNNstudies4/ -o VP5nodefinal -c ge4j_ge4t --plot -s -1 -S ttH --odd --printroc --dataera=2017,2018
