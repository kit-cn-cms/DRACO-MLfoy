'''
test to read out h5 file
'''
# imports
import numpy as np
import pandas as pd
import re
import base64

# path to data
path = '/ceph/jvautz/NN/CNNInputs/testCNN/test_CSV/CSV_rotation_3ch/ttbar_CSV_rot_MaxJetPt.h5'

# read input data out of h5 file
with pd.HDFStore(path, mode = "r" ) as store:
    df = store.select("data", stop = 3) #stop is arbitrary
    mi = store.select("meta_info")
    shape=list(mi["input_shape"])

# set channels to decode
columns_to_decode=[]
for col in df.columns:
    m=re.match("(.*_Hist)", col)
    if m!=None:
       columns_to_decode.append(m.group(1))

H_List_Dict={col:list() for col in columns_to_decode}

# decoding and normalisation
for column_name in columns_to_decode:
    empty_imgs_evtids=[]
    for index, row in df.iterrows():
        r=base64.b64decode(row[column_name])
        u=np.frombuffer(r,dtype=np.float64)

	# own normalisation for each image -> rework
        #maxjetinevt=np.max(u)
        #if(maxjetinevt!=0):
        #    u=u/maxjetinevt
        #else:
        #    empty_imgs_evtids.append(index[2])

        u=np.reshape(u,shape)

        H_List_Dict[column_name].append(u)
    print(column_name)    
    print(H_List_Dict[column_name])

    # test new normalisation
    if not column_name == 'Jet_CSV[0-16]_Hist':
        value_list = []
        for element in np.asarray(H_List_Dict[column_name]).flatten():
            if not element == 0:
                value_list.append(element)
    
        quantile = np.quantile(value_list, 0.9)
        print(quantile)

        H_List_Dict[column_name] = np.asarray(H_List_Dict[column_name])/quantile

        for image in H_List_Dict[column_name]:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if image[i][j] > 1.:
                        image[i][j] = 1.

        print(H_List_Dict[column_name])
    try:
        df[column_name]=H_List_Dict[column_name].tolist()
    except AttributeError:
        df[column_name]=H_List_Dict[column_name]

# set channels
train_variables = ['Jet_Pt[0-16]_Hist', 'Jet_CSV[0-16]_Hist', 'TaggedJet_Pt[0-9]_Hist']

#prepare image matrices
df_variables_tmp=[np.expand_dims(np.stack(df[ channel].values), axis=3) for channel in train_variables]
#print(df_variables_tmp)
image_data = np.concatenate(df_variables_tmp, axis = 3)

#choose random image out of h5 file
random_image_index = np.random.randint(image_data.shape[0])
input_image=image_data[random_image_index]

# print outs
#for channel in df:
#    print(channel)
#print(input_image)
#print('\n')

# show channels separate
#for i in range(2):
#	print(train_variables[i])
#	print(input_image[:,:,i])
