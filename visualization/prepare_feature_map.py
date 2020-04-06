'''
takes stored filter weight data to process one picture from h5 datafile through filter layer and saves output feature maps to textfile 
'''

# imports
import numpy as np
import pandas as pd
import re
import base64
from keras.models import Sequential
from keras.layers import Conv2D


def get_filters(string): 
	# takes filter data as string and converts it into an array 

	filter_output = string.replace(' ', '').replace('[', '').replace(']', '').split(',')
	for i in range(len(filter_output)): 
		filter_output[i] = float(filter_output[i])
	filter_output = np.array(filter_output).reshape(4,4,2,8)

	return filter_output


def get_image_inputs(path, train_variables):
	# takes image input data from h5 file and decodes it
	
	# open datafile 
	with pd.HDFStore(path, mode = "r" ) as store:
	    df = store.select("data", stop = 100) #stop is arbitrary for less wait time
	    mi = store.select("meta_info")
    	shape=list(mi["input_shape"])

        # select channels to decode
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
	        maxjetinevt=np.max(u)
	        if(maxjetinevt!=0):
	            u=u/maxjetinevt
	        else:
	            empty_imgs_evtids.append(index[2])

	        u=np.reshape(u,shape)
	        H_List_Dict[column_name].append(u)

	    df[column_name]=H_List_Dict[column_name]
	    
	#prepare image matrices
	df_variables_tmp=[np.expand_dims(np.stack(df[ channel ].values), axis=3) for channel in train_variables]
	image_data = np.concatenate(df_variables_tmp, axis = 3)
	
	return image_data


#set paths
path_to_filter_data = 'trainCNN/visualizeTraining/Tagged_channel/basic_settings/'
path_to_image_data = '/ceph/jvautz/NN/CNNInputs/testCNN/Tagged_channel/basic/ttH_cnn.h5'
filename = '_cnn'

#set channels
channels = ['Jet_Pt[0-16]_Hist', 'TaggedJet_Pt[0-9]_Hist']

#choose random image to analyse with filters
image_inputs = get_image_inputs(path_to_image_data, channels)
random_image_index = np.random.randint(image_inputs.shape[0])
input_image=image_inputs[random_image_index].reshape(1, 11, 15, 2)

# create model
model = Sequential()
model.add(Conv2D(1, (4,4), input_shape=image_inputs[0].shape))

#prepare stored filters
filters = get_filters(open(path_to_filter_data + 'filterOutputs_after_training' + filename + '.txt', 'r').read())
weights = [filters, np.zeros(8)]

# store weights in the model
model.set_weights(weights)

# apply filter to input data
output_image = model.predict(input_image)

# save data
text_file = open(path_to_filter_data + "input_image" + filename + ".txt", "w")
text_file.write(str(input_image.tolist()))
text_file.close()

text_file = open(path_to_filter_data + "output_image" + filename + ".txt", "w")
text_file.write(str(output_image.tolist()))
text_file.close()




