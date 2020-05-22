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

#################################################################################
def readOutFilters(l, path, position, name):
    # reads out filter data and saves it to textfile
    
    print('#'*60)
    print('read out filters '+ position + ' training')
    text_file = open(path + '/filterOutputs_' + position + '_training' + name + '.txt', "w")
    text_file.write(str(l.tolist()))
    text_file.close()
    print('#'*60)


def get_filters(string, num_filters, filter_size): 
	# takes filter data as string and converts it into an array 

	filter_output = string.replace(' ', '').replace('[', '').replace(']', '').split(',')
	for i in range(len(filter_output)): 
		filter_output[i] = float(filter_output[i])
	filter_output = np.array(filter_output).reshape(filter_size,filter_size,2,num_filters)

	return filter_output

#####################################################################################
def get_image_inputs(path, train_variables):
	# takes image input data from h5 file and decodes it
	
	# open datafile 
	with pd.HDFStore(path, mode = "r" ) as store:
	    df = store.select("data", stop = 1) #stop is arbitrary for less wait time
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
	    
	# prepare image matrices
	df_variables_tmp=[np.expand_dims(np.stack(df[ channel ].values), axis=3) for channel in train_variables]
	image_data = np.concatenate(df_variables_tmp, axis = 3)
	
	return image_data

#########################################################################################
def decode_input_data(path):
        # takes image input data from h5 file and decodes it

    # open datafile 
    with pd.HDFStore(path, mode = "r" ) as store:
        df = store.select("data", stop = 10000) #overlay 10000 pictures
        mi = store.select("meta_info")
    shape=list(mi["input_shape"])

    # select channels to decode
    columns_to_decode=[]
    for col in df.columns:
        m=re.match("(.*_Hist)", col)
        if m!=None:
           columns_to_decode.append(m.group(1))

    H_List_Dict={col:list() for col in columns_to_decode}
    column_name = 'Jet_Pt[0-16]_Hist'
    gr_0=[]
    for index, row in df.iterrows():

        r=base64.b64decode(row[column_name])
        u=np.frombuffer(r,dtype=np.float64)

        u=np.reshape(u,shape)

        for line in u:
            for element in line:
                if element > 0.:
                    gr_0.append(element)

        H_List_Dict[column_name].append(u)

    df[column_name]=H_List_Dict[column_name]
    image_data =  df[column_name].values

    #normalize data
  
    quantile = np.quantile(gr_0, 0.95)


    var = 'Jet_Pt[0-16]_Hist'
    normalisedData = []
    for image in image_data:
       image = image/quantile

       for line in image:
           for j in range(len(line)):
              if line[j] > 1.: line[j] = 1.

       normalisedData.append(image)

    return normalisedData

####################################################################################   
def prepare_feature_maps(path_to_filter_data, path_to_image_data, samplename, filename, channels, num_filters, filter_size):
    # prepare and save feature map data
    
    # choose first image to analyse with filters
    image_inputs = get_image_inputs(path_to_image_data + '/ttH' + samplename + '.h5', channels)
    input_image=image_inputs[0].reshape(1, 11, 15, 2)

  
    normalisedData_ttH = decode_input_data(path_to_image_data + '/ttH'  + samplename + '.h5')
    normalisedData_ttbar = decode_input_data(path_to_image_data + '/ttbar'  + samplename + '.h5')

    # predict for two channels
    # create model
    model_2ch = Sequential()
    model_2ch.add(Conv2D(1, (filter_size,filter_size), padding='same', input_shape=image_inputs[0].shape))

    # prepare stored filters
    filters = get_filters(open(path_to_filter_data + 'filterOutputs_after_training' + filename + '.txt', 'r').read(), num_filters, filter_size)
    weights_2ch = [filters, np.zeros(num_filters)]

    # store weights in the model
    model_2ch.set_weights(weights_2ch)

    # apply filter to input data
    output_image_2ch = model_2ch.predict(input_image)

    # repeat with one channel each
    output_images = []
    for i in range(len(channels)):

        # create model
        model_1ch = Sequential()
        model_1ch.add(Conv2D(1, (filter_size,filter_size), padding = 'same', input_shape=input_image[:,:,:,i].reshape(11,15,1).shape))

        # prepare stored filters
        weights_1ch = [filters[:,:,i,:].reshape(filter_size,filter_size,1,num_filters), np.zeros(num_filters)]

        # store weights in the model
        model_1ch.set_weights(weights_1ch)
        
        # apply filter to input data
        output_images.append(model_1ch.predict(input_image[:,:,:,i].reshape(1,11,15,1)))
        print input_image[:,:,:,i].reshape(1,11,15,1)
    # repeat for overlaying picuters 

    model = Sequential()
    model.add(Conv2D(1, (filter_size,filter_size), padding = 'same', input_shape=(11,15,1)))

    # prepare stored filters
    filters = get_filters(open(path_to_filter_data + 'filterOutputs_after_training' + filename + '.txt', 'r').read(), num_filters, filter_size)
    weights = [filters[:,:,0,:].reshape(filter_size,filter_size,1,num_filters), np.zeros(num_filters)]

    # store weights in the model
    model.set_weights(weights)

    output_picture_ttH= np.zeros((1,11,15,num_filters))
    for image in normalisedData_ttH:
        # apply filter to input data
        picture = model.predict(image.reshape(1,11,15,1))
        output_picture_ttH += picture

    output_picture_ttbar= np.zeros((1,11,15,num_filters))
    for image in normalisedData_ttbar:
        # apply filter to input data
        picture = model.predict(image.reshape(1,11,15,1))
        output_picture_ttbar += picture
'''
    # save data
    text_file = open(path_to_filter_data + "input_image" + filename + ".txt", "w")
    text_file.write(str(input_image.tolist()))
    text_file.close()

    text_file = open(path_to_filter_data + "output_image_2ch" + filename + ".txt", "w")
    text_file.write(str(output_image_2ch.tolist()))
    text_file.close()

    for i in range(len(channels)):
        text_file = open(path_to_filter_data + "output_image_" + channels[i][0:channels[i].find('[')]  + filename + ".txt", "w")
        text_file.write(str(output_images[i].tolist()))
        text_file.close()

    text_file = open(path_to_filter_data + "all_output_images_ttH" + filename + ".txt", "w")
    text_file.write(str(output_picture_ttH.tolist()))
    text_file.close()

    text_file = open(path_to_filter_data + "all_output_images_ttbar" + filename + ".txt", "w")
    text_file.write(str(output_picture_ttbar.tolist()))
    text_file.close()
'''

filter_size = 4
num_filters = 8
# for manual use
# set paths
rot = 'no_rot'
path_to_filter_data = '../../workdir/trainCNN/Results_Test_Series/all_rot_basic_CSV/' + rot + '_basic_model_CSV/'
path_to_image_data = '/ceph/jvautz/NN/CNNInputs/testCNN/CSV_channel/all_rotations'
filename = '_' + rot  +'_basic_model_CSV'
samplename = '_CSV_' + rot

# set channels
channels = ['Jet_Pt[0-16]_Hist', 'Jet_CSV[0-16]_Hist']

prepare_feature_maps(path_to_filter_data, path_to_image_data, samplename, filename, channels, num_filters, filter_size)
