'''
produces and combines feature maps of all images 
'''

# imports
import numpy as np
import pandas as pd
import re
import base64
from keras.models import Sequential
from keras.layers import Conv2D



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

    return image_data, gr_0

'''
def normalize_input_data(data, gr_0):

    #normalize data
  
   quantile = np.quantile(gr_0, 0.95)


   
   var = 'Jet_Pt[0-16]_Hist'
   normalisedData = []
   for image in data:
       image = image/quantile

       for line in image:
           for j in range(len(line)):
              if line[j] > 1.: line[j] = 1.

       normalisedData.append(image)

   return normalisedData


def get_filters(string, num_filters, filter_size):
        # takes filter data as string and converts it into an array 

        filter_output = string.replace(' ', '').replace('[', '').replace(']', '').split(',')
        for i in range(len(filter_output)):
                filter_output[i] = float(filter_output[i])
        filter_output = np.array(filter_output).reshape(filter_size,filter_size,2,num_filters)

        return filter_output
'''
fname = 'ttbar_CSV_no_rot'
path_to_image_data = '/ceph/jvautz/NN/CNNInputs/testCNN/CSV_channel/all_rotations/'+fname+'.h5'
#path_to_filter_data = '../workdir/trainCNN/visualizeTraining/CSV_channel/series_CSV_qu_95_all_rots_all_models/rot_MaxJetPt_basic_model/'
output = '../workdir/trainCNN/input_images/'
filename = '_'+fname

data, gr_0 = decode_input_data(path_to_image_data)

#print data

input_picture = np.zeros((data[0].shape))
for i in data:
	input_picture += i




'''
#print np.asarray(normalisedData).shape    


model = Sequential()
model.add(Conv2D(1, (4,4), padding = 'same', input_shape=(11,15,1)))

# prepare stored filters
filters = get_filters(open(path_to_filter_data + 'filterOutputs_after_training' + filename + '.txt', 'r').read())
weights = [filters[:,:,0,:].reshape(4,4,1,8), np.zeros(8)]

# store weights in the model
model.set_weights(weights)

output_picture= np.zeros((1,11,15,8))

for image in normalisedData:
    #print image.shape
	# apply filter to input data
    picture = model.predict(image.reshape(1,11,15,1))
    output_picture += picture
    #print picture
    #print output_picture.shape
'''
# save data
text_file = open(output + "all_input_images" + filename + ".txt", "w")
text_file.write(str(input_picture.tolist()))
text_file.close()
'''
text_file = open(output + "all_input_images_normed" + filename + ".txt", "w")
text_file.write(str(input_picture_normed.tolist()))
text_file.close()

text_file = open(path_to_filter_data + "all_output_images" + filename + ".txt", "w")
text_file.write(str(output_picture.tolist()))
text_file.close()
'''


