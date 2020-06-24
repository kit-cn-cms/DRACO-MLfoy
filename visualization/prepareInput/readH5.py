'''
test to read out h5 file
'''
# imports
import numpy as np
import pandas as pd
import re
import base64

def decode_Samples(path):
    # reads h5 files and decodes channels, saves entrys for histogram

    # read input data out of h5 file
    with pd.HDFStore(path, mode = "r" ) as store:
        df = store.select("data", stop = 5) #stop is arbitrary
        mi = store.select("meta_info")
        shape=list(mi["input_shape"])

    # set channels to decode
    columns_to_decode=[]
    for col in df.columns:
        m=re.match("(.*_Hist)", col)
        if m!=None:
            columns_to_decode.append(m.group(1))

    H_List_Dict={col:list() for col in columns_to_decode} 
    hist_data = []

    # decoding
    for i in range(len(columns_to_decode)):
        column_name = columns_to_decode[i]
        empty_imgs_evtids=[]
        gr_0 = []

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
        hist_data.append(gr_0)

    return df, hist_data


def normalisation(data, quantile, var):
    # normalises data

    normalisedData = []

    for index, row in data.iterrows():
        
        if not var == 'Jet_CSV[0-16]_Hist': 
            values = row[var]/quantile
        else:
            values = row[var]

        for line in values:
            for i in range(len(line)):
                if line[i] > 1.: line[i] = 1.

        normalisedData.append(values)
    data[var]=normalisedData

    return data


# paths
ttbar_path = '/ceph/jvautz/NN/CNNInputs/testCNN/3ch/ttbar_3ch_rot_MaxJetPt.h5'
ttH_path = '/ceph/jvautz/NN/CNNInputs/testCNN/3ch/ttH_3ch_rot_MaxJetPt.h5'

# prepare data for normalization 
train_samples = []
bkg_data, bkg_hist_data = decode_Samples(ttbar_path)
sig_data, sig_hist_data = decode_Samples(ttH_path)
train_samples.append(sig_data)
train_samples.append(bkg_data)
df = pd.concat(train_samples)

train_variables = ['Jet_Pt[0-16]_Hist', 'TaggedJet_Pt[0-9]_Hist','Jet_CSV[0-16]_Hist']
'''
# set quantile
hist_data = []
for i in range(len(train_variables)):
    hist_data.append(bkg_hist_data[i] + sig_hist_data[i])
print 'hist_data: ', hist_data

quantile =[np.quantile(data, 0.5) for data in hist_data]
'''

#print df['Jet_Pt[0-16]_Hist'].values
#print df['TaggedJet_Pt[0-9]_Hist'].values
#print df['Jet_CSV[0-16]_Hist'].values

all_img_ttH = []
all_img_ttbar = []
for i in range(5):
    img_ttH = []
    img_ttbar = []
    for train_var in train_variables:
        img_ttH.append(sig_data[train_var].values[i].tolist())
        img_ttbar.append(bkg_data[train_var].values[i].tolist())
    #print img
    all_img_ttH.append(img_ttH)
    all_img_ttbar.append(img_ttbar)
#print all_img_ttH

text_file_ttH = open("3ch_img_ttH.txt", "w")
text_file_ttH.write(str(all_img_ttH))
text_file_ttH.close()
text_file_ttbar = open("3ch_img_ttbar.txt", "w")
text_file_ttbar.write(str(all_img_ttbar))
text_file_ttbar.close()

'''
# normalise
for i in range(len(train_variables)):
    df = normalisation(df, quantile[i], train_variables[i])

print df['Jet_Pt[0-16]_Hist'].values
print df['Jet_CSV[0-16]_Hist'].values
'''


