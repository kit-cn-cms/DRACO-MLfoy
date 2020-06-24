import numpy as np
import pandas as pd
import re
import base64
import os


def decodeInputData(sample, channel, importStop):
		# decodes and normalizes data from h5 file

		# open file
		with pd.HDFStore(sample, mode = "r" ) as store:
			df = store.select("data", stop = importStop) 
			mi = store.select("meta_info")
			shape = list(mi["input_shape"])
	 
	 	# select channels to decode
   		for col in df.columns:
 			m = re.match(channel, col)
 			if m != None:
 				columnToDecode = m.group(1)

 		decodedImages = []
 		gr_0 = []

 		# decode data
 		for index, row in df.iterrows():
 			r = base64.b64decode(row[channel])
	                u = np.frombuffer(r,dtype=np.float64)
	                u = np.reshape(u,shape)

	                for line in u:
	                    for element in line:
	                        if element > 0.:
	                            gr_0.append(element)
	                decodedImages.append(u)

		df[channel] = decodedImages
		imageData = df[channel].values

	    # normalize data in case of Jet_Pt channel
	  	if channel == 'Jet_Pt[0-16]_Hist':
		    quantile = np.asarray(gr_0).max()

		    normalisedData = []
		    for image in imageData:
		       image = image/quantile

		       for line in image:
		           for j in range(len(line)):
		              if line[j] > 1.: line[j] = 1.

		       normalisedData.append(image)
		    return normalisedData

		else:
			return imageData


data = decodeInputData('../../h5/st_CSV_rot_MaxJetPt.h5', 'Jet_CSV[0-16]_Hist', 3)

print data

