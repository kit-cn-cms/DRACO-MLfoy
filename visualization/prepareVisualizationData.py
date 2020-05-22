import numpy as np
import pandas as pd
import re
import base64
import os

from keras.models import Sequential
from keras.layers import Conv2D

class visualizer():

        def __init__(self, inputDir, outputDir, inputShape, plotName, rotName, channels, filterNum, filterSize, quantile = 0.95):
	        # initiate all variables

                self.inputDir = inputDir
                self.outputDir = outputDir + 'visualization_data/'
                self.inputShape = inputShape
                self.plotName = plotName
                self.rotName = rotName
                self.channels = channels
                self.filterNum = filterNum
                self.filterSize = filterSize
                self.filterWeights = []
                self.quantile = quantile

                # calculate maximal amount of serial layers
                l = len(self.filterSize[0])
                for tower in filterSize:
                        if len(tower) > l:
         	                l = len(tower)

                self.maxLayers = l

                # prepare filter layer parameters
                self.networkArchitecture = []
                for tower in self.filterSize:
                        towerCopy = []
                        for i in range(self.maxLayers):
  
         	                if i == 0:
         	                    towerCopy.append([tower[i], self.filterNum, len(self.channels)])

         	                else:
         	                    try:
         		                towerCopy.append([tower[i], self.filterNum, self.filterNum])
         	                    except IndexError:
         		                towerCopy.append([0,0,0])
         
                        self.networkArchitecture.append(towerCopy)
       
    
                # create directory for output 
                if not os.path.exists(self.outputDir):
    	                os.mkdir(self.outputDir)

                # prepare input data
                ttHSample = inputDir + '/ttH' + rotName + '.h5'
                ttbarSample = inputDir + '/ttbar' + rotName + '.h5'
    	 
                self.ttHImages = self.decodeInputData(ttHSample, channels[0],10000)
                self.ttbarImages = self.decodeInputData(ttbarSample, channels[0], 10000)

                if len(channels) > 1:
                        secondChImage = self.decodeInputData(ttHSample, channels[1], 1)[0]
                        self.inputImage = np.concatenate((self.ttHImages[0].reshape(self.inputShape + [1]), secondChImage.reshape(self.inputShape + [1])), axis = 2)
           

	def decodeInputData(self, sample, channel, importStop):
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
		    quantile = np.quantile(gr_0, self.quantile)

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

		 
	def readOutFilters(self, weights, position):
	    # saves filter data from training to textfile

                # print outs for information
                print('#'*60)
                print('read out filters '+ position + ' training')
                print('#'*60)

                # copy shape of model
		counter = 0
		shapeCopy = []
		for tower in self.filterSize:
			towerCopy = []

			for i in range(len(tower)):
				counter += 1
				towerCopy.append(counter)
			shapeCopy.append(towerCopy)
                
                # imitate order of output weights
		for tower in shapeCopy:
			for i in range(self.maxLayers -len(tower)):
				tower.insert(0,0)
                
		# resort weights from layers (network outputs order is different from needed one)
		order = np.swapaxes(np.asarray(shapeCopy), 0, 1).flatten()
		order = order[order != 0]
                
		allFilters = []
		for i in range(order.max()):
			index = order.tolist().index(i+1)
			allFilters.append(weights[index*2].tolist())
                
                # save filter weights to textfile
		textFile = open(self.outputDir + 'filter_outputs_' + position + '_training' + self.plotName + '.txt', "w")
		textFile.write(str(allFilters))
		textFile.close()


	def getFilters(self): 
		# prepares filter weights for prediction

                filterData = eval(open(self.outputDir + 'filter_outputs_after_training' + self.plotName + '.txt', 'r').read())
                filterWeights = np.zeros(np.asarray(self.networkArchitecture).shape[0:2]).tolist()
                filterCounter = 0
                for i in range(len(self.networkArchitecture)):
                    for j in range(self.maxLayers):
                
                        if self.networkArchitecture[i][j] == [0,0,0]:
                            filterWeights[i][j] = []
                        else:
                            filterWeights[i][j] = np.asarray(filterData[filterCounter])
                            filterCounter += 1
                
                self.filterWeights = filterWeights

	def prepareImageData(self):
		# takes input images and passes them through several filter layers with weights learned in training 

                print('_'*60)
                print('prepare feature maps')
                print('_'*60)

		#empty lists for saving feature maps with the same shape as the model's
		self.outputImages = np.empty(np.asarray(self.networkArchitecture).shape[0:2]).tolist()
		self.outputImagesSeparate = np.empty(np.asarray(self.networkArchitecture).shape[0:1]).tolist()
		self.outputImagesTTH = np.empty(np.asarray(self.networkArchitecture).shape[0:2]).tolist()
		self.outputImagesTTbar = np.empty(np.asarray(self.networkArchitecture).shape[0:2]).tolist()
            
                self.getFilters()

		for i in range(len(self.filterWeights)):
    			
			# pass image through several layers in each tower and save feature map after each layer
    			weights = []
                        layer = []

			for j in range(self.maxLayers):

                                # all non-existing layers and pictures in the grid get vaule []
                                self.outputImages[i][j] = []
                                self.outputImagesTTH[i][j] = []
                                self.outputImagesTTbar[i][j] = []
                                
				if self.filterWeights[i][j] != []:
                                        
                                        #create new model with j layers every time
                                        model = Sequential()
                                        print 'predict feature maps in tower ', i+1, ', layer ', j+1    
					par = self.networkArchitecture[i][j]
					layer.append(Conv2D(par[1], par[0], padding = 'same', input_shape = self.inputShape + [par[2]]))
                                        for k in range(j+1):
                                            model.add(layer[k])
					
                                        weights.append(self.filterWeights[i][j])
					weights.append(np.zeros(self.filterNum))
					model.set_weights(weights)

			    		self.outputImages[i][j] = model.predict(self.inputImage.reshape([1] + self.inputShape + [len(self.channels)])).tolist()
                                        
					if len(self.channels) == 1:

						# if images have only one channel produce overlapping images also for several layers
                                                self.outputImagesTTH[i][j] = 0.0
                                                for image in self.ttHImages:
							fmap = model.predict(image.reshape([1]+ self.inputShape + [1]))
							self.outputImagesTTH[i][j] += fmap
                                                self.outputImagesTTH[i][j] = self.outputImagesTTH[i][j].tolist()
                
                                                self.outputImagesTTbar[i][j] = 0.0
						for image in self.ttbarImages:
							fmap = model.predict(image.reshape([1]+ self.inputShape + [1]))
							self.outputImagesTTbar[i][j] += fmap
                                                self.outputImagesTTbar[i][j] = self.outputImagesTTbar[i][j].tolist()
			
                        if len(self.channels) > 1:
                                # in case of multi-channel input the channels will be put through the first layers of each tower separately 
                                modelSeparate = Sequential()
                                par = self.networkArchitecture[i][0]
                                modelSeparate.add(Conv2D(par[1], par[0], padding = 'same', input_shape = self.inputShape + [1]))

                                # produce feature maps with filter from first channel
                                weightsSeparate = [self.filterWeights[i][0][:,:,0,:].reshape(par[0], par[0], 1, par[1]), np.zeros(par[1])]
                                modelSeparate.set_weights(weightsSeparate)

                                # overlapping feature maps from 10.000 images in both ttH and ttbar sample
                                self.outputImagesTTH[i][0] = 0.0
                                for image in self.ttHImages:
                                        fmap = modelSeparate.predict(image.reshape([1]+ self.inputShape + [1]))
                                        self.outputImagesTTH[i][0] += fmap
                                self.outputImagesTTH[i][0] = self.outputImagesTTH[i][0].tolist()

                                self.outputImagesTTbar[i][0] = 0.0
                                for image in self.ttbarImages:
                                        fmap = modelSeparate.predict(image.reshape([1]+ self.inputShape + [1]))
                                        self.outputImagesTTbar[i][0] += fmap
                                self.outputImagesTTbar[i][0] = self.outputImagesTTbar[i][0].tolist()
                                
                                # separate feature maps channel one
                                self.outputImagesSeparate[i] = []
                                outputImage = modelSeparate.predict(self.inputImage[:,:,0].reshape([1]+self.inputShape + [1]))
                                self.outputImagesSeparate[i].append(outputImage.reshape(self.inputShape + [par[1]]).tolist())
                                
                                # produce feature maps with filter from second channel
                                # separate feature maps channel two
                                weightsSeparate = [self.filterWeights[i][0][:,:,1,:].reshape(par[0], par[0], 1, par[1]), np.zeros(par[1])]
                                modelSeparate.set_weights(weightsSeparate)
                                    
                                outputImage = modelSeparate.predict(self.inputImage[:,:,1].reshape([1]+self.inputShape + [1]))
                                self.outputImagesSeparate[i].append(outputImage.reshape(self.inputShape + [par[1]]).tolist())


	    	self.saveData()


	def saveData(self):
		# saves all data as list in textfile for later visualization
		
		#save conv-layer-info for reference
		text_file = open(self.outputDir + "filter_layer_info" + self.plotName + ".txt", "w")
		text_file.write(str(self.networkArchitecture))
                text_file.write('\n')
                text_file.write(str(self.channels))
		text_file.write(str('\n'))
                text_file.write(str(self.inputShape))
                text_file.close()

		text_file = open(self.outputDir + "input_image" + self.plotName + ".txt", "w")
		text_file.write(str(self.inputImage.tolist()))
		text_file.close()

		text_file = open(self.outputDir + "output_images" + self.plotName + ".txt", "w")
		text_file.write(str(self.outputImages))
		text_file.close()
                
                if len(self.channels) > 1:
		    text_file = open(self.outputDir + "output_images_separate" + self.plotName + ".txt", "w")
		    text_file.write(str(self.outputImagesSeparate))
		    text_file.close()
                
		text_file = open(self.outputDir + "all_output_images_ttH" + self.plotName + ".txt", "w")
		text_file.write(str(self.outputImagesTTH))
		text_file.close()

		text_file = open(self.outputDir + "all_output_images_ttbar" + self.plotName + ".txt", "w")
		text_file.write(str(self.outputImagesTTbar))
		text_file.close()
                


# manual use
#my_vis = visualizer('testfolder', 'testfolder', [11,15], 'test', 'MJPt', ['Jet_Pt[0-16]_Hist','Jet_CSV[0-16]_Hist'], 1, [[1],[2,3],[4,5,6]])
#my_vis.readOutFilters(test.test_data2, 'after')
#my_vis.getFilters()



