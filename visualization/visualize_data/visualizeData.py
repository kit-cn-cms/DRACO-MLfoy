'''
visualizes the filter weights of the convolutional layer's filter after being read out before and after training as well as their differences
visualizes feature maps alltogether and for each channel separate after being lead through the filter of convolutional layer
visualizes overlapping feature maps from ttH and ttbar sample
'''

# imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import os
import sys
import math

# option_handler
import optionHandler_vis
options = optionHandler_vis.optionHandler_vis(sys.argv)


class Visualizer():

		def __init__(self, inputDir, fileName):
			# initiate all variables

			# path
			self.inputDir = inputDir
			self.fileName = fileName
			
			# read layer information file
			networkArchitecture = open(inputDir+'filter_layer_info_'+fileName+'.txt', 'r').read().split('\n')

			# mxn-matrix (m parallel columns with n serial layers) with layer information: filtersize, filternum and number of channels
			self.networkArchitecture = eval(networkArchitecture[0])
			self.filterNum = self.networkArchitecture[0][0][1]

			# calculate maximal amount of serial layers
			self.maxLayers = len(self.networkArchitecture[0])

			# get inputshape and channels
			channels = eval(networkArchitecture[1])
			self.channels = [channel[0:channel.index('[')] for channel in channels]
			self.inputShape = eval(networkArchitecture[2])

			# prepare visulization data from textfiles
			self.prepareData()


####################################################################################################################################################
		# data preparation


		def prepareData(self):
			# coordinates preparation data for visualization

			# prepare filter data
			self.filterBefore = self.prepareFilterData('filter_outputs_before_training_')
			self.filterAfter = self.prepareFilterData('filter_outputs_after_training_')

			# calculate differences between filters
			self.filterDiff = np.zeros(np.asarray(self.networkArchitecture).shape[0:2]).tolist()
			for i in range(len(self.networkArchitecture)):
				for j in range(self.maxLayers):
					self.filterDiff[i][j] = np.absolute(np.asarray(self.filterAfter[i][j]) - np.asarray(self.filterBefore[i][j]))
			
			# prepare input image
			rawInputImage = eval(open(self.inputDir + 'input_image_' + self.fileName + '.txt', 'r').read())
			self.inputImages = np.flip(np.swapaxes(np.asarray(rawInputImage), 0, 2), axis=1)
			
			# prepare output images
			self.outputImages = self.prepareOutputImages('output_images_')

			# prepare output images in separate channels
			self.outputImagesSeparate = self.prepareOutputImagesSeparate('output_images_separate_')
			
			# prepare overlapping feature maps from different samples
			self.allImagesTTH = self.prepareOutputImages('all_output_images_ttH_')
			self.allImagesTTbar = self.prepareOutputImages('all_output_images_ttbar_')
			

		def prepareFilterData(self, textfileName):
			# prepare data from filter weights for visualization

			# import filter data from textfile
			filterRawData = eval(open(self.inputDir + textfileName + self.fileName + '.txt', 'r').read())

			# initialize arrays with same shape as network architecture to fill with filter weights and labels for each layer
			filterWeights = np.zeros(np.asarray(self.networkArchitecture).shape[0:2]).tolist()
			labels = np.zeros(np.asarray(self.networkArchitecture).shape[0:2]).tolist()

			# iterate for every layer in mxn-matrix and list of raw filter data
			filterCounter = 0
			for i in range(len(self.networkArchitecture)):
				for j in range(self.maxLayers):
					labels[i][j] = []

					# only perform for existing layers (non-existing layers are set to [] or 0 in matrices to complete mxn-grid)
					if self.networkArchitecture[i][j] == [0,0,0]:
						filterWeights[i][j] = []
					
					# swap axes and concatenate channels and amount of filter, for getting list of 2D-filters 
					else:
						filters = np.asarray(filterRawData[filterCounter])
						filters = np.swapaxes(filters, 0, 2)
						filters = np.swapaxes(filters, 1, 3)
						filterWeights[i][j] = np.concatenate(filters, axis=0)
						filterCounter += 1

						# generate labels
						if j == 0:
							for c in range(len(self.channels)):
								for fn in range(self.filterNum):
									labels[i][j].append(self.channels[c] + ",\nfilter {}".format(fn+1))

						# labels are according to channels only in first layer, for further layers channels are the filters of the former layer
						else:
							for c in range(self.filterNum):
								for fn in range(self.filterNum):
									labels[i][j].append("channel {}".format(c+1) + ",\nfilter {}".format(fn+1))

			self.filterLabels = labels
			return filterWeights


		def prepareOutputImages(self, textfileName):
			# prepares resulting feature maps from several channels

			# import saved raw data 
			images = eval(open(self.inputDir + textfileName + self.fileName + '.txt', 'r').read())

			# iterate for all layers in mxn-matrix
			for i in range(len(self.networkArchitecture)):
				for j in range(self.maxLayers):

					# for existing layers: swap axes and flip y-axis for better display
					if images[i][j] != []:
						images[i][j] = np.flip(np.swapaxes(np.asarray(images[i][j]).reshape(self.inputShape + [self.networkArchitecture[i][j][1]]), 0, 2), axis=1)

			return images


		def prepareOutputImagesSeparate(self, textfileName):
			# prepares resulting feature maps of first layers separate for both channels
			
			# import saved raw data
			images = eval(open(self.inputDir + textfileName + self.fileName + '.txt', 'r').read())
			
			# iterate for parallel first layers and channels
			for i in range(len(self.networkArchitecture)):
				for j in range(len(self.channels)):

					# for existing layers: swap axis and flipy-axis for better display
					images[i][j] = np.flip(np.swapaxes(images[i][j], 0, 2), axis=1)
			return images


########################################################################################################################################################				
		# data visualization

		
		def doPlots(self):
			# coordinates visualization data

			# loop over all layers in mxn-matrix
			for i in range(len(self.networkArchitecture)):
				for j in range(self.maxLayers):

					# skip virtual layers '[0]'
					if self.networkArchitecture[i][j] != [0,0,0]:

						# print plotting information
						print 'create plots for column ', i+1, ', layer ', j+1

						# create name tag for identification of layer
						if len(self.networkArchitecture) > 1:
							layerIndex = 'column_{}_layer_{}_'.format(i+1,j+1)
						elif len(self.networkArchitecture) == 1 and self.maxLayers > 1:
							layerIndex = 'layer_{}_'.format(j+1)
						else:
							layerIndex = ''
						
						# create new folder for each layer to save plots
						outputDir = self.inputDir + layerIndex + 'visualization/'
						if not os.path.exists(outputDir):
							os.mkdir(outputDir)
						
						# plot filter from each layer
						self.plotFilter([self.filterBefore[i][j], self.filterAfter[i][j]], self.filterLabels[i][j], layerIndex)
						self.plotFilter([self.filterDiff[i][j]], self.filterLabels[i][j], layerIndex)
						
						# plot feature maps from each layer
						self.plotFMaps(self.inputImages, self.channels, self.outputImages[i][j], layerIndex, str(len(self.channels)) + 'ch_')

						# plot overlapping feature maps from ttH and ttbar sample (in case of several input channels only for first layer)
						if self.allImagesTTH[i][j] != []:
							self.plotFMaps([], self.channels[0], self.allImagesTTH[i][j], layerIndex, 'TTH_')
							self.plotFMaps([], self.channels[0], self.allImagesTTbar[i][j], layerIndex, 'TTbar_')

						# for first layer plot features maps from different channels separate
						if j == 0:
							for k in range(len(self.channels)):
								self.plotFMaps([self.inputImages[k]], [self.channels[k]], self.outputImagesSeparate[i][k], layerIndex, self.channels[k] + '_')
						

		def getGeometry(self, numImages):
			# get geometry for compact arrangement of feature maps

			# start with squareroot and search for integers nearby
			if type(np.sqrt(numImages)) == int:
				y = np.sqrt(numImages)
			else:
				test = int(math.floor(np.sqrt(numImages)))
				
				# test if number of feature maps can be divided by integer
				for i in range(test):
					if numImages%(test-i) == 0:
						y = test-i
						break

			# return geometry for plot
			return [numImages/y, y]	


		def plotFilter(self, filterData, labels, layerIndex):
			# plot filters before and after training as well as their differencd (two runs)

			# get geometry for arrangement of filters
			geom = self.getGeometry(len(filterData[0]))			

			# set norm and colormap for histogram
			maxElement = np.asarray(filterData).flatten().max()
			minElement = np.asarray(filterData).flatten().min()
			norm = cm.colors.Normalize(vmax=maxElement, vmin=minElement) 
			cmap = plt.cm.BuPu

			# set size of canvas and outer grid (in gridspec object: first argument for y and second for x direction)
			# for filters before and after in comparison
			if len(filterData) == 2:
				fig = plt.figure(figsize=(9.*1.5,65/18.*1.5)) # size calculated to get the right symmetry for resulting picture
				outerGrid = gridspec.GridSpec(1, 3, wspace=.3, width_ratios=[13,13,1]) # equal first and second section for filters, smaller third section for colorbar
			
			# for difference between filters before and after
			else:
				fig = plt.figure(figsize=(8., 6.63)) # size calculated for right symmetry
				outerGrid = gridspec.GridSpec(1, 2, wspace=.3, width_ratios=[20,1]) # first section for filters, second section for colorbar
			
			# set inner grid for all filters from all channels 
			for k in range(len(filterData)):
				innerGrid = gridspec.GridSpecFromSubplotSpec(geom[1], geom[0], subplot_spec=outerGrid[k], wspace=0.4, hspace=0.4)

				# plot histogram of filter weights with right norm and label
				for i in range(len(filterData[0])):
					ax=fig.add_subplot(innerGrid[i//geom[0],i%geom[0]])
					ax.set_xticks([])
					ax.set_yticks([])
					ax.set_title(labels[i], size=8)
					ax.imshow(filterData[k][i], cmap=cmap, vmin=minElement, vmax=maxElement)

			# create colorbar
			cbar_ax = fig.add_subplot(outerGrid[0, len(filterData)])
			mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm)

			# save figure (two runs for comparison and difference)
			#plt.suptitle('Visualization of CNN Filters before and after Training' + title)
			if len(filterData) == 2:
				plt.savefig(self.inputDir + layerIndex + 'visualization/filter_visualization_' + layerIndex + self.fileName + '.png')
			else:
				plt.savefig(self.inputDir + layerIndex + 'visualization/filter_difference_' + layerIndex + self.fileName + '.png')
			
			# close figure to save storage
			plt.close(fig)


		def plotFMaps(self, inputData, labels, fmaps, layerIndex, tag):
			# plots feature maps for given layer and label

			# get geometry for arrangement of feature maps
			geom = self.getGeometry(len(fmaps))	

			# set image labels (channels for input images and filter number for feature maps)
			labels_i = labels
			labels_o = []
			for i in range(self.filterNum):
				labels_o.append('filter {}:'.format(i+1))

			# set norm and colormaps of input images for histogram 
			if inputData != []: # not in case of overlapping feature maps
				maxElement_i = np.asarray(inputData).flatten().max()
				minElement_i = np.asarray(inputData).flatten().min()
				norm_i = cm.colors.Normalize(vmax=maxElement_i, vmin=minElement_i) 
				cmap_i = plt.cm.Purples
			
			# set norm and colormaps of feature maps for histogram
			maxElement_o = np.asarray(fmaps).flatten().max()
			minElement_o = np.asarray(fmaps).flatten().min()
			norm_o = cm.colors.Normalize(vmax=maxElement_o, vmin=minElement_o) 
			cmap_o = plt.cm.RdBu

			#symmetric range for negative and positive values so that white of color map matches value zero
			if np.abs(minElement_o) > maxElement_o:
				maxElement_o = np.abs(minElement_o)
			else:
				minElement_o = -maxElement_o
			
			# additional height of 1 in case of displaying input images
			height = int(inputData != [])
			
			# figure size geared to number of feature maps in x and y direction
			fig = plt.figure(figsize=(2.5 * geom[0], 2.5 * geom[1]+height))

			#change font size of axis labels
			mpl.rcParams.update({'font.size': 5})

			# outer arrangement: 2x2 grid for input images and resulting feature maps and their two colorbars
			outerGrid = gridspec.GridSpec(1, 2, wspace=.3, width_ratios=[50, 1])

			# set inner grids for better proportions between images (need to have same height and regular order)
			innerGrid = gridspec.GridSpecFromSubplotSpec(geom[1]+height, max(geom[0], 2)*2, subplot_spec=outerGrid[0,0], hspace = 0.4)
			innerGrid_bar = gridspec.GridSpecFromSubplotSpec(geom[1]+height, 1, subplot_spec=outerGrid[0,1])
			
			# plot histograms of input images in upper left section central in comparison to feature maps
			for i in range(len(inputData)):
				ax=fig.add_subplot(innerGrid[0,max(geom[0], 2)-len(labels)+len(labels)*i:max(geom[0], 2)-len(labels)+len(labels)*i+2])
				ax.set_title(labels_i[i], size=7)
				ax.set_xlabel('$\\eta$', size=6)
				ax.set_ylabel('$\\phi$', size=6, rotation=0)
				ax.imshow(inputData[i], cmap=cmap_i, vmin=minElement_i, vmax=maxElement_i, extent=[-2.3, 2.3, -np.pi, np.pi])

			# create colorbar for input images if displayed in upper right section
			if inputData != []:
				cbar_ax = fig.add_subplot(innerGrid_bar[0,0])
				mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_i, norm=norm_i)
			
			# plot histograms for all feature maps according to geometric arrangement in lower left section
			for i in range(self.filterNum):
				ax=fig.add_subplot(innerGrid[height + i//geom[0], (i%geom[0])*2:(i%geom[0])*2+2])
				ax.set_title(labels_o[i], size=7)
				ax.set_xlabel('$\\eta$', size=6)
				ax.set_ylabel('$\\phi$', size=6, rotation=0)
				ax.imshow(fmaps[i], cmap=cmap_o, vmin=minElement_o, vmax=maxElement_o, extent=[-2.3, 2.3, -np.pi, np.pi])
	
			# create colorbar for feature maps in lower right section
			cbar_ax = fig.add_subplot(innerGrid_bar[height:,0])
			mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_o, norm=norm_o)

			# save figure
			#plt.suptitle('Image before and after Convolutional Layer' + title, size= 12)
			plt.savefig(self.inputDir + layerIndex + 'visualization/feature_map_visualization_' + tag + layerIndex + self.fileName + '.png')
			
			# close figure to save storage
			plt.close(fig)


'''
TODO

name von ordner in case of genau 1 layer sollte nicht _output sein
Title bei allen plots hinzufuegen
layer Anordnung plotten
ueberpruefen ob filter difference Betrag sinnvoll ist
Ergebnisse von basic training reproduzieren -> sehen Bilder aus wie vorher?
anpassen fuer 1ch 
mit original trainingsdaten testen
code pushen
auch den von hier sichern
1ch pseudodaten testen
laborbuch
testreihen starten
'''


myVis = Visualizer('../test_multilayer2/visualization_data/', 'test_multilayer2')
myVis.doPlots()





