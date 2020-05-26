'''
visualizes the network architectures scheme
visualizes the filter weights of the convolutional layer's filter after being read out before and after training as well as their differences
visualizes feature maps alltogether and for each channel separate after being lead through the filter of convolutional layer
visualizes overlapping feature maps from ttH and ttbar sample
'''

# imports
import os
import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm
import colorsys
from PIL import Image, ImageDraw, ImageFont

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

			# get input information
			channels = eval(networkArchitecture[1])
			self.channels = [channel[0:channel.index('[')] for channel in channels]
			self.inputShape = eval(networkArchitecture[2])
			rotation = networkArchitecture[3].replace('_', ' ').replace('rot', 'Rotation')
			try:
				self.rotation = rotation[rotation.index('no'):]
			except ValueError:
				self.rotation = rotation[rotation.index('Rotation'):]
			if len(networkArchitecture) > 5: self.pseudoData = networkArchitecture[5]
			else: self.pseudoData = None


			# get further model information
			self.model = networkArchitecture[4]

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
			if len(self.channels) > 1:
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

			# plot the network architecture at first
			self.plotNetworkArchitecture()

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
						if j == 0 and len(self.channels) > 1:
							for k in range(len(self.channels)):
								self.plotFMaps([self.inputImages[k]], [self.channels[k]], self.outputImagesSeparate[i][k], layerIndex, self.channels[k] + '_')
						

		def drawFrame(self, draw_obj, x_min, y_min, x_max, y_max, width):
			# draws frame around tiles

			# vertical lines
			draw_obj.line( (x_min + width/2, y_min,
				            x_min + width/2, y_max), fill=(0,0,0), width=width)
			draw_obj.line( (x_max - width/2, y_min,
				            x_max - width/2, y_max), fill=(0,0,0), width=width)

			# horizontal lines
			draw_obj.line( (x_min, y_min + width/2,
				            x_max, y_min + width/2), fill=(0,0,0), width=width)
			draw_obj.line( (x_min, y_max + width/2,
				            x_max, y_max + width/2), fill=(0,0,0), width=width)


		def plotNetworkArchitecture(self):
			# draws a scheme of the network architecture

			# set fonttype and fontsize
			fontPath = "/Users/Amaterasu1/Library/Fonts/Arial Bold.ttf"
			fontSize = 20
			lineSpacing = 5
			titleFontSize = 60
			titleFont = ImageFont.truetype(fontPath, titleFontSize-30)
			font = ImageFont.truetype(fontPath, fontSize)

			# set horizontal tile width and spacing
			n_w = len(self.networkArchitecture)
			w_fig = 220
			w_space = w_fig/4
			w_img = n_w * w_fig + (n_w + 1) * w_space 

			# set vertical tile width and spacing
			n_h = len(self.networkArchitecture[0])
			h_fig = 130
			h_space = h_fig/3
			if self.model == 'basic':
				h_img = (n_h + 6) * h_fig + (n_h + 3) * h_space + titleFontSize
			else:
				h_img = (n_h + 5) * h_fig + (n_h + 2) * h_space + titleFontSize

			# set background
			model_img = Image.new('RGB', (w_img, h_img), 'white')
			model_draw = ImageDraw.Draw(model_img)

			# set title
			model_draw.multiline_text( (w_img/2 - 155, 30), 'Network Architecture', font=titleFont, fill='black', align='center', spacing=lineSpacing)

			# draw connection lines between conv layer tiles
			for x in range(n_w):

				model_draw.line( (w_img/2.                          , h_fig           + h_space       + titleFontSize, 
								  w_fig * (x+1/2.) + w_space * (x+1), h_fig *      2  + h_space       + titleFontSize), fill=(0,0,0), width=5)
				model_draw.line( (w_fig * (x+1/2.) + w_space * (x+1), h_fig *      2  + h_space       + titleFontSize, 
								  w_fig * (x+1/2.) + w_space * (x+1), h_fig * (n_h+2) + h_space * n_h + titleFontSize), fill=(0,0,0), width=5)
				model_draw.line( (w_fig * (x+1/2.) + w_space * (x+1), h_fig * (n_h+2) + h_space * n_h + titleFontSize,
								  w_img/2.                          , h_fig * (n_h+3) + h_space * n_h + titleFontSize), fill=(0,0,0), width=5)

			# draw connection line to output (different length for basic/reduced)
			if self.model == 'basic':  
				model_draw.line( (w_img/2., h_fig * (n_h+4) + h_space *  n_h    + titleFontSize, 
								  w_img/2., h_fig * (n_h+5) + h_space * (n_h+2) + titleFontSize), fill=(0,0,0), width=5)
			else:
				model_draw.line( (w_img/2., h_fig * (n_h+4) + h_space *  n_h    + titleFontSize, 
								  w_img/2., h_fig * (n_h+4) + h_space * (n_h+1) + titleFontSize), fill=(0,0,0), width=5)

			# draw the first tile with input information
			# text position in the middle of the tile
			textPos_i = [25, (h_fig - (4*fontSize + 3*lineSpacing))/2]

			# create text: input shape and input channels
			inputText = 'Input Shape ' + str(self.inputShape[0]) + 'x' + str(self.inputShape[1]) + '\n' + self.rotation + '\n'
			if len(self.channels) == 1:
				inputText += 'Channel ' + self.channels[0].replace('_', '')
				if self.pseudoData == 'pseudo': inputText += '\n Pseudo Data'
				else: textPos_i[1] = (h_fig - (3*fontSize + 2*lineSpacing))/2
			else:
				inputText += 'Channels:'
				inputText += '\n' + self.channels[0].replace('_', '')
				if len(self.channels) > 1:
					for i in range(len(self.channels)-1):
						inputText += ', ' + self.channels[i+1].replace('_', '')

			# draw input tile with frame and write text to it
			model_draw.rectangle(           ((w_img - w_fig)/2.,         		        h_space                + titleFontSize,   
								             (w_img + w_fig)/2.,                h_fig + h_space                + titleFontSize), outline=(0,0,0), fill=(180,180,180))
			self.drawFrame(model_draw,       (w_img - w_fig)/2.,         			    h_space                + titleFontSize, 
								             (w_img + w_fig)/2.,                h_fig + h_space                + titleFontSize, 5)
			model_draw.multiline_text(      ((w_img - w_fig)/2. + textPos_i[0], 	    h_space + textPos_i[1] + titleFontSize), inputText, font=font, fill='black', align='center', spacing=lineSpacing)

			# draw a tile for each layer with layer information
			# loop over all layers and only recognize the real layers (not virutal ones with [0,0,0])
			for x in range(n_w):
				for y in range(n_h):
					if not self.networkArchitecture[x][y] == [0,0,0]:

						# text position in the middle of the tile
						textPos = [45, (h_fig - (3*fontSize + 2*lineSpacing))/2]
						if self.networkArchitecture[x][y][0] >= 10: textPos[0] = 32

						# create text: filtersize, number of filters and numer of channels
						filterNum = str(self.networkArchitecture[x][y][1]) + ' Filters' 
						filterSize = 'Filter Size ' + str(self.networkArchitecture[x][y][0]) + 'x' + str(self.networkArchitecture[x][y][0])
						channels = str(self.networkArchitecture[x][y][2]) + ' Channel'
						if self.networkArchitecture[x][y][2] > 1:
							channels += 's'

						# set color of tile in HSV-mode according to filtersize and convert it to RGB-mode (needed for saving figure)
						colorHSV = (170. - (190./self.inputShape[0] * self.networkArchitecture[x][y][0]))%255.
						colorsRGB = []
						colors = colorsys.hsv_to_rgb(colorHSV/255.,180/255.,230/255.) # takes values from 0 to 1, but draw object needs range 0 to 255
						for i in range(3):
							colorsRGB.append(int(round(colors[i]*255.)))

						# draw a tile with frame and write text to it
						model_draw.rectangle(           (w_fig *  x    + w_space * (x+1),              h_fig * (y+2) + h_space * (y+1)               + titleFontSize, 
											             w_fig * (x+1) + w_space * (x+1),              h_fig * (y+3) + h_space * (y+1)               + titleFontSize), outline=(0,0,0), fill=tuple(colorsRGB))

						self.drawFrame(model_draw,       w_fig *  x    + w_space * (x+1),              h_fig * (y+2) + h_space * (y+1)               + titleFontSize, 
											             w_fig * (x+1) + w_space * (x+1),              h_fig * (y+3) + h_space * (y+1)               + titleFontSize, 5)

						model_draw.multiline_text(      (w_fig *  x    + w_space * (x+1) + textPos[0], h_fig * (y+2) + h_space * (y+1) + textPos[1]  + titleFontSize), filterNum +'\n' + filterSize + '\n' + channels, font=font, fill='black', align='center', spacing=lineSpacing)

			# draw the next tile to signal concatenation and transition to further non convolutional layers
			# text position in the middle of the tile
			textPos_ml = [80, (h_fig - fontSize)/2]

			# create merging layer text: concatenate and flatten
			mergingLayerText = ''
			if n_w > 1:
				mergingLayerText += 'Concatenate\n& '
				textPos_ml = [50,(h_fig - (2*fontSize + 2*lineSpacing))/2]
			mergingLayerText += 'Flatten'

			# draw merging layer tile with frame and write text to it
			model_draw.rectangle(           ((w_img - w_fig)/2.,                 h_fig * (n_h+3) + h_space * n_h                + titleFontSize,
								             (w_img + w_fig)/2.,                 h_fig * (n_h+4) + h_space * n_h                + titleFontSize), outline=(0,0,0), fill=(180,180,180))
			self.drawFrame(model_draw,       (w_img - w_fig)/2.,                 h_fig * (n_h+3) + h_space * n_h                + titleFontSize,
								             (w_img + w_fig)/2.,                 h_fig * (n_h+4) + h_space * n_h                + titleFontSize, 5)
			model_draw.multiline_text(      ((w_img - w_fig)/2. + textPos_ml[0], h_fig * (n_h+3) + h_space * n_h + textPos_ml[1] + titleFontSize), mergingLayerText, font=font, fill='black', align='center', spacing=lineSpacing)

			# draw tile for further layers (only if basic model) and output tile
			# text position in the middle of the tile
			textPos_fl = [20, (h_fig - (3*fontSize + 2*lineSpacing))/2]
			textPos_o =  [40, (h_fig - (2*fontSize +   lineSpacing))/2]

			# create text for further layer tile and output tile
			furtherLayerText = 'Further Layers\nDense (50 Neurons)\nDropout (Rate 0,5)'
			outputText = 'Output\nDense (1 Node)'
			if self.model == 'reduced_untr':
				outputText += '\nuntrainable'
				textPos_o[1] = (h_fig - (3*fontSize + 2*lineSpacing))/2

			# if basic model draw further layer tile and output tile
			if self.model == 'basic':

				model_draw.rectangle(           ((w_img - w_fig)/2.,                 h_fig * (n_h+4) + h_space * (n_h+1)                + titleFontSize,
									             (w_img + w_fig)/2.,                 h_fig * (n_h+5) + h_space * (n_h+1)                + titleFontSize), outline=(0,0,0), fill=(240,240,240))
				self.drawFrame(model_draw,       (w_img - w_fig)/2.,                 h_fig * (n_h+4) + h_space * (n_h+1)                + titleFontSize,
									             (w_img + w_fig)/2.,                 h_fig * (n_h+5) + h_space * (n_h+1)                + titleFontSize, 5)
				model_draw.multiline_text(      ((w_img - w_fig)/2. + textPos_fl[0], h_fig * (n_h+4) + h_space * (n_h+1) + textPos_fl[1] + titleFontSize), furtherLayerText, font=font, fill='black', align='center', spacing=lineSpacing)

				model_draw.rectangle(           ((w_img - w_fig)/2.,                 h_fig * (n_h+5) + h_space * (n_h+2)                + titleFontSize,
									             (w_img + w_fig)/2.,                 h_fig * (n_h+6) + h_space * (n_h+2)                + titleFontSize), outline=(0,0,0), fill=(180,180,180))
				self.drawFrame(model_draw,       (w_img - w_fig)/2.,                 h_fig * (n_h+5) + h_space * (n_h+2)                + titleFontSize,
									             (w_img + w_fig)/2.,                 h_fig * (n_h+6) + h_space * (n_h+2)                + titleFontSize, 5)
				model_draw.multiline_text(      ((w_img - w_fig)/2. + textPos_o[0],  h_fig * (n_h+5) + h_space * (n_h+2) + textPos_o[1] + titleFontSize), outputText, font=font, fill='black', align='center', spacing=lineSpacing)

			# if reduced model only draw output tile
			else:

				model_draw.rectangle(           ((w_img - w_fig)/2.,                 h_fig * (n_h+4) + h_space * (n_h+1)                + titleFontSize,
									             (w_img + w_fig)/2.,                 h_fig * (n_h+5) + h_space * (n_h+1)                + titleFontSize), outline=(0,0,0), fill=(180,180,180))
				self.drawFrame(model_draw,       (w_img - w_fig)/2.,                 h_fig * (n_h+4) + h_space * (n_h+1)                + titleFontSize,
									             (w_img + w_fig)/2.,                 h_fig * (n_h+5) + h_space * (n_h+1)                + titleFontSize, 5)
				model_draw.multiline_text(      ((w_img - w_fig)/2. + textPos_o[0],  h_fig * (n_h+4) + h_space * (n_h+1) + textPos_o[1] + titleFontSize), outputText, font=font, fill='black', align='center', spacing=lineSpacing)

			# save scheme as jpeg
			model_img.show()
			model_img.save(self.inputDir + 'network_architecture_' + self.fileName + '.jpg')


		def getGeometry(self, numImages):
			# get geometry for compact arrangement of feature maps

			# start with squareroot and search for integers nearby
			if int(np.sqrt(numImages)) == np.sqrt(numImages):
				y = int(np.sqrt(numImages))
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
				outerGrid = gridspec.GridSpec(2, 3, wspace=.3, width_ratios=[13,13,1], height_ratios=[1, geom[1]*20]) # upper line stays free to make room for title, lower line sections: equal first and second section for filters, smaller third section for colorbar
			
			# for difference between filters before and after
			else:
				fig = plt.figure(figsize=(8., 6.63)) # size calculated for right symmetry
				outerGrid = gridspec.GridSpec(2, 2, wspace=.3, width_ratios=[20,1], height_ratios=[1, geom[1]*20]) # upper line to make space, lower line first section for filters, second section for colorbar
			
			# set inner grid for all filters from all channels 
			for k in range(len(filterData)):
				innerGrid = gridspec.GridSpecFromSubplotSpec(geom[1], geom[0], subplot_spec=outerGrid[1,k], wspace=0.4, hspace=0.4)

				# plot histogram of filter weights with right norm and label
				for i in range(len(filterData[0])):
					ax=fig.add_subplot(innerGrid[i//geom[0],i%geom[0]])
					ax.set_xticks([])
					ax.set_yticks([])
					ax.set_title(labels[i], size=8)
					ax.imshow(filterData[k][i], cmap=cmap, vmin=minElement, vmax=maxElement)

			# create colorbar
			cbar_ax = fig.add_subplot(outerGrid[1, len(filterData)])
			mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm)

			# save figure with title (two runs for comparison and difference)
			if len(filterData) == 2:
				plt.suptitle('Visualization of CNN Filters before and after Training\n' + layerIndex.replace('_',' ') + '(' + self.fileName.replace('_', ' ').replace('rot', 'rotation') + ')', fontsize = 10)
				plt.savefig(self.inputDir + layerIndex + 'visualization/filter_visualization_' + layerIndex + self.fileName + '.png')
			else:
				plt.suptitle('Filters Difference\n' + layerIndex.replace('_',' ') + '(' + self.fileName.replace('_', ' ').replace('rot', 'rotation') + ')', fontsize = 10)
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

			# outer arrangement: first line stays empty to make room for the figure title, second line for input images and resulting feature maps and their two colorbars
			outerGrid = gridspec.GridSpec(2, 2, wspace=.3, width_ratios=[geom[0]*15, 1], height_ratios=[1, (geom[0]+height)*10])

			# set inner grids for better proportions between images (need to have same height and regular order)
			innerGrid = gridspec.GridSpecFromSubplotSpec(geom[1]+height, max(geom[0], 2)*2, subplot_spec=outerGrid[1,0], hspace = 0.4)
			innerGrid_bar = gridspec.GridSpecFromSubplotSpec(geom[1]+height, 1, subplot_spec=outerGrid[1,1])
			
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
				if self.filterNum == 1:
					ax=fig.add_subplot(innerGrid[height + i//geom[0], 1:3])
				else:
					ax=fig.add_subplot(innerGrid[height + i//geom[0], (i%geom[0])*2:(i%geom[0])*2+2])
				ax.set_title(labels_o[i], size=7)
				ax.set_xlabel('$\\eta$', size=6)
				ax.set_ylabel('$\\phi$', size=6, rotation=0)
				ax.imshow(fmaps[i], cmap=cmap_o, vmin=minElement_o, vmax=maxElement_o, extent=[-2.3, 2.3, -np.pi, np.pi])
	
			# create colorbar for feature maps in lower right section
			cbar_ax = fig.add_subplot(innerGrid_bar[height:,0])
			mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_o, norm=norm_o)

			# create figure title
			if tag == str(len(self.channels)) + 'ch_':
				titleTag = ' of both channels\n'
			elif tag == 'TTH_':
				titleTag = ' of overlapping TTH Samples\n'
			elif tag == 'TTbar_':
				titleTag = ' of overlapping TTbar Samples\n'
			else:
				for channel in self.channels:
					if tag == channel + '_':
						titleTag = ' of channel' + channel.replace('_', ' ') + '\n'
			
			# save figure with title
			plt.suptitle('Feature Map Visualization' + titleTag + layerIndex.replace('_',' ') + '(' + self.fileName.replace('_', ' ').replace('rot', 'rotation') + ')', fontsize = 10)
			plt.savefig(self.inputDir + layerIndex + 'visualization/feature_map_visualization_' + tag + layerIndex + self.fileName + '.png')
			
			# close figure to save storage
			plt.close(fig)


#######################################################################################################################################


myVis = Visualizer('../../Series_Multilayer/matrix_descending/visualization_data/', 'matrix_descending')
myVis.doPlots()





