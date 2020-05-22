'''
visualizes both channels of one input image and 8 feature maps after being lead through 8 filters of convolutional layer
manual visualization
'''

# imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec


def prepare_image_before_data(string): 
	# takes input image data as string and converts it into an array of two 2D-arrays one for each channel

	image = string.replace(' ', '').replace('[', '').replace(']', '').split(',')
	for i in range(len(image)): 
		image[i] = float(image[i])
	image = np.array(image).reshape(11,15,2) # settings: 11x15 image, two channels

	# swap axes for plots, turn image 90 degrees, put origin in lower left corner
	channels = []
	for i in range(2):
		channels.append(np.flip(np.swapaxes(np.array(image[:,:,i]),0, 1), axis = 0))

	return channels


def prepare_image_after_data(string): 
	# takes output image data as string and converts it into an array of eight 2D-arrays one for each filter

	image = string.replace(' ', '').replace('[', '').replace(']', '').split(',')
	for i in range(len(image)): 
		image[i] = float(image[i])
	image = np.array(image).reshape(8,12,8) # settings: 8x12 image without phi-padding, 8 filter outputs

	# swap axes for plots, turn image 90 degrees, put origin in lower left corner
	images = []
	for i in range(8): # num filters
		images.append(np.flip(np.swapaxes(np.array(image[:,:,i]),0, 1), axis = 0))

	return images


def get_max_min(l):
	# returns lowest and greatest value of all images

	maxElement = l[0][0][0]
	minElement = l[0][0][0]

	for i in l:
		for j in i:
			if j.max() >= maxElement:
				maxElement = j.max()
			if j.min() <= minElement:
				minElement = j.min()

	return maxElement, minElement


def do_plot_2ch(inputs, outputs, path, name, title, channels):
	# plots the two input channels and all eight resulting feature maps

	# set image labels
	labels_i = channels
	labels_o = []
	for i in range(8): #numfilters
		labels_o.append('filter {}:'.format(i+1))

	# set norms for histogram of input and output images seperately
	maxElement_i, minElement_i = get_max_min(inputs)
	norm_i = cm.colors.Normalize(vmax=maxElement_i, vmin=minElement_i) 
	maxElement_o, minElement_o = get_max_min(outputs)
	norm_o = cm.colors.Normalize(vmax=maxElement_o, vmin=minElement_o) 
	cmap_i = plt.cm.GnBu
	cmap_o = plt.cm.GnBu

	# set size of canvas and fontsize
	f = plt.figure(figsize=(14., 5.))
	mpl.rcParams.update({'font.size': 5})

	# outer arrangement: two channel input image, colorbar, eight feature maps, colorbar
	outerGrid = gridspec.GridSpec(1, 5, wspace=.3, width_ratios=[20, 0.8, 1, 30, 0.8])

	# inner grids for better proportions
	innerGrid_i = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=outerGrid[0], wspace=0.3, hspace=0.1) 
	innerGrid_bar_i = gridspec.GridSpecFromSubplotSpec(200, 1, subplot_spec=outerGrid[1], wspace=0.3, hspace=0.1)
	innerGrid_o = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outerGrid[3], wspace=0.5, hspace=0.1)
	innerGrid_bar_o = gridspec.GridSpecFromSubplotSpec(25, 1, subplot_spec=outerGrid[4], wspace=0.3, hspace=0.1) 

	# fill first section with input images
	for i in range(2):
		ax=f.add_subplot(innerGrid_i[1:3, i])
		ax.set_title(labels_i[i], size=8)
		ax.set_xlabel('$\\phi$', size=8)
		ax.set_ylabel('$\\eta$', size=8, rotation=0)
		ax.imshow(inputs[i], cmap=cmap_i, vmin=minElement_i, vmax=maxElement_i, extent=[-2.3, 2.3, -np.pi, np.pi])
	
	# colorbar norm of input images
	cbar_ax = f.add_subplot(innerGrid_bar_i[53:147, 0])
	mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_i, norm=norm_i)
	
	# fill fourth (third one is a gap) segment with output feature maps
	for i in range(8):
		ax=f.add_subplot(innerGrid_o[i//4, i%4])
		ax.set_title(labels_o[i], size=8)
		ax.set_xlabel('$\\phi$', size=8)
		ax.set_ylabel('$\\eta$', size=8, rotation=0)
		ax.imshow(outputs[i], cmap=cmap_o, vmin=minElement_o, vmax=maxElement_o, extent=[-2.2, 2.2, -np.pi, np.pi])
	
	# colorbar norm of output images
	cbar_ax = f.add_subplot(innerGrid_bar_o[2:23, 0])
	mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_o, norm=norm_o)

	# save figure
	plt.suptitle('Image before and after Convolutional Layer' + title, size= 12)
	plt.savefig(path+'feature_map_visualization'+name+'_2ch.png')
	plt.show()


def do_plot_1ch(inputs, outputs, path, name, title, channels):
	# repeat for one channel only and eight resulting feature maps

	# set image labels
	labels_i = channels
	labels_o = []
	for i in range(8): #numfilters
		labels_o.append('filter {}:'.format(i+1))

	# set norm for histogram of output images
	maxElement_i, minElement_i = get_max_min(inputs)
	norm_i = cm.colors.Normalize(vmax=maxElement_i, vmin=minElement_i) 
	cmap_i = plt.cm.GnBu

	# set size of canvas and fontsize
	f = plt.figure(figsize=(7., 6.))
	mpl.rcParams.update({'font.size': 5})

	# outer arrangement: two channel input image, colorbar, eight feature maps, colorbar
	outerGrid = gridspec.GridSpec(2, 5, wspace=.3, hspace=0.5, width_ratios=[7, 0.8, 1, 30, 0.8])

	for index in range(len(channels)):

		# set norm for histogram of output images
		maxElement_o, minElement_o = get_max_min(outputs[index])
		norm_o = cm.colors.Normalize(vmax=maxElement_o, vmin=minElement_o) 
		cmap_o = plt.cm.GnBu

		# inner grids for better proportions
		innerGrid_bar_i = gridspec.GridSpecFromSubplotSpec(50, 1, subplot_spec=outerGrid[index,1], wspace=0.3, hspace=0.1)
		innerGrid_o = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outerGrid[index,3], wspace=0.1, hspace=0.8)
		
		# fill first section with input image
		ax=f.add_subplot(outerGrid[index, 0])
		ax.set_title(labels_i[index], size=6)
		ax.set_xlabel('$\\phi$', size=6)
		ax.set_ylabel('$\\eta$', size=6, rotation=0)
		ax.imshow(inputs[index], cmap=cmap_i, vmin=minElement_i, vmax=maxElement_i, extent=[-2.3, 2.3, -np.pi, np.pi])
	
		# colorbar norm of input images
		cbar_ax = f.add_subplot(innerGrid_bar_i[11:39, 0])
		mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_i, norm=norm_i)
		
		# fill fourth (third one is a gap) segment with output feature maps
		for i in range(8):
			ax=f.add_subplot(innerGrid_o[i//4, i%4])
			ax.set_title(labels_o[i], size=6)
			ax.set_xlabel('$\\phi$', size=6)
			ax.set_ylabel('$\\eta$', size=6, rotation=0)
			ax.imshow(outputs[index][i], cmap=cmap_o, vmin=minElement_o, vmax=maxElement_o, extent=[-2.2, 2.2, -np.pi, np.pi])
		
		# colorbar norm of output images
		cbar_ax = f.add_subplot(outerGrid[index, 4])
		mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_o, norm=norm_o)

	# save figure
	plt.suptitle('Image before and after Convolutional Layer' + title, size= 12)
	plt.savefig(path+'feature_map_visualization'+name+'_1ch.png')
	plt.show()


# set paths
path = "test/"
filename = "_test"
title = " (test)"

# set channels
channels = ['Jet_Pt', 'Jet_CSV']

# read image data from textfile 
string_input = open(path+'input_image'+filename+'.txt', 'r').read() 
string_output_2ch = open(path+'output_image_2ch'+filename+'.txt', 'r').read()
string_output_Jet_Pt = open(path+'output_image_Jet_Pt'+filename+'.txt', 'r').read()
string_output_second_ch = open(path+'output_image_'+channels[1]+filename+'.txt', 'r').read()

# prepare data for plotting
inputs = prepare_image_before_data(string_input)
outputs_2ch = prepare_image_after_data(string_output_2ch)
outputs_Jet_Pt = prepare_image_after_data(string_output_Jet_Pt)
outputs_second_channel = prepare_image_after_data(string_output_second_ch)

# plot feature maps
do_plot_2ch(inputs, outputs_2ch, path, filename, title, channels)
do_plot_1ch(inputs, [outputs_Jet_Pt, outputs_second_channel], path, filename, title, channels)







