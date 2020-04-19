'''
visualizes the filter weights of the convolutional layer's filter after being read out before and after training
visualizes both channels of one input image and 8 feature maps after being lead through 8 filters of convolutional layer
'''

# imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import sys

# option_handler
import optionHandler_vis
options = optionHandler_vis.optionHandler_vis(sys.argv)


def prepare_filter_data(string, channels): 
	# takes filter data as string and converts it into an array of 2D-arrays (one for each filter)

	filter_output = string.replace(' ', '').replace('[', '').replace(']', '').split(',')
	for i in range(len(filter_output)): 
		filter_output[i] = float(filter_output[i])
	filter_output = np.array(filter_output).reshape(4,4,2,8) # settings: 8 filters, size 4x4, 2 channels

	#swap axes for plots
	plot_data = []
	labels = []
	for i in range(2):
		for j in range(8):
			labels.append(channels[i] +",\nfilter {}".format(j+1))
			plot_data.append(filter_output[:,:,i,j])

	return plot_data, labels


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


def do_plot_fmap_2ch(inputs, outputs, path, name, title, channels):
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


def do_plot_fmap_1ch(inputs, outputs, path, name, title, channels):
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


def do_plot_filter(mylist, labels, path, name, title):
	# plots filters before and after training next to each other for better observation

	# set norm for histogram
	maxElement, minElement = get_max_min(mylist)
	norm = cm.colors.Normalize(vmax=maxElement, vmin=minElement) 
	cmap = plt.cm.BuPu

	# set size of canvas
	f = plt.figure(figsize=(9.*1.5,65/18.*1.5)) # size calculated to get the right symmetry for resulting picture

	# outer arrangement: 16 filters (2 channels with 8 filters each) before training, 16 filters after training, colorbar
	outerGrid = gridspec.GridSpec(1, 3, wspace=.3, width_ratios=[13,13,1])

	# fill outer grid with 4x4 inner grid for 16 filters
	for k in range(2):
		innerGrid = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outerGrid[k], wspace=0.4, hspace=0.4)
		for i in range(4):
			for j in range(4):
				ax=f.add_subplot(innerGrid[i:i+1,j:j+1])
				ax.set_xticks([])
				ax.set_yticks([])
				ax.set_title(labels[i*4+j], size=8)
				ax.imshow(mylist[k*16+i*4+j], cmap=cmap, vmin=minElement, vmax=maxElement)

	# create colorbar
	cbar_ax = f.add_subplot(outerGrid[:,2:3])
	mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm)

	# save figure
	plt.suptitle('Visualization of CNN Filters before and after Training' + title)
	plt.savefig(path+'filter_visualization'+name+'.png')
	plt.show()


def do_plot_filter_diff(mylist, labels, path, name, title):
	# plots the differences between filter weights before and after training

	# set norm for histogram
	maxElement, minElement = get_max_min(mylist)
	norm = cm.colors.Normalize(vmax=maxElement, vmin=minElement) 
	cmap = plt.cm.BuPu

	# set size of canvas
	f = plt.figure(figsize=(8., 6.63)) # size calculated for right symmetry

	# outer arrangement: 16 filters and colorbar
	outerGrid = gridspec.GridSpec(1, 2, wspace=.3, width_ratios=[20,1])

	# fill outer grid with 4x4 inner grid for 16 filters
	innerGrid = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outerGrid[0], wspace=0.3, hspace=0.3)
	for i in range(4):
		for j in range(4):
			ax=f.add_subplot(innerGrid[i:i+1,j:j+1])
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_title(labels[i*4+j], size=8)
			ax.imshow(mylist[i*4+j], cmap=cmap, vmin=minElement, vmax=maxElement)

	# create colorbar
	cbar_ax = f.add_subplot(outerGrid[:,1:2])
	mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm)

	# save figure
	plt.suptitle('Filter Difference' + title)
	plt.savefig(path+'filter_visualization_diff'+name+'.png')
	plt.show()


# set paths
path = options.getInputDir()
filename = options.getFileName()
if options.getTitle() != "":
	title = " (" + options.getTitle() +")"
else:
	title = options.getTitle()

# set channels
channels = ['Jet_Pt', options.getSecondChannel()]

# read filter data from textfile
string_before = open(path+'filterOutputs_before_training'+filename+'.txt', 'r').read() 
string_after = open(path+'filterOutputs_after_training'+filename+'.txt', 'r').read()
string_input = open(path+'input_image'+filename+'.txt', 'r').read() 
string_output_2ch = open(path+'output_image_2ch'+filename+'.txt', 'r').read()
string_output_Jet_Pt = open(path+'output_image_Jet_Pt'+filename+'.txt', 'r').read()
string_output_second_ch = open(path+'output_image_'+channels[1]+filename+'.txt', 'r').read()

# prepare data for plotting
mylist_before, labels = prepare_filter_data(string_before, channels)
mylist_after, labels = prepare_filter_data(string_after, channels)
inputs = prepare_image_before_data(string_input)
outputs_2ch = prepare_image_after_data(string_output_2ch)
outputs_Jet_Pt = prepare_image_after_data(string_output_Jet_Pt)
outputs_second_ch = prepare_image_after_data(string_output_second_ch)

# plot filter weights before and after training on first canvas
mybiglist = mylist_before + mylist_after
do_plot_filter(mybiglist, labels, path, filename, title)

#plot differences between filter weights before and after training on second canvas
list_difference = []
for i in range(len(mylist_after)):
	list_difference.append(np.abs(np.asarray(mylist_after[i]) - np.asarray(mylist_before[i])))
do_plot_filter_diff(list_difference, labels, path, filename, title)

# plot feature maps on next two canvases
do_plot_fmap_2ch(inputs, outputs_2ch, path, filename, title, channels)
do_plot_fmap_1ch(inputs, [outputs_Jet_Pt, outputs_second_ch], path, filename, title, channels)







