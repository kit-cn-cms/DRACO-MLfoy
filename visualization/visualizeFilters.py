
'''
visualizes the filter weights of the convolutional layer's filter after being read out before and after training
'''

# imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec


def prepare_plot_data(string, channels): 
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
			labels.append("ch.: " + channels[i] +", f. {}:".format(j+1))
			plot_data.append(filter_output[:,:,i,j])

	return plot_data, labels


def get_max_min(l):
	# returns lowest and greatest value of all filter weights 

	maxElement = l[0][0][0]
	minElement = l[0][0][0]

	for i in l:
		for j in i:
			if j.max() >= maxElement:
				maxElement = j.max()
			if j.min() <= minElement:
				minElement = j.min()

	return maxElement, minElement


def do_plot(mylist, labels, path, name, title):
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
		innerGrid = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outerGrid[k], wspace=0.3, hspace=0.3)
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
	plt.suptitle('Visualization of CNN Filters before (left) and after (right) Training' + title)
	plt.savefig(path+'filter_visualization'+name+'.png')
	plt.show()


def do_plot_diff(mylist, labels, path, name, title):
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
	plt.suptitle('Filter Difference before and after Training' + title)
	plt.savefig(path+'filter_visualization_diff'+name+'.png')
	plt.show()


# set paths
path = 'switch_test/_ge6j_ge3t/'
filename = '_CSV_rot_MaxJetPt'
title = ' (test)'

# set channels
channels = ['Jet_CSV', 'Jet_Pt']

# read filter data from textfile
string_before = open(path+'filterOutputs_before_training'+filename+'.txt', 'r').read() 
string_after = open(path+'filterOutputs_after_training'+filename+'.txt', 'r').read()

# prepare data for plotting
mylist_before, labels = prepare_plot_data(string_before, channels)
mylist_after, labels = prepare_plot_data(string_after, channels)

# plot filter weights before and after training on first canvas
mybiglist = mylist_before + mylist_after
do_plot(mybiglist, labels, path, filename, title)

#plot differences between filter weights before and after training on second canvas
list_difference = []
for i in range(len(mylist_after)):
	list_difference.append(np.abs(np.asarray(mylist_after[i]) - np.asarray(mylist_before[i])))
do_plot_diff(list_difference, labels, path, filename, title)







