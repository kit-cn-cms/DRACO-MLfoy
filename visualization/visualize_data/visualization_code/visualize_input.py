
'''
overlaps 10.000 input pictures and draws one picture for every rotation type
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
	image = np.array(image).reshape(11,15) # settings: 11x15 image, 

	# swap axes for plots, turn image 90 degrees, put origin in lower left corner
	output_image = np.flip(np.swapaxes(np.array(image),0, 1), axis = 0)

	return output_image

def get_max_min(l):
	# returns lowest and greatest value of all input values

	maxElement = l[0][0][0]
	minElement = l[0][0][0]

	for i in l:
		for j in i:
			if j.max() >= maxElement:
				maxElement = j.max()
			if j.min() <= minElement:
				minElement = j.min()

	return maxElement, minElement

def do_plot(mylist, path, labels):
	# plots the overlapping input pictures

	# set norm for histogram
	maxElement, minElement = get_max_min(mylist)
	norm = cm.colors.Normalize(vmax=0, vmin=1) 
	cmap = plt.cm.Purples

	mylist = np.asarray(mylist)/maxElement

	# set size of canvas
	f = plt.figure(figsize=(8., 6.)) 

	# arrangement: 6 input pictures for 6 rotations
	outerGrid = gridspec.GridSpec(1, 2, wspace=.3, width_ratios=[30,1])
	innerGrid = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outerGrid[0], wspace=0.4, hspace=0.4)
	
	for i in range (2):
		for j in range(3):
			ax=f.add_subplot(innerGrid[i:i+1,j:j+1])
			ax.set_xlabel('$\\eta$', size=8)
			ax.set_ylabel('$\\phi$', size=8, rotation=0)
			ax.set_title(labels[i*3+j])
			ax.imshow(mylist[i*3+j], cmap=cmap, vmin=0, vmax=1, extent=[-2.2, 2.2, -np.pi, np.pi])

	# create colorbar
	cbar_ax = f.add_subplot(outerGrid[:,1:2])
	mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm)

	# save figure
	#plt.suptitle('Overlapping Input Images ttH', size= 12)
	plt.savefig(path+'all_images_ttH.png')
	plt.show()


# set paths
path = '../input_images/' 
filenames = ['_no_rot','_rot_MaxJetPt','_rot_TopLep','_rot_sph1','_rot_sph2','_rot_sph3']
labels = ['no rotation', 'MaxJetPt', 'TopLep', 'EV sphericity 1', 'EV sphericity 2', 'EV sphericity 3']
image_data = []
for name in filenames:
	# read filter data from textfile
	string = open(path+'all_input_images_ttH_CSV'+name+'.txt', 'r').read() 
	image_data.append(prepare_image_before_data(string))



print np.asarray(image_data).shape
do_plot(image_data, path, labels)







