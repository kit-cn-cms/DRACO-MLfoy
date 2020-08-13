
# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont
import colorsys


def drawFrame(draw_obj, x_min, y_min, x_max, y_max, width):
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


# set fonttype and fontsize
fontPath = "/Users/Amaterasu1/Library/Fonts/Arial Bold.ttf"
fontSize = 19
lineSpacing = 5
font = ImageFont.truetype(fontPath, fontSize, encoding="utf-8")

# set horizontal tile width and spacing
w_fig = 220
w_space = w_fig/4
w_img = 5*w_fig + 6*w_space 


# set vertical tile width and spacing
h_fig = 130
h_space = h_fig/3
h_img = h_fig + 2*h_space


# set background
model_img = Image.new('RGB', (w_img, h_img), 'white')
model_draw = ImageDraw.Draw(model_img)


# draw connection lines between conv layer tiles

model_draw.line( (w_fig, h_img/2.,
			      4*w_fig + 5*w_space, h_img/2.), fill=(0,0,0), width=5)


# draw the first tile with input information
# text position in the middle of the tile
textPos_i = [18, (h_fig - (4*fontSize + 3*lineSpacing))/2]

# create text: input shape and input channels
inputText = "Eingangsbild 11x15,\nRotation MaxJetPt,\n"
inputText += "Kan"+u"ä"+"le:\nJetPt, JetCSV"

# draw input tile with frame and write text to it
model_draw.rectangle(       (        w_space,         		 (h_img - h_fig)/2.,   
					        (w_fig + w_space,                (h_img + h_fig)/2.)), outline=(0,0,0), fill=(180,180,180))
drawFrame(model_draw,               w_space,           	 (h_img - h_fig)/2., 
					        w_fig + w_space,                (h_img + h_fig)/2., 5)
model_draw.multiline_text( (        w_space + textPos_i[0], (h_img - h_fig)/2. + textPos_i[1]), inputText, font=font, fill='black', align='center', spacing=lineSpacing)



# text position in the middle of the tile
textPos = [37, (h_fig - (4*fontSize + 3*lineSpacing))/2]


# create text: filtersize, number of filters and numer of channels
filterNum = '8 Filter,' 
filterSize = 'Filtergr'+u'ö'+u'ß'+'e 4x4,'
channels = '2 Kan'+u'ä'+'le'

textPos_f = [40, (h_fig - (3*fontSize + 2*lineSpacing))/2]

# set color of tile in HSV-mode according to filtersize and convert it to RGB-mode (needed for saving figure)
colorHSV = (170. - (190./11. * 4.))%255.
colorsRGB = []
colors = colorsys.hsv_to_rgb(colorHSV/255.,180/255.,230/255.) # takes values from 0 to 1, but draw object needs range 0 to 255
for i in range(3):
	colorsRGB.append(int(round(colors[i]*255.)))

# draw a tile with frame and write text to it
model_draw.rectangle(       (w_fig   + w_space*2,         		 (h_img - h_fig)/2.,   
					        (w_fig*2 + w_space*2,                (h_img + h_fig)/2.)), outline=(0,0,0), fill=tuple(colorsRGB))
drawFrame(model_draw,       w_fig   + w_space*2,           	 (h_img - h_fig)/2., 
					        w_fig*2 + w_space*2,                (h_img + h_fig)/2., 5)
model_draw.multiline_text( (w_fig   + w_space*2 + textPos_f[0], (h_img - h_fig)/2. + textPos_f[1]), filterNum +'\n' + filterSize + '\n' + channels, font=font, fill='black', align='center', spacing=lineSpacing)




# draw the next tile to signal concatenation and transition to further non convolutional layers
# text position in the middle of the tile
textPos_ml = [42, (h_fig - fontSize)/2]

# create merging layer text: concatenate and flatten

mergingLayerText = 'Flatten-Schicht'

# draw merging layer tile with frame and write text to it
model_draw.rectangle(       (w_fig*2 + w_space*3,         		 (h_img - h_fig)/2.,   
					        (w_fig*3 + w_space*3,                (h_img + h_fig)/2.)), outline=(0,0,0), fill=(180,180,180))
drawFrame(model_draw,       w_fig*2 + w_space*3,           	 (h_img - h_fig)/2., 
					        w_fig*3 + w_space*3,                (h_img + h_fig)/2., 5)
model_draw.multiline_text( (w_fig*2 + w_space*3 + textPos_ml[0], (h_img - h_fig)/2. + textPos_ml[1]), mergingLayerText, font=font, fill='black', align='center', spacing=lineSpacing)

# draw tile for further layers (only if basic model) and output tile
# text position in the middle of the tile
textPos_fl = [11, (h_fig - (3*fontSize + 2*lineSpacing))/2]
textPos_o =  [30, (h_fig - (2*fontSize +   lineSpacing))/2]

# create text for further layer tile and output tile
furtherLayerText = 'Zwischenschichten:\nDense (50 Neuronen),\nDropout (Rate 0,5)'
outputText = 'Ausgabeschicht:\nDense (1 Neuron)'


# if basic model draw further layer tile and output tile


model_draw.rectangle(       (w_fig*3 + w_space*4,         		 (h_img - h_fig)/2.,   
					        (w_fig*4 + w_space*4,                (h_img + h_fig)/2.)), outline=(0,0,0), fill=(240,240,240))
drawFrame(model_draw,       w_fig*3 + w_space*4,           	 (h_img - h_fig)/2., 
					        w_fig*4 + w_space*4,                (h_img + h_fig)/2., 5)
model_draw.multiline_text( ((w_fig*3 + w_space*4 + textPos_fl[0], (h_img - h_fig)/2. + textPos_fl[1])), furtherLayerText, font=font, fill='black', align='center', spacing=lineSpacing)

model_draw.rectangle(       (w_fig*4 + w_space*5,         		 (h_img - h_fig)/2.,   
					        (w_fig*5 + w_space*5,                (h_img + h_fig)/2.)), outline=(0,0,0), fill=(180,180,180))
drawFrame(model_draw,       w_fig*4 + w_space*5,           	 (h_img - h_fig)/2., 
					        w_fig*5 + w_space*5,                (h_img + h_fig)/2., 5)
model_draw.multiline_text( (w_fig*4 + w_space*5 + textPos_o[0], (h_img - h_fig)/2. + textPos_o[1]), outputText, font=font, fill='black', align='center', spacing=lineSpacing)


# save scheme as jpeg
model_img.show()
model_img.save('network_architecture_basic.pdf')
