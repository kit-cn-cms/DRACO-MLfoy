
# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont



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
w_img = w_fig + 2*w_space 


# set vertical tile width and spacing
h_fig = 130
h_space = h_fig/3
h_img = 4 * h_fig + 5 * h_space


# set background
model_img = Image.new('RGB', (w_img, h_img), 'white')
model_draw = ImageDraw.Draw(model_img)


# draw connection lines between conv layer tiles

model_draw.line( (w_img/2., h_fig +   h_space,
			      w_img/2., h_fig + 2*h_space), fill=(0,0,0), width=5)

# draw connection line to output (different length for basic/reduced)
 
model_draw.line( (w_img/2., h_fig * 2 + h_space * 2, 
				  w_img/2., h_fig * 4 + h_space * 2), fill=(0,0,0), width=5)

# draw the first tile with input information
# text position in the middle of the tile
textPos_i = [18, (h_fig - (4*fontSize + 3*lineSpacing))/2]

# create text: input shape and input channels
inputText = "Eingangsbild 11x15,\nRotation MaxJetPt,\n"
inputText += "Kan"+u"ä"+"le:\nJetPt, JetCSV"

# draw input tile with frame and write text to it
model_draw.rectangle(           ((w_img - w_fig)/2.,         		        h_space               ,   
					             (w_img + w_fig)/2.,                h_fig + h_space               ), outline=(0,0,0), fill=(180,180,180))
drawFrame(model_draw,       (w_img - w_fig)/2.,         			    h_space               , 
					             (w_img + w_fig)/2.,                h_fig + h_space               , 5)
model_draw.multiline_text(      ((w_img - w_fig)/2. + textPos_i[0], 	    h_space + textPos_i[1]), inputText, font=font, fill='black', align='center', spacing=lineSpacing)


# draw the next tile to signal concatenation and transition to further non convolutional layers
# text position in the middle of the tile
textPos_ml = [42, (h_fig - fontSize)/2]

# create merging layer text: concatenate and flatten

mergingLayerText = 'Flatten-Schicht'

# draw merging layer tile with frame and write text to it
model_draw.rectangle(           ((w_img - w_fig)/2.,                 h_fig   + h_space*2,
					             (w_img + w_fig)/2.,                 h_fig*2 + h_space*2), outline=(0,0,0), fill=(180,180,180))
drawFrame(model_draw,       (w_img - w_fig)/2.,                 h_fig   + h_space*2,
					             (w_img + w_fig)/2.,                 h_fig*2 + h_space*2, 5)
model_draw.multiline_text(      ((w_img - w_fig)/2. + textPos_ml[0], h_fig   + h_space*2 + textPos_ml[1]), mergingLayerText, font=font, fill='black', align='center', spacing=lineSpacing)

# draw tile for further layers (only if basic model) and output tile
# text position in the middle of the tile
textPos_fl = [11, (h_fig - (3*fontSize + 2*lineSpacing))/2]
textPos_o =  [30, (h_fig - (2*fontSize +   lineSpacing))/2]

# create text for further layer tile and output tile
furtherLayerText = 'Zwischenschichten:\nDense (50 Neuronen),\nDropout (Rate 0,5)'
outputText = 'Ausgabeschicht:\nDense (1 Neuron)'


# if basic model draw further layer tile and output tile


model_draw.rectangle(     ((w_img - w_fig)/2.,                 h_fig*2 + h_space*3,
					       (w_img + w_fig)/2.,                 h_fig*3 + h_space*3), outline=(0,0,0), fill=(240,240,240))
drawFrame(model_draw,      (w_img - w_fig)/2.,                 h_fig*2 + h_space*3,
					       (w_img + w_fig)/2.,                 h_fig*3 + h_space*3, 5)
model_draw.multiline_text(((w_img - w_fig)/2. + textPos_fl[0], h_fig*2 + h_space*3 + textPos_fl[1]), furtherLayerText, font=font, fill='black', align='center', spacing=lineSpacing)

model_draw.rectangle(     ((w_img - w_fig)/2.,                 h_fig*3 + h_space*4,
					       (w_img + w_fig)/2.,                 h_fig*4 + h_space*4), outline=(0,0,0), fill=(180,180,180))
drawFrame(model_draw,      (w_img - w_fig)/2.,                 h_fig*3 + h_space*4,
					       (w_img + w_fig)/2.,                 h_fig*4 + h_space*4, 5)
model_draw.multiline_text(((w_img - w_fig)/2. + textPos_o[0],  h_fig*3 + h_space*4 + textPos_o[1]), outputText, font=font, fill='black', align='center', spacing=lineSpacing)


# save scheme as jpeg
model_img.show()
model_img.save('network_architecture_noConv.pdf')
