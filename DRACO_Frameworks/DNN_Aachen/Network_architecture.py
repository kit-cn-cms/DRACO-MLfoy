from keras import optimizers


class architecture():

	def __init__(self):
		self.sl_4j_3b = {"prenet_layer":	[100,100],
						 "mainnet_layer":[100,100],
						 "Dropout":		1.,#0.7,
						 "L2_Norm":		0.00001,
						 "batch_size":	5000,
						 "optimizer":	optimizers.Adam(lr=0.0001),
						 "activation_function": "elu",
						 "Early_stop_percent": 0.01,
						 }

		self.sl_5j_3b = {"prenet_layer":	[100,100],
						 "mainnet_layer":[100],
						 "Dropout":		1.,#0.7,
					 	 "L2_Norm":		0.00001,
						 "batch_size":	5000,
						 "optimizer":	optimizers.Adam(lr=0.0001),
						 "activation_function": "elu",
						 "Early_stop_percent": 0.01,
						}

		self.sl_6j_3b = {"prenet_layer":	[100,100],
						 "mainnet_layer":[100,100],
						 "Dropout":		1.,#0.7,
						 "L2_Norm":		0.00001,
						 "batch_size":	5000,
						 "optimizer":	optimizers.Adam(lr=0.0001),
						 "activation_function": "elu",
						 "Early_stop_percent": 0.01,
						 }

	def get_architecture(self, event_category):

		if event_category == "(N_Jets >= 6 and N_BTagsM >= 3)":
			return self.sl_6j_3b
		elif event_category == "(N_Jets == 5 and N_BTagsM >= 3)":
			return self.sl_5j_3b
		elif event_category == "(N_Jets == 4 and N_BTagsM >= 3)":
			return self.sl_4j_3b
