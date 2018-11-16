from keras import optimizers

class Architecture():
	def __init__(self):
		self.sl_4j_3b = {
            "prenet_layer":	        [100,100],
            "prenet_loss":          'categorical_crossentropy',
            "mainnet_layer":        [100,100],
            "mainnet_loss":         "kullback_leibler_divergence",
            "Dropout":		        1.,
            "L2_Norm":		        1e-5,
            "batch_size":	        4096,
            "optimizer":	        optimizers.Adam(1e-4),
            "activation_function":  "elu",
            "Early_stop_percent":   0.01,
            }

		self.sl_5j_3b = {
            "prenet_layer":	        [100,100],
            "prenet_loss":          'categorical_crossentropy',
            "mainnet_layer":        [100],
            "mainnet_loss":         "kullback_leibler_divergence",
            "Dropout":		        1.,
            "L2_Norm":		        1e-5,
            "batch_size":	        4096,
            "optimizer":	        optimizers.Adam(1e-4),
            "activation_function":  "elu",
            "Early_stop_percent":   0.01,
            }

		self.sl_6j_3b = {
            "prenet_layer":	        [100,100],
            "prenet_loss":          'categorical_crossentropy',
            "mainnet_layer":        [100,100],
            "mainnet_loss":         "kullback_leibler_divergence",
            "Dropout":		        1.,
            "L2_Norm":		        1e-5,
            "batch_size":	        4096,
            "optimizer":	        optimizers.Adam(1e-4),
            "activation_function":  "elu",
            "Early_stop_percent":   0.01,
            }

	def get_architecture(self, event_category):

		if event_category == "(N_Jets >= 6 and N_BTagsM >= 3)":
			return self.sl_6j_3b
		elif event_category == "(N_Jets == 5 and N_BTagsM >= 3)":
			return self.sl_5j_3b
		elif event_category == "(N_Jets == 4 and N_BTagsM >= 3)":
			return self.sl_4j_3b
