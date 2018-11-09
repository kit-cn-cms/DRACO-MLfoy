from keras import optimizers

sl_4j_3b = {"prenet_layer":	[100,100],
			"mainnet_layer":[100,100],
			"Dropout":		0.7,
			"L2_Norm":		0.00001,
			"batch_size":	5000,
			"optimizer":	optimizers.Adam(lr=0.0001),
			"activation_function": "elu",
			"Early_stop_percent": 0.01,
			}

sl_5j_3b = {"prenet_layer":	[100,100],
			"mainnet_layer":[100],
			"Dropout":		0.7,
			"L2_Norm":		0.00001,
			"batch_size":	5000,
			"optimizer":	optimizers.Adam(lr=0.0001),
			"activation_function": "elu",
			"Early_stop_percent": 0.01,
			}

sl_6j_1b = {"prenet_layer":	[100,100],
			"mainnet_layer":[100,100],
			"Dropout":		0.7,
			"L2_Norm":		0.00001,
			"batch_size":	5000,
			"optimizer":	optimizers.Adam(lr=0.0001),
			"activation_function": "elu",
			"Early_stop_percent": 0.01,
			}