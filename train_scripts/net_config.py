from keras import optimizers
net_config = {
    "layers":                   [1000,1000,500,500,200,200],
    "loss_function":            "categorical_crossentropy",
    "Dropout":                  0.5,
    "L2_Norm":                  1e-4,
    "batch_size":               5000,
    "optimizer":                optimizers.Adagrad(decay=0.99),
    "activation_function":      "elu",
    "output_activation":        "Softmax",
    "earlystopping_percentage": 0.05,
    "batchNorm":                False,
    }

