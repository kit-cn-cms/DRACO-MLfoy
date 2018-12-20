from keras import optimizers

architecture = {}

architecture["4j_ge3t"] = {
    "layers":                   [100,100,100],
    "loss_function":            "categorical_crossentropy",
    "Dropout":                  0.50,
    "L2_Norm":                  1e-5,
    "batch_size":               4096,
    "optimizer":                optimizers.Adam(1e-4),
    "activation_function":      "elu",
    "output_activation":        "Softmax",
    "earlystopping_percentage": 0.02,
    "batchNorm":                False,
    }

architecture["5j_ge3t"] = {
    "layers":                   [100,100,100],
    "loss_function":            "categorical_crossentropy",
    "Dropout":                  0.50,
    "L2_Norm":                  1e-5,
    "batch_size":               4096,
    "optimizer":                optimizers.Adam(1e-4),
    "activation_function":      "elu",
    "output_activation":        "Softmax",
    "earlystopping_percentage": 0.02,
    "batchNorm":                False,
    }

architecture["ge6j_ge3t"] = {
    "layers":                   [100,100,100],
    "loss_function":            "categorical_crossentropy",
    "Dropout":                  0.50,
    "L2_Norm":                  1e-5,
    "batch_size":               4096,
    "optimizer":                optimizers.Adam(1e-4),
    "activation_function":      "elu",
    "output_activation":        "Softmax",
    "earlystopping_percentage": 0.02,
    "batchNorm":                False,
    }

def getArchitecture(cat):
    return architecture[cat]
