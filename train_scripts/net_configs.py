from keras import optimizers
config_dict = {}

config_dict["example_config_challenge"] = {
        "layers":                   [200,200],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.5,
        "L2_Norm":                  1e-5,
        "batch_size":               5000,
        "optimizer":                optimizers.Adam(),
        # "optimizer":                optimizers.Adagrad(decay=0.99),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }

config_dict["example_config_binary"] = {
        "layers":                   [500,500],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.3,
        "L2_Norm":                  1e-5,
        "batch_size":               500,
        # "optimizer":                optimizers.Adagrad(decay=0.99),
        "optimizer":                optimizers.Adam(),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }

