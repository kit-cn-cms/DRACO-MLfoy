from tensorflow.keras import optimizers

config_dict = {}

config_dict["ttH_SL_legacy"] = {
        "layers":                   [100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.4,
        "L2_Norm":                  1e-4,
        "L1_Norm":                  1e-4,
        "batch_size":               500,
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.04,
        "earlystopping_epochs":     20,
        }

config_dict["flavorTag"] = {
        "layers":                   [100,100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5,
        "L2_Norm":                  1e-5,
        "L1_Norm":                  1e-5,
        "batch_size":               2048,
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.1,
        "earlystopping_epochs":     50,
        }

config_dict["ttZ_2018_final"] = {
        "layers":                   [50,50],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5,
        "L2_Norm":                  1e-5,
        "L1_Norm":                  1e-5,
        "batch_size":               1000,
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.1,
        "earlystopping_epochs":     50,
        }

config_dict["reco_single_boson"] = {
        "layers":                   [50,50],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.2,
        "L1_Norm":                  1e-4,
        "L2_Norm":                  1e-4,
        "batch_size":               16,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.05,
        "earlystopping_epochs":      20,
}

config_dict["dnn_ttZ"] = {
        "layers":                   [100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.2,
        "L1_Norm":                  1e-4,
        "L2_Norm":                  1e-4,
        "batch_size":               256,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage":  0.05,
        "earlystopping_epochs":      20,
}

config_dict["dnn_ttZ_binary"] = {
        "layers":                   [100,100],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.2,
        "L1_Norm":                  1e-4,
        "L2_Norm":                  1e-4,
        "batch_size":               256,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "selu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.05,
        "earlystopping_epochs":      20,
}



config_dict["ttZ_or_ttH"] = {
        "layers":                   [100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.2,
        "L1_Norm":                  1e-4,
        "L2_Norm":                  1e-4,
        "batch_size":               256,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage":  0.05,
        "earlystopping_epochs":      20,
}

