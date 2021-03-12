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

config_dict["ttH_SL_legacy_opt44"] = {
        "layers":                   [1024,2048,512,512],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5,
        "L2_Norm":                  1e-5,
        "L1_Norm":                  1e-5,
        "batch_size":               4096,
        # "optimizer":                optimizers.Adam(lr = 0.0006),
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }

config_dict["ttH_SL_legacy_STXS_opt44"] = {
        "layers":                   [2048,2048,64],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5,
        "L2_Norm":                  1e-5,
        "L1_Norm":                  1e-5,
        "batch_size":               1024,
        # "optimizer":                optimizers.Adam(lr = 0.0006),
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }

config_dict["ttH_SL_legacy_opt43"] = {
        "layers":                   [2048,182,1024,64],
        "loss_function":            "categorical_crossentropy",
        # "Dropout":                  0.05,
        "Dropout":                  0.5,
        "L2_Norm":                  1e-5,
        "L1_Norm":                  1e-5,
        "batch_size":               2048,
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }

config_dict["ttH_SL_legacy_STXS_opt43"] = {
        "layers":                   [2048,2048,512,1024],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5,
        "L2_Norm":                  1e-5,
        "L1_Norm":                  1e-5,
        "batch_size":               1024,
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }

config_dict["ttZ_2018_final"] = {
        "layers":                   [50,50],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5,
        "L2_Norm":                  1e-5,
        "L1_Norm":                  1e-5,
        "batch_size":               1024,
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.1,
        "earlystopping_epochs":     50,
        }

config_dict["ttX_test_2020"] = {
        "layers":                   [300,300,300,300],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.2,
        "L2_Norm":                  0,
        "L1_Norm":                  0,
        "batch_size":               128,
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.1,
        "earlystopping_epochs":     50,
        }

config_dict["ttX_2020"] = {
        "layers":                   [200,200,200],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.1,
        "L2_Norm":                  0,
        "L1_Norm":                  0,
        "batch_size":               128,
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

