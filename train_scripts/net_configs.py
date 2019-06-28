from keras import optimizers
config_dict = {}

config_dict["example_config"] = {
        "layers":                   [200,200],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5,
        "L2_Norm":                  1e-5,
        "batch_size":               5000,
        "optimizer":                optimizers.Adagrad(decay=0.99),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }

config_dict["test_config"] = {
        "layers":                   [1000,1000,200,200],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.,
        "L2_Norm":                  0.,
        "batch_size":               5000,
        "optimizer":                optimizers.Adagrad(decay=0.99),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }

config_dict["ttH_2017"] = {
        "layers":                   [100,100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "L2_Norm":                  1e-5,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-4),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     100,
        }

config_dict["ttH_2017_baseline"] = {
        "layers":                   [100,100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "L2_Norm":                  1e-5,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-4),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.1,
        "earlystopping_epochs":     100,
        }

config_dict["legacy_2018"] = {
        "layers":                   [200,100,50],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.20,
        "L2_Norm":                  1e-5,
        "batch_size":               50000,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     100,
        }

config_dict["ttZ_2018"] = {
        "layers":                   [300,200,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.4,
        "L2_Norm":                  0.,
        "batch_size":               4096,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     100,
        }

config_dict["ttZ_2018_v2"] = {
        "layers":                   [300,200,100,50],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.4,
        "L2_Norm":                  1e-5,
        "batch_size":               5000,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }
config_dict["ttZ_2018_v3"] = {
        "layers":                   [200,100,50],
        "loss_function":            "mean_squared_error",
        "Dropout":                  0.3,
        "L2_Norm":                  1e-5,
        "batch_size":               5000,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }


config_dict["dnn_config"] = {
        "layers":                   [20],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.1,
        "L2_Norm":                  0.,
        "batch_size":               2000,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }


config_dict["binary_config"] = {
        "layers":                   [200,100],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.3,
        "L2_Norm":                  0.,
        "batch_size":               4000,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "selu",
        "output_activation":        "Tanh",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }
config_dict["binary_config_v2"] = {
        "layers":                   [200,100],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.3,
        "L2_Norm":                  1e-5,
        "batch_size":               4000,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "selu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }
