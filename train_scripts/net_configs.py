from tensorflow.keras import optimizers

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
        "layers":                   [200,200,1000,1000,200,200],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5,
        "L2_Norm":                  1e-5,
        "L1_Norm":                  1e-5,
        "batch_size":               1000,
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     50,
        }

config_dict["ttH_SL_legacy"] = {
        "layers":                   [50,50],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5,
        "L2_Norm":                  1e-5,
        "L1_Norm":                  1e-5,
        "batch_size":               1000,
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
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
        "batch_size":               1000,
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.1,
        "earlystopping_epochs":     50,
        }

config_dict["ttH_2017"] = {
        "layers":                   [100,100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "L1_Norm":                  0.,
        "L2_Norm":                  1e-5,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-4),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.05,
        "earlystopping_epochs":     100,
        }

config_dict["Legacy_ttH_2017"] = {
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
        "layers":                   [50,50],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.20,
        "L2_Norm":                  1e-5,
        "batch_size":               512,
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


config_dict["ttH_2017_DL"] = {
        "layers":                   [200,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.30,
        "L1_Norm":                  0.,
        "L2_Norm":                  1e-3,
        "batch_size":               64,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "relu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
        }

config_dict["binary_crossentropy_Adam"] = {
        "layers":                   [150],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.4,
        "L1_Norm":                  0.,
        "L2_Norm":                  1e-3,
        "batch_size":               2000,
        "optimizer":                optimizers.Adam(learning_rate=1e-3),
        "activation_function":      "relu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      150,
}

config_dict["ttbb_reco_v2"] = {
        "layers":                   [100,100],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.2,
        "L1_Norm":                  1e-3,
        "L2_Norm":                  1e-3,
        "batch_size":               128,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.1,
        "earlystopping_epochs":      100,
}

config_dict["ttbb_reco"] = {
        "layers":                   [100,100],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.2,
        "L1_Norm":                  1e-3,
        "L2_Norm":                  1e-3,
        "batch_size":               128,
        "optimizer":                optimizers.Adagrad(),
        "activation_function":      "leakyrelu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.1,
        "earlystopping_epochs":      100,
}

config_dict["binary_squared_Adadelta"] = {
        "layers":                   [200,100],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.3,
        "L1_Norm":                  0,
        "L2_Norm":                  0.,
        "batch_size":               4096,
        "optimizer":                optimizers.Adadelta(),
        "activation_function":      "elu",
        "output_activation":        "Tanh",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}

config_dict["binary_squared_SGD"] = {
        "layers":                   [100,100],
        "loss_function":            "squared_hinge",
        "Dropout":                  0.40,
        "L1_Norm":                  0,
        "L2_Norm":                  1e-5,
        "batch_size":               64,
        "optimizer":                optimizers.SGD(1e-3),
        "activation_function":      "tanh",
        "output_activation":        "Tanh",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
}

config_dict["adversary_multi"] = {
        "layers":                   [100,100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "L1_Norm":                  0,
        "L2_Norm":                  1e-5,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-4),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        "adversary_layers":         [100,100],
        "pretrain_class_epochs":    2,
        "pretrain_adv_epochs":      5,
        "adversary_epochs":         1,
        "adversary_iterations":     1,
}

config_dict["adversary_binary"] = {
        "layers":                   [200,100],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.30,
        "L1_Norm":                  1e-4,
        "L2_Norm":                  1e-3,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        "adversary_layers":         [100,100],
        "pretrain_class_epochs":    200,
        "pretrain_adv_epochs":      50,
        "adversary_epochs":         10,
        "adversary_iterations":     100,
}
config_dict["adversary_binary_test"] = {
        "layers":                   [200,100],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.30,
        "L1_Norm":                  1e-4,
        "L2_Norm":                  1e-3,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
        "adversary_layers":         [100,100],
        "pretrain_class_epochs":    20,
        "pretrain_adv_epochs":      5,
        "adversary_epochs":         1,
        "adversary_iterations":     1,
}

config_dict["BNN"] = {
        "layers":                   [50],
        #"loss_function":            "neg_log_likelihood",
        "Dropout":                  0,
        #"L1_Norm":                  0,
        #"L2_Norm":                  1e-5,
        "batch_size":               2000,
        "optimizer":                optimizers.Adam(learning_rate=1e-3),
        "activation_function":      "relu",
        "output_activation":        "sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}
