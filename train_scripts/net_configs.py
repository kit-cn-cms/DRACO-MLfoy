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
        "layers":                   [200,100],
        "loss_function":            "binary_crossentropy",
        "Dropout":                  0.30,
        "L1_Norm":                  1e-4,
        "L2_Norm":                  1e-3,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage":  0.02,
        "earlystopping_epochs":      100,
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
config_dict["jakob_test_5"] = {
        "layers":                   [5,5],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.30,
        "L2_Norm":                  1e-3,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}

config_dict["jakob_test_big"] = {
        "layers":                   [100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.30,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}

config_dict["jakob_test_low_dropout"] = {
        "layers":                   [100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.10,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}
config_dict["jakob_test_high_dropout"] = {
        "layers":                   [100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.70,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}
config_dict["jakob_test_relu"] = {
        "layers":                   [100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.30,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "relu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}
config_dict["jakob_test_tanh"] = {
        "layers":                   [100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.30,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "tanh",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}
config_dict["jakob_test_linear"] = {
        "layers":                   [100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.30,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "linear",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}
config_dict["jakob_test_sigmoid"] = {
        "layers":                   [100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.30,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "sigmoid",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}
config_dict["jakob_test_exponential"] = {
        "layers":                   [100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.30,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "exponential",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}
config_dict["jakob_test_l1_high"] = {
        "layers":                   [100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.30,
        "L1_Norm":                  1e-3,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}
config_dict["jakob_test_l2_high"] = {
        "layers":                   [100,100],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.30,
        "L2_Norm":                  1e-3,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}
config_dict["jakob_test_random_normal"] = {
        "layers":                   [100,100],
        "kernel_initializer":       'random_normal',
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}

config_dict["jakob_test_zeros"] = {
        "layers":                   [100,100],
        "kernel_initializer":      'zeros',
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}

config_dict["jakob_test_random_uniform"] = {
        "layers":                   [100,100],
        "kernel_initializer":       'random_uniform',
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}

config_dict["jakob_test_ones"] = {
        "layers":                   [100,100],
        "kernel_initializer":       'ones',
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}

config_dict["jakob_test_truncated_normal"] = {
        "layers":                   [100,100],
        "kernel_initializer":       'truncated_normal',
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}

config_dict["jakob_test_glorot_normal"] = {
        "layers":                   [100,100],
        "kernel_initializer":       'glorot_normal',
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}

config_dict["jakob_test_glorot_uniform"] = {
        "layers":                   [100,100],
        "kernel_initializer":       'glorot_uniform',
        "bias_initializer":         'zeros',
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.50,
        "batch_size":               4096,
        "optimizer":                optimizers.Adam(1e-3),
        "activation_function":      "elu",
        "output_activation":        "Sigmoid",
        "earlystopping_percentage": 0.02,
        "earlystopping_epochs":     100,
}
