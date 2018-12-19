from keras import optimizers

class Architecture():
    def __init__(self):
        self.sl_4j_ge3t = {
        "layers":                   [100,100],
        "loss_function":            "kullback_leibler_divergence",
        "Dropout":		    0.50,
        "L2_Norm":		    1e-5,
        "batch_size":	            4096,
        "optimizer":	            optimizers.Adam(1e-4),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.01,
        "batchNorm":                False,
        }

        self.sl_5j_ge3t = {
        "layers":                   [100,100],
        "loss_function":            "kullback_leibler_divergence",
        "Dropout":		    0.50,
        "L2_Norm":		    1e-5,
        "batch_size":	            4096,
        "optimizer":	            optimizers.Adam(1e-4),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.01,
        "batchNorm":                False,
        }

        self.sl_ge6j_ge3t = {
        "layers":                   [100,100],
        "loss_function":            "kullback_leibler_divergence",
        "Dropout":		    0.50,
        "L2_Norm":		    1e-5,
        "batch_size":	            4096,
        "optimizer":	            optimizers.Adam(1e-4),
        "activation_function":      "elu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 0.01,
        "batchNorm":                False,
        }

    def get_architecture(self, event_category):

        if event_category == "(N_Jets >= 6 and N_BTagsM >= 3)":
            return self.sl_ge6j_ge3t
        elif event_category == "(N_Jets == 5 and N_BTagsM >= 3)":
            return self.sl_5j_ge3t
        elif event_category == "(N_Jets == 4 and N_BTagsM >= 3)":
            return self.sl_4j_ge3t
