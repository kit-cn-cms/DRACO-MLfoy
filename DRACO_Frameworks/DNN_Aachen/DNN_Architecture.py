from keras import optimizers

class Architecture():
    def __init__(self):
        self.sl_4j_3b = {
            "prenet_layer":             [100,100],
            "prenet_loss":              'categorical_crossentropy',
            "mainnet_layer":            [100,100],
            "mainnet_loss":             "kullback_leibler_divergence",
            "Dropout":                  0.30,
            "L2_Norm":                  1e-5,
            "batch_size":               5000,
            "optimizer":                optimizers.Adam(1e-4),
            "activation_function":      "elu",
            "earlystopping_percentage": 0.01,
            "batchNorm":                False,
            }

        self.sl_5j_3b = {
            "prenet_layer":             [100,100],
            "prenet_loss":              'categorical_crossentropy',
            "mainnet_layer":            [100,100],
            "mainnet_loss":             "kullback_leibler_divergence",
            "Dropout":                  0.30,
            "L2_Norm":                  1e-5,
            "batch_size":               5000,
            "optimizer":                optimizers.Adam(1e-4),
            "activation_function":      "elu",
            "earlystopping_percentage": 0.01,
            "batchNorm":                False,
            }

        self.sl_6j_3b = {
            "prenet_layer":             [100,100],
            "prenet_loss":              'categorical_crossentropy',
            "mainnet_layer":            [100,100],
            "mainnet_loss":             "kullback_leibler_divergence",
            "Dropout":                  0.30,
            "L2_Norm":                  1e-5,
            "batch_size":               5000,
            "optimizer":                optimizers.Adam(1e-4),
            "activation_function":      "elu",
            "earlystopping_percentage": 0.01,
            "batchNorm":                False,
            }

    def get_architecture(self, event_category):

        if event_category == "(N_Jets >= 6 and N_BTagsM >= 3)":
            return self.sl_6j_3b
        elif event_category == "(N_Jets == 5 and N_BTagsM >= 3)":
            return self.sl_5j_3b
        elif event_category == "(N_Jets == 4 and N_BTagsM >= 3)":
            return self.sl_4j_3b
        elif event_category == "(N_Jets == 4 and N_BTagsM == 4)":
            return self.sl_4j_3b
