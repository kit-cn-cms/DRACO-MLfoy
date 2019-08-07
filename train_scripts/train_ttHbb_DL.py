# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import optparse

# option handler
import optionHandler
options = optionHandler.optionHandler(sys.argv)

# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for DNN training
import DRACO_Frameworks.DNN.DNN as DNN
import DRACO_Frameworks.DNN.data_frame as df

options.initArguments()

# load samples
input_samples = df.InputSamples(options.getInputDirectory(), options.getActivatedSamples(), options.getTestPercentage())

# during preprocessing half of the ttH sample is discarded (Even/Odd splitting),
# thus, the event yield has to be multiplied by two. This is done with normalization_weight = 2.
input_samples.addSample(options.getDefaultName("ttH"), label = "ttH", normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')

input_samples.addSample(options.getDefaultName("ttbb"), label = "ttbb" , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')
input_samples.addSample(options.getDefaultName("tt2b"), label = "tt2b" , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')
input_samples.addSample(options.getDefaultName("ttb"), label = "ttb"  , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')
input_samples.addSample(options.getDefaultName("ttcc"), label = "ttcc" , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')
input_samples.addSample(options.getDefaultName("ttlf"), label = "ttlf" , normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')

#!! input_samples.addSample("ttbar"+naming, label = "ttbar", normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')
#!! input_samples.addSample("ttbar"+naming, label = "background", normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')

#input_samples.addSample("ttZ"+naming,  label = "ttZ",  normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')
#input_samples.addSample("ST"+naming,   label = "ST",   normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')
#input_samples.addSample("tH"+naming,   label = "tH",   normalization_weight = options.getNomWeight(), total_weight_expr='x.weight')

if options.isBinary():
   input_samples.addBinaryLabel(options.getSignal(), options.getBinaryBkgTarget())

category_cutString_dict = {

    '3j_'+  '2t': '(N_jets == 3) & (N_btags == 2)',
    '3j_'+  '3t': '(N_jets == 3) & (N_btags == 3)',
  'ge3j_'+'ge3t': '(N_jets >= 3) & (N_btags >= 3)',
  'ge4j_'+  '2t': '(N_jets >= 4) & (N_btags == 2)',
  'ge4j_'+  '3t': '(N_jets >= 4) & (N_btags == 3)',
  'ge4j_'+'ge4t': '(N_jets >= 4) & (N_btags >= 4)',

  'ge4j_'+'ge3t': '(N_jets >= 4) & (N_btags >= 3)',
}

category_label_dict = {

    '3j_'+  '2t': 'N_jets = 3, N_btags = 2',
    '3j_'+  '3t': 'N_jets = 3, N_btags = 3',
  'ge3j_'+'ge3t': 'N_jets \\geq 3, N_btags \\geq 3',
  'ge4j_'+  '2t': 'N_jets \\geq 4, N_btags = 2',
  'ge4j_'+  '3t': 'N_jets \\geq 4, N_btags = 3',
  'ge4j_'+'ge4t': 'N_jets \\geq 4, N_btags \\geq 4',

  'ge4j_'+'ge3t': 'N_jets \\geq 4, N_btags \\geq 3',
}

# initializing DNN training class

dnn = DNN.DNN(

    save_path       =  options.getOutputDir(),
    input_samples   = input_samples,

    category_name   = options.getCategory(),

    category_cutString = category_cutString_dict[options.getCategory()],
    category_label     = category_label_dict[options.getCategory()],

    train_variables = options.getTrainVariables(),

    # number of epochs
    train_epochs    = options.getTrainEpochs(),
    # metrics for evaluation (c.f. KERAS metrics)
    eval_metrics    = ['acc'],
    # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
    test_percentage = options.getTestPercentage(),
    # balance samples per epoch such that there amount of samples per category is roughly equal
    balanceSamples  = options.doBalanceSamples(),
    evenSel         = options.doEvenSelection(),
    norm_variables  = options.doNormVariables()
)



# build DNN model
dnn.build_model(options.getNetConfig())

# perform the training
dnn.train_model()

# evalute the trained model
dnn.eval_model()

# save information
dnn.save_model(sys.argv, filedir)

# save configurations of variables for plotscript
#dnn.variables_configuration()

# save and print variable ranking according to the input layer weights
dnn.get_input_weights()

# save and print variable ranking according to all layer weights
dnn.get_weights()


# plotting
if options.doPlots():
    # plot the evaluation metrics
    dnn.plot_metrics(privateWork = options.isPrivateWork())

    if options.isBinary():
        # plot output node
        bin_range = options.getBinaryBinRange()
        dnn.plot_binaryOutput(
            log         = options.doLogPlots(),
            privateWork = options.isPrivateWork(),
            printROC    = options.doPrintROC(),
            bin_range   = bin_range,
            name        = options.getName())
    else:
        # plot the confusion matrix
        dnn.plot_confusionMatrix(
            privateWork = options.isPrivateWork(),
            printROC    = options.doPrintROC())

        # plot the output discriminators
        dnn.plot_discriminators(
            log                 = options.doLogPlots(),
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork(),
            printROC            = options.doPrintROC(),
            sigScale            = options.getSignalScale())

        # plot the output nodes
        dnn.plot_outputNodes(
            log                 = options.doLogPlots(),
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork(),
            printROC            = options.doPrintROC(),
            sigScale            = options.getSignalScale())

        # plot event yields
        dnn.plot_eventYields(
            log                 = options.doLogPlots(),
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork(),
            sigScale            = options.getSignalScale())

        # plot closure test
        dnn.plot_closureTest(
            log                 = options.doLogPlots(),
            signal_class        = options.getSignal(),
            privateWork         = options.isPrivateWork())
