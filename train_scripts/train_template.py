# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import numpy as np
import pprint

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

def power_of_two(a):
    return 2.0 ** a

options.initArguments()

# load samples
input_samples = df.InputSamples(options.getInputDirectory(), options.getActivatedSamples(), options.getTestPercentage(), options.getAddSampleSuffix())

weight_expr = 'x.Weight_XS * x.Weight_btagSF * x.Weight_GEN_nom * x.lumiWeight'
# define all samples
#input_samples.addSample(options.getDefaultName("ttH")  , label = "ttH"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttHbb")  , label = "ttHbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttHnonbb")  , label = "ttHnonbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttZ")  , label = "ttZ"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttZbb")  , label = "ttZbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttZnonbb")  , label = "ttZnonbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttX")  , label = "ttX"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttXbb")  , label = "ttXbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttXnonbb")  , label = "ttXnonbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )

#input_samples.addSample(options.getDefaultName("ttbar") , label = "ttbar" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttnonbb") , label = "ttnonbb" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttmb") , label = "ttmb" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttbb") , label = "ttbb" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttbb") , label = "ttbb" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("tt2b") , label = "tt2b" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttb")  , label = "ttb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttcc") , label = "ttcc" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttlf") , label = "ttlf" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )

#input_samples.addSample(options.getDefaultName("tHq") , label = "tHq" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("tHW") , label = "tHW" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )

###############################################################################################################################################################################
#  for JAN
###############################################################################################################################################################################
#input_samples.addSample(options.getDefaultName("sig") ,  label = "sig" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("Zbb") ,  label = "Zbb" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("Hbb") ,  label = "Hbb" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("bb") ,  label = "bb" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("cc") ,  label = "cc" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("ttTobb") , label = "ttTobb" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
input_samples.addSample(options.getDefaultName("bkg") ,  label = "bkg" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("bkg_Z") ,  label = "bkg_Z" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("bkg_Higgs") ,  label = "bkg_Higgs" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("bkg_bb") ,  label = "sig_cc" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("bkg_cc") ,  label = "bkg_cc" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )

################################################################################################################################################################################
# for Classification DNN
################################################################################################################################################################################
#input_samples.addSample(options.getDefaultName("ttH")  , label = "ttH"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttHbb")  , label = "ttHbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttHnonbb")  , label = "ttHnonbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttZ")  , label = "ttZ"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttZbb")  , label = "ttZbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttZnonbb")  , label = "ttZnonbb"  , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttbb") , label = "ttbb" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttcc") , label = "ttcc" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
#input_samples.addSample(options.getDefaultName("ttlf") , label = "ttlf" , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
################################################################################################################################################################################


# additional samples for adversary training
if options.isAdversary():
    input_samples.addSample(options.getAddSampleName("ttmb"), label = "ttmb"+options.getAddSampleSuffix(), normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
    input_samples.addSample(options.getAddSampleName("ttbb"), label = "ttbb"+options.getAddSampleSuffix(), normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
    input_samples.addSample(options.getAddSampleName("tt2b"), label = "tt2b"+options.getAddSampleSuffix(), normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )
    input_samples.addSample(options.getAddSampleName("ttb") , label = "ttb"+options.getAddSampleSuffix() , normalization_weight = options.getNomWeight(), total_weight_expr = weight_expr )

if options.isBinary():
    input_samples.addBinaryLabel(options.getSignal(), options.getBinaryBkgTarget())

if not options.isAdversary():
    # initializing DNN training class
    dnn = DNN.DNN(
        save_path       = options.getOutputDir(),
        input_samples   = input_samples,
        category_name   = options.getCategory(),
        train_variables = options.getTrainVariables(),
        # number of epochs
        train_epochs    = options.getTrainEpochs(),
        # metrics for evaluation (c.f. KERAS metrics)
        eval_metrics    = ["acc"],
        # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
        test_percentage = options.getTestPercentage(),
        # balance samples per epoch such that there amount of samples per category is roughly equal
        balanceSamples  = options.doBalanceSamples(),
        evenSel         = options.doEvenSelection(),
        norm_variables  = options.doNormVariables())
else:
    import DRACO_Frameworks.DNN.CAN as CAN
    # initializing CAN training class
    dnn = CAN.CAN(
        save_path       = options.getOutputDir(),
        input_samples   = input_samples,
        category_name   = options.getCategory(),
        train_variables = options.getTrainVariables(),
        # number of epochs
        train_epochs    = options.getTrainEpochs(),
        # metrics for evaluation (c.f. KERAS metrics)
        eval_metrics    = ["acc"],
        # percentage of train set to be used for testing (i.e. evaluating/plotting after training)
        test_percentage = options.getTestPercentage(),
        # balance samples per epoch such that there amount of samples per category is roughly equal
        balanceSamples  = options.doBalanceSamples(),
        evenSel         = options.doEvenSelection(),
        norm_variables  = options.doNormVariables(),
        addSampleSuffix = options.getAddSampleSuffix())


if not options.doHyperparametersOptimization():
    # build DNN model
    dnn.build_model(config=options.getNetConfig(), penalty=options.getPenalty())

    # perform the training
    dnn.train_model()

    # evalute the trained model
    dnn.eval_model()

    # save information
    dnn.save_model(sys.argv, filedir, options.getNetConfigName(), get_gradients = options.doGradients())

    # save and print variable ranking according to the input layer weights
    dnn.get_input_weights()

    # save and print variable ranking according to all layer weights
    dnn.get_weights()

    # variation plots
    if options.doVariations():
        dnn.get_variations(options.isBinary())

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
                nbins       = 15,
                bin_range   = bin_range,
                name        = options.getName(),
                sigScale    = options.getSignalScale())
            if options.isAdversary():
                dnn.plot_ttbbKS_binary(
                    log                 = options.doLogPlots(),
                    signal_class        = options.getSignal(),
                    privateWork         = options.isPrivateWork())
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

            # plot ttbb KS test
            if options.isAdversary():
                dnn.plot_ttbbKS(
                    log                 = options.doLogPlots(),
                    signal_class        = options.getSignal(),
                    privateWork         = options.isPrivateWork())

else:
    import hyperopt as hp
    from hyperopt import fmin, STATUS_OK, tpe, space_eval, Trials
    from hyperopt.pyll import scope
    from hyperas.distributions import choice, uniform, loguniform, quniform

    opt_search_space = choice('name',
                                [
                                    # {'name': 'adam',
                                    # 'learning_rate': loguniform('learning_rate_adam', -10, 0),
                                    #'beta_1': loguniform('beta_1_adam', -10, -1),# Note the name of the label to avoid duplicates
                                    #'beta_2': loguniform('beta_2_adam', -10, -1),
                                    # },
                                    # {'name': 'sgd',
                                    # 'learning_rate': loguniform('learning_rate_sgd', -10, 0), # Note the name of the label to avoid duplicates
                                    #'momentum': uniform('momentum_sgd', 0, 1.0),
                                    # },
                                    # {'name': 'Adagrad',
                                    # 'learning_rate': loguniform('learning_rate_adagrad', -10, 0), # Note the name of the label to avoid duplicates
                                    # },
                                    {'name': 'Adagrad',
                                    'learning_rate': loguniform('learning_rate_adagrad', -10, 0), # Note the name of the label to avoid duplicates
                                    },
                                    # {'name': 'Adadelta',
                                    # 'learning_rate': loguniform('learning_rate_adadelta', -10, 0), # Note the name of the label to avoid duplicates
                                    # },
                                    # {'name': 'Adamax',
                                    # 'learning_rate': loguniform('learning_rate_adamax', -10, 0), # Note the name of the label to avoid duplicates
                                    # 'beta_1': loguniform('beta_1_adamax', -10, 0),# Note the name of the label to avoid duplicates
                                    # 'beta_2': loguniform('beta_2_adamax', -10, 0),
                                    # }
                                    ])
    fourth_layer_search_space = choice('four_layer',
                                [
                                    {
                                    'include': False,
                                    },
                                    {
                                    'include': True,
                                    'layer_size_4': power_of_two(quniform('layer_size_4', 5, 10, q=1)),
                                    }

                                ])
    # @scope.define
    # def power_of_two(a):
    #         return 2.0 ** a

    search_space = {
        'layer_size_1'        : power_of_two(quniform('layer_size_1', 5, 11, q=1)),
        'layer_size_2'        : power_of_two(quniform('layer_size_2', 5, 11, q=1)),
        'layer_size_3'        : power_of_two(quniform('layer_size_3', 5, 11, q=1)),
        # 'layer_size_1'        : 32,
        # 'layer_size_2'        : 32,
        # 'layer_size_3'        : 32,
        'four_layer'          : fourth_layer_search_space,
        # 'dropout'             : uniform('dropout', 0, 1),
        'dropout'             : 0.5,
        'batch_size'          : power_of_two(quniform('batch_size', 10, 12, q=1)),
        # 'batch_size'          : 10000,
        # 'optimizer'           : opt_search_space,
        'optimizer'           : opt_search_space,
        # 'l2_regularizer'      : loguniform('l2_regularizer', -10,-1)
        'l2_regularizer'      : 1E-5
    }

    trials = Trials()
    best = fmin(dnn.hyperopt_fcn, search_space, algo=tpe.suggest, max_evals=25, trials=trials)
    # best = fmin(dnn.hyperopt_fcn, search_space, algo=tpe.suggest, max_evals=50, trials=trials)
    params = space_eval(search_space, best)
    f = open(options.getOutputDir()+"/hyperparamsOptimizations.txt","w")
    f.write( str(params) )
    f.close()
    pprint.pprint(params)

    if options.doPlots():
        import matplotlib.pyplot as plt
        plt.figure()
        xs = [t['tid'] for t in trials.trials]
        ys = [-t['result']['loss'] for t in trials.trials]
        plt.xlim(xs[0]-1, xs[-1]+1)
        plt.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
        plt.xlabel('Iteration', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.savefig(options.getOutputDir()+"/Accuracy.png", bbox_inches='tight')
        plt.savefig(options.getOutputDir()+"/Accuracy.pdf", bbox_inches='tight')
        plt.close()
        # Some additional visualization
        parameters = search_space.keys()
        cmap = plt.cm.Dark2

        # for t in trials.trials:
            # pprint.pprint(t)
        pprint.pprint(trials.trials[0])
        # pprint.pprint(trials.trials[0]['misc']['vals'])
        for i, name in enumerate(trials.trials[0]['misc']['vals']):
            plt.figure()
            x = []
            for t in trials.trials:
                if np.array(t['misc']['vals'][name]).ravel().size == 0:
                    continue
                x+=t['misc']['vals'][name]
            xs = np.array(x).ravel()
            # xs = np.array([t['misc']['vals'][name] for t in trials.trials if size(np.array(t['misc']['vals'][name])).ravel() != 0 ).ravel() 
            print("-"*20)
            print(name)
            print(xs)
            if xs.size == 0: continue
            ys = [-t['result']['loss'] for t in trials.trials ]
            zs = [t['tid'] for t in trials.trials]
            xs, ys, zs = zip(*sorted(zip(xs, ys, zs)))
            ys = np.array(ys)
            plt.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75, c=cmap(float(i)/len(parameters)))
            plt.savefig(options.getOutputDir()+"/"+name+"_vs_Accuracy.png", bbox_inches='tight')
            plt.savefig(options.getOutputDir()+"/"+name+"_vs_Accuracy.pdf", bbox_inches='tight')
            plt.close()

            plt.figure()
            plt.scatter(xs, zs, s=20, linewidth=0.01, alpha=0.75, c=cmap(float(i)/len(parameters)))
            plt.savefig(options.getOutputDir()+"/"+name+"_vs_Iteration.png", bbox_inches='tight')
            plt.savefig(options.getOutputDir()+"/"+name+"_vs_Iteration.pdf", bbox_inches='tight')
            plt.close()

if options.doGradients():
    dnn.get_gradients(options.isBinary())
