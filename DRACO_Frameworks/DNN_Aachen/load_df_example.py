from data_frame import DataFrame as data_frame
import variable_info

workdir = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/AachenDNN_files"
classes = ["ttHbb", "ttbb", "ttb", "tt2b", "ttcc", "ttlf"]


categories = {
    "(N_Jets == 6 and N_BTagsM >= 3)": variable_info.variables_4j_3b,
    "(N_Jets == 5 and N_BTagsM >= 3)": variable_info.variables_5j_3b,
    "(N_Jets == 4 and N_BTagsM >= 3)": variable_info.variables_6j_3b,
} 

example_cat = "(N_Jets == 6 and N_BTagsM >= 3)"

data_set = data_frame(
    path_to_input_files         = workdir,
    classes                     = classes,
    event_category              = example_cat,
    train_variables             = categories[example_cat],
    prenet_targets              = ["N_Jets", "N_BTagsM"], # placeholders
    test_percentage             = 0.1,
    norm_variables              = False)

data_set.hist_train_variables(signal_hists = ["ttHbb"])


