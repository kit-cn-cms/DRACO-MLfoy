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
    inFile_path                 = workdir,
    classes                     = classes,
    category                    = example_cat,
    variables                   = categories[example_cat],
    intermediate_variables      = ["N_Jets", "N_BTagsM"], # placeholders
    test_percentage             = 0.3,
    train_percentage            = 0.5,
    norm_variables              = True)


