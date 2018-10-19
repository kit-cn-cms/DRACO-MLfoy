import miniAOD_preprocessing as preproc

# !!! change this !!!
basedir = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir"

sample_dict = {
    "ttHbb": basedir + "/out_files/ttHbb_*.h5",
    "ttZqq": basedir + "/out_files/ttZqq_*.h5"
    }

outFile = basedir + "/out_files/base_train_set"

preproc.prepare_training_sample(
    sample_dict = sample_dict, 
    outFile     = outFile,
    train_pc    = 0.7,
    val_pc      = 0.2
    )
