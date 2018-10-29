import miniAOD_preprocessing as preproc

# !!! change this !!!
basedir = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir"

sample_dict = {
    "ttHbb": basedir + "/out_files/ttHbb_pfc_*.h5",
    "ttZqq": basedir + "/out_files/ttZqq_pfc_*.h5"
    }

outFile = basedir + "/out_files/pfc_train_set"

preproc.prepare_training_sample(
    sample_dict = sample_dict, 
    outFile     = outFile,
    train_pc    = 0.83334,
    val_pc      = 0.083334
    )
