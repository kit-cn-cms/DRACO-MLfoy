import numpy as np
import pandas as pd
import uproot as root
import re
import glob
import os

import variable_info


categories = {
    "(N_Jets == 6 and N_BTagsM >= 3)": variable_info.variables_4j_3b,
    "(N_Jets == 5 and N_BTagsM >= 3)": variable_info.variables_5j_3b,
    "(N_Jets == 4 and N_BTagsM >= 3)": variable_info.variables_6j_3b,
    }      


# handle vector variables:
def get_vars_and_vecvars(varlist):
    # list for variables
    variables = []
    # dictionary for vector variables
    vector_variables = {}

    # loop over variables in list
    for var in varlist:

        # search for index in name (dummyvar[index])
        found_vector_variable = re.search("\[\d+?\]$", var)
        # append variable to list if not a vector variable
        if not found_vector_variable:
            variables.append(var)
        # add index information to dictionary if it is a vector variable
        else:
            index = found_vector_variable.group(0)
            var_name = var[:-len(index)]
            var_index = int(index[1:-1])
            if var_name in vector_variables:
                vector_variables[var_name].append( var_index )
            else:
                vector_variables[var_name] = [var_index]
            
    return variables, vector_variables

# TODO:
# add weights
# add train targets for pre net




def create_datasets(df, class_label, path):
    ''' write dataset with events of given classlabel to workdir '''
    cut_df = df.query("(class_label == '"+str(class_label)+"')")
    print("creating dataset for class label {} with {} entries".format(class_label, cut_df.shape[0]))

    out_path = path + "/" + str(class_label)+".h5"
    with pd.HDFStore(out_path, "a") as store:
        store.append("data", cut_df, index = False)

def apply_cut(df, condition, drop_variables = None):
    ''' apply cut to data
        optionally drop variables from dataframe if it is not needed anymore '''
    cut_df = df.query(condition)
    # drop cut variables if not further specified
    if drop_variables:
        cut_df.drop(drop_variables, axis = 1, inplace = True)
    return cut_df

def add_class_labels(df, is_ttH = False):
    ''' add column of class labels for higgs and ttbar events '''
    if is_ttH:
        df["class_label"] = pd.Series( ["ttHbb"]*df.shape[0], index = df.index )
    else:
        # split ttbar processes
        ttbb_df = df.query("(GenEvt_I_TTPlusBB == 3 and GenEvt_I_TTPlusCC == 0)")
        ttbb_df["class_label"] = pd.Series( ["ttbb"]*ttbb_df.shape[0], index = ttbb_df.index )

        tt2b_df = df.query("(GenEvt_I_TTPlusBB == 2 and GenEvt_I_TTPlusCC == 0)")
        tt2b_df["class_label"] = pd.Series( ["tt2b"]*tt2b_df.shape[0], index = tt2b_df.index )

        ttb_df = df.query("(GenEvt_I_TTPlusBB == 1 and GenEvt_I_TTPlusCC == 0)")
        ttb_df["class_label"] = pd.Series( ["ttb"]*ttb_df.shape[0], index = ttb_df.index )

        ttcc_df = df.query("(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 1)")
        ttcc_df["class_label"] = pd.Series( ["ttcc"]*ttcc_df.shape[0], index = ttcc_df.index )

        ttlf_df = df.query("(GenEvt_I_TTPlusBB == 0 and GenEvt_I_TTPlusCC == 0)")
        ttlf_df["class_label"] = pd.Series( ["ttlf"]*ttlf_df.shape[0], index = ttlf_df.index )

        # concatenate them
        df = pd.concat( [ttbb_df, tt2b_df, ttb_df, ttcc_df, ttlf_df] )
    
    df.drop(["GenEvt_I_TTPlusBB", "GenEvt_I_TTPlusCC"], axis = 1, inplace = True)
    return df

def process_files(files, vars, vecvars, sel, workdir, is_ttH = False):
    ''' helper to process a list of files '''
    # loop over files
    for f in files:
        print("-"*50)
        print("loading file "+str(f))
        # open root file
        with root.open(f) as rf:
            # get MVATree
            tree = rf["MVATree"]
    
            # convert tree to dataframe but only extract the variables needed
            df = tree.pandas.df(vars)

            # handle vector variables, loop over them
            for vecvar in vecvars:
                # load dataframe with vector variable
                vec_df = tree.pandas.df(vecvar)

                # loop over indices in vecvar list
                for idx in vecvars[vecvar]:
                    # slice the index
                    idx_df = vec_df.loc[ (slice(None), slice(idx,idx)), :]
                    # define name for column in df
                    col_name = str(vecvar)+"["+str(idx)+"]"
                    # append column to original dataframe
                    df[col_name] = pd.Series( idx_df[vecvar].values, index = df.index )

            # apply event selection
            df = apply_cut(df, sel, drop_variables = ["Evt_Odd"])
            # add class labels
            df = add_class_labels(df, is_ttH = is_ttH)

            # write data to file
            if is_ttH:
                create_datasets(df, class_label = "ttHbb", path = workdir)
            else:
                create_datasets(df, class_label = "ttbb", path = workdir)
                create_datasets(df, class_label = "tt2b", path = workdir)
                create_datasets(df, class_label = "ttb",  path = workdir)
                create_datasets(df, class_label = "ttcc", path = workdir)
                create_datasets(df, class_label = "ttlf", path = workdir)

                

def preprocess_data(ttH_location, ttbar_location, base_selection, workdir):
    # remove old workdir
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    h5_files_existing = glob.glob(workdir+"/*.h5")
    for f in h5_files_existing:
        os.remove(f)


    # glob input files
    ttH_files = glob.glob(ttH_location)
    ttbar_files = glob.glob(ttbar_location)

    
    # import variable infos
    variable_list = variable_info.all_variables_list
    # get variables and dictionary of vector variables
    variables, vecvars = get_vars_and_vecvars(variable_list)

    # add some more variables needed -------------------------
    # append variables for category cutting to varlist
    variables += ["N_Jets", "N_BTagsM"]

    # append variables for class labels
    variables += ['GenEvt_I_TTPlusBB', 'GenEvt_I_TTPlusCC']

    # append variables for prenet-targets
    variables += [
        "GenTopHad_B_inacceptance_part",
        "GenTopHad_Q_inacceptance_part",
        "GenTopHad_QQ_inacceptance_part",
        "GenTopLep_B_inacceptance_part",
        "GenHiggs_B_inacceptance_part",
        "GenHiggs_BB_inacceptance_part",
        "GenAdd_B_inacceptance_part",
        "GenAdd_BB_inacceptance_part"]

    # append variable for train/test splitting
    variables += ["Evt_Odd"]

    # variables for weighting events
    variables += ["Weight_XS"]

    # append variables for MEM matching
    variabels += ["evt_id", "run", "lumi"]
    # --------------------------------------------------------

    process_files( ttH_files,   variables, vecvars, base_selection, workdir, is_ttH = True)
    process_files( ttbar_files, variables, vecvars, base_selection, workdir, is_ttH = False)



# location of ttH and ttbar samples
ttH = "/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/*nominal*.root"
ttbar = "/nfs/dust/cms/user/kelmorab/ttH_2018/ntuples/TTToSemiLeptonic_TuneCP5_PSweights_13TeV-powheg-pythia8/*nominal*.root"

workdir = "/nfs/dust/cms/user/vdlinden/DRACO-MLfoy/workdir/AachenDNN_files/"

# base event selection to drop unneccesary events immediately
base_event_selection = "(N_Jets >= 4 and N_BTagsM >= 3 and Evt_Odd == 1)"

preprocess_data(ttH, ttbar, base_event_selection, workdir)
