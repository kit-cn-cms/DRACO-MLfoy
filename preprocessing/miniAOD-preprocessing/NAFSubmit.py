import os
import numpy as np
import subprocess 
import stat


def writeShellScripts( workdir, inFiles, nameBase ):
    ''' write shell script to execute 'preprocessing_single_file.py'
        to process a single root file '''

    shellpath = workdir + "/shell_scripts"
    single_file_path = os.path.dirname(os.path.realpath(__file__)) + "/preprocessing_single_file.py"

    if not os.path.exists(shellpath):
        os.makedirs(shellpath)

    outpath = workdir + "/out_files"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    script_list = []
    for iF, inFile in enumerate(inFiles):
        name = nameBase+"_"+str(iF)
        
        script =  "#!/bin/bash\n"
        script += "export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch \n"
        script += "source $VO_CMS_SW_DIR/cmsset_default.sh \n"
        script += "export SCRAM_ARCH="+os.environ['SCRAM_ARCH']+"\n"
        script += "cd /nfs/dust/cms/user/vdlinden/CMSSW/CMSSW_9_2_4/src\n"
        script += "eval `scram runtime -sh`\n"
        script += "cd - \n"
        script += "python "+str(single_file_path)+" "+str(inFile)+" "+str(outpath)+"/"+str(name)+".h5\n"

        save_path = shellpath+"/"+str(name)+".sh"
        with open(save_path, "w") as f:
            f.write(script)

        st = os.stat(save_path)
        os.chmod(save_path, st.st_mode | stat.S_IEXEC)
        print("wrote shell script "+str(save_path))
        script_list.append( save_path )
 
    return script_list
       
def submitToBatch(workdir, list_of_shells ):
    ''' submit the list of shell script to the NAF batch system '''

    # write array script for submission
    arrayScript = writeArrayScript(workdir, list_of_shells)

    # write submit script for submission
    submitScript = writeSubmitScript(workdir, arrayScript, len(list_of_shells))
        
    # submit the whole thing
    condorSubmit( submitScript )


def writeArrayScript(workdir, files):
    shellpath = workdir + "/shell_scripts"
    path = shellpath+"/arraySubmit.sh"

    code = "#!/bin/bash\n"
    code+= "subtasklist=(\n"
    for f in files:
        code += f+"\n"
    code += ")\n"
    code += "thescript=${subtasklist[$SGE_TASK_ID]}\n"
    code += "echo \"${thescript}\"\n"
    code += "echo \"$SGE_TASK_ID\"\n"
    code += ". $thescript"

    with open(path, "w") as f:
        f.write(code)

    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)
    
    print("wrote array script "+str(path))
    return path


def writeSubmitScript(workdir, arrayScript, nScripts):
    shellpath = workdir + "/shell_scripts"
    path = shellpath+"/submitScript.sub"
    logdir = shellpath+"/logs"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    code = "universe = vanilla\n"
    code +="executable = /bin/zsh\n"
    code +="arguments = " + arrayScript + "\n"
    code +="request_memory = 10000M\n"
    code +="error = "+logdir+"/submitScript.$(Cluster)_$(ProcId).err\n"
    code +="log = "+logdir+"/submitScript.$(Cluster)_$(ProcId).log\n"
    code +="output = "+logdir+"/submitScript.$(Cluster)_$(ProcId).out\n"
    code +="Queue Environment From (\n"
    for taskID in range(nScripts):
        code += "\"SGE_TASK_ID="+str(taskID+1)+"\"\n"
    code += ")"

    with open(path, "w") as f:
        f.write(code)

    print("wrote submit script "+str(path))
    return path

def condorSubmit(submitPath):
    submitCommand = "condor_submit -name bird-htc-sched02.desy.de " + submitPath
    print("submitting:")
    print(submitCommand)
    process = subprocess.Popen(submitCommand.split(), stdout = subprocess.PIPE, stderr = subprocess.STDOUT, stdin = subprocess.PIPE)
    process.wait()
    output = process.communicate()
    print(output)






