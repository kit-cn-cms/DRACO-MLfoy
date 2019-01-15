import os
import numpy as np
import subprocess 
import stat
import re
import time

def submitToBatch(workdir, list_of_shells ):
    ''' submit the list of shell script to the NAF batch system '''

    # write array script for submission
    arrayScript = writeArrayScript(workdir, list_of_shells)

    # write submit script for submission
    submitScript = writeSubmitScript(workdir, arrayScript, len(list_of_shells))
        
    # submit the whole thing
    jobID = condorSubmit( submitScript )
    return [jobID]

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
    #code +="request_memory = 10000M\n"
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
    submitCommand = "condor_submit -terse -name bird-htc-sched12.desy.de " + submitPath
    print("submitting:")
    print(submitCommand)
    tries = 0
    jobID = None
    while not jobID:
        process = subprocess.Popen(submitCommand.split(), stdout = subprocess.PIPE, stderr = subprocess.STDOUT, stdin = subprocess.PIPE)
        process.wait()
        output = process.communicate()
        try:
            jobID = int(output[0].split(".")[0])
        except:
            print("something went wrong with calling the condir_submit command, submission of jobs was not successful")
            print("DEBUG:")
            print(output)
            tries += 1
            jobID = None
            time.sleep(60)
        if tries>10:
            print("job submission was not successful after ten tries - exiting without JOBID")
            sys.exit(-1)
    return jobID




def monitorJobStatus(jobIDs = None):
    allfinished = False
    errorcount = 0
    print("checking job status in condor_q ...")

    command = ["condor_q", "-name", "bird-htc-sched02.desy.de"]
    if jobIDs:
        command += jobIDs
        command = [str(c) for c in command]
    command.append("-totals")

    while not allfinished:
        time.sleep(30)
        
        a = subprocess.Popen(command, stdout=subprocess.PIPE,stderr=subprocess.STDOUT,stdin=subprocess.PIPE)
        a.wait()
        qstat = a.communicate()[0]

        nrunning = -1
        queryline = [line for line in qstat.split("\n") if "Total for query" in line] 
        if len(queryline) == 1:
            jobsRunning = int(re.findall(r'\ [0-9]+\ running', queryline[0])[0][1:-8])
            jobsIdle = int(re.findall(r'\ [0-9]+\ idle', queryline[0])[0][1:-5])
            jobsHeld = int(re.findall(r'\ [0-9]+\ held', queryline[0])[0][1:-5])

            nrunning = jobsRunning + jobsIdle + jobsHeld

            print("{:4d} running | {:4d} idling | {:4d} held |\t total: {:4d}".format(jobsRunning, jobsIdle, jobsHeld, nrunning))

            errorcount = 0
            if nrunning == 0:
                print("waiting on no more jobs - exiting loop")
                allfinished=True
        else:
            errorcount += 1
            # sometimes condor_q is not reachable - if this happens a lot something is probably wrong
        
            print("line does not match query")
            if errorcount == 30:
                print("something is off - condor_q has not worked for 15 minutes ...")
                print("exiting condor_q (jobs are probably still in queue")
                sys.exit()


    print("all jobs are finished - exiting monitorJobStatus")
    return
