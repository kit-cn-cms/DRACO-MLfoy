import optparse
import os
import sys

usage = "python generate_set.py [opts] [args]\nargs are jet-tag regions"
parser = optparse.OptionParser(usage = usage)
parser.add_option("-i", dest = "input",
    help = "path to trained DNNs. Needs to be given as /path/to/dnns/name_CATEGORY, where CATEGORY will be replaced by the given categories. \
if the input string does not contain the 'CATEGORY' string the category is just appended")
parser.add_option("-o", dest = "output",
    help = "path to output DNN set")
parser.add_option("--evenodd", dest = "evenodd", action = "store_true", default = False,
    help = "add even and odd DNNs")

(opts, args) = parser.parse_args()

if not os.path.exists(opts.output):
    os.makedirs(opts.output)
else:
    sys.exit("DNN set directory already exists")

mode = "replace"
if not "CATEGORY" in opts.input:
    mode = "append"
    print("'CATEGORY' not found in input string - path will be appended")

for jt in args:
    print("\nhandling region {}".format(jt))
    directory = opts.output+"/"+jt
    os.makedirs(directory)
    
    if mode == "replace":
        path = opts.input.replace("CATEGORY", jt)
    elif mode == "append":
        path = opts.input+jt

    if opts.evenodd:
        os.makedirs(directory+"/even")
        os.makedirs(directory+"/odd")

        cmd = "cp {}_even/checkpoints/* {}/even/".format(path, directory)
        print(cmd)
        os.system(cmd)

        cmd = "cp {}_odd/checkpoints/* {}/odd/".format(path, directory)
        print(cmd)
        os.system(cmd)

    else:
        cmd = "cp {}/checkpoints/* {}".format(path, directory)
        print(cmd)
        os.system(cmd)
