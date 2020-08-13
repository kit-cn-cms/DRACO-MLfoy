import optparse

usage = ""
parser = optparse.OptionParser(usage=usage)

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="../data/CSV_channel/basic_CSV/",
        help="DIR for input", metavar="inputDir")
parser.add_option("-f", "--filname", dest="fileName",default="_CSV",
        help="STR of ending of file to be visualized", metavar="fileName")
parser.add_option("-t", "--title", dest="title",default="",
        help="STR of title of figure", metavar="title")
parser.add_option("-s", "--secondChannel", dest="secondChannel",default="Jet_CSV",
        help="STR of second channel to use (supported: Jet_CSV and TaggedJet_Pt", metavar="secondChannel")


class optionHandler_vis:
    def __init__(self, argv):
        (options, args) = parser.parse_args(argv[1:])
        self.__options  = options
        self.__args     = args

    def getInputDir(self):
        return self.__options.inputDir

    def getFileName(self):
        return self.__options.fileName

    def getTitle(self):
        return self.__options.title

    def getSecondChannel(self):
        return self.__options.secondChannel
