### import stuff



#####################################################################################



### some usefull functions

# function to append a list with sample, label and normalization_weight to a list samples
def createSampleList(sList, sample, label = None, nWeight = 1):
    """ takes a List a sample and appends a list with category, label and weight. Checks if even/odd splitting was made and therefore adjusts the normalization weight """
    if sample.even_odd: nWeight*=2.
    for cat in sample.categories.categories:
        if label==None:
            sList.append([cat, cat, nWeight])
        else:
            sList.append([cat, label, nWeight])
    return sList

# function to create a file with all preprocessed samples
def createSampleFile(outPath, sampleList):
    # create file
    processedSamples=""
    # write samplenames in file
    for sample in sampleList:
        processedSamples+=str(sample[0])+" "+str(sample[1])+" "+str(sample[2])+"\n"
    with open(outPath+"/sampleFile.dat","w") as sampleFile:
        sampleFile.write(processedSamples)

# function to read Samplefile
def readSampleFile(inPath):
    """ reads file and returns a list with all samples, that are not uncommented. Uncomment samples by adding a '#' in front of the line"""
    sampleList = []
    with open(outPath+"/sampleFile.dat","w") as sampleFile:
        for row in sampleFile:
            if row[0] != "#":
                sample = row.split()
                sampleList.append(dict( sample=sample[0], label=sample[1], normWeight=sample[2]) )
    return sampleList

# function to add samples to InputSamples
def addToInputSamples(inputSamples, samples, naming="_dnn.h5"):
    """ adds each sample in samples to the inputSample """
    for sample in samples:
        inputSamples.addSample(sample["sample"]+naming, label=sample["label"], normalization_weight = sample["normWeight"])
