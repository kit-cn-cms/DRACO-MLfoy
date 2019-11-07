# global imports
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import os
import sys
import json


# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

# import class for DNN training
import DRACO_Frameworks.DNN.DNN as DNN 

dnn1 = DNN.loadDNN(inputDirectory="/nfs/dust/cms/user/lreuter/hiwi/robusttests/workdir/RT_2_ge4j_ge4t_odd/",
		outputDirectory="/nfs/dust/cms/user/lreuter/hiwi/robusttests/workdir/",
		inputData="/nfs/dust/cms/user/lreuter/hiwi/DRACO-MLfoy/workdir/Robusttest/",
		data_era="2016")
# plot the output discriminators
dnn1.get_discriminators(signal_class="ttH")

dnn2 = DNN.loadDNN(inputDirectory="/nfs/dust/cms/user/lreuter/hiwi/robusttests/workdir/RT_3_ge4j_ge4t_odd/",
		outputDirectory="/nfs/dust/cms/user/lreuter/hiwi/robusttests/workdir/",
		inputData="/nfs/dust/cms/user/lreuter/hiwi/DRACO-MLfoy/workdir/Robusttest/",
		data_era="2016")
# plot the output discriminators
dnn2.get_discriminators(signal_class="ttH")

dnn3 = DNN.loadDNN(inputDirectory="/nfs/dust/cms/user/lreuter/hiwi/robusttests/workdir/RT_4_ge4j_ge4t_odd/",
		outputDirectory="/nfs/dust/cms/user/lreuter/hiwi/robusttests/workdir/",
		inputData="/nfs/dust/cms/user/lreuter/hiwi/DRACO-MLfoy/workdir/Robusttest/",
		data_era="2016")
# plot the output discriminators
dnn3.get_discriminators(signal_class="ttH")

dnn4 = DNN.loadDNN(inputDirectory="/nfs/dust/cms/user/lreuter/hiwi/robusttests/workdir/RT_5_ge4j_ge4t_odd/",
		outputDirectory="/nfs/dust/cms/user/lreuter/hiwi/robusttests/workdir/",
		inputData="/nfs/dust/cms/user/lreuter/hiwi/DRACO-MLfoy/workdir/Robusttest/",
		data_era="2016")
# plot the output discriminators
dnn4.get_discriminators(signal_class="ttH")


dnn5 = DNN.loadDNN(inputDirectory="/nfs/dust/cms/user/lreuter/hiwi/robusttests/workdir/RT_1_ge4j_ge4t_odd/",
		outputDirectory="/nfs/dust/cms/user/lreuter/hiwi/robusttests/workdir/",
		inputData="/nfs/dust/cms/user/lreuter/hiwi/DRACO-MLfoy/workdir/Robusttest/",
		data_era="2016")
# plot the output discriminators
dnn5.get_discriminators(signal_class="ttH")

RMSscores = {}
print dnn1.Histograms
for node in dnn1.Histograms:
	print node
	r=0
	v=0
	RMSscores[node]={}
	for i,hist in enumerate(dnn1.Histograms[node]["bkgHists"]):
		histname=hist.GetName()
		print histname
		RMSscores[node][histname]={}
		# RMSscores[node][histname]["RMSperbin"]=[]
		# RMSscores[node][histname]["varianceperbin"]=[]
		# RMSscores[node][histname]["standardDerivationperbin"]=[]
		RMS=0
		Variance=0
		standardDerivation=0
	
	# 	RMS=0
		for n in range(dnn1.Histograms[node]["bkgHists"][i].GetNbinsX()):

			dnnbincontent = []
			dnnbincontent.append(dnn1.Histograms[node]["bkgHists"][i].GetBinContent(n))
			dnnbincontent.append(dnn2.Histograms[node]["bkgHists"][i].GetBinContent(n))
			dnnbincontent.append(dnn3.Histograms[node]["bkgHists"][i].GetBinContent(n))
			dnnbincontent.append(dnn4.Histograms[node]["bkgHists"][i].GetBinContent(n))
			dnnbincontent.append(dnn5.Histograms[node]["bkgHists"][i].GetBinContent(n))

			#get RMS
			temp=0
			meancontent=0
			for bincontent in dnnbincontent:
				temp+=bincontent*bincontent
				meancontent+=bincontent
			temp = temp*0.2
			temp = temp**(1.0/2)
			#RMSscores[node][histname]["RMSperbin"].append(temp)
			if temp>0:
				r+=1
			RMS+= temp

			#get variance
			meancontent=meancontent/5.0
			temp=0

			for bincontent in dnnbincontent:
				temp+=(bincontent-meancontent)*(bincontent-meancontent)
			#RMSscores[node][histname]["varianceperbin"].append(temp)
			if temp>0:
				v+=1
			Variance+=temp

			#get standard derivation
			temp=temp**(1.0/2)
			#RMSscores[node][histname]["standardDerivationperbin"].append(temp)
			standardDerivation+=temp
		print RMS
		RMSscores[node][histname]["RMSall"]=RMS
		RMSscores[node][histname]["RMS"]=RMS/r
		print Variance
		RMSscores[node][histname]["Variance"]=Variance/v
		RMSscores[node][histname]["Varianceall"]=Variance
		print standardDerivation
		RMSscores[node][histname]["standardDerivationall"]=standardDerivation
		RMSscores[node][histname]["standardDerivation"]=standardDerivation/v



	RMSscores[node]["ttH"]={}
	# RMSscores[node]["ttH"]["RMSperbin"]=[]
	# RMSscores[node]["ttH"]["varianceperbin"]=[]
	# RMSscores[node]["ttH"]["standardDerivationperbin"]=[]
	RMS=0
	Variance=0
	standardDerivation=0
	r=0
	v=0
	for n in range(dnn1.Histograms[node]["sigHists"].GetNbinsX()):

		dnnbincontent = []
		dnnbincontent.append(dnn1.Histograms[node]["sigHists"].GetBinContent(n))
		dnnbincontent.append(dnn2.Histograms[node]["sigHists"].GetBinContent(n))
		dnnbincontent.append(dnn3.Histograms[node]["sigHists"].GetBinContent(n))
		dnnbincontent.append(dnn4.Histograms[node]["sigHists"].GetBinContent(n))
		dnnbincontent.append(dnn5.Histograms[node]["sigHists"].GetBinContent(n))

		#get RMS
		temp=0
		meancontent=0
		for bincontent in dnnbincontent:
			temp+=bincontent*bincontent
			meancontent+=bincontent
		temp = temp*0.2
		temp = temp**(1.0/2)
		#RMSscores[node]["ttH"]["RMSperbin"].append(temp)
		if temp>0:
			r+=1
		RMS+= temp

		#get variance
		meancontent=meancontent/5.0
		temp=0

		for bincontent in dnnbincontent:
			temp+=(bincontent-meancontent)*(bincontent-meancontent)
		#RMSscores[node]["ttH"]["varianceperbin"].append(temp)
		if temp>0:
			v+=1
		Variance+=temp

		#get standard derivation
		temp=temp**(1.0/2)
		#RMSscores[node]["ttH"]["standardDerivationperbin"].append(temp)
		standardDerivation+=temp




	print RMS
	RMSscores[node]["ttH"]["RMSall"]=RMS
	RMSscores[node]["ttH"]["RMS"]=RMS/r
	print Variance
	RMSscores[node]["ttH"]["Varianceall"]=Variance
	RMSscores[node]["ttH"]["Variance"]=Variance/v
	print standardDerivation
	RMSscores[node]["ttH"]["standardDerivationall"]=standardDerivation
	RMSscores[node]["ttH"]["standardDerivation"]=standardDerivation/v

print RMSscores

with open('DNNcombined.json','w') as file:
	json.dump(RMSscores,file,indent = 2, separators = (",", ": "))