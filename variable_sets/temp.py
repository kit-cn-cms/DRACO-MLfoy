import importlib
ListOfFiles = ['allVars', 'allVars_noReco', 'RecoVarsOnly', 'allVars_S01', 'RecoVarsOnly_S01', 'MergeAll_S02', 'MergeAllExclBin_S02', 'MergeNoRecoReco_S02', 'MergeNoRecoRecoExclBin_S02', 'Fit_S01', 'topVariables_validated_decorr']

for i in ListOfFiles:
    print(20*"*")
    print(i+":")
    globals()[i] = importlib.import_module(i) 
    a = globals()[i].variables
    for cat in ["4j_ge3t","5j_ge3t","ge6j_ge3t","ge4j_3t","ge4j_ge4t"]:
        
        try: print(str(cat)+": "+str(len(a[cat])))
        except: 
             print("skipping "+cat)
      
    
