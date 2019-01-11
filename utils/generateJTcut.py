def getJTstring(cat):
    # jtstring is '(ge)[NJets]j_(ge)[NTags]t'
    # output format is '(N_Jets (>=/==) [NJets] and N_BTagsM (>=/==) [NTags])'
    jtstring_config = "'(ge)[NJets]j_(ge)[NTags]t'"
    jets, tags = cat.split("_")

    string = "(N_Jets"

    # get jetstring
    if not jets.endswith("j"):
        print("JTstring has to be of format: "+jtstring_config)
        print("at the moment it is:          "+cat)
        exit()

    if jets.startswith("ge"): 
        njets  =  jets[2:-1]
        string += " >= "+njets
    else:
        njets  =  jets[:-1]
        string += " == "+njets

    string += " and N_BTagsM"

    # get tagstring
    if not tags.endswith("t"):
        print("JTstring has to be of format: "+jtstring_config)
        print("at the moment it is:          "+cat)
        exit()

    if tags.startswith("ge"):
        ntags  =  tags[2:-1]
        string += " >= "+ntags
    else:
        ntags  =  tags[:-1]
        string += " == "+ntags

    string += ")"

    #print("generated JTstring: '"+string+"'.")
    return string


def getJTlabel(cat):
    # jtstring is '(ge)[NJets]j_(ge)[NTags]t'
    # output format is '1 lepton, (\geq) 6 jets, (\geq) 3 b-tags'
    jtstring_config = "'(ge)[NJets]j_(ge)[NTags]t'"
    jets, tags = cat.split("_")

    string = "1 lepton,"

    # get jetstring
    if not jets.endswith("j"):
        print("JTstring has to be of format: "+jtstring_config)
        print("at the moment it is:          "+cat)
        exit()

    if jets.startswith("ge"):
        njets  =  jets[2:-1]
        string += " \geq"
    else:
        njets  =  jets[:-1]
    string += " "+njets+" jets,"

    # get tagstring
    if not tags.endswith("t"):
        print("JTstring has to be of format: "+jtstring_config)
        print("at the moment it is:          "+cat)
        exit()

    if tags.startswith("ge"):
        ntags  =  tags[2:-1]
        string += " \geq"
    else:
        ntags  =  tags[:-1]
    string += " "+ntags+" b-tags"

    #print("generated JTlabel: '"+string+"'.")
    return string
