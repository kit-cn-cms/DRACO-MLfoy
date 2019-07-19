def getJTstring(cat):
    # jtstring is '(ge)[NJets]j_(ge)[NTags]t'
    # output format is '(N_Jets (>=/==) [NJets] and N_BTagsM (>=/==) [NTags])'

    string_parts = cat.split("_")
    
    cutstring = "("
    for part in string_parts:
        if part.endswith("l"):
            cutstring += "N_LooseLeptons"
        elif part.endswith("j"):
            cutstring += "N_Jets"
        elif part.endswith("t"):
            cutstring += "N_BTagsM"
        else:
            print("invalid format of category substring '{}' - IGNORING".format(part))
            continue

        if part.startswith("ge"):
            cutstring += " >= "+part[2:-1]
        elif part.startswith("le"):
            cutstring += " <= "+part[2:-1]
        else:
            cutstring += " == "+part[:-1]
        
        if not part == string_parts[-1]:
            cutstring += " and "
    
    cutstring += ")"

    return cutstring


def getJTlabel(cat):
    # jtstring is '(ge)[NJets]j_(ge)[NTags]t'
    # output format is '1 lepton, (\geq) 6 jets, (\geq) 3 b-tags'

    # special labels:
    if cat == "inclusive":  return "inclusive"
    if cat == "SL":         return "semileptonic t#bar{t}"

    string_parts = cat.split("_")

    cutstring = ""
    for part in string_parts:
        partstring = ""
        if part.startswith("ge"):
            n = part[2:-1]
            partstring += "\geq "
        elif part.startswith("le"):
            n = part[2:-1]
            partstring += "\leq "
        else:
            n = part[:-1]
            partstring += ""
        partstring += n


        if part.endswith("l"):
            partstring += " lepton"
        elif part.endswith("j"):
            partstring += " jet"
        elif part.endswith("t"):
            partstring += " b-tag"
        else:
            print("invalid format of category substring '{}' - IGNORING".format(part))
            continue

        # plural
        if int(n)>1: partstring += "s"

        if not part == string_parts[-1]:
            partstring += ", "
        cutstring += partstring

    return cutstring
