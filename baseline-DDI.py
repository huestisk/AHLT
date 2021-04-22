import os
import sys
from xml.dom.minidom import parse

#from eval.evaluator import evaluate
from helper_functions_DDI import analyze, check_interaction

# parse arguments
datadir = sys.argv[1]
outfile = sys.argv[2]

# delete old file
if os.path.exists(outfile):
    os.remove(outfile)

# process each file in directory
for f in os.listdir(datadir):
    # parse XML file , obtaining a DOM tree
    tree = parse(datadir + "/" + f)
    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value  # get sentence id
        stext = s.attributes["text"].value  # get sentence text
        # load sentence entities into a dictionary
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents:
            eid = e.attributes["id"].value
            entities[eid] = e.attributes["charOffset"].value.split("-")
        # Tokenize, tag, and parse sentence
        analysis = analyze(stext)
        # for each pair in the sentence , decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            ddi_type = check_interaction(analysis, entities, id_e1, id_e2)
            if ddi_type is not None:
                with open(outfile, 'a') as f:
                    print(sid + "|" + id_e1 + "|" + id_e2 + "|" + ddi_type, file=f)
# get performance score
#evaluate("DDI", datadir, outfile)
