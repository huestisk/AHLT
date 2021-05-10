import os
import sys
from xml.dom.minidom import parse

sys.path.append('common/')
from helper_functions_DDI import getInfo, check_interaction

# parse arguments
datadir = sys.argv[1]
outfile = sys.argv[2]

# delete old file
if os.path.exists(outfile):
    os.remove(outfile)

### Get dependency tress
info = getInfo(datadir)

# process each file in directory
for f in os.listdir(datadir):
    # parse XML file , obtaining a DOM tree
    tree = parse(datadir + "/" + f)
    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        # we're only considering pairs
        pairs = s.getElementsByTagName("pair")
        if len(pairs) > 0:
            # get sentence id
            sid = s.attributes["id"].value
            stext = s.attributes["text"].value
            # get dependency tree
            analysis = info[sid].getDependencyGraph()
            # load sentence entities into a dictionary
            entities = {}
            ents = s.getElementsByTagName("entity")
            for e in ents:
                eid = e.attributes["id"].value
                entities[eid] = e.attributes["charOffset"].value.split("-")
            # get DDI
            for p in pairs:
                id_e1 = p.attributes["e1"].value
                id_e2 = p.attributes["e2"].value
                ddi_type = check_interaction(analysis, entities, id_e1, id_e2, stext)
                if ddi_type is not None:
                    with open(outfile, 'a') as f:
                        print(sid + "|" + id_e1 + "|" + id_e2 + "|" + ddi_type, file=f)
