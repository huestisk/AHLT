import sys
import os
from xml.dom.minidom import parse

from helper_functions import tokenize, extract_features, get_tag

# parse arguments
datadir = sys.argv[1]
outfile = sys.argv[2]

# delete old file
if os.path.exists(outfile): 
    os.remove(outfile)

# process each file in directory
for f in os.listdir(datadir):
    # parse XML file, obtaining a DOM tree
    tree = parse(datadir + f)
    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text
        # load ground truth entities
        gold = []
        entities = s.getElementsByTagName("entity")
        for e in entities:
            # for discontinuous entities, we only get the first span
            offset = e.attributes["charOffset"].value
            (start, end) = offset.split(";")[0].split("-")
            gold.append((int(start), int(end), e.attributes["type"].value))
            # tokenize text
            tokens = tokenize(stext)
            # extract features for each word in the sentence
            features = extract_features(tokens)
            # print features in format suitable for the learner/classifier
            for i in range(0,len(tokens)):
                # see if the token is part of an entity, and which part (B/I)
                tag = get_tag(tokens[i], gold)
                with open(outfile, 'a') as f:
                    print(sid , tokens[i][0], tokens[i][1], tokens[i][2],
                        tag, "\t".join(features[i]), sep='\t', file=f)
            # blank line to separate sentences 
            with open(outfile, 'a') as f:
                print('\n', file=f)






