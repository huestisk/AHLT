import sys
from os import listdir
from xml.dom.minidom import parse

from eval.evaluator import evaluate
from src.tokenize import tokenize
from src.extract_entities import extract_entities


datadir = sys.argv[1]
outfile = sys.argv[2]

# process  each  file in  directory
for f in listdir(datadir):
    # parse  XML file , obtaining a DOM  tree
    tree = parse(datadir + "/" + f)
    # process  each  sentence  in the  file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"]. value   # get  sentence  id
        stext = s.attributes["text"]. value   # get  sentence  text
        # tokenize  text
        tokens = tokenize(stext)
        # extract  entities  from  tokenized  sentence  text
        entities = extract_entities(tokens)

        # print  sentence  entities  in  format  requested  for  evaluation
        for e in entities:
            print(sid+"|"+e["offset"]+"|"+e["text"]+"|"+e["type"], file=outf)
            
# print  performance  score
evaluate("NER", datadir, outfile)
