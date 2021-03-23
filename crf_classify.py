import sys
import os
import numpy as np

import pycrfsuite

from eval.evaluator import evaluate

model = sys.argv[1]

# Output file
outfile = sys.argv[3]
if os.path.exists(outfile):
    os.remove(outfile)

# Load data
datafile = sys.argv[2]
with open(datafile, 'r') as f:
    data = f.readlines()

# Preprocessing
data = [line[:-1].split('\t') for line in data]
X_test = []
x_seq = []
tokens = []

for line in data:
    if line[0] == '':
        X_test.append(x_seq)
        x_seq = []
    else:
        tokens.append(line[:4])
        x_seq.append(line[5:])


# Classifier
tagger = pycrfsuite.Tagger()
tagger.open(model)

y_pred = [tagger.tag(xseq) for xseq in X_test]
labels = [label for seq in y_pred for label in seq]

lines = set()
prevPos = None
prevLabel = None
prevDocId = None
for token, label in zip(tokens, labels):

    if label == '0':
        continue
    
    docId = token[0]
    pos, fLabel = label.split('-')

    if pos == 'I' and prevPos == 'B' and docId == prevDocId and prevLabel == fLabel and int(end)+2 == int(token[2]): 
        # If tag is I, previous tag is B (with same document ID and label) and they follow each other 
        lines.pop()     # remove previous, since it will be combined
        name += " " + token[1]
    else:
        name = token[1]
        start = token[2]

    end = token[3]
        
    lines.add(
        "{}|{}-{}|{}|{}".format(docId, start, end, name, fLabel)
    )

    prevPos = pos
    prevLabel = fLabel
    prevDocId = docId

for line in lines:
    with open(outfile, 'a') as f:
        print(line, file=f)

# print performance score
evaluate("NER", "data/test/", outfile)