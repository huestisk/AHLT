import os
import sys
import pickle
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import OneHotEncoder

from eval.evaluator import evaluate

CHECK_EXTERNAL = False

# Output file
outfile = sys.argv[2]
if os.path.exists(outfile):
    os.remove(outfile)

# Load data
datafile = sys.argv[1]
with open(datafile, 'r') as f:
    data = f.readlines()

if CHECK_EXTERNAL:
    drugbank = './resources/DrugBank.txt'
    hsbd = './resources/HSDB.txt'

    brands = []
    groups = []

    with open(hsbd, 'r') as f:
        drugs = f.read().splitlines()

    with open(drugbank, 'r') as f:
        for line in f.readlines():
            raw = line.split('|')
            if raw[1] == "drug\n":
                drugs.append(raw[0])
            elif raw[1] == "brand\n":
                brands.append(raw[0])
            elif raw[1] == "group\n":
                groups.append(raw[0])

# Preprocessing
data = np.array([np.array(line[:-1].split('\t'))
                 for line in data if line != '\n'])
classes = data[:, 4]
features = data[:, 5:]

# Load model & encoder
svm = pickle.load(open('svm.model', 'rb'))
feat_encoder = pickle.load(open('encoder.pkl', 'rb'))
X = feat_encoder.transform(features)

# # Eval
# test_acc = svm.score(X, classes)
# print("Test Accurracy: {}".format(test_acc))

# Print to file
labels = svm.predict(X)

lines = set()
prevPos = None
prevLabel = None
prevDocId = None
for token, label in zip(data[:,:4], labels):

    if CHECK_EXTERNAL and label == '0':  # override result
        if token[1] in drugs:
            label = "B-drug"
        elif token[1] in brands:
            label = "B-brand"
        elif token[1] in groups:
            label = "B-group"
    
    if label == '0':
        continue
    
    docId = token[0]
    pos, fLabel = label.split('-')

    # if False and pos == 'I' and prevPos == 'B' and docId == prevDocId and prevLabel == fLabel and int(end)+2 == int(token[2]): 
    #     # If tag is I, previous tag is B (with same document ID and label) and they follow each other 
    #     lines.pop()     # remove previous, since it will be combined
    #     name += " " + token[1]
    # else:
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