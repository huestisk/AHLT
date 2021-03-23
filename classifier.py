import os
import sys
import pickle
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import OneHotEncoder

from eval.evaluator import evaluate

CHECK_EXTERNAL = True

# Output file
outfile = sys.argv[2]
if os.path.exists(outfile):
    os.remove(outfile)

# Load data
datafile = sys.argv[1]
with open(datafile, 'r') as f:
    data = f.readlines()

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

if CHECK_EXTERNAL:
    drugbank = './resources/DrugBank.txt'
    hsbd = './resources/HSDB.txt'

    drugs = []
    brands = []
    groups = []

    drug_suffixes = ['racin', 'NaFlu','teine', 'butin','ampin', 'navir', 'azone', 'ncers', 'moter', 'orine', 'limus', 'ytoin', 'angin', 'ucose', 'sulin', 'odone', 'xacin', 'udine', 'osine', 'zepam', 'hacin', 'etine', 'pride', 'ridol', 'apine', 'idone', 'apine', 'nafil', 'otine', 'oxide', 'iacin', 'tatin', 'cline', 'kacin','xacin', 'illin', 'azole', 'idine', 'amine', 'mycin', 'tatin', 'ridin', 'caine', 'micin', 'hanol', 'dolac', 'feine', 'lline', 'amide', 'afine', 'rafin', 'trast','goxin', 'coxib', 'phine', 'coids', 'isone', 'oride', 'ricin', 'lipin', 'cohol', 'otine', 'taxel', 'tinib', 'rbose', 'ipine', 'idine', 'nolol', 'uride', 'lurea', 'ormin', 'amide', 'estin']
    # plural of drugs can be group names
    group_suffixes = [s + "s" for s in drug_suffixes]
    group_suffixes = [s[1:] for s in group_suffixes]
    group_suffixes = group_suffixes + ['trate', 'otics', 'pioid', 'zones', ]

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

# Print to file
results = svm.predict(X)

idx = results != '0'
tokens = data[idx, :4]
labels = results[idx]

lines = set()
prevPos = None
prevLabel = None
prevDocId = None
for token, label in zip(tokens, labels):
    
    docId = token[0]
    pos, fLabel = label.split('-')

    if CHECK_EXTERNAL:  # override result
        if token[1] in drugs:
            pos, fLabel = '', "drug"
        elif token[1] in brands:
            pos, fLabel = '', "brand"
        elif token[1] in groups:
            pos, fLabel = '', "group"

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