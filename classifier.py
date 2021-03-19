import os
import sys
import pickle
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import OneHotEncoder

from eval.evaluator import evaluate


# Output file
outfile = sys.argv[2]
# delete old file
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

# Print to file
results = svm.predict(X)

idx = results != '0'
tokens = data[idx, :4]
labels = results[idx]

lines = set()
prevPos = None
prevLabel = None
for token, label in zip(tokens, labels):
    
    pos, fLabel = label.split('-')

    docId = token[0]
    start = token[2]
    end = token[3]
    name = token[1]

    # if pos == 'I' and docId == token[0] and int(end)+2 == int(token[2]):
    #     end = token[3]
    #     name += " " + token[1]
    #     lines.pop()
    # elif pos != 'B':
    #     print('Error')
    # else:
    #     docId = token[0]
    #     start = token[2]
    #     end = token[3]
    #     name = token[1]
        
    lines.add(
        "{}|{}-{}|{}|{}".format(docId, start, end, name, fLabel)
    )

    prevPos = pos
    prevLabel = fLabel

for line in lines:
    with open(outfile, 'a') as f:
        print(line, file=f)

# print performance score
evaluate("NER", "data/test/", outfile)