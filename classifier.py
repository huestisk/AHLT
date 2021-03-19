import os
import sys
import pickle
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import OneHotEncoder


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

# Eval
test_acc = svm.score(X, classes)
print("Test Accurracy: {}".format(test_acc))

# Print to file
results = svm.predict(X)

idx = results != '0'
tokens = data[idx, :4]
labels = results[idx]

lines = set()
docId = None

for token, label in zip(tokens, labels):
    
    if token[0] != docId:        # if new doc, then can't be the same name
        docId = token[0]

    pos, fLabel = label.split('-')

    start = token[2]
    end = token[3]
    name = token[1]

    lines.add(
        "{}|{}-{}|{}|{}".format(docId, start, end, name, fLabel)
    )
# else:
#     lines.append(
#         "{}|{}-{}|{}|{}".format(docId, start, end, name, final_label)
#     )


# FIXME: combine results
for line in lines:
    with open(outfile, 'a') as f:
        print(line, file=f)
