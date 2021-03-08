import os
import pickle
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import OneHotEncoder

outfile = 'results.out'  # TODO: read from commandline

# delete old file
if os.path.exists(outfile):
    os.remove(outfile)

# Load data
datafile = 'devel.feat'  # TODO: read from commandline
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
for idx, token in enumerate(data):
    line = "{}|{}-{}|{}|{}".format(token[0],
                                   token[2], token[3], token[1], results[idx])
    # FIXME: Remove backslashes, Duplicates
    with open(outfile, 'a') as f:
        print(line, file=f)
