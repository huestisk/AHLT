import numpy as np
import pickle

# arguments
outfile = 'devel.out'
featfile = 'devel.feat'
svmfile = 'train.model'
encoder = 'encoder.pkl'

# Load data
with open(featfile, 'rb') as f:
    data = pickle.load(f)

classes = data[:, 3]
features = data[:, 4:]

# Load model & encoder
svm = pickle.load(open(svmfile, 'rb'))
feat_encoder = pickle.load(open(encoder, 'rb'))
X = feat_encoder.transform(features)

# Classify
ddi = svm.predict(X)

# Print to file
for idx in range(len(ddi)):
    sid = data[idx, 0]
    e1 = data[idx, 1]
    e2 = data[idx, 2]
    with open(outfile, 'a') as f:
        print(f"{sid}|{e1}|{e2}|{ddi[idx]}", file=f)

