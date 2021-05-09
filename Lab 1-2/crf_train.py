# Base on https://github.com/scrapinghub/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb

import sys
import numpy as np

import pycrfsuite

# Load data
datafile = sys.argv[1]
outfile = sys.argv[2]

with open(datafile, 'r') as f:
    data = f.readlines()

# Preprocessing
data = [line[:-1].split('\t') for line in data]
X_train = []
y_train = []
x_seq = []
y_seq = []

for line in data:
    if line[0] == '':
        X_train.append(x_seq)
        x_seq = []
        y_train.append(y_seq)
        y_seq = []
    else:
        x_seq.append(line[5:])
        y_seq.append(line[4])

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 5000,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})


trainer.train(outfile)



