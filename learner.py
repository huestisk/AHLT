import sys
import numpy as np
import pickle

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import OneHotEncoder

# Load data
datafile = sys.argv[1]
outfile = sys.argv[2]

with open(datafile, 'r') as f:
    data = f.readlines()

# Preprocessing
data = np.array([np.array(line[:-1].split('\t'))
                 for line in data if line != '\n'])
classes = data[:, 4]
features = data[:, 5:]

feat_encoder = OneHotEncoder(handle_unknown='ignore')
X = feat_encoder.fit_transform(features)
pickle.dump(feat_encoder, open('encoder.pkl','wb'))

# Create & train model
svm = LinearSVC(verbose=0)
svm.fit(X, classes)
pickle.dump(svm, open('svm.model','wb'))

# Eval
train_acc = svm.score(X, classes)
print("Training Accurracy: {}".format(train_acc))


# FIXME: For future reference
# all_classes = set(classes)
# class_encoder = OneHotEncoder()
# y = class_encoder.fit_transform(classes.reshape(-1, 1))
# all_features = set(features.flatten())
# loaded_model = pickle.load(open(filename, 'rb'))