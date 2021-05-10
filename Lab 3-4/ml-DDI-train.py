import sys
import pickle
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import OneHotEncoder

sys.path.append(sys.path[0] + '/../common/')
from helper_functions_DDI import getFeatures

# parse arguments
datadir = sys.argv[1]
featfile = sys.argv[2]
# svmfile = sys.argv[3]

""" Get Features """
features = getFeatures(featfile, datadir, recompute=True)

# cheat = features[features[:,3]!='None']
# cheat = np.array([cheat[:,3], cheat[:,6]]).T  
# gold = cheat[:,0]
# lemma = [word[10:] for word in cheat[:,1]]

# from itertools import compress
# int_lemma = list(compress(lemma, gold=='int'))
# mech_lemma = list(compress(lemma, gold=='mechanism'))
# eff_lemma = list(compress(lemma, gold=='effect'))
# ad_lemma = list(compress(lemma, gold=='advise'))


""" Train Network """
# Preprocessing
classes = features[:, 3]
features = features[:, 4:]

feat_encoder = OneHotEncoder(handle_unknown='ignore')
X = feat_encoder.fit_transform(features)
pickle.dump(feat_encoder, open('encoder.pkl','wb'))

# Create & train model
svm = LinearSVC(dual=False, class_weight='balanced', max_iter=5000)
svm.fit(X, classes)
pickle.dump(svm, open(svmfile,'wb'))

# Eval
train_acc = svm.score(X, classes)
print("Training Accurracy: {}".format(train_acc))

