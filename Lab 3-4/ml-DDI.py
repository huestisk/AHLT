import os
import sys
import pickle
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import OneHotEncoder

sys.path.append(sys.path[0] + '/../common/')
from helper_functions_DDI import getFeatures


""" Get Features """
train = getFeatures("train.feat", "data/train/", recompute=True)
devel = getFeatures("devel.feat", "data/devel/", recompute=True)

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
classes = train[:, 3]
features = train[:, 4:]

feat_encoder = OneHotEncoder(handle_unknown='ignore')
X = feat_encoder.fit_transform(features)
pickle.dump(feat_encoder, open('encoder.pkl','wb'))

# Create & train model
svm = LinearSVC(dual=False, class_weight='balanced', max_iter=10000)
svm.fit(X, classes)
pickle.dump(svm, open("svm.model",'wb'))

# Eval
train_acc = svm.score(X, classes)
print("Training Accurracy: {}".format(train_acc))


""" Classify """
features = devel[:, 4:]
X = feat_encoder.transform(features)
ddi = svm.predict(X)

# delete old file
if os.path.exists("devel.out"):
    os.remove("devel.out")

# Print to file
for idx in range(len(ddi)):
    sid = devel[idx, 0]
    e1 = devel[idx, 1]
    e2 = devel[idx, 2]
    with open("devel.out", 'a') as f:
        print(f"{sid}|{e1}|{e2}|{ddi[idx]}", file=f)