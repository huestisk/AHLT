#from eval.evaluator import evaluate
import os
import sys
import shutil
import numpy as np
os.system("python --version")

from helper_functions import *
from neural_network import *

# you can comment out this, it just sets my path to the root folder
#sys.path.append('/Users/betty/Desktop/AHLT/AHLT')

# parse arguments
datadir = sys.argv[1]

learn('data/train/', 'data/test/', 'LSTMTEST' )

# test - these should be used in neural_network.py
#data = load_data(datadir)
#print(data)
#idx = create_indexs(dataset=data,max_length=100)
#print(idx)
#encoded = encode_labels(dataset=data, idx=idx)
#print(encoded[0])
#idx = create_indexs(dataset=data, max_length=10)
#X = encode_words(dataset=data, idx=idx)
#y = encode_labels(dataset=data, idx=idx)

# not tested with a real model yet
#model = build_network(test_idx)
#save_model_and_indexs(model,test_idx, 'test')
#model,idx = load_model_and_indexs('test')
