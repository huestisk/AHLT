#from eval.evaluator import evaluate
import sys
from helper_functions import *
from neural_network import *
import sys
# you can comment out this, it just sets my path to the root folder instead of lab5
sys.path.append('/Users/betty/Desktop/AHLT/AHLT')

# parse arguments
datadir = sys.argv[1]


test_idx = {'words': { '<PAD >':0 , '<UNK >':1 , '11-day ':2 , 'murine':3 , 'criteria ':4 ,
            'stroke':5,'levodopa':8511, 'terfenadine':8512},
            'labels ': {'<PAD >':0, 'B- group':1, 'B- drug_n':2, 'I- drug_n':3, 'O':4,
            'I- group':5, 'B- drug':6, 'I- drug':7, 'B- brand':8, 'I- brand':9},
            'maxlen':100
            }

#TODO split data into dirs for train and test

# test - these should be used in neural_network.py
data = load_data(datadir)
idx = create_indexs(dataset=data,max_length=100)
print(idx)
#idx = create_indexs(dataset=data, max_length=10)
#X = encode_words(dataset=data, idx=idx)
#y = encode_labels(dataset=data, idx=idx)

# not tested with a real model yet
#model = build_network(test_idx)
#save_model_and_indexs(model,test_idx, 'test')
#model,idx = load_model_and_indexs('test')
