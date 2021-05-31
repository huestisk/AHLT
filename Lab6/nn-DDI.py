#from eval.evaluator import evaluate
import os
import sys
import shutil
import numpy as np
path = sys.path.append('/Users/betty/Desktop/AHLT/AHLT')
from helper_functions import *
#from neural_network import *


# parse arguments
datadir = sys.argv[1]

#learn('data/train/', 'data/test/', 'LSTMTEST' )

# test - these should be used in neural_network.py
data = load_data('/Users/betty/Desktop/AHLT/AHLT/data/test/')
print(data)