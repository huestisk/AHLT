import re
import numpy as np

with open('branches.txt', 'r') as f:
    branches = f.read().splitlines()

branches = np.array([np.array((re.sub('[^A-Za-z0-9 ]+', '', line)).split(' ')).astype(int) for line in branches])
max_length = np.max(branches, axis=1)
min_length = np.min(branches, axis=1)

stats = [[np.mean(max_length), np.median(max_length), np.std(max_length)],
         [np.mean(min_length), np.median(min_length), np.std(min_length)]]

# [[3.8420870318250704, 4.0, 1.7463423738578836], 
#  [2.5592985494695824, 2.0, 0.8778267521174778]]
         
print(0)