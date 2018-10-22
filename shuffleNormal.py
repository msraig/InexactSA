import numpy as np
import os, sys

if len(sys.argv) != 2:
    print('Invalid input! Please input: data folder')

datacnt = 0
with open('{}/sr.txt'.format(sys.argv[1])) as f:
    while True:
        line = f.readline()
        if len(line) < 2:
            break
        datacnt += 1

per = np.random.permutation(datacnt)
for i in range(datacnt):
    os.rename('{}/{}_normal.pfm'.format(sys.argv[1],i), '{}/_{}_normal.pfm'.format(sys.argv[1],i))

for i in range(datacnt):
    os.rename('{}/_{}_normal.pfm'.format(sys.argv[1],i), '{}/{}_normal.pfm'.format(sys.argv[1],per[i]))
