import numpy as np
import sys, os

if len(sys.argv) != 3:
    print('Invalid input! Please input: number to generate, output folder')
    sys.exit()
    
os.makedirs(sys.argv[2], exist_ok=True)

with open('{}/sr.txt'.format(sys.argv[2]), 'w') as f:
    for i in range(int(sys.argv[1])):
        
        s = 10**(np.random.rand()*1.602-2.0)
        r = 10**(np.random.rand()*1.778-2.0)
        
        f.write('{} {} {}\n'.format(i, s, r))
