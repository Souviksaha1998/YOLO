import numpy as np

a = np.zeros((3,3,5))

print(a)

a[0,2,0:4] = [4,4,4,4]

print(a)