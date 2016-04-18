import hickle
import numpy as np

test = {}
cells = {}
labels = {}
hickleCount = 7
hickleCount += 1

for x in range(hickleCount):
    test = hickle.load('C:/Users/thoma/Documents/00GitHub/rbc_cnn/data/April{}.hkl'.format(x))
    cells["hk{}".format(x)] = test['X']
    labels["hk{}".format(x)] = test['y']

print len(cells)
print len(labels)

totalCellArray = np.concatenate()

print len(totalCellArray)

class HickleCombine(object):
    """description of class"""

