import os
import numpy as np
import pandas as pd
import hickle

df = pd.DataFrame(np.random.randint(0,10,size=(10, 4)), columns=list('ABCD'))

for n in range(len(df)+10):
    print len(df), n
    if n >= len(df):
        break
    # select one cell to compare to rest of dataset
    cell = df.index[n]



class RbcCNN(object):
    """description of class"""


