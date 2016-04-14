import os
import numpy as np
import pandas as pd
import hickle


from PIL import Image

# http://stackoverflow.com/questions/15514593/importerror-no-module-named-when-trying-to-run-python-script

import sys

sys.path.append('C:/Anaconda/Lib/site-packages')

import cv2
import PIL

# Set DATA directory
DATA_DIR = 'C:/Users/thoma/Documents/00GitHub/rbc_cnn/'

# Set IMAGE directory: Keep off of GitHub
IMG_DIR = 'C:/Users/thoma/Documents/00GitHub/00_LOCAL_ONLY/00RbcCNN_Sln_Images/'

# read in cell labels from Cell Label Tool 
classes = pd.read_csv(DATA_DIR + 'dataset.csv', index_col=0, parse_dates=True)

# get rid of reject cells
classes = classes[classes.label != 'Reject']
classes = classes[classes.label != 'Abnormal']

# counts the number of cells in each calss
print classes['label'].value_counts()

# gets byte array from smear image using xywh coordinates 
def getData(ind, dataframe, im):
    # im = cv2.imread(IMG_DIR + dataframe.image[ind])
    df = dataframe.ix[ind]
    label = df.label
    
    new_x = df.x - 30
    new_y = df.y - 30
    new_w = 60
    new_h = 60

    croppedCell = im[new_y:new_y+new_h, new_x:new_x+new_w]

    if croppedCell.shape == (60, 60, 3):
        return croppedCell, label


def CreateLabelImageArrays(dataframe):
    
    # chunking counter
    i = 0

                
    # hickle counter
    hickleCount = 0

    # byte array
    X = []
    # label array
    Y = []
    # not sure
    IDs = []
    
    # iterates over image column of dataframe
    for n in dataframe.image.iteritems():
        
        # this is the entire index object
        index_obj = dataframe.ix[n[0]]
        
        # n[0] = index
        index = n[0]

        # get smear image array      
        im = cv2.imread(IMG_DIR + dataframe.image[index])

        try:
            # pass index, dataframe, and image array to function
            x, y = getData(index, dataframe, im)
        except Exception, e:
            #type, value, tb = sys.exc_info()
            #traceback.print_exc()
            print e
            continue

        X.append(x)
        Y.append(y)
        IDs.append(n)
        i += 1
        i
        if i > 300:
            
            # individual cell byte arrays    
            X = np.array(X)

            # individual cell labels
            Y = np.array(Y)

            # individual cell associated smear files
            IDs = np.array(IDs)

            # create dictionary for arrays
            d = {}
            d['X'] = X
            d['y'] = Y
            
            # (X, Y) -- ((N,3,w,h), label)
            hickle.dump(d, open('C:/Users/thoma/Documents/00GitHub/rbc_cnn/data/April{}.hkl'.format(hickleCount),'w'))
            
            hickleCount += 1

            i = 0
            hickleCount
             
            # reset byte array
            X = []
            # reset label array
            Y = []
            # reset IDs
            IDs = []

    # individual cell byte arrays    
    X = np.array(X)
    
    # individual cell labels
    Y = np.array(Y)
    
    # individual cell associated smear files
    IDs = np.array(IDs)

    # create dictionary for arrays
    d = {}
    d['X'] = X
    d['y'] = Y
            
    # (X, Y) -- ((N,3,w,h), label)
    hickle.dump(d, open('C:/Users/thoma/Documents/00GitHub/rbc_cnn/data/April{}.hkl'.format(hickleCount),'w'))

    return hickleCount

# Get image arrays for test3 dataset
hickleCount = CreateLabelImageArrays(classes)


