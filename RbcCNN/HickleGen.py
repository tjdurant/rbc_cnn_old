import os
import numpy as np
import pandas as pd
import hickle
import cv2
import PIL
import time

from PIL import Image

# http://stackoverflow.com/questions/15514593/importerror-no-module-named-when-trying-to-run-python-script
import sys
sys.path.append('C:/Anaconda/Lib/site-packages')

start = time.time()

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
    
    # iteration counter
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

        # append getData value to list
        X.append(x)
        Y.append(y)
        IDs.append(n)

        # add 1 to iteration counter
        i += 1

        # instantiates list to np.arrays
        if i == 300:
            
            # individual cell byte arrays    
            npX = np.array(X)

            # individual cell labels
            npY = np.array(Y)

            # individual cell associated smear files
            npIDs = np.array(IDs)

            hickleCount += 1
             
            # reset byte array
            X = []
            # reset label array
            Y = []
            # reset IDs
            IDs = []

        elif i >= 600:
    
            # Cell Arrays: concatenates lists to np.arrays
            X = np.array(X)
            npX = np.concatenate((npX, X))

            # Label Arrays: concatenates lists to np.arrays
            Y = np.array(Y)
            npY = np.concatenate((npY, Y))

            # IDs: concatenates lists to np.arrays
            IDs = np.array(IDs)
            npIDs = np.concatenate((npIDs, IDs))

            hickleCount += 1
             
            # reset byte array
            X = []
            # reset label array
            Y = []
            # reset IDs
            IDs = []
            
            i = 301

    # Cell Arrays: concatenates lists to np.arrays
    X = np.array(X)
    npX = np.concatenate((npX, X))

    # Label Arrays: concatenates lists to np.arrays
    Y = np.array(Y)
    npY = np.concatenate((npY, Y))

    # IDs: concatenates lists to np.arrays
    IDs = np.array(IDs)
    npIDs = np.concatenate((npIDs, IDs))

    # create dictionary for arrays
    d = {}
    d['X'] = npX
    d['y'] = npY
    
    hickleCount += 1
    
    # (X, Y) -- ((N,3,w,h), label)
    hickle.dump(d, open('C:/Users/thoma/Documents/00GitHub/rbc_cnn/data/April.hkl','w'))


# Get image arrays for "classes" dataset
CreateLabelImageArrays(classes)

end = time.time()
print (end - start) 