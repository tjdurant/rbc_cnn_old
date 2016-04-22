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



# Set DATA directory
DATA_DIR = 'C:/Users/thoma/Documents/00GitHub/rbc_cnn/'
CSV_DIR = DATA_DIR + 'csv/'

# Set IMAGE directory: Keep off of GitHub
IMG_DIR = 'C:/Users/thoma/Documents/00GitHub/00_LOCAL_ONLY/00RbcCNN_Sln_Images/'

# read in cell labels from Cell Label Tool 
classes = pd.read_csv(DATA_DIR + 'dataset.csv', index_col=0, parse_dates=True)

# get rid of reject cells
classes = classes[classes.label != 'Reject']
classes = classes[classes.label != 'Abnormal']

# counts the number of cells in each calss
#print classes['label'].value_counts()

def csv_to_dataFrame(path):
    
    # output csv
    #fout=open(outputFile,"a")
    i = 0 

    for root, dirs, files in os.walk(path):
        frame = pd.DataFrame()
        list_ = []
        for file_ in files:
            csv_path = os.path.join(root, file_)
            df = pd.read_csv(csv_path,index_col=0, header=0)
            list_.append(df)
        frame = pd.concat(list_)
    frame = frame.reset_index(drop=True)
    return frame


def overlap_cell_check(df, cell, second_cell):
    
    overlap_list = []

    # absolute < 30 
    if cell.x in range(second_cell.x-30, second_cell.x+30):
        if cell.y in range(second_cell.y-30, second_cell.y+30):
            if cell.image == second_cell.image:
                if cell.annotator != second_cell.annotator:
                    overlap_list.append(cell)
                    overlap_list.append(second_cell)
                    return overlap_list
                else:
                    return None
            else:
                return None
        else:
            return None
    else:
        return None

    return 


def parse_dataFrame(df):
    
    x = np.array(df['x'])
    y = np.array(df['y'])
    i = np.array(df['image'])

    # create linkage table 
    dx = x[:, np.newaxis] - x
    dy = y[:, np.newaxis] - y
    di = (i[:, np.newaxis] != i)*1e9

    # set diagonal of linkage table to large number
    dx[np.diag_indices(len(dx))] = 1e9
    dy[np.diag_indices(len(dy))] = 1e9
    di[np.diag_indices(len(di))] = 1e9

    # get absolute values
    dx = np.abs(dx)
    dy = np.abs(dy)

    # sum vector of x, y, and image
    d = dx + dy + di

    # set variable = to min of 0 for vector summation
    b = d.min(0)

    # create array with zeros of length b 
    exclude = np.zeros(len(b))

    # iterate through index of length b 
    for i in range(len(b)):
        # if True/anything present at exclude[i]; skip
        if exclude[i]:
            continue
        if b[i] < 30:
            # add True to index position of exclude if d[i] < 30     
            exclude[(d[i] < 30).nonzero()] = True

    # create overlap frame
    df_overlap = df[exclude == True]
    df_overlap = df_overlap.reset_index(drop=True)

    # create non_overlap frame
    df_nonoverlap = df[exclude == False]
    df_nonoverlap = df_nonoverlap.reset_index(drop=True)

    # create rick frame
    df_r = df[df.annotator == 'rick']
    df_r = df_r.reset_index(drop=True)

    # create tommy frame
    df_t = df[df.annotator == 'tommy']
    df_t = df_t.reset_index(drop=True)

    return df_nonoverlap, df_overlap, df_r, df_t


# gets byte array from smear image using xywh coordinates 
def get_cropped_array(ind, dataframe, im):
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


def create_hickle(dataframe, hickle_name):
    
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
        if n[1].endswith('.jpg'):
            im = cv2.imread(IMG_DIR + n[1])
        else:
            im_name = n[1] + '.jpg'
            im_path = IMG_DIR + im_name
            im = cv2.imread(im_path)
        im
        n[1]
        try:
            # pass index, dataframe, and image array to function
            x, y = get_cropped_array(index, dataframe, im)
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
    hickle.dump(d, open('C:/Users/thoma/Documents/00GitHub/rbc_cnn/data/April_{}.hkl'.format(hickle_name),'w'))


# return df from all csv files in CSV_DIR
df = csv_to_dataFrame(CSV_DIR)

# parse dataframes into appropriate dfs and return to list
arrays = []
arrays = parse_dataFrame(df)

for frame in arrays:
    try:
        create_hickle(frame, str(len(frame)))
    except Exception, e:
        print e


