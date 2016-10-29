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

WORK_DIR = 'C:\\Users\\thoma\\Documents\\00GitHub\\01_rbc_cnn\\rbc_cnn_old\\'
CSV_DIR = WORK_DIR + 'csv/'
DATA_DIR =  WORK_DIR + 'data/'

# Set IMAGE directory: Keep off of GitHub
IMG_DIR = 'C:/Users/thoma/Documents/00GitHub/00_LOCAL_ONLY/00RbcCNN_Sln_Images/'

hickle_date = "October_14_new"


def csv_to_dataFrame(path):
    """
    reads each csv in path and concatenates into single dataframe using os.walk
    os.join, and pd.concat. Returns a dataframe which contains all csvs in a given 
    directory. 
    """
    # output csv
    #fout=open(outputFile,"a")
    i = 0 

    for root, dirs, files in os.walk(path):

        frame = pd.DataFrame()
        list_ = []

        # iterates through files in directory 
        for file_ in files:

            # joins path with each file name
            csv_path = os.path.join(root, file_)

            # reads each file into df
            df = pd.read_csv(csv_path,index_col=0, header=0)
            
            # appends df to list
            list_.append(df)

        # concat dfs in list 
        frame = pd.concat(list_)
    frame = frame.reset_index(drop=True)
    return frame



def remove_overlaps(df):
    """
    removes overlapping cells by x, y, and source image. returns a dictionary
    """    
    dic = {} 
    
    df_dict = {}
    df_dict['total'] = df
    df_tommy = df[df.annotator == 'tommy']
    df_dict['tommy'] = df_tommy
    df_rick = df[df.annotator == 'rick']
    df_dict['rick'] = df_rick
    
    for key in df_dict:
        
        df = df_dict[key]
        
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
                # add True to index position of exclude if d[i] < 30     
            exclude[(d[i] < 20).nonzero()] = True
        
        # create overlap frame
        df_overlap = df[exclude == True]
        df_overlap = df_overlap.reset_index(drop=True)

        # create non_overlap frame
        df_nonoverlap = df[exclude == False]
        df_nonoverlap = df_nonoverlap.reset_index(drop=True)
        dic['{}_non_overlap'.format(key)] = df_nonoverlap

    return dic


def get_cropped_array(ind, dataframe, im):
    """
    takes each labeled cell from dataframe and gets cropped byte array.
    """

    # im = cv2.imread(IMG_DIR + dataframe.image[ind])
    df = dataframe.ix[ind]
    label = df.label
    
    uid = df.pk

    new_x = df.x - 35
    new_y = df.y - 35
    new_w = 70
    new_h = 70

    croppedCell = im[new_y:new_y+new_h, new_x:new_x+new_w]

    try:
        if croppedCell.shape == (70, 70, 3):
            return croppedCell, label, uid
    except Exception, e:
        print e

def create_hickle(dataframe, hickle_name):
    """
    Creates hickle of cropped byte arrays. 
    """

    # iteration counter
    i = 0

    # hickle counter
    hickleCount = 0

    X = []
    Y = []
    jpg = []
    pk = []

    # iterates over image column of dataframe
    for n in dataframe.image.iteritems():
        
        # this is the entire index object
        index_obj = dataframe.ix[n[0]]
        
        # n[0] = index
        index = n[0]
        imgName = n[1].replace (" ", "_")
        imgPath = IMG_DIR + imgName
        imgPath

        # get smear image array
        if n[1].endswith('.jpg'):
            im = cv2.imread(imgPath)        
        else:
            im_name = n[1] + '.jpg'
            im_path = IMG_DIR + im_name
            im = cv2.imread(im_path)
        im
        try:
            # pass index, dataframe, and image array to function
            x, y, uid = get_cropped_array(index, dataframe, im)
        except Exception, e:
            #type, value, tb = sys.exc_info()
            #traceback.print_exc()
            print e, n[1]
            continue

        # append getData value to list
        X.append(x)
        Y.append(y)
        jpg.append(n)
        pk.append(uid)

        # add 1 to iteration counter
        i += 1

        # instantiates list to np.arrays
        if i == 300:
            
            # individual cell byte arrays    
            npX = np.array(X)

            # individual cell labels
            npY = np.array(Y)

            # individual cell associated smear files
            npJPG = np.array(jpg)

            # individual unique ID of each cell
            npPK = np.array(pk)

            hickleCount += 1
             
            # reset byte array
            X = []
            # reset label array
            Y = []
            # reset jpgs
            jpg = []
            # reset pk
            pk = []

        elif i >= 600:
    
            # Cell Arrays: concatenates lists to np.arrays
            X = np.array(X)
            npX = np.concatenate((npX, X))

            # Label Arrays: concatenates lists to np.arrays
            Y = np.array(Y)
            npY = np.concatenate((npY, Y))

            # jpg: concatenates lists to np.arrays
            jpg = np.array(jpg)
            npJPG = np.concatenate((npJPG, jpg))

            # pk: concatenates lists to np.arrays
            pk = np.array(pk)
            npPK = np.concatenate((npPK, pk))

            hickleCount += 1
             
            # reset byte array
            X = []
            # reset label array
            Y = []
            # reset jpg
            jpg = []
            # reset pk
            pk = []

            i = 301

    
    # Cell Arrays: concatenates lists to np.arrays
    try:
        X = np.array(X)
        npX = np.concatenate((npX, X))
    except:
        npX = np.array(X)

    # Label Arrays: concatenates lists to np.arrays
    try:
        Y = np.array(Y)
        npY = np.concatenate((npY, Y))
    except:
        npY = np.array(Y)

    # jpg: concatenates lists to np.arrays
    try:
        jpg = np.array(jpg)
        npJPG = np.concatenate((npJPG, jpg))
    except:
        npJPG = np.array(jpg)

    # pk: concatenates lists to np.arrays
    try:
        pk = np.array(pk)
        npPK = np.concatenate((npPK, pk))
    except:
        npPK = np.array(pk)
        
    # create dictionary for arrays
    d = {}
    d['X'] = npX
    d['y'] = npY
    d['z'] = npJPG
    d['pk'] = npPK

    # (X, Y) -- ((N,3,w,h), label)
    hickle_path = DATA_DIR + '{}_{}.hkl'.format(hickle_date, hickle_name)
    hickle.dump(d, open(hickle_path, 'w'))


# read all csv cell 
df = csv_to_dataFrame(CSV_DIR)

# remove overlap cells
non_overlap_dic = remove_overlaps(df)

# create hickle for each dataframe
for key in non_overlap_dic:
    try:
        create_hickle(non_overlap_dic[key], key)
    except Exception, e:
        print e


