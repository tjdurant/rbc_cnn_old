import os
import numpy as np
import pandas as pd
import hickle

from PIL import Image

import matplotlib.pyplot as plt
# % matplotlib inline


# Set working directory
DATA_DIR = 'C:/Users/thoma/Documents/00GitHub/rbc_cnn/data/'
# Set image directory which is off of GitHub
IMG_DIR_1 = 'C:/Users/thoma/Documents/00GitHub/00_LOCAL_ONLY/Cellavision_Trial_images/Test3_AltFebruary/'
IMG_DIR_2 = 'C:/Users/thoma/Pictures/HemePictures/'

WRONG_CSV = 'C:/Users/thoma/Documents/00GitHub/rbc_cnn/wrong_cell_csv/'
WRONG_CELLS = 'C:/Users/thoma/Documents/00GitHub/rbc_cnn/wrong_cells/'

# http://stackoverflow.com/questions/15514593/importerror-no-module-named-when-trying-to-run-python-script

import sys
sys.path.append('C:/Anaconda/Lib/site-packages')

import cv2
import PIL

# Set DATA directory
CSV_DIR = 'C:/Users/thoma/Documents/00GitHub/rbc_cnn/csv/'
path =r'C:\Users\thoma\Documents\00GitHub\rbc_cnn\csv' # use your path

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


def wrong_image(dataframe):
    
    cropped_array = []
    i = 0
    
    for cell in wrong_df:
        correct_label = cell['Correct(x)']
        wrong_label = cell['Predicted(y)']
        pk_value = cell['PK_Value']
        
        cell_ix = df.ix[df['pk']==pk_value]
        img = df.get_value(cell_ix.index[0], 'image')
        path = "C:/Users/thoma/Documents/00GitHub/00_LOCAL_ONLY/00RbcCNN_Sln_Images/{}".format(img)
        
        im = cv2.imread(path)

        # im = cv2.imread(imgPath) 

        x = df.get_value(cell_ix.index[0], 'x')
        y = df.get_value(cell_ix.index[0], 'y')
        label = df.get_value(cell_ix.index[0], 'label')
        
        new_x = x - 30
        new_y = y - 30
        new_w = 60
        new_h = 60

        try:
            croppedCell = im[new_y:new_y+new_h, new_x:new_x+new_w]
            cropped_array.append(croppedCell)
            
            ## save each object
            cv2.imwrite(WRONG_CELLS+'{}_{}_{}.jpg'.format(correct_label,wrong_label,pk_value), croppedCell)
            
        except Exception, e:
            print e, img
    return cropped_array

    # pic = pic / 255        

GROUPS = {
        0:'NORMAL',
        1:'Echinocyte',
        2:'Dacrocyte',
        3:'Schistocyte',
        4:'Elliptocyte',
        5:'Acanthocyte',
        6:'Target cell',
        7:'Stomatocyte',
        8:'Spherocyte',
        9:'Overlap'
    }
wrong_df = pd.DataFrame.from_csv(WRONG_CELLS+'wrong_cells.csv',index_col=None)
for num in range(10):
    wrong_df = wrong_df.replace(to_replace=num, value=GROUPS[num])


wrong = np.genfromtxt(WRONG_CSV+'wrong.csv', delimiter=',',dtype=int)
df = csv_to_dataFrame(CSV_DIR)
wrong_cells = wrong_image(wrong)