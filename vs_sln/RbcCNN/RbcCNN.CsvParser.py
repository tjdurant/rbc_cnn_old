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
CSV_DIR = 'C:/Users/thoma/Documents/00GitHub/rbc_cnn/csv/'

# Set IMAGE directory: Keep off of GitHub
IMG_DIR = 'C:/Users/thoma/Documents/00GitHub/00_LOCAL_ONLY/00RbcCNN_Sln_Images/'

COMBINED_CSV = CSV_DIR + 'combined_output.csv'

def combine_csv_files(path, outputFile):
    
    # output csv
    fout=open(outputFile,"a")
    i = 0 

    for root, dirs, files in os.walk(path):

        for file in files:

            # skips created csv created
            if file.endswith('combined_output.csv'):
                continue

            # first elif includes header line
            elif i == 0:
                csv_path = os.path.join(root, file)
                try:
                    with open(csv_path, 'rb') as f:
                        for line in f:
                            fout.write(line)
                        f.close() # not really needed
                        i += 1
                except IOError, e:
                    print e


            elif file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                try:
                    with open(csv_path, 'rb') as f:
                        f.next()
                        for line in f:
                            fout.write(line)
                        f.close() # not really needed
                except IOError, e:
                    print e

            else:
                print "no csvs present in directory"

combine_csv_files(CSV_DIR,COMBINED_CSV)

class RbcCNN(object):
    """description of class"""


