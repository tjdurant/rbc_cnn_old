# rbc_cnn
conv net

rbc_cnn contains notebooks which are meant to get the individual cell images from the smear images. 

Dataset_prep will take smear images, and crop individual cells from the CellLabelTool dataset to produce a byte array of that cell image. This is then put into a hickle which can be passed as an object into the CNN. 
