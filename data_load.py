import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array , load_img
import numpy as np
import cv2
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np

import os
import random
import gc

import seaborn as sns
from sklearn.model_selection import train_test_split
import random
from imutils import paths
fname = 'val2017'


# def read_to_imgarray(img_list):
#     X = []
#     Y = []

#     for img in img_list:
#         X.append(cv2.resize(cv2.imread(fname+'/'+   img),(224,224), interpolation=cv2.INTER_CUBIC))
#         # if 'dog' in img:
#         # Y.append(1)
#         # elif 'cat' in img:
#             # Y.append(0)
#     return X



def data_load(fname):
# h = 128
# w = 128

    # train_imgs = os.listdir(fname)
    train_imgs = list(paths.list_images(fname))

    # l = len(listOfFiles)
    random.shuffle(train_imgs)

    gc.collect()


    # nrows = 256
    # ncolumns = 256
    # channels = 3




    # X,y = read_to_imgarray(train_imgs)
    # del train_imgs
    # gc.collect()

    # X = np.array(X)
    # y = np.array(y)

    # sns.countplot(y)
    # plt.title("DOG Vs. Cat Data")

    X_train, X_val = train_test_split(train_imgs,test_size=0.1,random_state=2)

    print(len(X_train), len(X_val))

    return (X_train, X_val)

