#CHANGES from ipynn

# py -m pip show keras 
#tested on raptor, gtx 1050 ti 4gb, keras 2.3.1, tensorflow 2.3.1



import sys
import numpy as np
import pandas as pd
from imageio.v2 import imread
import pickle
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from PIL import Image
from raptor_hambla28 import h28_get_siamese_model
import matplotlib.pyplot as plt
# %matplotlib inline

import cv2
import time
import traceback as tb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.initializers import glorot_uniform

from tensorflow.keras.layers import Layer

from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K





from sklearn.utils import shuffle

import numpy.random as rng

from x270_lib_03 import create_batch_test, create_filepath_cat_from_htk_list
from raptor_hambla28 import zoel_test_accuracy


base_dir_cat="d:\\rsync\\RESEARCHS\\finger_board_det\\github_jurnal\\siamese"
htk_train="d:\\rsync\\RESEARCHS\\finger_board_det\\python\\finger_board\\cat_train.txt"
htk_test="d:\\rsync\\RESEARCHS\\finger_board_det\\python\\finger_board\\cat_test.txt"
cat=["fp_image","answer_sheet"]


catdir=["",""]
for i in range(2):
    catdir[i]="{}\\data\\smaller\\{}".format(base_dir_cat,cat[i])

cat_ptrain=create_filepath_cat_from_htk_list(htk_train,catdir)
cat_ptest=create_filepath_cat_from_htk_list(htk_test,catdir)

imiov2_size=(498,280,3) #imageio io : w,h is reversed
inputs,targets=input,target=create_batch_test(imiov2_size,cat_ptrain,cat_ptest)


model_path = './siamese/weights/'
req_size=(498,280,3)
model=h28_get_siamese_model(req_size)
# model.summary()

optimizer = Adam(lr = 0.00006) #this Adam . lr is not recognized in x270 python only in raptor
# optimizer =
model.compile(loss="binary_crossentropy",optimizer=optimizer) 

t_start = time.time()
start=7000  #iteration used by test
model.load_weights(os.path.join(model_path, "weights."+str(start)+".h5"))

print("Predicting ... !")
print("-------------------------------------")
output=model.predict(inputs)
# print (targets)
# print(output)
print("Time for {0} images testing: {1} seconds".format(len(targets), (time.time()-t_start)))

acc,terr=zoel_test_accuracy(targets,output)
print("Correct percentage  is  {} %, error={}".format(round(acc*100,2),terr))


