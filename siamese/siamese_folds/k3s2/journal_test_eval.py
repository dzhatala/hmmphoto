#CHANGES from ipynn
 
# py -m pip show keras 
#tested on raptor, nvidia gtx 1050 ti 4gb, keras 2.3.1, tensorflow 2.3.1
#cuda_11.3.r11.3/compiler.29745058_0

import sys,pathlib
import numpy as np
import pandas as pd
from imageio.v2 import imread
import pickle
import os

#disable gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#enable gpu 
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

from x270_lib_05 import create_batch_test, create_filepath_cat_from_htk_list
from raptor_hambla28 import zoel_test_accuracy

from env_global import *

base_dir_cat=smaller_dir
# htk_train="d:\\rsync\\RESEARCHS\\finger_board_det\\python\\finger_board\\cat_train.txt"
# htk_train="./cat_train.txt"
# htk_test="d:\\rsync\\RESEARCHS\\finger_board_det\\python\\finger_board\\cat_test.txt"
# htk_test="./cat_test.txt"
# cat=["fp_image","answer_sheet"]
cat=categs # from env_global



catdir=["",""] #initializers
for i in range(2):
    catdir[i]="{}/{}".format(base_dir_cat,cat[i])

# k=2#fold number
# k=0#fold number
# if True:
for k in range(num_folds):
    print("\n\nfold k={}".format(k))
    htk_train="{}\\k{}\\siamese_train.lst".format(htk_folds_dir,k)
    htk_test="{}\\k{}\\test.lst".format(htk_folds_dir,k)

    # print (catdir)
    # quit()
    cat_ptrain=create_filepath_cat_from_htk_list(htk_train,catdir)
    cat_ptest=create_filepath_cat_from_htk_list(htk_test,catdir)

    imiov2_size=(498,280,3) #imageio io : w,h is reversed
    inputs,targets,cats,path_fns=create_batch_test(imiov2_size,cat_ptrain,cat_ptest)


    model_path = '{}/k{}/weights/'.format(project_name,k)
    req_size=(498,280,3)
    model=h28_get_siamese_model(req_size)
    # model.summary()

    optimizer = Adam(lr = 0.00006) #this Adam . lr is not recognized in x270 python only in raptor
    # optimizer =
    model.compile(loss="binary_crossentropy",optimizer=optimizer) 

    t_start = time.time()
    # start=7000  #iteration used by test
    # start=1800  #iteration used by test
    start=1000
    start=0 #get latest
    model.load_weights(os.path.join(model_path, "weights."+str(start)+".h5"))
    
    if(start<=0):
        model.load_weights(os.path.join(model_path, "weights."+"latest"+".h5"))

    print("Predicting ... !")
    print("-------------------------------------")
    outputs=model.predict(inputs)
    # print (targets)
    # print(outputs)
    print("Time for {0} images testing: {1} seconds".format(len(targets), (time.time()-t_start)))

    acc,terr,rec_results=zoel_test_accuracy(targets,outputs)
    print("\nfold(k={}) :  Correct percentage  is  {} %, error={}".format(k,round(acc*100,2),terr))


    #confusion matrix ?
    # print (rec_results)

    resdir="{}/k{}/results".format(project_name,k)
    # print (resdir) ;quit()
    pathlib.Path(resdir).mkdir(parents=True, exist_ok=True)

    with open('{}/nrec_targets.pickle'.format(resdir), 'wb') as handle:
        pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

    with open('{}/nrec_outputs.pickle'.format(resdir), 'wb') as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


    with open('{}/nrec_results.pickle'.format(resdir), 'wb') as handle:
        pickle.dump(rec_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
       
    with open('{}/ncats.pickle'.format(resdir), 'wb') as handle:
        pickle.dump(cats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
        

    with open('{}/npath_fns.pickle'.format(resdir), 'wb') as handle:
        pickle.dump(path_fns, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()


# for k in range(num_folds):
    