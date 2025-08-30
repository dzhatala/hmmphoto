#CHANGES from ipynn
 
# py -m pip show keras 
#tested on raptor, gtx 1050 ti 4gb, keras 2.3.1, tensorflow 2.3.1


import sys, pathlib
import numpy as np
import pandas as pd
# from scipy.misc import imread
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

# from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers.pooling import MaxPooling2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers.merge import Concatenate
# from tensorflow.keras.layers.core import Lambda, Flatten, Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform

from tensorflow.keras.initializers import glorot_uniform



# from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.layers import Layer

from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

from sklearn.utils import shuffle

import numpy.random as rng

# from x270_lib_01 import init_journal_train
from x270_lib_05 import folds_init_journal_train


# inputs,targets=init_journal_train("k3s2\k0")

# htk_folds_dir="D:\\RESEARCHS\\finger_board\\github_jurnal\\hmm\\scr_folds\\k3_shot2"

# from env_global import htk_folds_dir, smaller_dir, num_folds, categs
from env_global import *


model_path=[""]*num_folds
for k in range (num_folds):
    model_path[k] = "{}/k{}/weights/".format(project_name,k)
    pathlib.Path(model_path[k]).mkdir(parents=True, exist_ok=True)

epsilon=1e-2
for k in range (num_folds):
    # if k==0:
        # continue
    ftrain="{}\\k{}\\siamese_train.lst".format(htk_folds_dir,k)
    # ftest="{}\\k{}\\test.lst".format(htk_folds_dir,k)
    
    inputs,targets=folds_init_journal_train(ftrain,smaller_dir,categs)
    # print_text_file(ftrain)
    req_size=(498,280,3)
    model=h28_get_siamese_model(req_size)
    # model.summary()

    optimizer = Adam(lr = 0.00006) #this Adam . lr is not recognized in x270 python only in raptor
    # optimizer =
    model.compile(loss="binary_crossentropy",optimizer=optimizer) 

    print("Starting training process!")
    print("-------------------------------------")
    t_start = time.time()
    start=0
    # start=800
    # n_iter=800
    n_iter=1000
    # n_iter=200
    
    evaluate_every=100
    evaluate_every=300
    
    if start>0:
        model.load_weights(os.path.join(model_path, "weights."+str(start)+".h5"))
    for i in range(start,start+ n_iter+1):
        # (inputs,targets) = get_batch(batch_size) #why this is call in every iter?
        loss = model.train_on_batch(inputs, targets)
        if i % evaluate_every == 0:
            print("\n ------------- \n")
            print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
            print("Train Loss: {0}".format(loss)) 
            # val_acc = test_oneshot(model, N_way, n_val, verbose=True) #not done
            model.save_weights(os.path.join(model_path[k], 'weights.{}.h5'.format(i)))

            if (loss<epsilon):
                break
            # if val_acc >= best:
            #     print("Current best: {0}, previous best: {1}".format(val_acc, best))
            #     best = val_acc

    model.save_weights(os.path.join(model_path[k], 'weights.latest.h5'.format(i)))
for k in range (num_folds):
    print("models save in {}",model_path[k])
