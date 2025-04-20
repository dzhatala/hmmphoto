#!/usr/bin/env python3
import sys
import numpy as np
# import pandas as pd
#from scipy.misc import imread
from imageio.v2 import imread
import pickle
import os
import matplotlib.pyplot as plt
#%matplotlib inline

import cv2
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform

#from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K #https://stackoverflow.com/questions/63467984/attributeerror-module-tensorflow-python-keras-has-no-attribute-abs

from sklearn.utils import shuffle

import numpy.random as rng

from pprint import pprint


# train_folder = "./few_background"
# val_folder = './few_evaluation'
# save_path = './data/'

import traceback as tb;



def crop_image(image,req_size=None):
    '''
    crop center based on req_size
    https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    '''
    print("cropping image into =>", req_size)
    if not req_size is None:
        # print(image.shape,req_size)
        if(image.shape==req_size):
            print("size already match, no crop")
            return image
        print("fix here #0003: img size is ", image.shape," req is: ",req_size) ; quit()    
    return image

def zoel_loadimgs(path,n = 0,req_size=None):
    '''
    path => Path of train directory or test directory
    '''
    X=[]
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n
    
    # we load every alphabet seperately so we can isolate them later
    for alphabet in os.listdir(path):
        print("loading alphabet: " + alphabet)
        lang_dict[alphabet] = [curr_y,None]
        alphabet_path = os.path.join(path,alphabet)
        print(alphabet_path)
        # every letter/category has it's own column in the array, so  load seperately
        for letter in os.listdir(alphabet_path):
            cat_dict[curr_y] = (alphabet, letter)
            category_images=[]
            letter_path = os.path.join(alphabet_path, letter)
            # read all the images in the current category
            for filename in os.listdir(letter_path):
                if filename=="Thumbs.db":
                    continue
                image_path = os.path.join(letter_path, filename)
                image_pre = imread(image_path)
                image=crop_image(image_pre,req_size) #req_size
                category_images.append(image)
                y.append(curr_y)
                print(image_path)
            try:
                X.append(np.stack(category_images))
            # edge case  - last one
            except ValueError as e:
                # print(e)
                tb.print_exc()
                # print("error - category_images:", category_images)
            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1
    print ("shape y to stack"); print (type(y)) ; print (len(y)), print (len(X))
    if (len(X)==0):
        print("list X is 0 length")
        quit()
    y = np.vstack(y)
    X = np.stack(X)
    for fruit in sorted(lang_dict, reverse=True):
        print(fruit, "->", lang_dict[fruit])

    for fruit in sorted(cat_dict, reverse=True):
        print(fruit, "->", cat_dict[fruit])
    
    return X,y,lang_dict,cat_dict



def h28_initialize_weights(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def h28_initialize_bias(shape, dtype=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def h28_get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    model = Sequential()     # sequences or cascading layers

    # model.add(Conv2D(32, (10,10), activation='relu', input_shape=input_shape,
                   # kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(32, (10,10),strides=(3,3), activation='relu', input_shape=input_shape,
                   kernel_initializer=h28_initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (7,7), activation='relu',
                     kernel_initializer=h28_initialize_weights,
                     bias_initializer=h28_initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (4,4), activation='relu', kernel_initializer=h28_initialize_weights,
                     bias_initializer=h28_initialize_bias, kernel_regularizer=l2(2e-4)))

    model.add(MaxPooling2D())
    # model.add(Conv2D(256, (4,4), activation='relu', kernel_initializer=initialize_weights,
                     # bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
 
 
    model.add(Flatten())
    # model.add(Dense(4096, activation='sigmoid',
    model.add(Dense(128, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=h28_initialize_weights,bias_initializer=h28_initialize_bias))
 
    # model.summary()
    # quit()#debug
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    # L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]), output_shape=input_shape)
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=h28_initialize_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    # siamese_net.summary()
    # quit()
    # return the model
    return siamese_net
    # return model

def check_type(vara):
    if isinstance(vara, np.ndarray):
        print(vara.shape, " is a numpy array")
        return
    if isinstance(vara, list):
        print(" is a list: ", len(vara))
        return
    return

def zoel_get_batch_ok001(batch_size,s="train", Xtrain=None,train_classes=None, Xval=None,val_classes=None):
    """Create batch of n pairs, half same class, half different class"""
    print ("###Function: zoel Get Batch()###")
    
    if (batch_size!=1):
        print ("test: batch_size must be 1 now=",batch_size)
        quit()
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h,rgb = X.shape
    print("n_classes:")
    check_type(n_classes)
    print("X:")
    check_type(X)
    
    # initialize 2 empty arrays for the input image batch
    pairs=[np.zeros((batch_size, h, w,rgb)) for i in range(2)]
    # pairs=[np.zeros((batch_size, h, w,1)) ]
    print("pairs type:")
    check_type(pairs)
    print("pairs len ",len(pairs))
    # initialize vector for the targets
    targets=np.zeros((batch_size,))
    
    batch_size=2
    N=batch_size

    #the rgb=3 is included in :
    test_image0 = np.asarray([X[0,0:N,:,:]]).reshape(N, w, h,rgb)
    # print( test_image0.shape)
    test_image1 = np.asarray([X[0,N:2*N,:,:]])
    # print( test_image1.shape)
    test_image1=test_image1.reshape(N, w, h,rgb)
    # quit()
    #ok for N=1
    #asarray() convert to numpy array, 
    #X is numpy array, but its element NOT, elemen of X need to be converted
    #using asarray()
    # test_image0 = np.asarray([X[0,0,:,:]]*N).reshape(N, w, h,1)
    # test_image1 = np.asarray([X[0,2,:,:]]*N).reshape(N, w, h,1)
    pairs[0] = [test_image0]
    pairs[1] = [test_image1]
    targets=np.ones((batch_size,))
    # quit()
    # pairs=X;targets=np.ones((1,))
    return [pairs,targets]


def h28_get_batch_predict_ok001(batch_size,s="train", Xtrain=None,train_classes=None, Xval=None,val_classes=None):
    """to call teras.predict() only twin/pair is passed without output
    """
    print ("###Function: zoel Get Batch for predictions()###")
    
    if (batch_size!=1):
        print ("test: batch_size must be 1 now=",batch_size)
        quit()
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h,rgb = X.shape
    print("n_classes:")
    check_type(n_classes)
    print("X:")
    check_type(X)
    
    # initialize 2 empty arrays for the input image batch
    pairs=[np.zeros((batch_size, h, w,rgb)) for i in range(2)]
    # pairs=[np.zeros((batch_size, h, w,1)) ]
    print("pairs type:")
    check_type(pairs)
    print("pairs len ",len(pairs))
    # initialize vector for the targets
    targets=np.zeros((batch_size,))
    
    batch_size=2
    N=batch_size

    #the rgb=3 is included in :
    test_image0 = np.asarray([X[0,0:N,:,:]]).reshape(N, w, h,rgb)
    # print( test_image0.shape)
    test_image1 = np.asarray([X[0,N:2*N,:,:]])
    # print( test_image1.shape)
    test_image1=test_image1.reshape(N, w, h,rgb)
    # quit()
    #ok for N=1
    #asarray() convert to numpy array, 
    #X is numpy array, but its element NOT, elemen of X need to be converted
    #using asarray()
    # test_image0 = np.asarray([X[0,0,:,:]]*N).reshape(N, w, h,1)
    # test_image1 = np.asarray([X[0,2,:,:]]*N).reshape(N, w, h,1)
    pairs[0] = [test_image0]
    pairs[1] = [test_image1]
    targets=np.ones((batch_size,))
    # quit()
    # pairs=X;targets=np.ones((1,))
    return [pairs]



def h28_get_batch_predict_001(batch_size,iteration=0, Xmix=None):
    """to call teras.predict() only twin/pair is passed without required (softmax) output
    NO RANDOM, just passing the FIRST batch_size. the first sequence is similairty=1
    the next sequences is similarity=0.
    data for first siames input in Xmix [0].
    data for next siamese network input in XMix [1].
    first half of Xmix 1 is  for similarity=1.
    last half of Xmix[1] is for similarity=0.
    """ 
    print ("###h28_get_batch_predict_001()###")
    
    if iteration!=0:
        print("iteration=", iteration, " not implemented")
        quit()
    X = Xmix
    
    # p_sh=X.shape;
    # print(p_sh)
     
    # list_begin=(1,2,1)
    # if p_sh[0:3]!=list_begin:
    #     print("no list begin, extpected: ", list_begin)
    #     print(p_sh)
    #     quit()


    n_classes, n_examples, w, h,rgb = X.shape

    if n_classes!=2:
        print("n_classes=",n_classes, " not implemented")
        quit()
    X = Xmix




    print("n_classes:")
    check_type(n_classes)
    print("X:")
    check_type(X)
    
    # initialize 2 empty arrays for the input image batch
    pairs=[np.zeros((batch_size, h, w,rgb)) for i in range(2)]
    # pairs=[np.zeros((batch_size, h, w,1)) ]
    print("pairs type:")
    check_type(pairs)
    print("pairs len ",len(pairs))
    # initialize vector for the targets
    # targets=np.zeros((batch_size,))
    
    # batch_size=2

    N=min(batch_size,n_examples)

    if N % 2 != 0:
        N=N-1

    if N <4:
        print("Example at least 4")
        quit()


    print("batch size approved:", N)

    # #the rgb=3 is included in :
    test_image0 = np.asarray([X[0,0:N,:,:]])
    test_image0=test_image0.reshape(N, w, h,rgb)
    print(" test_image0.shape ", test_image0.shape)
    # print( test_image1.shape)
    
    N_half=(np.floor((N/2))).astype("uint8")
    temp_arr = np.asarray([X[0,1:N_half+1,:,:]])
    temp_arr=temp_arr.reshape(N_half, w, h,rgb)
    print("temp_arr.shape ",temp_arr.shape)
    temp_arr=np.hstack((temp_arr,np.asarray([X[1,0:N_half,:,:]]).reshape(N_half, w, h,rgb   )))
    print("temp_arr.shape ",temp_arr.shape)
    temp_arr=temp_arr.reshape(N, w, h,rgb)
    print(temp_arr.shape)
    pairs[0] = [test_image0]
    pairs[1] = [temp_arr]


    # test only
    # #the rgb=3 is included in :
    # test_image0 = np.asarray([X[0,0:N,:,:]]).reshape(N, w, h,rgb)
    # # print( test_image0.shape)
    # test_image1 = np.asarray([X[0,N:2*N,:,:]])
    # # print( test_image1.shape)
    # test_image1=test_image1.reshape(N, w, h,rgb)
    # pairs[0] = [test_image0]
    # pairs[1] = [test_image1]

    return [pairs]



def zoel_get_batch(batch_size,s="train", Xtrain=None,train_classes=None, Xval=None,val_classes=None):
    """Create batch of n pairs, half same class, half different class"""
    print ("###Function: zoel Get Batch()###")
    
    if (batch_size!=1):
        print ("test: batch_size must be 1 now=",batch_size)
        quit()
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape
    print("n_classes:")
    check_type(n_classes); print("nclassess",n_classes)
    # print("X:") ;     check_type(X)
    print(X.shape)
    # initialize 2 empty arrays for the input image batch
    pairs=[np.zeros((batch_size, h, w,1)) for i in range(2)]
    # pairs=[np.zeros((batch_size, h, w,1)) ]
    print("pairs type:")
    check_type(pairs)
    print("pairs len ",len(pairs))
    # initialize vector for the targets
    targets=np.zeros((batch_size,))
    
    batch_size=5
    N=batch_size
    test_image0 = np.asarray([X[0,0:5,:,:]]).reshape(N, w, h,1)
    test_image1 = np.asarray([X[0,5:10,:,:]]).reshape(N, w, h,1)

    #ok for N=1
    #asarray() convert to numpy array, 
    #X is numpy array, but its element NOT, elemen of X need to be converted
    #using asarray()
    # test_image0 = np.asarray([X[0,0,:,:]]*N).reshape(N, w, h,1)
    # test_image1 = np.asarray([X[0,2,:,:]]*N).reshape(N, w, h,1)
    pairs[0] = [test_image0]
    pairs[1] = [test_image1]
    targets=np.ones((batch_size,))
    # quit()
    # pairs=X;targets=np.ones((1,))
    return [pairs,targets]

def zoel_get_batch_train(Xtrain=None,train_classes=None):
    '''
    using ALL available training data
    use permutation. for same class use targets value 1,
    for different class use targets value 0
    '''
    print("zoel_get_batch_Train")
    X=Xtrain
    n_classes, n_examples, w, h = X.shape
    print("X.shape",X.shape); 
    print("train_classes",train_classes); quit()
    return [pairs, targets]



def h28_get_batch(batch_size,s="train", Xtrain=None,train_classes=None, Xval=None,val_classes=None):
    """Create batch of n pairs, half same class, half different class"""
    print ("###Function: h28 Get Batch()###")
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape
    print ("X.shape")
    print (X.shape)
    # randomly sample several classes to use in the batch
    # categories = rng.choice(n_classes,size=(batch_size,),replace=False)
    categories = rng.choice(n_classes,size=(batch_size,),replace=True)
    
    # initialize 2 empty arrays for the input image batch
    pairs=[np.zeros((batch_size, h, w,1)) for i in range(2)]
    
    # initialize vector for the targets
    targets=np.zeros((batch_size,))
    
    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size//2:] = 1
    for i in range(batch_size):
        category = categories[i]
        # print("category");print(category)
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, 1)
        idx_2 = rng.randint(0, n_examples)
        
        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category  
        else: 
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1,n_classes)) % n_classes
        
        pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(w, h,1)
    
    return pairs, targets

def generate(batch_size, s="train"):
    """a generator for batches, so model.fit_generator can be used. """
    print ("###Function: Generate()###")
    while True:
        pairs, targets = get_batch(batch_size,s)
        yield (pairs, targets)

def make_oneshot_task(N, s="val", language=None,Xtrain=None,train_classes=None, Xval=None,val_classes=None):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    print ("###Function: make_oneshot_task()###")
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape
    
    indices = rng.randint(0, n_examples,size=(N,))
    if language is not None: # if language is specified, select characters for that language
        low, high = categories[language]
        print("low high");print (low) ;         print (high)
        if N > high - low:
            raise ValueError("This language ({}) has less than {} letters".format(language, N))
        # categories = rng.choice(range(low,high),size=(N,),replace=False)
        categories = rng.choice(range(low,high),size=(N,),replace=True)

    else: # if no language specified just pick a bunch of random letters
        # categories = rng.choice(range(n_classes),size=(N,),replace=False)            
        categories = rng.choice(range(n_classes),size=(N,),replace=True)            
    true_category = categories[0]
    # ex1, ex2 = rng.choice(n_examples,replace=False,size=(2,))
    ex1, ex2 = rng.choice(n_examples,replace=True,size=(2,))
    test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N, w, h,1)
    support_set = X[categories,indices,:,:]
    support_set[0,:,:] = X[true_category,ex2]
    support_set = support_set.reshape(N, w, h,1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)
    pairs = [test_image,support_set]

    return pairs, targets

def test_oneshot(model, N, k, s = "val", verbose = 0,Xtrain=None,train_classes=None, Xval=None,val_classes=None):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N,s,Xval=Xval, val_classes=val_classes,Xtrain=Xtrain,train_classes=train_classes)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct+=1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
    return percent_correct


def nearest_neighbour_correct(pairs,targets):
    """returns 1 if nearest neighbour gets the correct answer for a one-shot task
        given by (pairs, targets)"""
    L2_distances = np.zeros_like(targets)
    for i in range(len(targets)):
        L2_distances[i] = np.sum(np.sqrt(pairs[0][i]**2 - pairs[1][i]**2))
    if np.argmin(L2_distances) == np.argmax(targets):
        return 1
    return 0

def test_nn_accuracy(N_ways,n_trials,Xval=None, val_classes=None):
    """Returns accuracy of NN approach """
    # global Xval, val_classes
    print("Evaluating nearest neighbour on {} unique {} way one-shot learning tasks ...".format(n_trials,N_ways))

    n_right = 0
    
    for i in range(n_trials):
        pairs,targets = make_oneshot_task(N_ways,"val",Xval=Xval, val_classes=val_classes)
        correct = nearest_neighbour_correct(pairs,targets)
        n_right += correct
    return 100.0 * n_right / n_trials



def concat_images(X):
    """Concatenates a bunch of images into a big matrix for plotting purposes."""
    nc, h , w, _ = X.shape
    X = X.reshape(nc, h, w)
    n = np.ceil(np.sqrt(nc)).astype("int8")
    img = np.zeros((n*w,n*h))
    x = 0
    y = 0
    for example in range(nc):
        img[x*w:(x+1)*w,y*h:(y+1)*h] = X[example]
        y += 1
        if y >= n:
            y = 0
            x += 1
    return img


def plot_oneshot_task(pairs):
    fig,(ax1,ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.matshow(pairs[0][0].reshape(105,105), cmap='gray')
    img = concat_images(pairs[1])
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.matshow(img,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def describe_inputs_targets(inputs=None,targets=None):
    if not inputs is None:
        print("inputs is list not none")
        # pprint(inputs)
        if not isinstance(inputs,list):
            print ("list required as inputs")
            return
        else:
            print("input is list len=",len(inputs))
            print("inputs[0].shape",inputs[0].shape)

            # df=pd.DataFrame(inputs)
            # df.info()
    
    else:
        print("None inputs")
    if not targets is None:
        print("targets is not none")
        print(targets)
    else:
        print("None tergets")


def zoel_test_accuracy(targets=None,predicted=None):
    '''return percentage of correct(matched) between targets and predicted.
    None mean CANNOT None
    '''
    acc=0.0
    tarray=[]
    parray=[]
    if targets is None:
        print("targets can't be None")
        return
    if predicted is None:
        print("predicted can't be None")
        return
    if not isinstance(targets,np.ndarray):
        print("targets is not numpy array")
        # print(targets.shape)
        return
    if not isinstance(predicted,np.ndarray):
        # print("predicted is not numpy array")
        # print(predicted.shape)
        return
    maxlen=max(targets.shape)
    parray=np.rint(predicted.reshape(maxlen,1))
    tarray=targets.reshape(maxlen,1)
    
    # print(parray)
    # print(tarray)
    if(len(parray)!=len(targets)):
        print("need same size to compare")
        return
    

    rsub=parray==tarray
    # print(rsub)
    terr=(rsub==False).sum()
    acc=1-terr/maxlen
    # print("err", terr)
    return acc,terr
