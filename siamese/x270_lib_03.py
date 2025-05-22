#!/usr/bin/env python3
import os, datetime, numpy as np
from pathlib import Path
from imageio.v2 import imread
from itertools import combinations
# from raptor_hambla28 import crop_im   age # heavy load tensorflow
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

def convert(date_time):  
     '''
     convert date from htk format
     '''
    #  format = '%b %d %Y %I:%M%p'   
     format = '%Y%m'   
     datetime_str = datetime.datetime.strptime(date_time, format)  
     return datetime_str

def create_filepath_cat_from_htk_list(htk_list,real_dir):
    '''
    make real file path and it's category.
    
    its assumed images are kept in dir structure
        smaller/cat1/yy-mm1/xxxx.jpeg
                    /yy-mm2/xxxx.jpeg
                    
    htk_list : HTK style list file for training or testing
    real_dir : array of full  path of image /classes/categories
    return : list of file path and its category index number of real_dir
    as input to siamese network

    return array/list of (dir_index, image_fullpath)
            first dir/category is assigned number 0
    '''
    print("Reading  {} ..".format(htk_list))
    retlist=[]
    fileh=open(htk_list,"r")
    lines=fileh.readlines()
    count=0
    total_found=0
    for line in lines:
        # print("{}. {}".format(count,line))
        basen = line.split('/')[-1].split('.')[0] #basename
        # print(basen)
        datestr=basen[8:14]
        # print(datestr)
        # break
        dtm=convert(datestr)
        # print(dtm) ; break
        monthdir=dtm.strftime("%y-%m")
        # print (monthdir) ; break
        # monthdir="to extract from basen"  #24-01
        found=False
        dir_idx=0
        for testdir in real_dir:
            f1="{}\\{}\\{}.jpeg".format(testdir,monthdir,basen)
            # print(f1)
            if os.path.isfile(f1):
                found=True
                total_found+=1
                # print ("found {} ".format(f1))
                retlist.append([dir_idx,f1])
                break
            dir_idx+=1
        
        if not found:
            print("{} not found in all dirs.".format(f1))
            print(real_dir)
            exit()

        count+=1

    print("Found {} of {} ".format(total_found,count))

    return retlist

def crop_image_light(image,req_size=None):
    '''
    crop center based on req_size
    https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    '''
    # print("cropping image into =>", req_size)
    if not req_size is None:
        # print(image.shape,req_size)
        if(image.shape==req_size):
            # print("size already match, no crop")
            return image
        print("fix here #0003: img size is ", image.shape," req is: ",req_size) ; quit()    
    return image


def create_batch_train(req_img_size, list_path_cat,batch_no=-1):
    '''
    
    load img in list_path_cat
    perform combinatorial
    return batch of siamese network based on list and cat. return combination
    return full combination if batch_no=-1
    crop image left and right
    return error if height is higher
    
    return input , target to train siamese
    '''

    #load_imad

    image_catl=[]
    countimg=0
    dict_path_img=dict()
    for path in list_path_cat:
        image_pre = imread(path[1])  #dimension w,h is reverse from windows explorer
        image=crop_image_light(image_pre,req_img_size) #req_size
        image_info=[countimg,path[1],path[0],image]   #number path category/class matrix(490,280,3)
        image_catl.append(image_info)
        dict_path_img[path[1]]=image_info

        countimg+=1

    # print("image number are : {}".format(len(image_catl)))

    # for img in image_catl:
    #     print("{}".format(img[0]))

    ##create cNr then check if the same car set 0 for output of siamese

    comb2 = combinations(list_path_cat, r=2)
    counter=0
    input0=[]; input1=[]
    target=[]
    for path in comb2:
        # print("comb ",counter)
        # print("\tpath",path)
        pathinfo0=path[0]
        pathinfo1=path[1]
        image_info0=dict_path_img[pathinfo0[1]]
        image_info1=dict_path_img[pathinfo1[1]]
        # print("\timg number: {}   vs {}".format(image_info0[0],image_info1[0]))
        cat1=image_info0[2]; cat2=image_info1[2]
        rxor= cat1^cat2
        rxor =rxor ^1
        # rxor=!rxor
        # print("\tcat: {} not xor {} = {} ".format(cat1,cat2,rxor))
        input0.append(image_info0[3])
        input1.append(image_info1[3])   #twin image
        target.append(rxor)

        counter+=1
    target=np.asarray(target)
    # print(target.shape)
    input=[]
    input0=np.asarray(input0)
    input.append(input0)
    input1=np.asarray(input1)
    input.append(input1)
    # input[np.asarray(input0),np.asarray(input1)]
    print(len(input))
    print(input[0].shape)
    return input,target

last_batch_test_no=0 #global var ?
last_batch_train_no=0 #global var ?

def create_batch_test(req_img_size, list_path_cat_train
    ,list_path_cat_test,batch_size=None,batch_no=None):
    '''
    same as create_batch_train() except:

    the test is put in first input, the trains(master) is put in second input. if test samples
        are larger than master, just rotate the master, so all master are tested fairly.
        
    return inputs (list that contain 2 ndarrays) and target
    
    if batch_size not specified , use ALL sample
    if batch_no is not specified, use no 0
        
    '''

    #load_image

    image_catl=[] # for another trainer/master rotation algorithm
    countimg=0
    dict_path_img=dict()
    # print(list_path_cat_train)
    for path in list_path_cat_train:
        # print("try reading : ",path[1])
        image_pre = imread(path[1])
        # print(image_pre.shape)
        image=crop_image_light(image_pre,req_img_size) #req_size
        image_info=[countimg,path[1],path[0],image]   #number path category/class matrix(490,280,3)
        # image_catl.append(image_info)
        dict_path_img[path[1]]=image_info
        countimg+=1

    train_len=countimg
    print("image train number are : {}".format(train_len))
    # print (image_catl)
    
    input0=[]; input1=[]
    target=[]
    train_idx=0
    test_idx=0
    paths_fn=[]
    cats=[]
    for test_path in list_path_cat_test:
        # print("try reading : ",test_path[1])
        image_pre = imread(test_path[1])
        image=crop_image_light(image_pre,req_img_size) #req_size
        #
        pathinfo0=test_path
        pathinfo1=list_path_cat_train[train_idx]
        paths_fn.append([pathinfo0,pathinfo1])
        image_info1=dict_path_img[pathinfo1[1]]
        # print("\timg number: {}   vs {}".format(image_info0[0],image_info1[0]))
        
        cat1=test_path[0]
        cat2=image_info1[2]
        rxor= cat1^cat2
        rxor =rxor ^1
        # rxor=!rxor
        # print("\tcat: {} not xor {} = {} ".format(cat1,cat2,rxor))

        input0.append(image)
        input1.append(image_info1[3])   #twin image
        target.append(rxor)
        cats.append([cat1,cat2,rxor])

       
        train_idx+=1
        if train_idx==train_len:
            train_idx=0 #rotating train/master data
        test_idx+=1

    target=np.asarray(target)
    # print(target.shape)
    input=[]
    input0=np.asarray(input0)
    input.append(input0)
    input1=np.asarray(input1)
    input.append(input1)
    # print(len(input))
    # print(input[0].shape)

        
    return input,target,cats,paths_fn

def init_journal_train(base_p=None, ftrain=None,cat=None):
    '''
    base_p : base directory for images must 'contain data/smaller'
    ftrain : htk train list filename
    cat : category filters and sorted under data/smaller
    '''
    # print ("ftrain:" , ftrain)


    # cat=["fp_image","answer_sheet"]
    catdir=["",""]
    for i in range(2):
        catdir[i]="{}\\data\\smaller\\{}".format(base_p,cat[i])

    # print (catdir)
    # ret=create_filepath_cat_from_htk_list(ftest,catdir)
    # print("finish reading {}".format(ftest))

    ret=create_filepath_cat_from_htk_list(ftrain,catdir)
    print("finish reading {} len={}".format(ftrain,len(ret)))
    # print("ret len {}".format(len(ret)))
    # print(ret)
    (input,target)=create_batch_train((498,280,3),ret)
    # print (input)
    return input,target

def concat_images(pairs_list):
    """Concatenates a bunch of images into a big matrix for plotting purposes.
    X : ndarray on shame nc,h,w,c
    
    """
    X=pairs_list[0][0]
    Y=pairs_list[1][0]

        #todo check sampe height here
    img=np.concatenate((X, Y), axis=1)   #horizontal
    
    return img


def plot_input_target(inputs,targets,max_pair=4,plot_num=0, num_col=2,start_idx=0):
    
    '''
    

    plot siamese twin inputs in single subplot, and render number (target, prediction results)
    
    
    can be use to plot (input_pair,predicted) and (input_pair,trained), so 
    we can visualized the prediction or training

    inputs: list of input image pairs into siamese arch.
    targets: target for training, or prediction results from predict()
    max_pair : maximum pairs to show on single plot
    plot_no : 0 default, if >0 forward max_pairt *plot_no
    
    num_col : column to create using subplot.
    
    '''
    
    font_size = 150
    font = ImageFont.truetype("arial.ttf", font_size) #replace "arial.ttf" with your font file if needed
    nrow=int(max_pair/num_col)
    # fig,(ax1,ax2) = plt.subplots(nrow, num_col)
    fig,axes = plt.subplots(nrow, num_col)
    print(axes.shape)
    
    for r in range(nrow):
        for c in range(num_col):
            X=inputs[0][start_idx]
            Y=inputs[1][start_idx]
            text=str(targets[start_idx])

            img=np.concatenate((X, Y), axis=1)   #horizontal
            img=Image.fromarray(img)
            draw = ImageDraw.Draw(img)

            text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
         
            image_width, image_height = img.size
            x = (image_width - text_width) / 2
            y = (image_height - text_height) / 2
            draw.text((x, y), text, font=font, fill=(255, 255, 255))  # White text
            draw = ImageDraw.Draw(img )
            axes[r][c].matshow(img,cmap='gray')
            start_idx+=1
    # img=inputs[0][0]
    # ax1.get_yaxis().set_visible(False)
    # ax1.get_xaxis().set_visible(False)
    # ax2.matshow(img,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()


    
    return
    

