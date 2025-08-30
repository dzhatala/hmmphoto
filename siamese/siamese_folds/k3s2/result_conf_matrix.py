#result_conf_matrix.py
from env_global import * #import global variables
import pickle, numpy as np
num_folds=3 # for testing/sub simulation  only


S_TP=0 #true positif
S_FP= 0#
S_TN = 0 # TRUE negatif
S_FN = 0# false negatif
for k in range(num_folds):
    resdir="{}/k{}/results".format(project_name,k)
    with open('{}/nrec_outputs.pickle'.format(resdir), 'rb') as handle:
        outputs=pickle.load(handle)
        # print(targets)
        # handle.close()  
    with open('{}/nrec_results.pickle'.format(resdir), 'rb') as handle:
        rec_results=pickle.load(handle)
        # print(rec_results)
        handle.close()
    with open('{}/ncats.pickle'.format(resdir), 'rb') as handle:
        cats_target=pickle.load( handle)
        # print(cats_target)
        handle.close()

    with open('{}/npath_fns.pickle'.format(resdir), 'rb') as handle:
        path_fns=pickle.load(handle)
        # print(path_fns)
        handle.close()
        
    value_to_find = False
    indices = np.where(rec_results == value_to_find)[0]
    print(f"Indices of {value_to_find}: {indices}")

    print ("\nfiles/path")
    # for i in indices:
    for i in range(2):  #for testing only
        print(path_fns[i])
    # [[1, 'y:/answer_sheet\\24-01\\280_IMG-20240101-WA0136.jpeg'], [0, 'y:/fp_image\\24-01\\280_IMG-20240102-WA0000.jpeg']]
    # [[1, 'y:/answer_sheet\\24-01\\280_IMG-20240101-WA0152.jpeg'], [0, 'y:/fp_image\\24-01\\280_IMG-20240103-WA0000.jpeg']]

    print("\ntwo inputs and its TRUE target")
    # for i in indices:
    for i in range(2):  #for testing only
        print(cats_target[i])
    # [1, 0, 0]
    # [1, 0, 0]

    print("\npredicted similarity output/results")
    # print(outputs[indices])  #outpus that are wrong ..
    for i in range(2):
        print(outputs[i])  #outpus that are wrong ..

    # [[0.60199124]
     # [0.60199124]

    from sklearn.metrics import confusion_matrix, accuracy_score

    input2_target_arr=np.array(cats_target) #convert list to numpy array
    input2_target_arr=input2_target_arr[:,2].astype(int) #transpose the third columns
    int_outputs=np.rint(outputs).astype(int) #rint output still floating point
    # print (int_outputs.shape)
    # print(input2_target_arr.shape)

    print("\nfold k={}".format(k))
    cm = confusion_matrix(input2_target_arr, int_outputs)
    print(cm)
    TP=cm[0,0]; #print ("TP {}".format(TP))
    TN=cm[1,1]; #print ("TN {}".format(TN))
    FN=cm[0,1]; #print ("FP {}".format(FP))
    FP=cm[1,0]; #print ("FN {}".format(FN))
    S_TP=S_TP+TP;
    S_TN=S_TN+TN
    S_FP=S_FP+FP
    S_FN=S_FN+FN
    
avg_TP=S_TP/num_folds
avg_TN=S_TN/num_folds
avg_FP=S_FP/num_folds
avg_FN=S_FN/num_folds

avg_CM=[[avg_TP , avg_FN] , [avg_FP , avg_TN]]
print (avg_CM)
precision = avg_𝑇𝑃 /(avg_TP+avg_FP)
print ("Precision: {}".format(precision))
recall = avg_𝑇𝑃 /(avg_𝑇𝑃 + avg_𝐹𝑁)
print ("Recal: {}".format(recall))

f1_score=2*precision*recall/(precision+recall)
print ("f1_score: {}".format(f1_score))

recall_P=recall
recall_N=avg_𝑇N /(avg_𝑇N + avg_𝐹P)
# AUC= 0,5*recall_P*recall_N #impossible to get 1, maks is 0,5
AUC= recall_P*recall_N #impossible to get 1, maks is 0,5
print ("AUC: {}".format(AUC))

