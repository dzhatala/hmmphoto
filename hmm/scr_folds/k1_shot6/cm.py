import numpy
cm={} #dynamic variable using dictionary
cm[0]=[ [56, 0], [ 2 ,53]]
# print (cm_k0)

cm[1]=[ [54, 2], [ 0 ,55]]
# print (cm_k1)

cm[2]=[ [56, 0], [ 0 ,55]]
# print (cm_k2)

cm[3]=[ [48, 8], [ 1 ,54]]
# print (cm_k3)

cm[4]=[ [3, 53], [ 0 ,55]]
# print (cm_k4)

cm[5]=[ [52, 4], [ 0 ,55]]
# print (cm_k5)

num_folds=6
s=1 # num shots

S_TP=0 #true positif
S_FP= 0#
S_TN = 0 # TRUE negatif
S_FN = 0# false negatif
# cf_mat=np.zeros((num_folds,2,2))

TP=0;
TN=0;
FN=0;
FP=0
for ik in range(num_folds):
    icm=numpy.asarray(cm[ik])
    # print (icm);
    TP=icm[0,0]; #print ("TP {}".format(TP))
    TN=icm[1,1]; #print ("TN {}".format(TN))
    FN=icm[0,1]; #print ("FP {}".format(FP))
    FP=icm[1,0]; #print ("FN {}".format(FN))
    # cf_mat[k]=[[TP, FN], [FP,TN]]
    S_TP=S_TP+TP;
    S_TN=S_TN+TN
    S_FP=S_FP+FP
    S_FN=S_FN+FN

# avg_TP=S_TP/num_folds
# avg_TN=S_TN/num_folds
# avg_FP=S_FP/num_folds
# avg_FN=S_FN/num_folds

avg_TP=S_TP
avg_TN=S_TN
avg_FP=S_FP
avg_FN=S_FN

avg_CM=[[avg_TP , avg_FN] , [avg_FP , avg_TN]]
print (avg_CM)
precision = avg_𝑇𝑃 /(avg_TP+avg_FP)
print ("Precision: {}".format(precision))
recall = avg_𝑇𝑃 /(avg_𝑇𝑃 + avg_𝐹𝑁)
print ("Recal: {}".format(recall))

f1_score=2*precision*recall/(precision+recall)
print ("f1_score: {}".format(f1_score))
