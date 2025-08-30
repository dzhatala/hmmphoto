currdir=pwd
addpath(pwd)
[num_folds,cm,s]=my_cm();

S_TP=0 %true positif
S_FP= 0%
S_TN = 0 % TRUE negatif
S_FN = 0% false negatif
% cf_mat=np.zeros((num_folds,2,2))

TP=0;
TN=0;
FN=0;
FP=0
for ik=1:num_folds
	% continue
	ik
    icm=cm(:,:,ik)
    % % print (icm);
    TP=icm(1,1); %print ("TP {}".format(TP))
    TN=icm(2,2); %print ("TN {}".format(TN))
    FN=icm(1,2); %print ("FP {}".format(FP))
    FP=icm(2,1); %print ("FN {}".format(FN))
    % cf_mat(k)=((TP, FN), (FP,TN))
    S_TP=S_TP+TP;
    S_TN=S_TN+TN;
    S_FP=S_FP+FP;
    S_FN=S_FN+FN;
	total_test=TP+TN+FP+FN
end
% avg_TP=S_TP/num_folds
% avg_TN=S_TN/num_folds
% avg_FP=S_FP/num_folds
% avg_FN=S_FN/num_folds

avg_TP=S_TP
avg_TN=S_TN
avg_FP=S_FP
avg_FN=S_FN

avg_CM=[avg_TP  avg_FN ; avg_FP , avg_TN]
total_test_K=sum(sum(avg_CM))
fprintf ("num folds: %i, num shots: %i\n",num_folds,s);
% print (avg_CM)
format long
precision_acc = avg_TP /(avg_TP+avg_FP)
% fprintf ("Precision/Accuracy: %.3f",precision)
recall = avg_TP /(avg_TP + avg_FN)
% fprintf ("Recal: %.3f",recall)

f1_score=2*precision_acc*recall/(precision_acc+recall)
% fprintf ("f1_score: %.3f",f1_score)
	