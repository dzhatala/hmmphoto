function [num_folds,cm,s]=my_cm()
num_folds=3
s=2 % num shots


% cm=[]
cm(:,:,1)= [54 1 ;  13 41];

cm(:,:,2)=[51 4 ; 0  54];

cm(:,:,3)=[ 30 25 ; 0  54];
