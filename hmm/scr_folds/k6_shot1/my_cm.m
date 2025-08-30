function [num_folds,cm,s]=my_cm()
num_folds=6
s=1 % num shots

fprintf ("k=%i s=%i\n",num_folds,s)

% return

% cm=[]
cm(:,:,1)= [56 0 ;  2 53];

cm(:,:,2)=[54 2 ; 0  55];

cm(:,:,3)=[ 56 0 ; 0  55];

cm(:,:,4)= [48 8 ;  1 54];
cm(:,:,5)= [3 53 ;  0 55];

cm(:,:,6)=[52 4 ; 0  55];

