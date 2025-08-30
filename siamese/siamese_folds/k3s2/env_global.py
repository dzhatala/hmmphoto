num_folds=3
num_shots=2


#another project
# num_folds=6
# num_shots=1

#another project
# num_folds=2
# num_shots=3


num_folds=1
num_shots=4

num_folds=1
num_shots=5

num_folds=1
num_shots=6


project_name="./k{}_shot{}".format(num_folds,num_shots) # the relative dir
htk_folds_dir="z:\\{}".format(project_name)  #via samba htk hmm
smaller_dir="y:"  #samba based
categs=["fp_image","answer_sheet"] # the first is indexed 0
