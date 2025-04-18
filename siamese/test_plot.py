from x270_lib_03 import *
# test_plot
# uncomment to test
# base_dir_cat="g:\\rsync\\RESEARCHS\\finger_board_detection_image\\github_jurnal"
# htk_train="g:\\rsync\\RESEARCHS\\finger_board_detection_image\scripts\\python\\finger_board\\cat_train.txt"
# htk_test="g:\\rsync\\RESEARCHS\\finger_board_detection_image\scripts\\python\\finger_board\\cat_test.txt"
base_dir_cat="d:\\rsync\\RESEARCHS\\finger_board_det\\github_jurnal\\siamese"
htk_train="d:\\rsync\\RESEARCHS\\finger_board_det\\python\\finger_board\\cat_train.txt"
htk_test="d:\\rsync\\RESEARCHS\\finger_board_det\\python\\finger_board\\cat_test.txt"
cat=["fp_image","answer_sheet"]

# inputs,targets=init_journal_train(base_dir_cat,htk_train,cat) # test without tensorflow

catdir=["",""]
for i in range(2):
    catdir[i]="{}\\data\\smaller\\{}".format(base_dir_cat,cat[i])

cat_ptrain=create_filepath_cat_from_htk_list(htk_train,catdir)
cat_ptest=create_filepath_cat_from_htk_list(htk_test,catdir)

#https://stackoverflow.com/questions/29762972/image-height-and-width-getting-swapped-when-read-using-opencv-imread
# https://stackoverflow.com/questions/29762972/image-height-and-width-getting-swapped-when-read-using-opencv-imread
imiov2_size=(498,280,3) #imageio io : w,h is reversed
input,target=create_batch_test(imiov2_size,cat_ptrain,cat_ptest)
plot_input_target(input,target,8,2)
