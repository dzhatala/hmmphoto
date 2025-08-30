#HTKTOOLS_DIR="../../../small_vocabulary_/HTK-3.4.1_pure/htk/HTKTools"
#HTKTOOLS_DIR="/cygdrive/f/rsync/RESEARCHS/small_vocabulary_/HTK-3.4.1_pure/htk/HTKTools"
#HTKTOOLS_DIR="../HTKTools_x86"
#HTKTOOLS_DIR="/home/joesmart/RESEARCHS/htk/htk/HTKTools"
#HTKTOOLS_DIR="/home/joesmart/table_det/HTKTools"
# HTKTOOLS_DIR=~/gRESEARCHS/table_detection/HTKTools_x64
HTKTOOLS_DIR=~/gRes/htk_cygwin/HTK-3.4.1/htk/HTKTools
datadir=./tmp_genobs
labdir=${datadir}
project=k6_shots1

num_k=2 #k fold
folds="0 1 " #overide env_cygwin.sh

num_s=3 # s shots

categs="answer_sheet fpboard"
labelme_parsed=/cygdrive/c/rps/opencv/tmp/scp_mlfs #output of 
