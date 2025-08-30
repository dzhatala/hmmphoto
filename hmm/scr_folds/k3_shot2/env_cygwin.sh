#HTKTOOLS_DIR="../../../small_vocabulary_/HTK-3.4.1_pure/htk/HTKTools"
#HTKTOOLS_DIR="/cygdrive/f/rsync/RESEARCHS/small_vocabulary_/HTK-3.4.1_pure/htk/HTKTools"
#HTKTOOLS_DIR="../HTKTools_x86"
#HTKTOOLS_DIR="/home/joesmart/RESEARCHS/htk/htk/HTKTools"
#HTKTOOLS_DIR="/home/joesmart/table_det/HTKTools"
# HTKTOOLS_DIR=~/gRESEARCHS/table_detection/HTKTools_x64
HTKTOOLS_DIR=~/gRes/htk_cygwin/HTK-3.4.1/htk/HTKTools
datadir=./tmp_genobs
labdir=${datadir}

num_s=2 # s shots
num_k=3 #k fold
project=k${num_k}_shot${num_s}
folds="0 1 2" #overide env_cygwin.sh
categs="fpboard answer_sheet "
labelme_parsed=/cygdrive/c/rps/opencv/tmp/scp_mlfs #output of 
