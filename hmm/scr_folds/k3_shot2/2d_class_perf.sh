source env_cygwin.sh

# cmd="cp phonems.lst monophones1"
# echo $cmd ; eval $cmd
# mlfs=./tmp_genobs/${project}.mlf

echo answer_sheet > cat_2dphonems.lst
echo fp_image >> cat_2dphonems.lst

phones=cat_2dphonems.lst
# cat ${phones}

#cmd="$HTKTOOLS_DIR/HResults -e NNN sil  -p -L $labdir monophones1 recph.mlf"
# cmd="$HTKTOOLS_DIR/HResults -p -I ${mlfs} ${phones} recph.mlf"
# cmd="$HTKTOOLS_DIR/HResults -p -I ${mlfs} ${phones} recph.mlf"

#trace="-T 3"
trace="-T 0"
trace="-T 15"
basedir="/cygdrive/g/rsync/RESEARCHS/finger_board_detection_image"
master_lab_dir=${basedir}/github_jurnal/hmm/master_labels/cat_labs

rec_dir=./results/cats

cmd1="$HTKTOOLS_DIR/HResults $trace -p -L ${master_lab_dir} ${phones} ${rec_dir}"
# cmd1="$HTKTOOLS_DIR/HResults $trace -p -L ${master_lab_dir} ${phones} "
cmdex="${cmd1}/*.recph"
cmdecho="${cmd1}\/\*.recph"
echo $cmdecho
eval $cmdex
echo $cmdecho;
