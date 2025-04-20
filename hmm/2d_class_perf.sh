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
basedir="."
master_lab_dir=${basedir}/master_labs

rec_dir=./results/cats

cmd="$HTKTOOLS_DIR/HResults $trace -p -L ${master_lab_dir} ${phones} ${rec_dir}/*.rec"
cmd_star="$HTKTOOLS_DIR/HResults $trace -p -L ${master_lab_dir} ${phones} ${rec_dir}/\	*.rec"
echo $cmd_star
eval $cmd
