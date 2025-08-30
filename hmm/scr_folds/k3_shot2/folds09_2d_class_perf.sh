source env_cygwin.sh
set -e

echo answer_sheet > cat_2dphonems.lst
echo fp_image >> cat_2dphonems.lst

phones=cat_2dphonems.lst


	# k=0
	#trace="-T 3"
	trace="-T 0"
	trace="-T 3"
for k in ${folds}; do
	echo "fold ${k}"
	basedir="/cygdrive/g/rsync/RESEARCHS/finger_board_detection_image"
	master_lab_dir=${basedir}/github_jurnal/hmm/master_labels/cat_labs

	rec_dir=k${k}/results/cats

	cmd1="$HTKTOOLS_DIR/HResults $trace -p -L ${master_lab_dir} ${phones} ${rec_dir}"
	cmdex="${cmd1}/*.recph"
	cmdecho="${cmd1}/\*.recph"
	echo $cmdecho
	eval $cmdex
	echo $cmdecho;
done

