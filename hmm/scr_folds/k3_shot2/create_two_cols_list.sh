source env_cygwin.sh

basedir="/cygdrive/g/rsync/RESEARCHS/finger_board_detection_image"
master_lab_dir=${basedir}/github_jurnal/hmm/master_labels/cat_labs

for x in `cat $1`; do 
	cat=`cat ${master_lab_dir}/${x}.lab`
	echo -e "${x}\t${cat}"
done
