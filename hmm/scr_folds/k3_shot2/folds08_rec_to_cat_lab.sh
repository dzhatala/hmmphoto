#parse .rec contain rows and create its HTK format cat. lab file

source env_cygwin.sh

for k in ${folds}; do

	recsdir="k${k}/results"
	outlabdir=${recsdir}/cats


	echo "from ${recsdir} to ${outlabdir} ?  ENTER | CTRL_C" ;read
	mkdir -p ${outlabdir}
	for x in `ls ${recsdir}/*.recph`
	# for x in `ls ${recsdir}/*113.rec`
	do
		
		
		base_fn=`basename $x`
		fp_image=`grep fp_screen $x`
		answer_sheet=`grep bg_white $x`
		
		if [ "${fp_image}" == "" ]; then
			echo "${base_fn} NOT fp_image"
		else
			echo "${base_fn} IS fp_image"
			echo fp_image > ${outlabdir}/${base_fn}
		fi
		
		if [ "${answer_sheet}" == "" ]; then
			echo "${base_fn} NOT answer_sheet"
		else
			echo "${base_fn} IS answer_sheet"
			echo answer_sheet > ${outlabdir}/${base_fn}
			
		fi

	done
	echo "from rec dir. ${recsdir}"

done