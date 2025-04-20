#parse .rec contain rows and create its HTK format cat. lab file


recsdir="./results"
outlabdir=${recsdir}/cats

mkdir -p ${outlabdir}
for x in `ls ${recsdir}/*.rec`
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
