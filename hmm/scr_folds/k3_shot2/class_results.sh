source env_cygwin.sh

#HRest Outputs
dir=hmm1
dir=hmm10_class #herest
phonems=phonems.lst
phonems_2d=phonems_2d_class.lst
# dict=phone1_2D.dic
dict=phone_class.dict
ftest=tmp_genobs/${project}_tb_2d_class.lst
outdir=results
 
for rec in `ls ${outdir}`
do

	# echo $rec
	# echo ${outdir}/$rec
	grepout=`grep fp_ ${outdir}/$rec`
	# echo ${grepout}
	size=${#grepout}
	if [ $size -gt 0 ]; then
		echo $rec fp_board 
	fi

	grepout=`grep bg_white ${outdir}/$rec`
	# echo ${grepout}
	size=${#grepout}
	if [ $size -gt 0 ]; then
		echo $rec student_paper 
	fi
	
	
done