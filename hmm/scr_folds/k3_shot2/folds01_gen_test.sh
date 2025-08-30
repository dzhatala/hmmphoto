#!/bin/sh
source ./env_cygwin.sh
#folds_gen_train.sh must be executed before
# num_k=3  #folds

all_list=114_two_cols_test_train_cat.lst

#every lines must be uncommented iszsn turn

usage="usage: folds_gen_test.sh all_list"

if [ "${all_list}" == "" ]; then
	echo "${usage}"
	kill -INT $$ #cygwin ok
	# return 1;
fi 


# lnum1=`wc -l ${f_label_train} | cut -f 1 -d " "`
lnum1=`wc -l ${all_list} | cut -f 1 -d " "`

# echo "gen_test_folds.sh   ${all_list} " #debug

for k in ${folds}; do
  echo "k value: $k"
  cp /dev/null "k${k}"/test.lst
  
done

#checking
SPACE=" "
mfc_dir="/cygdrive/c/rps/tmp/tmp_6/tmp"

# echo awk....
# echo "categs are $categs"
awk  -v output_dir="."  -v mfc_dir=${mfc_dir}  \
-v num_folds="${num_k}" -v categs="${categs}"   $SPACE '
BEGIN {
	# print "categs: " categs
	print "num_folds is  " num_folds

	# x=(getline <	"./k0/answer_sheet/train.lst")
	# print x	" " $0
	# getline <	"./k0/answer_sheet/train.lst"
	# print x	" " $0

	# exit
	
	FNUM=0.0;
	LNUM=0;
	# num_folds=1
	for (ik=0;ik<num_folds; ++ik) {
	# for (ik=2;ik<num_folds; ++ik) {
		print "fold "ik
		split(categs,carr," ")
		hash_table_trains[ik]=""
		num_train=0
		for (c = 1; c <= length(carr); c++) {
			print "categ " carr[c]
			ftrain="k" ik "/" carr[c] "/train.lst"
			print "reading " ftrain
			
			RCS=1 #first positive
			while(RCS>0) {
				RCS=(getline < ftrain)   #getline only read A SINGLE line
				print RCS " " $0 > /dev/null
				tmp=$0
				gsub("-", "",tmp);
				gsub("_", "",tmp);
				print "tmp: " tmp
				if(index(hash_table_trains[ik],tmp)<=0){ ## getline read double last entry
					hash_table_trains[ik] = hash_table_trains[ik] " " tmp
					num_train++
				}
			}
		}
		print "total_num_train: " num_train
		# print hash_table_trains[2]
		numtest[ik]=0.0
		
	}
	# print "FNUM " FNUM
	# exit #goto END{}
}	
	

{


	
		# print $1
		
		# num_folds=1#
			cleanLine=$1
			gsub("_","",cleanLine)
			gsub("-","",cleanLine)
			# print cleanLine
		for (ik=0;ik<num_folds; ++ik){
			
			# if(cleanLine=="280IMG20240101WA0052"){
				# print cleanLine
				# print hash_table_trains[ik]
				# exit;
			# }
			
			if(index(hash_table_trains[ik],cleanLine)>0){
				printf("#duplicate %s in %s at line %d for fold k=%d\r\n", $1, FILENAME,NR,ik)
			}else {
				out_list= "k" ik "/test.lst"
				 print mfc_dir "/" $1 ".mfc" >> out_list
				 numtest[ik]++;
			}
	
		}


	


}
END {
	for (ik=0; ik<num_folds;++ik){
		print "k" ik " num test: " numtest[ik]
	}
}

'  ${all_list}



if [ "${error}" != "" ]; then
	echo "error :${erros}"
	kill -INT $$ #cygwin ok
fi



