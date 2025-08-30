#!/bin/sh
num_k=3  #folds
num_s=2  #shots

all_list=114_two_cols_test_train_cat.lst
f_label_train=train_cat_answer_sheet.lst
# f_label_test=

k=0
#!/bin/bash
k=0

usage="usage: gen_train_test_folds.sh all_list train_list"


if [ "${all_list}" == "" ]; then
	echo "${usage}"
	kill -INT $$ #cygwin ok
	# return 1;
fi 

if [ "${f_label_train}" == "" ]; then
	echo "${usage}"
	kill -INT $$ #cygwin ok
	# return 1;
fi 

echo "gen_train_test_folds.sh ${all_list} ${f_label_train}" #debug

while (( k < num_k )); do
  echo "k value: $k"
  
  mkdir -p "gen_folds/k${k}"
  
  ((k++)) # Increment i
  
done

#checking
all_trains=`cat ${f_label_train}`
SPACE=" "
mfc_dir="/cygdrive/c/rps/tmp/tmp_6/tmp"
awk -v all_trains="$all_trains" -v output_dir="." -v k="1" -v mfc_dir=${mfc_dir} $SPACE '  
BEGIN {
	# print all_trains
	# print output_dir
	out_mfc =output_dir "/k" k "/test.lst"
}
{
	if(index(all_trains,$1)>0){
		printf("#duplicate %s in %s at line %d\r\n", $1, FILENAME,NR)
	}else {
		print mfc_dir "/" $1 ".mfc" >> out_mfc
	}
	#print $1
}
' ${all_list}


# for x in `cat ${all_list}| cut -f1`; do
	# echo ${x} #must there is a line
	# found=`grep $x ${all_list}`
	
	# if [ "${found}" == "" ]; then
		# echo "remove1 ${x} from1 ${all_list}"
		# error="duplicate"
		# echo $x
	# fi
	
# done

if [ "${error}" != "" ]; then
	echo "error :${erros}"
	kill -INT $$ #cygwin ok
fi

	kill -INT $$ #cygwin ok


while (( k < num_k )); do
  echo "k value: $k"
  
  # mkdir -p
  
  ((k++)) # Increment i
  
done
