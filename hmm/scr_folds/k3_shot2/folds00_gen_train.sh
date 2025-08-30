#!/bin/sh
source ./env_cygwin.sh
all_list=114_two_cols_test_train_cat.lst

#every lines must be uncommented in turn

#k only no categs
k=0
for k in ${folds}; do  #do not forget folds= in your env_cywin.sh
# while (( k < num_k )); do
  # echo "k value: $k"
  
  for cat in ${categs}; do
	echo "making dir k${k}/${cat}" 
	mkdir -p "k${k}/${cat}"
  done
  
  cp /dev/null "k${k}"/siamese_train.lst #master
  # cp /dev/null "k${k}"/mfc_train.lst # moved to baum_viterbi_init
  ((k++)) # Increment i
  echo "k${k}/siamese_train.lst created"
done

	#k categ first, k next level
	for categ in $categs; do 
		echo $categ
		if [ "$categ" == "answer_sheet" ]; then
		
			f_label_train=train_cat_answer_sheet.lst #; categ=answer_sheet
		else 
			f_label_train=train_cat_fpboard.lst ; #categ=fpboard
		fi

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

		lnum1=`wc -l ${f_label_train} | cut -f 1 -d " "`
		lnum2=`wc -l ${all_list} | cut -f 1 -d " "`


		k=0
		for k in ${folds}; do  #do not forget folds= in your env_cywin.sh
		# while (( k < num_k )); do
		  echo "k value: $k"
		  
		  mkdir -p "k${k}/${categ}"
		  cp /dev/null "k${k}"/${categ}/train.lst #master
		  # cp /dev/null "k${k}"/mfc_train.lst # moved to baum_viterbi_init
# qs		  ((k++)) # Increment i
		  
		done

		#checking
		all_trains=`cat ${f_label_train}`
		SPACE=" "
		mfc_dir="/cygdrive/c/rps/tmp/tmp_6/tmp"
		awk  -v output_dir="." -v k="1" -v mfc_dir=${mfc_dir} -v num_shots="${num_s}" \
		-v num_folds="${num_k}" -v categ="${categ}" -v lnum1="${lnum1}" -v lnum2="${lnum2}" $SPACE '

		function floor(num) {
			if (num >= 0) {
				return int(num)
			} else {
				return int(num) == num ? num : int(num) - 1
			}
		}

		BEGIN {
			# print output_dir
			# print mfc_dir
			# out_test_mfc =output_dir "/k" k "/test.lst"
			# out_train_mfc =output_dir "/k" k "/test.lst"

			num_folds=num_folds+0.0
			num_shots=num_shots+0.0
			print "categ: " categ
			print "num_shots is  " num_shots
			print "num_folds is  " num_folds
			print "lnum1 (train) " lnum1
			# print "lnum2 (test)" lnum2

			req_train_line=num_shots *num_folds
			 if(req_train_line>lnum1){
				print "invalid train numbers " lnum1 " != "  num_folds "*" num_shots 
				exit
			 }
			
			
			
			FNUM=0.0;
			LNUM=0;
			
			for (ik=0;ik<num_folds; ik=++ik) {
				hash_table_trains	[ik]= "" #hash contain all trains
			}
			print "FNUM " FNUM
			
		}

		{

			
			# print "CURRN FNUM " FNUM ", FNR: " FNR
			
			#round robin create folds
			#divide trains first into k shots
			k=floor((FNR-1)/num_shots);  #????
			if(k>=num_folds){
				exit
			}
			print "k for folds is " k
			out_list = "k" k "/" categ "/train.lst"
			print out_list
			print $1  >> out_list
			hash_table_trains[k] = hash_table_trains[k]  $1   "\n" 
			# print hash_table_trains[k]
			print  "add " $1 " into " outlist
			
			
			#siamese list
			out_list= "k" k "/" "/siamese_train.lst"
			print $1  >> out_list
				

			if(length($0)>1)  #windows files has \r
				LNUM=LNUM+1.0;
			# print FILENAME " " FNUM

		}

		' ${f_label_train} 

		echo ""
		echo "cat: $categ is done, you must repeat for another categories">&2

	done



# done

if [ "${error}" != "" ]; then
	echo "error :${erros}"
	kill -INT $$ #cygwin ok
fi


			# kill -INT $$ #cygwin ok
