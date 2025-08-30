set -e
source env_cygwin.sh
mkdir -p hinitoutput

init_file=init_6


# folds="0 1 2" #overide env_cygwin.sh
cmd=""
categs="answer_sheet fpboard"
trace_flag="-T 0"
for k in ${folds}; do
# for k in 2; do
echo -e "\t fold: $k "
	scp_hinit=k${k}/mfc_train_hinit.lst
	cp /dev/null ${scp_hinit}
	for c in ${categs}; do
		echo "Categ: ${c}"
		cp /dev/null "k${k}"/${c}/mfc_train.lst
		for prefix in `cat k${k}/${c}/train.lst`; do
			cat "${labelme_parsed}/${prefix}.lst" >> k${k}/${c}/mfc_train.lst
			
		done
		cat k${k}/${c}/mfc_train.lst >> ${scp_hinit}
		echo "k${k}/${c}/mfc_train.lst created "
		echo "${scp_hinit} created"
	done
	mkdir -p k${k}/hinitoutput 
	cmd="$HTKTOOLS_DIR/HInit  -C configtrain.txt $trace_flag -m 1 -M k${k}/hinitoutput $init_file -S ${scp_hinit}"
	echo $cmd; 
	eval $cmd
	echo $cmd ; 
	# break

done
