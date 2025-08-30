sync
source env_cygwin.sh
HVITE=$HTKTOOLS_DIR/HVite

echo "Isolated training with HRest"

# k="0"

trace="-T 0"
set -e
for k in ${folds}; do
# for k in 0; do
	echo "k: $k" >&2
	dir_src=k${k}/hmm0
	dir_tgt=k${k}/hmm1

	prune_param=" -t 50.0 150.0 1000.0"
	mkdir -p ${dir_tgt}
	echo creating k${k}/hmms in k${k}/hmm1 ... 
	counter=1

	numiter=50


	echo "#!MLF!#" > k${k}/embedded_mfc_train.mlf #moved to HRest version
	cp /dev/null k${k}/embedded_mfc_train.lst
	for c in $categs; do
		echo "#!MLF!#" > k${k}/${c}/mfc_train.mlf #moved to HRest version
		for prefix in `cat k${k}/${c}/train.lst`; do
			cat "${labelme_parsed}/${prefix}.mlf" >> k${k}/${c}/mfc_train.mlf #moved to HRest version
			
			cat "${labelme_parsed}/${prefix}.mlf" >> k${k}/embedded_mfc_train.mlf #later used by embedded training
			cat k${k}/${c}/mfc_train.lst >> k${k}/embedded_mfc_train.lst
		done
	done

	rm -f k${k}/hmm1/*

	for c in $categs; do
		echo -e "\tcateg: $c"
		# continue

		for x in `cat categs/${c}/phonems_2d_class_train.lst`
		do
			scp=k${k}/${c}/mfc_train.lst
			mlfs="-I k${k}/${c}/mfc_train.mlf"
			cmd="$HTKTOOLS_DIR/HRest -C configtrain.txt -i ${numiter} -S ${scp} $trace   ${mlfs}  \
			   -M ${dir_tgt} -l $x ${dir_src}/$x"
			#cmd="$HTKTOOLS_DIR/HRest "
			echo $cmd ; echo "ENTER ? [CTRL+c] "; read; 
			eval $cmd || { kill -INT $$; }
			if [ "$counter" == "1" ]; then 
				if [ -f k${k}/hmm1/$x ]; then
					#cp /dev/null k${k}/hmm0/k${k}/hmmdefs
					cp k${k}/hmm1/$x k${k}/hmm1/hmmdefs
					echo "##ALL"
				fi
			else
				if [ -f k${k}/hmm1/$x ]; then
					tail -n +4 k${k}/hmm1/$x >>  k${k}/hmm1/hmmdefs
					echo "##head ignored"
				else
					echo "NOT found k${k}/hmm1/${x} "
						break;
				fi
			fi
			echo "counter: $counter $x"
			counter=`expr $counter + 1`; 

			echo [ENTER! CONT ]  CTRL+C EXIT!
			read 
		done
		echo "output at: ${dir_tgt}"
	done
	# break #uncomment to exit
done