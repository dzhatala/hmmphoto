# previous steps ?
# labelme\categs> ..\labelme.bat waxxxxx.json
#  /cygdrive/c/rps/opencv/mlp.sh 

#set -e #exit immediately

for cmd in ../k3_shot2/folds00_gen_train.sh \
../k3_shot2/folds01_gen_test.sh \
../k3_shot2/folds02_baum_viterbi_init_cyg.sh \
../k3_shot2/folds03_makehmm_cyg.sh \
../k3_shot2/folds04_train_hmm1_cyg.sh \
../k3_shot2/folds05_HERest_cyg.sh \
../k3_shot2/folds06_clean_results.sh \
../k3_shot2/folds07_2d_class_test_cyg.sh \
../k3_shot2/folds08_rec_to_cat_lab.sh \
../k3_shot2/folds09_2d_class_perf.sh
do
	echo $cmd >&2; 
	echo "Continue ? any char (bypass), just ENTER  or y (exee), CTRL+C" >&2
	read key

	if  [ "$key" == "" ] || [ "$key" == "yes" ] || [ "$key" == "y" ]; then
		echo "continue..."
		eval $cmd
	else
		
		echo "bypass by '${key}'"
	fi
done

yes | ../k3_shot2/folds09_2d_class_perf.sh > log.perf.txt

grep Correct log.perf.txt  -B 3 -A 11 > log.conf.txt
