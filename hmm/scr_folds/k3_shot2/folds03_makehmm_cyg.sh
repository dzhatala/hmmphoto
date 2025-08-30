sync
source env_cygwin.sh
HVITE=$HTKTOOLS_DIR/HVite
#rm -f hmm0/*


# k="2"
# for c in ${categs}
for k in ${folds}; do
	# mkdir -p ${k}/hmm0
	init_name=init_6
	#init_name=init_16
	mkdir -p k${k}/hmm0
	rm -f k${k}/hmm0/*
	cp k${k}/hinitoutput/$init_name k${k}/hmm0/
	# kill -INT $$ #cygwin ok

	cp /dev/null k${k}/hmm0/hmmdefs

	for x in `cat phonems_2d_class_train.lst`
	do
		echo "creating hmm $x from  k${k}/hmm0/$init_name "
		cat k${k}/hmm0/$init_name | sed s/$init_name/$x/ >> k${k}/hmm0/$x
		cat k${k}/hmm0/$init_name | sed s/$init_name/$x/ >> k${k}/hmm0/hmmdefs
	done

	#head -3 hmm0/proto > hmm0/macros ; head -3 hmm0/vFloors >> hmm0/macros ; read
	echo "echo are phonems in phonems.lst correct ? [ENTER! CONT ] CTRL+C EXIT!" ; read
	# break
	echo "k:${k}  creating hmm completed"
done
