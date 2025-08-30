source env_cygwin.sh
set -e


gram=wgram_2d.class
rm phone.net
cmd="$HTKTOOLS_DIR/HParse ${gram} phone.net"
echo $cmd; eval $cmd ; echo "enter .. " ;read

# k=0
for k in ${folds}; do 
# for k in 0; do 
	echo "fold k is : ${k}"
	dir=k${k}/hmm30 #herest output
	phonems_2d=phonems_2d_class_test.lst
	dict=new_class.dict
	ftest=k${k}/test.lst
	outdir=k${k}/results  # refine the previous project
	mkdir -p ${outdir}



	marker="-H marker/MARKER_6"

	multi_level="-m"


	#output to single file instead of directory
	# single_mlf="-i recph_class.mlf"

	#ignore score S and time T, print word W
	oppress="-o ST" #oppress time and score left only W : word

	date1=`date`
	cmd="$HTKTOOLS_DIR/HVite  ${multi_level} -l ${outdir} ${single_mlf} ${trace} -w phone.net -C configtrain.txt  -H $dir/hmmdefs \
	 ${marker} \
	 ${oppress} -S $ftest $dict ${phonems_2d}"
	 # echo $cmd ; echo "Enter [ctrl c]" ; read
	# eval $cmd

	# echo $date1
	# date
	# echo $cmd;

	#omit -i so individual label will be created
	mkdir -p ${outdir}		#
	rm -rf ${outdir}/*
	mkdir -p ${outdir}/cats
	rm -rf ${outdir}/cats/*
	trace="-T 0" # see per file, if 0 mean correct, 1 mean wrong

	cmd="$HTKTOOLS_DIR/HVite ${multi_level} -l ${outdir}  -y recph ${trace} \
	-w phone.net -C configtrain.txt  -H $dir/hmmdefs \
	 ${marker} -o S -S $ftest $dict ${phonems_2d}"
	echo "output at ${outdir}"
	echo $cmd ; echo "Enter ?" ;read
	eval $cmd
	echo $cmd
done
