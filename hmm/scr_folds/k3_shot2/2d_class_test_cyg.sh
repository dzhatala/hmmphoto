source env_cygwin.sh

#HRest Outputs
# dir=hmm1
dir=hmm10_class #herest
dir=hmm30 #herest
# phonems=phonems.lst
phonems_2d=phonems_2d_class_test.lst
# dict=phone1_2D.dic
# dict=phone_class.dict
dict=new_class.dict
ftest=tmp_genobs/${project}_tb_2d_class_test.lst
outdir=./results  # refine the previous project
# outdir=../scr_03/results  # refine the previous project
 
# cp /dev/null $dict ;sync
# awk '{
	# if (index($0,"sp")>0){
		# print "sil\tsp"
	# }else
	# print $1,"\t",$1 
	
# }' $phonems > $dict
# sync


rm phone.net

# cp phone1.dic ${dict}
# echo "MARKER	MARKER" >> ${dict}
# cp ${phonems} ${phonems_2d}
# echo "MARKER" >> ${phonems_2d}

gram=wgram_2d.class
cmd="$HTKTOOLS_DIR/HParse ${gram} phone.net"
echo $cmd; eval $cmd ; echo "enter .. " ;read

marker="-H marker/MARKER_6"

multi_level="-m"
trace="-T 1"


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
rm ${outdir}/*
mkdir -p ${outdir}/cats

rm ${outdir}/cats/*
trace="-T 1" # see per file, if 0 mean correct, 1 mean wrong
# trace="-T 0" # see per file, if 0 mean correct, 1 mean wrong
cmd="$HTKTOOLS_DIR/HVite ${multi_level} -l ${outdir}  -y recph ${trace} \
-w phone.net -C configtrain.txt  -H $dir/hmmdefs \
 ${marker} -o S -S $ftest $dict ${phonems_2d}"
echo "output at ${outdir}"
echo $cmd ; echo "Enter ?" ;read
eval $cmd
echo $cmd




