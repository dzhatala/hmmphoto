#usage 2d_class_test_cyg.sh list_file out_dir hmm_dir
#hmm_dir must contain hmmdefs
SECONDS=0
source env_cygwin.sh
dir=hmm30 #herest output
phonems_2d=phonems_2d_class_test.lst
dict=new_class.dict
ftest=tmp_genobs/${project}_tb_2d_class_test.lst
outdir=../scr_03/results  # refine the previous project

if [ "$1" != "" ]; then
	ftest="$1"
fi  

if [ "$2" != "" ]; then
	outdir="$2"
fi  

if [ "$3" != "" ]; then
	dir="$3"
fi  


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

marker="-H models/marker/MARKER"

multi_level="-m"
trace="-T 1"
trace="-T 0"


#output to single file instead of directory
# single_mlf="-i recph_class.mlf"

#ignore score S and time T, print word W
oppress="-o ST" #oppress time and score left only W : word

date1=`date`
cmd="$HTKTOOLS_DIR/HVite  ${multi_level} -l ${outdir} ${single_mlf} ${trace} -w phone.net -C configtrain.txt  -H $dir/hmmdefs \
 ${marker} \
 ${oppress} -S $ftest $dict ${phonems_2d}"
echo $cmd ; echo "Enter [ctrl c]" ; read
eval $cmd


#omit -i so individual label will be created
mkdir -p ${outdir}		#
cmd="$HTKTOOLS_DIR/HVite ${multi_level} -l ${outdir}  -y recph ${trace} -w phone.net -C configtrain.txt  -H $dir/hmmdefs \
 ${marker} -o S -S $ftest $dict ${phonems_2d}"
# echo $cmd ; eval $cmd
duration=$SECONDS
echo "time: $((duration / 60)) minutes and $((duration % 60)) s."



