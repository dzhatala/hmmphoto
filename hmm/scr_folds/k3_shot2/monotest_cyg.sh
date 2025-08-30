source env_cygwin.sh

#HRest Outputs
dir=hmm1
dir=hmm10 #herest
phonems=phonems.lst
dict=phone1.dic
ftest=tmp_genobs/${project}_tb.lst
outdir=results
 
cp /dev/null $dict ;sync
awk '{
	if (index($0,"sp")>0){
		print "sil\tsp"
	}else
	print $1,"\t",$1 
	
}' $phonems > $dict
sync

rm phone.net
$HTKTOOLS_DIR/HBuild $dict phone.net

gram=wgram
#gram=wgram_row
cmd="$HTKTOOLS_DIR/HParse ${gram} phone.net"
# echo $cmd; eval $cmd ; echo "enter .. " ;read

# multi_level="-m"
cmd="$HTKTOOLS_DIR/HVite  ${multi_level} -i recph.mlf -T 1 -w phone.net -C configtrain.txt  -H $dir/hmmdefs \
 -o S -S $ftest $dict $phonems"
echo $cmd ; eval $cmd
#omit -i so individual label will be created
mkdir -p $outdir		#
cmd="$HTKTOOLS_DIR/HVite ${multi_level} -l $outdir  -y recph -T 1 -w phone.net -C configtrain.txt  -H $dir/hmmdefs \
 -o S -S $ftest $dict $phonems"
# echo $cmd ; eval $cmd

