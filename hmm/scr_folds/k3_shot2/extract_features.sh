trace_level="2"  #IMG
trace_level="4"  #FRAME
trace_level="8"  #DCT
trace_level="15"  # 8+...
trace_level="0"  # 8+...
trace_level="3"  # 8+...


dir_to_remove=./tmp
echo "Enter to remove files " ; read
rm ${dir_to_remove}/*

# HImCopy=~/gRes/htk_cygwin/HTK-3.4.1/htk/HImc_16/HImc_16 #wihtout double quote ""
HImCopy=~/gRes/htk_cygwin/HTK-3.4.1/htk/HImCopy/im2feature #wihtout double quote ""
feature_type=2
ls -l ${HImCopy}
# echo `test -f ${HImCopy}`
if [ ! -f "${HImCopy}" ]; then  #must be space in if .. 
	echo "bad HImCopy"
else
	cmd="${HImCopy} -f ${feature_type} -T ${trace_level} -C configcopy.txt -S image2dct.lst"
	echo $cmd ;
	echo "Enter .... " ; read
	eval $cmd
	echo $cmd;
fi
