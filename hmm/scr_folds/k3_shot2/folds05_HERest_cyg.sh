sync
source env_cygwin.sh
set -e
for k in ${folds}; do
# for k in 0; do
	# k=2
	dirout=k${k}/hmm30
	mkdir -p ${dirout}
	rm -f ${dirout}/*

	hmmIN=hmm1 #last isolated output see Hrest_cyg.sh
	scp=k${k}/embedded_mfc_train.lst
	mlfs="-I k${k}/embedded_mfc_train.mlf"

	cmd="$HTKTOOLS_DIR/HERest -C configtrain.txt  -S ${scp} -T 3 ${mlfs}    \
	   -M ${dirout}  -H k${k}/${hmmIN}/hmmdefs phonems_2d_class_train.lst"
	echo $cmd ; echo "Enter ? [CTRL+C]" ; read
	eval $cmd
	echo $cmd
	echo "output in ${dirout}"
	# break
done
