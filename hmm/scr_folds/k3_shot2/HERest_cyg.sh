sync
source env_cygwin.sh
HVITE=$HTKTOOLS_DIR/HVite
rm -f hmm10/*
dirout=hmm30

mkdir -p ${dirout}
hmmIN=hmm1
scp=tmp_genobs/${project}_tb_2d_class_train.lst
mlfs="-I ./tmp_genobs/${project}_2d_class.mlf"

cmd="$HTKTOOLS_DIR/HERest -C configtrain.txt  -S ${scp} -T 3 ${mlfs}    \
   -M ${dirout}  -H ${hmmIN}/hmmdefs phonems_2d_class_train.lst"
echo $cmd ; echo "Enter ? [CTRL+C]" ; read
eval $cmd
echo $cmd
echo "output in ${dirout}"