source env_cygwin.sh
sdir=../../tmp_genlst
scp=tmp_genobs/${project}_tb.lst
mlfs=./tmp_genobs/${project}.mlf
cat ${sdir}/*.lst > ${scp}
cat ${sdir}/*.mlf > ${mlfs}
