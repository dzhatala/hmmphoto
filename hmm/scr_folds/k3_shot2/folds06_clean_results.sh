#parse .rec contain rows and create its HTK format cat. lab file

source env_cygwin.sh

for k in ${folds}; do
	echo "fold: ${k}"
	recsdir="k${k}/results"
	cmd="rm -rf ${recsdir}/"
	echo $cmd;
	rm -rf ${recsdir}
done