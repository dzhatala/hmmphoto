#usage rec2cat.sh rec_file
if [ ! -f "$1" ]; then
	echo "Not a file: $1"
	exit
fi

pattern=`grep fp_key $1`
if [ ! "${pattern}" == "" ]; then
	echo "$1 is a fingerprint board"
fi

pattern=`grep bg_white $1`
if [ ! "${pattern}" == "" ]; then
	echo "$1 is an answer sheet"
fi
