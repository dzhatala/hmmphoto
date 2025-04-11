#recognizing single image file
tmp=./tmp
full_f=$1
if [ "${full_f}" = "" ]; then
	echo "no image to recognize"
	exit
fi

filename_with_ext=$(basename "${full_f}")
filename="${filename_with_ext%.*}"
extension="${filename_with_ext##*.}"

if [ "${extension^^}" == "JPG" ] || [ "${extension^^}" == "JPEG" ]\
|| [ "${extension^^}" == "BMP" ]\
|| [ "${extension^^}" == "PNG" ] ; then 
	if [ ! -f ${full_f} ]; then
		echo "Not a file: ${full_f}"
		exit
	fi	
else
	echo "${extension} is not image type"
	exit
fi

octave=/cygdrive/c/rps/Octave-9.4.0/mingw64/bin/octave.exe

mkdir -p ${tmp}
smaller_f="${tmp}/280x${filename_with_ext}"
smaller_mfc="${tmp}/280x${filename}.mfc"
smaller_rec="${tmp}/280x${filename}.rec"
# echo $smaller_mfc
# exit
cmd="cp ${full_f} ${smaller_f}"
echo $cmd; eval $cmd

cmd="mogrify -resize 280x  ${smaller_f}"
echo -e "${tab}exec: $cmd" ; eval $cmd

# smaller_f=g:\\rsync\\RESEARCHS\\finger_board_detection_image\\data\\smaller\\fp_image\\24-01\\280_IMG-20240102-WA0000.jpeg
mkdir -p tmp
# hmm=	#model
${octave} octave\\img2mfcc.m ${smaller_f} .\\tmp
echo $smaller_mfc > tmp/test.lst
./2d_class_test_cyg.sh tmp/test.lst tmp models/hmm30
cmd="./rec2class.sh ${smaller_rec}"
echo $cmd ;eval $cmd
# do some work
