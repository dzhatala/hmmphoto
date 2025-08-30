#sync THIS DIRECTORY into /media/joesmart/sda5/
#execute this in ubuntu after mounted and this script sync.sh can be seen

#$pwd
#/media/joesmart/x240rsync
#$./sync.sh

rsync -rtv --exclude="!NODELETE" \
	--exclude=\$RECYCLE.BIN \
	--exclude=RESEARCHS/finger_board/data.rar \
	--exclude=master \
	--exclude=rps \
	--exclude=RESEARCHS/finger_board/github_jurnal/.git \
	--exclude=RESEARCHS/finger_board/github_jurnal/data \
	--exclude=RESEARCHS/finger_board/github_jurnal/hmm \
	--exclude=RESEARCHS/finger_board/github_jurnal/siamese/data \
	--exclude=RESEARCHS/finger_board/python/finger_board/data \
	--exclude=RESEARCHS/finger_board/scripts/scr_folds \
	--exclude=RESEARCHS/htk_cygwin \
	--exclude=RESEARCHS/moodle \
	--exclude=books \
/cygdrive/z/RESEARCHS/finger_board/github_jurnal/siamese_folds/*  ./


