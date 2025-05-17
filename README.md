DATA
 /data contain original full size taken photos
	/data/fp_board
	/data/answer_sheet

 /data/smaller : converted into 280 width
	-see resize_image.sh (cygwin shell examples)
 /data/labelme :   labelled images (for HMM training)

A. HMM
 -feature extraction: see two examples batch files
	create_mfcc_obs_fpboard.bat
	create_mfcc_obs_sheet.bat
 =HMM Model construction and training
	Global initialization: 
		see baum_viterbi_init_cyg.sh
	Isolated training	  : 
		see  HRest_cyg.sh
	Embedded/join training: 
		see HERest_cyg.sh
	final model:
		models/hmm30/hmmdefs
		models/marker
	
 -recognizing one single image full image:
	recognize_single.sh
	rec2class.sh #convert from sub label into category

 -recognize  photo in dir
	2d_class_test_cyg.sh
 -performance of dir. recognition
	rec_to_cat_lab.sh #convert all .recs into cat labels to be used by higher level HResults
	2d_class_perf-cyg.sh
	
B. Siamese
 -training
 -recognize one
 -recognize all and performance all
	>py journal_eval.py

Software requisites:
-Microsoft Windows [Version 10.0.19045.5608]
-CYGWIN_NT-10.0-19045 x86_64 Cygwin
	-imageqick 7.0.10-27 : mogrify
-Octave-9.4.0
