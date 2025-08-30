./extract_features.sh
./baum_viterbi_init_cyg.sh
./make_hmm_cyg.sh
./train1_hmm_cyg.sh
./HERest_cyg.sh
./2d_class_test.sh
./rec_to_cat_lab.sh
./2d_class_perf.sh



create your own directory
	change env_cygwin.sh
		project="your relative working dir."

changing vector size:
wehenever the vector size of hmm observations is changed (feature type is changed)
you need to change
	init_X (x depend on your vector size)
	wgram_2d.class
	new_class.dict
	
	