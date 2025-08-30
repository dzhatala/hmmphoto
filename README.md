

Artikel Bahasa Indonesia:

@article{hatala2025smartphone,
  title={Smartphone Photos Categorization Using Markov Model with Limited Training Data},
  author={Hatala, Zulkarnaen and Hudzaly, Muhammad},
  journal={Journal of Artificial Intelligence and Software Engineering},
  volume={5},
  number={2},
  pages={573--579},
  year={2025}
}

English article:

@article{Hatala_Hudzaly_2025, title={Few-shot Classification of Smartphone Photos using Hidden Markov Model and Siamese Network}, volume={7}, url={https://ijeeemi.org/index.php/ijeeemi/article/view/116}, DOI={10.35882/ijeeemi.v7i3.116}, abstractNote={
Images from the increasing use of smartphones are so large that they are nearly impossible to handle by hand. The problem arises when a person needs to classify these photos into groups or classes. Smartphones are low-performance devices in contrast to desktop or cloud-based computers. Many solutions of image classification using various types of Convolutional Neural Network (CNN) are performed on massive cloud-based supercomputers. These computers often equipped with very high-end additional specialized graphics processing units (GPUs) at remarkable prices. In fact, to implement classification in most smartphones currently on the market, we need an algorithm that has less computation. Based on this fact, we propose HMM that requires fewer parameters. The aim of this research is to examine HMM method for classification of photos taken with a smartphone. For a comparison we also outline the results from Siamese CNN. The same data are used for training and testing for both models. For HMM, we use Discrete Cosine Transform (DCT) to extract salient features of images. The number of training examples is very small compared to the test set. Here we carried out few-shot classification method. In the training phase, we used Maximum Likelihood (ML) criterion-based, Baum-welch algorithm. Two versions are used; isolated training is applied first and later followed by jointly-embedded Baum-welch estimation of parameters. For recognition of the HMM, Viterbi algorithm is applied. Performances of both procedures were measured. Based on the test results, HMM achieves 0,94 precision, 0.85 recall, F1 score 0.89 and accuracy 0.90 while Siamese claims 0.87, 0.98, 0.92 and 0.91. The result shows that HMM, which has advantage over Siamese in term of fewer parameters number, still competes Siamese CNN with only slight decrease in performance. We conclude that HMM are suitable over Siamese CNN to be implemented in low-performance devices such as cellphones.
}, number={3}, journal={Indonesian Journal of Electronics, Electromedical Engineering, and Medical Informatics}, author={Hatala, Zulkarnaen and Hudzaly, Muhammad}, year={2025}, month={Aug.}, pages={549–558} }


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

