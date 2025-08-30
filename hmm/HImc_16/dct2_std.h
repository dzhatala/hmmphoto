#ifndef __DCT2_STD_H__
#define __DCT2_STD_H__`
/*
to adapt fftw into octave/matlab dct 2d
*/

#include <stdlib.h>
#include <fftw3.h>
/**compute 2D dct as octave/matlab 

in octave/matlab rows are separated by semi colon ';'

in C (this file) follow fftw where image is stored in single dimension 
memory block

double image[nrows * ncols];

image at first row second col (0,1)  can be accessed as image[0*cols+1

**/

typedef double **DbMatrix;

/**subblock of an image*/
typedef struct {
	size_t width; /**number of cols of whole image**/
	size_t height;/**number of rows of whole image**/
	size_t size; /** sizexsize (square) of this block **/
	size_t row_stride; /**vertical_stride from left to right **/
	size_t col_stride; /**vertical_stride from left to right **/
	size_t left_pad;/**left image padding **/
	size_t right_pad;/**right image padding **/
	size_t top_pad;  /**top image padding **/
	size_t bottom_pad; /**bottom image padding **/
	size_t frame_processed; /**total frame*/
	size_t total_frame_cols; /**total frame along cols/horizontal/width**/
	size_t total_frame_rows; /**total frame along rows/vertical/height**/
	size_t col_start; /**y coordinate,start of current block**/
	size_t row_start;  /**relative to full matrix**/
	unsigned char *current ; /**cache current subblock with ->size */
	double *features ; /**cache current features **/
	size_t feature_length;/**length of a vector features**/
	
} SubBlockInfo;



void compute2Ddct(const double* input, double* output, int rows, int cols) ;


/* treat liner block as image with size nrows ncols**/
DbMatrix CreateDbMatrix(int rows, int cols);

/**dump matrix from start until start+length **/
void dumpDbMatrix(const DbMatrix mtr,int start_row, size_t start_col,int len_row, size_t len_col );
void dump_double_image(const  double *mtr,const size_t width
	,const size_t len);


void getSubblock(const DbMatrix mtr, DbMatrix sub, size_t start, size_t length );
void DoubleVectFromCharVect(double *target, const unsigned char *source,size_t length);

#endif