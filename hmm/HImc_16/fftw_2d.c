/** 

https://stackoverflow.com/questions/79152885/differences-between-opencv-and-fftw-2d-dct

*/
#include "HShell.h"
#include "dct2_std.h"
#include "HMem.h"
#include "MagickWand/MagickWand.h"

MagickWand *magick_wand;
 MagickBooleanType status;
 

int main() {

/*const int rows = 8;
const int cols = 6;
*/
	int rows = 2;
	int cols = 2;
	double *image=NULL;
	double *dct_output=NULL;



	rows=2;cols=2;
	image=calloc(sizeof(double),rows * cols);
	image[0] = 2 ; image[1]=6; image[2]=4; image[3]=4;
	
	dct_output=calloc(sizeof(double),rows * cols);
	printf("A[]: \r\n");
	dump_double_image(image,rows,cols,rows);

	compute2Ddct(image, dct_output, rows, cols);
	printf("dct2 \r\n");
	dump_double_image(dct_output,rows,cols,cols);
	free(image);
	free(dct_output);


	/*https://groups.google.com/g/comp.soft-sys.matlab/c/td_sbgAOltE?pli=1
	difference between matlab dct2 and FFTW lib(FFTW_REDFT10
	*/
	/*rows=3;cols=3;
	image=calloc(sizeof(double),rows * cols);
	image[0] = 0.2 ; image[1]=0.3; image[2]=1; 
	image[3] = 0 ; image[4]=12; image[5]=5; 
	image[6] = 0.3 ; image[7]=0.3; image[8]=1.0; 
	
	dct_output=calloc(sizeof(double),rows * cols);
	printf("\r\n");
	printf("A[]: \r\n");
	dump_double_image(image,rows,cols,rows);

	compute2Ddct(image, dct_output, rows, cols);
	printf("dct2 \r\n");
	dump_double_image(dct_output,rows,cols,cols);
	free(image);
	free(dct_output);
	*/

	rows=4;cols=4;
	image=calloc(sizeof(double),rows * cols);
	image[0] = 0.2 ; image[1]=0.3; image[2]=1; image[3] = 0 ;
	image[4] = 0 ; image[5]=12; image[6]=5; image[7] = 0 ;
	image[8] = 0.3 ; image[9]=0.3; image[10]=1.0; image[11] = 0 ;
	image[12] = 0.0 ; image[13]=0.0; image[14]=0.0; image[15] = 0 ;
	
	dct_output=calloc(sizeof(double),rows * cols);
	printf("\r\n");
	printf("A[]: \r\n");
	dump_double_image(image,rows,cols,rows);

	compute2Ddct(image, dct_output, rows, cols);
	printf("dct2 \r\n");
	dump_double_image(dct_output,rows,cols,cols);
	free(image);
	free(dct_output);
 
	return 0;
}


/**get pixel block put on buffer**/
int getBlock(Matrix image, int imheight,int xstart, int ystart, int block_size,unsigned char **block_buf){
	
}
