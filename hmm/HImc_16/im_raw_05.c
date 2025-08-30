/* $ ./im_raw_04 img_fn start_col start_row len

 */
 
 
#include <stdio.h>
#include <stdlib.h>
#include "MagickWand/MagickWand.h"

#include "convert.h"
#include "dct2_std.h"


#define STRINGIFY(...) #__VA_ARGS__ "\n"
#define xSTRINGIFY(x) STRINGIFY(x)

#define xstr(s) str(s)
#define str(s) #s



 MagickWand *magick_wand;
 MagickBooleanType status;
 
 size_t stride_v=4,stride_h=4
	,block_size=4,padding_v=0,padding_h=0;

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <image_file>\n", argv[0]);
    return 1;
  }

  ImageGrayscale gray;
  MagickWandGenesis();
  magick_wand = NewMagickWand();

  getRGBDataDimension(magick_wand,argv[1],&gray);

  fprintf(stdout,"Image dimension w x h: %u x %u \r\n",gray.width,gray.height);
	
  MagickDisplayImage(magick_wand,   "localhost:0.0");
  printf("dct2_std #1\r\n");
	getGrayscaleData(magick_wand, &gray);
  // printf("dct2_std #1\r\n");
	DestroyMagickWand(magick_wand);
  MagickWandTerminus();

	//dct test
	DbMatrix mat=CreateDbMatrix(gray.height,gray.width); // the dimension is h*width
	DbMatrixFromCharVect(mat, gray.data,gray.width,gray.height);		
	printf("double DbMat full matrix\r\n");
	dumpDbMatrix(mat,0,0,9,9);

	// if(FALSE == FALSE ) return 1;
	size_t block_size=4;
	size_t col_start=0,col_len=block_size;
	size_t row_start=0,row_len=block_size;
  

	if(argc>2){
		col_start=atoi(argv[2]);
	}
	if(argc>3){
		row_start=atoi(argv[3]);
	}

	if(argc>4){
		block_size=atoi(argv[4]);
	}

	printf("double DbMat full matrix shift(%u,%u):%u\r\n",col_start,row_start,block_size);
	dumpDbMatrix(mat,col_start,row_start,block_size,block_size);

  
  double *fft_mat;
  fft_mat=(double*)malloc(sizeof(double)*block_size*block_size);

  DbVectFromMatrix(mat,fft_mat,  
					col_start, block_size,
					 row_start,block_size	);
  printf("double vec subblock %u,%u : %u\r\n",col_start,row_start,block_size);
  dump_double_image(fft_mat,block_size,block_size,block_size);

  double *fft_result=(double*)malloc(sizeof(double)*block_size*block_size);
  compute2Ddct(fft_mat,fft_result,block_size,block_size);
  printf("fft results \r\n");
  dump_double_image(fft_result,block_size,block_size,block_size);

  free(fft_result);
  free(gray.data);
  return 0;


}
