/* $ ./im_raw_04 img_fn start_col start_row len

 */
 
 
#include <stdio.h>
#include <stdlib.h>
#include "MagickWand/MagickWand.h"

#include "dct2_std.h"


#define STRINGIFY(...) #__VA_ARGS__ "\n"
#define xSTRINGIFY(x) STRINGIFY(x)

#define xstr(s) str(s)
#define str(s) #s

MagickWand *magick_wand;
MagickBooleanType status;


int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <image_file>\n", argv[0]);
    return 1;
  }


  MagickWandGenesis();
  magick_wand = NewMagickWand();

  status = MagickReadImage(magick_wand, argv[1]);
  if (status == MagickFalse) {
    fprintf(stderr, "Error reading image.\n");
    DestroyMagickWand(magick_wand);
    MagickWandTerminus();
    return 1;
  }

  size_t width = MagickGetImageWidth(magick_wand);
  size_t height = MagickGetImageHeight(magick_wand);
  

	Image *img=GetImageFromMagickWand(magick_wand);
    // printf("image property wxh: %zu x%zu \r\n", width, height);
	if(img==NULL){
		fprintf(stderr,"error getting img");
		exit(-1);
	}
    printf("image property H x W: %zux%zu , num. channel: %zu\r\n", height, width, img->number_channels);
	
	if(img->number_channels!=3){
		fprintf(stderr,"error img channel is not 3 %z",img->number_channels);
		exit(-1);
	}
    
	ColorspaceType type=MagickGetImageColorspace(magick_wand);
	
	// printf("colorspace: %s \r\n",xstr(type));
	// printf("colorspace hexa: %x int: %i \r\n",type,type);
	char isRGB=0 ;
	switch (type){
	case RGBColorspace : 	
		isRGB=1;
		break;
	case scRGBColorspace:         /* ??? */
		isRGB=1;
		break;
	case sRGBColorspace:          /* Default: non-linear sRGB colorspace */
		isRGB=1;
		break;
	default:
		break;
	}
	
	if(isRGB==0){
		fprintf(stderr,"image %s not RGB exiting ..",argv[1]);
		exit(-1);
	}

  
  // size_t channels = 3; // RGB
  size_t channels = 3; // RGBA
  // size_t channels =1; // grayscale

  
  size_t data_length = sizeof(unsigned char) * channels * height*width;
  unsigned char *data = malloc(data_length);
	
  status = MagickExportImagePixels(magick_wand, 0, 0, width, height, "RGB", CharPixel, data);
  // status = MagickExportImagePixels(magick_wand, 0, 0, width, height, "Grayscale", CharPixel, data);

	unsigned char R=0,G=0,B=0,row=0,col=0; 
  if (status == MagickTrue) {
	
    printf("Wrote %zu bytes of data to %p address\n", data_length, data);
    // Example: Access the first pixel
    // printf("First Pixel: R=%u, G=%u, B=%u\n", data[0], data[1], data[2]);
	
	
	R=data[width*row*3+col*3];
	G=data[width*row*3+col*3+1];
	B=data[width*row*3+col*3+2];
    printf("1st Pixel: R=%u, G=%u, B=%u\n", data[0], data[1], data[2]);
	printf("r:%u c:%u    R=%u, G=%u, B=%u \n",row,col, R, G, B);

	row=0;col=1;
	R=data[width*row*3+col*3];
	G=data[width*row*3+col*3+1];
	B=data[width*row*3+col*3+2];
    printf("2nd pxls: R=%u, G=%u, B=%u\n", data[3], data[4], data[5]);
    printf("r:%u c:%u    R=%u, G=%u, B=%u \n",row,col, R, G, B);

	row=1;col=0;
	R=data[width*row*3+col*3];
	G=data[width*row*3+col*3+1];
	B=data[width*row*3+col*3+2];
    printf("r:%u c:%u    R=%u, G=%u, B=%u \n",row,col, R, G, B);

	
  } else {
    fprintf(stderr, "Error exporting pixels. err code %zu \n",status);
	 // ThrowWandException(magick_wand);
  }
	
	// gtk_init (&argc, &argv);
	// showimage2(data);
	
  // memset(data,0,data_length); //blacken
  eyeFitRGB2Gray3(data,width,height);
  // averageRGB2Gray3(data,width,height);
  col=0,row=0;
  R=data[width*row*3+col*3];
  printf("gray3 r:%u c:%u    gray=%u \n",row,col, R);
  col=0,row=1;
  R=data[width*row*3+col*3];
  printf("gray3 r:%u c:%u    gray=%u \n",row,col, R);
  col=1,row=0;
  R=data[width*row*3+col*3];
  printf("gray3 r:%u c:%u    gray=%u \n",row,col, R);
  
  unsigned char *gray1_buf=malloc(height*width);
  eyeFitRGB2Gray1(data,gray1_buf,width,height);
  // col=0,row=0;
  // R=gray1_buf[width*row+col];
  // printf("gray1 r:%u c:%u    gray=%u \n",row,col, R);
  // col=0,row=1;
  // R=gray1_buf[width*row+col];
  // printf("gray1 r:%u c:%u    gray=%u \n",row,col, R);
  // col=1,row=0;
  // R=gray1_buf[width*row+col];
  // printf("gray1 r:%u c:%u    gray=%u \n",row,col, R);
  printf("check gray1_buf \r\n");
  dump_uchar_image(gray1_buf,width,height,9);
  
  status = MagickImportImagePixels(magick_wand, 0, 0, width, height, "RGB", CharPixel, data);
  if (status != MagickTrue) {
		fprintf(stderr,"error upload data to wand\r\n");
		exit(-1);
  }

  MagickDisplayImage(magick_wand,   "localhost:0.0");
  free(data);
  DestroyMagickWand(magick_wand);
  MagickWandTerminus();
  
  
  //
  double *dvec =calloc(sizeof(double),height*width);
  DoubleVectFromCharVect(dvec, gray1_buf,height*width);
  printf("recheck  dvect to fft in double: \r\n");
  dump_double_image(dvec,width,9);
  
	DbMatrix mat=CreateDbMatrix(height,width); // the dimension is h*width
	DbMatrixFromCharVect(mat, gray1_buf,width,height);		
	free(gray1_buf);
	printf("double DbMat full matrix\r\n");
	dumpDbMatrix(mat,0,0,9,9);

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
  dump_double_image(fft_mat,block_size,block_size);

  double *fft_result=(double*)malloc(sizeof(double)*block_size*block_size);
  compute2Ddct(fft_mat,fft_result,block_size,block_size	);
  printf("fft results \r\n");
  dump_double_image(fft_result,block_size,block_size);


  return 0;


}
