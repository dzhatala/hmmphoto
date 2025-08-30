// convert.convert

#include <stdio.h>
#include <math.h>
#include "MagickWand/MagickWand.h"
#include "dct2_std.h"
#include "convert.h"
/*
convert to grayscale 1 channel, 8 level depth

*/
void eyeFitRGB2Gray1(const unsigned char* rgb_buf, unsigned char *gray_buf, int width, int height){
	
	float R=0,G=0,B=0,f_g=0;
	int idgray=0,idfloat=0;
	
	for (idgray=0;idgray<width*height;idgray++){
		R=(float)rgb_buf[idfloat];idfloat++;
		G=(float)rgb_buf[idfloat];idfloat++;
		B=(float)rgb_buf[idfloat];idfloat++;
		
		f_g=0.299*R + 0.587*G + 0.114*B; 
		gray_buf[idgray]=(unsigned char)f_g;
		/*if (idgray==0|idgray==1|idgray==width){
			printf("uchar RGB: idfloat:%u R=%u, G=%u, B=%u  \n",idfloat, rgb_buf[idfloat], rgb_buf[idfloat+1], rgb_buf[idfloat+2]);
			// printf("avrRGB: idgray:%u R=%.1f, G=%.1f, B=%.1f gr:f_g \n",idgray, R, G, B,c_g);
		}*/
	}
}

/***
	convert 2 gray scale preserve 3 channels
**/
void eyeFitRGB2Gray3(unsigned char* rgb_buf, int width, int height){
	
	float R=0,G=0,B=0,f_g=0;
	int idgray=0,idfloat=0;
	char c_g=0;
	for (idgray=0;idgray<width*height;idgray++){
		
		R=(float)rgb_buf[idfloat];
		G=(float)rgb_buf[idfloat+1];
		B=(float)rgb_buf[idfloat+2];
		
		f_g=0.299*R + 0.587*G + 0.114*B;
		c_g=(unsigned char)f_g;
		if (idgray==0|idgray==1|idgray==width){
			printf("uchar RGB: idfloat:%u R=%u, G=%u, B=%u  \n",idfloat, rgb_buf[idfloat], rgb_buf[idfloat+1], rgb_buf[idfloat+2]);
			// printf("avrRGB: idgray:%u R=%.1f, G=%.1f, B=%.1f gr:f_g \n",idgray, R, G, B,c_g);
		}
		rgb_buf[idfloat]=c_g;
		rgb_buf[idfloat+1]=c_g;
		rgb_buf[idfloat+2]=c_g;
		idfloat+=3;
	}
}

					/**DO NOT remove 'unsigned'**/
void averageRGB2Gray3(unsigned char* rgb_buf, int width, int height){
	
	float R=0,G=0,B=0,f_g=0;
	int idgray=0,idfloat=0;
	char c_g=0;
	for (idgray=0;idgray<width*height;idgray++){
		
		R=(float)rgb_buf[idfloat];
		G=(float)rgb_buf[idfloat+1];
		B=(float)rgb_buf[idfloat+2];
		f_g=(R + G + B)/3;
		if (idgray==0|idgray==1|idgray==width){
			printf("uchar RGB: idfloat:%u R=%u, G=%u, B=%u  \n",idfloat, rgb_buf[idfloat], rgb_buf[idfloat+1], rgb_buf[idfloat+2]);
			// printf("avrRGB: idgray:%u R=%.1f, G=%.1f, B=%.1f gr:f_g \n",idgray, R, G, B,c_g);
		}
		c_g=(char)f_g;
		rgb_buf[idfloat]=c_g;
		rgb_buf[idfloat+1]=c_g;
		rgb_buf[idfloat+2]=c_g;
		idfloat+=3;
	}
}


/**return brightenss the same 
https://stackoverflow.com/questions/13125939/relation-between-light-intensity-and-r-g-b
https://www.nbdtech.com/Blog/archive/2008/04/27/Calculating-the-Perceived-Brightness-of-a-Color.aspx
https://alienryderflex.com/hsp.html

**/
float brightnessFromRGB(char R, char G, char B){
	return sqrt( .241*R*R + .691*G*G + .068*B*B ) ;
}


/**convert from vector grayscale into  Matrix  **/
int grayscale2Matrix(const unsigned char *grayVect, DbMatrix gray, int width, int height){
	/** width=number of cols, height = number of rows**/
	int row=0, col=0, idgray=0; 
	for (row=0;row<height;row++){
		for(col=0;col<width; col++){
			idgray=row*width+col;  
			gray[row][col]=(double)grayVect[idgray];
			idgray++;
		}
	}		
	
	return -1;
}

/**convert from Matrix into vector grayscale   **/
int Matrix2grayscale(const DbMatrix gray, unsigned char *grayVect,  int width, int height){
	/** width=number of cols, height = number of rows**/
	int row=0, col=0, idgray=0; 
	for (row=0;row<height;row++){
		for(col=0;col<width; col++){
			idgray=row*width+col;  
			grayVect[idgray]=(unsigned char)gray;
			idgray++;
		}
	}		
	
	return -1;
}


	extern MagickWand *magick_wand;
	extern MagickBooleanType status;

int getRGBDataDimension(const MagickWand *wand, const char *fn,
	ImageGrayscale *outputinfo 
	){

	MagickWandGenesis();
	magick_wand = NewMagickWand();
	
	
	status = MagickReadImage(magick_wand, fn);
	if (status == MagickFalse) {
		fprintf(stderr, "Error reading image.\n");
		DestroyMagickWand(magick_wand);
		MagickWandTerminus();
		return 1;
	}

	outputinfo->width = MagickGetImageWidth(magick_wand);
	outputinfo->height = MagickGetImageHeight(magick_wand);
  
	return 0;
}  

/** read RGB convert to gray put in data 
magickwand must already read with
@getRGBDataDimension()
@outputinfo.data will be filled 

caller should NOT allocate memory for outputinfo->data.
or caller should free the memory when it was allocated before.
when violated the memory will not be freed by this function

**/
int getGrayscaleData(const MagickWand *wand, ImageGrayscale *outputinfo){

	Image *img=GetImageFromMagickWand(magick_wand);
	// unsigned char *data;
    // printf("image property wxh: %zu x%zu \r\n", width, height);
	if(img==NULL){
		fprintf(stderr,"error getting img");
		exit(-1);
	}
	
	outputinfo->width = MagickGetImageWidth(magick_wand);
	outputinfo->height = MagickGetImageHeight(magick_wand);

	
    /*printf("image property H x W: %zux%zu , num. channel: %zu\r\n"
		, outputinfo->height, outputinfo->width, img->number_channels);
	*/
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
	
	// printf("image is RGB \r\n"); fflush(stdout);
	if(isRGB==0){
		fprintf(stderr,"image %s not RGB exiting ..");
		return -1;
	}

  
  size_t channels = 3; // RGBA

  
  size_t data_length = sizeof(unsigned char) * channels * outputinfo->height*outputinfo->width;
  unsigned char *datargb = malloc(data_length);

	// printf("#######34#######3 \r\n"); fflush(stdout);
	
  status = MagickExportImagePixels(magick_wand, 0, 0, outputinfo->width
		, outputinfo->height, "RGB", CharPixel,datargb);

  if (status == MagickTrue) {
	
    // printf("Wrote %zu bytes of data to %p address\n", data_length, datargb);
	;
	
  } else {
    fprintf(stderr, "Error exporting pixels. err code %zu \n",status);
	 // ThrowWandException(magick_wand);
	 return -1;
  }
	
  /**no free() called, becarefull*/
  outputinfo->data=malloc(outputinfo->height*outputinfo->width);
  eyeFitRGB2Gray1(datargb,outputinfo->data,outputinfo->width,outputinfo->height);
  free(datargb);

   return 0;
}
