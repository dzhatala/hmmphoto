#ifndef _CONVERT_H_
#define _CONVERT_H_

#include "HShell.h"
#include "HMem.h"
#include "dct2_std.h"
#include "MagickWand/MagickWand.h"


typedef struct {
	unsigned char *data; // 8 bit level grayscale
		//caller should free data when no use
	size_t width; //number of columns
	size_t height; //number of rows
}ImageGrayscale;

// int grayscale2Matrix(const unsigned char *grayVect, Matrix gray, int width, int height);
// int Matrix2grayscale(const DbMatrix gray, unsigned char *grayVect,  int width, int height);

#endif