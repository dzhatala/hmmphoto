#include <stdio.h>
#include <stdlib.h>
#include "MagickWand/MagickWand.h"

#define COLOR_MAX 65535.0

typedef struct {
        float         **px;
        unsigned long width;
        unsigned long height;
} RawPixels;

RawPixels getRawImage (char *path)
{
    MagickWand        *mw;
    MagickBooleanType status;
    PixelIterator     *iter;
    PixelPacket pixel;
    PixelWand         **pixels;
    RawPixels         rp;

    long          x;
    long          y;
    unsigned long width;

    unsigned long count;

    MagickWandGenesis();

    mw     = NewMagickWand();
    status = MagickReadImage(mw, path);
    rp.px   = NULL;

    if (status == MagickFalse) {
        return rp;
    }

    iter = NewPixelIterator(mw);

    if (iter == (PixelIterator *) NULL) {
        return rp;
    }

    rp.width  = 0;
    rp.height = (unsigned long) MagickGetImageHeight(mw);
    count     = 0;

    for (y = 0; y < rp.height; y++) {
        pixels   = PixelGetNextIteratorRow(iter, &width);
        rp.width = (unsigned long) width;

        if (rp.px == NULL) {
            rp.px = malloc(sizeof(float *) * (width * rp.height + 1)); 
        }

        if (pixels == (PixelWand **) NULL) {
            break;
        }

        for (x = 0; x < (long) width; x++) {
            count++;
            rp.px[count - 1] = malloc(sizeof(float) * 3);

            PixelGetMagickColor(pixels[x], &pixel);

            rp.px[count - 1][0] = pixel.red / COLOR_MAX;
            rp.px[count - 1][1] = pixel.green / COLOR_MAX;
            rp.px[count - 1][2] = pixel.blue / COLOR_MAX;
        }
    }

    rp.px[count] = NULL;

    return rp;
}

void freeRawImage (RawPixels rp)
{
    for (int i = 0; rp.px[i] != NULL; i++) {
        free(rp.px[i]);
    }

    free(rp.px);
}

void writeRawImage (RawPixels rp, char *path)
{
    // This function is the one that gives me a headache.
    // Basically, I take float **rp.px, which has the following structure
    // at this point:
    //
    // {
    //     {float red, float green, float blue},
    //     {float red, float green, float blue},
    //     ...
    // }
    //
    // Now, the documentation at https://www.imagemagick.org/api/magick-image.php#MagickConstituteImage
    // says the following:
    //
    //     "The pixel data must be in scanline order top-to-bottom"
    //
    // So up until the call to MagickConstituteImage() I am trying
    // to restructure the data and write it into float *scanline
    // to satisfy that requirement. However, once I call
    // MagickConstituteImage(), my program segfaults.


    MagickWand    *mw;
    float         *scanline;
    unsigned long pxcount;

    mw     = NewMagickWand();
    
	for (pxcount = 0; rp.px[pxcount] != NULL; pxcount++);
    pxcount *= 3;
    scanline = malloc(sizeof(float) * pxcount);

    pxcount = 0;

    for (int i = 0; rp.px[i] != NULL; i++) {
        for (int j = 0; j < 3; j++) {
            scanline[pxcount++] = rp.px[i][j];
            // printf("%f\n", scanline[pxcount - 1]);
        }
    }

    // This function causes a segfault
    MagickConstituteImage(mw, rp.width, rp.height, "RGB", FloatPixel, scanline);
	MagickWriteImage(mw,path);
    free(scanline);
}

int main ()
{
        RawPixels rp; 

        if ((rp = getRawImage("tmp/x.jpeg")).px == NULL) {
                fprintf(stderr, "ERROR: Failed to process image\n");
                return 1;
        }


        // Some image processing using raw pixels here


        writeRawImage(rp, "tmp/raw_out.png");

        freeRawImage(rp);

        return 0;
}
