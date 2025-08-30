/*test_imagemagick.c
not working
*/

#include "MagickCore/MagickCore.h"
#include "MagickWand/MagickWand.h"

/*#include "magick/magick_types.h"
#include "wand/magick_wand.h"*/
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  MagickWand *magick_wand;
  MagickBooleanType status;

  if (argc < 2) {
    printf("Usage: %s <image_file>\n", argv[0]);
    return 1;
  }

  // Initialize MagickWand
  MagickWandGenesis();

  // Create a new MagickWand
  magick_wand = NewMagickWand();
  if (!magick_wand) {
    fprintf(stderr, "Error: Could not create MagickWand.\n");
    /*MagickWandTerminus();*/
    return 1;
  }

  // Read the image from file
  status = MagickReadImage(magick_wand, argv[1]);
  if (status == MagickFalse) {
    fprintf(stderr, "Error: Could not read image '%s'.\n", argv[1]);
    magick_wand = DestroyMagickWand(magick_wand);
    MagickWandTerminus();
    return 1;
  }

  // Get image width and height
  size_t width = MagickGetImageWidth(magick_wand);
  size_t height = MagickGetImageHeight(magick_wand);

  printf("Image loaded successfully: widthxheight: %zu x %zu\n", width, height);

	Image *img=GetImageFromMagickWand(magick_wand);
	
	
	PixelIterator *iterator = NewPixelIterator(magick_wand);
    const char* pixel_data = NULL;
	pixel_data=(char*) GetPixels(magick_wand, 0, 0, width, height);

    // Verify that pixel data is valid
    if (pixel_data == NULL) {
        fprintf(stderr, "Error: Could not retrieve pixel data.\n");
        DestroyMagickWand(magick_wand);
        return 1;
    }

    // You can now access and manipulate the pixel data as needed.
    // For example, printing the first few bytes
    printf("First few bytes of raw pixel data: ");
    for (size_t i = 0; i < 10; i++) {
        printf("%02X ", (unsigned char)pixel_data[i]);
    }
	printf("\r\n");
	printf("enter to display image, or ctrl+c to break!");

	char line[256];
	fgets(line, sizeof(line), stdin); 
	
	MagickDisplayImage(magick_wand,   "localhost:0.0");
	
	

  // Destroy the wand
  magick_wand = DestroyMagickWand(magick_wand);

  // Terminate MagickWand
  MagickWandTerminus();

  return 0;
  

}