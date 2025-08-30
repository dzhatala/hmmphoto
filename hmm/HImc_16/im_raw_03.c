#include <stdio.h>
#include <stdlib.h>
#include "MagickWand/MagickWand.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <image_file>\n", argv[0]);
    return 1;
  }

  MagickWand *magick_wand;
  MagickBooleanType status;

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
  // size_t channels = 3; // RGB
  size_t channels = 4; // RGBA
  // size_t channels =1; // grayscale
  size_t data_length = sizeof(unsigned char) * channels * width * height;
  unsigned char *data = malloc(data_length);

  // status = MagickExportImagePixels(magick_wand, 0, 0, width, height, "RGB", CharPixel, data);
  status = MagickExportImagePixels(magick_wand, 0, 0, width, height, "RGBA", CharPixel, data);
  // status = MagickExportImagePixels(magick_wand, 0, 0, width, height, "Grayscale", CharPixel, data);
  if (status == MagickTrue) {
    printf("image property wxh: %zu x%zu \r\n", width, height);
    printf("Wrote %zu bytes of data to %p address\n", data_length, data);
    // Example: Access the first pixel
    // printf("First Pixel: R=%u, G=%u, B=%u\n", data[0], data[1], data[2]);
    printf("First Pixel: R=%u, G=%u, B=%u A=%u\n", data[0], data[1], data[3]);
  } else {
    fprintf(stderr, "Error exporting pixels. err code %zu \n",status);
	 // ThrowWandException(magick_wand);
  }

	
	// gtk_init (&argc, &argv);
	// showimage2(data);

  free(data);
  DestroyMagickWand(magick_wand);
  MagickWandTerminus();
  return 0;
}


#include <gtk/gtk.h>

static gboolean
on_window_draw (GtkWidget *da, GdkEvent *event, gpointer data)
{
    (void)event; (void)data;
    GdkPixbuf *pix;
    GError *err = NULL;
    /* Create pixbuf */
    pix = gdk_pixbuf_new_from_file("/usr/share/icons/cab_view.png", &err);
    if(err)
    {
        printf("Error : %s\n", err->message);
        g_error_free(err);
        return FALSE;
    }
    cairo_t *cr;
    cr = gdk_cairo_create (gtk_widget_get_window(da));
    //    cr = gdk_cairo_create (da->window);
    gdk_cairo_set_source_pixbuf(cr, pix, 0, 0);
    cairo_paint(cr);
    //    cairo_fill (cr);
    cairo_destroy (cr);
    //    return FALSE;
}

int showImage ( int argc, char **argv) {
    GtkWidget *window;
    GtkWidget *canvas;
    gtk_init (&argc , &argv);
    window = gtk_window_new (GTK_WINDOW_TOPLEVEL);
    gtk_widget_set_size_request (window,
        50, 50);

    g_signal_connect (window, "destroy",
        G_CALLBACK (gtk_main_quit) , NULL);
    canvas = gtk_drawing_area_new ();
    gtk_container_add (GTK_CONTAINER (window), canvas);
    g_signal_connect (canvas, "draw", (GCallback) on_window_draw, NULL);

    gtk_widget_set_app_paintable(canvas, TRUE);
    gtk_widget_show_all (window);
    gtk_main ();
    return 0;
}

void on_destroy (GtkWidget *widget G_GNUC_UNUSED, gpointer user_data G_GNUC_UNUSED)
{
    gtk_main_quit ();
}


int showimage2(const char*buf2){

FILE *f;
    guint8 buffer[100000];
	// guint8 *buffer=NULL;
    gsize length;
    GdkPixbufLoader *loader;
    GdkPixbuf *pixbuf;
    GtkWidget *window;
    GtkWidget *image;

    
    f = fopen ("tmp/x.jpeg", "r");
	// buffer=(guint8*)calloc(sizeof(guint8),100000);
    length = fread (buffer, 1, sizeof(buffer), f);
    fclose (f);

    loader = gdk_pixbuf_loader_new ();
    gdk_pixbuf_loader_write (loader, buffer, length, NULL);
    // gdk_pixbuf_loader_write (loader, buffer, length, buf2);
    gdk_pixbuf_loader_close(loader, NULL);
    pixbuf = gdk_pixbuf_loader_get_pixbuf (loader);

    window = gtk_window_new (GTK_WINDOW_TOPLEVEL);
    image = gtk_image_new_from_pixbuf (pixbuf);
    gtk_container_add (GTK_CONTAINER (window), image);
    gtk_widget_show_all (GTK_WIDGET (window));
    g_signal_connect (window, "destroy", G_CALLBACK(on_destroy), NULL);
    gtk_main ();

    return 0;
	
}


