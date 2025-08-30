# ----------------------------------------------------------- 
#                                                             
#                          ___                                
#                       |_| | |_/   SPEECH                    
#                       | | | | \   RECOGNITION               
#                       =========   SOFTWARE                  
#                                                             
#                                                             
# ----------------------------------------------------------- 
#         Copyright: Cambridge University
#          1995-2005 Engineering Department
#                    http://htk.eng.cam.ac.uk
#                    http://mi.eng.cam.ac.uk
#                 
#   Use of this software is governed by a License Agreement   
#    ** See the file License for the Conditions of Use  **    
#    **     This banner notice must not be removed      **    
#                                                             
# ----------------------------------------------------------- 
# File: HTKTools/Makefile.  Generated from Makefile.in by configure.
# ----------------------------------------------------------- 

SHELL =	/bin/sh
inc = ../HTKLib
HTKLIB = $(inc)/HTKLib.a 
srcdir = .
top_srcdir = ..

prefix = /usr/local
exec_prefix = ${prefix}
bindir = ${exec_prefix}/bin
sbindir = ${exec_prefix}/sbin
libexecdir = ${exec_prefix}/libexec
datadir = ${prefix}/share
sysconfdir = ${prefix}/etc
sharedstatedir = ${prefix}/com
localstatedir = ${prefix}/var
libdir = ${exec_prefix}/lib
infodir = ${prefix}/share/info
mandir = ${prefix}/share/man
includedir = ${prefix}/include
oldincludedir = /usr/include

CC      = 	gcc
#CANSI  =  -ansi  #comment this for imageqick which ANSI C89 is obselete
magickdir=/cygdrive/c/rps/Octave-9.4.0/mingw64/lib
#magick=$(magickdir)/libGraphicsMagick.dll.a $(magickdir)/libGraphicsMagickWand.dll.a
# magick_inc=-I/cygdrive/c/rps/Octave-9.4.0/mingw64/include/GraphicsMagick
# magick_inc=-I./include/magick
magick_inc=-I../HImCopy/include/magick
# CFLAGS  = 	$(CANSI) -DCYGWIN -D'ARCH="ASCII"' -Wall -Wno-switch -g -O2 -I$(inc) -DPHNALG -I./tmp/ImageMagick-7.0.10-27
CFLAGS  = 	$(CANSI) -DCYGWIN -D'ARCH="ASCII"' -Wall -Wno-switch -g -O2 -I. -I$(inc) -DPHNALG ${magick_inc} `pkg-config gtk+-3.0 --cflags`
LDFLAGS = 	-L/usr/X11R6/lib  -L/bin -lm -lfftw3  -lcygMagickWand-7.Q16HDRI-7 -lcygMagickCore-7.Q16HDRI-7 ${magick} `pkg-config gtk+-3.0 --libs`
INSTALL = 	/usr/bin/install -c
# PROGS   = im_raw_04 im_raw_03 test_im_raw	fftw_2d fftw_test test_imagemagick HImCopy 
objects   = convert.o dct2_std.o feature.o
PROGS   = 	HImc_16 ZInit
# tests = test_subblock im_raw_04 im_raw_05 fftw_2d testHParm
tests = testHParm
all: $(objects) $(PROGS)
test: 	$(objects) $(tests)

	
%: %.c $(HTKLIB) $(objects)
	if [ ! -d $(bindir) -a X_ = X_yes ] ; then mkdir -p $(bindir) ; fi
	
		$(CC) -o $@ $(CFLAGS) $^ $(LDFLAGS) 

	
	if [ X_ = X_yes ] ; then $(INSTALL) -m 755 $@ $(bindir)  ; fi

strip: $(PROGS)
	-strip $(PROGS)

clean:
	-rm -f *.o 

cleanup:
	-rm -f *.o $(PROGS) *.exe

distclean:
	-rm -f *.o $(PROGS) *.exe Makefile

install: mkinstalldir $(PROGS)
	for program in $(PROGS) ; do $(INSTALL) -m 755 $${program} $(bindir) ; done

mkinstalldir:
	if [ ! -d $(bindir) ] ; then mkdir $(bindir) ; fi


.PHONY: all strip clean cleanup distclean install mkinstalldir
