dll_1="/cygdrive/c/rps/Octave-9.4.0/mingw64/lib/libGraphicsMagick.dll.a"
dll_1=/cygdrive/c/rps/Octave-9.4.0/mingw64/lib/libGraphicsMagickWand.dll.a

cmd="objdump --syms ${dll_1}"
echo $cmd ; eval $cmd
