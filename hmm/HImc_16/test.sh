cmd="./HImCopy -T 15 -C config_image_mfc.txt -S image.lst"
trace_level="1"  #TOP
trace_level="2"  #IMG
trace_level="4"  #FRAME
trace_level="8"  #DCT
trace_level="15"  # 8+...
trace_level="0"  # 8+...
trace_level="1"  # 8+...
cmd="./HImc_16 -T ${trace_level} -C config_image_mfc.txt -S image.lst"
echo $cmd ;
eval $cmd
