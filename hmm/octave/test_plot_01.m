pkg load signal
pkg load image
jpg_prefix=['G:\\rsync\\RESEARCHS\\table_detection\\source_code\\github\\data\\labelme_01\\bw_10' ]
jpgFN=[jpg_prefix   '.jpg' ]

RGB = imread(jpgFN);
I = rgb2gray(RGB);

J = dct2(I);
size_J=size(J)
h1=figure;
imshow(log(abs(J)),[])
% colormap parula
colorbar
% J(abs(J) < 10) = 0; 
% K = idct2(J);
% K = rescale(K);
% montage({I,K})
title('Original Grayscale Image (Left) and Processed Image (Right)')

waitfor(h1)