function [coefs,Adct2]=feature_wavelet_db8(imgblock)
	Adct2=fwt(imgblock,'db8',3);
	% size_imgblock=size(imgblock)
	coefs=zigzag(Adct2)';
			