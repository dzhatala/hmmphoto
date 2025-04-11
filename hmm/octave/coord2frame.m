%calculate image position converted to htk frame number
%fnumber : smallest frame number, start from 0
%coord : start from 0 
function [fnumber, complete_width, overpixel, padding_width]=coord2frame(position, block_size, stride)


if position < block_size
	fnumber=0;
	complete_width=block_size;
	overpixel=position-block_size;
	padding_width=block_size-position;
	return;
end

ratio1=(position-block_size)/stride+1;
fnumber=0;
if ratio1>=1
    fnumber=floor(ratio1);
end
complete_width=(fnumber-1)*stride+block_size;
overpixel=position-complete_width;
padding_width=stride-overpixel;
if overpixel==0
    padding_width=0;
end
return;