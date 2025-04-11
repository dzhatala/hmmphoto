function [coefs,Adct2]=feature_grayli_2000(imgblock)
size_block=size(imgblock);
if max(size_block) != 4
	error(["block size " num2str(size_block) " != 4 4"]);
end	
% imgblock
Adct2=dct2(imgblock);
coefs=zeros(1,6);
coefs(1,1)=Adct2(1,1);
Abs_d=abs(Adct2);
coefs(1,2)=Abs_d(2,1);
coefs(1,3)=Abs_d(1,2);
% coefs(1,4)=
sum=0.0;
for i=3:4
	for j=1:2
		sum+=Abs_d(i,j);
	end
end
coefs(1,4)=sum/4.0;

sum=0.0;
for j=3:4
	for i=1:2
		sum+=Abs_d(i,j);
	end
end
coefs(1,5)=sum/4.0;


sum=0.0;
for j=3:4
	for i=3:4
		sum+=Abs_d(i,j);
	end
end
coefs(1,6)=sum/4.0;



