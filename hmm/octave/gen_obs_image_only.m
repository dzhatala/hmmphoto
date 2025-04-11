#! /bin/octave -qf
printf ("%s", program_name ());
arg_list = argv ();
for i = 1:nargin
  printf (" %s", arg_list{i});
endfor
printf ("\n");

% return
%load image and convert to .mfc htk 

pkg load signal
pkg load image
addpath(".");
addpath("./octave")

[datadir workdir vboxpath maven_repo eclipse22ws]=win10
addpath(vboxpath)
% return 
disp("generating observations .. ");
CHECK_TODO=0  ; % take care TODO list



imgname=[datadir "/labelme_01/bw_10.jpg"]

[A, map, alpha]= imread(imgname);

% max_A=max(max(A))
% min_A=min(min(A))
% A=im2bw(A);
A=rgb2gray(A);

h1=figure;imshow(A); colorbar;title("rgbplot"); waitfor(h1)
[imheight,imwidth]=size(A)
sA=size(size(A));
dimA=sA(2)
FORCE_BW=0;
if dimA != 2 && FORCE_BW==1
	disp(["error image is not black/white " num2str(dimA) "!=2" ]);
	return
end
% A(1,1:3)
% return ;
% irowb_end=min(3,numBlock_vertical)
h_blocksize=16
h_stride=16 %stride != 0

ratio1=(imwidth-v_block_size)/v_stride+1;
fnumber=0;
if ratio1>=1
	fnumber=floor(ratio1)
end
complete_width=(fnumber-1)*stride+h_block_size
overpixel=imwidth-complete_width
padding_width=v_stride-overpixel;
if overpixel==0
	padding_width=0;
end
padding_width

v_stride=32; %stride vertical
v_block_size=32;
ratio1=imheight /v_block_size
if stride_v >0
	ratio1=(imheight-block_size)/v_stride_v+1;
end
fnumber=0;
if ratio1>=1
	fnumber=floor(ratio1)
end

advance_v=v_stride; %multiplier
if advance_v==0
	advance_v=v_block_size;
end

complete_height=(fnumber-1)*advance_v+v_block_size
overpixel=imheight-complete_height

padding_height=advance_v-overpixel;
if overpixel==0
	padding_height=0;
end
padding_height
% return 


% disp('#### padding here ####')
% return
tmpdir="./tmp_genobs"

MFCCS=[];
row_width=imwidth
row_height=imheight;

numBlock_horz=floor( (row_width-h_block_size)/h_stride)+1
numBlock_vertical=floor((row_height-v_block_size)/advance_v)+1

irowb_end=numBlock_vertical
% irowb_end=numBlock_vertical-1
	
	for irowblock=1:irowb_end
	% for irowblock=1:3
		irowblock
		% TODO what block/frame is the labels ?
		
		y1=(irowblock-1)*advance_v+1;
		y2=y1+block_size-1;
		for ihorzblock=1:numBlock_horz
	
			x1=(ihorzblock-1)*stride+1;
			x2=x1+block_size-1 ;
			% if CHECK_TODO>0
				% disp ("TODO: no padding here");
				% return 
			% end
			imgblock=A(y1:y2,x1:x2); 
			
			Adct2=dct2(imgblock);
			coefs=zigzag(Adct2)';
			size_c=size(coefs);
			MFCCS=[MFCCS ; coefs];

		end
	end
% end
size_MFCCS=size(MFCCS);
disp("saving octave vars into gen_obs.mat");
% save ("gen_obs_bw_06.mat");
nSamples = size_MFCCS(1)

%HTK PARAMS
% sampPeriod = 100000*1E-7
sampSize = size_MFCCS(2)
% parmKind = 6 % MFCC six
[sampPeriod,parmKind]=mfcParams


mfc_prefix=["./tmp_genobs/bw_10"  ]
mfcFN=[mfc_prefix   ".mfc" ]
writehtk(mfcFN, MFCCS,sampPeriod,parmKind);
clear MFCCS;

%reread test
% mfcfile = fopen( mfcFN, 'r', 'b' );
% [d,fp,dt,tc]=readhtk(mfcFN);
% fp,dt,tc
% size(d)
