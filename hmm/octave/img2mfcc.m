#! /bin/octave -qf
printf ("%s", program_name ());
arg_list = argv ();
for i = 1:nargin
  printf (" %s", arg_list{i});
endfor
printf ("\n");

file_path = fileparts(mfilename('fullpath'));

% return
%load image and convert to .mfc htk 

% pkg load signal
pkg load image
pkg load signal
addpath(".");
% addpath("./octave")
addpath(file_path)
addpath([file_path "./../"]) %for win10 

[datadir workdir vboxpath maven_repo eclipse22ws]=win10
addpath(vboxpath)
% return 
disp("generating observations .. ");
CHECK_TODO=0  ; % take care TODO list


tmpdir="./tmp"
imgname=[datadir "/labelme_01/many_lines.jpg"]
if nargin >0
	imgname=arg_list{1}
end
if nargin >1
	tmpdir=arg_list{2}
end
[dir outfname ext]= fileparts(imgname);

%test 
save([tmpdir "/" outfname ".sav"])
% clear all;
load ([tmpdir "/" outfname ".sav"])
% return


	[A, map, alpha]= imread(imgname);
	A=rgb2gray(A);
	
	% h1=figure;imshow(A); colorbar;
	% waitfor(h1);
	
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
	block_size=4
	stride=4 %stride != 0 %stride is advancement
	
	ratio1=(imwidth-block_size)/stride+1
	fnumber=0;
	if ratio1>=1
		fnumber=floor(ratio1);
	end
	complete_width=(fnumber-1)*stride+block_size
	overpixel=imwidth-complete_width
	padding_width=stride-overpixel;
	if overpixel==0
		padding_width=0;
	end
	padding_width
	if padding_width > 0
		fnumber=fnumber+1;
		A=padarray(A, [0  padding_width],0,"post");
	end
	numBlock_horz=fnumber
		% return ;
	% stride_v=0; %stride vertical
	stride_v=stride; %stride vertical
	ratio1=imheight /block_size
	if stride_v >0
		ratio1=(imheight-block_size)/stride_v+1;
	end
	fnumber=0;
	if ratio1>=1
		fnumber=floor(ratio1);
	end
	
	advance_v=stride_v; %multiplier
	if advance_v==0
		advance_v=block_size;
	end
	
	complete_height=(fnumber-1)*advance_v+block_size
	overpixel=imheight-complete_height

	padding_height=advance_v-overpixel;
	if overpixel==0
		padding_height=0;
	end
	padding_height
	if padding_height > 0
		fnumber=fnumber+1;
		A=padarray(A, [ padding_height 0],0,"post");
	end
	numBlock_vertical=fnumber
	% return 
	
	
	% disp('#### padding here ####')
	% return

MFCCS=[];
	row_width=imwidth
	row_height=imheight;
	
	% numBlock_horz=floor( (row_width-block_size)/stride)+1
	% numBlock_vertical=floor((row_height-block_size)/advance_v)+1
	

% for rbig=1:rnum
% for rbig=1:1
	% return
	
	% if numBlock_vertical <3
		% disp(['not enough 3 data .. ']);
		% return
	% end

	% numlabel=row1.getColumnsNumber()
	% labelcol=0
	% if numlabel>0
		% labelcol=row1.getColumns()
	% end 
	
	irowb_end=numBlock_vertical
	% irowb_end=numBlock_vertical-1
	
	for irowblock=1:irowb_end
	% for irowblock=1:3
		%irowblock
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
			
			% x1,y1,x2,y2 
			% return 
			
			imgblock=A(y1:y2,x1:x2); 
			
			% Adct2=dct2(imgblock);
			% size_imgblock=size(imgblock)
			
			% isnumeric=isnumeric(imgblock)
			% isinteger=isinteger(imgblock)
			% return
			imgblock+=0.0;% islogical()?
			coefs=feature_grayli_2000(imgblock) ; %dct2 inside funct.
			size_c=size(coefs);
			% return
			
			% pad
		

			if sum(size_c)!=7  %6 coefs only
				% sum(size_c)
				% block_size+1
				disp ('bad grayli vector size');
				return
			end


			% if sum(size_c)!=block_size*block_size+1
				% sum(size_c)
				% block_size+1
				% disp ('bad dct vector size');
				% return
			% end

			% return
			
			MFCCS=[MFCCS ; coefs];

		end
	end
% end
size_MFCCS=size(MFCCS)
delta= diff(MFCCS(:,2:6));

accell=diff(delta);
% accell=accell';
delta=padarray(delta,[1 0 ], 0,'pre');
accell=padarray(accell,[2 0 ], 0, 'pre');
size_delta=size(delta)
size_accell=size(accell)

%HTK PARAMS
% sampPeriod = 100000*1E-7
MFCCS=[MFCCS  delta  accell];
size_MFCCS=size(MFCCS)
feature_size = size_MFCCS(2)

disp("saving octave vars into gen_obs.mat"); save -binary tmp/mfccs.mat MFCCS;


MARKER_LENGTH=3
if MARKER_LENGTH >0
	padded=[];
	pads=zeros(MARKER_LENGTH,feature_size)+3.4e37;
	sz_pad=size(pads)
	% return ;

	fr_start=1;
	disp("MARKER adding..");
	for irowblock=1:irowb_end
		% disp (["padding MARKER ... " num2str(irowblock) ]); 
	    fr_end=fr_start+numBlock_horz-1;
		padded=[padded ; MFCCS(fr_start:fr_end,:) ; pads];
		% size_padd=size(padded);
		fr_start=fr_end+1;
	end;

	
end
MFCCS=padded; clear padded;
size_MFCCS=size(MFCCS)
sampSize = size_MFCCS(2)
% return ;
% parmKind = 6 % MFCC six
[sampPeriod,parmKind]=mfcParams

%outdir="G:/rsync/RESEARCHS/table_detection/source_code/github/bw_14/tmp_genobs"; 
outdir="./tmp_genobs"; 
if nargin >1
	outdir=arg_list{2}
end

mfc_prefix=[outdir "/" outfname  ];
mfcFN=[mfc_prefix   ".mfc" ]
writehtk(mfcFN, MFCCS,sampPeriod,parmKind);
disp(["written " mfcFN]);

%reread test
disp(["reload " mfcFN]);
mfcfile = fopen( mfcFN, 'r', 'b' );
[d,fp,dt,tc]=readhtk(mfcFN);
fp,dt,tc
size_d=size(d)
should_length=(numBlock_horz+MARKER_LENGTH)*numBlock_vertical
if size_d(1) != should_length
	error (["size NOT match" ]);
else 
	disp (["size ALREADY match" ]);
end

sum_sum_check=sum(sum(MFCCS-d))

%test 
%disp("saving octave vars into mat file"); save -binary tmp/mfccs_padded.mat MFCCS;
% clear MFCCS;save([tmpdir "/" outfname ".sav"])
clear all;
% load ([tmpdir "/" outfname ".sav"])
