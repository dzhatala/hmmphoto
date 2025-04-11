#! /bin/octave -qf
printf ("%s", program_name ());
arg_list = argv ();
for i = 1:nargin
  printf (" %s", arg_list{i});
endfor
printf ("\n");

% return
%load image and convert to .mfc htk 

% pkg load signal
pkg load image
pkg load ltfat
addpath(".");
addpath("./octave")

[datadir workdir vboxpath maven_repo eclipse22ws]=win10
addpath(vboxpath)
% return 
disp("generating observations .. ");
CHECK_TODO=0  ; % take care TODO list

% lib=[maven_repo "/com/googlecode/json-simple/json-simple/1.1.1/json-simple-1.1.1.jar"]
% javaaddpath (lib);

% lib=[maven_repo "/log4j/log4j/1.2.17/log4j-1.2.17.jar"]
% javaaddpath (lib);


% lib=[eclipse22ws "/test_01/target/classes"]
% javaaddpath (lib);


imgname=[datadir "/labelme_01/bw_05.jpg"]
% fn=[datadir "/labelme_01/bw_05.json"];
% jo1=javaObject("labelme.LabelmeTableParser", fn);
% jo1.parse();
% rnum=jo1.getRowNumber()

	[A, map, alpha]= imread(imgname);
	% max_A=max(max(A))
	% min_A=min(min(A))
	A=im2bw(A);
	% A=rgb2gray(A);
	
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
	block_size=16
	stride=8 %stride != 0
	
	ratio1=(imwidth-block_size)/stride+1;
	fnumber=0;
	if ratio1>=1
		fnumber=floor(ratio1)
	end
	complete_width=(fnumber-1)*stride+block_size
	overpixel=imwidth-complete_width
	padding_width=stride-overpixel;
	if overpixel==0
		padding_width=0;
	end
	padding_width

		% return ;
	% stride_v=0; %stride vertical
	stride_v=stride; %stride vertical
	ratio1=imheight /block_size
	if stride_v >0
		ratio1=(imheight-block_size)/stride_v+1;
	end
	fnumber=0;
	if ratio1>=1
		fnumber=floor(ratio1)
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
	% return 
	
	
	% disp('#### padding here ####')
	% return
tmpdir="./tmp_genobs"

MFCCS=[];
	% jo1.doTest();
	% disp (rnum)
	% row1=jo1.getRow(rbig-1);
	% row1.toString()
	% x1=row1.getX1()
	% x2=row1.getX2()
	% y1=row1.getY1()
	% y2=row1.getY2()
	% row_width=x2-x1 
	row_width=imwidth
	% if imwidth != row_width
		% disp(["WARNING row_width:" num2str(row_width) "<> imwidth:" num2str(imwidth)]);
		% return
	% end
	% return
	% row_height=y2-y1
	row_height=imheight;
	
	numBlock_horz=floor( (row_width-block_size)/stride)+1
	numBlock_vertical=floor((row_height-block_size)/advance_v)+1
	

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
			
			% Adct2=dct2(imgblock);
			% size_imgblock=size(imgblock)
			
			% isnumeric=isnumeric(imgblock)
			% isinteger=isinteger(imgblock)
			% return
			imgblock+=0.0;% islogical()?
			Adct2=fwt(imgblock,'db8',3);
			% size_imgblock=size(imgblock)
			
			coefs=zigzag(Adct2)';
			size_c=size(coefs); 
			if sum(size_c)!=block_size*block_size+1
				sum(size_c)
				block_size+1
				disp ('bad vector size');
				return
			end
			% return
			
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


mfc_prefix=["./tmp_genobs/bw_10_gray_wavelet"  ]
mfcFN=[mfc_prefix   ".mfc" ]
writehtk(mfcFN, MFCCS,sampPeriod,parmKind);
clear MFCCS;

%reread test
mfcfile = fopen( mfcFN, 'r', 'b' );
[d,fp,dt,tc]=readhtk(mfcFN);
fp,dt,tc
size(d)
