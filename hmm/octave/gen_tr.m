#! /bin/octave -qf
printf ("%s", program_name ());
arg_list = argv ();
for i = 1:nargin
  printf (" %s", arg_list{i});
endfor
printf ("\n");

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
	[imheight,imwidth]=size(A)
	sA=size(size(A));
	dimA=sA(2)
	if dimA != 2
		disp(["error image is not black/white " num2str(dimA) "!=2" ]);
		return
	end
	% A(1,1:3)
	% return ;
	% irowb_end=min(3,numBlock_vertical)
	block_size=16
	stride=6
	
	ratio1=(imwidth-block_size)/stride+1;
	fnumber=0;
	if ratio1>=1
		fnumber=floor(ratio1);
	end
	fnum_horz=fnumber
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
		fnumber=floor(ratio1);
	end
	fnum_ver=fnumber

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

[sampPeriod,parmKind]=mfcParams


mfc_prefix=["./tmp_genobs/bw_5.jpg"  ]
mfcFN=[mfc_prefix   ".mfc" ]

% reread test
mfcfile = fopen( mfcFN, 'r', 'b' );
[d,fp,dt,tc]=readhtk(mfcFN);
fp,dt,tc
size(d)
check1=fnum_horz*fnum_ver
