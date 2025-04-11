%gen MLF and gen 
%
#! /bin/octave -qf
printf ("%s", program_name ());
arg_list = argv ();
for i = 1:nargin
    printf (" %s", arg_list{i});
end


[datadir workdir vboxpath maven_repo eclipse22ws]=win10
fn=[datadir "/labelme_01/bw_05.json"]
imgname=[datadir "/labelme_01/bw_05.jpg"]


printf ("\n");

%load image and convert to .mfc htk

pkg load signal
pkg load image
addpath(".");
addpath("./octave")
[sampPeriod,parmKind,block_size,stride_v,stride]=mfcParams

addpath(vboxpath)
% return
disp("generating observations .. ");
CHECK_TODO=0  ; % take care TODO list

lib=[maven_repo "/com/googlecode/json-simple/json-simple/1.1.1/json-simple-1.1.1.jar"]
javaaddpath (lib);

lib=[maven_repo "/log4j/log4j/1.2.17/log4j-1.2.17.jar"]
javaaddpath (lib);


lib=[eclipse22ws "/test_01/target/classes"]
javaaddpath (lib);



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

jo1=javaObject("labelme.LabelmeTableParser", fn);
jo1.parse();
rnum=jo1.getRowNumber()

if rnum <=0 
	return
end



rows_startY1=[];
rows_frames=[];

for ir=1:rnum
	row=jo1.getRow(ir-1);
	
	Y1=row.getY1();
	rows_startY1=[rows_startY1 Y1]
	
	
	%cek columns here 

	% row.getY2()
	% [fnumber, complete_width, overpixel, padding_width]=coord2frame(Y1, block_size, stride);
	

end

mfc_prefix=["./tmp_genobs/bw_5.jpg"  ]
mfcFN=[mfc_prefix   ".lab" ]
