pkg load statistics
file_path = fileparts(mfilename('fullpath'));
addpath(file_path)
addpath([file_path "./../"]) %for win10 

[datadir workdir vboxpath maven_repo eclipse22ws]=win10
addpath(vboxpath)

arg_list = argv ();
if nargin >0
	imgname=arg_list{1}
end

mfc_prefix=["./tmp_genobs/bw_10_gray_wavelet"  ]
mfcFN=[mfc_prefix   ".mfc" ]

if nargin >0
	mfcFN=arg_list{1}
end

%reread test
mfcfile = fopen( mfcFN, 'r', 'b' );
[d,fp,dt,tc]=readhtk(mfcFN);
fp,dt,tc
size_d=size(d)

h1=figure;
mean_data=mean(d)
max_data=max(d)
min_data=min(d)
% return ;
plot([mean_data ; var(d)]');
legend('mean', 'var');
h2=figure;

disp("showing selected coefs");
% dc=d(:,1:3,16:19); %selected coefs
dc=d(:,16:19); %selected coefs
max(max(dc))
min(min(dc))
size_dc=size(dc)
normplot(dc);
% normplot(d(2,:));
waitfor(h1)
waitfor(h2)


