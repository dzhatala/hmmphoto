[datadir workdir vboxpath maven_repo eclipse22ws]=win10% in win10.m 
addpath(vboxpath)
mfcFN='tmp/x.mfc'
% mfcFN='G:\\rsync\\RESEARCHS\\finger_board_detection_image\\github_jurnal\\hmm\\tmp\\280_IMG-20240723-WA0001.mfc';

mfcFN='tmp/x.mfc'
% mfcFN='c:\\rps\features\\280_IMG-20240101-WA0052.dct'
 disp(["load " mfcFN]);
mfcfile = fopen( mfcFN, 'r', 'b' );
[d,fp,dt,tc]=readhtk(mfcFN);
fp,dt,tc
size_d=size(d)
sd1=size_d(1)
% d
first_data=d(1,:)
second_data=d(2,:)
third_data=d(3,:)

% last_data= d(sd1,:)
% last_data_1= d(sd1-1,:)
% last_data_2= d(sd1-2,:)
% d(4001:4003,:)
% d(8608:8610,:)
% d
