[datadir workdir vboxpath maven_repo eclipse22ws]=win10
addpath(vboxpath)

% mfc_prefix=["./tmp_genobs/bw_06"  ]
mfc_prefix=["G:\\rsync\\RESEARCHS\\table_detection\\source_code\\github\\bw_08\\tmp_genobs\\bw_08_20"  ]
mfc_prefix=["G:\\rsync\\RESEARCHS\\table_detection\\source_code\\github\\bw_08\\tmp_genobs\\bw_08"  ]
mfcFN=[mfc_prefix   ".mfc" ]

%reread test
mfcfile = fopen( mfcFN, 'r', 'b' );
[d,fp,dt,tc]=readhtk(mfcFN);
fp,dt,tc
size_d=size(d)

h1=figure;
mean_d=mean(d)

% return ;
plot([mean_d ; var(d)]');
legend('mean', 'var');

% h2=figure;
% absd=abs(d);
% maxd=max(max(absd));

% imshow(abs(d)*255/maxd,[])

waitfor(h1)
% waitfor(h2)