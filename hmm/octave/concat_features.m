[datadir workdir vboxpath maven_repo eclipse22ws]=win10
addpath(vboxpath)

mfc_prefix=["./tmp_genobs/bw_06"  ];
mfcFN=[mfc_prefix   ".mfc" ]

%reread test
mfcfile = fopen( mfcFN, 'r', 'b' );
[d,fp,dt,tc]=readhtk(mfcFN);
% fp,dt,tc;
% size(d)


flength=flength;
mfc_prefix=["./tmp_genobs/bw_08_" num2str(flength) ];
mfcFN=[mfc_prefix   ".mfc" ]
MFCCS1=d(:,1:flength);
MFCCS=(log(abs(MFCCS1)));
writehtk(mfcFN, MFCCS,fp,tc);
clear MFCCS;


%reread test
mfcfile = fopen( mfcFN, 'r', 'b' );
[d,fp,dt,tc]=readhtk(mfcFN);
fp,dt,tc;
size(d)


% h1=figure;
% plot([mean(d) ; var(d)]');
% legend('mean', 'var');
% waitfor(h1)
