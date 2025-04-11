[datadir workdir vboxpath maven_repo eclipse22ws]=win10
addpath(vboxpath)

mfc_prefix=["./tmp_genobs/bw_07_20"  ]
mfcFN=[mfc_prefix   ".mfc" ]

%reread test
mfcfile = fopen( mfcFN, 'r', 'b' );
[d,fp,dt,tc]=readhtk(mfcFN);
fp,dt,tc
size(d)

h1=figure;
plot([mean(d) ; var(d)]');
legend('mean', 'var');
h2=figure;
imshow(abs(d));
colorbar
max_d=(max(max(d)))
min_d=(min(min(d)))
waitfor(h1)
waitfor(h2)