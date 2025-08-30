pkg load signal

%https://groups.google.com/g/comp.soft-sys.matlab/c/td_sbgAOltE?pli=1
%difference between matlab dct2 and FFTW lib(FFTW_REDFT10)
A=[0.2 0.3 1;
0 12 5;
0.3 0.3 1];

A
adct=dct2(A)

A=[0.2 0.3 1 0;
0 12 5 0 ;
0.3 0.3 1 0 ;
0 0 0 0
 ];

A
adct=dct2(A)

A=[
90.00 92.00 89.00 86.00;
90.00 92.00 91.00 89.00;
89.00 91.00 90.00 90.00;
89.00 90.00 90.00 91.00;
]
adct=dct2(A)


