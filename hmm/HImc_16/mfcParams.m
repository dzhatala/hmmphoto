%HTK PARAMS
function [sampPeriod,parmKind,block_size,stride_v,stride]=mfcParams
sampPeriod = 100000*1E-7;
parmKind = 6; % MFCC six
block_size=16;
stride=6;
stride_v=6;
return