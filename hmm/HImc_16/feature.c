#include <stdio.h>
#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HWave.h"
#include "HAudio.h"
#include "HParm.h"
#include "HLabel.h"
#include "feature.h"

/***
$ ./testHParm tmp/x.jpeg tmp/x.mfc

G:\rsync\rsync_2025_21_02\RESEARCHS\htk_cygwin\HTK-3.4.1\htk\HImCopy>call C:\rps\Octave-9.4.0\mingw64\bin\octave.bat test_htk_MFC.m .\tmp\x.jpeg .\tmp
vboxpath = g:\rsync\RESEARCHS\htkbook_to_c_code\matlab\voicebox
maven_repo = C:\Users\User\.m2\repository
eclipse22ws = G:\rsync\RESEARCHS\table_detection\source_code\github\x270eclipse22_ws
datadir = ../data
workdir = .
vboxpath = g:\rsync\RESEARCHS\htkbook_to_c_code\matlab\voicebox
maven_repo = C:\Users\User\.m2\repository
eclipse22ws = G:\rsync\RESEARCHS\table_detection\source_code\github\x270eclipse22_ws
mfcFN = tmp/x.mfc
load tmp/x.mfc
fp = 0.010000
dt = 6
tc = 6
size_d =

    4   16

*/

//test multiple float
void testSaveMFCFile(const char*s, short vectorSize,long numSamples,float *data){
	FILE *f=fopen(s,"wb");
	if(f==NULL){
		HError(20000,"open %s \r\n failed",s);
	}
	OpenForSaveMFCFile(f,  vectorSize,  numSamples*2);
	appendFloat(f, vectorSize, numSamples,data);
	appendFloat(f, vectorSize, numSamples,data);//demonstrate appending ??? check in octave ?
	closeMFCFile(f);
}

/**TOTAL numsamples must be calculated first***/
void OpenForSaveMFCFile(FILE *f, short vectorSize,long numSamples){
	
    // FILE *f;  
	short kind=MFCC;
    Boolean *bSwap=FALSE;

	// f=fopen(s,"wb");//open a file for writing
	
	if(f==NULL){
		HError(20000,"error open mfc target\r\n");
	}
	kind = MFCC; //kind = cf->tgtPK & ~(HASNULLE|HASVQ);
  
	long samPeriod=1e5; //mimicking sampling period pretending waveform
	size_t numbytes=sizeof(float); //HTK use 4 byte floating point ?

	// return 0;//debug
	
	WriteHTKHeader(f,numSamples,samPeriod,vectorSize*numbytes,kind,&bSwap);
	// void WriteFloat(FILE *f, float *x, int n, Boolean binary);
	return f;
}

void appendFloat(FILE *f, const short vectorSize,const long numSamples,const float *data){
	
	// printf("appendfloat %zu %zu %i %.4f ",vectorSize, numSamples,data[0]);fflush(stdout);

		// for (int i=0;i<18 ; ++i){
			// printf("%.4f ",data[i]);fflush(stdout);//debug
		// }
		WriteFloat(f, data, numSamples*vectorSize, TRUE); //does this WriteFloat appending ?
		// WriteFloat(f, data, numSamples, TRUE); //does this WriteFloat appending ?

}

void closeMFCFile(FILE *f){
		fclose(f);
}



/**caller must allocate memory 16 float**/
void feature_hatala_16(const double *fft_result
		,const size_t dct_size,const float *previousFrame, float *gray_lee){
	int i=0;
	float *gray_18=calloc(sizeof(float),18);
	feature_hatala_18(fft_result,dct_size,previousFrame,gray_18);

	for( i=0; i<6; ++i){
		// printf("%zu\r\n",i);fflush(stdout);
		gray_lee[i]=gray_18[i];
	}
	/**6,7,8,9,10 = 7 - 11 **/
	for(i=6; i<11; ++i){
		gray_lee[i]=gray_18[i+1];

	}		
	/** 11,12,13,14,15 = 13,14,15,16,17  **/
	for(i=11; i<16; ++i){
		gray_lee[i]=gray_18[i+2];

	}		
	free(gray_18);//segmentation fault
}		
/**caller must allocate memory 18 float **/
void feature_hatala_18(const double *fft_result
		,const size_t dct_size,const float *previousFrame, float *gray_lee_18){

	int p;
	double *fft_copy=calloc(sizeof(double),16); //assume 4x4
	memcpy(fft_copy,fft_result,16*sizeof(double));
	for(p=0;p<16; ++p){
		if(fft_copy[p]<0)fft_copy[p]*=-1; //absolute
		
	}
	gray_lee_18[0]=(float)fft_result[0]; //oc: 1,1
	gray_lee_18[1]=(float)fft_copy[4];  //0,1 0+1
	gray_lee_18[2]=(float)fft_copy[1];   //1,0 1*4+0
	
	
	// for i=3:4
		// for j=1:2
			// sum+=Abs_d(i,j);
		// end
	// end

	
	gray_lee_18[3]=0;
	gray_lee_18[3]+=(float)fft_copy[8]; // oc 3,1 2*4+0
	gray_lee_18[3]+=(float)fft_copy[9]; // oc 3,2 2*4+1
	gray_lee_18[3]+=(float)fft_copy[12]; // oc 4,1 3*4+0
	gray_lee_18[3]+=(float)fft_copy[13]; // oc 4,2 3*4+1
	gray_lee_18[3]/=4;
	
	gray_lee_18[4]=0;
	gray_lee_18[4]+=(float)fft_copy[2]; // oc 
	gray_lee_18[4]+=(float)fft_copy[3]; // oc 
	gray_lee_18[4]+=(float)fft_copy[6]; // oc 
	gray_lee_18[4]+=(float)fft_copy[7]; // oc 
	gray_lee_18[4]/=4;


	gray_lee_18[5]=0;
	gray_lee_18[5]+=(float)fft_copy[10]; // oc 
	gray_lee_18[5]+=(float)fft_copy[11]; // oc 
	gray_lee_18[5]+=(float)fft_copy[14]; // oc 
	gray_lee_18[5]+=(float)fft_copy[15]; // oc 
	gray_lee_18[5]/=4;

	
	for ( p=0; p<6 ; ++p){
		gray_lee_18[6+p]=gray_lee_18[p]-previousFrame[p];
		// printf("%zu\r\n",i);fflush(stdout);
	}
	
	// printf("dumping coefs 12> \r\n");fflush(stdout);
	for (p=0; p<6 ; ++p){
		gray_lee_18[12+p]=gray_lee_18[6+p]-previousFrame[6+p];
		// printf("%zu:%.4f-%.4f=%.4f\r\n",12+p,gray_lee_18[6+p]
			// ,previousFrame[6+p],gray_lee_18[12+p]);fflush(stdout);
		
	}
	// printf("\r\n");
	free(fft_copy);

}
			
