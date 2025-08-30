#include <stdlib.h>
#include <math.h>
#include <fftw3.h>

#include <dct2_std.h>
#include "convert.h"
void compute2Ddct(const double* input, double* output, int rows, int cols) {
	int i=0, j=0;
	double factor=1.0;
	fftwf_r2r_kind kind1 =FFTW_REDFT10;
	fftwf_r2r_kind kind2 =FFTW_REDFT10;
	// size_t flags =FFTW_PATIENT;
	size_t flags =FFTW_ESTIMATE;
	
	if ((rows %2 != 0 )|| (cols %2 !=0)){
		fprintf(stderr, " matrix is NOT even %u x %u \r\n",rows,cols);
		exit(-1);
	}
	
	
	fftw_plan plan = fftw_plan_r2r_2d(rows, cols, input, output, kind1, kind2, flags);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
	
	factor = 4*sqrt(rows/2)*sqrt(cols/2);
	for (i = 0; i < rows; ++i) {
		for ( j = 0; j < cols; ++j) {
			output[i*rows+j]/=factor; 
		}
	}
	
	factor=sqrt(2);
	i = 0; 
	for ( j = 0; j < cols; ++j) {
		output[i*rows+j]/=factor; 
	}	
	
	for (i = 0; i < rows; ++i) {
		j=0;
		output[i*rows+j]/=factor;
	}
	//'thus ' is a consequence, google discussion
}

/** show snapshot lenxlen for an image with widht=nrows and height=ncols
in vector array from zero,zero (topleft)
 ***/
void dump_double_image(const  double *mtr,const size_t width
	,const size_t len){
	int i=0, j=0;
	for (i = 0; i < len; ++i) {
		for ( j = 0; j < len; ++j) {
			// printf("%.2f ",mtr[j]);
			printf("%.2f ",mtr[i * width + j]);
			// printf("%u,%u ",i,j);
			
			// printf("%.2f (%u,%u) ",mtr[i * width + j],i,j);
			
		}
		printf("\r\n");
	}
}
//float version BUT stop after len elementh
void dump_float_image(const  float *mtr,const size_t width
	,const size_t len){
	int i=0, j=0;
	for (i = 0; i < len/width+1; ++i) {
		for ( j = 0; j < width; ++j) {
			// printf("%.2f ",mtr[j]);
			printf("%.4f ",mtr[i * width + j]);
			// printf("%u,%u ",i,j);
			
			// printf("%.2f (%u,%u) ",mtr[i * width + j],i,j);
			if(i*width+j>=len-1)break;
		}
		printf("\r\n");
			if(i*width+j>=len-1)break;
	}
}

void seq_double_image(const  double *mtr,size_t len){
		size_t j=0;
		for ( j = 0; j < len; ++j) {
			// printf("%.2f ",mtr[i * width + j]);
			// printf("%u,%u ",i,j);
			
			printf("%.2f (%u,%u) ",mtr[j]);
			
		}
		
		printf("\r\n");


}

void dump_uchar_image(const unsigned char *mtr,size_t width, size_t height
	,size_t len){
	size_t i=0, j=0;
	size_t start=0;
	for (i = 0; i < len; ++i) {
		for ( j = 0; j < len; ++j) {
			printf("%u ",mtr[i * width+start + j]);
		}
		printf("\r\n");
	}
}


/** caller must CHECK start end length 

this must consistent with DbVectFromMatrix (octave,matlab)

[i][j] (row,column) == (y,x) (reversed cartesius) 

**/
void dumpDbMatrix(const DbMatrix mtr,int start_row
	, size_t start_col,int len_row, size_t len_col ){
	int i=0,j=0;
	for (i = start_row; i < start_row+len_row; ++i) {
		for ( j = start_col; j < start_col+len_col; ++j) {
			// printf("%u,%u ",i,j);
			printf("%.2f ",mtr[i][j]); 
		}
		printf("\r\n");
	}
}


/** copy from mtr and put into sub**/
void getSubblock(const DbMatrix mtr, DbMatrix sub, size_t start, size_t length ){
	int i=0,j=0;
	for (i = start; i < start+length; ++i) {
		for ( j = start; j < start+length; ++j) {
			// printf("i:%u j:%u val:%.2f",i,j,mtr[i][j]);fflush(stdout);
			sub[i-start][j-start]=mtr[i][j];
		}
		// printf("\r\n");
	}
}


/***do not reversed,
rows is the height
cols is the width of an image

@example
DbMatrix mat=CreateDbMatrix(gray.height,gray.width); 


***/
DbMatrix CreateDbMatrix(int rows, int cols){
	
	DbMatrix arr = (DbMatrix)malloc(rows * sizeof(double*));
        if (arr == NULL) {
            return NULL;
        }
        for (int i = 0; i < rows; i++) {
            arr[i] = (double*)malloc(cols * sizeof(double));
            if (arr[i] == NULL) {
                for(int j = 0; j < i; j++){
                  free(arr[j]);
                }
                free(arr);
                return NULL;
            }
        }
        return arr;
}

/**convert RGB to double before fft**/
void DoubleVectFromCharVect(double *target, const unsigned char *source,size_t length){
		size_t i=0;
		for (i=0;i<length;i++){
					target[i]=(double)source[i];
			
		}
}

/****
		CONVERT VECTOR grayscale into matrix 2 dimension
**/
void DbMatrixFromCharVect(DbMatrix target, const unsigned char *source,
					const size_t mat_width,const size_t mat_height
					){
		size_t i=0,j=0,id_vect=0;
		double gr=0;
		
		for (i=0; i<mat_height; i++){
			for (j=0;j<mat_width; j++){
				// gr=(double)id_vect;
				gr=(double)source[id_vect];
				target[i][j]=gr;
				id_vect++;
			}
			
		}
			
}

/*** from matrix to vector, caller must aware of length  ***/
/*void CharVectFromDbMatrix(const DbMatrix source,  unsigned char *target,
	size_t mat_size,const size_t start, const size_t length){

		size_t i=0,j=0,id_vect=0;
		for (i = start; i < start+length; ++i) {
			for ( j = start; j < start+length; ++j) {
				// printf("i:%u j:%u val:%.2f",i,j,source[i][j]);fflush(stdout);
				target[i*mat_size+start+j]=(unsigned char)source[i-start][j-start];
			}
		// printf("\r\n");
		}
						
						
}
*/


void DbVectFromMatrix(const DbMatrix source,double *target,  
					const size_t start_col,const size_t len_col,
					const size_t start_row,const size_t len_row	){
	size_t idvec=0,i=0,j=0;
											   /* i++ is WRONG*/
	// printf("dct2_std #1\r\n");
	
	/** inner i outer j vs inner j outer i has DIFFRENT/TRANSPOSE?
	because we're linerizing/unfolding matrix into vector...
	***/
	for (i = start_row; i < start_row+len_row; ++i) {
		for ( j = start_col; j < start_col+len_col; ++j) {
			// printf("%.2f ",source[j][i]);
			// printf("%.2f ",source[i][j]);
			target[idvec]=source[i][j];
			idvec++;
		}
		// printf("\r\n");
	}
					

}


 
int initializeParametersBlock(const ImageGrayscale *gray,SubBlockInfo *info, const size_t blocksize){
	
	info->width=gray->width;
	info->height=gray->height;	
	info->size=blocksize;
	if(info->row_stride==0){
		info->row_stride=info->size; //Nan protection
	}
	
	if(info->col_stride==0){ //horizontal along into rigtest width size
		info->col_stride=info->size; //Nan protection
	}
	
	int ratio1=info->width/info->row_stride; //floored
	info->total_frame_cols=ratio1;
	int over_pixel=info->width-ratio1*info->size;
	if(over_pixel>0){
		info->total_frame_cols++;
		info->right_pad=info->size-over_pixel;
	}
	
	
	ratio1=info->height/info->col_stride;
	info->total_frame_rows=ratio1;
	over_pixel=info->height-ratio1*info->size;
	if(over_pixel>0){
		info->total_frame_rows++;
		info->bottom_pad=info->size-over_pixel;
	}
	
	return 0;
}

					
void frame_summary(SubBlockInfo *info){
	
	printf("\r\n");
	printf("img width:%u\r\n",info->width);
	printf("img height:%u\r\n",info->height);
	printf("row stride:%u\r\n",info->row_stride);
	printf("col stride:%u\r\n",info->col_stride);
	printf("total_frame_cols:%u\r\n",info->total_frame_cols);
	printf("right_pad:%u\r\n",info->right_pad);
	printf("total_frame_rows:%u\r\n",info->total_frame_rows);
	printf("bottom_pad:%u\r\n",info->bottom_pad);
	

	
}



void ZeroDbMatrix(DbMatrix mat, size_t row_start, size_t row_len,
	size_t col_start, size_t col_len){
	
        for (int i = row_start; i < row_start+row_len; ++i) {
			for (int j = col_start; j < col_start+col_len; ++j) {
				mat[i][j]=0;
			}
        }
}

