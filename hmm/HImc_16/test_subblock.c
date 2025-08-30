/* 

	test_Subblock start length;

$ ./test_subblock 2 3
0.00    1.00    2.00    3.00    4.00    5.00    6.00    7.00    8.00    9.00
1.00    1.00    2.00    3.00    4.00    5.00    6.00    7.00    8.00    9.00
2.00    2.00    2.00    3.00    4.00    5.00    6.00    7.00    8.00    9.00
3.00    3.00    3.00    3.00    4.00    5.00    6.00    7.00    8.00    9.00
4.00    4.00    4.00    4.00    4.00    5.00    6.00    7.00    8.00    9.00
5.00    5.00    5.00    5.00    5.00    5.00    6.00    7.00    8.00    9.00
6.00    6.00    6.00    6.00    6.00    6.00    6.00    7.00    8.00    9.00
7.00    7.00    7.00    7.00    7.00    7.00    7.00    7.00    8.00    9.00
8.00    8.00    8.00    8.00    8.00    8.00    8.00    8.00    8.00    9.00
9.00    9.00    9.00    9.00    9.00    9.00    9.00    9.00    9.00    9.00
start:2 length:3
2.00    3.00    4.00
3.00    3.00    4.00
4.00    4.00    4.00
2.00    3.00    4.00
3.00    3.00    4.00
4.00    4.00    4.00


*/
#include <stdio.h>
#include <stdlib.h>

#include "convert.h"
#include <dct2_std.h>

MagickWand *magick_wand;
MagickBooleanType status;
/** **/
int main(int argc, char** argv){
	int rows=10, cols=10,x=0,y=0;
	unsigned char *data=NULL;
   
	if(argc<3){
		fprintf(stderr,"usage: test_subblock start _length\r\n");
		fprintf(stderr,"start from 0 to 9 \r\n");
		return -1;
	}
	
	
   
	data=calloc(sizeof(char),rows*cols);
	DbMatrix mat=CreateDbMatrix(rows,cols);
		for (x=0;x<rows;x++){
			for(y=0;y<rows;y++){
			// mat[x][y]=(double)(x-y)/(x+y+1);
			mat[x][y]=(double)(x>y?x:y);
		}
	}
	dumpDbMatrix(mat,0,0,cols,cols);
	x=atoi(argv[1]);
	if(x<0)x=0;
	y=atoi(argv[2]);
	if(x+y>rows-1)y=cols-x;
	if(y<0){
			fprintf(stderr,"len of blocks minimal 1\r\n");
			return -1;
	}
	printf("start:%i length:%i\r\n",x,y);
	dumpDbMatrix(mat,x,x,y,y);
	
	DbMatrix sub=CreateDbMatrix(y,y);
	
	getSubblock(mat, sub, x, y );
	dumpDbMatrix(sub,0,0,y,y);
	
	return 0;
}

