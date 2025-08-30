/* ----------------------------------------------------------- */
/*                                                             */
/*                          ___                                */
/*                       |_| | |_/   SPEECH                    */
/*                       | | | | \   RECOGNITION               */
/*                       =========   SOFTWARE                  */ 
/*                                                             */
/*                                                             */
/* ----------------------------------------------------------- */
/*         Copyright: Microsoft Corporation                    */
/*          1995-2000 Redmond, Washington USA                  */
/*                    http://www.microsoft.com                 */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*      File: HIMcopy.c: Copy Image to MFCC       */
/*      author: dzulqarnaenhatala@gmail.com       */
/* ----------------------------------------------------------- */

char *himcopy_version = "!HVER!HImCopy:   3.4.1 [dzhatala@github.com 12/03/25]";
char *himcopy_vc_id = "$Id: HImCopy.c,v 1.1.1.1 2025/10/11 ";

#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HSigP.h"
#include "HWave.h"
#include "HVQ.h"
#include "HAudio.h"
#include "HParm.h"
#include "HLabel.h"
#include "HModel.h"

#include "MagickWand/MagickWand.h"
#include "dct2_std.h"
#include "convert.h" /***/
#include "feature.h"

/* -------------------------- Trace Flags & Vars ------------------------ */
/**octal **/
#define T_TOP     001           /* basic progress reporting */
#define T_IMG     002           /* report file formats and parm kinds */
#define T_FRAME   004           /* output segment label calculations */
#define T_DCT 	  010           /* output segment label calculations */
#define T_MEM     020           /* debug memory usage */

static int  trace  = 0;         /* Trace level */




typedef struct _TrList *TrPtr;  /* simple linked list for trace info */
typedef struct _TrList {      
   char *str;                   /* output string */
   TrPtr next;                  /* pointer to next in list */
} TrL;
static TrL trList;              /* 1st element in trace linked list */
static TrPtr trStr = &trList;   /* ptr to it */

static int traceWidth = 70;     /* print this many chars before wrapping ln */

static ConfParam *cParm[MAXGLOBS];
static int nParm = 0;            /* total num params */

/* ---------------------- Global Variables ----------------------- */

FileFormat srcFF     = UNDEFF;   /* I/O configuration options */
FileFormat tgtFF     = UNDEFF;
FileFormat srcLabFF  = UNDEFF;
FileFormat tgtLabFF  = UNDEFF;
ParmKind srcPK       = ANON;
ParmKind tgtPK       = ANON;
HTime srcSampRate    = 0.0;
HTime tgtSampRate    = 0.0;
Boolean saveAsVQ = FALSE;
int swidth0 = 1;

Boolean convert_2_gray =TRUE;

static HTime st=0.0;            /* start of samples to copy */
static HTime en=0.0;            /* end of samples to copy */
static HTime xMargin=0.0;       /* margin to include around extracted labs */
static Boolean stenSet=FALSE;   /* set if either st or en set */
static int labstidx=0;          /* label start index (if set) */
static int labenidx=0;          /* label end index (if set) */
static int labRep=1;            /* repetition of named label */
static int auxLab = 0;          /* auxiliary label to use (0==primary) */
static Boolean chopF = FALSE;   /* set if we should truncate files/trans */

static LabId labName = NULL;    /* name of label to extract (if set) */
static Boolean useMLF=FALSE;    /* set if we are saving to an mlf */
static Boolean labF=FALSE;      /* set if we should  process label files too */
static char *labDir = NULL;     /* label file directory */
static char *outLabDir = NULL;  /* output label dir */
static char *labExt = "lab";    /* label file extension */

static Transcription *trans=NULL;/* main labels; cat all input to this */
static HTime off = 0.0;         /* length of files appended so far */

/* ---------------- zoel ------------------------- */
MagickWand *magick_wand;
MagickBooleanType status;
static ImageGrayscale gray; /**current grayscale data*/
static SubBlockInfo blkinfo;
static char imageSRCFN[256];
static Boolean use_delta5=TRUE;
static use_accel5=TRUE;
/**MARKER */
#define MARKER_LENGTH 3
static float MARKER_OBS[]={3.4E+37,3.4E+37,3.4E+37,3.4E+37,3.4E+37,3.4E+37,3.4E+37,3.4E+37,3.4E+37,
                                 3.4E+37,3.4E+37,3.4E+37,3.4E+37,3.4E+37,3.4E+37,3.4E+37,3.4E+37,3.4E+37};//default is 18

static float gray_lee[]={0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0};//default is 18
static float gray_lee_prev[]={0,0,0,0,0,0,0,0,0,
                       0,0,0,0,0,0,0,0,0}; //default is 18

// static size_t []DCT_DEBUG[]={0,35,46}; errod undefined?

static size_t DCT_DEBUG[]={0,1,2,4000,4001,4002,8607,8608,8609};
// static size_t DCT_DEBUG[]={0};
// static size_t DEBUG_EL=3;//size of DCT_DEBUG
static size_t DEBUG_EL=9;//size of DCT_DEBUG


/* ---------------- Memory Management ------------------------- */

#define STACKSIZE 100000        /* assume ~100K wave files */
static MemHeap iStack;          /* input stack */
static MemHeap oStack;          /* output stack */
static MemHeap cStack;          /* chop stack */
static MemHeap lStack;          /* label i/o  stack */
static MemHeap tStack;          /* trace list  stack */

/* ---------------- Process Command Line ------------------------- */

#define MAXTIME 1E13            /* maximum HTime (1E6 secs) for GetChkdFlt */

void ReportUsage(void)
{
   printf("\nUSAGE: HImCopyCopy [options] src [ + src ...] tgt ...\n\n");
   printf(" Option                                       Default\n\n");
   /*
   printf(" -a i     Use level i labels                  1\n");
   printf(" -e t     End copy at time t                  EOF\n");
   printf(" -i mlf   Save labels to mlf s                null\n");
   printf(" -l dir   Output target label files to dir    current\n");
   printf(" -m t     Set margin of t around x/n segs     0\n");
   printf(" -n i [j] Extract i'th [to j'th] label        off\n");
   printf(" -s t     Start copy at time t                0\n");
   printf(" -t n     Set trace line width to n           70\n");
   printf(" -x s [n] Extract [n'th occ of] label  s      off\n");
   */
   // PrintStdOpts("FGILPOX");
   PrintStdOpts("S");
   
}

/* SetConfParms: set conf parms relevant to this tool */
void SetConfParms(void)
{
   int i;
   Boolean b;
   char buf[MAXSTRLEN];

   nParm = GetConfig("HCOPY", TRUE, cParm, MAXGLOBS);
   if (nParm>0){
      if (GetConfInt(cParm,nParm,"TRACE",&i)) trace = i;
      if (GetConfBool(cParm,nParm,"SAVEASVQ",&b)) saveAsVQ = b;
      if (GetConfInt(cParm,nParm,"NSTREAMS",&i)) swidth0 = i;
      if (GetConfStr(cParm,nParm,"SOURCEFORMAT",buf))
         srcFF = Str2Format(buf);
      if (GetConfStr(cParm,nParm,"TARGETFORMAT",buf))
         tgtFF = Str2Format(buf);
      if (GetConfStr(cParm,nParm,"SOURCEKIND",buf))
         srcPK = Str2ParmKind(buf);
      if (GetConfStr(cParm,nParm,"TARGETKIND",buf)) {
         tgtPK = Str2ParmKind(buf);
         if (tgtPK&HASNULLE) 
            HError(1019, "SetConfParms: incompatible TARGETKIND=%s for coding", buf);
      }
   }
}

/* FixOptions: Check and set config options */
void FixOptions(void)
{
   if (stenSet && (labstidx>0 || labName != NULL))
      HError(1019,"FixOptions: Specify -s/-e or -x but not both");
   if (labstidx>0 && labName != NULL)
      HError(1019,"FixOptions: Specify label index or name but not both");
   if (srcFF == UNDEFF) srcFF = HTK;
   if (tgtFF == UNDEFF) tgtFF = HTK;
   if (tgtPK == ANON) tgtPK = srcPK;
}




int main(int argc, char *argv[])
{
   char *s;                     /* next file to process */
   void SetImageFileName(const char *s);
   // void AppendImageFile(char *s);
   void PutTargetFile(const char *s);

   if(InitShell(argc,argv,himcopy_version,himcopy_vc_id)<SUCCESS)
      HError(1000,"HCopy: InitShell failed");
   InitMem();   
   /*InitLabel();
   InitMath();  InitSigP();
   InitWave();  InitAudio();
   InitVQ();    InitModel();*/
   if(InitParm()<SUCCESS)  
      HError(1000,"HCopy: InitParm failed");

   if (!InfoPrinted() && NumArgs() == 0)
      ReportUsage();
   if (NumArgs() == 0) Exit(0);

   SetConfParms();
   /* initial trace string is null */
   trList.str = NULL;

   CreateHeap(&iStack, "InBuf",   MSTAK, 1, 0.0, STACKSIZE, LONG_MAX);
   CreateHeap(&oStack, "OutBuf",  MSTAK, 1, 0.0, STACKSIZE, LONG_MAX);
   /*CreateHeap(&cStack, "ChopBuf", MSTAK, 1, 0.0, STACKSIZE, LONG_MAX);
   CreateHeap(&lStack, "LabBuf",  MSTAK, 1, 0.0, 10000, LONG_MAX);
   */
   CreateHeap(&tStack, "Trace",   MSTAK, 1, 0.0, 100, 200);

	
   while (NextArg() == SWITCHARG) {
      s = GetSwtArg();
      if (strlen(s)!=1) 
         HError(1019,"HCopy: Bad switch %s; must be single letter",s);
      switch(s[0]){
      case 'a':
         if (NextArg() != INTARG)
            HError(1019,"HCopy: Auxiliary label index expected");
         auxLab = GetChkedInt(1,100000,s) - 1;
         break;
      case 'e':              /* end time in seconds, max 10e5 secs */
         en = GetChkedFlt(-MAXTIME,MAXTIME,s);
         stenSet = TRUE; chopF = TRUE;
         break;
      case 'i':
         if (NextArg() != STRINGARG)
            HError(1019,"HCopy: Output MLF name expected");
         if(SaveToMasterfile(GetStrArg())<SUCCESS)
            HError(1014,"HCopy: Cannot write to MLF");
         useMLF = TRUE; labF = TRUE; break;
      case 'l':
         if (NextArg() != STRINGARG)
            HError(1019,"HCopy: Target label file directory expected");
         outLabDir = GetStrArg();
         labF = TRUE; break;
      case 'm':
         xMargin = GetChkedFlt(-MAXTIME,MAXTIME,s);
         chopF = TRUE; break;
      case 'n':
         if (NextArg() != INTARG)
            HError(1019,"HCopy: Label index expected");
         labstidx= GetChkedInt(-100000,100000,s);
         if (NextArg() == INTARG)
            labenidx = GetChkedInt(-100000,100000,s);
         chopF = TRUE; break;          
      case 's':      /* start time in seconds */
         st = GetChkedFlt(0,MAXTIME,s);
         stenSet = TRUE; chopF = TRUE; break;
      case 't':
         if (NextArg() != INTARG)
            HError(1019,"HCopy: Trace line width expected");
         traceWidth= GetChkedInt(10,100000,s); break;
      case 'x':
         if (NextArg() != STRINGARG)
            HError(1019,"HCopy: Label name expected");
         labName = GetLabId(GetStrArg(),TRUE);
         if (NextArg() == INTARG)
            labRep = GetChkedInt(1,100000,s);
         chopF = TRUE; labF = TRUE; break;
      case 'F':
         if (NextArg() != STRINGARG)
            HError(1019,"HCopy: Source file format expected");
         if((srcFF = Str2Format(GetStrArg())) == ALIEN)
            HError(-1089,"HCopy: Warning ALIEN src file format set");
         break;
      case 'G':
         if (NextArg() != STRINGARG)
            HError(1019,"HCopy: Source label File format expected");
         if((srcLabFF = Str2Format(GetStrArg())) == ALIEN)
            HError(-1089,"HCopy: Warning ALIEN Label output file format set");
         labF= TRUE; break;
      case 'I':
         if (NextArg() != STRINGARG)
            HError(1019,"HCopy: MLF file name expected");
         LoadMasterFile(GetStrArg());
         labF = TRUE; break;
      case 'L':
         if (NextArg()!=STRINGARG)
            HError(1019,"HCopy: Label file directory expected");
         labDir = GetStrArg();
         labF = TRUE; break;
      case 'P':
         if (NextArg() != STRINGARG)
            HError(1019,"HCopy: Label File format expected");
         if((tgtLabFF = Str2Format(GetStrArg())) == ALIEN)
            HError(-1089,"HCopy: Warning ALIEN Label file format set");
         labF = TRUE; break;
      case 'O':
         if (NextArg() != STRINGARG)
            HError(1019,"HCopy: Target file format expected");
         if((tgtFF = Str2Format(GetStrArg())) == ALIEN)
            HError(-1089,"HCopy: Warning ALIEN target file format set");
         break;
      case 'T':
         trace = GetChkedInt(0,16,s); break;
      case 'X':
         if (NextArg()!=STRINGARG)
            HError(1019,"HCopy: Label file extension expected");
         labExt = GetStrArg();
         labF = TRUE; break;     
      default:
         HError(1019,"HCopy: Unknown switch %s",s);
      }
   }
   if (NumArgs() == 1)  
      HError(1019,"HCopy: Target file or + operator expected");
   FixOptions();
   

	MagickWandGenesis();
	magick_wand = NewMagickWand();

   
   while (NumArgs()>1) { /* process group S1 + S2 + ... TGT */
      off = 0.0;
      if (NextArg()!=STRINGARG)
         HError(1019,"HCopy: Source file name expected");    
      s = GetStrArg();     
      SetImageFileName(s); /*first s is .jpeg**/               /* Load initial file  S1 */
      if (NextArg()!=STRINGARG)
         HError(1019,"HCopy: Target file or + operator expected");
      s = GetStrArg();
		
      PutTargetFile(s);  /**next s is .mfc**/
      
	  if(trace & T_MEM) PrintAllHeapStats();
      if(trans != NULL){
         trans = NULL;
         ResetHeap(&lStack);
      }
      ResetHeap(&iStack);
      ResetHeap(&oStack);
      if(chopF) ResetHeap(&cStack);
   }
   
    DestroyMagickWand(magick_wand);
	MagickWandTerminus();

   
   if(useMLF) CloseMLFSaveFile();
   if (NumArgs() != 0) HError(-1019,"HCopy: Unused args ignored");
   Exit(0);
   return (0);          /* never reached -- make compiler happy */
}

/* ----------------- Trace linked list handling ------------------------ */

/* AppendTrace: insert a string to trStr for basic tracing */
void AppendTrace(char *str)
{
   TrPtr tmp = trStr;

   /* Seek to end of list */
   while (tmp->str != NULL) tmp = tmp->next;
   tmp->str =  CopyString(&tStack, str);
   tmp->next = (TrPtr)New(&tStack,sizeof(trList));
   tmp->next->str = NULL;
   tmp->next->next = NULL;
}

/* PrintTrace: Print trace linked list */
void PrintTrace(void)
{
   int linelen = 0;
   TrPtr tmp = trStr;

   /* print all entries in list */
   while (tmp->next != NULL){
      printf("%s ",tmp->str);
      linelen += strlen(tmp->str) + 1;
      if (linelen > traceWidth && tmp->next->next!=NULL){
         printf("\n    ");  /* wrap line where appropriate */
         linelen = 0;
      }
      tmp = tmp->next;
   }
   if(linelen > 0) printf("\n");
}


/* isImage: check config parms to see if target is a waveform */
Boolean IsImage(char *srcFile)
{
   Boolean isImage;
   
   isImage = tgtPK == MFCC;
   return isImage;
}

/* src has .jpg extension */
HTime ExtractAndSaveFeature(const char *targetMFCfn)
{

	int i;
	Boolean in_array(size_t x, size_t y, size_t *z);
	MagickBooleanType status;
	status = MagickReadImage(magick_wand, imageSRCFN);
	// double *gray_lee=calloc(sizeof(double)),16);
	if (status == MagickFalse) {
		HError(20000, "Error reading image %s\n",imageSRCFN);
		return 0;
	}
	
	getRGBDataDimension(magick_wand,imageSRCFN,&gray);
	// return 0;
   if (trace & T_IMG) {
		fprintf(stdout,"Image dimension w x h: %u x %u \r\n",gray.width,gray.height);
   }
  // MagickDisplayImage(magick_wand,   "localhost:0.0");
	if (trace & T_DCT) {
		printf("dct2_std #1\r\n");
	}
	getGrayscaleData(magick_wand, &gray);
  // printf("dct2_std #1\r\n");
	// return 0; //debug
	//dct test
	initializeParametersBlock(gray,&blkinfo,4);
   if (trace & T_IMG) {
		fprintf(stdout,"blkinfo w x h: %u x %u \r\n",blkinfo.width,blkinfo.height);
   }
	// return 0; //debug
	
	/**onlt padding rightest and lowest/bottom**/
	DbMatrix mat=CreateDbMatrix(gray.height+blkinfo.bottom_pad
			,gray.width+blkinfo.right_pad); // the dimension is h*width
	DbMatrixFromCharVect(mat, gray.data,gray.width,gray.height);		
	free(gray.data);//no longer use, only use mat since

	long totSamples=(blkinfo.total_frame_rows*(blkinfo.total_frame_cols
						+MARKER_LENGTH));
	// short obs_vector_size=18 ; //6+6+6
	short obs_vector_size=6 ; //6+6+6
	if(use_delta5)obs_vector_size+=11;
	if(use_accel5){
		use_delta5=TRUE;
		obs_vector_size=16;
		
	}
	
	// return 0; //debug
	if (trace & T_IMG) {
		printf("totSamples=%zu\r\n",totSamples);
	}
	FILE *fmfc=	fopen(targetMFCfn,"wb");//open a file for writing
	// return 0; //debug
	if(fmfc==NULL){
		HError(20000,"open file %s for writing failed \r\n",targetMFCfn);

	}
	OpenForSaveMFCFile(fmfc,  obs_vector_size,  totSamples);
	
	
	// return 0; //debug
		
	if (trace & T_IMG) {
		frame_summary(&blkinfo);
	}


	if (trace & T_DCT) {
		printf("double DbMat full matrix\r\n");
		dumpDbMatrix(mat,0,0,9,9);
	}

	// return 0; //debug
	
	blkinfo.col_start=0;
	blkinfo.row_start=0;
	
	size_t irow=0,icol=0;
	size_t end_block=0;
	size_t to_pad=0; //
	double * fft_mat=(double*)malloc(sizeof(double)*blkinfo.size*blkinfo.size);
	double *fft_result=(double*)malloc(sizeof(double)*blkinfo.size*blkinfo.size);
	if(sizeof(float)!=4){
		HError(20000,"sizeof float !=4 \r\n");
	}
	if (trace & T_DCT) {
		printf("double DbMat full matrix shift(%u,%u):%u\r\n",blkinfo.col_start,blkinfo.row_start,blkinfo.size);
		dumpDbMatrix(mat,blkinfo.col_start,blkinfo.row_start,blkinfo.size,blkinfo.size);
	}
	
	
	// return 0; //debug
	Boolean in_arrq;
	size_t total_frame_written=0;
	for (irow=0;irow<blkinfo.total_frame_rows;irow++){
		for (icol=0;icol<blkinfo.total_frame_cols;icol++){
			blkinfo.col_start=icol*blkinfo.col_stride;
			blkinfo.row_start=irow*blkinfo.row_stride;

			in_arrq=in_array(irow*blkinfo.total_frame_cols+icol,DEBUG_EL,DCT_DEBUG);
			if(in_arrq)
			if (trace & T_FRAME){
				printf("s.block: %u %u y,x:%u,%u\r\n",irow,icol,blkinfo.row_start,
				blkinfo.col_start	);
			}
			
			
			//padding check
			end_block=blkinfo.col_start+blkinfo.col_stride;
			
			/* if(icol==blkinfo.total_frame_cols-1&&end_block>blkinfo.width){
					printf("Need padding right  %u<%u \r\n",end_block,blkinfo.width);
			}
			end_block=blkinfo.row_start+blkinfo.row_stride;
			if(irow==blkinfo.total_frame_rows-1&&end_block>blkinfo.height){
					printf("Need padding bottom %u<%u\r\n",end_block,blkinfo.height);
			} */
			
			//need to zero rightest, lowest blocks if any paddings
			/**only padding right and bottom**/
			if(icol==blkinfo.total_frame_cols &&
				blkinfo.right_pad>0){
				DbVectFromMatrix(mat,fft_mat,  
					blkinfo.col_start, blkinfo.size-blkinfo.right_pad,
							 blkinfo.row_start,blkinfo.size	);
						
			
			} else if(irow==blkinfo.total_frame_rows &&
				blkinfo.bottom_pad>0){
				DbVectFromMatrix(mat,fft_mat,  
					blkinfo.col_start, blkinfo.size,
							 blkinfo.row_start,blkinfo.size-blkinfo.bottom_pad	);
					
			} else {
			
			
				DbVectFromMatrix(mat,fft_mat,  
					blkinfo.col_start, blkinfo.size,
							 blkinfo.row_start,blkinfo.size	);
			}
			
			if(in_arrq)
			if(trace & T_DCT){
				printf("(row,col) (%u,%u) : irow*totfram+icol=%u  \r\n",irow
				,icol,irow*blkinfo.total_frame_cols+icol);
				printf("double vec subblock %u,%u : %u\r\n",blkinfo.col_start,blkinfo.row_start,blkinfo.size);
				dump_double_image(fft_mat,blkinfo.size,blkinfo.size);
			}
			
			compute2Ddct(fft_mat,fft_result,blkinfo.size,blkinfo.size);
			
			if(in_arrq)
			if(trace & T_DCT){
				printf("fftw results \r\n");
				dump_double_image(fft_result,blkinfo.size,blkinfo.size);
			}
			
			// exit(0);
			if(in_arrq)
			if(trace & T_DCT){
				printf("gray_lee_16 prev \r\n");
				// dump_float_image(gray_lee_prev,7,16);
				
			}
			feature_hatala_16(fft_result,blkinfo.size,gray_lee_prev,gray_lee);
			// feature_hatala_18(fft_result,blkinfo.size,gray_lee_prev,gray_lee);
			
			if(use_delta5){
				if(irow==0&&icol==0){ //first block delta =0
					// for (i=6;i<16;i++)gray_lee[i]=0;
					for (i=6;i<11;i++)gray_lee[i]=0; //vcszie=18
				}
				
			}
			if(use_accel5){
				if(irow==0&&icol<=1){//second block accell-0
					// for (i=11;i<16;i++)gray_lee[i]=0;//vcsize=16
					for (i=11;i<16;i++)gray_lee[i]=0;//vcsize=16
				}
			}
			
			if(in_arrq)
			if(trace & T_DCT){
				// printf("fft_result[0]:%.2f,  gray_lee_18[0]%.2f\r\n",fft_result[0],gray_lee[0]);fflush(stdout);

				printf("gray_lee_16 \r\n");
				// dump_float_image(gray_lee,7,16);
				// dump_float_image(gray_lee,7,18);
				
			}			
			
			// printf("writing ... ");fflush(stdout); //debug

			// appendFloat(fmfc, obs_vector_size, 1,gray_lee);//write to file
			WriteFloat(fmfc, gray_lee, obs_vector_size, TRUE); //does this WriteFloat appending ?
	
			// size_t wrlen=fwrite(gray_lee,sizeof(float)*18,1,fmfc);
			// fclose(fmfc);

			// printf("completed %zu\r\n",wrlen);fflush(stdout);//debug
			// printf("completed \r\n");fflush(stdout);//debug

			for (i=0;i<obs_vector_size;i++)gray_lee_prev[i]=gray_lee[i];//copy
			// writecoefs();

			total_frame_written+=obs_vector_size;
		}	
		// pad_MARKER
			for(int i=0;i<MARKER_LENGTH;++i){
				WriteFloat(fmfc, MARKER_OBS, obs_vector_size, TRUE); //does this WriteFloat appending ?
				++total_frame_written;
			}
		// total_frame_written+=MARKER_LENGTH*obs_vector_size;
	}
	
	free(fft_mat);
	free(fft_result);
	fclose(fmfc);//
	
	

   if (trace & T_TOP) {
		printf("%zu frames written\r\n",total_frame_written);
   }

   if (trace & T_TOP) {
	   AppendTrace(" open image ");
	   // AppendTrace(src);
   }
   
   return 0;
}




/* --------------------- Image File Handling ---------------------- */

/* OpenSpeechFile: open waveform or parm file */
void SetImageFileName(const char *s)
{

   
   if(IsImage(s)) { 
	   if(s!=NULL){
		  size_t lenFN=strlen(s);
		  if (lenFN>0){
			  memset(imageSRCFN,0,256);
			  strncpy(imageSRCFN,s,lenFN); //just mark/book filename
			  if(trace&T_TOP){
				  AppendTrace("SrcFile: ");
				  AppendTrace(imageSRCFN);
				  AppendTrace("\r\n");
			  }
		  }
	   }
   }
   else { 
      /*len = OpenParmFile(s);*/
		fprintf(stderr, "open Parm file not implemented\r\n");
		return ;
   }
}


/* s has .mfc extension */
void PutTargetFile(const char *target)
{
	
	// printf("puttargetfile: src is %s\r\n",imageSRCFN); fflush(stdout);
	ExtractAndSaveFeature(target);
	// testSaveMFCFile(target,18,1,gray_lee);
   if (trace & T_TOP){
      AppendTrace("creating target");

   }

   if (trace & T_TOP){
      AppendTrace("->"); AppendTrace(target);
      PrintTrace();     
      ResetHeap(&tStack);
      trList.str = NULL;
   }
   
}

Boolean in_array(const size_t number,size_t size,size_t *arr){
	// if(trace&&T_DCT){
		// printf("comparing %u to \r\n",number);
		// fflush(stdout);
	// }
	  for (size_t i = 0; i < size; i++) {
		   // if(number==4000){
				// printf("comparing %u to %u\r\n",number,arr[i]);
				// fflush(stdout);
		   // }
           if (arr[i] == number) {
               return TRUE;
           }
       }
	return FALSE;
}

