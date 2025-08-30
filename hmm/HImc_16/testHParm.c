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

char *himcopy_version = "!HVER!HImCopy:   3.4.1 [fftw magickwand 12/03/25]";
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


/* -------------------------- Trace Flags & Vars ------------------------ */
/**octal **/
#define T_TOP     001           /* basic progress reporting */
#define T_IMG     002           /* report file formats and parm kinds */
#define T_FRAME   003           /* output segment label calculations */
#define T_DCT 	  004           /* output segment label calculations */
#define T_MEM     010           /* debug memory usage */

static int  trace  = 0;         /* Trace level */


// static size_t []DCT_DEBUG[]={0,35,46}; errod undefined?

static size_t DCT_DEBUG[]={0,1,2};
static size_t DEBUG_EL=3;//size of DCT_DEBUG


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
   printf("\nUSAGE: HCopy [options] src [ + src ...] tgt ...\n\n");
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
   PrintStdOpts("FGILPOX");
   
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
   void OpenImageFile(char *s);
   // void AppendImageFile(char *s);
   void PutTargetFile(char *s);

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
      OpenImageFile(s); /*first s is .jpeg**/               /* Load initial file  S1 */
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




/* --------------------- Image File Handling ---------------------- */

/* OpenSpeechFile: open waveform or parm file */
void OpenImageFile(char *s)
{
   HTime len;
   char buf[MAXSTRLEN];
   
      // len = ExtractFeature(s);
   if (tgtPK == ANON) tgtPK = srcPK;      
}


/* s has .mfc extension */
void PutTargetFile(char *s)
{
	  openAndSaveParm(s);

   if (trace & T_TOP){
      AppendTrace("->"); AppendTrace(s);
      PrintTrace();     
      ResetHeap(&tStack);
      trList.str = NULL;
   }
   
}

Boolean in_array(size_t number,size_t size,size_t *arr){
	  for (size_t i = 0; i < size; i++) {
           if (arr[i] == number) {
               return TRUE;
           }
       }
	return FALSE;
}

void openAndSaveParm(const char*s){
	long numSamples=2,sampPeriod=1e5;
	short vectorSize=18;
	// float *data=calloc(sizeof(float),vectorSize*numSamples); //allocate memory for all observations
	
	float gray_lee[]={0,0,0,0,0,0,0,0,0,
                  0,0,0,0,0,0,0,0,0};//default is 18

	
		//simulating data..
	for (int i=0;i<numSamples;i++){
		
		for (int j=0;j<vectorSize;j++){
			gray_lee[i*vectorSize+j]=i+(float)j/10;
		}
	}
	
	testSaveMFCFile("tmp/x.mfc",vectorSize,numSamples,gray_lee);

}

