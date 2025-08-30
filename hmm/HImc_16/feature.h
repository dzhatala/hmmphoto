#ifndef __FEATURE_H__
#define __FEATURE_H__
#include <stdlib.h>

void OpenForSaveMFCFile(FILE *f, short vectorSize,long numSamples);
void saveMFCFile(const char*s, short vectorSize,long samPeriod,long numSamples,float *data);

#endif
