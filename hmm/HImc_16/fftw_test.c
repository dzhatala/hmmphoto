/*ffwt_test.c
https://stackoverflow.com/questions/37530613/using-fftw3-library-for-dct

*/
#include "HShell.h"

#include <fftw3.h>
void dump_vector(int n, double* vec) {
	int i=0;
    for ( i = 0; i < n; i++)
        printf("%f ", vec[i]);
    printf("\n");
}
int main()
{
    // double a[] = {0.5, 0.6, 0.7, 0.8};
    double a[] = {2, 4, 6, 8};
    // double b[] = {0, 0, 0, 0};
    printf("Original vector\n");
    dump_vector(4, a);
    fftw_plan plan = fftw_plan_r2r_1d(4, a, a, FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_execute(plan);
    printf("DCT\n");
    dump_vector(4, a);
    fftw_plan plani = fftw_plan_r2r_1d(4, a, a, FFTW_REDFT01, FFTW_ESTIMATE);
    fftw_execute(plani);
    printf("IDCT\n");
    dump_vector(4, a);
    return 0;
}
