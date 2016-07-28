#include <stdio.h>
#include <stdlib.h>
#include "math.h"

#define MAX(I,J)        ((I) > (J) ? (I) : (J))
#define MIN(I,J)        ((I) < (J) ? (I) : (J))

void unpackE3(char* file, char* fout, int norb) {
    FILE *f = fopen(file, "rb");
    size_t norb2 = norb*norb;
    size_t e3slicesize = (norb2*norb2*norb2 + 3*norb2*norb2 + 2*norb2)/6;
    double *fj = (double*)malloc(e3slicesize*sizeof(double));
    fread(fj, sizeof(*fj), e3slicesize, f);
    fclose(f);
    double *e3 = (double*)malloc(norb2*norb2*norb2*sizeof(double));
#pragma omp parallel default(none) \
        shared(norb, norb2, e3, fj)
{
        int i, j, k, l, m, n;
#pragma omp parallel for
    for (i=0; i<norb; i++)
      for (j=0; j<norb; j++)
        for (k=0; k<norb; k++)
          for (l=0; l<norb; l++)
            for (m=0; m<norb; m++)
              for (n=0; n<norb; n++)
                {
                  size_t a = i*norb+l, b = j*norb+m, c = k*norb+n;
                  size_t A=0, B=0, C=0;

                  A = MAX(a,MAX(b,c));
                  if (A==a) {
                    B = MAX(b,c); C = MIN(b,c);
                  }
                  else if (A==b) {
                    B = MAX(a,c); C = MIN(a,c);
                  }
                  else {
                    B = MAX(a,b); C = MIN(a,b);
                  }

                  size_t p = (A*A*A + 3*A*A + 2*A)/6  +  (B*B + B)/2 + C ;

                  e3[i+j*norb+k*norb2+l*norb*norb2+m*norb2*norb2+n*norb2*norb2*norb] = fj[p];
                }
}
    FILE *f2 = fopen(fout, "wb");
    fwrite(e3, sizeof(*e3), norb2*norb2*norb2, f2);
    fclose(f2);
    free(e3);
    free(fj);
};
