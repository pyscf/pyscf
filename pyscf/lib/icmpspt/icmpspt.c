/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Sandeep Sharma
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "math.h"

#define MAX(I,J)        ((I) > (J) ? (I) : (J))
#define MIN(I,J)        ((I) < (J) ? (I) : (J))

void unpackE3(char* file, char* fout, int norb) {
    FILE *f = fopen(file, "rb");
    size_t norb2 = norb*norb;
    // 6-fold symmetry
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

                  // E_ABC=E_ACB=...
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

                  // tetrahedral number + triangular number + square number
                  // A(A+1)(A+2)/3!     + B(B+1)/2!         + C/1!
                  size_t p = (A*A*A + 3*A*A + 2*A)/6  +  (B*B + B)/2 + C ;

                  // fully square number
                  int q = i+j*norb+k*norb2+l*norb*norb2+m*norb2*norb2+n*norb2*norb2*norb;

                  e3[q] = fj[p];
                }
}
    FILE *f2 = fopen(fout, "wb");
    fwrite(e3, sizeof(*e3), norb2*norb2*norb2, f2);
    fclose(f2);
    free(e3);
    free(fj);
};

void unpackE4(char* file, char* fout, int norb) {
    FILE *f = fopen(file, "rb");
    size_t norb2 = norb*norb;
    // 8-fold symmetry
    size_t e4slicesize = (norb2*norb2*norb2*norb2 + 6*norb2*norb2*norb2 + 11*norb2*norb2 + 6*norb2)/24;
    double *fj = (double*)malloc(e4slicesize*sizeof(double));
    fread(fj, sizeof(*fj), e4slicesize, f);
    fclose(f);
    double *e4 = (double*)malloc(norb2*norb2*norb2*norb2*sizeof(double));
    int i, j, k, h, l, m, n, o;
    for (i=0; i<norb; i++)
      for (j=0; j<norb; j++)
        for (k=0; k<norb; k++)
        for (h=0; h<norb; h++)
          for (l=0; l<norb; l++)
            for (m=0; m<norb; m++)
              for (n=0; n<norb; n++)
              for (o=0; o<norb; o++)
                {
                  size_t a = i*norb+l, b = j*norb+m, c = k*norb+n, d = h*norb+o;

                  // E_ABCD=E_ACBD=...
                  size_t A=0, B=0, C=0, D=0;
                  size_t prov=MAX(a,b);
                  A = MAX(prov,MAX(c,d));
                  if (A==a) {
                    if (MAX(b,MAX(c,d))==b) {
                      B=b;
                      C=MAX(c,d);
                      D=MIN(c,d);
                    } else if (MAX(b,MAX(c,d))==c) { 
                      B=c;
                      C=MAX(b,d);
                      D=MIN(b,d);
                    } else if (MAX(b,MAX(c,d))==d) { 
                      B=d;
                      C=MAX(b,c);
                      D=MIN(b,c);
                    }
                  } else if (A==b) {
                    if (MAX(a,MAX(c,d))==a) {
                      B=a;
                      C=MAX(c,d);
                      D=MIN(c,d);
                    } else if (MAX(a,MAX(c,d))==c) { 
                      B=c;
                      C=MAX(a,d);
                      D=MIN(a,d);
                    } else if (MAX(a,MAX(c,d))==d) { 
                      B=d;
                      C=MAX(a,c);
                      D=MIN(a,c);
                    }
                  } else if (A==c) {
                    if (MAX(b,MAX(a,d))==b) {
                      B=b;
                      C=MAX(a,d);
                      D=MIN(a,d);
                    } else if (MAX(b,MAX(a,d))==a) { 
                      B=a;
                      C=MAX(b,d);
                      D=MIN(b,d);
                    } else if (MAX(b,MAX(a,d))==d) { 
                      B=d;
                      C=MAX(b,a);
                      D=MIN(b,a);
                    }
                  } else if (A==d) {
                    if (MAX(b,MAX(c,a))==b) {
                      B=b;
                      C=MAX(c,a);
                      D=MIN(c,a);
                    } else if (MAX(b,MAX(c,a))==c) { 
                      B=c;
                      C=MAX(b,a);
                      D=MIN(b,a);
                    } else if (MAX(b,MAX(c,a))==a) { 
                      B=a;
                      C=MAX(b,c);
                      D=MIN(b,c);
                    }
                  };

                  // pentatopic number   + tetrahedral number + triangular number + square number
                  // A(A+1)(A+2)(A+3)/4! + B(B+1)(B+2)/3!     + C(C+1)/2!         + D/1!
                  size_t p = (A*A*A*A + 6*A*A*A + 11*A*A + 6*A)/24 + (B*B*B + 3*B*B + 2*B)/6  +  (C*C + C)/2 + D ;

                  // fully square number
                  int q = i+j*norb+k*norb2+h*norb*norb2+l*norb2*norb2+m*norb*norb2*norb2+n*norb2*norb2*norb2+o*norb*norb2*norb2*norb2;

                  e4[q] = fj[p];
                }
    FILE *f2 = fopen(fout, "wb");
    fwrite(e4, sizeof(*e4), norb2*norb2*norb2*norb2, f2);
    fclose(f2);
    free(e4);
    free(fj);
};

void unpackE3_BLOCK(char* file, char* fout, int norb) {
    FILE *f = fopen(file, "rb");
    size_t norb2 = norb*norb;
    // no symmetry!
    size_t e3slicesize = (norb2*norb2*norb2);
    double *fj = (double*)malloc(e3slicesize*sizeof(double));
    fseek(f,93,SEEK_SET);
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
                  // is given as E^ijk_nml and is expected to come out as E^ijk_lmn
                  int p = i+j*norb+k*norb2  +n*norb*norb2+m*norb2*norb2+l*norb2*norb2*norb;
                  int q = i+j*norb+k*norb2  +l*norb*norb2+m*norb2*norb2+n*norb2*norb2*norb;
                  e3[q] = fj[p];
                };
}
    FILE *f2 = fopen(fout, "wb");
    fwrite(e3, sizeof(*e3), norb2*norb2*norb2, f2);
    fclose(f2);
    free(e3);
    free(fj);
};

void unpackE4_BLOCK(char* file, char* fout, int norb) {
    FILE *f = fopen(file, "rb");
    size_t norb2 = norb*norb;
    // no symmetry!
    size_t e4slicesize = (norb2*norb2*norb2*norb2);
    double *fj = (double*)malloc(e4slicesize*sizeof(double));
    fseek(f,109,SEEK_SET);
    fread(fj, sizeof(*fj), e4slicesize, f);
    fclose(f);
    double *e4 = (double*)malloc(norb2*norb2*norb2*norb2*sizeof(double));
    int i, j, k, h, l, m, n, o;
    for (i=0; i<norb; i++)
      for (j=0; j<norb; j++)
        for (k=0; k<norb; k++)
        for (h=0; h<norb; h++)
          for (l=0; l<norb; l++)
            for (m=0; m<norb; m++)
              for (n=0; n<norb; n++)
              for (o=0; o<norb; o++)
                {
                  // is given as E^ijkh_onml and is expected to come out as E^ijkh_lmno
                  int p = i+j*norb+k*norb2+h*norb*norb2 +o*norb2*norb2+n*norb*norb2*norb2+m*norb2*norb2*norb2+l*norb*norb2*norb2*norb2;
                  int q = i+j*norb+k*norb2+h*norb*norb2 +l*norb2*norb2+m*norb*norb2*norb2+n*norb2*norb2*norb2+o*norb*norb2*norb2*norb2;
                  e4[q] = fj[p];
                };
    FILE *f2 = fopen(fout, "wb");
    fwrite(e4, sizeof(*e4), norb2*norb2*norb2*norb2, f2);
    fclose(f2);
    free(e4);
    free(fj);
};

