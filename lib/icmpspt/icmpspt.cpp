#include <stdio.h>
#include "math.h"
#include <algorithm> 
#include <iostream>

using namespace std;
extern "C" {
  void unpackE3(char* file, char* fout, size_t norb) {
    FILE *f = fopen(file, "rb");
    size_t norb2 = norb*norb;
    size_t e3slicesize = (norb2*norb2*norb2 + 3*norb2*norb2 + 2*norb2)/6;
    double *fj = (double*)malloc(e3slicesize*sizeof(double));
    fread(fj, sizeof(*fj), e3slicesize, f);
    fclose(f);
    double *e3 = (double*)malloc(norb2*norb2*norb2*sizeof(double));
#pragma omp parallel for 
    for (int i=0; i<norb; i++)
      for (int j=0; j<norb; j++)
	for (int k=0; k<norb; k++)
	  for (int l=0; l<norb; l++)
	    for (int m=0; m<norb; m++)
	      for (int n=0; n<norb; n++)
		{
		  size_t a = i*norb+l, b = j*norb+m, c = k*norb+n;
		  size_t A=0, B=0, C=0;
		  
		  A = max(a,max(b,c));
		  if (A==a) {
		    B = max(b,c); C = min(b,c);
		  }
		  else if (A==b) {
		    B = max(a,c); C = min(a,c);
		  }
		  else {
		    B = max(a,b); C = min(a,b);
		  }
		  
		  size_t p = (A*A*A + 3*A*A + 2*A)/6  +  (B*B + B)/2 + C ;
		  
		  e3[i+j*norb+k*norb2+l*norb*norb2+m*norb2*norb2+n*norb2*norb2*norb] = fj[p];
		}
    
    FILE *f2 = fopen(fout, "wb");
    fwrite(e3, sizeof(*e3), norb2*norb2*norb2, f2);
    fclose(f2);
    free(e3);
    free(fj);
  };
}
