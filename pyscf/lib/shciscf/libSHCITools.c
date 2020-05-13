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
 * Author: James E. T. Smith james.e.smith@colorado.edu (2/7/17)
 *
 * This is a shared library for use interfacing the pyscf package with the Dice
 * package.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include <complex.h>


  void transformRDMDinfh(int norbs, int* nRows, int* rowInds, double* rowCoeffs, double* int2, double* newint2)
  {
    size_t n3 = norbs*norbs*norbs, n2=norbs*norbs;
#pragma omp parallel 
    {
      int i,j,k,l,ia,ja,ka,la;
      
      for (i=0; i<norbs; i++) {
	if (i%omp_get_num_threads() != omp_get_thread_num()) continue;
	for (j=0; j<norbs; j++)
	  for (k=0; k<norbs; k++)
	    for (l=0; l<norbs; l++)
	      {
		double _Complex comp = 0.0;
		for (ia=0; ia<nRows[i]; ia++)
		  for (ja=0; ja<nRows[j]; ja++)
		    for (ka=0; ka<nRows[k]; ka++)
		      for (la=0; la<nRows[l]; la++) {
			int ii,jj,kk,ll; 
			ii = rowInds[2*i+ia], jj = rowInds[2*j+ja], kk = rowInds[2*k+ka], ll = rowInds[2*l+la];
			double _Complex ci = rowCoeffs[4*i+2*ia] + rowCoeffs[4*i+2*ia+1]*I;
			double _Complex cj = rowCoeffs[4*j+2*ja] + rowCoeffs[4*j+2*ja+1]*I; 
			double _Complex ck = rowCoeffs[4*k+2*ka] + rowCoeffs[4*k+2*ka+1]*I; 
			double _Complex cl = rowCoeffs[4*l+2*la] + rowCoeffs[4*l+2*la+1]*I;
			comp = comp + conj(ci)*cj*conj(ck)*cl*int2[ii*n3+jj*n2+kk*norbs+ll];
		      }
		if (cimag(comp) > 1.e-4) {
		  printf("Error in %d %d %d %d element of rdm (%g,%g)\n", i, j, k, l, creal(comp), cimag(comp));
		  exit(0);
		}

		newint2[i*n3+j*n2+k*norbs+l] = creal(comp);
	      }
      }
    }
    printf("Done rdm \n");
  }

  void transformDinfh(int norbs, int* nRows, int* rowInds, double* rowCoeffs, double* int2, double* newint2)
  {
    size_t n3 = norbs*norbs*norbs, n2=norbs*norbs;
#pragma omp parallel 
    {
      int i,j,k,l,ia,ja,ka,la;
      
      for (i=0; i<norbs; i++) {
	if (i%omp_get_num_threads() != omp_get_thread_num()) continue;
	for (j=0; j<norbs; j++)
	  for (k=0; k<norbs; k++)
	    for (l=0; l<norbs; l++)
	      {
		double _Complex comp = 0.0;
		for (ia=0; ia<nRows[i]; ia++)
		  for (ja=0; ja<nRows[j]; ja++)
		    for (ka=0; ka<nRows[k]; ka++)
		      for (la=0; la<nRows[l]; la++) {
			int ii, jj, kk, ll;
			ii = rowInds[2*i+ia], jj = rowInds[2*j+ja], kk = rowInds[2*k+ka], ll = rowInds[2*l+la];
			int sgnf = ia+ja+ka+la;
			double sign = sgnf==2 ? -1. : 1.;
			if (sgnf%2 == 0)
			  newint2[i*n3+j*n2+k*norbs+l] += sign*pow(-1., ia)*pow(-1., ka)*int2[ii*n3+jj*n2+kk*norbs+ll]*rowCoeffs[2*i+ia]*rowCoeffs[2*j+ja]*rowCoeffs[2*k+ka]*rowCoeffs[2*l+la];
		      }
	      }
      }
    }
    
  }
  
  void writeIntNoSymm(int norbs, double* int1, double* int2, double coreE, int nelec, int* irrep)
  {
    size_t n3 = norbs*norbs*norbs, n2=norbs*norbs;
    FILE *fp;
    fp=fopen("FCIDUMP", "w");
    
    fprintf(fp, "&FCI NORBS=%d, NELEC=%d, MS2=0\n", norbs, nelec);
    fprintf(fp, "ORBSYM=");
    int i,j,k,l,ia,ja,ka,la;
    for (i=0; i<norbs; i++) {
      fprintf(fp, "%d,", irrep[i]);
    }
    fprintf(fp,"\nISYM=1\nKSYM\n&END\n");
    
    
    for (i=0; i<norbs; i++) 
      for (j=0; j<norbs; j++)
	for (k=0; k<norbs; k++)
	  for (l=0; l<norbs; l++) {
	    if (fabs(int2[i*n3+j*n2+k*norbs+l]) >= 1.e-9 && i*norbs+j >= k*norbs+l) {
	      fprintf(fp, "%20.12f  %d  %d  %d  %d\n", int2[i*n3+j*n2+k*norbs+l], i+1, j+1, k+1, l+1);
	    }
	  }
    
    for (i=0; i<norbs; i++) 
      for (j=i; j<norbs; j++)
	if (fabs(int1[i*norbs+j]) > 1.e-9)
	  fprintf(fp, "%20.12f  %d  %d  %d  %d\n", int1[i*norbs+j], i+1, j+1, 0, 0);
    
    fprintf(fp, "%20.12f  %d  %d  %d  %d\n", coreE, 0,0,0,0);
    
    fclose(fp);
  }

  /* This function is the basic reader for second order spatialRDM files and
	should return arrays that can be iterated over. The switching of the
	idices is intentional when saving the elements of the 2RDM and arises
	from a difference in notation between pyscf and SHCI.
  */
void r2RDM( double * twoRDM, size_t norb,  char * fIn ){

  char line[255];
  FILE *fp = fopen( fIn, "r" );
  fgets(line, sizeof(line), fp);
  
  int norbs = atoi( strtok(line, " ,\t\n") );
  assert( norbs == norb );
  int norbs2 = norbs*norbs;
  int norbs3 = norbs2*norbs;
	  
  while ( fgets(line, sizeof(line), fp) != NULL ) {
    int i = atoi( strtok(line, " ,\t\n") );
    int k = atoi( strtok(NULL, " ,\t\n") );
    int j = atoi( strtok(NULL, " ,\t\n") );
    int l = atoi( strtok(NULL, " ,\t\n") );
    float val = atof( strtok(NULL, " ,\t\n") );
    int indx = i*norbs3 + j*norbs2 + k*norbs + l;
    twoRDM[indx] = val;
  }
  fclose(fp);
}


// This function is used to create the header for the FCI dump file
void writeFDHead ( FILE* fOut, size_t norb, size_t nelec, size_t ms, int* orbsym )
{
  fprintf(fOut, " &FCI NORB=%zu ,NELEC=%zu ,MS2=%zu,\n", norb, nelec, ms);
  fprintf(fOut, "  ORBSYM=");
  size_t i;

  for ( i = 0; i < norb; ++i ) {
      fprintf(fOut,"%d,", orbsym[i]);
  }
  
  fprintf(fOut, "\n  ISYM=1,\n &END\n");
}

// Only for 8-fold symmetry
void writeERI ( FILE* fOut, double * eri, size_t norb, double tol)
{
  size_t ij = 0;
  size_t kl = 0;
  size_t ijkl = 0;
  size_t n2 = norb*norb;
  size_t n3 = n2*norb;

  size_t i,j,k,l;
  for ( i=0; i<norb; ++i ) {
    for( j=0; j<i+1; ++j ) {
      kl = 0;
      for ( k=0; k<i+1; ++k ) {
	for (l=0; l<k+1; ++l ) {
	  
	  if ( ij >= kl ) { 
	    if ( fabs(eri[ijkl]) > tol ) {
	    fprintf(fOut,  "%20.12e    %zu  %zu  %zu  %zu\n",eri[ ijkl ],i+1,j+1,k+1,l+1);
	    }
	    ++ijkl;
	  }
	  ++kl;
	}
      }
      ++ij;
    }
  }  
}

void writeHCore ( FILE * fOut, double * h1e, size_t norb, double tol )
{
  int i,j;

  for ( i = 0; i < norb; ++i ) {
    for ( j = 0; j < i+1; ++j ) {
      if ( fabs( h1e[ i*norb + j ] ) > tol ) {
	fprintf( fOut, "%20.12e    %d  %d  %d  %d\n", h1e[ i*norb + j ],
		 i+1, j+1, 0, 0 );
      }
    }
  }
}

// This function is an alternative to pyscf.tools.fcidump.from_integral and
// should be used when working with large active spaces.
void fcidumpFromIntegral ( char * fileName, double * h1eff, double * eri,
			   size_t norb, size_t nelec, double ecore, 
			   int* orbsym, size_t ms )
{
  FILE * fOut = fopen( fileName, "w");
  writeFDHead( fOut, norb, nelec, ms, orbsym );
  writeERI( fOut, eri, norb, 1e-12 );
  writeHCore( fOut, h1eff, norb, 1e-12 );
  fprintf( fOut, "%20.12f   %d  %d  %d  %d\n", ecore, 0,0,0,0);
  fclose(fOut);
}

/*
  Sum over m1s rows and m2s columns. The indx variable determines which
  index of m1 is summed over, e.g. 0 is the first index and so on. m1 should
  be the one-body matrix, m2 should be the unitary matrix. This function has
  a temporal cost of O(n^3).
*/
void multMat2D ( size_t n, double * m1, double * m2, double * mout,
		 size_t indx )
{
  size_t i, j, k, ik, kj, ij, ki;
  
  if ( indx == 1 )
    {
      for ( i = 0; i < n; ++i )
	for ( j = 0; j < n; ++j )
	  {
	    ij = j + i*n;
	    mout[ij] = 0;
	    
	    for ( k = 0; k < n; ++k )
	      {
		ik = k + i*n;
		kj = j + k*n;
		mout[ij] += m1[ik] * m2[kj];
	      }
	  }
    }
  
  else if ( indx == 0 )
    {
      for ( i = 0; i < n; ++i )
	for ( j = 0; j < n; ++j )
	  {
	    ij = j + i*n;
	    mout[ij] = 0;
	    
	    for ( k = 0; k < n; ++k )
	      {
		ki = i + k * n;
		kj = j + k * n;
		mout[ij] += m1[ki] * m2[kj];
	      }
	  }
    }
}

/*
  Multiplies the matrices m1[n^4] by m2[n^2] and allows the user to choose
  the index of m1 to sum over using the indx parameter, e.g. 0 sums over
  the last index of m1. n is the common dimension of m1 and m2. mout is the
  output array for the multiplication. When used with two-body integrals,
  m1 shold be the eri (or intermediate eri) array and m2 should be the unitary
  matrix. This function has a temporal cost of O(n^5).
  
  NOTE: Chemistry notation is used for the two-body matrix indices, e.g.
  (ab|cd) = \int \phi_{a^+} \phi_b O_2 \phi_{c^+} \phi_d dx
*/
void multMat4D ( size_t n, double * m1, double * m2, double * mout,
		 size_t indx )
{
  size_t n2 = n*n, n3 = n2*n;
  
  // Switch statement deals with the different indices to sum over.
  switch ( indx )
    {
    case 0:
#pragma omp parallel
      {
	size_t a,b,c,d,i;
	for ( a = 0; a < n; ++a )
	  for ( b = 0; b < n; ++b )
	    for ( c = 0; c < n; ++c )
	      for ( d = 0; d < n; ++d )
		{
		  size_t abcd = d + c*n + b*n2 + a*n3;
		  mout[abcd] = 0;
		  
		  for ( i = 0; i < n; ++i )
		    {
		      size_t ia = i*n + a;
		      size_t ibcd = d + c*n + b*n2 + i*n3;
		      mout[abcd] += m1[ibcd] * m2[ia];
		    }
		}
      }
      break;
      
    case 1:
#pragma omp parallel
      {
	size_t i,b,c,d,j;
	for ( i = 0; i < n; ++i )
	  for ( b = 0; b < n; ++b )
	    for ( c = 0; c < n; ++c )
	      for ( d = 0; d < n; ++d )
		{
		  size_t ibcd = d + c*n + b*n2 + i*n3;
		  mout[ibcd] = 0;
		  
		  for ( j = 0; j < n; ++j )
		    {
		      size_t jb = j*n + b;
		      size_t ijcd = d + c*n + j*n2 + i*n3;
		      mout[ibcd] += m1[ijcd] * m2[jb];
		    }
		}
      }
      break;
      
    case 2:
#pragma omp parallel
      {
	size_t i,j,c,d,k;
	for ( i = 0; i < n; ++i )
	  for ( j = 0; j < n; ++j )
	    for ( c = 0; c < n; ++c )
	      for ( d = 0; d < n; ++d )
		{
		  size_t ijcd = d + c*n + j*n2 + i*n3;
		  mout[ijcd] = 0;
		  
		  for ( k = 0; k < n; ++k )
		    {
		      size_t kc = k*n + c;
		      size_t ijkd = d + k*n + j*n2 + i*n3;
		      mout[ijcd] += m1[ijkd] * m2[kc];
		    }
		}
      }
      break;
      
    default:
#pragma omp parallel
      {
	size_t i,j,k,l,d;
	for ( i = 0; i < n; ++i )
	  for ( j = 0; j < n; ++j )
	    for ( k = 0; k < n; ++k )
	      for ( d = 0; d < n; ++d )
		{
		  size_t ijkd = d + k*n + j*n2 + i*n3;
		  mout[ijkd] = 0;
		  
		  for ( l = 0; l < n; ++l )
		    {
		      size_t ld = l*n + d;
		      size_t ijkl = l + k*n + j*n2 + i*n3;
		      mout[ijkd] += m1[ijkl] * m2[ld];
		    }
		}
      }
      break;
    }
  
}

/*
  This transformation of h1 from MO to NO basis is performed with a
  temporal cost of O(n^3).
*/
void oneBodyTrans ( size_t norb, double * un, double * h1 )
{
  double mp[ norb * norb ]; // Initialize intermediate matrix
  multMat2D ( norb, h1, un, mp, 1 );
  multMat2D ( norb, mp, un, h1, 0 );
  
}

/*
  This function transforms the eri array sequentially from the MO to NO
  basis. Like oneBodyTrans() this function uses no special algorithms to
  optimize performance and has a time cost of O(n^5).
*/
void twoBodyTrans ( size_t norb, double * un, double * eri )
{
  size_t norb4 = norb * norb * norb * norb;
  double interMat1[ norb4 ];
  
  multMat4D( norb, eri, un, interMat1, 3 );
  multMat4D( norb, interMat1, un, eri, 2 );
  multMat4D( norb, eri, un, interMat1, 1 );
  multMat4D( norb, interMat1, un, eri, 0 );
  
}

//}
