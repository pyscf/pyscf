// Author: James E. T. Smith james.e.smith@colorado.edu (2/7/17)
//
// This is a shared library for use interfacing the pyscf package with the SHCI
// module.
//
// TODO: Not stable
//
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <sstream>
#include <math.h>
#include <omp.h>
#include <complex>
#include <typeinfo>

using namespace std;

extern "C"
{

  void transformRDMDinfh(int norbs, int* nRows, int* rowInds, double* rowCoeffs, double* int2, double* newint2)
  {
    //int norbs = *pnorbs;
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
		std::complex<double> complex = 0.0;
		for (ia=0; ia<nRows[i]; ia++)
		  for (ja=0; ja<nRows[j]; ja++)
		    for (ka=0; ka<nRows[k]; ka++)
		      for (la=0; la<nRows[l]; la++) {
			int I = rowInds[2*i+ia], J = rowInds[2*j+ja], K = rowInds[2*k+ka], L = rowInds[2*l+la];
			std::complex<double> ci(rowCoeffs[4*i+2*ia], rowCoeffs[4*i+2*ia+1]), 
			  cj(rowCoeffs[4*j+2*ja], rowCoeffs[4*j+2*ja+1]), 
			  ck(rowCoeffs[4*k+2*ka], rowCoeffs[4*k+2*ka+1]), 
			  cl(rowCoeffs[4*l+2*la], rowCoeffs[4*l+2*la+1]);

			complex += conj(ci)*cj*conj(ck)*cl*int2[I*n3+J*n2+K*norbs+L];
			//newint2[i*n3+j*n2+k*norbs+l] += sign*pow(-1., ia)*pow(-1., ka)*int2[I*n3+J*n2+K*norbs+L]*rowCoeffs[2*i+ia]*rowCoeffs[2*j+ja]*rowCoeffs[2*k+ka]*rowCoeffs[2*l+la];
		      }
		if (complex.imag() > 1.e-4) {
		  cout << "Error in "<<i<<"  "<<j<<"  "<<k<<"  "<<l<<"  element of rdm "<<complex<<endl;
		  exit(0);
		}

		newint2[i*n3+j*n2+k*norbs+l] = complex.real();
	      }
      }
    }
    cout << "Done rdm "<<endl;
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
		double complex = 0.0;
		for (ia=0; ia<nRows[i]; ia++)
		  for (ja=0; ja<nRows[j]; ja++)
		    for (ka=0; ka<nRows[k]; ka++)
		      for (la=0; la<nRows[l]; la++) {
			int I = rowInds[2*i+ia], J = rowInds[2*j+ja], K = rowInds[2*k+ka], L = rowInds[2*l+la];
			int sgnf = ia+ja+ka+la;
			double sign = sgnf==2 ? -1. : 1.; //i^2 = -1
			if (sgnf%2 == 0)
			  newint2[i*n3+j*n2+k*norbs+l] += sign*pow(-1., ia)*pow(-1., ka)*int2[I*n3+J*n2+K*norbs+L]*rowCoeffs[2*i+ia]*rowCoeffs[2*j+ja]*rowCoeffs[2*k+ka]*rowCoeffs[2*l+la];
		      }
	      }
      }
    }
    
  }
  
  void writeIntNoSymm(int norbs, double* int1, double* int2, double coreE, int nelec, int* irrep)
  {
    //int norbs = *pnorbs;
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
	from a difference in notation between pyscf and SHCI. */
	void r2RDM( double * twoRDM, size_t norb,  char * fIn ){

		size_t norb2 = norb * norb;
		size_t norb3 = norb2 * norb;
		size_t ikjl[4]; // Indices to be used to for twoRDM.
		size_t counter;
		size_t i, j, k ,l, indx;
		string line;

		ifstream myfile ( fIn ); 	// Begin reading fIn

		// Grab first line (number of orbitals) and check to make sure it
		// matches the input norb.
		if ( myfile.is_open() ) { getline( myfile, line ); }
		size_t chkOrbs = strtol( line.c_str(), NULL, 10 );
		assert( chkOrbs == norb );

		if ( myfile.is_open() )
		{
			while ( getline( myfile, line ) )
			{
				// Converts the line into a c string so it can be broken up
				// with strtok().
				// TODO and check that "new" is necessary here.
				char *cStr = new char[line.length() + 1];
				strcpy(cStr, line.c_str());
				char * chPtr; // Resets the ptr every iteration so that we
							  // don't use a ptr to the wrong location.

				// Begin Cutting up line of data.
				chPtr = strtok( cStr, " ");
				counter = 0; // Reset counter.

				// Selects the remaining tokens in the line, one at a time so
				// there is no need to keep track of the pointers in strtol and
				// strtod.
				while ( chPtr != NULL )
				{
					if ( counter < 4 )
					{
						ikjl[ counter ] = strtol( chPtr, NULL, 10 );
					}

					else
					{
						i=ikjl[0], j=ikjl[2], k=ikjl[1], l=ikjl[3];
						indx = i*norb3 + j*norb2 + k*norb + l;
						twoRDM[indx] = strtod( chPtr, NULL );
					}

					// Cut up the line and adjust the counter.
					chPtr = strtok ( NULL, " " );
					++counter;
				}

				// Delete variables to avoid any memory leaks.
				delete chPtr;
				delete [] cStr;
			}

			myfile.close();
		}

		else cout << "Error opening spatalRDM file.";

	}


	// This function is used to create the header for the FCI dump file
  void writeFDHead ( ofstream& fOut, size_t norb, size_t nelec, size_t ms, int* orbsym )
	{
		// Creating the four lines of the header.
		ostringstream h1, h2, h34;
		h1 << " &FCI NORB=" << norb << ",NELEC=" << nelec << ",MS2=" << ms << ",";

		// TODO: Temporarily setting the orbsym to be 1 for every orbital.
		h2 << "  ORBSYM=";

		for ( size_t i = 0; i < norb; ++i )
		{
		  h2 << orbsym[i]<<",";
		}

		h34 << "  ISYM=1,\n" << " &END";

		if ( fOut.is_open() )
		{
			fOut << h1.str() << endl;
			fOut << h2.str() << endl;
			fOut << h34.str() << endl;
		}
	}

	// Only for 8-fold symmetry
	void writeERI ( ofstream& fOut, double * eri, size_t norb, double tol)
	{
		// Currently only compatible with 8-fold symmetry.
		size_t ij = 0;
		size_t kl = 0;
		size_t ijkl = 0;

		// Add assert statements.
		for ( size_t i = 0; i < norb; ++i )
		{
			for ( size_t j = 0; j < i + 1; ++j )
			{
				kl = 0;
				for ( size_t k = 0; k < i + 1; ++k )
				{
					for ( size_t l = 0; l < k + 1; ++l )
					{
						if ( ij >= kl )
						{
							if ( fabs( eri[ijkl] ) > tol )
							{
								fOut << eri[ ijkl ] << "    ";
								fOut << i+1 << "  " << j+1 << "  ";
								fOut << k+1 << "  " << l+1 << "\n";
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

	void writeHCore ( ofstream& fOut, double * h1e, size_t norb, double tol )
	{
		size_t maxIdx = norb * norb;
		size_t p;
		size_t q;
		string endStr = "  0  0\n";

		for ( size_t i = 0; i < maxIdx; ++i )
		{
			if ( fabs( h1e[ i ] ) > tol )
			{
				div_t pq = div( i, norb );
				p = pq.rem + 1;
				q = pq.quot + 1;
				fOut << h1e[ i ] << "    " << p << "  " << q << endStr;
			}
		}
	}

	// This function is an alternative to pyscf.tools.fcidump.from_integral and
	// should be used when working with large active spaces.
	void fcidumpFromIntegral ( char * fileName, double * h1eff, double * eri,
				   size_t norb, size_t nelec, double ecore, 
				   int* orbsym, size_t ms,
				   double tol = 1e-15 )
	{

		ofstream fOut ( fileName );
		fOut.precision(16);

		// Write FCIDUmp File
		writeFDHead( fOut, norb, nelec, ms, orbsym );
		writeERI( fOut, eri, norb, tol );
		writeHCore( fOut, h1eff, norb, tol );
		fOut << ecore << "  0  0  0  0\n" << endl;

		fOut.close();

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
		// Initialize index variables.
		// size_t ijkl, ijkd, ijcd, ibcd, abcd;
		// size_t ia, jb, kc, ld, i, j ,k, l, a, b, c, d;
		size_t n2 = n*n, n3 = n2*n;

		// Switch statement deals with the different indices to sum over.
		switch ( indx )
		{
			case 0:
				#pragma omp parallel for
					for ( size_t a = 0; a < n; ++a )
						for ( size_t b = 0; b < n; ++b )
							for ( size_t c = 0; c < n; ++c )
								for ( size_t d = 0; d < n; ++d )
								{
									size_t abcd = d + c*n + b*n2 + a*n3;
									mout[abcd] = 0;

									for ( size_t i = 0; i < n; ++i )
									{
										size_t ia = i*n + a;
										size_t ibcd = d + c*n + b*n2 + i*n3;
										mout[abcd] += m1[ibcd] * m2[ia];
									}
								}
				break;

			case 1:
				#pragma omp parallel for
					for ( size_t i = 0; i < n; ++i )
						for ( size_t b = 0; b < n; ++b )
							for ( size_t c = 0; c < n; ++c )
								for ( size_t d = 0; d < n; ++d )
								{
									size_t ibcd = d + c*n + b*n2 + i*n3;
									mout[ibcd] = 0;

									for ( size_t j = 0; j < n; ++j )
									{
										size_t jb = j*n + b;
										size_t ijcd = d + c*n + j*n2 + i*n3;
										mout[ibcd] += m1[ijcd] * m2[jb];
									}
								}
				break;

			case 2:
				#pragma omp parallel for
					for ( size_t i = 0; i < n; ++i )
						for ( size_t j = 0; j < n; ++j )
							for ( size_t c = 0; c < n; ++c )
								for ( size_t d = 0; d < n; ++d )
								{
									size_t ijcd = d + c*n + j*n2 + i*n3;
									mout[ijcd] = 0;

									for ( size_t k = 0; k < n; ++k )
									{
										size_t kc = k*n + c;
										size_t ijkd = d + k*n + j*n2 + i*n3;
										mout[ijcd] += m1[ijkd] * m2[kc];
									}
								}
				break;

			default:
				#pragma omp parallel for
					for (size_t i = 0; i < n; ++i )
						for (size_t j = 0; j < n; ++j )
							for (size_t k = 0; k < n; ++k )
								for (size_t d = 0; d < n; ++d )
								{
									size_t ijkd = d + k*n + j*n2 + i*n3;
									mout[ijkd] = 0;

									for (size_t l = 0; l < n; ++l )
									{
										size_t ld = l*n + d;
										size_t ijkl = l + k*n + j*n2 + i*n3;
										mout[ijkd] += m1[ijkl] * m2[ld];
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
		// size_t i, j, a, b, ij, jb, ia, ib, ab;

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

}
