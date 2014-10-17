/*
 *
 */

#include <string.h>
#include <complex.h>
#include "np_helper.h"

void NPdsymm_triu(int n, double *mat, int hermi)
{
        int ic, ic1, jc, jc1;
        long i, j, i1;

        if (hermi == HERMITIAN) {
                for (ic = 0; ic < n; ic += BLOCK_DIM) {
                        ic1 = ic + BLOCK_DIM;
                        if (ic1 > n) {
                                ic1 = n;
                        }
                        for (jc = 0; jc < ic; jc += BLOCK_DIM) {
                                jc1 = jc + BLOCK_DIM;
                                for (i1 = ic; i1 < ic1; i1++) {
                                for (j = jc; j < jc1; j++) {
                                        mat[j*n+i1] = mat[i1*n+j];
                                } }
                        }
                        for (i = ic; i < ic1; i++) {
                        for (j = ic; j < i; j++) {
                                mat[j*n+i] = mat[i*n+j];
                        } }
                }
        } else {
                for (ic = 0; ic < n; ic += BLOCK_DIM) {
                        ic1 = ic + BLOCK_DIM;
                        if (ic1 > n) {
                                ic1 = n;
                        }
                        for (jc = 0; jc < ic; jc += BLOCK_DIM) {
                                jc1 = jc + BLOCK_DIM;
                                for (i1 = ic; i1 < ic1; i1++) {
                                for (j = jc; j < jc1; j++) {
                                        mat[j*n+i1] = -mat[i1*n+j];
                                } }
                        }
                        for (i = ic; i < ic1; i++) {
                        for (j = ic; j < i; j++) {
                                mat[j*n+i] = -mat[i*n+j];
                        } }
                }
        }
}

void NPzhermi_triu(int n, double complex *mat, int hermi)
{
        int ic, ic1, jc, jc1;
        long i, j, i1;

        if (hermi == HERMITIAN) {
                for (ic = 0; ic < n; ic += BLOCK_DIM) {
                        ic1 = ic + BLOCK_DIM;
                        if (ic1 > n) {
                                ic1 = n;
                        }
                        for (jc = 0; jc < ic; jc += BLOCK_DIM) {
                                jc1 = jc + BLOCK_DIM;
                                for (i1 = ic; i1 < ic1; i1++) {
                                for (j = jc; j < jc1; j++) {
                                        mat[j*n+i1] = conj(mat[i1*n+j]);
                                } }
                        }
                        for (i = ic; i < ic1; i++) {
                        for (j = ic; j < i; j++) {
                                mat[j*n+i] = conj(mat[i*n+j]);
                        } }
                }
        } else {
                for (ic = 0; ic < n; ic += BLOCK_DIM) {
                        ic1 = ic + BLOCK_DIM;
                        if (ic1 > n) {
                                ic1 = n;
                        }
                        for (jc = 0; jc < ic; jc += BLOCK_DIM) {
                                jc1 = jc + BLOCK_DIM;
                                for (i1 = ic; i1 < ic1; i1++) {
                                for (j = jc; j < jc1; j++) {
                                        mat[j*n+i1] = -conj(mat[i1*n+j]);
                                } }
                        }
                        for (i = ic; i < ic1; i++) {
                        for (j = ic; j < i; j++) {
                                mat[j*n+i] = -conj(mat[i*n+j]);
                        } }
                }
        }
}


void NPdunpack_tril(int n, double *tril, double *mat, int hermi)
{
        long i, j, ij;

        for (ij = 0, i = 0; i < n; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        mat[i*n+j] = tril[ij];
                }
        }
        if (hermi) {
                NPdsymm_triu(n, mat, hermi);
        }
}

// unpack one row from the compact matrix-tril coefficients
void NPdunpack_row(int ndim, int row_id, double *tril, double *row)
{
        int i;
        unsigned long idx = (unsigned long)row_id * (row_id + 1) / 2;
        memcpy(row, tril+idx, sizeof(double)*row_id);
        for (i = row_id; i < ndim; i++) {
                idx += i;
                row[i] = tril[idx];
        }
}

void NPzunpack_tril(int n, double complex *tril, double complex *mat,
                    int hermi)
{
        long i, j, ij;
        for (ij = 0, i = 0; i < n; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        mat[i*n+j] = tril[ij];
                }
        }
        if (hermi) {
                NPzhermi_triu(n, mat, hermi);
        }
}

void NPdpack_tril(int n, double *tril, double *mat)
{
        long i, j, ij;
        for (ij = 0, i = 0; i < n; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        tril[ij] = mat[i*n+j];
                }
        }
}

void NPzpack_tril(int n, double complex *tril, double complex *mat)
{
        long i, j, ij;
        for (ij = 0, i = 0; i < n; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        tril[ij] = mat[i*n+j];
                }
        }
}
