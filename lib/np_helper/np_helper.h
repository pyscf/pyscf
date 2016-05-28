/*
 * numpy helper
 */

#include <complex.h>

#define BLOCK_DIM    120

#define HERMITIAN    1
#define ANTIHERMI    2
#define SYMMETRIC    3

void NPdsymm_triu(int n, double *mat, int hermi);
void NPzhermi_triu(int n, double complex *mat, int hermi);
void NPdunpack_tril(int n, double *tril, double *mat, int hermi);
void NPdunpack_row(int ndim, int row_id, double *tril, double *row);
void NPzunpack_tril(int n, double complex *tril, double complex *mat,
                    int hermi);
void NPdpack_tril(int n, double *tril, double *mat);
void NPzpack_tril(int n, double complex *tril, double complex *mat);

void NPdtranspose(int n, int m, double *a, double *at, int blk);
void NPztranspose(int n, int m, double complex *a, double complex *at, int blk);

void NPdunpack_tril_2d(int count, int n, double *tril, double *mat, int hermi);
void NPzunpack_tril_2d(int count, int n,
                       double complex *tril, double complex *mat, int hermi);
void NPdpack_tril_2d(int count, int n, double *tril, double *mat);

