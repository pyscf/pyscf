/*
 * numpy helper
 */

#include <complex.h>

#define BLOCK_DIM    120

#define HERMITIAN    1
#define ANTIHERMI    2
#define SYMMETRIC    3

#define MIN(X, Y)       ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y)       ((X) > (Y) ? (X) : (Y))

#define TRIU_LOOP(I, J) \
        for (j0 = 0; j0 < n; j0+=BLOCK_DIM) \
                for (I = 0, j1 = MIN(j0+BLOCK_DIM, n); I < j1; I++) \
                        for (J = MAX(I,j0); J < j1; J++)

void NPdsymm_triu(int n, double *mat, int hermi);
void NPzhermi_triu(int n, double complex *mat, int hermi);
void NPdunpack_tril(int n, double *tril, double *mat, int hermi);
void NPdunpack_row(int ndim, int row_id, double *tril, double *row);
void NPzunpack_tril(int n, double complex *tril, double complex *mat,
                    int hermi);
void NPdpack_tril(int n, double *tril, double *mat);
void NPzpack_tril(int n, double complex *tril, double complex *mat);

void NPdtranspose(int n, int m, double *a, double *at);
void NPztranspose(int n, int m, double complex *a, double complex *at);
void NPdtranspose_021(int *shape, double *a, double *at);
void NPztranspose_021(int *shape, double complex *a, double complex *at);

void NPdunpack_tril_2d(int count, int n, double *tril, double *mat, int hermi);
void NPzunpack_tril_2d(int count, int n,
                       double complex *tril, double complex *mat, int hermi);
void NPdpack_tril_2d(int count, int n, double *tril, double *mat);

void NPomp_dsum_reduce_inplace(double **vec, size_t count);
void NPomp_dprod_reduce_inplace(double **vec, size_t count);
void NPomp_zsum_reduce_inplace(double complex **vec, size_t count);
void NPomp_zprod_reduce_inplace(double complex **vec, size_t count);
