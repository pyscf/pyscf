/*
 * numpy helper
 */

#include <complex.h>

#define BLOCK_DIM    120

#define HERMITIAN    1
#define ANTIHERMI    2

void NPdot_aibj_cidj(double *t, int *nt, double *g, int *ng,
                     double alpha, double beta, double *prod);
void NPdot_aijb_cijd(double *t, int *nt, double *g, int *ng,
                     double alpha, double beta, double *prod);
void NPdot_aibj_cijd(double *t, int *nt, double *g, int *ng,
                     double alpha, double beta, double *prod);
void NPdot_aibj_icjd(double *t, int *nt, double *g, int *ng,
                     double alpha, double beta, double *prod);
void NPdot_aijb_icjd(double *t, int *nt, double *g, int *ng,
                     double alpha, double beta, double *prod);

void NPdsymm_triu(int n, double *mat, int hermi);
void NPzhermi_triu(int n, double complex *mat, int hermi);
void NPdunpack_tril(int n, double *tril, double *mat);
void NPdunpack_row(int ndim, int row_id, double *tril, double *row);
void NPzunpack_tril(int n, double complex *tril, double complex *mat);
void NPdpack_tril(int n, double *tril, double *mat);
void NPzpack_tril(int n, double complex *tril, double complex *mat);
