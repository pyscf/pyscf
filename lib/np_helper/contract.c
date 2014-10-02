/*
 *
 */

#include <string.h>
#include <assert.h>
#include "vhf/fblas.h"


/* prod(a,b,c,d) += da * \sum_{pq} t(a,p,b,q) * g(c,p,d,q) */
void NPdot_aibj_cidj(double *t, int *nt, double *g, int *ng,
                     double alpha, double beta, double *prod)
{
        assert(nt[1] == ng[1]);
        assert(nt[3] == ng[3]);

        const int INC1 = 1;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        int i, j, k;
        int ng02 = ng[0] * ng[2];
        double *pt, *pg, *pprod;

        int n = nt[0] * nt[2] * ng[0] * ng[2];
        if (beta == 0) {
                memset(prod, 0, sizeof(double)*n);
        } else {
                dscal_(&n, &beta, prod, &INC1);
        }

        for (i = 0; i < nt[0]; i++) {
        for (j = 0; j < ng[0]; j++) {
                pt = t + nt[3] * nt[2] * nt[1] * i;
                pg = g + ng[3] * ng[2] * ng[1] * j;
                pprod = prod + ng[2] * ng[0] * nt[2] * i + ng[2] * j;
                for (k = 0; k < ng[1]; k++) {
                        dgemm_(&TRANS_T, &TRANS_N, &ng[2], &nt[2], &ng[3],
                               &alpha, pg, &ng[3], pt, &nt[3],
                               &D1, pprod, &ng02);
                        pt += nt[3] * nt[2];
                        pg += ng[3] * ng[2];
                }
        } }
}

/* prod(a,b,c,d) += da * \sum_{pq} t(a,p,q,b) * g(c,p,q,d) */
void NPdot_aijb_cijd(double *t, int *nt, double *g, int *ng,
                     double alpha, double beta, double *prod)
{
        assert(nt[1] == ng[1]);
        assert(nt[2] == ng[2]);

        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        int i, k;
        int ng12 = ng[1] * ng[2];
        int ng03 = ng[0] * ng[3];
        double *pt, *pg, *pprod;

        for (i = 0; i < nt[0]; i++) {
        for (k = 0; k < ng[0]; k++) {
                pt = t + nt[3] * nt[2] * nt[1] * i;
                pg = g + ng[3] * ng[2] * ng[1] * k;
                pprod = prod + ng[3] * ng[0] * nt[3] * i + ng[3] * k;
                dgemm_(&TRANS_N, &TRANS_T, &ng[3], &nt[3], &ng12,
                       &alpha, pg, &ng[3], pt, &nt[3],
                       &beta, pprod, &ng03);
        } }
}

/* prod(a,b,c,d) += da * \sum_{pq} t(a,p,b,q) * g(c,p,q,d) */
void NPdot_aibj_cijd(double *t, int *nt, double *g, int *ng,
                     double alpha, double beta, double *prod)
{
        assert(nt[1] == ng[1]);
        assert(nt[3] == ng[2]);

        const int INC1 = 1;
        const char TRANS_N = 'N';
        const double D1 = 1;
        int i, k, p;
        int ng03 = ng[0] * ng[3];
        double *pt, *pg, *pprod;
        int n = nt[0] * nt[3] * ng[1] * ng[3];
        if (beta == 0) {
                memset(prod, 0, sizeof(double)*n);
        } else {
                dscal_(&n, &beta, prod, &INC1);
        }

        for (i = 0; i < nt[0]; i++) {
        for (k = 0; k < ng[0]; k++) {
                pprod = prod + ng[3] * ng[0] * nt[2] * i + ng[3] * k;
                pt = t + nt[3] * nt[2] * nt[1] * i;
                pg = g + ng[3] * ng[2] * ng[1] * k;
                for (p = 0; p < ng[0]; p++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ng[3], &nt[2], &ng[2],
                               &alpha, pg, &ng[3], pt, &nt[3],
                               &D1, pprod, &ng03);
                        pt += nt[3] * nt[2];
                        pg += ng[3] * ng[2];
                }
        } }
}

/* prod(a,b,c,d) += da * \sum_{pq} t(a,p,b,q) * g(p,c,q,d) */
void NPdot_aibj_icjd(double *t, int *nt, double *g, int *ng,
                     double alpha, double beta, double *prod)
{
        assert(nt[1] == ng[0]);
        assert(nt[3] == ng[2]);

        const int INC1 = 1;
        const double D1 = 1;
        const char TRANS_N = 'N';
        int i, k, p;
        int ng13 = ng[1] * ng[3];
        double *pt, *pg, *pprod;

        int n = nt[0] * nt[2] * ng[1] * ng[3];
        if (beta == 0) {
                memset(prod, 0, sizeof(double)*n);
        } else {
                dscal_(&n, &beta, prod, &INC1);
        }

        for (i = 0; i < nt[0]; i++) {
        for (k = 0; k < ng[1]; k++) {
                pprod = prod + ng[3] * ng[1] * nt[2] * i + ng[3] * k;
                pt = t + nt[3] * nt[2] * nt[1] * i;
                pg = g + ng[3] * ng[2] * k;
                for (p = 0; p < ng[0]; p++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ng[3], &nt[2], &nt[3],
                               &alpha, pg, &ng[3], pt, &nt[3],
                               &D1, pprod, &ng13);
                        pt += nt[3] * nt[2];
                        pg += ng[3] * ng[2] * ng[1];
                }
        } }
}
/* prod(a,b,c,d) += da * \sum_{pq} t(a,p,q,b) * g(p,c,q,d) */
void NPdot_aijb_icjd(double *t, int *nt, double *g, int *ng,
                     double alpha, double beta, double *prod)
{
        assert(nt[1] == ng[0]);
        assert(nt[2] == ng[2]);

        const int INC1 = 1;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        int i, k, p;
        int ng13 = ng[1] * ng[3];
        double *pt, *pg, *pprod;

        int n = nt[0] * nt[3] * ng[1] * ng[3];
        if (beta == 0) {
                memset(prod, 0, sizeof(double)*n);
        } else {
                dscal_(&n, &beta, prod, &INC1);
        }

        for (i = 0; i < nt[0]; i++) {
        for (k = 0; k < ng[1]; k++) {
                pprod = prod + ng[3] * ng[1] * nt[3] * i + ng[3] * k;
                pt = t + nt[3] * nt[2] * nt[1] * i;
                pg = g + ng[3] * ng[2] * k;
                for (p = 0; p < ng[0]; p++) {
                        dgemm_(&TRANS_N, &TRANS_T, &ng[3], &nt[3], &nt[2],
                               &alpha, pg, &ng[3], pt, &nt[3],
                               &D1, pprod, &ng13);
                        pt += nt[3] * nt[2];
                        pg += ng[3] * ng[2] * ng[1];
                }
        } }
}
