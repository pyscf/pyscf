/*
 * File: nr_ao2mo_o4.c
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 */

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
//#define NDEBUG

//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "vhf/cvhf.h"
#include "vhf/fblas.h"
#include "nr_ao2mo_o3.h"

#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y)        ((X) > (Y) ? (X) : (Y))


static void nr_reorder(int nao, int kloc, int dk, int lloc, int dl,
                       int *idx_tri, double *eri, double *buf)
{
        int i, j, k, l, k0, l0, ij, kl;
        double tmp;
        for (k0 = kloc, k = 0; k < dk; k++, k0++) {
        for (l0 = lloc, l = 0; l < dl; l++, l0++) {
                if (k0 >= l0) {
                        kl = l * dk + k;
                        for (i = 0, ij = 0; i < nao; i++) {
                        for (j = 0; j <= i; j++, ij++) {
                                tmp = eri[idx_tri[ij]*dk*dl+kl];
                                buf[i*nao+j] = tmp;
                                buf[j*nao+i] = tmp;
                        } }
                        buf += nao * nao;
                }
        } }
}
/* [[.] [..] [...] [....] [.....]] => transform electron 1 from AO to MO
 *                                              ...,kl1,kl2,...
 *    (Fortran order) [. + + . .]    (C order) [      +      ]
 *                    [  + + . .]          ... [      +      ]
 * =>                 [    + . .] =>       ij1 [ ...  +  ... ]
 *                    [      . .]          ij2 [      +      ]
 *                    [        .]          ... [      +      ]
 * ksh and lsh are kept in AO representation, all i and j for (ksh,lsh)
 * are transformed to MO representation
 */
/* prefer i_count > j_count, and i_start >= j_start+j_count
 *      __ j_start
 * j_count [.        ]
 *      __ [. .      ]
 * i_start [+ + .    ]
 *         [+ + . .  ]
 *         [+ + . . .]
 */
void AO2MOnr_tri_e1_o3(double *vout, double *eri, double *mo_coeff,
                       int ksh, int lsh, struct _AO2MOEnvs *envs)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int nao = envs->nao;
        const int i_start = envs->bra_start;
        const int i_count = envs->bra_count;
        const int j_start = envs->ket_start;
        const int j_count = envs->ket_count;
        const int dk = CINTcgto_spheric(ksh, envs->bas);
        const int dl = CINTcgto_spheric(lsh, envs->bas);
        double *vc = malloc(sizeof(double) * nao*j_count*dk*dl);
        double *cvc = malloc(sizeof(double) * nao*nao*dk*dl);
        int *idx_tri = envs->idx_tri;
        int *ao_loc = envs->ao_loc;
        int m;
        int lenkl = (ksh == lsh) ? (dk*(dk+1)/2) : (dk*dl);

        nr_reorder(nao, ao_loc[ksh], dk, ao_loc[lsh], dl, idx_tri, eri, cvc);

        m = lenkl * nao;
        dgemm_(&TRANS_T, &TRANS_N, &m, &j_count, &nao,
               &D1, cvc, &nao, mo_coeff+j_start*nao, &nao,
               &D0, vc, &m);
        m = lenkl * j_count;
        dgemm_(&TRANS_T, &TRANS_N, &m, &i_count, &nao,
               &D1, vc, &nao, mo_coeff+i_start*nao, &nao,
               &D0, cvc, &m);

        int npair = nao*(nao+1)/2;
        int ish_start = envs->ish_start;
        int kloff = ao_loc[ish_start]*(ao_loc[ish_start]+1)/2;
        int i, j, k, l, kl, j1;
        double *pcvc, *pvout;

        // (kl,ij) => (ij, ..kl..)
        if (ksh == lsh) {
                for (i = 0; i < i_count; i++) {
                        // j < j_count and j+j_start <= i+i_start
                        j1 = i+i_start-j_start+1;
                        for (j = 0; j < MIN(j_count,j1); j++) {
                                pcvc = cvc + (i*j_count+j)*lenkl;
                                for (k = ao_loc[ksh]; k < ao_loc[ksh]+dk; k++) {
                                for (l = ao_loc[lsh]; l < ao_loc[lsh]+dl; l++) {
                                        if (k >= l) {
                                                kl = k*(k+1)/2+l - kloff;
                                                vout[kl] = *pcvc;
                                                pcvc++;
                                        }
                                } }
                                vout += npair;
                        }
                }
        } else {
                for (i = 0; i < i_count; i++) {
                        // j < j_count and j+j_start <= i+i_start
                        j1 = i+i_start-j_start+1;
                        for (j = 0; j < MIN(j_count,j1); j++) {
                                pcvc = cvc + (i*j_count+j)*lenkl;
                                k = ao_loc[ksh];
                                pvout = vout + k*(k+1)/2 - kloff;
                                for (; k < ao_loc[ksh]+dk; k++, pvout+=k) {
                                for (l = ao_loc[lsh]; l < ao_loc[lsh]+dl; l++) {
                                        pvout[l] = *pcvc;
                                        pcvc++;
                                } }
                                vout += npair;
                        }
                }
        }

        free(vc);
        free(cvc);
}

/* prefer i_count < j_count, and i_start >= j_start+j_count
 *      __ j_start
 *         [.        ]
 * j_count [. .      ]
 *      __ [. . .    ]
 * i_start [+ + + .  ]
 *         [+ + + . .]
 */
void AO2MOnr_tri_e1_o4(double *vout, double *eri, double *mo_coeff,
                       int ksh, int lsh, struct _AO2MOEnvs *envs)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int nao = envs->nao;
        const int i_start = envs->bra_start;
        const int i_count = envs->bra_count;
        const int j_start = envs->ket_start;
        const int j_count = envs->ket_count;
        const int dk = CINTcgto_spheric(ksh, envs->bas);
        const int dl = CINTcgto_spheric(lsh, envs->bas);
        double *vc = malloc(sizeof(double) * nao*j_count*dk*dl);
        double *cvc = malloc(sizeof(double) * nao*nao*dk*dl);
        int *idx_tri = envs->idx_tri;
        int *ao_loc = envs->ao_loc;
        int m;
        int lenkl = (ksh == lsh) ? (dk*(dk+1)/2) : (dk*dl);

        nr_reorder(nao, ao_loc[ksh], dk, ao_loc[lsh], dl, idx_tri, eri, cvc);

        m = lenkl * nao;
        dgemm_(&TRANS_T, &TRANS_N, &m, &i_count, &nao,
               &D1, cvc, &nao, mo_coeff+i_start*nao, &nao,
               &D0, vc, &m);
        m = lenkl * i_count;
        dgemm_(&TRANS_T, &TRANS_N, &m, &j_count, &nao,
               &D1, vc, &nao, mo_coeff+j_start*nao, &nao,
               &D0, cvc, &m);

        int npair = nao*(nao+1)/2;
        int ish_start = envs->ish_start;
        int kloff = ao_loc[ish_start]*(ao_loc[ish_start]+1)/2;
        int i, j, k, l, kl, j1;
        double *pcvc, *pvout;

        // (kl,ij) => (ij, ..kl..)
        if (ksh == lsh) {
                for (i = 0; i < i_count; i++) {
                        // j < j_count and j+j_start <= i+i_start
                        j1 = i+i_start-j_start+1;
                        for (j = 0; j < MIN(j_count,j1); j++) {
                                pcvc = cvc + (j*i_count+i)*lenkl;
                                for (k = ao_loc[ksh]; k < ao_loc[ksh]+dk; k++) {
                                for (l = ao_loc[lsh]; l < ao_loc[lsh]+dl; l++) {
                                        if (k >= l) {
                                                kl = k*(k+1)/2+l - kloff;
                                                vout[kl] = *pcvc;
                                                pcvc++;
                                        }
                                } }
                                vout += npair;
                        }
                }
        } else {
                for (i = 0; i < i_count; i++) {
                        // j < j_count and j+j_start <= i+i_start
                        j1 = i+i_start-j_start+1;
                        for (j = 0; j < MIN(j_count,j1); j++) {
                                pcvc = cvc + (j*i_count+i)*lenkl;
                                k = ao_loc[ksh];
                                pvout = vout + k*(k+1)/2 - kloff;
                                for (; k < ao_loc[ksh]+dk; k++, pvout+=k) {
                                for (l = ao_loc[lsh]; l < ao_loc[lsh]+dl; l++) {
                                        pvout[l] = *pcvc;
                                        pcvc++;
                                } }
                                vout += npair;
                        }
                }
        }

        free(vc);
        free(cvc);
}

