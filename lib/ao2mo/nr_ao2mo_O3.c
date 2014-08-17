/*
 * File: nr_ao2mo_O3.c
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <assert.h>
//#define NDEBUG

#include "cint.h"
#include "vhf/cvhf.h"
#include "vhf/fblas.h"
#include "nr_ao2mo_O3.h"

#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y)        ((X) > (Y) ? (X) : (Y))


/* ***************************************************
 * calculate the lower triangle part (of Fortran order matrix)
 *   _  |-- jdiag_off -|
 *   |  [ . . . . . . . .     ]
 *   m  [ . . . . . . . . .   ]
 *   |  [ . . . . . . . . . . ] __ idiag_end
 *   _  [ . . . . . . . . . . ]
 *      |--------- n ---------|
 */
void AO2MOdtrimm_o1(int m, int n, int k, int jdiag_off,
                    double *a, double *b, double *c)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int BLK = 56;
        const int idiag_end = (int)n - jdiag_off;
        int mstart, mleft, mend;
        // lower triangle part
        for (mstart = 0; mstart+BLK < idiag_end; mstart+=BLK) {
                mend = jdiag_off + mstart + BLK;
                dgemm_(&TRANS_T, &TRANS_N, &BLK, &mend, &k,
                       &D1, a+mstart*k, &k, b, &k, &D0, c+mstart, &m);
        }
        // the below rectangle part
        mleft = m - mstart;
        dgemm_(&TRANS_T, &TRANS_N, &mleft, &n, &k,
               &D1, a+mstart*k, &k, b, &k, &D0, c+mstart, &m);
}

/*
 * calculate the lower triangle part (of Fortran order matrix)
 *   _  |-- jdiag_off -|
 *   |  [ . . . . . . . .     ]
 *   m  [ . . . . . . . . .   ]
 *   |  [ . . . . . . . . . . ]
 *   _  [ . . . . . . . . . . ]
 *      |--------- n ---------|
 *      |----- nend -----|  nleft = nblk_diag * BLK
 *                       |----|
 */
void AO2MOdtrimm_o2(int m, int n, int k, int jdiag_off,
                    double *a, double *b, double *c)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int BLK = 48;
        const int nblk_diag = MAX(0, (n-jdiag_off)/BLK);
        const int nend = n - nblk_diag * BLK;

        dgemm_(&TRANS_T, &TRANS_N, &m, &nend, &k,
               &D1, a, &k, b, &k, &D0, c, &m);

        // lower triangle part
        int mstart, nleft;
        for (nleft = nblk_diag*BLK, b = b+nend*k, c = c+nend*m;
             nleft > 0; b+=BLK*k, c+=BLK*m, nleft-=BLK) {
                mstart = m - nleft;
                dgemm_(&TRANS_T, &TRANS_N, &nleft, &BLK, &k,
                       &D1, a+mstart*k, &k, b, &k, &D0, c+mstart, &m);
        }
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
static void trans_nr_tri_e1_o0(double *vout, double *eri, double *mo_coeff,
                               int ksh, int lsh, struct _AO2MOEnvs *envs)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        const int nao = envs->nao;
        const int i_start = envs->bra_start;
        const int i_count = envs->bra_count;
        const int j_start = envs->ket_start;
        const int j_count = envs->ket_count;
        const int dk = CINTcgto_spheric(ksh, envs->bas);
        const int dl = CINTcgto_spheric(lsh, envs->bas);
        double *vc = malloc(sizeof(double) * nao*j_count);
        double *cvc = malloc(sizeof(double) * nao*nao);
        const int *idx_tri = envs->idx_tri;
        const int *ao_loc = envs->ao_loc;
        int i, j, k, l, k0, l0, j1, ij, kl;
        const unsigned long npair = nao*(nao+1)/2;
        const int ish_start = envs->ish_start;
        const int offeri = ao_loc[ish_start]*(ao_loc[ish_start]+1)/2;

        for (l0 = ao_loc[lsh], l = 0; l < dl; l++, l0++) {
        for (k0 = ao_loc[ksh], k = 0; k < dk; k++, k0++) {
        if (k0 >= l0) {
                kl = l * dk + k;
                for (i = 0, ij = 0; i < nao; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        // It should store the upper triangle in Fortran order
                        //cvc[j*nao+i] = eri[idx_tri[ij]*dk*dl+kl];
                        cvc[i*nao+j] = eri[idx_tri[ij]*dk*dl+kl];
                } }
                dsymm_(&SIDE_L, &UPLO_U, &nao, &j_count,
                       &D1, cvc, &nao, mo_coeff+j_start*nao, &nao,
                       &D0, vc, &nao);
                dgemm_(&TRANS_T, &TRANS_N, &i_count, &j_count, &nao,
                       &D1, mo_coeff+i_start*nao, &nao, vc, &nao,
                       &D0, cvc, &i_count);

                // (kl,ij) => (ij, ..kl..)
                kl = k0*(k0+1)/2+l0 - offeri;
                for (i = 0, ij = 0; i < i_count; i++) {
                        // j < j_count and j+j_start <= i+i_start
                        j1 = i+i_start-j_start+1;
                        for (j = 0; j < MIN(j_count,j1); j++, ij++) {
                                vout[ij*npair+kl] = cvc[j*i_count+i];
                        }
                }
        } } }

        free(vc);
        free(cvc);
}
static void trans_nr_tri_e1_o1(double *vout, double *eri, double *mo_coeff,
                               int ksh, int lsh, struct _AO2MOEnvs *envs)
{
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        const int nao = envs->nao;
        const int i_start = envs->bra_start;
        const int i_count = envs->bra_count;
        const int j_start = envs->ket_start;
        const int j_count = envs->ket_count;
        const int dk = CINTcgto_spheric(ksh, envs->bas);
        const int dl = CINTcgto_spheric(lsh, envs->bas);
        double *vc = malloc(sizeof(double) * nao*j_count);
        double *cvc = malloc(sizeof(double) * nao*nao);
        const int *idx_tri = envs->idx_tri;
        const int *ao_loc = envs->ao_loc;
        int i, j, k, l, k0, l0, j1;
        int ij = 0, kl;
        double *cache = malloc(sizeof(double) * dk*dl*i_count*j_count);
        int idxkl[4096];
        int lastkl = 0;

        for (l0 = ao_loc[lsh], l = 0; l < dl; l++, l0++) {
        for (k0 = ao_loc[ksh], k = 0; k < dk; k++, k0++) {
        if (k0 >= l0) {
                kl = l * dk + k;
                for (i = 0, ij = 0; i < nao; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        // It should store the upper triangle in Fortran order
                        //cvc[j*nao+i] = eri[idx_tri[ij]*dk*dl+kl];
                        cvc[i*nao+j] = eri[idx_tri[ij]*dk*dl+kl];
                } }
                dsymm_(&SIDE_L, &UPLO_U, &nao, &j_count,
                       &D1, cvc, &nao, mo_coeff+j_start*nao, &nao,
                       &D0, vc, &nao);
                AO2MOdtrimm_o1(i_count, j_count, nao, i_start,
                               mo_coeff+i_start*nao, vc, cvc);

                for (i = 0, ij = 0; i < i_count; i++) {
                        // j < j_count and j+j_start <= i+i_start
                        j1 = i+i_start-j_start+1;
                        for (j = 0; j < MIN(j_count,j1); j++, ij++) {
                                cache[lastkl*i_count*j_count+ij] = cvc[j*i_count+i];
                        }
                }

                idxkl[lastkl] = k0*(k0+1)/2+l0;
                lastkl++;
        } } }

        // (kl,ij) => (ij, ..kl..)
        const unsigned long npair = nao*(nao+1)/2;
        const int ish_start = envs->ish_start;
        const int offeri = ao_loc[ish_start]*(ao_loc[ish_start]+1)/2;
        for (i = 0; i < ij; i++) {
        for (j = 0; j < lastkl; j++) {
                vout[i*npair+idxkl[j]-offeri] = cache[j*i_count*j_count+i];
        } }

        free(cache);
        free(vc);
        free(cvc);
}
static void trans_nr_tri_e1_o2(double *vout, double *eri, double *mo_coeff,
                               int ksh, int lsh, struct _AO2MOEnvs *envs)
{
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        const int nao = envs->nao;
        const int i_start = envs->bra_start;
        const int i_count = envs->bra_count;
        const int j_start = envs->ket_start;
        const int j_count = envs->ket_count;
        const int dk = CINTcgto_spheric(ksh, envs->bas);
        const int dl = CINTcgto_spheric(lsh, envs->bas);
        double *vc = malloc(sizeof(double) * nao*i_count);
        double *cvc = malloc(sizeof(double) * nao*nao);
        const int *idx_tri = envs->idx_tri;
        const int *ao_loc = envs->ao_loc;
        int i, j, k, l, k0, l0, j1;
        int ij = 0, kl;
        double *cache = malloc(sizeof(double) * dk*dl*i_count*j_count);
        int idxkl[4096];
        int lastkl = 0;

        for (l0 = ao_loc[lsh], l = 0; l < dl; l++, l0++) {
        for (k0 = ao_loc[ksh], k = 0; k < dk; k++, k0++) {
        if (k0 >= l0) {
                kl = l * dk + k;
                for (i = 0, ij = 0; i < nao; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        // It should store the upper triangle in Fortran order
                        //cvc[j*nao+i] = eri[idx_tri[ij]*dk*dl+kl];
                        cvc[i*nao+j] = eri[idx_tri[ij]*dk*dl+kl];
                } }
                dsymm_(&SIDE_L, &UPLO_U, &nao, &i_count,
                       &D1, cvc, &nao, mo_coeff+i_start*nao, &nao,
                       &D0, vc, &nao);
                //dgemm_(&TRANS_T, &TRANS_N, &i_count, &j_count, &nao,
                //       &D1, vc, &nao, mo_coeff+j_start*nao, &nao,
                //       &D0, cvc, &i_count);
                AO2MOdtrimm_o2(i_count, j_count, nao, i_start,
                               vc, mo_coeff+j_start*nao, cvc);

                for (i = 0, ij = 0; i < i_count; i++) {
                        // j < j_count and j+j_start <= i+i_start
                        j1 = i+i_start-j_start+1;
                        for (j = 0; j < MIN(j_count,j1); j++, ij++) {
                                cache[lastkl*i_count*j_count+ij] = cvc[j*i_count+i];
                        }
                }

                idxkl[lastkl] = k0*(k0+1)/2+l0;
                lastkl++;
        } } }

        // (kl,ij) => (ij, ..kl..)
        const unsigned long npair = nao*(nao+1)/2;
        const int ish_start = envs->ish_start;
        const int offeri = ao_loc[ish_start]*(ao_loc[ish_start]+1)/2;
        for (i = 0; i < ij; i++) {
        for (j = 0; j < lastkl; j++) {
                vout[i*npair+idxkl[j]-offeri] = cache[j*i_count*j_count+i];
        } }

        free(cache);
        free(vc);
        free(cvc);
}

static void transform_kl(double *meri, double *mo_coeff, int ksh, int lsh,
                         struct _AO2MOEnvs *envs, CINTOpt *opt, CVHFOpt *vhfopt,
                         void (*ftrans_e1)())
{
        const int nao = envs->nao;
        const int dk = CINTcgto_spheric(ksh, envs->bas);
        const int dl = CINTcgto_spheric(lsh, envs->bas);
        double *eribuf = (double *)malloc(sizeof(double)*dk*dl*nao*nao);

        if (CVHFfill_nr_eri_o2(eribuf, ksh, lsh, envs->nbas,
                               envs->atm, envs->natm, envs->bas, envs->nbas,
                               envs->env, opt, vhfopt)) {
                (*ftrans_e1)(meri, eribuf, mo_coeff, ksh, lsh, envs);
        } else {
                const int i_start = envs->bra_start;
                const int i_count = envs->bra_count;
                const int j_start = envs->ket_start;
                const int j_count = envs->ket_count;
                const int *ao_loc = envs->ao_loc;
                const unsigned long npair = nao*(nao+1)/2;
                const int ish_start = envs->ish_start;
                const int offeri = ao_loc[ish_start]*(ao_loc[ish_start]+1)/2;

                int i, j, k, l, j1, ij, kl;
                for (i = 0, ij = 0; i < i_count; i++) {
                        // j < j_count and j+j_start <= i+i_start
                        j1 = i+i_start-j_start+1;
                        for (j = 0; j < MIN(j_count,j1); j++, ij++) {
                                for (k = ao_loc[ksh]; k < ao_loc[ksh]+dk; k++) {
                                for (l = ao_loc[lsh]; l < MIN(ao_loc[lsh]+dl,k+1); l++) {
                                        kl = k*(k+1)/2 + l - offeri;
                                        meri[ij*npair+kl] = 0;
                                } }
                        }
                }
        }

        free(eribuf);
}


void AO2MOnr_tri_e2_o0(double *vout, double *vin, double *mo_coeff, int nao,
                       int i_start, int i_count, int j_start, int j_count)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        double *vc = malloc(sizeof(double) * nao*j_count);
        double *cvc = malloc(sizeof(double) * nao*nao);
        int i, j, j1, ij;

        for (i = 0, ij = 0; i < nao; i++) {
        for (j = 0; j <= i; j++, ij++) {
                cvc[i*nao+j] = vin[ij];
        } }

        dsymm_(&SIDE_L, &UPLO_U, &nao, &j_count,
               &D1, cvc, &nao, mo_coeff+j_start*nao, &nao,
               &D0, vc, &nao);
        dgemm_(&TRANS_T, &TRANS_N, &i_count, &j_count, &nao,
               &D1, mo_coeff+i_start*nao, &nao, vc, &nao,
               &D0, cvc, &i_count);

        for (i = 0, ij = 0; i < i_count; i++) {
                // j < j_count and j+j_start <= i+i_start
                j1 = i+i_start-j_start+1;
                for (j = 0; j < MIN(j_count,j1); j++, ij++) {
                        vout[ij] = cvc[j*i_count+i];
                }
        }

        free(cvc);
        free(vc);
}

void AO2MOnr_tri_e2_o1(double *vout, double *vin, double *mo_coeff, int nao,
                       int i_start, int i_count, int j_start, int j_count)
{
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        double *vc = malloc(sizeof(double) * nao*j_count);
        double *cvc = malloc(sizeof(double) * nao*nao);
        int i, j, j1, ij;

        for (i = 0, ij = 0; i < nao; i++) {
        for (j = 0; j <= i; j++, ij++) {
                cvc[i*nao+j] = vin[ij];
        } }

        dsymm_(&SIDE_L, &UPLO_U, &nao, &j_count,
               &D1, cvc, &nao, mo_coeff+j_start*nao, &nao,
               &D0, vc, &nao);
        AO2MOdtrimm_o1(i_count, j_count, nao, i_start,
                       mo_coeff+i_start*nao, vc, cvc);

        for (i = 0, ij = 0; i < i_count; i++) {
                // j < j_count and j+j_start <= i+i_start
                j1 = i+i_start-j_start+1;
                for (j = 0; j < MIN(j_count,j1); j++, ij++) {
                        vout[ij] = cvc[j*i_count+i];
                }
        }

        free(cvc);
        free(vc);
}
void AO2MOnr_tri_e2_o2(double *vout, double *vin, double *mo_coeff, int nao,
                       int i_start, int i_count, int j_start, int j_count)
{
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        double *vc = malloc(sizeof(double) * nao*i_count);
        double *cvc = malloc(sizeof(double) * nao*nao);
        int i, j, j1, ij;

        for (i = 0, ij = 0; i < nao; i++) {
        for (j = 0; j <= i; j++, ij++) {
                cvc[i*nao+j] = vin[ij];
        } }

        dsymm_(&SIDE_L, &UPLO_U, &nao, &i_count,
               &D1, cvc, &nao, mo_coeff+i_start*nao, &nao,
               &D0, vc, &nao);
        //const char TRANS_N = 'N';
        //const char TRANS_T = 'T';
        //dgemm_(&TRANS_T, &TRANS_N, &i_count, &j_count, &nao,
        //       &D1, vc, &nao, mo_coeff+j_start*nao, &nao,
        //       &D0, cvc, &i_count);
        AO2MOdtrimm_o2(i_count, j_count, nao, i_start,
                       vc, mo_coeff+j_start*nao, cvc);

        for (i = 0, ij = 0; i < i_count; i++) {
                // j < j_count and j+j_start <= i+i_start
                j1 = i+i_start-j_start+1;
                for (j = 0; j < MIN(j_count,j1); j++, ij++) {
                        vout[ij] = cvc[j*i_count+i];
                }
        }

        free(cvc);
        free(vc);
}

/*
 * count the number of elements in the "lower triangle" block offset by
 * i_start, i_count, j_start, j_count
 * where j_start <= i_start and j_end <= i_end are assumed
 */
int AO2MOcount_ij(int i_start, int i_count, int j_start, int j_count)
{
        int ntri;
        if (j_start+j_count <= i_start) {
                ntri = 0;
        } else {
                int noff = j_start+j_count - (i_start+1);
                ntri = noff*(noff+1)/2;
        }
        return i_count * j_count - ntri;
}




/*
 * ************************************************
 * transform [ij] of integrals (i>j|k>l) from AO to MO representation.
 * i_start, j_start are row_id (not shell_id)
 */

void AO2MOnr_e1_o0(double *eri, double *mo_coeff,
                   int i_start, int i_count, int j_start, int j_count,
                   const int *atm, const int natm,
                   const int *bas, const int nbas, const double *env)
{
        assert(j_start <= i_start);
        assert(j_start+j_count <= i_start+i_count);
        const int nao = CINTtot_cgto_spheric(bas, nbas);
        int *ij2i = malloc(sizeof(int)*nbas*nbas);
        int *idx_tri = malloc(sizeof(int)*nao*nao);
        CVHFset_ij2i(ij2i, nbas);
        int *ao_loc = malloc(sizeof(int)*nbas);
        CINTshells_spheric_offset(ao_loc, bas, nbas);
        CVHFindex_blocks2tri(idx_tri, ao_loc, bas, nbas);

        struct _AO2MOEnvs envs = {natm, nbas, atm, bas, env, nao,
                                  0, nbas,
                                  i_start, i_count, j_start, j_count,
                                  ao_loc, idx_tri};
        CINTOpt *opt;
        cint2e_optimizer(&opt, atm, natm, bas, nbas, env);

        int k, l, kl;
#pragma omp parallel default(none) \
        shared(eri, mo_coeff, envs, opt, ij2i) \
        private(kl, k, l)
#pragma omp for nowait schedule(guided)
        for (kl = 0; kl < nbas*(nbas+1)/2; kl++) {
                k = ij2i[kl];
                l = kl - k*(k+1)/2;
                transform_kl(eri, mo_coeff, k, l, &envs, opt, NULL,
                             trans_nr_tri_e1_o0);
        }

        free(idx_tri);
        free(ij2i);
        free(ao_loc);
        CINTdel_optimizer(&opt);
}

void AO2MOnr_e2_o0(const int nrow, double *vout, double *vin,
                    double *mo_coeff, const int nao,
                    int i_start, int i_count, int j_start, int j_count)
{
        assert(j_start <= i_start);
        assert(j_start+j_count <= i_start+i_count);
        const unsigned long npair = (unsigned long)nao*(nao+1)/2;

        int i;
        const unsigned long nij = AO2MOcount_ij(i_start, i_count, j_start, j_count);
#pragma omp parallel default(none) \
        shared(vout, vin, mo_coeff, i_start, i_count, j_start, j_count) \
        private(i)
#pragma omp for nowait schedule(static)
        for (i = 0; i < nrow; i++) {
                AO2MOnr_tri_e2_o0(vout+i*nij, vin+i*npair, mo_coeff, nao,
                                  i_start, i_count, j_start, j_count);
        }
}

/*
 * ************************************************
 * i_count > j_count is more efficient
 */
void AO2MOnr_e1_o1(double *eri, double *mo_coeff,
                   int i_start, int i_count, int j_start, int j_count,
                   const int *atm, const int natm,
                   const int *bas, const int nbas, const double *env)
{
        AO2MOnr_e1_drv(eri, mo_coeff, &transform_kl, &trans_nr_tri_e1_o1,
                       0, nbas, i_start, i_count, j_start, j_count,
                       atm, natm, bas, nbas, env);
}

void AO2MOnr_e2_o1(const int nrow, double *vout, double *vin,
                   double *mo_coeff, const int nao,
                   int i_start, int i_count, int j_start, int j_count)
{
        AO2MOnr_e2_drv(nrow, vout, vin, mo_coeff, nao, &AO2MOnr_tri_e2_o1,
                       i_start, i_count, j_start, j_count);
}


/*
 * ************************************************
 * i_count < j_count is more efficient
 */
void AO2MOnr_e1_o2(double *eri, double *mo_coeff,
                   int i_start, int i_count, int j_start, int j_count,
                   const int *atm, const int natm,
                   const int *bas, const int nbas, const double *env)
{
        AO2MOnr_e1_drv(eri, mo_coeff, &transform_kl, &trans_nr_tri_e1_o2,
                       0, nbas, i_start, i_count, j_start, j_count,
                       atm, natm, bas, nbas, env);
}

void AO2MOnr_e2_o2(const int nrow, double *vout, double *vin,
                   double *mo_coeff, const int nao,
                   int i_start, int i_count, int j_start, int j_count)

{
        AO2MOnr_e2_drv(nrow, vout, vin, mo_coeff, nao, &AO2MOnr_tri_e2_o2,
                       i_start, i_count, j_start, j_count);
}


/*
 * ************************************************
 */
void AO2MOnr_e1_drv(double *eri, double *mo_coeff,
                    void (*ftransform_kl)(), void (*ftrans_e1)(),
                    int ish_start, int ish_count,
                    int k_start, int k_count, int l_start, int l_count,
                    const int *atm, const int natm,
                    const int *bas, const int nbas, const double *env)
{
        assert(l_start <= k_start);
        assert(l_start+l_count <= k_start+k_count);

        const int nao = CINTtot_cgto_spheric(bas, nbas);
        int *idx_tri = malloc(sizeof(int)*nao*nao);
        int *ao_loc = malloc(sizeof(int)*nbas);
        CINTshells_spheric_offset(ao_loc, bas, nbas);
        CVHFindex_blocks2tri(idx_tri, ao_loc, bas, nbas);

        struct _AO2MOEnvs envs = {natm, nbas, atm, bas, env, nao,
                                  ish_start, ish_count,
                                  k_start, k_count, l_start, l_count,
                                  ao_loc, idx_tri};
        CINTOpt *opt;
        cint2e_optimizer(&opt, atm, natm, bas, nbas, env);
        CVHFOpt *vhfopt;
        CVHFnr_optimizer(&vhfopt, atm, natm, bas, nbas, env);
        vhfopt->fprescreen = &CVHFnr_schwarz_cond;

        int i, j, ij;
        const int ijsh_start = ish_start*(ish_start+1)/2;
        const int ijsh_end = (ish_start+ish_count)*(ish_start+ish_count+1)/2;
#pragma omp parallel default(none) \
        shared(eri, mo_coeff, envs, opt, vhfopt, \
               ftransform_kl, ftrans_e1) \
        private(ij, i, j)
#pragma omp for nowait schedule(dynamic)
        for (ij = ijsh_start; ij < ijsh_end; ij++) {
                // ij = i*(i+1)/2+j, ij => (i,j)
                i = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                j = ij - i*(i+1)/2;
                (*ftransform_kl)(eri, mo_coeff, i, j, &envs, opt, vhfopt,
                                 ftrans_e1);
        }

        free(idx_tri);
        free(ao_loc);
        CVHFdel_optimizer(&vhfopt);
        CINTdel_optimizer(&opt);
}

void AO2MOnr_e2_drv(const int nrow, double *vout, double *vin,
                    double *mo_coeff, const int nao,
                    void (*const ftrans_e2)(),
                    int i_start, int i_count, int j_start, int j_count)
{
        assert(j_start <= i_start);
        assert(j_start+j_count <= i_start+i_count);
        const unsigned long npair = (unsigned long)nao*(nao+1)/2;

        int i;
        const unsigned long nij = AO2MOcount_ij(i_start, i_count, j_start, j_count);
#pragma omp parallel default(none) \
        shared(vout, vin, mo_coeff, i_start, i_count, j_start, j_count) \
        private(i)
#pragma omp for nowait schedule(static)
        for (i = 0; i < nrow; i++) {
                (*ftrans_e2)(vout+i*nij, vin+i*npair, mo_coeff, nao,
                             i_start, i_count, j_start, j_count);
        }
}
