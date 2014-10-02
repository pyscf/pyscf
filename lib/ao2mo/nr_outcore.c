/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 * for ao2mo.outcore
 *
 */

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <assert.h>
//#define NDEBUG

#include "cint.h"
#include "vhf/cvhf.h"
#include "vhf/fblas.h"
#include "nr_ao2mo_o3.h"

#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))

int CVHFfill_nr_eri_o2(double *eri, int ish, int jsh, int ksh_lim,
                       int *atm, int natm, int *bas, int nbas, double *env,
                       CINTOpt *opt, CVHFOpt *vhfopt);

/* without transposing e1,e2 index */
static void transform_kl_tt(double *meri, double *mo_coeff, int ish, int jsh,
                            struct _AO2MOEnvs *envs, CINTOpt *opt, CVHFOpt *vhfopt,
                            void (*ftrans_e1)())
{
        const int nao = envs->nao;
        const int di = CINTcgto_spheric(ish, envs->bas);
        const int dj = CINTcgto_spheric(jsh, envs->bas);
        double *eribuf = (double *)malloc(sizeof(double)*di*dj*nao*nao);

        if (CVHFfill_nr_eri_o2(eribuf, ish, jsh, envs->nbas,
                               envs->atm, envs->natm, envs->bas, envs->nbas,
                               envs->env, opt, vhfopt)) {
                (*ftrans_e1)(meri, eribuf, mo_coeff, ish, jsh, envs);
        } else {
                const int k_start = envs->bra_start;
                const int k_count = envs->bra_count;
                const int l_start = envs->ket_start;
                const int l_count = envs->ket_count;
                const unsigned long nkl = AO2MOcount_ij(k_start, k_count,
                                                        l_start, l_count);
                const int *ao_loc = envs->ao_loc;
                const int ish_start = envs->ish_start;
                const int offeri = ao_loc[ish_start]*(ao_loc[ish_start]+1)/2;
                int i, j, ij;
                for (i = ao_loc[ish]; i < ao_loc[ish]+di; i++) {
                for (j = ao_loc[jsh]; j < MIN(ao_loc[jsh]+dj,i+1); j++) {
                        ij = i*(i+1)/2+j - offeri;
                        memset(meri+ij*nkl, 0, sizeof(double)*nkl);
                } }
        }

        free(eribuf);
}

void AO2MOnr_e1range_o0(double *vout, double *eri, double *mo_coeff,
                        int ish, int jsh, struct _AO2MOEnvs *envs)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        const int nao = envs->nao;
        const int k_start = envs->bra_start;
        const int k_count = envs->bra_count;
        const int l_start = envs->ket_start;
        const int l_count = envs->ket_count;
        const unsigned long nkl = AO2MOcount_ij(k_start, k_count,
                                                l_start, l_count);

        const int di = CINTcgto_spheric(ish, envs->bas);
        const int dj = CINTcgto_spheric(jsh, envs->bas);
        double *vc = malloc(sizeof(double) * nao*l_count);
        double *cvc = malloc(sizeof(double) * nao*nao);
        double *pvout;
        const int *idx_tri = envs->idx_tri;
        const int *ao_loc = envs->ao_loc;
        const int ish_start = envs->ish_start;
        const int offeri = ao_loc[ish_start]*(ao_loc[ish_start]+1)/2;
        int i, j, k, l, i0, j0, l1, ij, kl;

        for (j0 = ao_loc[jsh], j = 0; j < dj; j++, j0++) {
        for (i0 = ao_loc[ish], i = 0; i < di; i++, i0++) {
        if (i0 >= j0) {
                ij = j * di + i;
                for (k = 0, kl = 0; k < nao; k++) {
                for (l = 0; l <= k; l++, kl++) {
                        cvc[k*nao+l] = eri[idx_tri[kl]*di*dj+ij];
                } }
                dsymm_(&SIDE_L, &UPLO_U, &nao, &l_count,
                       &D1, cvc, &nao, mo_coeff+l_start*nao, &nao,
                       &D0, vc, &nao);
                dgemm_(&TRANS_T, &TRANS_N, &k_count, &l_count, &nao,
                       &D1, mo_coeff+k_start*nao, &nao, vc, &nao,
                       &D0, cvc, &k_count);

                pvout = vout + (i0*(i0+1)/2+j0-offeri) * nkl;
                for (k = 0, kl = 0; k < k_count; k++) {
                        // l < l_count and l+l_start <= k+k_start
                        l1 = k+k_start-l_start+1;
                        for (l = 0; l < MIN(l_count,l1); l++, kl++) {
                                pvout[kl] = cvc[l*k_count+k];
                        }
                }
        } } }

        free(vc);
        free(cvc);
}

void AO2MOnr_e1range_o1(double *vout, double *eri, double *mo_coeff,
                        int ish, int jsh, struct _AO2MOEnvs *envs)
{
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        const int nao = envs->nao;
        const int k_start = envs->bra_start;
        const int k_count = envs->bra_count;
        const int l_start = envs->ket_start;
        const int l_count = envs->ket_count;
        const unsigned long nkl = AO2MOcount_ij(k_start, k_count,
                                                l_start, l_count);

        const int di = CINTcgto_spheric(ish, envs->bas);
        const int dj = CINTcgto_spheric(jsh, envs->bas);
        double *vc = malloc(sizeof(double) * nao*l_count);
        double *cvc = malloc(sizeof(double) * nao*nao);
        double *pvout, *pcvc;
        const int *idx_tri = envs->idx_tri;
        const int *ao_loc = envs->ao_loc;
        const int ish_start = envs->ish_start;
        const int offeri = ao_loc[ish_start]*(ao_loc[ish_start]+1)/2;
        int i, j, k, l, i0, j0, l1, ij, kl;

        for (j0 = ao_loc[jsh], j = 0; j < dj; j++, j0++) {
        for (i0 = ao_loc[ish], i = 0; i < di; i++, i0++) {
        if (i0 >= j0) {
                ij = j * di + i;
                for (k = 0, kl = 0; k < nao; k++) {
                for (l = 0; l <= k; l++, kl++) {
                        cvc[k*nao+l] = eri[idx_tri[kl]*di*dj+ij];
                } }
                dsymm_(&SIDE_L, &UPLO_U, &nao, &l_count,
                       &D1, cvc, &nao, mo_coeff+l_start*nao, &nao,
                       &D0, vc, &nao);
                AO2MOdtriumm_o2(l_count, k_count, nao, k_start,
                                vc, mo_coeff+k_start*nao, cvc);

                pvout = vout + (i0*(i0+1)/2+j0-offeri) * nkl;
                for (k = 0, kl = 0; k < k_count; k++) {
                        // l < l_count and l+l_start <= k+k_start
                        l1 = k+k_start-l_start+1;
                        pcvc = cvc + k * l_count;
                        for (l = 0; l < MIN(l_count,l1); l++, kl++) {
                                pvout[kl] = pcvc[l];
                        }
                }
        } } }

        free(vc);
        free(cvc);
}

void AO2MOnr_e1range_o2(double *vout, double *eri, double *mo_coeff,
                        int ish, int jsh, struct _AO2MOEnvs *envs)
{
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        const int nao = envs->nao;
        const int k_start = envs->bra_start;
        const int k_count = envs->bra_count;
        const int l_start = envs->ket_start;
        const int l_count = envs->ket_count;
        const unsigned long nkl = AO2MOcount_ij(k_start, k_count,
                                                l_start, l_count);

        const int di = CINTcgto_spheric(ish, envs->bas);
        const int dj = CINTcgto_spheric(jsh, envs->bas);
        double *vc = malloc(sizeof(double) * nao*l_count);
        double *cvc = malloc(sizeof(double) * nao*nao);
        double *pvout, *pcvc;
        const int *idx_tri = envs->idx_tri;
        const int *ao_loc = envs->ao_loc;
        const int ish_start = envs->ish_start;
        const int offeri = ao_loc[ish_start]*(ao_loc[ish_start]+1)/2;
        int i, j, k, l, i0, j0, l1, ij, kl;

        for (j0 = ao_loc[jsh], j = 0; j < dj; j++, j0++) {
        for (i0 = ao_loc[ish], i = 0; i < di; i++, i0++) {
        if (i0 >= j0) {
                ij = j * di + i;
                for (k = 0, kl = 0; k < nao; k++) {
                for (l = 0; l <= k; l++, kl++) {
                        cvc[k*nao+l] = eri[idx_tri[kl]*di*dj+ij];
                } }
                dsymm_(&SIDE_L, &UPLO_U, &nao, &k_count,
                       &D1, cvc, &nao, mo_coeff+k_start*nao, &nao,
                       &D0, vc, &nao);
                AO2MOdtriumm_o1(l_count, k_count, nao, k_start,
                                mo_coeff+l_start*nao, vc, cvc);

                pvout = vout + (i0*(i0+1)/2+j0-offeri) * nkl;
                for (k = 0, kl = 0; k < k_count; k++) {
                        // l < l_count and l+l_start <= k+k_start
                        l1 = k+k_start-l_start+1;
                        pcvc = cvc + k * l_count;
                        for (l = 0; l < MIN(l_count,l1); l++, kl++) {
                                pvout[kl] = pcvc[l];
                        }
                }
        } } }

        free(vc);
        free(cvc);
}

/*
 * ************************************************
 * for a given range of i, transform [kl] of integrals (i>j|k>l)
 * ish_start, is shell_id
 * k_start, l_start are row_id (not shell_id)
 */
void AO2MOnr_e1outcore_drv(double *eri, double *mo_coeff, void (*ftrans_e1)(),
                           int ish_start, int ish_end,
                           int i_start, int i_count, int j_start, int j_count,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        AO2MOnr_e1_drv(eri, mo_coeff, &transform_kl_tt, ftrans_e1,
                       ish_start, ish_end-ish_start,
                       i_start, i_count, j_start, j_count,
                       atm, natm, bas, nbas, env);
}

