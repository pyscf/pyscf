/*
 *
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
//#include <omp.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"

void CVHFnrs8_tridm_vj(double *eri, double *tri_dm, double *vj,
                       int nao, int ic, int jc);
void CVHFnrs8_jk_s2il(double *eri, double *dm, double *vk,
                      int nao, int ic, int jc);

struct _AO2MOEnvs {
        int natm;
        int nbas;
        int *atm;
        int *bas;
        double *env;
        int nao;
        int klsh_start;
        int klsh_count;
        int bra_start;
        int bra_count;
        int ket_start;
        int ket_count;
        int ncomp;
        int *ao_loc;
        double *mo_coeff;
        void *cintopt;
        void *vhfopt;
};

/*
 * transform ket, s2 to label AO symmetry
 * copy from RIhalfmmm_nr_s2_ket
 */
int MCSCFhalfmmm_nr_s2_ket(double *vout, double *vin, struct _AO2MOEnvs *envs,
                           int seekdim)
{
        switch (seekdim) {
                case 1: return envs->nao * envs->ket_count;
                case 2: return envs->nao * (envs->nao+1) / 2;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';
        int nao = envs->nao;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        double *mo_coeff = envs->mo_coeff;
        double *buf = malloc(sizeof(double)*nao*j_count);
        int i, j;

        dsymm_(&SIDE_L, &UPLO_U, &nao, &j_count,
               &D1, vin, &nao, mo_coeff+j_start*nao, &nao,
               &D0, buf, &nao);
        for (j = 0; j < nao; j++) {
                for (i = 0; i < j_count; i++) {
                        vout[i] = buf[i*nao+j];
                }
                vout += j_count;
        }
        free(buf);
        return 0;
}

/*
 * eri[dk,dl,nao,nao] the lower triangular nao x nao array is filled in
 * the input.
 */
void MCSCFnrs4_corejk(double *eri, double *dm, double *vj, double *vk,
                      int klsh_start, int klsh_count, int nbas, int *ao_loc)
{
        const int nao = ao_loc[nbas];
        const size_t nao2 = nao * nao;
        int k, l, kl, ksh, lsh, dk, dl;
        double *tridm = malloc(sizeof(double) * nao2);
        int *kshs = malloc(sizeof(int) * klsh_count);
        int *lshs = malloc(sizeof(int) * klsh_count);
        double **peris = malloc(sizeof(double*) * klsh_count);
        double *buf, *jpriv, *kpriv;
        double *peri;

        for (k = 0; k < klsh_count; k++) {
                // kl = k * (k+1) / 2 + l
                kl = klsh_start + k;
                ksh = (int)(sqrt(2*kl+.25) - .5 + 1e-7);
                lsh = kl - ksh * (ksh+1) / 2;
                kshs[k] = ksh;
                lshs[k] = lsh;
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                peris[k] = eri;
                if (ksh != lsh) {
                        eri += nao2 * dk * dl;
                } else {
                        eri += nao2 * dk*(dk+1)/2;
                }
        }
        for (k = 0, kl = 0; k < nao; k++) {
                for (l = 0; l < k; l++, kl++) {
                        tridm[kl] = dm[k*nao+l] + dm[l*nao+k];
                }
                tridm[kl] = dm[k*nao+l];
                kl++;
        }

#pragma omp parallel default(none) \
        shared(klsh_count, ao_loc, dm, tridm, vj, vk, kshs, lshs, peris) \
        private(k, l, kl, ksh, lsh, peri, buf, jpriv, kpriv)
        {
                buf = malloc(sizeof(double) * nao2);
                jpriv = malloc(sizeof(double) * nao2);
                kpriv = malloc(sizeof(double) * nao2);
                memset(jpriv, 0, sizeof(double) * nao2);
                memset(kpriv, 0, sizeof(double) * nao2);
#pragma omp for nowait schedule(dynamic)
                for (kl = 0; kl < klsh_count; kl++) {
                        ksh = kshs[kl];
                        lsh = lshs[kl];
                        peri = peris[kl];
                        if (ksh != lsh) {
                                for (k = ao_loc[ksh]; k < ao_loc[ksh+1]; k++) {
                                for (l = ao_loc[lsh]; l < ao_loc[lsh+1]; l++) {
                                        NPdpack_tril(nao, buf, peri);
                                        CVHFnrs8_tridm_vj(buf, tridm, jpriv, nao, k, l);
                                        CVHFnrs8_jk_s2il (buf, dm   , kpriv, nao, k, l);
                                        peri += nao2;
                                } }
                        } else {
                                for (k = ao_loc[ksh]; k < ao_loc[ksh+1]; k++) {
                                for (l = ao_loc[lsh]; l <= k; l++) {
                                        NPdpack_tril(nao, buf, peri);
                                        CVHFnrs8_tridm_vj(buf, tridm, jpriv, nao, k, l);
                                        CVHFnrs8_jk_s2il (buf, dm   , kpriv, nao, k, l);
                                        peri += nao2;
                                } }
                        }
                }
#pragma omp critical
                for (k = 0; k < nao2; k++) {
                        vj[k] += jpriv[k];
                        vk[k] += kpriv[k];
                }
                free(buf);
                free(jpriv);
                free(kpriv);
        }
        free(tridm);
        free(peris);
        free(kshs);
        free(lshs);
}
