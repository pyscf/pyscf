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
#include "ao2mo/nr_ao2mo.h"

int AO2MOmmm_nr_s2_igtj(double *vout, double *eri, struct _AO2MOEnvs *envs,
                        int seekdim);

void CVHFics8_tridm_vj(const double *eri, const double *tri_dm, double *vj,
                       int n, int i, int j);
void CVHFics8_jk_s2il(const double *eri, const double *dm, double *vk,
                      int n, int i, int j);

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
void MCSCFnrs4_aapp_jk(double *eri, double *xpa, double *mo_coeff,
                       double *dm, double *vj, double *vk,
                       int klsh_start, int klsh_count,
                       int i_start, int i_count, int j_start, int j_count,
                       int *ao_loc, int nbas)
{
        const int nao = ao_loc[nbas];
        const size_t nao2 = nao * nao;
        const size_t nao_pair = nao*(nao+1)/2;
        int n, k, l, kl, ksh, lsh;
        double *tridm = malloc(sizeof(double) * nao_pair);
        int *klocs = malloc(sizeof(int) * nao2);
        int *llocs = malloc(sizeof(int) * nao2);
        double *peri, *jpriv, *kpriv, *buf;

        const int ij_pair = i_count * j_count;
        struct _AO2MOEnvs envs;
        envs.bra_start = i_start;
        envs.bra_count = i_count;
        envs.ket_start = j_start;
        envs.ket_count = j_count;
        envs.nao = nao;
        envs.nbas = nbas;
        envs.ao_loc = ao_loc;
        envs.mo_coeff = mo_coeff;

        n = 0;
        for (kl = 0; kl < klsh_count; kl++) {
                // kl = k * (k+1) / 2 + l
                ksh = (int)(sqrt(2*(klsh_start+kl)+.25) - .5 + 1e-7);
                lsh = klsh_start+kl - ksh * (ksh+1) / 2;

                if (ksh != lsh) {
                        for (k = ao_loc[ksh]; k < ao_loc[ksh+1]; k++) {
                        for (l = ao_loc[lsh]; l < ao_loc[lsh+1]; l++, n++) {
                                klocs[n] = k;
                                llocs[n] = l;
                        } }
                } else {
                        for (k = ao_loc[ksh]; k < ao_loc[ksh+1]; k++) {
                        for (l = ao_loc[lsh]; l <= k; l++, n++) {
                                klocs[n] = k;
                                llocs[n] = l;
                        } }
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
        shared(eri, xpa, dm, tridm, vj, vk, klocs, llocs, n, envs) \
        private(k, l, kl, ksh, lsh, jpriv, kpriv, peri, buf)
        {
                buf = malloc(sizeof(double) * nao2);
                jpriv = malloc(sizeof(double) * nao2);
                kpriv = malloc(sizeof(double) * nao2);
                memset(jpriv, 0, sizeof(double) * nao2);
                memset(kpriv, 0, sizeof(double) * nao2);
#pragma omp for nowait schedule(dynamic)
                for (kl = 0; kl < n; kl++) {
                        k = klocs[kl];
                        l = llocs[kl];
                        peri = eri + kl * nao_pair;
                        CVHFics8_jk_s2il(peri, dm, kpriv, nao, k, l);
                        CVHFics8_tridm_vj(peri, tridm, jpriv, nao, k, l);
                        NPdunpack_tril(nao, peri, buf, 0);
                        AO2MOmmm_nr_s2_igtj(xpa+kl*ij_pair, buf, &envs, 0);
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
        free(klocs);
        free(llocs);
}

