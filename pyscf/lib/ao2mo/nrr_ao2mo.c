/* Copyright 2014-2022 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Xubo Wang <wangxubo0201@outlook.com>
 *         Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <assert.h>

//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "np_helper/np_helper.h"
#include "vhf/cvhf.h"
#include "vhf/fblas.h"
#include "vhf/nr_direct.h"
#include "r_ao2mo.h"

#define MIN(X,Y)        ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y)        ((X) > (Y) ? (X) : (Y))
#define NCTRMAX         128

int AO2MOmmm_nrr_iltj(double complex *vout, double *eri,
                    struct _AO2MOEnvs *envs, int seekdim)
{
        switch (seekdim) {
                case 1: return envs->bra_count * envs->ket_count;
                case 2: return envs->nao * envs->nao;
        }
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        int n2c = envs->nao;
        int i_start = envs->bra_start;
        int i_count = envs->bra_count;
        int j_start = envs->ket_start;
        int j_count = envs->ket_count;
        int i;
        double *buf1 = malloc(sizeof(double)*n2c*i_count*3);
        double *buf2 = buf1 + n2c*i_count;
        double *buf3 = buf2 + n2c*i_count;
        double *bufr, *bufi;
        double *mo1 = malloc(sizeof(double) * n2c*MAX(i_count,j_count)*2);
        double *mo2, *mo_r, *mo_i;
        double *eri_r = malloc(sizeof(double) * n2c*n2c*3);
        double *eri_i = eri_r + n2c*n2c;
        double *eri1  = eri_i + n2c*n2c;
        double *vout1, *vout2, *vout3;

        // Gauss complex multiplication, C_pi^* (pq| = (iq|, where (pq| is in C-order
        mo_r = envs->mo_r + i_start * n2c;
        mo_i = envs->mo_i + i_start * n2c;
        mo2 = mo1 + n2c*i_count;
        for (i = 0; i < n2c*i_count; i++) {
                mo1[i] = mo_r[i] - mo_i[i];
                mo2[i] =-mo_i[i] - mo_r[i];
        }
        for (i = 0; i < n2c*n2c; i++) {
                eri_r[i] = eri[i];
                eri_i[i] = 0.0;
                eri1 [i] = eri_r[i] + eri_i[i];
        }
        dgemm_(&TRANS_N, &TRANS_N, &n2c, &i_count, &n2c,
               &D1, eri1, &n2c, mo_r, &n2c, &D0, buf1, &n2c);
        dgemm_(&TRANS_N, &TRANS_N, &n2c, &i_count, &n2c,
               &D1, eri_r, &n2c, mo2, &n2c, &D0, buf2, &n2c);
        dgemm_(&TRANS_N, &TRANS_N, &n2c, &i_count, &n2c,
               &D1, eri_i, &n2c, mo1, &n2c, &D0, buf3, &n2c);
        free(eri_r);

        // C_qj^* (iq| = (ij|
        bufr = buf3;
        bufi = buf2;
        for (i = 0; i < n2c*i_count; i++) {
                buf3[i] = buf1[i] - buf3[i];
                buf2[i] = buf1[i] + buf2[i];
        }
        for (i = 0; i < n2c*i_count; i++) {
                buf1[i] = bufr[i] + bufi[i];
        }
        mo_r = envs->mo_r + j_start * n2c;
        mo_i = envs->mo_i + j_start * n2c;
        mo2 = mo1 + n2c*j_count;
        for (i = 0; i < n2c*j_count; i++) {
                mo1[i] = mo_r[i] + mo_i[i];
                mo2[i] = mo_i[i] - mo_r[i];
        }
        vout1 = malloc(sizeof(double)*i_count*j_count*3);
        vout2 = vout1 + i_count * j_count;
        vout3 = vout2 + i_count * j_count;
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &n2c,
               &D1, mo_r, &n2c, buf1, &n2c, &D0, vout1, &j_count);
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &n2c,
               &D1, mo2, &n2c, bufr, &n2c, &D0, vout2, &j_count);
        dgemm_(&TRANS_T, &TRANS_N, &j_count, &i_count, &n2c,
               &D1, mo1, &n2c, bufi, &n2c, &D0, vout3, &j_count);
        for (i = 0; i < i_count*j_count; i++) {
                vout[i] = (vout1[i]-vout3[i]) + (vout1[i]+vout2[i])*_Complex_I;
        }
        free(vout1);
        free(buf1);
        free(mo1);
        return 0;
}
int AO2MOmmm_nrr_s1_iltj(double complex *vout, double *eri,
                       struct _AO2MOEnvs *envs, int seekdim)
{
        return AO2MOmmm_nrr_iltj(vout, eri, envs, seekdim);
}
void AO2MOfill_nrr_s1(int (*intor)(), int (*fprescreen)(),
                    double *eri, int nkl, int ish,
                    struct _AO2MOEnvs *envs)
{
        const int nao = envs->nao;
        const size_t nao2 = nao * nao;
        const int *ao_loc = envs->ao_loc;
        const int klsh_start = envs->klsh_start;
        const int klsh_end = klsh_start + envs->klsh_count;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int jshtot = envs->nbas;
        int kl, jsh, ksh, lsh, dj, dk, dl;
        int icomp, i, j, k, l, n;
        int shls[4];
        double *buf = malloc(sizeof(double) *di*nao*NCTRMAX*NCTRMAX*envs->ncomp);
        assert(buf);
        double *pbuf, *pbuf1, *peri;

        shls[0] = ish;

        for (kl = klsh_start; kl < klsh_end; kl++) {
                ksh = kl / envs->nbas;
                lsh = kl - ksh * envs->nbas;
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                dl = ao_loc[lsh+1] - ao_loc[lsh];
                shls[2] = ksh;
                shls[3] = lsh;

                pbuf = buf;
                for (jsh = 0; jsh < jshtot; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        shls[1] = jsh;
                        n = di * dj * dk * dl * envs->ncomp;
                        if ((*fprescreen)(shls, envs->vhfopt,
                                          envs->atm, envs->bas, envs->env)) {
                                (*intor)(pbuf, NULL, shls, envs->atm, envs->natm,
                                         envs->bas, envs->nbas, envs->env,
                                         envs->cintopt, NULL);
                        } else {
                                NPdset0(pbuf, n);
                        }
                        pbuf += n;
                }

                pbuf = buf;
                for (jsh = 0; jsh < jshtot; jsh++) {
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        for (icomp = 0; icomp < envs->ncomp; icomp++) {
                                peri = eri + nao2 * nkl * icomp
                                     + ao_loc[ish] * nao + ao_loc[jsh];
                                for (k = 0; k < dk; k++) {
                                for (l = 0; l < dl; l++) {
                                        pbuf1 = pbuf + di * dj * (l*dk+k);
                                        for (i = 0; i < di; i++) {
                                        for (j = 0; j < dj; j++) {
                                                peri[i*nao+j] = pbuf1[j*di+i];
                                        } }
                                        peri += nao2;
                                } }
                                pbuf += di * dj * dk * dl;
                        }
                }
                eri += nao2 * dk * dl;
        }
        free(buf);
}
void AO2MOtranse1_nrr_s1(int (*fmmm)(),
                       double complex *vout, double *vin, int row_id,
                       struct _AO2MOEnvs *envs)
{
        size_t ij_pair = (*fmmm)(NULL, NULL, envs, 1);
        size_t nao2 = envs->nao * envs->nao;
        (*fmmm)(vout+ij_pair*row_id, vin+nao2*row_id, envs, 0);
}
void AO2MOnrr_e1_drv(int (*intor)(), void (*fill)(),
                   void (*ftrans)(), int (*fmmm)(),
                   double complex *eri, double complex *mo_a,
                   double complex *mo_b,
                   int klsh_start, int klsh_count, int nkl, int ncomp,
                   int *orbs_slice, int *tao, int *ao_loc,
                   CINTOpt *cintopt, CVHFOpt *vhfopt,
                   int *atm, int natm, int *bas, int nbas, double *env)
{
        const int i_start = orbs_slice[0];
        const int i_count = orbs_slice[1] - orbs_slice[0];
        const int j_start = orbs_slice[2];
        const int j_count = orbs_slice[3] - orbs_slice[2];
        int ij_count = i_count*j_count;
        int nao = ao_loc[nbas];
        int nmo = MAX(orbs_slice[1], orbs_slice[3]);
        int i;
        double *mo_ra = malloc(sizeof(double) * nao * nmo);
        double *mo_ia = malloc(sizeof(double) * nao * nmo);
        double *mo_rb = malloc(sizeof(double) * nao * nmo);
        double *mo_ib = malloc(sizeof(double) * nao * nmo);
        for (i = 0; i < nao*nmo; i++) {
                mo_ra[i] = creal(mo_a[i]);
                mo_ia[i] = cimag(mo_a[i]);
                mo_rb[i] = creal(mo_b[i]);
                mo_ib[i] = cimag(mo_b[i]);
        }
        struct _AO2MOEnvs envs = {natm, nbas, atm, bas, env, nao,
                                  klsh_start, klsh_count,
                                  i_start, i_count, j_start, j_count,
                                  ncomp, tao, ao_loc, mo_a,
                                  mo_ra, mo_ia, cintopt, vhfopt};
        struct _AO2MOEnvs envs2 = {natm, nbas, atm, bas, env, nao,
                                  klsh_start, klsh_count,
                                  i_start, i_count, j_start, j_count,
                                  ncomp, tao, ao_loc, mo_b,
                                  mo_rb, mo_ib, cintopt, vhfopt};


        double *eri_ao = malloc(sizeof(double)* nao*nao*nkl*ncomp);
        if (eri_ao == NULL) {
                fprintf(stderr, "malloc(%zu) failed in AO2MOnrr_e1_drv\n",
                        sizeof(double) * nao*nao*nkl*ncomp);
                exit(1);
        }
        int ish, kl;
        int (*fprescreen)();
        if (vhfopt) {
                fprescreen = vhfopt->fprescreen;
        } else {
                fprescreen = CVHFnoscreen;
        }

#pragma omp parallel default(none) \
        shared(fill, fprescreen, eri_ao, envs, intor, nkl, nbas) \
        private(ish)
#pragma omp for nowait schedule(dynamic)
        for (ish = 0; ish < nbas; ish++) {
                (*fill)(intor, fprescreen, eri_ao, nkl, ish, &envs, 0);
        }

#pragma omp parallel default(none) \
        shared(ftrans, fmmm, eri, eri_ao, nkl, ncomp, ij_count, envs, envs2) \
        private(kl)
#pragma omp for nowait schedule(static)
        for (kl = 0; kl < nkl*ncomp; kl++) {
                (*ftrans)(fmmm, eri, eri_ao, kl, &envs);
                (*ftrans)(fmmm, eri+ncomp*nkl*ij_count, eri_ao, kl, &envs2);
        }

        free(eri_ao);
        free(mo_ra);
        free(mo_rb);
        free(mo_ia);
        free(mo_ib);
}
