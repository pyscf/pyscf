/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.

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
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <complex.h>
#include "config.h"
#include "gto/grid_ao_drv.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"
#include <assert.h>

#define BOXSIZE         56

int VXCao_empty_blocks(int8_t *empty, uint8_t *non0table, int *shls_slice,
                       int *ao_loc);

static void dot_ao_dm(double complex *vm, double complex *ao, double complex *dm,
                      int nao, int nocc, int ngrids, int bgrids,
                      uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        int nbox = (nao+BOXSIZE-1) / BOXSIZE;
        int8_t empty[nbox];
        int has0 = VXCao_empty_blocks(empty, non0table, shls_slice, ao_loc);

        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double complex Z1 = 1;
        double complex beta = 0;

        if (has0) {
                int box_id, blen, i, j;
                size_t b0;
                for (box_id = 0; box_id < nbox; box_id++) {
                        if (!empty[box_id]) {
                                b0 = box_id * BOXSIZE;
                                blen = MIN(nao-b0, BOXSIZE);
                                zgemm_(&TRANS_N, &TRANS_T, &bgrids, &nocc, &blen,
                                       &Z1, ao+b0*ngrids, &ngrids, dm+b0*nocc, &nocc,
                                       &beta, vm, &ngrids);
                                beta = 1.0;
                        }
                }
                if (beta == 0) { // all empty
                        for (i = 0; i < nocc; i++) {
                                for (j = 0; j < bgrids; j++) {
                                        vm[i*ngrids+j] = 0;
                                }
                        }
                }
        } else {
                zgemm_(&TRANS_N, &TRANS_T, &bgrids, &nocc, &nao,
                       &Z1, ao, &ngrids, dm, &nocc, &beta, vm, &ngrids);
        }
}


/* vm[nocc,ngrids] = ao[i,ngrids] * dm[i,nocc] */
void VXCzdot_ao_dm(double complex *vm, double complex *ao, double complex *dm,
                   int nao, int nocc, int ngrids, int nbas,
                   uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;

#pragma omp parallel
{
        int ip, ib;
#pragma omp for nowait schedule(static)
        for (ib = 0; ib < nblk; ib++) {
                ip = ib * BLKSIZE;
                dot_ao_dm(vm+ip, ao+ip, dm,
                          nao, nocc, ngrids, MIN(ngrids-ip, BLKSIZE),
                          non0table+ib*nbas, shls_slice, ao_loc);
        }
}
}



/* conj(vv[n,m]) = ao1[n,ngrids] * conj(ao2[m,ngrids]) */
static void dot_ao_ao(double complex *vv, double complex *ao1, double complex *ao2,
                      int nao, int ngrids, int bgrids, int hermi,
                      uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        int nbox = (nao+BOXSIZE-1) / BOXSIZE;
        int8_t empty[nbox];
        int has0 = VXCao_empty_blocks(empty, non0table, shls_slice, ao_loc);

        const char TRANS_C = 'C';
        const char TRANS_N = 'N';
        const double complex Z1 = 1;
        if (has0) {
                int ib, jb, leni, lenj;
                int j1 = nbox;
                size_t b0i, b0j;

                for (ib = 0; ib < nbox; ib++) {
                if (!empty[ib]) {
                        b0i = ib * BOXSIZE;
                        leni = MIN(nao-b0i, BOXSIZE);
                        if (hermi) {
                                j1 = ib + 1;
                        }
                        for (jb = 0; jb < j1; jb++) {
                        if (!empty[jb]) {
                                b0j = jb * BOXSIZE;
                                lenj = MIN(nao-b0j, BOXSIZE);
                                zgemm_(&TRANS_C, &TRANS_N, &lenj, &leni, &bgrids, &Z1,
                                       ao2+b0j*ngrids, &ngrids, ao1+b0i*ngrids, &ngrids,
                                       &Z1, vv+b0i*nao+b0j, &nao);
                        } }
                } }
        } else {
                zgemm_(&TRANS_C, &TRANS_N, &nao, &nao, &bgrids,
                       &Z1, ao2, &ngrids, ao1, &ngrids, &Z1, vv, &nao);
        }
}


/* vv[nao,nao] = conj(ao1[i,nao]) * ao2[i,nao] */
void VXCzdot_ao_ao(double complex *vv, double complex *ao1, double complex *ao2,
                   int nao, int ngrids, int nbas, int hermi,
                   uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
        size_t Nao = nao;
        NPzset0(vv, Nao * Nao);

#pragma omp parallel
{
        int ip, ib;
        double complex *v_priv = calloc(nao*nao+2, sizeof(double complex));
#pragma omp for nowait schedule(static)
        for (ib = 0; ib < nblk; ib++) {
                ip = ib * BLKSIZE;
                dot_ao_ao(v_priv, ao1+ip, ao2+ip,
                          nao, ngrids, MIN(ngrids-ip, BLKSIZE), hermi,
                          non0table+ib*nbas, shls_slice, ao_loc);
        }
#pragma omp critical
        {
                for (ip = 0; ip < nao*nao; ip++) {
                        vv[ip] += conj(v_priv[ip]);
                }
        }
        free(v_priv);
}
        if (hermi != 0) {
                NPzhermi_triu(nao, vv, hermi);
        }
}

void VXC_zscale_ao(double complex *aow, double complex *ao, double *wv,
                    int comp, int nao, int ngrids)
{
#pragma omp parallel
{
        size_t Ngrids = ngrids;
        size_t ao_size = nao * Ngrids;
        int i, j, ic;
        double complex *pao = ao;
#pragma omp for schedule(static)
        for (i = 0; i < nao; i++) {
                pao = ao + i * Ngrids;
                for (j = 0; j < Ngrids; j++) {
                        aow[i*Ngrids+j] = pao[j] * wv[j];
                }
                for (ic = 1; ic < comp; ic++) {
                for (j = 0; j < Ngrids; j++) {
                        aow[i*Ngrids+j] += pao[ic*ao_size+j] * wv[ic*Ngrids+j];
                } }
        }
}
}

void VXC_dzscale_ao(double complex *aow, double *ao, double complex *wv,
                    int comp, int nao, int ngrids)
{
#pragma omp parallel
{
        size_t Ngrids = ngrids;
        size_t ao_size = nao * Ngrids;
        int i, j, ic;
        double *pao = ao;
#pragma omp for schedule(static)
        for (i = 0; i < nao; i++) {
                pao = ao + i * Ngrids;
                for (j = 0; j < Ngrids; j++) {
                        aow[i*Ngrids+j] = pao[j] * wv[j];
                }
                for (ic = 1; ic < comp; ic++) {
                for (j = 0; j < Ngrids; j++) {
                        aow[i*Ngrids+j] += pao[ic*ao_size+j] * wv[ic*Ngrids+j];
                } }
        }
}
}

void VXC_zzscale_ao(double complex *aow, double complex *ao, double complex *wv,
                    int comp, int nao, int ngrids)
{
#pragma omp parallel
{
        size_t Ngrids = ngrids;
        size_t ao_size = nao * Ngrids;
        int i, j, ic;
        double complex *pao = ao;
#pragma omp for schedule(static)
        for (i = 0; i < nao; i++) {
                pao = ao + i * Ngrids;
                for (j = 0; j < Ngrids; j++) {
                        aow[i*Ngrids+j] = pao[j] * wv[j];
                }
                for (ic = 1; ic < comp; ic++) {
                for (j = 0; j < Ngrids; j++) {
                        aow[i*Ngrids+j] += pao[ic*ao_size+j] * wv[ic*Ngrids+j];
                } }
        }
}
}

// 'ip,ip->p'
void VXC_zcontract_rho(double *rho, double complex *bra, double complex *ket,
                       int nao, int ngrids)
{
#pragma omp parallel
{
        size_t Ngrids = ngrids;
        int nthread = omp_get_num_threads();
        int blksize = MAX((Ngrids+nthread-1) / nthread, 1);
        int ib, b0, b1, i, j;
#pragma omp for
        for (ib = 0; ib < nthread; ib++) {
                b0 = ib * blksize;
                b1 = MIN(b0 + blksize, ngrids);
                for (j = b0; j < b1; j++) {
                        rho[j] = creal(bra[j]) * creal(ket[j])
                               + cimag(bra[j]) * cimag(ket[j]);
                }
                for (i = 1; i < nao; i++) {
                for (j = b0; j < b1; j++) {
                        rho[j] += creal(bra[i*Ngrids+j]) * creal(ket[i*Ngrids+j])
                                + cimag(bra[i*Ngrids+j]) * cimag(ket[i*Ngrids+j]);
                } }
        }
}
}
