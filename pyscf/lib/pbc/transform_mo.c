/* Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.

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
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include "config.h"
#include "cint.h"
#include "gto/gto.h"
#include "np_helper/np_helper.h"
#include "vhf/nr_direct.h"
#include "vhf/fblas.h"
#include "pbc/pbc.h"

// # Transform AO indices
//  k_phase = np.eye(nkpts, dtype=np.complex128)
//  k_phase[[[k],[k_conj]],[k,k_conj]] = [[1., 1j], [1., -1j]] * .5**.5
// C_gamma = np.einsum('Rk,kum,kh->uRhm', phase, C_k, k_phase)
// C_gamma.reshape(Nao*NR, Nk*Nmo)[ao_map]
void PBCmo_k2gamma(double *dmfR, double *dmfI,
                   double complex *mo, double complex *expRk,
                   int *sh_loc, int *ao_loc, int *cell0_ao_loc,
                   int bvk_ncells, int nbasp, int s_nao, int naop, int nmo,
                   int nkpts, int nimgs)
{
        int bvk_nbas = bvk_ncells * nbasp;
        size_t knmo = nkpts * nmo;

#pragma omp parallel
{
        int ip0, ip1, i0, di, i, j, k;
        int ish_bvk, ish, ish0, ish1, ishp, iL;
        double complex phase, v;
        double complex *pmo_k;
        double *pdmfR, *pdmfI;
#pragma omp for schedule(dynamic, 4)
        for (ish_bvk = 0; ish_bvk < bvk_nbas; ish_bvk++) {
                ish0 = sh_loc[ish_bvk];
                ish1 = sh_loc[ish_bvk+1];
                if (ish0 == ish1) {
                        continue;
                }

                iL = ish_bvk / nbasp;
                ishp = ish_bvk % nbasp; // shell Id in cell 0
                ip0 = cell0_ao_loc[ishp];
                ip1 = cell0_ao_loc[ishp+1];
                di = ip1 - ip0;
                for (ish = ish0; ish < ish1; ish++) {
                        i0 = ao_loc[ish];
                        for (i = 0; i < di; i++) {
                                pdmfR = dmfR + (i0+i) * knmo;
                                pdmfI = dmfI + (i0+i) * knmo;
                                pmo_k = mo + (ip0+i) * nmo;
                                for (k = 0; k < nkpts; k++) {
                                        phase = expRk[iL*nkpts+k];
                                        for (j = 0; j < nmo; j++) {
                                                v = phase * pmo_k[k*naop*nmo+j];
                                                pdmfR[k*nmo+j] = creal(v);
                                                pdmfI[k*nmo+j] = cimag(v);
                                        }
                                }
                        }
                }
        }
}
}

void PBC_kcontract_fake_gamma(double *vkR, double *vkI, double *moR, double *moI,
                              double *GpqR, double *GpqI, double *coulG,
                              int *ki_idx, int *kj_idx, int8_t *k_to_compute, int swap_2e,
                              int s_nao, int nao, int nmo, int ngrids, int nkpts)
{
        size_t nao2 = nao * nao;
        size_t size_vk = nkpts * nao2;
        double *vtmpR = calloc(sizeof(double), size_vk*2);
        double *vtmpI = vtmpR + size_vk;

        int naog = nao * ngrids;
        int nmog = nmo * ngrids;
        int knmo = nkpts * nmo;
        size_t Naog = naog;
        double *GpiRR = malloc(sizeof(double) * Naog * knmo * 4);
        double *GpiRI = GpiRR + naog * knmo;
        double *GpiIR = GpiRI + naog * knmo;
        double *GpiII = GpiIR + naog * knmo;

        char TRANS_T = 'T';
        char TRANS_N = 'N';
        double D1 = 1.;
        double D0 = 0.;
        double N1 = -1.;
//        dgemm_(&TRANS_N, &TRANS_T, &naog, &knmo, &s_nao, &D1, GpqR, &naog, moR, &knmo, &D0, GpiRR, &naog);
//        dgemm_(&TRANS_N, &TRANS_T, &naog, &knmo, &s_nao, &D1, GpqI, &naog, moR, &knmo, &D0, GpiRI, &naog);
//        dgemm_(&TRANS_N, &TRANS_T, &naog, &knmo, &s_nao, &D1, GpqR, &naog, moI, &knmo, &D0, GpiIR, &naog);
//        dgemm_(&TRANS_N, &TRANS_T, &naog, &knmo, &s_nao, &D1, GpqI, &naog, moI, &knmo, &D0, GpiII, &naog);
//
#pragma omp parallel
{
        size_t k0, k1;
        NPomp_split(&k0, &k1, knmo);
        int dk = k1 - k0;
        double *pRR = GpiRR + Naog * k0;
        double *pRI = GpiRI + Naog * k0;
        double *pIR = GpiIR + Naog * k0;
        double *pII = GpiII + Naog * k0;
        dgemm_(&TRANS_N, &TRANS_T, &naog, &dk, &s_nao, &D1, GpqR, &naog, moR+k0, &knmo, &D0, pRR, &naog);
        dgemm_(&TRANS_N, &TRANS_T, &naog, &dk, &s_nao, &D1, GpqI, &naog, moR+k0, &knmo, &D0, pRI, &naog);
        dgemm_(&TRANS_N, &TRANS_T, &naog, &dk, &s_nao, &D1, GpqR, &naog, moI+k0, &knmo, &D0, pIR, &naog);
        dgemm_(&TRANS_N, &TRANS_T, &naog, &dk, &s_nao, &D1, GpqI, &naog, moI+k0, &knmo, &D0, pII, &naog);

        double *bufR = malloc(sizeof(double) * Naog * nmo * 4);
        double *bufI = bufR + Naog * nmo;
        double *buf1R = bufI + Naog * nmo;
        double *buf1I = buf1R + Naog * nmo;
        double *outR, *outI, *out1R, *out1I;
        int k, i, j, ig, ki, kj;
        double cR, cI;

#pragma omp for schedule(dynamic)
        for (k = 0; k < nkpts; k++) {
                ki = ki_idx[k];
                kj = kj_idx[k];
                if (!(k_to_compute[ki] || (swap_2e && k_to_compute[kj]))) {
                        continue;
                }

                if (k_to_compute[ki]) {
                        // (GpiRR[kj] - GpiII[kj]).transpose(1,0,2)
                        // (GpiRI[kj] + GpiIR[kj]).transpose(1,0,2)
                        for (i = 0; i < nmo; i++) {
                                outR = bufR + i * ngrids;
                                outI = bufI + i * ngrids;
                                out1R = buf1R + i * ngrids;
                                out1I = buf1I + i * ngrids;
                                pRR = GpiRR + (kj * nmo + i) * Naog;
                                pRI = GpiRI + (kj * nmo + i) * Naog;
                                pIR = GpiIR + (kj * nmo + i) * Naog;
                                pII = GpiII + (kj * nmo + i) * Naog;
                                for (j = 0; j < nao; j++) {
#pragma GCC ivdep
                                for (ig = 0; ig < ngrids; ig++) {
                                        cR = pRR[j*ngrids+ig] - pII[j*ngrids+ig];
                                        cI = pRI[j*ngrids+ig] + pIR[j*ngrids+ig];
                                        outR[j*nmog+ig] = cR;
                                        outI[j*nmog+ig] = cI;
                                        out1R[j*nmog+ig] = cR * coulG[ig];
                                        out1I[j*nmog+ig] = cI * coulG[ig];
                                } }
                        }
// zdotNC(buf, buf1.T).T
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &D1, buf1R, &nmog, bufR, &nmog, &D1, vkR+ki*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &D1, buf1I, &nmog, bufI, &nmog, &D1, vkR+ki*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &D1, buf1R, &nmog, bufI, &nmog, &D1, vkI+ki*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &N1, buf1I, &nmog, bufR, &nmog, &D1, vkI+ki*nao2, &nao);
                }

                if (swap_2e && k_to_compute[kj]) {
                        // (GpiRR[ki] + GpiII[ki]).transpose(1,0,2)
                        // (GpiRI[ki] - GpiIR[ki]).transpose(1,0,2)
                        for (i = 0; i < nmo; i++) {
                                outR = bufR + i * ngrids;
                                outI = bufI + i * ngrids;
                                out1R = buf1R + i * ngrids;
                                out1I = buf1I + i * ngrids;
                                pRR = GpiRR + (ki * nmo + i) * Naog;
                                pRI = GpiRI + (ki * nmo + i) * Naog;
                                pIR = GpiIR + (ki * nmo + i) * Naog;
                                pII = GpiII + (ki * nmo + i) * Naog;
                                for (j = 0; j < nao; j++) {
#pragma GCC ivdep
                                for (ig = 0; ig < ngrids; ig++) {
                                        cR = pRR[j*ngrids+ig] + pII[j*ngrids+ig];
                                        cI = pRI[j*ngrids+ig] - pIR[j*ngrids+ig];
                                        outR[j*nmog+ig] = cR;
                                        outI[j*nmog+ig] = cI;
                                        out1R[j*nmog+ig] = cR * coulG[ig];
                                        out1I[j*nmog+ig] = cI * coulG[ig];
                                } }
                        }
// zdotCN(buf, buf1.T).T
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &D1, buf1R, &nmog, bufR, &nmog, &D1, vtmpR+kj*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &D1, buf1I, &nmog, bufI, &nmog, &D1, vtmpR+kj*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &N1, buf1R, &nmog, bufI, &nmog, &D1, vtmpI+kj*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &D1, buf1I, &nmog, bufR, &nmog, &D1, vtmpI+kj*nao2, &nao);
                }
        }
        free(bufR);

#pragma omp barrier
#pragma omp for schedule(static)
        for (i = 0; i < size_vk; i++) {
                vkR[i] += vtmpR[i];
                vkI[i] += vtmpI[i];
        }
}
        free(GpiRR);
        free(vtmpR);
}
