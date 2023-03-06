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

#define SQRTHALF        0.707106781186547524

// # Transform AO indices
//  k_phase = np.eye(nkpts, dtype=np.complex128)
//  k_phase[[[k],[k_conj]],[k,k_conj]] = [[1., 1j], [1., -1j]] * .5**.5
// C_gamma = np.einsum('Rk,kum,kh->uRhm', phase, C_k, k_phase)
// C_gamma.reshape(Nao*NR, Nk*Nmo)[ao_map]
void PBCmo_k2gamma(double *dmfR, double *dmfI,
                   double complex *mo, double complex *expRk,
                   int *sh_loc, int *ao_loc, int *cell0_ao_loc,
                   int *k_conj_groups, int ngroups, int bvk_ncells, int nbasp,
                   int s_nao, int naop, int nmo, int nkpts, int nimgs)
{
        int bvk_nbas = bvk_ncells * nbasp;
        size_t knmo = nkpts * nmo;
        int *k_list = k_conj_groups;
        int *kc_list = k_conj_groups + ngroups;

#pragma omp parallel
{
        int ip0, ip1, i0, di, i, j, n, k, kc;
        int ish_bvk, ish, ish0, ish1, ishp, iL;
        double complex phase, phasec, v1, v2, c1, c2;
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
for (n = 0; n < ngroups; n++) {
        k = k_list[n];
        kc = kc_list[n];
        if (k == kc) {
                phase = expRk[iL*nkpts+k];
                for (j = 0; j < nmo; j++) {
                        v1 = phase * pmo_k[k*naop*nmo+j];
                        pdmfR[k*nmo+j] = creal(v1);
                        pdmfI[k*nmo+j] = cimag(v1);
                }
        } else {
                // multiply phase
                //   [1  i]
                //   [1 -i] / sqrt(2)
                phase = expRk[iL*nkpts+k];
                phasec = expRk[iL*nkpts+kc];
                for (j = 0; j < nmo; j++) {
                        v1 = phase * pmo_k[k*naop*nmo+j];
                        v2 = phasec * pmo_k[kc*naop*nmo+j];
                        c1 = v1 + v2;
                        c2 = v1 - v2;
                        pdmfR[k*nmo+j] = creal(c1) * SQRTHALF;
                        pdmfI[k*nmo+j] = cimag(c1) * SQRTHALF;
                        pdmfR[kc*nmo+j] = cimag(c2) * -SQRTHALF;
                        pdmfI[kc*nmo+j] = creal(c2) * SQRTHALF;
                }
        }
}
                        }
                }
        }
}
}

static void hermi_assign(double *outR, double *outI, double *inR, double *inI, int n)
{
        int i, j;
        for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
                outR[j*n+i] = inR[i*n+j];
                outI[j*n+i] = -inI[i*n+j];
        } }
}

static void plain_add(double *outR, double *outI, double *inR, double *inI, int n)
{
        int i;
        for (i = 0; i < n*n; i++) {
                outR[i] += inR[i];
                outI[i] += inI[i];
        }
}

static void transpose_add(double *outR, double *outI, double *inR, double *inI, int n)
{
        int i, j;
        for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
                outR[j*n+i] += inR[i*n+j];
                outI[j*n+i] += inI[i*n+j];
        } }
}

// Vpq = Gpi dot pp dot (Gqi*)
// u = [1  i]
//     [1 -i] / sqrt(2)
// U = (u^-1)[:,0]
// pp = U U^dagger = [ .5  .5i]
//                   [-.5i .5 ]
static void pp_add(double *outR, double *outI,
                   double *i00R, double *i00I, double *i11R, double *i11I,
                   double *i01R, double *i01I, double *i10R, double *i10I, int n)
{
        int i;
        for (i = 0; i < n*n; i++) {
                outR[i] += (i00R[i] + i11R[i] + i10I[i] - i01I[i]) * .5;
                outI[i] += (i00I[i] + i11I[i] - i10R[i] + i01R[i]) * .5;
        }
}

// Vpq = Gpi dot cc dot (Gqi*)
// U = (u^-1)[:,1]
// cc = U U^\dagger = [.5  -.5i]
//                    [.5i  .5 ]
static void cc_add(double *outR, double *outI,
                   double *i00R, double *i00I, double *i11R, double *i11I,
                   double *i01R, double *i01I, double *i10R, double *i10I, int n)
{
        pp_add(outR, outI, i00R, i00I, i11R, i11I, i10R, i10I, i01R, i01I, n);
}

// Vqp = Gpi dot pp dot (Gqi*) = transpose(pp_add)
static void pp_tadd(double *outR, double *outI,
                   double *i00R, double *i00I, double *i11R, double *i11I,
                   double *i01R, double *i01I, double *i10R, double *i10I, int n)
{
        int i, j, ij;
        for (i = 0, ij = 0; i < n; i++) {
        for (j = 0; j < n; j++, ij++) {
                outR[j*n+i] += (i00R[ij] + i11R[ij] + i10I[ij] - i01I[ij]) * .5;
                outI[j*n+i] += (i00I[ij] + i11I[ij] - i10R[ij] + i01R[ij]) * .5;
        } }
}

// Vqp = Gpi dot cc dot (Gqi*) = transpose(cc_add)
static void cc_tadd(double *outR, double *outI,
                   double *i00R, double *i00I, double *i11R, double *i11I,
                   double *i01R, double *i01I, double *i10R, double *i10I, int n)
{
        pp_tadd(outR, outI, i00R, i00I, i11R, i11I, i10R, i10I, i01R, i01I, n);
}

void PBC_kcontract_fake_gamma(double *vkR, double *vkI, double *moR,
                              double *GpqR, double *GpqI, double *coulG,
                              int *ki_idx, int *kj_idx, int8_t *k_to_compute,
                              int *k_conj_groups, int ngroups, int swap_2e,
                              int s_nao, int nao, int nmo, int ngrids, int nkpts)
{
        size_t nao2 = nao * nao;
        size_t size_vk = nkpts * nao2;
        double *v00R = calloc(sizeof(double), size_vk*4);
        double *v00I = v00R + size_vk;
        double *v01R = v00I + size_vk;
        double *v01I = v01R + size_vk;
        int *k_list = k_conj_groups;
        int *kc_list = k_conj_groups + ngroups;
        int *ki2kj = malloc(sizeof(int) * nkpts * 2);
        int *kj2ki = ki2kj + nkpts;
        int n;
#pragma GCC ivdep
        for (n = 0; n < nkpts; n++) {
                ki2kj[ki_idx[n]] = kj_idx[n];
                kj2ki[kj_idx[n]] = ki_idx[n];
        }

        int naog = nao * ngrids;
        int nmog = nmo * ngrids;
        int knmo = nkpts * nmo;
        size_t Naog = naog;
        double *GpiR = malloc(sizeof(double) * Naog * knmo * 2);
        double *GpiI = GpiR + naog * knmo;

        char TRANS_T = 'T';
        char TRANS_N = 'N';
        double D1 = 1.;
        double D0 = 0.;
        double N1 = -1.;

#pragma omp parallel
{
        size_t k0, k1;
        NPomp_split(&k0, &k1, knmo);
        int dk = k1 - k0;
        double *pR = GpiR + Naog * k0;
        double *pI = GpiI + Naog * k0;
        dgemm_(&TRANS_N, &TRANS_T, &naog, &dk, &s_nao, &D1, GpqR, &naog, moR+k0, &knmo, &D0, pR, &naog);
        dgemm_(&TRANS_N, &TRANS_T, &naog, &dk, &s_nao, &D1, GpqI, &naog, moR+k0, &knmo, &D0, pI, &naog);
#pragma omp barrier

        double *buf0R = malloc(sizeof(double) * Naog * nmo * 6);
        double *buf0I = buf0R + Naog * nmo;
        double *buf1R = buf0I + Naog * nmo;
        double *buf1I = buf1R + Naog * nmo;
        double *bufwR = buf1I + Naog * nmo;
        double *bufwI = bufwR + Naog * nmo;
        double *out0R, *out0I, *out1R, *out1I, *outwR, *outwI;
        int kp, k, kc, i, j, ig, ki, kj, kic, kjc;

#pragma omp for schedule(dynamic)
        for (kp = 0; kp < ngroups; kp++) {
                k = k_list[kp];
                for (i = 0; i < nmo; i++) {
                        out0R = buf0R + i * ngrids;
                        out0I = buf0I + i * ngrids;
                        outwR = bufwR + i * ngrids;
                        outwI = bufwI + i * ngrids;
                        pR = GpiR + (k * nmo + i) * Naog;
                        pI = GpiI + (k * nmo + i) * Naog;
                        for (j = 0; j < nao; j++) {
#pragma GCC ivdep
                        for (ig = 0; ig < ngrids; ig++) {
                                out0R[j*nmog+ig] = pR[j*ngrids+ig];
                                out0I[j*nmog+ig] = pI[j*ngrids+ig];
                                outwR[j*nmog+ig] = pR[j*ngrids+ig] * coulG[ig];
                                outwI[j*nmog+ig] = pI[j*ngrids+ig] * coulG[ig];
                        } }
                }
// zdotNC(buf*w, buf.T).T
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &D1, bufwR, &nmog, buf0R, &nmog, &D1, v00R+k*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &D1, bufwI, &nmog, buf0I, &nmog, &D1, v00R+k*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &D1, bufwR, &nmog, buf0I, &nmog, &D1, v00I+k*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &N1, bufwI, &nmog, buf0R, &nmog, &D1, v00I+k*nao2, &nao);

                kc = kc_list[kp];
                if (k == kc) {
                        continue;
                }

                for (i = 0; i < nmo; i++) {
                        out1R = buf1R + i * ngrids;
                        out1I = buf1I + i * ngrids;
                        outwR = bufwR + i * ngrids;
                        outwI = bufwI + i * ngrids;
                        pR = GpiR + (kc * nmo + i) * Naog;
                        pI = GpiI + (kc * nmo + i) * Naog;
                        for (j = 0; j < nao; j++) {
#pragma GCC ivdep
                        for (ig = 0; ig < ngrids; ig++) {
                                out1R[j*nmog+ig] = pR[j*ngrids+ig];
                                out1I[j*nmog+ig] = pI[j*ngrids+ig];
                                outwR[j*nmog+ig] = pR[j*ngrids+ig] * coulG[ig];
                                outwI[j*nmog+ig] = pI[j*ngrids+ig] * coulG[ig];
                        } }
                }
// zdotNC(buf*w, buf.T).T
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &D1, bufwR, &nmog, buf1R, &nmog, &D1, v00R+kc*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &D1, bufwI, &nmog, buf1I, &nmog, &D1, v00R+kc*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &D1, bufwR, &nmog, buf1I, &nmog, &D1, v00I+kc*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &N1, bufwI, &nmog, buf1R, &nmog, &D1, v00I+kc*nao2, &nao);

// zdotNC(buf*w, buf.T).T
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &D1, bufwR, &nmog, buf0R, &nmog, &D1, v01R+k*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &D1, bufwI, &nmog, buf0I, &nmog, &D1, v01R+k*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &D1, bufwR, &nmog, buf0I, &nmog, &D1, v01I+k*nao2, &nao);
dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &nmog, &N1, bufwI, &nmog, buf0R, &nmog, &D1, v01I+k*nao2, &nao);
// Put v10[k] in v01[kc]
hermi_assign(v01R+kc*nao2, v01I+kc*nao2, v01R+k*nao2, v01I+k*nao2, nao);
        }
        free(buf0R);
#pragma omp barrier

        // k as kj
#pragma omp for schedule(dynamic)
        for (kp = 0; kp < ngroups; kp++) {
                k = k_list[kp];
                kc = kc_list[kp];
                ki = kj2ki[k];
if (k == kc) {
        if (k_to_compute[ki]) {
                plain_add(vkR+ki*nao2, vkI+ki*nao2, v00R+k*nao2, v00I+k*nao2, nao);
        }
} else { // k != kc
        if (k_to_compute[ki]) {
                pp_add(vkR+ki*nao2, vkI+ki*nao2,
                       v00R+k*nao2, v00I+k*nao2, v00R+kc*nao2, v00I+kc*nao2,
                       v01R+k*nao2, v01I+k*nao2, v01R+kc*nao2, v01I+kc*nao2, nao);
        }
        kic = kj2ki[kc];
        if (k_to_compute[kic]) {
                cc_add(vkR+kic*nao2, vkI+kic*nao2,
                       v00R+k*nao2, v00I+k*nao2, v00R+kc*nao2, v00I+kc*nao2,
                       v01R+k*nao2, v01I+k*nao2, v01R+kc*nao2, v01I+kc*nao2, nao);
        }
}
        }
#pragma omp barrier

        if (swap_2e) {
                // k as ki
#pragma omp for schedule(dynamic)
                for (kp = 0; kp < ngroups; kp++) {
                        k = k_list[kp];
                        kc = kc_list[kp];
                        kj = ki2kj[k];
if (k == kc) {
        if (k_to_compute[kj]) {
                transpose_add(vkR+kj*nao2, vkI+kj*nao2, v00R+k*nao2, v00I+k*nao2, nao);
        }
} else { // k != kc
        if (k_to_compute[kj]) {
                cc_tadd(vkR+kj*nao2, vkI+kj*nao2,
                        v00R+k*nao2, v00I+k*nao2, v00R+kc*nao2, v00I+kc*nao2,
                        v01R+k*nao2, v01I+k*nao2, v01R+kc*nao2, v01I+kc*nao2, nao);
        }
        kjc = ki2kj[kc];
        if (k_to_compute[kjc]) {
                pp_tadd(vkR+kjc*nao2, vkI+kjc*nao2,
                        v00R+k*nao2, v00I+k*nao2, v00R+kc*nao2, v00I+kc*nao2,
                        v01R+k*nao2, v01I+k*nao2, v01R+kc*nao2, v01I+kc*nao2, nao);
        }
}
                }
        }
}
        free(GpiR);
        free(v00R);
        free(ki2kj);
}
