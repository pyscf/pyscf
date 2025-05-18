/* Copyright 2014-2020 The PySCF Developers. All Rights Reserved.

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
#include <stdbool.h>
#include <assert.h>
#include "config.h"
#include "gto/grid_ao_drv.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"

#define BOXSIZE         56

typedef struct { double x, y, z; } double3;

int VXCao_empty_blocks(int8_t *empty, uint8_t *non0table, int *shls_slice,
                       int *ao_loc)
{
        if (non0table == NULL || shls_slice == NULL || ao_loc == NULL) {
                return 0;
        }

        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];

        int bas_id;
        int box_id = 0;
        int bound = BOXSIZE;
        int has0 = 0;
        empty[box_id] = 1;
        for (bas_id = sh0; bas_id < sh1; bas_id++) {
                if (ao_loc[bas_id] == bound) {
                        has0 |= empty[box_id];
                        box_id++;
                        bound += BOXSIZE;
                        empty[box_id] = 1;
                }
                empty[box_id] &= !non0table[bas_id];
                if (ao_loc[bas_id+1] > bound) {
                        has0 |= empty[box_id];
                        box_id++;
                        bound += BOXSIZE;
                        empty[box_id] = !non0table[bas_id];
                }
        }
        return has0;
}

static void dot_ao_dm(double *vm, double *ao, double *dm,
                      int nao, int nocc, int ngrids, int bgrids,
                      uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        int nbox = (nao+BOXSIZE-1) / BOXSIZE;
        int8_t empty[nbox];
        int has0 = VXCao_empty_blocks(empty, non0table, shls_slice, ao_loc);

        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D1 = 1;
        double beta = 0;

        if (has0) {
                int box_id, blen, i, j;
                size_t b0;

                for (box_id = 0; box_id < nbox; box_id++) {
                        if (!empty[box_id]) {
                                b0 = box_id * BOXSIZE;
                                blen = MIN(nao-b0, BOXSIZE);
                                dgemm_(&TRANS_N, &TRANS_T, &bgrids, &nocc, &blen,
                                       &D1, ao+b0*ngrids, &ngrids, dm+b0*nocc, &nocc,
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
                dgemm_(&TRANS_N, &TRANS_T, &bgrids, &nocc, &nao,
                       &D1, ao, &ngrids, dm, &nocc, &beta, vm, &ngrids);
        }
}


/* vm[nocc,ngrids] = ao[i,ngrids] * dm[i,nocc] */
void VXCdot_ao_dm(double *vm, double *ao, double *dm,
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



/* vv[n,m] = ao1[n,ngrids] * ao2[m,ngrids] */
static void dot_ao_ao(double *vv, double *ao1, double *ao2,
                      int nao, int ngrids, int bgrids, int hermi,
                      uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        int nbox = (nao+BOXSIZE-1) / BOXSIZE;
        int8_t empty[nbox];
        int has0 = VXCao_empty_blocks(empty, non0table, shls_slice, ao_loc);

        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D1 = 1;
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
                                dgemm_(&TRANS_T, &TRANS_N, &lenj, &leni, &bgrids, &D1,
                                       ao2+b0j*ngrids, &ngrids, ao1+b0i*ngrids, &ngrids,
                                       &D1, vv+b0i*nao+b0j, &nao);
                        } }
                } }
        } else {
                dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &bgrids,
                       &D1, ao2, &ngrids, ao1, &ngrids, &D1, vv, &nao);
        }
}


/* vv[nao,nao] = ao1[i,nao] * ao2[i,nao] */
void VXCdot_ao_ao(double *vv, double *ao1, double *ao2,
                  int nao, int ngrids, int nbas, int hermi,
                  uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
        size_t Nao = nao;
        NPdset0(vv, Nao * Nao);

#pragma omp parallel
{
        int ip, ib;
        double *v_priv = calloc(Nao*Nao+2, sizeof(double));
#pragma omp for nowait schedule(static)
        for (ib = 0; ib < nblk; ib++) {
                ip = ib * BLKSIZE;
                dot_ao_ao(v_priv, ao1+ip, ao2+ip,
                          nao, ngrids, MIN(ngrids-ip, BLKSIZE), hermi,
                          non0table+ib*nbas, shls_slice, ao_loc);
        }
#pragma omp critical
        {
                for (ip = 0; ip < Nao*Nao; ip++) {
                        vv[ip] += v_priv[ip];
                }
        }
        free(v_priv);
}
        if (hermi != 0) {
                NPdsymm_triu(nao, vv, hermi);
        }
}

// 'nip,np->ip'
void VXC_dscale_ao(double *aow, double *ao, double *wv,
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

// 'ip,ip->p'
void VXC_dcontract_rho(double *rho, double *bra, double *ket,
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
                        rho[j] = bra[j] * ket[j];
                }
                for (i = 1; i < nao; i++) {
                for (j = b0; j < b1; j++) {
                        rho[j] += bra[i*Ngrids+j] * ket[i*Ngrids+j];
                } }
        }
}
}

void VXC_vv10nlc(double *Fvec, double *Uvec, double *Wvec,
                 double *vvcoords, double *coords,
                 double *W0p, double *W0, double *K, double *Kp, double *RpW,
                 int vvngrids, int ngrids)
{
#pragma omp parallel
{
        double DX, DY, DZ, R2;
        double gp, g, gt, T, F, U, W;
        int i, j;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
                F = 0;
                U = 0;
                W = 0;
                for (j = 0; j < vvngrids; j++) {
                        DX = vvcoords[j*3+0] - coords[i*3+0];
                        DY = vvcoords[j*3+1] - coords[i*3+1];
                        DZ = vvcoords[j*3+2] - coords[i*3+2];
                        R2 = DX*DX + DY*DY + DZ*DZ;
                        gp = R2*W0p[j] + Kp[j];
                        g  = R2*W0[i] + K[i];
                        gt = g + gp;
                        T = RpW[j] / (g*gp*gt);
                        F += T;
                        T *= 1./g + 1./gt;
                        U += T;
                        W += T * R2;
                }
                Fvec[i] = F * -1.5;
                Uvec[i] = U;
                Wvec[i] = W;
        }
}
}

void VXC_vv10nlc_grad(double *Fvec, double *vvcoords, double *coords,
                      double *W0p, double *W0, double *K, double *Kp, double *RpW,
                      int vvngrids, int ngrids)
{
#pragma omp parallel
{
        double DX, DY, DZ, R2;
        double gp, g, gt, T, Q, FX, FY, FZ;
        int i, j;
#pragma omp for schedule(static)
        for (i = 0; i < ngrids; i++) {
                FX = 0;
                FY = 0;
                FZ = 0;
                for (j = 0; j < vvngrids; j++) {
                        DX = vvcoords[j*3+0] - coords[i*3+0];
                        DY = vvcoords[j*3+1] - coords[i*3+1];
                        DZ = vvcoords[j*3+2] - coords[i*3+2];
                        R2 = DX*DX + DY*DY + DZ*DZ;
                        gp = R2*W0p[j] + Kp[j];
                        g  = R2*W0[i] + K[i];
                        gt = g + gp;
                        T = RpW[j] / (g*gp*gt);
                        Q = T * (W0[i]/g + W0p[j]/gp + (W0[i]+W0p[j])/gt);
                        FX += Q * DX;
                        FY += Q * DY;
                        FZ += Q * DZ;
                }
                Fvec[i*3+0] = FX * -3;
                Fvec[i*3+1] = FY * -3;
                Fvec[i*3+2] = FZ * -3;
        }
}
}

void VXC_vv10nlc_hessian_eval_UWABCE(double* __restrict__ U, double* __restrict__ W, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, double* __restrict__ E,
                                     const double* __restrict__ grid_coord, const double* __restrict__ grid_weight,
                                     const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                     const int ngrids)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ngrids; i++) {
        const double omega_i = omega[i];
        const double kappa_i = kappa[i];
        const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

        double U_i = 0;
        double W_i = 0;
        double A_i = 0;
        double B_i = 0;
        double C_i = 0;
        double E_i = 0;

        for (int j = 0; j < ngrids; j++) {
            const double omega_j = omega[j];
            const double kappa_j = kappa[j];
            const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
            const double weight_j = grid_weight[j];
            const double rho_j = rho[j];

            const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
            const double g_ij = omega_i * r_ij2 + kappa_i;
            const double g_ji = omega_j * r_ij2 + kappa_j;
            const double g_ij_1 = 1 / g_ij;
            const double g_sum_1 = 1 / (g_ij + g_ji);
            const double Phi_ij = -1.5 / g_ji * g_ij_1 * g_sum_1;

            const double E_ij = weight_j * rho_j * Phi_ij;
            const double U_ij = E_ij * (g_sum_1 + g_ij_1);
            const double W_ij = U_ij * r_ij2;
            const double A_ij = E_ij * (g_sum_1 * g_sum_1 + g_sum_1 * g_ij_1 + g_ij_1 * g_ij_1);
            const double B_ij = A_ij * r_ij2;
            const double C_ij = B_ij * r_ij2;

            U_i += U_ij;
            W_i += W_ij;
            A_i += A_ij;
            B_i += B_ij;
            C_i += C_ij;
            E_i += E_ij;
        }

        U[i] = -U_i;
        W[i] = -W_i;
        A[i] = 2 * A_i;
        B[i] = 2 * B_i;
        C[i] = 2 * C_i;
        E[i] = E_i;
    }
}

void VXC_vv10nlc_hessian_eval_omega_derivative(double* __restrict__ domega_drho, double* __restrict__ domega_dgamma,
                                               double* __restrict__ d2omega_drho2, double* __restrict__ d2omega_dgamma2, double* __restrict__ d2omega_drho_dgamma,
                                               const double* __restrict__ rho, const double* __restrict__ gamma, const double C_factor,
                                               const int ngrids)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ngrids; i++) {
        const double rho_i = rho[i];
        const double gamma_i = gamma[i];

        const double rho_1 = 1 / rho_i;
        const double rho_2 = rho_1 * rho_1;
        const double rho_3 = rho_1 * rho_2;
        const double rho_4 = rho_2 * rho_2;
        const double rho_5 = rho_1 * rho_4;
        const double gamma2 = gamma_i * gamma_i;
        const double four_pi_over_three = 4.0 / 3.0 * M_PI;
        const double omega2 = C_factor * gamma2 * rho_4 + four_pi_over_three * rho_i;
        const double omega = sqrt(omega2);
        const double omega_1 = 1 / omega;

        domega_drho[i] = 0.5 * (four_pi_over_three - 4 * C_factor * gamma2 * rho_5) * omega_1;
        domega_dgamma[i] = C_factor * gamma_i * rho_4 * omega_1;

        const double omega_3 = omega_1 / omega2;
        d2omega_drho2[i] = (-0.25 * four_pi_over_three * four_pi_over_three
                            + 12 * four_pi_over_three * C_factor * gamma2 * rho_5
                            + 6 * C_factor * C_factor * gamma2 * gamma2 * rho_5 * rho_5) * omega_3;
        d2omega_dgamma2[i] = four_pi_over_three * C_factor * rho_3 * omega_3;
        d2omega_drho_dgamma[i] = -C_factor * gamma_i * (4.5 * four_pi_over_three * rho_4
                                                        + 2 * C_factor * gamma2 * rho_4 * rho_5) * omega_3;
    }
}

void VXC_vv10nlc_hessian_eval_f_t(double* __restrict__ f_rho_t, double* __restrict__ f_gamma_t,
                                  const double* __restrict__ grid_coord, const double* __restrict__ grid_weight,
                                  const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                  const double* __restrict__ U, const double* __restrict__ W, const double* __restrict__ A, const double* __restrict__ B, const double* __restrict__ C,
                                  const double* __restrict__ domega_drho, const double* __restrict__ domega_dgamma, const double* __restrict__ dkappa_drho,
                                  const double* __restrict__ d2omega_drho2, const double* __restrict__ d2omega_dgamma2, const double* __restrict__ d2omega_drho_dgamma, const double* __restrict__ d2kappa_drho2,
                                  const double* __restrict__ rho_t, const double* __restrict__ gamma_t,
                                  const int ngrids, const int ntrial)
{
    const int n_trial_per_thread = 6;

    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < ngrids; i++) {
        for (int i_trial_start = 0; i_trial_start < ntrial; i_trial_start += n_trial_per_thread) {
            const double omega_i = omega[i];
            const double kappa_i = kappa[i];
            const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

            const double rho_i = rho[i];
            const double domega_drho_i = domega_drho[i];
            const double domega_dgamma_i = domega_dgamma[i];
            const double dkappa_drho_i = dkappa_drho[i];

            double f_rho_t_i[n_trial_per_thread];
            double f_gamma_t_i[n_trial_per_thread];
            #pragma GCC ivdep
            for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
                f_rho_t_i  [i_trial] = 0;
                f_gamma_t_i[i_trial] = 0;
            }

            for (int j = 0; j < ngrids; j++) {
                const double omega_j = omega[j];
                const double kappa_j = kappa[j];
                const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
                const double rho_j = rho[j];

                const double domega_drho_j = domega_drho[j];
                const double domega_dgamma_j = domega_dgamma[j];
                const double dkappa_drho_j = dkappa_drho[j];

                const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
                const double g_ij = omega_i * r_ij2 + kappa_i;
                const double g_ji = omega_j * r_ij2 + kappa_j;
                const double g_ij_1 = 1 / g_ij;
                const double g_ji_1 = 1 / g_ji;
                const double g_sum_1 = 1 / (g_ij + g_ji);
                const double Phi_ij = -1.5 * g_ij_1 * g_ji_1 * g_sum_1;

                const double rho_dgdrho_i = rho_i * (r_ij2 * domega_drho_i + dkappa_drho_i);
                const double rho_dgdrho_j = rho_j * (r_ij2 * domega_drho_j + dkappa_drho_j);
                const double d2Phi_dgij_dgji_over_Phi = 2 * (g_sum_1 * g_sum_1 + g_ij_1 * g_ji_1);

                const double f_rho_rho_ij = Phi_ij * (rho_dgdrho_i * rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi
                                                    - rho_dgdrho_i * (g_sum_1 + g_ij_1)
                                                    - rho_dgdrho_j * (g_sum_1 + g_ji_1) + 1);
                const double f_gamma_rho_ij = rho_i * domega_dgamma_i * r_ij2 * Phi_ij * (rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ij_1));
                const double f_rho_gamma_ij = rho_j * domega_dgamma_j * r_ij2 * Phi_ij * (rho_dgdrho_i * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ji_1));
                const double f_gamma_gamma_ij = rho_i * rho_j * domega_dgamma_i * domega_dgamma_j * r_ij2 * r_ij2 * Phi_ij * d2Phi_dgij_dgji_over_Phi;

                const double weight_j = grid_weight[j];

                #pragma GCC ivdep
                for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
                    if (i_trial + i_trial_start >= ntrial) continue;
                    const double   rho_t_j =   rho_t[(i_trial + i_trial_start) * ngrids + j];
                    const double gamma_t_j = gamma_t[(i_trial + i_trial_start) * ngrids + j];
                    f_rho_t_i  [i_trial] += weight_j * (  f_rho_rho_ij * rho_t_j +   f_rho_gamma_ij * gamma_t_j);
                    f_gamma_t_i[i_trial] += weight_j * (f_gamma_rho_ij * rho_t_j + f_gamma_gamma_ij * gamma_t_j);
                }
            }

            const double U_i = U[i];
            const double W_i = W[i];
            const double A_i = A[i];
            const double B_i = B[i];
            const double C_i = C[i];
            const double d2omega_drho2_i = d2omega_drho2[i];
            const double d2omega_dgamma2_i = d2omega_dgamma2[i];
            const double d2omega_drho_dgamma_i = d2omega_drho_dgamma[i];
            const double d2kappa_drho2_i = d2kappa_drho2[i];

            const double f_rho_rho_ii = 2 * domega_drho_i * W_i + 2 * dkappa_drho_i * U_i
                                        + rho_i * (d2omega_drho2_i * W_i + d2kappa_drho2_i * U_i + dkappa_drho_i * dkappa_drho_i * A_i
                                                + domega_drho_i * domega_drho_i * C_i + 2 * domega_drho_i * dkappa_drho_i * B_i);
            const double f_gamma_rho_ii = domega_dgamma_i * W_i + rho_i * (d2omega_drho_dgamma_i * W_i
                                                                        + domega_dgamma_i * (dkappa_drho_i * B_i + domega_drho_i * C_i));
            const double f_rho_gamma_ii = f_gamma_rho_ii;
            const double f_gamma_gamma_ii = rho_i * (d2omega_dgamma2_i * W_i + domega_dgamma_i * domega_dgamma_i * C_i);

            #pragma GCC ivdep
            for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
                if (i_trial + i_trial_start >= ntrial) continue;
                const double rho_t_i   =   rho_t[(i_trial + i_trial_start) * ngrids + i];
                const double gamma_t_i = gamma_t[(i_trial + i_trial_start) * ngrids + i];
                f_rho_t_i  [i_trial] += (  f_rho_rho_ii * rho_t_i +   f_rho_gamma_ii * gamma_t_i);
                f_gamma_t_i[i_trial] += (f_gamma_rho_ii * rho_t_i + f_gamma_gamma_ii * gamma_t_i);

                f_rho_t  [(i_trial + i_trial_start) * ngrids + i] = f_rho_t_i  [i_trial];
                f_gamma_t[(i_trial + i_trial_start) * ngrids + i] = f_gamma_t_i[i_trial];
            }
        }
    }
}

void VXC_vv10nlc_hessian_eval_EUW_grid_response(double* __restrict__ Egr, double* __restrict__ Ugr, double* __restrict__ Wgr,
                                                const double* __restrict__ grid_coord, const double* __restrict__ grid_weight,
                                                const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                                const int* __restrict__ grid_associated_atom,
                                                const int ngrids, const int natoms)
{
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < ngrids; i++) {
        for (int B_atom = 0; B_atom < natoms; B_atom++) {
            const int i_associated_atom = grid_associated_atom[i];
            if (i_associated_atom < 0) {
                Egr[B_atom * 3 * ngrids + 0 * ngrids + i] = 0;
                Egr[B_atom * 3 * ngrids + 1 * ngrids + i] = 0;
                Egr[B_atom * 3 * ngrids + 2 * ngrids + i] = 0;
                Ugr[B_atom * 3 * ngrids + 0 * ngrids + i] = 0;
                Ugr[B_atom * 3 * ngrids + 1 * ngrids + i] = 0;
                Ugr[B_atom * 3 * ngrids + 2 * ngrids + i] = 0;
                Wgr[B_atom * 3 * ngrids + 0 * ngrids + i] = 0;
                Wgr[B_atom * 3 * ngrids + 1 * ngrids + i] = 0;
                Wgr[B_atom * 3 * ngrids + 2 * ngrids + i] = 0;
                continue;
            }
            const bool i_in_B = (i_associated_atom == B_atom);

            const double omega_i = omega[i];
            const double kappa_i = kappa[i];
            const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

            double3 Egr_i = { 0, 0, 0 };
            double3 Ugr_i = { 0, 0, 0 };
            double3 Wgr_i = { 0, 0, 0 };

            for (int j = 0; j < ngrids; j++) {
                const int j_associated_atom = grid_associated_atom[j];
                if (j_associated_atom < 0)
                    continue;
                const int j_in_B = (j_associated_atom == B_atom);
                if (!i_in_B && !j_in_B)
                    continue;
                if (i_in_B && j_in_B)
                    continue;

                const double omega_j = omega[j];
                const double kappa_j = kappa[j];
                const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
                const double weight_j = grid_weight[j];
                const double rho_j = rho[j];

                const double3 r_ji = { r_j.x - r_i.x, r_j.y - r_i.y, r_j.z - r_i.z };
                const double r_ij2 = r_ji.x * r_ji.x + r_ji.y * r_ji.y + r_ji.z * r_ji.z;
                const double g_ij = omega_i * r_ij2 + kappa_i;
                const double g_ji = omega_j * r_ij2 + kappa_j;
                const double g_ij_1 = 1 / g_ij;
                const double g_ji_1 = 1 / g_ji;
                const double g_sum_1 = 1 / (g_ij + g_ji);
                const double Phi_ij = -1.5 * g_ij_1 * g_ji_1 * g_sum_1;

                const double E_ij = weight_j * rho_j * Phi_ij;
                const double dPhi_drj_over_Phi = omega_i * g_ij_1 + omega_j * g_ji_1 + (omega_i + omega_j) * g_sum_1;
                const double d2Phi_dgij_drj_over_Phi = omega_i * g_ij_1 * g_ij_1 + (omega_i + omega_j) * g_sum_1 * g_sum_1;
                const double dPhi_dgij_over_Phi = g_sum_1 + g_ij_1;

                const double Egr_ij = E_ij * dPhi_drj_over_Phi;
                const double Ugr_ij = E_ij * (dPhi_drj_over_Phi * dPhi_dgij_over_Phi + d2Phi_dgij_drj_over_Phi);
                const double Wgr_ij = E_ij * (r_ij2 * (dPhi_drj_over_Phi * dPhi_dgij_over_Phi + d2Phi_dgij_drj_over_Phi) - dPhi_dgij_over_Phi);

                Egr_i.x += Egr_ij * r_ji.x;
                Egr_i.y += Egr_ij * r_ji.y;
                Egr_i.z += Egr_ij * r_ji.z;
                Ugr_i.x += Ugr_ij * r_ji.x;
                Ugr_i.y += Ugr_ij * r_ji.y;
                Ugr_i.z += Ugr_ij * r_ji.z;
                Wgr_i.x += Wgr_ij * r_ji.x;
                Wgr_i.y += Wgr_ij * r_ji.y;
                Wgr_i.z += Wgr_ij * r_ji.z;
            }

            if (i_in_B) {
                Egr_i.x *= -1;
                Egr_i.y *= -1;
                Egr_i.z *= -1;
                Ugr_i.x *= -1;
                Ugr_i.y *= -1;
                Ugr_i.z *= -1;
                Wgr_i.x *= -1;
                Wgr_i.y *= -1;
                Wgr_i.z *= -1;
            }

            Egr[B_atom * 3 * ngrids + 0 * ngrids + i] = -2 * Egr_i.x;
            Egr[B_atom * 3 * ngrids + 1 * ngrids + i] = -2 * Egr_i.y;
            Egr[B_atom * 3 * ngrids + 2 * ngrids + i] = -2 * Egr_i.z;
            Ugr[B_atom * 3 * ngrids + 0 * ngrids + i] =  2 * Ugr_i.x;
            Ugr[B_atom * 3 * ngrids + 1 * ngrids + i] =  2 * Ugr_i.y;
            Ugr[B_atom * 3 * ngrids + 2 * ngrids + i] =  2 * Ugr_i.z;
            Wgr[B_atom * 3 * ngrids + 0 * ngrids + i] =  2 * Wgr_i.x;
            Wgr[B_atom * 3 * ngrids + 1 * ngrids + i] =  2 * Wgr_i.y;
            Wgr[B_atom * 3 * ngrids + 2 * ngrids + i] =  2 * Wgr_i.z;
        }
    }
}

void VXC_vv10nlc_hessian_eval_EUW_with_weight1(double* __restrict__ Ew, double* __restrict__ Uw, double* __restrict__ Ww,
                                               const double* __restrict__ grid_coord, const double* __restrict__ grid_weight1,
                                               const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                               const int ngrids, const int nderivative)
{
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < ngrids; i++) {
        for (int i_derivative = 0; i_derivative < nderivative; i_derivative++) {
            const double omega_i = omega[i];
            const double kappa_i = kappa[i];
            const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

            double Ew_i = 0;
            double Uw_i = 0;
            double Ww_i = 0;

            for (int j = 0; j < ngrids; j++) {
                const double omega_j = omega[j];
                const double kappa_j = kappa[j];
                const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
                const double rho_j = rho[j];

                const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
                const double g_ij = omega_i * r_ij2 + kappa_i;
                const double g_ji = omega_j * r_ij2 + kappa_j;
                const double g_ij_1 = 1 / g_ij;
                const double g_sum_1 = 1 / (g_ij + g_ji);
                const double Phi_ij = -1.5 / g_ji * g_ij_1 * g_sum_1;

                const double E_ij = rho_j * Phi_ij;
                const double U_ij = E_ij * (g_sum_1 + g_ij_1);
                const double W_ij = U_ij * r_ij2;

                const double weight_j = grid_weight1[i_derivative * ngrids + j];
                Ew_i += weight_j * E_ij;
                Uw_i += weight_j * U_ij;
                Ww_i += weight_j * W_ij;
            }

            Ew[i_derivative * ngrids + i] =  Ew_i;
            Uw[i_derivative * ngrids + i] = -Uw_i;
            Ww[i_derivative * ngrids + i] = -Ww_i;
        }
    }
}
