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
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <math.h>
#include "cint.h"
#include "config.h"
#include "gto/grid_ao_drv.h"
#include "np_helper/np_helper.h"

// 1k grids per block
#define GRIDS_BLOCK     512
#define ALIGNMENT       8

void VXCgen_grid(double *out, double *coords, double *atm_coords,
                 double *radii_table, int natm, int ngrids)
{
        const size_t Ngrids = ngrids;
        int i, j;
        double dx, dy, dz;
        double *atom_dist = malloc(sizeof(double) * natm*natm);
        for (i = 0; i < natm; i++) {
                for (j = 0; j < i; j++) {
                        dx = atm_coords[i*3+0] - atm_coords[j*3+0];
                        dy = atm_coords[i*3+1] - atm_coords[j*3+1];
                        dz = atm_coords[i*3+2] - atm_coords[j*3+2];
                        atom_dist[i*natm+j] = 1 / sqrt(dx*dx + dy*dy + dz*dz);
                }
        }

#pragma omp parallel private(i, j, dx, dy, dz)
{
        double *_buf = malloc(sizeof(double) * ((natm*2+1)*GRIDS_BLOCK + ALIGNMENT));
        double *buf = (double *)((uintptr_t)(_buf + ALIGNMENT - 1) & (-(uintptr_t)(ALIGNMENT*8)));
        double *g = buf + natm * GRIDS_BLOCK;
        double *grid_dist = g + GRIDS_BLOCK;
        size_t ig0, n, ngs;
        double fac, s;
#pragma omp for nowait schedule(static)
        for (ig0 = 0; ig0 < Ngrids; ig0 += GRIDS_BLOCK) {
                ngs = MIN(Ngrids-ig0, GRIDS_BLOCK);
                for (i = 0; i < natm; i++) {
#pragma GCC ivdep
                for (n = 0; n < ngs; n++) {
                        dx = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                        dy = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                        dz = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
                        grid_dist[i*GRIDS_BLOCK+n] = sqrt(dx*dx + dy*dy + dz*dz);
                        buf[i*GRIDS_BLOCK+n] = 1;
                } }

                for (i = 0; i < natm; i++) {
                for (j = 0; j < i; j++) {

                        fac = atom_dist[i*natm+j];
                        for (n = 0; n < ngs; n++) {
                                g[n] = (grid_dist[i*GRIDS_BLOCK+n] -
                                        grid_dist[j*GRIDS_BLOCK+n]) * fac;
                        }
                        if (radii_table != NULL) {
                                fac = radii_table[i*natm+j];
                                for (n = 0; n < ngs; n++) {
                                        g[n] += fac * (1 - g[n]*g[n]);
                                }
                        }
#pragma GCC ivdep
                        for (n = 0; n < ngs; n++) {
                                s = g[n];
                                s = (3 - s*s) * s * .5;
                                s = (3 - s*s) * s * .5;
                                s = ((3 - s*s) * s * .5) * .5;
                                buf[i*GRIDS_BLOCK+n] *= .5 - s;
                                buf[j*GRIDS_BLOCK+n] *= .5 + s;
                        }
                } }

                for (i = 0; i < natm; i++) {
                        for (n = 0; n < ngs; n++) {
                                out[i*Ngrids+ig0+n] = buf[i*GRIDS_BLOCK+n];
                        }
                }
        }
        free(_buf);
}
        free(atom_dist);
}

typedef struct { double x, y, z; } double3;

static inline double3 d3_plus(const double3 v1, const double3 v2) { double3 v = { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z }; return v; }
static inline double3 d3_minus(const double3 v1, const double3 v2) { double3 v = { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z }; return v; }
static inline double3 d3_negate(const double3 v) { double3 nv = { -v.x, -v.y, -v.z }; return nv; }
static inline double3 d3_scale(const double k, const double3 v) { double3 kv = { k * v.x, k * v.y, k * v.z }; return kv; }
static inline double norm(const double3 v) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
static inline double inv(const double x)
{
    if (x > 1e-14) return 1.0 / x;
    else return 0.0;
}

static double switch_function(const double mu, const double a_factor)
{
    const double nu = mu + a_factor * (1.0 - mu * mu);
    double s = nu;
    s = (3.0 - s * s) * s * 0.5;
    s = (3.0 - s * s) * s * 0.5;
    s = (3.0 - s * s) * s * 0.5;
    s = 0.5 * (1.0 - s);
    return s;
}

static double switch_function_dmuds_over_s(const double mu, const double a_factor)
{
    const double nu = mu + a_factor * (1 - mu * mu);
    const double dnu_dmu = 1.0 - 2.0 * a_factor * mu;
    const double f1 = (3.0 - nu * nu) * nu * 0.5;
    const double f2 = (3.0 - f1 * f1) * f1 * 0.5;
    const double f3 = (3.0 - f2 * f2) * f2 * 0.5;
    const double s = 0.5 * (1.0 - f3);
    const double dmuds = -0.5 * 1.5 * (1 - f2 * f2) * 1.5 * (1 - f1 * f1) * 1.5 * (1 - nu * nu) * dnu_dmu;
    return dmuds * inv(s);
}

void VXCbecke_weight_derivative(double* __restrict__ dwdG, const double* __restrict__ grid_coords, const double* __restrict__ grid_quadrature_weights,
                                const double* __restrict__ atm_coords, const double* __restrict__ a_factor,
                                const int* __restrict__ atm_idx, const int ngrids, const int natm)
{
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i_grid = 0; i_grid < ngrids; i_grid++) {
        for (int i_derivative_atom = 0; i_derivative_atom < natm; i_derivative_atom++) {
            const int i_associated_atom = atm_idx[i_grid];
            if (i_associated_atom < 0) // Pad grid
                continue;
            if (i_associated_atom == i_derivative_atom) // Dealt with later by translation invariance.
                continue;

            const double3 grid_r = { grid_coords[i_grid * 3 + 0], grid_coords[i_grid * 3 + 1], grid_coords[i_grid * 3 + 2] };
            const double3 atom_A = { atm_coords[i_associated_atom * 3 + 0], atm_coords[i_associated_atom * 3 + 1], atm_coords[i_associated_atom * 3 + 2] };
            const double3 atom_G = { atm_coords[i_derivative_atom * 3 + 0], atm_coords[i_derivative_atom * 3 + 1], atm_coords[i_derivative_atom * 3 + 2] };
            const double3 Ar = d3_minus(atom_A, grid_r);
            const double3 Gr = d3_minus(atom_G, grid_r);
            const double norm_Ar = norm(Ar);
            const double norm_Gr = norm(Gr);
            const double norm_Gr_1 = inv(norm_Gr);

            double P_A = 1.0;
            double sum_P_B = 0.0;
            double3 sum_dPB_dG = { 0.0, 0.0, 0.0 };
            double P_G = 1.0;
            double3 dPG_dG = { 0.0, 0.0, 0.0 };

            for (int j_atom = 0; j_atom < natm; j_atom++) {
                const double3 atom_B = { atm_coords[j_atom * 3 + 0], atm_coords[j_atom * 3 + 1], atm_coords[j_atom * 3 + 2] };
                const double3 Br = d3_minus(atom_B, grid_r);
                const double norm_Br = norm(Br);

                const double3 AB = d3_minus(atom_A, atom_B);
                const double norm_AB_1 = inv(norm(AB));

                const double mu_AB = (norm_Ar - norm_Br) * norm_AB_1;
                const double a_factor_AB = a_factor[i_associated_atom * natm + j_atom];
                const double s_AB = switch_function(mu_AB, a_factor_AB);

                P_A *= s_AB;

                double P_B = 1.0;

                for (int k_atom = 0; k_atom < natm; k_atom++) {
                    const double3 atom_C = { atm_coords[k_atom * 3 + 0], atm_coords[k_atom * 3 + 1], atm_coords[k_atom * 3 + 2] };
                    const double3 Cr = d3_minus(atom_C, grid_r);
                    const double3 BC = d3_minus(atom_B, atom_C);
                    const double norm_Cr = norm(Cr);
                    const double norm_BC_1 = inv(norm(BC));

                    const double mu_BC = (norm_Br - norm_Cr) * norm_BC_1;
                    const double a_factor_BC = a_factor[j_atom * natm + k_atom];
                    const double s_BC = switch_function(mu_BC, a_factor_BC);

                    P_B *= s_BC;
                }

                sum_P_B += P_B;

                const double3 BG = d3_minus(atom_B, atom_G);
                const double norm_BG_1 = inv(norm(BG));
                const double mu_BG = (norm_Br - norm_Gr) * norm_BG_1;
                const double3 dmuBG_dG = d3_scale(norm_BG_1, d3_plus(d3_scale(-norm_Gr_1, Gr), d3_scale(mu_BG * norm_BG_1, BG)));
                const double a_factor_BG = a_factor[j_atom * natm + i_derivative_atom];
                const double3 dPB_dG = d3_scale(switch_function_dmuds_over_s(mu_BG, a_factor_BG) * P_B, dmuBG_dG);

                sum_dPB_dG = d3_plus(sum_dPB_dG, dPB_dG);

                const double a_factor_GB = a_factor[i_derivative_atom * natm + j_atom];
                const double s_GB = switch_function(-mu_BG, a_factor_GB);
                P_G *= s_GB;

                const double3 dmuGB_dG = d3_negate(dmuBG_dG);
                dPG_dG = d3_plus(dPG_dG, d3_scale(switch_function_dmuds_over_s(-mu_BG, a_factor_GB), dmuGB_dG));
            }

            sum_dPB_dG = d3_plus(sum_dPB_dG, d3_scale(P_G, dPG_dG));

            const double3 AG = d3_minus(atom_A, atom_G);
            const double norm_AG_1 = inv(norm(AG));
            const double mu_AG = (norm_Ar - norm_Gr) * norm_AG_1;
            const double3 dmuAG_dG = d3_scale(norm_AG_1, d3_plus(d3_scale(-norm_Gr_1, Gr), d3_scale(mu_AG * norm_AG_1, AG)));
            const double a_factor_AG = a_factor[i_associated_atom * natm + i_derivative_atom];
            const double3 dPA_dG = d3_scale(switch_function_dmuds_over_s(mu_AG, a_factor_AG) * P_A, dmuAG_dG);

            const double quadrature_weight = grid_quadrature_weights[i_grid];
            const double3 dwi_dG = d3_scale(quadrature_weight, d3_minus(d3_scale(inv(sum_P_B), dPA_dG), d3_scale(inv(sum_P_B * sum_P_B) * P_A, sum_dPB_dG)));

            dwdG[i_derivative_atom * ngrids * 3 + 0 * ngrids + i_grid] = dwi_dG.x;
            dwdG[i_derivative_atom * ngrids * 3 + 1 * ngrids + i_grid] = dwi_dG.y;
            dwdG[i_derivative_atom * ngrids * 3 + 2 * ngrids + i_grid] = dwi_dG.z;
        }
    }
}
