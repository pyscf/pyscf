/* Copyright 2021- The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 */

#ifndef HAVE_DEFINED_GRID_COMMON_H
#define HAVE_DEFINED_GRID_COMMON_H

#include "cint.h"

#define EIJCUTOFF        60
#define PTR_EXPDROP      16

extern double CINTsquare_dist(const double *r1, const double *r2);
extern double CINTcommon_fac_sp(int l);

int get_lmax(int ish0, int ish1, int* bas);
int get_nprim_max(int ish0, int ish1, int* bas);
int get_nctr_max(int ish0, int ish1, int* bas);
void get_cart2sph_coeff(double** contr_coeff, double** gto_norm,
                        int ish0, int ish1, int* bas, double* env, int cart);
void del_cart2sph_coeff(double** contr_coeff, double** gto_norm, int ish0, int ish1);

static inline int _has_overlap(int nx0, int nx1, int nx_per_cell)
{
    return nx0 <= nx1;
}

static inline int _num_grids_on_x(int nimgx, int nx0, int nx1, int nx_per_cell)
{
    int ngridx;
    if (nimgx == 1) {
        ngridx = nx1 - nx0;
    } else if (nimgx == 2 && !_has_overlap(nx0, nx1, nx_per_cell)) {
        ngridx = nx1 - nx0 + nx_per_cell;
    } else {
        ngridx = nx_per_cell;
    }
    return ngridx;
}


static inline void _get_grid_mapping(int* xmap, int nx0, int nx1, int ngridx, int nimgx, bool is_x_split)
{
    int ix, nx;
    if (nimgx == 1) {
        for (ix = 0; ix < ngridx; ix++) {
            xmap[ix] = ix + nx0;
        }
    } else if (is_x_split) {
        for (ix = 0; ix < nx1; ix++) {
            xmap[ix] = ix;
        }
        nx = nx0 - nx1;
        for (ix = nx1; ix < ngridx; ix++) {
            xmap[ix] = ix + nx;
        }
    } else {
        for (ix = 0; ix < ngridx; ix++) {
            xmap[ix] = ix;
        }
    }
}


static inline int modulo(int i, int n)
{
    return (i % n + n) % n;
}


static inline int get_upper_bound(int x0, int nx_per_cell, int ix, int ngridx)
{
    return x0 + MIN(nx_per_cell - x0, ngridx - ix);
}

int _orth_components(double *xs_exp, int *img_slice, int *grid_slice,
                     double a, double b, double cutoff,
                     double xi, double xj, double ai, double aj,
                     int periodic, int nx_per_cell, int topl, double *cache);
int _init_orth_data(double **xs_exp, double **ys_exp, double **zs_exp,
                    int *img_slice, int *grid_slice, int *mesh,
                    int topl, int dimension, double cutoff,
                    double ai, double aj, double *ri, double *rj,
                    double *a, double *b, double *cache);

int init_orth_data(double **xs_exp, double **ys_exp, double **zs_exp,
                   int *grid_slice, double* dh, int* mesh, int topl, double radius,
                   double ai, double aj, double *ri, double *rj, double *cache);
void get_grid_spacing(double* dh, double* a, int* mesh);

void _get_dm_to_dm_xyz_coeff(double* coeff, double* rij, int lmax, double* cache);
void _dm_to_dm_xyz(double* dm_xyz, double* dm, int li, int lj, double* ri, double* rj, double* cache);
void _dm_xyz_to_dm(double* dm_xyz, double* dm, int li, int lj, double* ri, double* rj, double* cache);
void get_dm_pgfpair(double* dm_pgf, double* dm_cart,
                    PGFPair* pgfpair, int* ish_bas, int* jsh_bas, int hermi);
int get_max_num_grid_orth(double* dh, double radius);
#endif
