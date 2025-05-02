/* Copyright 2021-2024 The PySCF Developers. All Rights Reserved.

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
#include "np_helper/np_helper.h"

#if defined(_OPENMP) && _OPENMP >= 201307
    #define PRAGMA_OMP_SIMD _Pragma("omp simd")
#else
    #define PRAGMA_OMP_SIMD
#endif

#define EIJCUTOFF        60
#define PTR_EXPDROP      16
#define PTR_RADIUS        5

extern double CINTsquare_dist(const double *r1, const double *r2);
extern double CINTcommon_fac_sp(int l);

static inline double fac(const int i) {
    static const double fac_table[] = {
        1.00000000000000000000000000000000E+00,
        1.00000000000000000000000000000000E+00,
        2.00000000000000000000000000000000E+00,
        6.00000000000000000000000000000000E+00,
        2.40000000000000000000000000000000E+01,
        1.20000000000000000000000000000000E+02,
        7.20000000000000000000000000000000E+02,
        5.04000000000000000000000000000000E+03,
        4.03200000000000000000000000000000E+04,
        3.62880000000000000000000000000000E+05,
        3.62880000000000000000000000000000E+06,
        3.99168000000000000000000000000000E+07,
        4.79001600000000000000000000000000E+08,
        6.22702080000000000000000000000000E+09,
        8.71782912000000000000000000000000E+10,
        1.30767436800000000000000000000000E+12,
        2.09227898880000000000000000000000E+13,
        3.55687428096000000000000000000000E+14,
        6.40237370572800000000000000000000E+15,
        1.21645100408832000000000000000000E+17,
        2.43290200817664000000000000000000E+18,
        5.10909421717094400000000000000000E+19,
        1.12400072777760768000000000000000E+21,
        2.58520167388849766400000000000000E+22,
        6.20448401733239439360000000000000E+23,
        1.55112100433309859840000000000000E+25,
        4.03291461126605635584000000000000E+26,
        1.08888694504183521607680000000000E+28,
        3.04888344611713860501504000000000E+29,
        8.84176199373970195454361600000000E+30,
        2.65252859812191058636308480000000E+32, //30!
    };
    return fac_table[i];
}

int get_lmax(int ish0, int ish1, int* bas);
int get_nprim_max(int ish0, int ish1, int* bas);
int get_nctr_max(int ish0, int ish1, int* bas);
void get_cart2sph_coeff(double** contr_coeff, double** gto_norm,
                        int ish0, int ish1, int* bas, double* env, int cart);
void del_cart2sph_coeff(double** contr_coeff, double** gto_norm, int ish0, int ish1);

static inline int modulo(int i, int n)
{
    return (i % n + n) % n;
}

static inline int get_upper_bound(int x0, int nx_per_cell, int ix, int ngridx)
{
    return x0 + MIN(nx_per_cell - x0, ngridx - ix);
}

static inline void get_lattice_coords(double* r_latt, double* r, double* dh_inv)
{
    const double rx = r[0];
    const double ry = r[1];
    const double rz = r[2];
    r_latt[0] = rx * dh_inv[0] + ry * dh_inv[3] + rz * dh_inv[6];
    r_latt[1] = rx * dh_inv[1] + ry * dh_inv[4] + rz * dh_inv[7];
    r_latt[2] = rx * dh_inv[2] + ry * dh_inv[5] + rz * dh_inv[8];
}

int init_orth_data(double **xs_exp, double **ys_exp, double **zs_exp,
                   int *grid_slice, double* dh, int* mesh, int topl, double radius,
                   double ai, double aj, double *ri, double *rj, double *cache);

size_t init_nonorth_data(double **xs_exp, double **ys_exp, double **zs_exp,
                         double **exp_corr, int *bounds,
                         double* dh, double* dh_inv,
                         int* mesh, int topl, double radius,
                         double ai, double aj, double *ri, double *rj, double *cache);

void get_grid_spacing(double* dh, double* dh_inv, double* a, double* b, int* mesh);

void get_dm_to_dm_xyz_coeff(double* coeff, double* rij, int lmax, double* cache);
void dm_to_dm_xyz(double* dm_xyz, double* dm, int li, int lj, double* ri, double* rj, double* cache);
void dm_xyz_to_dm(double* dm_xyz, double* dm, int li, int lj, double* ri, double* rj, double* cache);
void dm_xyz_to_dm_ijk(double* dm_ijk, double* dm_xyz, double* dh, int topl);
void dm_ijk_to_dm_xyz(double* dm_ijk, double* dm_xyz, double* dh, int topl);
void get_dm_pgfpair(double* dm_pgf, double* dm_cart,
                    PGFPair* pgfpair, int* ish_bas, int* jsh_bas, int hermi);
int get_max_num_grid_orth(double* dh, double radius);
int get_max_num_grid_nonorth(double* dh_inv, double radius);
int get_max_num_grid_nonorth_tight(double* dh, double* dh_inv, double radius);

void dgemm_wrapper(const char transa, const char transb,
                   const int m, const int n, const int k,
                   const double alpha, const double* a, const int lda,
                   const double* b, const int ldb,
                   const double beta, double* c, const int ldc);

static inline void vadd(double* c, double* a, double* b)
{
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
    c[2] = a[2] + b[2];
}

static inline void vsub(double* c, double* a, double* b)
{
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];
}

static inline double vdot(double* a, double* b)
{
    double out;
    out = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    return out;
}

static inline void vscale(double* c, double alpha, double* a)
{
    c[0] = a[0] * alpha;
    c[1] = a[1] * alpha;
    c[2] = a[2] * alpha;
}

static inline double vnorm(double *a)
{
    double norm;
    norm = sqrt(vdot(a, a));
    return norm;
}

#endif
