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

 *
 * Author: Xing Zhang <zhangxing.nju@gmail.com>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include "config.h"
#include "dft/multigrid.h"
#include "dft/grid_common.h"
#include "vhf/fblas.h"

#define MAX_THREADS     256

static void transform_dm(double* dm_cart, double* dm,
                         double* ish_contr_coeff, double* jsh_contr_coeff,
                         int* ish_ao_loc, int* jsh_ao_loc,
                         int* ish_bas, int* jsh_bas, int ish, int jsh,
                         int ish0, int jsh0, int naoj, double* cache)
{
    int i0 = ish_ao_loc[ish] - ish_ao_loc[ish0];
    int i1 = ish_ao_loc[ish+1] - ish_ao_loc[ish0];
    int j0 = jsh_ao_loc[jsh] - jsh_ao_loc[jsh0];
    int j1 = jsh_ao_loc[jsh+1] - jsh_ao_loc[jsh0];

    int nrow = i1 - i0;
    int ncol = j1 - j0;
    double* pdm = dm + ((size_t)naoj) * i0 + j0;

    int l_i = ish_bas[ANG_OF+ish*BAS_SLOTS];
    int ncart_i = _LEN_CART[l_i];
    int nprim_i = ish_bas[NPRIM_OF+ish*BAS_SLOTS];
    int nao_i = nprim_i*ncart_i;
    int l_j = jsh_bas[ANG_OF+jsh*BAS_SLOTS];
    int ncart_j = _LEN_CART[l_j];
    int nprim_j = jsh_bas[NPRIM_OF+jsh*BAS_SLOTS];
    int nao_j = nprim_j*ncart_j;

    const char TRANS_T = 'T';
    const char TRANS_N = 'N';
    const double D1 = 1;
    const double D0 = 0;
    //einsum("pi,ij,qj->pq", coeff_i, dm, coeff_j)
    dgemm_wrapper(TRANS_T, TRANS_N, nao_j, nrow, ncol,
           D1, jsh_contr_coeff, ncol, pdm, naoj, D0, cache, nao_j);
    dgemm_wrapper(TRANS_N, TRANS_N, nao_j, nao_i, nrow,
           D1, cache, nao_j, ish_contr_coeff, nrow, D0, dm_cart, nao_j);
}


static void add_rho_submesh(double* rho, double* pqr,
                            int* mesh_lb, int* mesh_ub, int* submesh_lb,
                            const int* mesh, const int* submesh)
{
    const int x0 = mesh_lb[0];
    const int y0 = mesh_lb[1];
    const int z0 = mesh_lb[2];

    const int nx = mesh_ub[0] - x0;
    const int ny = mesh_ub[1] - y0;
    const int nz = mesh_ub[2] - z0;

    const int x0_sub = submesh_lb[0];
    const int y0_sub = submesh_lb[1];
    const int z0_sub = submesh_lb[2];

    const size_t mesh_yz = ((size_t) mesh[1]) * mesh[2];
    const size_t submesh_yz = ((size_t) submesh[1]) * submesh[2];

    int ix, iy, iz;
    for (ix = 0; ix < nx; ix++) {
        double* __restrict ptr_rho = rho + (ix + x0) * mesh_yz + y0 * mesh[2] + z0;
        double* __restrict ptr_pqr = pqr + (ix + x0_sub) * submesh_yz + y0_sub * submesh[2] + z0_sub;
        for (iy = 0; iy < ny; iy++) {
            //#pragma omp simd
            PRAGMA_OMP_SIMD
            for (iz = 0; iz < nz; iz++) {
                ptr_rho[iz] += ptr_pqr[iz];
            }
            ptr_rho += mesh[2];
            ptr_pqr += submesh[2];
        }
    }
}


static void map_rho_submesh_to_mesh(double* rho, double* pqr, int *bounds, int* mesh)
{
    const int nx = mesh[0];
    const int ny = mesh[1];
    const int nz = mesh[2];
    const int nx0 = bounds[0];
    const int ny0 = bounds[2];
    const int nz0 = bounds[4];
    const int ngridx = bounds[1] - nx0;
    const int ngridy = bounds[3] - ny0;
    const int ngridz = bounds[5] - nz0; 

    const int submesh[3] = {ngridx, ngridy, ngridz};
    int lb[3], ub[3];
    int ix, iy, iz;
    for (ix = 0; ix < ngridx;) {
        lb[0] = modulo(ix + nx0, nx);
        ub[0] = get_upper_bound(lb[0], nx, ix, ngridx);
        for (iy = 0; iy < ngridy;) {
            lb[1] = modulo(iy + ny0, ny);
            ub[1] = get_upper_bound(lb[1], ny, iy, ngridy);
            for (iz = 0; iz < ngridz;) {
                lb[2] = modulo(iz + nz0, nz);
                ub[2] = get_upper_bound(lb[2], nz, iz, ngridz);
                int lb_sub[3] = {ix, iy, iz};
                add_rho_submesh(rho, pqr, lb, ub, lb_sub, mesh, submesh);
                iz += ub[2] - lb[2];
            }
            iy += ub[1] - lb[1];
        }
        ix += ub[0] - lb[0];
    }
}


static void _orth_rho(double *rho, double *dm_xyz,
                      double fac, int topl,
                      int *mesh, int *grid_slice,
                      double *xs_exp, double *ys_exp, double *zs_exp,
                      double *cache)
{
    const int l1 = topl + 1;
    const int l1l1 = l1 * l1;
    const int nx0 = grid_slice[0];
    const int nx1 = grid_slice[1];
    const int ny0 = grid_slice[2];
    const int ny1 = grid_slice[3];
    const int nz0 = grid_slice[4];
    const int nz1 = grid_slice[5];
    const int ngridx = nx1 - nx0;
    const int ngridy = ny1 - ny0;
    const int ngridz = nz1 - nz0;
    if (ngridx == 0 || ngridy == 0 || ngridz == 0) {
        return;
    }

    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const double D0 = 0;
    const double D1 = 1;
    const int xcols = ngridy * ngridz;
    double *xyr = cache;
    double *xqr = xyr + l1l1 * ngridz;
    double *pqr = xqr + l1 * xcols;
    int l;

    dgemm_wrapper(TRANS_N, TRANS_N, ngridz, l1l1, l1,
                  fac, zs_exp, ngridz, dm_xyz, l1,
                  D0, xyr, ngridz);
    for (l = 0; l <= topl; l++) {
        dgemm_wrapper(TRANS_N, TRANS_T, ngridz, ngridy, l1,
                      D1, xyr+l*l1*ngridz, ngridz, ys_exp, ngridy,
                      D0, xqr+l*xcols, ngridz);
    }
    dgemm_wrapper(TRANS_N, TRANS_T, xcols, ngridx, l1,
                  D1, xqr, xcols, xs_exp, ngridx,
                  D0, pqr, xcols);

    map_rho_submesh_to_mesh(rho, pqr, grid_slice, mesh);
}


void make_rho_lda_orth(double *rho, double *dm, int comp,
                       int li, int lj, double ai, double aj,
                       double *ri, double *rj, double fac, double cutoff,
                       int dimension, double *dh, double *dh_inv,
                       int *mesh, double *cache)
{
    int topl = li + lj;
    int l1 = topl + 1;
    int l1l1l1 = l1 * l1 * l1;
    int grid_slice[6];
    double *xs_exp, *ys_exp, *zs_exp;
    int data_size = init_orth_data(&xs_exp, &ys_exp, &zs_exp,
                                   grid_slice, dh, mesh, topl, cutoff,
                                   ai, aj, ri, rj, cache);

    if (data_size == 0) {
        return;
    }
    cache += data_size;

    double *dm_xyz = cache;
    cache += l1l1l1;
    memset(dm_xyz, 0, l1l1l1*sizeof(double));

    dm_to_dm_xyz(dm_xyz, dm, li, lj, ri, rj, cache);

    _orth_rho(rho, dm_xyz, fac, topl, mesh, grid_slice,
              xs_exp, ys_exp, zs_exp, cache);
}


static void _nonorth_rho(double *rho, double *dm_ijk,
                         double fac, int topl,
                         int *mesh, int *grid_slice,
                         double *xs_exp, double *ys_exp, double *zs_exp,
                         double* exp_corr, double *cache)
{
    const int l1 = topl + 1;
    const int l1l1 = l1 * l1;
    const int nx0 = grid_slice[0], nx1 = grid_slice[1];
    const int ny0 = grid_slice[2], ny1 = grid_slice[3];
    const int nz0 = grid_slice[4], nz1 = grid_slice[5];
    const int ngridx = nx1 - nx0;
    const int ngridy = ny1 - ny0;
    const int ngridz = nz1 - nz0;
    if (ngridx == 0 || ngridy == 0 || ngridz == 0) return;

    const size_t ng_xy  = (size_t)ngridx * ngridy;
    const size_t ng_yz  = (size_t)ngridy * ngridz;

    double *xyr = cache;
    double *xqr = xyr + l1l1 * ngridz;
    double *pqr = xqr + l1 * ng_yz;

    int l, i, j, k;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const double D0 = 0;
    const double D1 = 1;

    dgemm_wrapper(TRANS_N, TRANS_N, ngridz, l1l1, l1,
                  fac, zs_exp, ngridz, dm_ijk, l1,
                  D0, xyr, ngridz);

    for (l = 0; l <= topl; l++) {
        dgemm_wrapper(TRANS_N, TRANS_T, ngridz, ngridy, l1,
                      D1, xyr + l * l1 * ngridz, ngridz, ys_exp, ngridy,
                      D0, xqr + l * ng_yz, ngridz);
    }

    double *exp_corr_ij = exp_corr;
    double *exp_corr_jk = exp_corr_ij + ng_xy;
    double *exp_corr_ik = exp_corr_jk + ng_yz;

    double *ptr_xqr = xqr;
    for (l = 0; l <= topl; l++) {
        double *ptmp = exp_corr_jk;
        for (j = 0; j < ngridy; j++) {
            for (k = 0; k < ngridz; k++) {
                ptr_xqr[k] *= ptmp[k];
            }
            ptr_xqr += ngridz;
            ptmp += ngridz;
        }
    }

    dgemm_wrapper(TRANS_N, TRANS_T, ng_yz, ngridx, l1,
                  D1, xqr, ng_yz, xs_exp, ngridx,
                  D0, pqr, ng_yz);

    double *ptr_pqr = pqr;
    for (i = 0; i < ngridx; i++) {
        for (j = 0; j < ngridy; j++) {
            double tmp = exp_corr_ij[j];
            for (k = 0; k < ngridz; k++) {
                ptr_pqr[k] *= tmp * exp_corr_ik[k];
            }
            ptr_pqr += ngridz;
        }
        exp_corr_ij += ngridy;
        exp_corr_ik += ngridz;
    }

    map_rho_submesh_to_mesh(rho, pqr, grid_slice, mesh);
}


void make_rho_lda_nonorth(double *rho, double *dm, int comp,
                          int li, int lj, double ai, double aj,
                          double *ri, double *rj, double fac, double cutoff,
                          int dimension, double *dh, double *dh_inv,
                          int *mesh, double *cache)
{
    int topl = li + lj;
    int l1 = topl + 1;
    int l1l1l1 = l1 * l1 * l1;
    int grid_slice[6];
    double *xs_exp, *ys_exp, *zs_exp, *exp_corr;
    size_t data_size = init_nonorth_data(&xs_exp, &ys_exp, &zs_exp, &exp_corr,
                                         grid_slice, dh, dh_inv, mesh, topl, cutoff,
                                         ai, aj, ri, rj, cache);

    if (data_size == 0) {
        return;
    }
    cache += data_size;

    double *dm_ijk = cache;
    memset(dm_ijk, 0, l1l1l1 * sizeof(double));

    double *dm_xyz = dm_ijk + l1l1l1;
    memset(dm_xyz, 0, l1l1l1 * sizeof(double));

    cache = dm_xyz + l1l1l1;
    dm_to_dm_xyz(dm_xyz, dm, li, lj, ri, rj, cache);
    dm_xyz_to_dm_ijk(dm_ijk, dm_xyz, dh, topl);

    cache = dm_ijk + l1l1l1;
    _nonorth_rho(rho, dm_ijk, fac, topl, mesh, grid_slice,
                 xs_exp, ys_exp, zs_exp, exp_corr, cache);
}


static void _apply_rho(void (*eval_rho)(), double *rho, double *dm,
                       PGFPair *pgfpair, int comp, int dimension,
                       double *dh, double *dh_inv, int *mesh,
                       double *ish_gto_norm, double *jsh_gto_norm,
                       int *ish_atm, int *ish_bas, double *ish_env,
                       int *jsh_atm, int *jsh_bas, double *jsh_env,
                       double *Ls, double *cache)
{
    int ish = pgfpair->ish;
    int jsh = pgfpair->jsh;
    int ipgf = pgfpair->ipgf;
    int jpgf = pgfpair->jpgf;
    int iL = pgfpair->iL;
    double cutoff = pgfpair->radius;

    double *ri = ish_env + ish_atm[PTR_COORD+ish_bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
    double *rj = jsh_env + jsh_atm[PTR_COORD+jsh_bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
    double *rL = Ls + iL*3;
    double rjL[3];
    rjL[0] = rj[0] + rL[0];
    rjL[1] = rj[1] + rL[1];
    rjL[2] = rj[2] + rL[2];

    const int li = ish_bas[ANG_OF+ish*BAS_SLOTS];
    const int lj = jsh_bas[ANG_OF+jsh*BAS_SLOTS];
    double ai = ish_env[ish_bas[PTR_EXP+ish*BAS_SLOTS]+ipgf];
    double aj = jsh_env[jsh_bas[PTR_EXP+jsh*BAS_SLOTS]+jpgf];
    double ci = ish_gto_norm[ipgf];
    double cj = jsh_gto_norm[jpgf];
    double aij = ai + aj;
    double rrij = CINTsquare_dist(ri, rjL);
    double eij = (ai * aj / aij) * rrij;
    if (eij > EIJCUTOFF) {
        return;
    }
    double fac = exp(-eij) * ci * cj * CINTcommon_fac_sp(li) * CINTcommon_fac_sp(lj);
    if (fac < ish_env[PTR_EXPDROP] && fac < jsh_env[PTR_EXPDROP]) {
        return;
    }

    (*eval_rho)(rho, dm, comp, li, lj, ai, aj, ri, rjL,
                fac, cutoff, dimension, dh, dh_inv, mesh, cache);
}


static size_t _orth_rho_cache_size(int l, int nprim, int nctr, int* mesh, double radius, double* dh)
{
    size_t size = 0;
    size_t mesh_size = ((size_t)mesh[0]) * mesh[1] * mesh[2];
    size_t nmx = get_max_num_grid_orth(dh, radius);
    int l1 = 2 * l + 1;
    int l1l1 = l1 * l1;
    int max_mesh = MAX(MAX(mesh[0], mesh[1]), mesh[2]);
    size += (nprim * _LEN_CART[l]) * (nprim * _LEN_CART[l]); // dm_cart
    size += _LEN_CART[l]*_LEN_CART[l]; // dm_pgf
    size += nctr * _LEN_CART[l] * nprim * _LEN_CART[l]; // transform_dm
    size += l1 * (mesh[0] + mesh[1] + mesh[2]); // xs_exp, ys_exp, zs_exp
    size += l1l1 * l1; // dm_xyz
    size += 3 * (_LEN_CART[l] + l1); // dm_to_dm_xyz

    size_t size_orth_components = l1 * nmx + nmx; // orth_components
    size_t size_orth_rho = 0; // _orth_rho
    if (nmx < max_mesh) {
        size_orth_rho = l1l1*nmx + l1*nmx*nmx + nmx*nmx*nmx;
    } else {
        size_orth_rho = l1l1*mesh[2] + l1*mesh[1]*mesh[2] + mesh_size;
    }
    size += MAX(size_orth_rho, size_orth_components);
    //size += 1000000;
    //printf("Memory allocated per thread for make_rho: %ld MB.\n", (size+mesh_size)*sizeof(double) / 1000000);
    return size;
}


static size_t _nonorth_rho_cache_size(int l, int nprim, int nctr, int* mesh, double radius, double*dh, double* dh_inv)
{
    size_t size = 0;
    //size_t nmx = get_max_num_grid_nonorth(dh_inv, radius);
    size_t nmx = get_max_num_grid_nonorth_tight(dh, dh_inv, radius);
    size_t nmx2 = nmx * nmx;
    int l1 = 2 * l + 1;
    int l1l1 = l1 * l1;
    size += (nprim * _LEN_CART[l]) * (nprim * _LEN_CART[l]); // dm_cart
    size += _LEN_CART[l]*_LEN_CART[l]; // dm_pgf
    size += nctr * _LEN_CART[l] * nprim * _LEN_CART[l]; // transform_dm
    size += l1 * nmx * 3; // xs_exp, ys_exp, zs_exp
    size += nmx2 * 3; // exp_corr

    size_t tmp = nmx * 2; // exp_corr cache
    if (l > 0) {
        tmp += nmx; // xs_exp cache
    }
    size_t tmp1 = 3 * (_LEN_CART[l] + l1) + l1l1 * l1; // dm_to_dm_xyz cache
    tmp1 = MAX(tmp1, l1l1 * nmx + nmx2 * l1 + nmx2 * nmx); // _nonorth_rho
    tmp1 += l1l1 * l1; // dm_ijk
    size += MAX(tmp, tmp1);
    return size;
}


static size_t _rho_cache_size(int l, int nprim, int nctr, int* mesh, double radius,
                              double* dh, double* dh_inv, bool orth)
{
    if (orth) {
        return _orth_rho_cache_size(l, nprim, nctr, mesh, radius, dh);
    } else {
        return _nonorth_rho_cache_size(l, nprim, nctr, mesh, radius, dh, dh_inv);
    }
}


static size_t _orth_rho_core_cache_size(int* mesh, double radius, double* dh)
{
    size_t size = 0;
    size_t mesh_size = ((size_t)mesh[0]) * mesh[1] * mesh[2];
    size_t nmx = get_max_num_grid_orth(dh, radius);
    int l = 0;
    int l1 = 1;
    int l1l1 = l1 * l1;
    int max_mesh = MAX(MAX(mesh[0], mesh[1]), mesh[2]);
    size += l1 * (mesh[0] + mesh[1] + mesh[2]);
    size += l1l1 * l1;
    size += 3 * (_LEN_CART[l] + l1);

    size_t size_orth_components = l1 * nmx + nmx;
    size_t size_orth_rho = 0;
    if (nmx < max_mesh) {
        size_orth_rho = l1l1*nmx + l1*nmx*nmx + nmx*nmx*nmx;
    } else {
        size_orth_rho = l1l1*mesh[2] + l1*mesh[1]*mesh[2] + mesh_size;
    }
    size += MAX(size_orth_rho, size_orth_components);
    return size;
}


static size_t _nonorth_rho_core_cache_size(int* mesh, double radius, double* dh, double* dh_inv)
{
    size_t size = 0;
    //size_t nmx = get_max_num_grid_nonorth(dh_inv, radius);
    size_t nmx = get_max_num_grid_nonorth_tight(dh, dh_inv, radius);
    size_t nmx2 = nmx * nmx;
    int l = 0;
    int l1 = 1;
    int l1l1 = l1 * l1;
    size += l1 * nmx * 3;
    size += nmx2 * 3;

    size_t tmp = nmx * 2; // exp_corr cache
    size_t tmp1 = 3 * (_LEN_CART[l] + l1) + l1l1 * l1; // dm_to_dm_xyz cache
    tmp1 = MAX(tmp1, l1l1 * nmx + nmx2 * l1 + nmx2 * nmx); // _nonorth_rho
    tmp1 += l1l1 * l1; // dm_ijk
    size += MAX(tmp, tmp1);
    return size;
}


static size_t _rho_core_cache_size(int* mesh, double radius, double* dh, double* dh_inv, bool orth)
{
    if (orth) {
        return _orth_rho_core_cache_size(mesh, radius, dh);
    } else {
        return _nonorth_rho_core_cache_size(mesh, radius, dh, dh_inv);
    }
}


void grid_collocate_drv(void (*eval_rho)(), RS_Grid** rs_rho, double* dm, TaskList** task_list,
                        int comp, int hermi, int *shls_slice, int* ish_ao_loc, int* jsh_ao_loc,
                        int dimension, double* Ls, double* a, double* b,
                        int* ish_atm, int* ish_bas, double* ish_env,
                        int* jsh_atm, int* jsh_bas, double* jsh_env,
                        int cart, bool orth)
{
    TaskList* tl = *task_list;
    GridLevel_Info* gridlevel_info = tl->gridlevel_info;
    int nlevels = gridlevel_info->nlevels;

    assert (comp == (*rs_rho)->comp);

    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    //const int nijsh = nish * njsh;
    //const int naoi = ish_ao_loc[ish1] - ish_ao_loc[ish0];
    const int naoj = jsh_ao_loc[jsh1] - jsh_ao_loc[jsh0];

    double **gto_norm_i = (double**) malloc(sizeof(double*) * nish);
    double **cart2sph_coeff_i = (double**) malloc(sizeof(double*) * nish);
    get_cart2sph_coeff(cart2sph_coeff_i, gto_norm_i, ish0, ish1, ish_bas, ish_env, cart);
    double **gto_norm_j = gto_norm_i;
    double **cart2sph_coeff_j = cart2sph_coeff_i;
    if (hermi != 1) {
        gto_norm_j = (double**) malloc(sizeof(double*) * njsh);
        cart2sph_coeff_j = (double**) malloc(sizeof(double*) * njsh);
        get_cart2sph_coeff(cart2sph_coeff_j, gto_norm_j, jsh0, jsh1, jsh_bas, jsh_env, cart);
    }

    int ish_lmax = get_lmax(ish0, ish1, ish_bas);
    int jsh_lmax = ish_lmax;
    if (hermi != 1) {
        jsh_lmax = get_lmax(jsh0, jsh1, jsh_bas);
    }

    int ish_nprim_max = get_nprim_max(ish0, ish1, ish_bas);
    int jsh_nprim_max = ish_nprim_max;
    if (hermi != 1) {
        jsh_nprim_max = get_nprim_max(jsh0, jsh1, jsh_bas);
    }

    int ish_nctr_max = get_nctr_max(ish0, ish1, ish_bas);
    int jsh_nctr_max = ish_nctr_max;
    if (hermi != 1) {
        jsh_nctr_max = get_nctr_max(jsh0, jsh1, jsh_bas);
    } 

    int ilevel;
    int *mesh;
    double max_radius;
    double *rho, *rhobufs[MAX_THREADS];
    Task* task;
    size_t ntasks;
    PGFPair** pgfpairs;
    double dh[9], dh_inv[9];
    for (ilevel = 0; ilevel < nlevels; ilevel++) {
        task = (tl->tasks)[ilevel];
        ntasks = task->ntasks;
        if (ntasks <= 0) {
            continue;
        }
        pgfpairs = task->pgfpairs;
        max_radius = task->radius;

        rho = (*rs_rho)->data[ilevel];
        mesh = gridlevel_info->mesh + ilevel*3;

        get_grid_spacing(dh, dh_inv, a, b, mesh);

        int *task_loc;
        int nblock = get_task_loc(&task_loc, pgfpairs, ntasks, ish0, ish1, jsh0, jsh1, hermi);

        size_t cache_size = _rho_cache_size(MAX(ish_lmax,jsh_lmax), 
                                            MAX(ish_nprim_max, jsh_nprim_max),
                                            MAX(ish_nctr_max, jsh_nctr_max), mesh, max_radius,
                                            dh, dh_inv, orth);
        size_t ngrids = ((size_t)mesh[0]) * mesh[1] * mesh[2];

#pragma omp parallel
{
    PGFPair *pgfpair = NULL;
    int iblock, itask, ish, jsh;
    double *ptr_gto_norm_i, *ptr_gto_norm_j;
    double *cache0 = malloc(sizeof(double) * cache_size);
    double *dm_cart = cache0;
    double *dm_pgf = dm_cart + ish_nprim_max*_LEN_CART[ish_lmax]*jsh_nprim_max*_LEN_CART[jsh_lmax];
    double *cache = dm_pgf + _LEN_CART[ish_lmax]*_LEN_CART[jsh_lmax]; 

    int thread_id = omp_get_thread_num();
    double *rho_priv;
    if (thread_id == 0) {
        rho_priv = rho;
    } else {
        rho_priv = calloc(comp*ngrids, sizeof(double));
    }
    rhobufs[thread_id] = rho_priv;

    #pragma omp for schedule(dynamic)
    for (iblock = 0; iblock < nblock; iblock+=2) {
        itask = task_loc[iblock];
        pgfpair = pgfpairs[itask];
        ish = pgfpair->ish;
        jsh = pgfpair->jsh;
        ptr_gto_norm_i = gto_norm_i[ish];
        ptr_gto_norm_j = gto_norm_j[jsh];
        transform_dm(dm_cart, dm, cart2sph_coeff_i[ish],
                     cart2sph_coeff_j[jsh], ish_ao_loc, jsh_ao_loc,
                     ish_bas, jsh_bas, ish, jsh, ish0, jsh0, naoj, cache);
        for (; itask < task_loc[iblock+1]; itask++) {
            pgfpair = pgfpairs[itask];
            get_dm_pgfpair(dm_pgf, dm_cart, pgfpair, ish_bas, jsh_bas, hermi);
            _apply_rho(eval_rho, rho_priv, dm_pgf, pgfpair, comp, dimension, dh, dh_inv, mesh,
                       ptr_gto_norm_i, ptr_gto_norm_j, ish_atm, ish_bas, ish_env,
                       jsh_atm, jsh_bas, jsh_env, Ls, cache);
        }
    }

    free(cache0);
    NPomp_dsum_reduce_inplace(rhobufs, comp*ngrids);
    if (thread_id != 0) {
        free(rho_priv);
    }
}
    if (task_loc) {
        free(task_loc);
    }
    } // loop ilevel

    del_cart2sph_coeff(cart2sph_coeff_i, gto_norm_i, ish0, ish1);
    if (hermi != 1) {
        del_cart2sph_coeff(cart2sph_coeff_j, gto_norm_j, jsh0, jsh1);
    }
}


void build_core_density(void (*eval_rho)(), double* rho,
                        int* atm, int* bas, int nbas, double* env,
                        int* mesh, int dimension, double* a, double* b,
                        double max_radius, bool orth)
{
    size_t ngrids;
    ngrids = ((size_t) mesh[0]) * mesh[1] * mesh[2];

    double dh[9], dh_inv[9];
    get_grid_spacing(dh, dh_inv, a, b, mesh);

    double *rhobufs[MAX_THREADS];
    size_t cache_size = _rho_core_cache_size(mesh, max_radius, dh, dh_inv, orth);

#pragma omp parallel
{
    int ia, ib;
    double alpha, coeff, charge, rad, fac;
    double dm[] = {1.0};
    double *r0;
    double *cache = (double*) malloc(sizeof(double) * cache_size);

    int thread_id = omp_get_thread_num();
    double *rho_priv;
    if (thread_id == 0) {
        rho_priv = rho;
    } else {
        rho_priv = calloc(ngrids, sizeof(double));
    }
    rhobufs[thread_id] = rho_priv;

    #pragma omp for schedule(static)
    for (ib = 0; ib < nbas; ib++) {
        ia = bas[ib*BAS_SLOTS+ATOM_OF];
        alpha = env[bas[ib*BAS_SLOTS+PTR_EXP]];
        coeff = env[bas[ib*BAS_SLOTS+PTR_COEFF]];
        charge = (double)atm[ia*ATM_SLOTS+CHARGE_OF];
        r0 = env + atm[ia*ATM_SLOTS+PTR_COORD];
        fac = -charge * coeff;
        rad = env[atm[ia*ATM_SLOTS+PTR_RADIUS]];
        if (rad > 1e-15) {
            eval_rho(rho_priv, dm, 1, 0, 0, alpha, 0., r0, r0,
                     fac, rad, dimension, dh, dh_inv, mesh, cache);
        }
    }
    free(cache);

    NPomp_dsum_reduce_inplace(rhobufs, ngrids);
    if (thread_id != 0) {
        free(rho_priv);
    }
}
}

