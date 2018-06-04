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
 * Numerical integration on uniform grids
 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <complex.h>
#include "config.h"
#include "cint.h"
#include "np_helper/np_helper.h"
#include "gto/ft_ao.h"
#include "gto/grid_ao_drv.h"
#include "vhf/fblas.h"

#define PLAIN           0
#define HERMITIAN       1
#define ANTIHERMI       2
#define SYMMETRIC       3

#define SQUARE(x)       (*(x) * *(x) + *(x+1) * *(x+1) + *(x+2) * *(x+2))

double CINTsquare_dist(const double *r1, const double *r2);
double CINTcommon_fac_sp(int l);
void c2s_sph_1e(double *opij, double *gctr, int *dims,
                CINTEnvVars *envs, double *cache);
void c2s_cart_1e(double *opij, double *gctr, int *dims,
                 CINTEnvVars *envs, double *cache);

int CINTinit_int1e_EnvVars(CINTEnvVars *envs, const int *ng, const int *shls,
                           const int *atm, const int natm,
                           const int *bas, const int nbas, const double *env);

static const int _LEN_CART[] = {
        1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136
};
static const int _CUM_LEN_CART[] = {
        1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364, 455, 560, 680, 816,
};

static double gto_rcut(double alpha, int l, double c, double log_prec)
{
        double r = 5.;
        double log_r = log(r);
        double log_c = 0;//log(abs(c));
        log_r = .5 * log(((l+2) * log_r + log_c - log_prec) / alpha);
        r = sqrt(((l+2) * log_r + log_c - log_prec) / alpha);
        return r;
}

static void _cartesian_components(double *xs_exp, int *img_slice, int *grid_slice,
                                  double *a, double xi, double xij, double aij,
                                  int periodic, int nx_per_cell, int topl,
                                  double edge0, double edge1, double *cache)
{
        int nimg0 = 0;
        int nimg1 = 1;
        if (periodic) {
                nimg0 = (int)floor(edge0);
                nimg1 = (int)ceil (edge1);
        }

        int nx0 = (int)floor(edge0 * nx_per_cell);
        int nx1 = (int)ceil (edge1 * nx_per_cell);
        nx0 = MAX(nx0, nimg0 * nx_per_cell);
        nx1 = MIN(nx1, nimg1 * nx_per_cell);
        img_slice[0] = nimg0;
        img_slice[1] = nimg1;
        grid_slice[0] = nx0;
        grid_slice[1] = nx1;

        int nimg = nimg1 - nimg0;
        int nmx = nimg * nx_per_cell;

        double img0_x = *a * nimg0;
        double dx = *a / nx_per_cell;
        double x0xij = img0_x - xij;
        double x0xi = img0_x - xi;
        double x;

        double *gridx = cache;
        double *xs_all = cache + nimg * nx_per_cell;

        int i, m, l;
        double *px0;

        for (i = 0; i < nmx; i++) {
                x = x0xij + i * dx;
                xs_all[i] = exp(-aij * x * x);
        }

        for (i = 0; i < nmx; i++) {
                gridx[i] = x0xi + i * dx;
        }
        for (l = 1; l <= topl; l++) {
                px0 = xs_all + (l-1) * nmx;
                for (i = 0; i < nmx; i++) {
                        px0[nmx+i] = px0[i] * gridx[i];
                }
        }

        for (i = 0; i < nx_per_cell*(topl+1); i++) {
                xs_exp[i] = 0;
        }
        for (l = 0; l <= topl; l++) {
                for (m = 0; m < nimg; m++) {
                        px0 = xs_all + l * nmx + m*nx_per_cell;
                        for (i = 0; i < nx_per_cell; i++) {
                                xs_exp[l*nx_per_cell+i] += px0[i];
                        }
                }
        }
}

void GTOnumint_3d_orth(double *out, int floorl, int topl,
                       double ai, double aj, double fac, double log_prec,
                       int dimension, double *a, double *b, int *mesh,
                       double *weights, CINTEnvVars *envs, double *cache)
{
        const double aij = ai + aj;
        const double *ri = envs->ri;
        const double *rj = envs->rj;
        double rij[3];
        rij[0] = (ai * ri[0] + aj * rj[0]) / aij;
        rij[1] = (ai * ri[1] + aj * rj[1]) / aij;
        rij[2] = (ai * ri[2] + aj * rj[2]) / aij;

        double cutoff = gto_rcut(aij, topl, fac, log_prec);
        //double x_frac = rij[0] * b[0] + rij[1] * b[0] + rij[2] * b[0];
        //double y_frac = rij[0] * b[1] + rij[1] * b[1] + rij[2] * b[1];
        //double z_frac = rij[0] * b[2] + rij[1] * b[2] + rij[2] * b[2];
        //double xheights_inv = sqrt(SQUARE(b  ));
        //double yheights_inv = sqrt(SQUARE(b+3));
        //double zheights_inv = sqrt(SQUARE(b+6));
        double x_frac = rij[0] * b[0];
        double y_frac = rij[1] * b[4];
        double z_frac = rij[2] * b[8];
        double xheights_inv = b[0];
        double yheights_inv = b[4];
        double zheights_inv = b[8];
        double xedge0 = x_frac - cutoff * xheights_inv;
        double xedge1 = x_frac + cutoff * xheights_inv;
        double yedge0 = y_frac - cutoff * yheights_inv;
        double yedge1 = y_frac + cutoff * yheights_inv;
        double zedge0 = z_frac - cutoff * zheights_inv;
        double zedge1 = z_frac + cutoff * zheights_inv;

        double *xs_exp = cache;
        double *ys_exp = xs_exp + (topl+1) * mesh[0];
        double *zs_exp = ys_exp + (topl+1) * mesh[1];
        cache = zs_exp + (topl+1) * mesh[2];

        int img_slice[6];
        int grid_slice[6];
        _cartesian_components(xs_exp, img_slice, grid_slice, a, ri[0], rij[0], aij,
                              (dimension>=1), mesh[0], topl, xedge0, xedge1, cache);
        _cartesian_components(ys_exp, img_slice+2, grid_slice+2, a+4, ri[1], rij[1], aij,
                              (dimension>=2), mesh[1], topl, yedge0, yedge1, cache);
        _cartesian_components(zs_exp, img_slice+4, grid_slice+4, a+8, ri[2], rij[2], aij,
                              (dimension>=3), mesh[2], topl, zedge0, zedge1, cache);

        int nimgx0 = img_slice[0];
        int nimgx1 = img_slice[1];
        int nimgy0 = img_slice[2];
        int nimgy1 = img_slice[3];
        int nimgz0 = img_slice[4];
        int nimgz1 = img_slice[5];
        int nimgx = nimgx1 - nimgx0;
        int nimgy = nimgy1 - nimgy0;
        int nimgz = nimgz1 - nimgz0;

        int nx0 = grid_slice[0];
        int nx1 = grid_slice[1];
        int ny0 = grid_slice[2];
        int ny1 = grid_slice[3];
        int nz0 = grid_slice[4];
        int nz1 = grid_slice[5];
        int ngridx = nx1 - nx0;
        int ngridy = ny1 - ny0;

        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        int l1 = topl + 1;
        int xcols = mesh[1] * mesh[2];
        int ycols = mesh[2];
        double *weightyz = cache;
        double *weightz = weightyz + l1*xcols;

        int lx, ly, lz;
        int l, n, i;
        double *pz, *pweightz;
        double val;

        if (nimgx == 1) {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &ngridx,
                       &fac, weights, &xcols, xs_exp+nx0, mesh,
                       &D0, weightyz, &xcols);
        //TODO:} elif (nimgx == 2 && nx0_nx1_overlap) {
        } else {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, mesh,
                       &fac, weights, &xcols, xs_exp, mesh,
                       &D0, weightyz, &xcols);
        }

        // TODO: merge the loop of lz?
        if (nimgy == 1) {
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ngridy,
                               &D1, weightyz+lx*xcols, &ycols, ys_exp+ny0, mesh+1,
                               &D0, weightz+lx*l1*ycols, &ycols);
                }
        //TODO:} elif (nimgy == 2 && ny0_ny1_overlap) {
        } else {
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, mesh+1,
                               &D1, weightyz+lx*xcols, &ycols, ys_exp+ny0, mesh+1,
                               &D0, weightz+lx*l1*ycols, &ycols);
                }
        }

        if (nimgz == 1) {
                for (n = 0, l = floorl; l <= topl; l++) {
                for (lx = l; lx >= 0; lx--) {
                for (ly = l - lx; ly >= 0; ly--, n++) {
                        lz = l - lx - ly;
                        pz = zs_exp + lz * mesh[2];
                        pweightz = weightz + (lx * l1 + ly) * mesh[2];
                        val = 0;
                        for (i = nz0; i < nz1; i++) {
                                val += pweightz[i] * pz[i];
                        }
                        out[n] = val;
                } } }
        } else {
                for (n = 0, l = floorl; l <= topl; l++) {
                for (lx = l; lx >= 0; lx--) {
                for (ly = l - lx; ly >= 0; ly--, n++) {
                        lz = l - lx - ly;
                        pz = zs_exp + lz * mesh[2];
                        pweightz = weightz + (lx * l1 + ly) * mesh[2];
                        val = 0;
                        for (i = 0; i < mesh[2]; i++) {
                                val += pweightz[i] * pz[i];
                        }
                        out[n] = val;
                } } }
        }
}

void GTOnumint_3d_nonorth(double *out, int floorl, int topl,
                          double ai, double aj, double fac, double log_prec,
                          int dimension, double *a, double *b, int *mesh,
                          double *weights, CINTEnvVars *envs, double *cache)
{
        fprintf(stderr, "GTOnumint_3d_nonorth not available\n");
        exit(1);
}

static void plain_prim_to_ctr(double *gc, const size_t nf, double *gp,
                              const int nprim, const int nctr,
                              const double *coeff, int empty)
{
        size_t n, i;
        double c;

        if (empty) {
                for (n = 0; n < nctr; n++) {
                        c = coeff[nprim*n];
                        for (i = 0; i < nf; i++) {
                                gc[i] = gp[i] * c;
                        }
                        gc += nf;
                }
        } else {
                for (n = 0; n < nctr; n++) {
                        c = coeff[nprim*n];
                        if (c != 0) {
                                for (i = 0; i < nf; i++) {
                                        gc[i] += gp[i] * c;
                                }
                        }
                        gc += nf;
                }
        }
}

int GTOnumint1e_loop(double *out, double fac, double log_prec,
                     int dimension, double *a, double *b,
                     int *mesh, double *weights,
                     CINTEnvVars *envs, double *cache)
{
        const int *shls  = envs->shls;
        const int *bas = envs->bas;
        double *env = envs->env;
        const int i_sh = shls[0];
        const int j_sh = shls[1];
        const int i_l = envs->i_l;
        const int j_l = envs->j_l;
        const int i_ctr = envs->x_ctr[0];
        const int j_ctr = envs->x_ctr[1];
        const int i_prim = bas(NPRIM_OF, i_sh);
        const int j_prim = bas(NPRIM_OF, j_sh);
        const int nf = envs->nf;
        double *ri = envs->ri;
        double *rj = envs->rj;
        double *ai = env + bas(PTR_EXP, i_sh);
        double *aj = env + bas(PTR_EXP, j_sh);
        double *ci = env + bas(PTR_COEFF, i_sh);
        double *cj = env + bas(PTR_COEFF, j_sh);
        double fac1i;
        double aij, eij;
        int ip, jp, n;
        int empty[3] = {1, 1, 1};
        int *jempty = empty + 0;
        int *iempty = empty + 1;
        //int *gempty = empty + 2;
        const int offset_g1d = _CUM_LEN_CART[i_l] - _LEN_CART[i_l];
        const int len_g1d = _CUM_LEN_CART[i_l+j_l] - offset_g1d;
        const size_t leni = len_g1d * i_ctr;
        const size_t lenj = len_g1d * i_ctr * j_ctr;
        double *gctrj = malloc(sizeof(double)*(lenj + leni +  len_g1d + nf+10000));
        double *gctri = gctrj + lenj;
        double *g = gctri + leni;

        double rrij = CINTsquare_dist(ri, rj);
        double fac1 = fac * CINTcommon_fac_sp(i_l) * CINTcommon_fac_sp(j_l);
        double logc;

        *jempty = 1;
        for (jp = 0; jp < j_prim; jp++) {
                *iempty = 1;
                for (ip = 0; ip < i_prim; ip++) {
                        aij = ai[ip] + aj[jp];
                        eij = (ai[ip] * aj[jp] / aij) * rrij;
                        if (eij > EXPCUTOFF) {
                                continue;
                        }

                        fac1i = fac1 * exp(-eij);
                        logc = log_prec - 10; // TODO: estimate log(c)
                        GTOnumint_3d_orth(g, i_l, i_l+j_l, ai[ip], aj[jp],
                                          fac1i, logc, dimension,
                                          a, b, mesh, weights, envs, cache);
                        plain_prim_to_ctr(gctri, len_g1d, g,
                                          i_prim, i_ctr, ci+ip, *iempty);
                        *iempty = 0;
                }
                if (!*iempty) {
                        plain_prim_to_ctr(gctrj, i_ctr*len_g1d, gctri,
                                          j_prim, j_ctr, cj+jp, *jempty);
                        *jempty = 0;
                }
        }

        if (!*jempty) {
                for (n = 0; n < i_ctr*j_ctr; n++) {
                        GTOplain_vrr2d(out+n*nf, gctrj+n*len_g1d, cache, envs);
                }
        }
        free(gctrj);

        return !*jempty;
}

int GTO_numint1e_drv(double *out, int *dims, void (*f_c2s)(),
                     double fac, double log_prec,
                     int dimension, double *a, double *b,
                     int *mesh, double *weights, CINTEnvVars *envs)
{
        const int i_ctr = envs->x_ctr[0];
        const int j_ctr = envs->x_ctr[1];
        const int n_comp = envs->ncomp_e1 * envs->ncomp_tensor;
        const size_t nc = envs->nf * i_ctr * j_ctr;
        double *gctr = malloc(sizeof(double) * (nc * n_comp + 1000000));
        double *cache = gctr + nc * n_comp;
        size_t n;

        int has_value = GTOnumint1e_loop(gctr, fac, log_prec, dimension,
                                         a, b, mesh, weights, envs, cache);
        if (!has_value) {
                for (n = 0; n < nc*n_comp; n++) { gctr[n] = 0; }
        }

        int counts[4];
        if (f_c2s == &c2s_sph_1e) {
                counts[0] = (envs->i_l*2+1) * i_ctr;
                counts[1] = (envs->j_l*2+1) * j_ctr;
        } else { // f_c2s == &GTO_ft_c2s_cart
                counts[0] = envs->nfi * i_ctr;
                counts[1] = envs->nfj * j_ctr;
        }
        if (dims == NULL) {
                dims = counts;
        }
        size_t nout = dims[0] * dims[1];

        if (has_value) {
                for (n = 0; n < n_comp; n++) {
                        (*f_c2s)(out+nout*n, gctr+nc*n, dims, envs, cache);
                }
        }
        free(gctr);
        return has_value;
}

int NUMINT1e_ovlp_cart(double *out, int *dims, int *shls,
                      int *atm, int natm, int *bas, int nbas, double *env,
                      double log_prec, int dimension, double *a, double *b,
                      int *mesh, double *weights)
{
        CINTEnvVars envs;
        int ng[] = {0, 0, 0, 0, 0, 1, 0, 1};
        CINTinit_int1e_EnvVars(&envs, ng, shls, atm, natm, bas, nbas, env);
        return GTO_numint1e_drv(out, dims, &c2s_cart_1e, 1., log_prec,
                                dimension, a, b, mesh, weights, &envs);
}

int NUMINT1e_ovlp_sph(double *out, int *dims, int *shls,
                      int *atm, int natm, int *bas, int nbas, double *env,
                      double log_prec, int dimension, double *a, double *b,
                      int *mesh, double *weights)
{
        CINTEnvVars envs;
        int ng[] = {0, 0, 0, 0, 0, 1, 0, 1};
        CINTinit_int1e_EnvVars(&envs, ng, shls, atm, natm, bas, nbas, env);
        return GTO_numint1e_drv(out, dims, &c2s_sph_1e, 1., log_prec,
                                dimension, a, b, mesh, weights, &envs);
}

void NUMINT1e_fill2c(int (*intor)(), double *mat,
                     int comp, int hermi, int *shls_slice, int *ao_loc,
//?double complex *out, int nkpts, int comp, int nimgs,
//?double *Ls, double complex *expkL,
                     double log_prec, int dimension,
                     double *a, double *b, int *mesh, double *weights,
                     int *atm, int natm, int *bas, int nbas, double *env, int nenv)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
#pragma omp parallel default(none) \
        shared(intor, mat, comp, hermi, ao_loc, \
               log_prec, dimension, a, b, mesh, weights, \
               atm, natm, bas, nbas, env)
{
        int dims[] = {naoi, naoj};
        int ish, jsh, ij, i0, j0;
        int shls[2];
#pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                if (hermi != PLAIN && ish > jsh) {
                        // fill up only upper triangle of F-array
                        continue;
                }

                ish += ish0;
                jsh += jsh0;
                shls[0] = ish;
                shls[1] = jsh;
                i0 = ao_loc[ish] - ao_loc[ish0];
                j0 = ao_loc[jsh] - ao_loc[jsh0];
                (*intor)(mat+j0*naoi+i0, dims, shls, atm, natm, bas, nbas, env,
                         log_prec, dimension, a, b, mesh, weights);
        }
}
        if (hermi != PLAIN) { // lower triangle of F-array
                int ic;
                for (ic = 0; ic < comp; ic++) {
                        NPdsymm_triu(naoi, mat+ic*naoi*naoi, hermi);
                }
        }
}
