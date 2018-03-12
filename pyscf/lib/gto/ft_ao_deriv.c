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
#include <complex.h>
#include "cint.h"
#include "gto/ft_ao.h"

/*
 * derivative over r on <i|
 */
void GTO_ft_nabla1i(double complex *f, double complex *g, int li, int lj,
                    double *Gv, size_t NGv, CINTEnvVars *envs)
{
        const int dj = envs->g_stride_j;
        const double ai2 = -2 * envs->ai;
        const size_t g_size = envs->g_size * NGv;
        double complex *gx = g;
        double complex *gy = g + g_size;
        double complex *gz = g + g_size * 2;
        double complex *fx = f;
        double complex *fy = f + g_size;
        double complex *fz = f + g_size * 2;
        int i, j, k, ptr;
        double vi;

        for (j = 0; j <= lj; j++) {
                ptr = dj * j;
                //f(...,0,...) = -2*ai*g(...,1,...)
                for (k = 0; k < NGv; k++) {
                        fx[ptr*NGv+k] = ai2 * gx[(ptr+1)*NGv+k];
                        fy[ptr*NGv+k] = ai2 * gy[(ptr+1)*NGv+k];
                        fz[ptr*NGv+k] = ai2 * gz[(ptr+1)*NGv+k];
                }
                //f(...,i,...) = i*g(...,i-1,...)-2*ai*g(...,i+1,...)
                for (vi = 1, i = ptr+1; i <= ptr+li; i++, vi+=1) {
                for (k = 0; k < NGv; k++) {
                        fx[i*NGv+k] = vi * gx[(i-1)*NGv+k] + ai2 * gx[(i+1)*NGv+k];
                        fy[i*NGv+k] = vi * gy[(i-1)*NGv+k] + ai2 * gy[(i+1)*NGv+k];
                        fz[i*NGv+k] = vi * gz[(i-1)*NGv+k] + ai2 * gz[(i+1)*NGv+k];
                } }
        }
}

/*
 * derivative over r on |j>
 */
void GTO_ft_nabla1j(double complex *f, double complex *g, int li, int lj,
                    double *Gv, size_t NGv, CINTEnvVars *envs)
{
        const int dj = envs->g_stride_j;
        const double aj2 = -2 * envs->aj;
        const size_t g_size = envs->g_size * NGv;
        double complex *gx = g;
        double complex *gy = g + g_size;
        double complex *gz = g + g_size * 2;
        double complex *fx = f;
        double complex *fy = f + g_size;
        double complex *fz = f + g_size * 2;
        int i, j, k, ptr;
        double vj;

        //f(...,0,...) = -2*aj*g(...,1,...)
        for (i = 0; i <= li; i++) {
                for (k = 0; k < NGv; k++) {
                        fx[i*NGv+k] = aj2 * gx[(i+dj)*NGv+k];
                        fy[i*NGv+k] = aj2 * gy[(i+dj)*NGv+k];
                        fz[i*NGv+k] = aj2 * gz[(i+dj)*NGv+k];
                }
        }
        //f(...,j,...) = j*g(...,j-1,...)-2*aj*g(...,j+1,...)
        for (j = 1; j <= lj; j++) {
                ptr = dj * j;
                vj = j;
                for (i = ptr; i <= ptr+li; i++) {
                for (k = 0; k < NGv; k++) {
                        fx[i*NGv+k] = vj * gx[(i-dj)*NGv+k] + aj2 * gx[(i+dj)*NGv+k];
                        fy[i*NGv+k] = vj * gy[(i-dj)*NGv+k] + aj2 * gy[(i+dj)*NGv+k];
                        fz[i*NGv+k] = vj * gz[(i-dj)*NGv+k] + aj2 * gz[(i+dj)*NGv+k];
                } }
        }
}

#define G1E_D_I(f, g, li, lj)   GTO_ft_nabla1i(f, g, li, lj, Gv, NGv, envs);
#define G1E_D_J(f, g, li, lj)   GTO_ft_nabla1j(f, g, li, lj, Gv, NGv, envs);

static void inner_prod_pdotp(double complex *g, double complex *gout,
                             int *idx, CINTEnvVars *envs,
                             double *Gv, size_t NGv, int empty)
{
        int nf = envs->nf;
        int ix, iy, iz, n, k;
        const size_t g_size = envs->g_size * NGv;
        double complex *g0 = g;
        double complex *gz = g  + g_size * 2;
        double complex *g1 = g0 + g_size * 3;
        double complex *g2 = g1 + g_size * 3;
        double complex *g3 = g2 + g_size * 3;
        double complex s;
        G1E_D_J(g1, g0, envs->i_l+1, envs->j_l);
        G1E_D_I(g2, g0, envs->i_l  , envs->j_l);
        G1E_D_I(g3, g1, envs->i_l  , envs->j_l);
        if (empty) {
                for (n = 0; n < nf; n++) {
                        ix = idx[0+n*3];
                        iy = idx[1+n*3];
                        iz = idx[2+n*3];
                        for (k = 0; k < NGv; k++) {
                                if (gz[k] != 0) {
                                        s  = g3[ix*NGv+k] * g0[iy*NGv+k] * g0[iz*NGv+k];
                                        s += g0[ix*NGv+k] * g3[iy*NGv+k] * g0[iz*NGv+k];
                                        s += g0[ix*NGv+k] * g0[iy*NGv+k] * g3[iz*NGv+k];
                                        gout[n*NGv+k] = s;
                                } else {
                                        gout[n*NGv+k] = 0;
                                }
                        }
                }
        } else {
                for (n = 0; n < nf; n++) {
                        ix = idx[0+n*3];
                        iy = idx[1+n*3];
                        iz = idx[2+n*3];
                        for (k = 0; k < NGv; k++) {
                                if (gz[k] != 0) {
                                        s  = g3[ix*NGv+k] * g0[iy*NGv+k] * g0[iz*NGv+k];
                                        s += g0[ix*NGv+k] * g3[iy*NGv+k] * g0[iz*NGv+k];
                                        s += g0[ix*NGv+k] * g0[iy*NGv+k] * g3[iz*NGv+k];
                                        gout[n*NGv+k] += s;
                                }
                        }
                }
        }
}

int GTO_ft_pdotp_cart(double complex *out, int *shls, int *dims,
                     int (*eval_aopair)(), void (*eval_gz)(), double complex fac,
                     double *Gv, double *b, int *gxyz, int *gs, int nGv,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        CINTEnvVars envs;
        int ng[] = {1, 1, 0, 0, 2, 1, 0, 1};
        GTO_ft_init1e_envs(&envs, ng, shls, atm, natm, bas, nbas, env);
        envs.f_gout = &inner_prod_pdotp;
        return GTO_ft_aopair_drv(out, dims, eval_aopair, eval_gz, &GTO_ft_c2s_cart,
                                 fac, Gv, b, gxyz, gs, nGv, &envs);
}

int GTO_ft_pdotp_sph(double complex *out, int *shls, int *dims,
                    int (*eval_aopair)(), void (*eval_gz)(), double complex fac,
                    double *Gv, double *b, int *gxyz, int *gs, int nGv,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        CINTEnvVars envs;
        int ng[] = {1, 1, 0, 0, 2, 1, 0, 1};
        GTO_ft_init1e_envs(&envs, ng, shls, atm, natm, bas, nbas, env);
        envs.f_gout = &inner_prod_pdotp;
        return GTO_ft_aopair_drv(out, dims, eval_aopair, eval_gz, &GTO_ft_c2s_sph,
                                 fac, Gv, b, gxyz, gs, nGv, &envs);
}


static void inner_prod_pxp(double complex *g, double complex *gout,
                           int *idx, CINTEnvVars *envs,
                           double *Gv, size_t NGv, int empty)
{
        int nf = envs->nf;
        int ix, iy, iz, n, k;
        const size_t g_size = envs->g_size * NGv;
        double complex *g0 = g;
        double complex *gz = g  + g_size * 2;
        double complex *g1 = g0 + g_size * 3;
        double complex *g2 = g1 + g_size * 3;
        double complex s[6];
        G1E_D_J(g1, g0, envs->i_l+1, envs->j_l);
        G1E_D_I(g2, g0, envs->i_l  , envs->j_l);
        if (empty) {
                for (n = 0; n < nf; n++) {
                        ix = idx[0+n*3];
                        iy = idx[1+n*3];
                        iz = idx[2+n*3];
                        for (k = 0; k < NGv; k++) {
                                if (gz[k] != 0) {
                                        s[0] = g2[ix*NGv+k] * g1[iy*NGv+k] * g0[iz*NGv+k];
                                        s[1] = g2[ix*NGv+k] * g0[iy*NGv+k] * g1[iz*NGv+k];
                                        s[2] = g1[ix*NGv+k] * g2[iy*NGv+k] * g0[iz*NGv+k];
                                        s[3] = g0[ix*NGv+k] * g2[iy*NGv+k] * g1[iz*NGv+k];
                                        s[4] = g1[ix*NGv+k] * g0[iy*NGv+k] * g2[iz*NGv+k];
                                        s[5] = g0[ix*NGv+k] * g1[iy*NGv+k] * g2[iz*NGv+k];
                                        gout[(n*3+0)*NGv+k] = s[3] - s[5];
                                        gout[(n*3+1)*NGv+k] = s[4] - s[1];
                                        gout[(n*3+2)*NGv+k] = s[0] - s[2];
                                } else {
                                        gout[(n*3+0)*NGv+k] = 0;
                                        gout[(n*3+1)*NGv+k] = 0;
                                        gout[(n*3+2)*NGv+k] = 0;
                                }
                        }
                }
        } else {
                for (n = 0; n < nf; n++) {
                        ix = idx[0+n*3];
                        iy = idx[1+n*3];
                        iz = idx[2+n*3];
                        for (k = 0; k < NGv; k++) {
                                if (gz[k] != 0) {
                                        s[0] = g2[ix*NGv+k] * g1[iy*NGv+k] * g0[iz*NGv+k];
                                        s[1] = g2[ix*NGv+k] * g0[iy*NGv+k] * g1[iz*NGv+k];
                                        s[2] = g1[ix*NGv+k] * g2[iy*NGv+k] * g0[iz*NGv+k];
                                        s[3] = g0[ix*NGv+k] * g2[iy*NGv+k] * g1[iz*NGv+k];
                                        s[4] = g1[ix*NGv+k] * g0[iy*NGv+k] * g2[iz*NGv+k];
                                        s[5] = g0[ix*NGv+k] * g1[iy*NGv+k] * g2[iz*NGv+k];
                                        gout[(n*3+0)*NGv+k] += s[3] - s[5];
                                        gout[(n*3+1)*NGv+k] += s[4] - s[1];
                                        gout[(n*3+2)*NGv+k] += s[0] - s[2];
                                }
                        }
                }
        }
}

int GTO_ft_pxp_cart(double complex *out, int *shls, int *dims,
                    int (*eval_aopair)(), void (*eval_gz)(), double complex fac,
                    double *Gv, double *b, int *gxyz, int *gs, int nGv,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        CINTEnvVars envs;
        int ng[] = {1, 1, 0, 0, 2, 1, 0, 3};
        GTO_ft_init1e_envs(&envs, ng, shls, atm, natm, bas, nbas, env);
        envs.f_gout = &inner_prod_pxp;
        return GTO_ft_aopair_drv(out, dims, eval_aopair, eval_gz, &GTO_ft_c2s_cart,
                                 fac, Gv, b, gxyz, gs, nGv, &envs);
}

int GTO_ft_pxp_sph(double complex *out, int *shls, int *dims,
                   int (*eval_aopair)(), void (*eval_gz)(), double complex fac,
                   double *Gv, double *b, int *gxyz, int *gs, int nGv,
                   int *atm, int natm, int *bas, int nbas, double *env)
{
        CINTEnvVars envs;
        int ng[] = {1, 1, 0, 0, 2, 1, 0, 3};
        GTO_ft_init1e_envs(&envs, ng, shls, atm, natm, bas, nbas, env);
        envs.f_gout = &inner_prod_pxp;
        return GTO_ft_aopair_drv(out, dims, eval_aopair, eval_gz, &GTO_ft_c2s_sph,
                                 fac, Gv, b, gxyz, gs, nGv, &envs);
}

