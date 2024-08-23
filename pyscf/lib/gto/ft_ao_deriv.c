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
#include <complex.h>
#include "cint.h"
#include "gto/ft_ao.h"

/*
 * derivative over r on <i|
 */
void GTO_ft_nabla1i(double *f, double *g, int li, int lj, FTEnvVars *envs)
{
        int dj = envs->g_stride_j;
        int bs = envs->block_size;
        size_t g_size = envs->g_size * bs;
        double ai2 = -2 * envs->ai[0];
        double *gxR = g;
        double *gyR = gxR + g_size;
        double *gzR = gyR + g_size;
        double *gxI = gzR + g_size;
        double *gyI = gxI + g_size;
        double *gzI = gyI + g_size;
        double *fxR = f;
        double *fyR = fxR + g_size;
        double *fzR = fyR + g_size;
        double *fxI = fzR + g_size;
        double *fyI = fxI + g_size;
        double *fzI = fyI + g_size;
        int i, j, k, ptr;
        double vi;

        for (j = 0; j <= lj; j++) {
                ptr = dj * j;
                //f(...,0,...) = -2*ai*g(...,1,...)
#pragma GCC ivdep
                for (k = 0; k < bs; k++) {
                        fxR[ptr*bs+k] = ai2 * gxR[(ptr+1)*bs+k];
                        fxI[ptr*bs+k] = ai2 * gxI[(ptr+1)*bs+k];
                        fyR[ptr*bs+k] = ai2 * gyR[(ptr+1)*bs+k];
                        fyI[ptr*bs+k] = ai2 * gyI[(ptr+1)*bs+k];
                        fzR[ptr*bs+k] = ai2 * gzR[(ptr+1)*bs+k];
                        fzI[ptr*bs+k] = ai2 * gzI[(ptr+1)*bs+k];
                }
                //f(...,i,...) = i*g(...,i-1,...)-2*ai*g(...,i+1,...)
                for (vi = 1, i = ptr+1; i <= ptr+li; i++, vi+=1) {
#pragma GCC ivdep
                for (k = 0; k < bs; k++) {
                        fxR[i*bs+k] = vi * gxR[(i-1)*bs+k] + ai2 * gxR[(i+1)*bs+k];
                        fxI[i*bs+k] = vi * gxI[(i-1)*bs+k] + ai2 * gxI[(i+1)*bs+k];
                        fyR[i*bs+k] = vi * gyR[(i-1)*bs+k] + ai2 * gyR[(i+1)*bs+k];
                        fyI[i*bs+k] = vi * gyI[(i-1)*bs+k] + ai2 * gyI[(i+1)*bs+k];
                        fzR[i*bs+k] = vi * gzR[(i-1)*bs+k] + ai2 * gzR[(i+1)*bs+k];
                        fzI[i*bs+k] = vi * gzI[(i-1)*bs+k] + ai2 * gzI[(i+1)*bs+k];
                } }
        }
}

/*
 * derivative over r on |j>
 */
void GTO_ft_nabla1j(double *f, double *g, int li, int lj, FTEnvVars *envs)
{
        int dj = envs->g_stride_j;
        int bs = envs->block_size;
        size_t g_size = envs->g_size * bs;
        double aj2 = -2 * envs->aj[0];
        double *gxR = g;
        double *gyR = gxR + g_size;
        double *gzR = gyR + g_size;
        double *gxI = gzR + g_size;
        double *gyI = gxI + g_size;
        double *gzI = gyI + g_size;
        double *fxR = f;
        double *fyR = fxR + g_size;
        double *fzR = fyR + g_size;
        double *fxI = fzR + g_size;
        double *fyI = fxI + g_size;
        double *fzI = fyI + g_size;
        int i, j, k, ptr;

        //f(...,0,...) = -2*aj*g(...,1,...)
        for (i = 0; i <= li; i++) {
#pragma GCC ivdep
                for (k = 0; k < bs; k++) {
                        fxR[i*bs+k] = aj2 * gxR[(i+dj)*bs+k];
                        fxI[i*bs+k] = aj2 * gxI[(i+dj)*bs+k];
                        fyR[i*bs+k] = aj2 * gyR[(i+dj)*bs+k];
                        fyI[i*bs+k] = aj2 * gyI[(i+dj)*bs+k];
                        fzR[i*bs+k] = aj2 * gzR[(i+dj)*bs+k];
                        fzI[i*bs+k] = aj2 * gzI[(i+dj)*bs+k];
                }
        }
        //f(...,j,...) = j*g(...,j-1,...)-2*aj*g(...,j+1,...)
        for (j = 1; j <= lj; j++) {
                ptr = dj * j;
                for (i = ptr; i <= ptr+li; i++) {
#pragma GCC ivdep
                for (k = 0; k < bs; k++) {
                        fxR[i*bs+k] = j * gxR[(i-dj)*bs+k] + aj2 * gxR[(i+dj)*bs+k];
                        fxI[i*bs+k] = j * gxI[(i-dj)*bs+k] + aj2 * gxI[(i+dj)*bs+k];
                        fyR[i*bs+k] = j * gyR[(i-dj)*bs+k] + aj2 * gyR[(i+dj)*bs+k];
                        fyI[i*bs+k] = j * gyI[(i-dj)*bs+k] + aj2 * gyI[(i+dj)*bs+k];
                        fzR[i*bs+k] = j * gzR[(i-dj)*bs+k] + aj2 * gzR[(i+dj)*bs+k];
                        fzI[i*bs+k] = j * gzI[(i-dj)*bs+k] + aj2 * gzI[(i+dj)*bs+k];
                } }
        }
}

#define G1E_D_I(f, g, li, lj)   GTO_ft_nabla1i(f##R, g##R, li, lj, envs);
#define G1E_D_J(f, g, li, lj)   GTO_ft_nabla1j(f##R, g##R, li, lj, envs);

static void inner_prod_pdotp(double *gout, double *g, int *idx, FTEnvVars *envs, int empty)
{
        int nf = envs->nf;
        int bs = envs->block_size;
        size_t g_size = envs->g_size * bs;
        double *g0R = g;
        double *g0I = g0R + g_size * 3;
        double *g1R = g0I + g_size * 3;
        double *g1I = g1R + g_size * 3;
        double *g2R = g1I + g_size * 3;
        double *g2I = g2R + g_size * 3;
        double *g3R = g2I + g_size * 3;
        double *g3I = g3R + g_size * 3;
        double *goutR = gout;
        double *goutI = gout + nf * bs;
        double xyR, xyI, sR, sI;
        int ix, iy, iz, n, k;

        G1E_D_J(g1, g0, envs->i_l+1, envs->j_l);
        G1E_D_I(g2, g0, envs->i_l  , envs->j_l);
        G1E_D_I(g3, g1, envs->i_l  , envs->j_l);
        if (empty) {
                for (n = 0; n < nf; n++) {
                        ix = idx[0+n*3];
                        iy = idx[1+n*3];
                        iz = idx[2+n*3];
#pragma GCC ivdep
                        for (k = 0; k < bs; k++) {
                                ZMUL(sR, sI, g3, g0, g0);
                                ZMAD(sR, sI, g0, g3, g0);
                                ZMAD(sR, sI, g0, g0, g3);
                                goutR[n*bs+k] = sR;
                                goutI[n*bs+k] = sI;
                        }
                }
        } else {
                for (n = 0; n < nf; n++) {
                        ix = idx[0+n*3];
                        iy = idx[1+n*3];
                        iz = idx[2+n*3];
#pragma GCC ivdep
                        for (k = 0; k < bs; k++) {
                                ZMUL(sR, sI, g3, g0, g0);
                                ZMAD(sR, sI, g0, g3, g0);
                                ZMAD(sR, sI, g0, g0, g3);
                                goutR[n*bs+k] += sR;
                                goutI[n*bs+k] += sI;
                        }
                }
        }
}

int GTO_ft_pdotp_cart(double *outR, double *outI, int *shls, int *dims,
                      FPtr_eval_gz eval_gz, double complex fac,
                      double *Gv, double *b, int *gxyz, int *gs, int nGv, int block_size,
                      int *atm, int natm, int *bas, int nbas, double *env, double *cache)
{
        FTEnvVars envs;
        int ng[] = {1, 1, 0, 0, 2, 1, 0, 1};
        GTO_ft_init1e_envs(&envs, ng, shls, fac, Gv, b, gxyz, gs, nGv, block_size,
                           atm, natm, bas, nbas, env);
        envs.f_gout = &inner_prod_pdotp;
        return GTO_ft_aopair_drv(outR, outI, dims, eval_gz, cache, &GTO_ft_c2s_cart, &envs);
}

int GTO_ft_pdotp_sph(double *outR, double *outI, int *shls, int *dims,
                     FPtr_eval_gz eval_gz, double complex fac,
                     double *Gv, double *b, int *gxyz, int *gs, int nGv, int block_size,
                     int *atm, int natm, int *bas, int nbas, double *env, double *cache)
{
        FTEnvVars envs;
        int ng[] = {1, 1, 0, 0, 2, 1, 0, 1};
        GTO_ft_init1e_envs(&envs, ng, shls, fac, Gv, b, gxyz, gs, nGv, block_size,
                           atm, natm, bas, nbas, env);
        envs.f_gout = &inner_prod_pdotp;
        return GTO_ft_aopair_drv(outR, outI, dims, eval_gz, cache, &GTO_ft_c2s_sph, &envs);
}


static void inner_prod_pxp(double *gout, double *g, int *idx, FTEnvVars *envs, int empty)
{
        int nf = envs->nf;
        int ix, iy, iz, n, k;
        int bs = envs->block_size;
        size_t g_size = envs->g_size * bs;
        double *g0R = g;
        double *g0I = g0R + g_size * 3;
        double *g1R = g0I + g_size * 3;
        double *g1I = g1R + g_size * 3;
        double *g2R = g1I + g_size * 3;
        double *g2I = g2R + g_size * 3;
        double *goutR = gout;
        double *goutI = gout + nf * bs * 3;
        double xyR, xyI;
        double sR[6], sI[6];

        G1E_D_J(g1, g0, envs->i_l+1, envs->j_l);
        G1E_D_I(g2, g0, envs->i_l  , envs->j_l);
        if (empty) {
                for (n = 0; n < nf; n++) {
                        ix = idx[0+n*3];
                        iy = idx[1+n*3];
                        iz = idx[2+n*3];
#pragma GCC ivdep
                        for (k = 0; k < bs; k++) {
                                ZMUL(sR[0], sI[0], g2, g1, g0);
                                ZMUL(sR[1], sI[1], g2, g0, g1);
                                ZMUL(sR[2], sI[2], g1, g2, g0);
                                ZMUL(sR[3], sI[3], g0, g2, g1);
                                ZMUL(sR[4], sI[4], g1, g0, g2);
                                ZMUL(sR[5], sI[5], g0, g1, g2);
                                goutR[(n*3+0)*bs+k] = sR[3] - sR[5];
                                goutR[(n*3+1)*bs+k] = sR[4] - sR[1];
                                goutR[(n*3+2)*bs+k] = sR[0] - sR[2];
                                goutI[(n*3+0)*bs+k] = sI[3] - sI[5];
                                goutI[(n*3+1)*bs+k] = sI[4] - sI[1];
                                goutI[(n*3+2)*bs+k] = sI[0] - sI[2];
                        }
                }
        } else {
                for (n = 0; n < nf; n++) {
                        ix = idx[0+n*3];
                        iy = idx[1+n*3];
                        iz = idx[2+n*3];
#pragma GCC ivdep
                        for (k = 0; k < bs; k++) {
                                ZMUL(sR[0], sI[0], g2, g1, g0);
                                ZMUL(sR[1], sI[1], g2, g0, g1);
                                ZMUL(sR[2], sI[2], g1, g2, g0);
                                ZMUL(sR[3], sI[3], g0, g2, g1);
                                ZMUL(sR[4], sI[4], g1, g0, g2);
                                ZMUL(sR[5], sI[5], g0, g1, g2);
                                goutR[(n*3+0)*bs+k] += sR[3] - sR[5];
                                goutR[(n*3+1)*bs+k] += sR[4] - sR[1];
                                goutR[(n*3+2)*bs+k] += sR[0] - sR[2];
                                goutI[(n*3+0)*bs+k] += sI[3] - sI[5];
                                goutI[(n*3+1)*bs+k] += sI[4] - sI[1];
                                goutI[(n*3+2)*bs+k] += sI[0] - sI[2];
                        }
                }
        }
}

int GTO_ft_pxp_cart(double *outR, double *outI, int *shls, int *dims,
                    void (*eval_gz)(), double complex fac,
                    double *Gv, double *b, int *gxyz, int *gs, int nGv, int block_size,
                    int *atm, int natm, int *bas, int nbas, double *env, double *cache)
{
        FTEnvVars envs;
        int ng[] = {1, 1, 0, 0, 2, 1, 0, 3};
        GTO_ft_init1e_envs(&envs, ng, shls, fac, Gv, b, gxyz, gs, nGv, block_size,
                           atm, natm, bas, nbas, env);
        envs.f_gout = &inner_prod_pxp;
        return GTO_ft_aopair_drv(outR, outI, dims, eval_gz, cache, &GTO_ft_c2s_cart, &envs);
}

int GTO_ft_pxp_sph(double *outR, double *outI, int *shls, int *dims,
                   void (*eval_gz)(), double complex fac,
                   double *Gv, double *b, int *gxyz, int *gs, int nGv, int block_size,
                   int *atm, int natm, int *bas, int nbas, double *env, double *cache)
{
        FTEnvVars envs;
        int ng[] = {1, 1, 0, 0, 2, 1, 0, 3};
        GTO_ft_init1e_envs(&envs, ng, shls, fac, Gv, b, gxyz, gs, nGv, block_size,
                           atm, natm, bas, nbas, env);
        envs.f_gout = &inner_prod_pxp;
        return GTO_ft_aopair_drv(outR, outI, dims, eval_gz, cache, &GTO_ft_c2s_sph, &envs);
}
