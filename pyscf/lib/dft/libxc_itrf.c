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
 * Authors: Qiming Sun <osirpt.sun@gmail.com>
 *          Susi Lehtola <susi.lehtola@gmail.com>
 *          Xing Zhang <zhangxing.nju@gmail.com>
 *
 * libxc from
 * https://libxc.gitlab.io
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <xc.h>
#include <string.h>
#include "config.h"
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX_THREADS     256

// TODO: register python signal
#define raise_error     return

/* Extracted from libxc:functionals.c since this function is not exposed
 * currently. See issue #2756.
 */
static int
xc_func_find_ext_params_name(const xc_func_type *p, const char *name) {
  int ii;
  assert(p != NULL && p->info->ext_params.n > 0);
  for(ii=0; ii<p->info->ext_params.n; ii++){
    if(strcmp(p->info->ext_params.names[ii], name) == 0) {
      return ii;
    }
  }
  return -1;
}

/* Extracted from comments of libxc:gga.c

    sigma_st          = grad rho_s . grad rho_t
    zk                = energy density per unit particle

    vrho_s            = d n*zk / d rho_s
    vsigma_st         = d n*zk / d sigma_st

    v2rho2_st         = d^2 n*zk / d rho_s d rho_t
    v2rhosigma_svx    = d^2 n*zk / d rho_s d sigma_tv
    v2sigma2_stvx     = d^2 n*zk / d sigma_st d sigma_vx

    v3rho3_stv        = d^3 n*zk / d rho_s d rho_t d rho_v
    v3rho2sigma_stvx  = d^3 n*zk / d rho_s d rho_t d sigma_vx
    v3rhosigma2_svxyz = d^3 n*zk / d rho_s d sigma_vx d sigma_yz
    v3sigma3_stvxyz   = d^3 n*zk / d sigma_st d sigma_vx d sigma_yz

 if nspin == 2
    rho(2)          = (u, d)
    sigma(3)        = (uu, ud, dd)

 * vxc(N*5):
    vrho(2)         = (u, d)
    vsigma(3)       = (uu, ud, dd)

 * fxc(N*45):
    v2rho2(3)       = (u_u, u_d, d_d)
    v2rhosigma(6)   = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
    v2sigma2(6)     = (uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)
    v2lapl2(3)
    vtau2(3)
    v2rholapl(4)
    v2rhotau(4)
    v2lapltau(4)
    v2sigmalapl(6)
    v2sigmatau(6)

 * kxc(N*35):
    v3rho3(4)       = (u_u_u, u_u_d, u_d_d, d_d_d)
    v3rho2sigma(9)  = (u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, d_d_ud, d_d_dd)
    v3rhosigma2(12) = (u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd)
    v3sigma(10)     = (uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, ud_ud_dd, ud_dd_dd, dd_dd_dd)

 */

#define LDA_NVAR        1
#define GGA_NVAR        4
#define MGGA_NVAR       5

/*
 * rho_u/rho_d = (den,grad_x,grad_y,grad_z,laplacian,tau)
 * In spin restricted case (spin == 1), rho_u is assumed to be the
 * spin-free quantities, rho_d is not used.
 */
static void _eval_rho(double *rho, double *rho_u, int spin, int nvar, int np, int ld_rho_u)
{
        int i;
        double *sigma, *tau;
        double *gxu, *gyu, *gzu, *gxd, *gyd, *gzd;
        double *tau_u, *tau_d;
        double *rho_d = rho_u + ld_rho_u * nvar;

        switch (nvar) {
        case LDA_NVAR:
                if (spin == 1) {
                        for (i = 0; i < np; i++) {
                                rho[i*2+0] = rho_u[i];
                                rho[i*2+1] = rho_d[i];
                        }
                } else {
                        for (i = 0; i < np; i++) {
                                rho[i] = rho_u[i];
                        }
                }
                break;
        case GGA_NVAR:
                if (spin == 1) {
                        sigma = rho + np * 2;
                        gxu = rho_u + ld_rho_u;
                        gyu = rho_u + ld_rho_u * 2;
                        gzu = rho_u + ld_rho_u * 3;
                        gxd = rho_d + ld_rho_u;
                        gyd = rho_d + ld_rho_u * 2;
                        gzd = rho_d + ld_rho_u * 3;
                        for (i = 0; i < np; i++) {
                                rho[i*2+0] = rho_u[i];
                                rho[i*2+1] = rho_d[i];
                                sigma[i*3+0] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                                sigma[i*3+1] = gxu[i]*gxd[i] + gyu[i]*gyd[i] + gzu[i]*gzd[i];
                                sigma[i*3+2] = gxd[i]*gxd[i] + gyd[i]*gyd[i] + gzd[i]*gzd[i];
                        }
                } else {
                        sigma = rho + np;
                        gxu = rho_u + ld_rho_u;
                        gyu = rho_u + ld_rho_u * 2;
                        gzu = rho_u + ld_rho_u * 3;
                        for (i = 0; i < np; i++) {
                                rho[i] = rho_u[i];
                                sigma[i] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                        }
                }
                break;
        case MGGA_NVAR:
                if (spin == 1) {
                        sigma = rho + np * 2;
                        tau = sigma + np * 3;
                        gxu = rho_u + ld_rho_u;
                        gyu = rho_u + ld_rho_u * 2;
                        gzu = rho_u + ld_rho_u * 3;
                        gxd = rho_d + ld_rho_u;
                        gyd = rho_d + ld_rho_u * 2;
                        gzd = rho_d + ld_rho_u * 3;
                        tau_u  = rho_u + ld_rho_u * 4;
                        tau_d  = rho_d + ld_rho_u * 4;
                        for (i = 0; i < np; i++) {
                                rho[i*2+0] = rho_u[i];
                                rho[i*2+1] = rho_d[i];
                                tau[i*2+0] = tau_u[i];
                                tau[i*2+1] = tau_d[i];
                        }
                        for (i = 0; i < np; i++) {
                                sigma[i*3+0] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                                sigma[i*3+1] = gxu[i]*gxd[i] + gyu[i]*gyd[i] + gzu[i]*gzd[i];
                                sigma[i*3+2] = gxd[i]*gxd[i] + gyd[i]*gyd[i] + gzd[i]*gzd[i];
                        }
                } else {
                        sigma = rho + np;
                        tau  = sigma + np;
                        gxu = rho_u + ld_rho_u;
                        gyu = rho_u + ld_rho_u * 2;
                        gzu = rho_u + ld_rho_u * 3;
                        tau_u = rho_u + ld_rho_u * 4;
                        for (i = 0; i < np; i++) {
                                rho[i] = rho_u[i];
                                sigma[i] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                                tau[i] = tau_u[i];
                        }
                }
                break;
        }
}
static void _eval_xc(xc_func_type *func_x, int spin, int deriv, int np,
                     double *rho, double *exc, int offset, int blksize)
{
        double *sigma, *tau;
        double *lapl = rho;
        double *vrho   = NULL;
        double *vsigma = NULL;
        double *vlapl  = NULL;
        double *vtau   = NULL;
        double *v2rho2      = NULL;
        double *v2rhosigma  = NULL;
        double *v2sigma2    = NULL;
        double *v2lapl2     = NULL;
        double *v2tau2      = NULL;
        double *v2rholapl   = NULL;
        double *v2rhotau    = NULL;
        double *v2sigmalapl = NULL;
        double *v2sigmatau  = NULL;
        double *v2lapltau   = NULL;
        double *v3rho3         = NULL;
        double *v3rho2sigma    = NULL;
        double *v3rhosigma2    = NULL;
        double *v3sigma3       = NULL;
        double *v3rho2lapl     = NULL;
        double *v3rho2tau      = NULL;
        double *v3rhosigmalapl = NULL;
        double *v3rhosigmatau  = NULL;
        double *v3rholapl2     = NULL;
        double *v3rholapltau   = NULL;
        double *v3rhotau2      = NULL;
        double *v3sigma2lapl   = NULL;
        double *v3sigma2tau    = NULL;
        double *v3sigmalapl2   = NULL;
        double *v3sigmalapltau = NULL;
        double *v3sigmatau2    = NULL;
        double *v3lapl3        = NULL;
        double *v3lapl2tau     = NULL;
        double *v3lapltau2     = NULL;
        double *v3tau3         = NULL;
        double *v4rho4           = NULL;
        double *v4rho3sigma      = NULL;
        double *v4rho3lapl       = NULL;
        double *v4rho3tau        = NULL;
        double *v4rho2sigma2     = NULL;
        double *v4rho2sigmalapl  = NULL;
        double *v4rho2sigmatau   = NULL;
        double *v4rho2lapl2      = NULL;
        double *v4rho2lapltau    = NULL;
        double *v4rho2tau2       = NULL;
        double *v4rhosigma3      = NULL;
        double *v4rhosigma2lapl  = NULL;
        double *v4rhosigma2tau   = NULL;
        double *v4rhosigmalapl2  = NULL;
        double *v4rhosigmalapltau= NULL;
        double *v4rhosigmatau2   = NULL;
        double *v4rholapl3       = NULL;
        double *v4rholapl2tau    = NULL;
        double *v4rholapltau2    = NULL;
        double *v4rhotau3        = NULL;
        double *v4sigma4         = NULL;
        double *v4sigma3lapl     = NULL;
        double *v4sigma3tau      = NULL;
        double *v4sigma2lapl2    = NULL;
        double *v4sigma2lapltau  = NULL;
        double *v4sigma2tau2     = NULL;
        double *v4sigmalapl3     = NULL;
        double *v4sigmalapl2tau  = NULL;
        double *v4sigmalapltau2  = NULL;
        double *v4sigmatau3      = NULL;
        double *v4lapl4          = NULL;
        double *v4lapl3tau       = NULL;
        double *v4lapl2tau2      = NULL;
        double *v4lapltau3       = NULL;
        double *v4tau4           = NULL;

        switch (func_x->info->family) {
        case XC_FAMILY_LDA:
#ifdef XC_FAMILY_HYB_LDA
        case XC_FAMILY_HYB_LDA:
#endif
                // ex is the energy density
                // NOTE libxc library added ex/ec into vrho/vcrho
                // vrho = rho d ex/d rho + ex, see work_lda.c:L73
                if (spin == 1) {
                        if (deriv > 0) {
                                vrho = exc + np;
                        }
                        if (deriv > 1) {
                                v2rho2 = vrho + np * 2;
                        }
                        if (deriv > 2) {
                                v3rho3 = v2rho2 + np * 3;
                        }
                        if (deriv > 3) {
                                v4rho4 = v3rho3 + np * 4;
                        }

                        // set offset
                        exc += offset;
                        if (deriv > 0) {
                                vrho += offset * 2;
                        }
                        if (deriv > 1) {
                                v2rho2 += offset * 3;
                        }
                        if (deriv > 2) {
                                v3rho3 += offset * 4;
                        }
                        if (deriv > 3) {
                                v4rho4 += offset * 5;
                        }
                } else {
                        if (deriv > 0) {
                                vrho = exc + np;
                        }
                        if (deriv > 1) {
                                v2rho2 = vrho + np;
                        }
                        if (deriv > 2) {
                                v3rho3 = v2rho2 + np;
                        }
                        if (deriv > 3) {
                                v4rho4 = v3rho3 + np;
                        }

                        // set offset
                        exc += offset;
                        if (deriv > 0) {
                                vrho += offset;
                        }
                        if (deriv > 1) {
                                v2rho2 += offset;
                        }
                        if (deriv > 2) {
                                v3rho3 += offset;
                        }
                        if (deriv > 3) {
                                v4rho4 += offset;
                        }
                }
                xc_lda(func_x, blksize, rho, exc, vrho, v2rho2, v3rho3, v4rho4);
                break;
        case XC_FAMILY_GGA:
#ifdef XC_FAMILY_HYB_GGA
        case XC_FAMILY_HYB_GGA:
#endif
                if (spin == 1) {
                        sigma = rho + blksize * 2;
                        if (deriv > 0) {
                                vrho = exc + np;
                                vsigma = vrho + np * 2;
                        }
                        if (deriv > 1) {
                                v2rho2 = vsigma + np * 3;
                                v2rhosigma = v2rho2 + np * 3;
                                v2sigma2 = v2rhosigma + np * 6; // np*6
                        }
                        if (deriv > 2) {
                                v3rho3 = v2sigma2 + np * 6;
                                v3rho2sigma = v3rho3 + np * 4;
                                v3rhosigma2 = v3rho2sigma + np * 9;
                                v3sigma3 = v3rhosigma2 + np * 12; // np*10
                        }
                        if (deriv > 3) {
                                v4rho4       = v3sigma3     + np * 10  ;
                                v4rho3sigma  = v4rho4       + np * 5   ;
                                v4rho2sigma2 = v4rho3sigma  + np * 4*3 ;
                                v4rhosigma3  = v4rho2sigma2 + np * 3*6 ;
                                v4sigma4     = v4rhosigma3  + np * 2*10;
                        }

                        // set offset
                        exc += offset;
                        if (deriv > 0) {
                                vrho += offset * 2;
                                vsigma += offset * 3;
                        }
                        if (deriv > 1) {
                                v2rho2 += offset * 3;
                                v2rhosigma += offset * 6;
                                v2sigma2 += offset * 6;
                        }
                        if (deriv > 2) {
                                v3rho3 += offset * 4;
                                v3rho2sigma += offset * 9;
                                v3rhosigma2 += offset * 12;
                                v3sigma3 += offset * 10;
                        }
                        if (deriv > 3) {
                                v4rho4 += offset * 5;
                                v4rho3sigma += offset * 4*3;
                                v4rho2sigma2 += offset * 3*6;
                                v4rhosigma3 += offset * 2*10;
                                v4sigma4 += offset * 15;
                        }
                } else {
                        sigma = rho + blksize;
                        if (deriv > 0) {
                                vrho = exc + np;
                                vsigma = vrho + np;
                        }
                        if (deriv > 1) {
                                v2rho2 = vsigma + np;
                                v2rhosigma = v2rho2 + np;
                                v2sigma2 = v2rhosigma + np;
                        }
                        if (deriv > 2) {
                                v3rho3 = v2sigma2 + np;
                                v3rho2sigma = v3rho3 + np;
                                v3rhosigma2 = v3rho2sigma + np;
                                v3sigma3 = v3rhosigma2 + np;
                        }
                        if (deriv > 3) {
                                v4rho4       = v3sigma3     + np;
                                v4rho3sigma  = v4rho4       + np;
                                v4rho2sigma2 = v4rho3sigma  + np;
                                v4rhosigma3  = v4rho2sigma2 + np;
                                v4sigma4     = v4rhosigma3  + np;
                        }

                        // set offset
                        exc += offset;
                        if (deriv > 0) {
                                vrho += offset;
                                vsigma += offset;
                        }
                        if (deriv > 1) {
                                v2rho2 += offset;
                                v2rhosigma += offset;
                                v2sigma2 += offset;
                        }
                        if (deriv > 2) {
                                v3rho3 += offset;
                                v3rho2sigma += offset;
                                v3rhosigma2 += offset;
                                v3sigma3 += offset;
                        }
                        if (deriv > 3) {
                                v4rho4 += offset;
                                v4rho3sigma += offset;
                                v4rho2sigma2 += offset;
                                v4rhosigma3 += offset;
                                v4sigma4 += offset;
                        }
                }
                xc_gga(func_x, blksize, rho, sigma,
                       exc, vrho, vsigma,
                       v2rho2, v2rhosigma, v2sigma2,
                       v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3,
                       v4rho4, v4rho3sigma, v4rho2sigma2, v4rhosigma3, v4sigma4);
                break;
        case XC_FAMILY_MGGA:
#ifdef XC_FAMILY_HYB_MGGA
        case XC_FAMILY_HYB_MGGA:
#endif
                if (spin == 1) {
                        sigma = rho + blksize * 2;
                        tau = sigma + blksize * 3;
                        if (deriv > 0) {
                                vrho = exc + np;
                                vsigma = vrho + np * 2;
                                vtau = vsigma + np * 3;
                        }
                        if (deriv > 1) {
                                v2rho2      = vtau        + np * 2;
                                v2rhosigma  = v2rho2      + np * 3;
                                v2sigma2    = v2rhosigma  + np * 6;
                                v2rhotau    = v2sigma2    + np * 6;
                                v2sigmatau  = v2rhotau    + np * 4;
                                v2tau2      = v2sigmatau  + np * 6;
                        }
                        if (deriv > 2) {
                                v3rho3         = v2tau2         + np * 3 ;
                                v3rho2sigma    = v3rho3         + np * 4 ;
                                v3rhosigma2    = v3rho2sigma    + np * 9 ;
                                v3sigma3       = v3rhosigma2    + np * 12;
                                v3rho2tau      = v3sigma3       + np * 10;
                                v3rhosigmatau  = v3rho2tau      + np * 6 ;
                                v3rhotau2      = v3rhosigmatau  + np * 12;
                                v3sigma2tau    = v3rhotau2      + np * 6 ;
                                v3sigmatau2    = v3sigma2tau    + np * 12;
                                v3tau3         = v3sigmatau2    + np * 9 ;
                        }
                        if (deriv > 3) {
                                v4rho4         = v3tau3         + np * 4    ;
                                v4rho3sigma    = v4rho4         + np * 5    ;
                                v4rho2sigma2   = v4rho3sigma    + np * 4*3  ;
                                v4rhosigma3    = v4rho2sigma2   + np * 3*6  ;
                                v4sigma4       = v4rhosigma3    + np * 2*10 ;
                                v4rho3tau      = v4sigma4       + np * 15   ;
                                v4rho2sigmatau = v4rho3tau      + np * 4*2  ;
                                v4rho2tau2     = v4rho2sigmatau + np * 3*3*2;
                                v4rhosigma2tau = v4rho2tau2     + np * 3*3  ;
                                v4rhosigmatau2 = v4rhosigma2tau + np * 2*6*2;
                                v4rhotau3      = v4rhosigmatau2 + np * 2*3*3;
                                v4sigma3tau    = v4rhotau3      + np * 2*4  ;
                                v4sigma2tau2   = v4sigma3tau    + np * 10*2 ;
                                v4sigmatau3    = v4sigma2tau2   + np * 6*3  ;
                                v4tau4         = v4sigmatau3    + np * 3*4  ;
                        }

                        // set offset
                        exc += offset;
                        if (deriv > 0) {
                                vrho   += offset * 2;
                                vsigma += offset * 3;
                                vtau   += offset * 2;
                        }
                        if (deriv > 1) {
                                v2rho2      += offset * 3;
                                v2rhosigma  += offset * 6;
                                v2sigma2    += offset * 6;
                                v2rhotau    += offset * 4;
                                v2sigmatau  += offset * 6;
                                v2tau2      += offset * 3;
                        }
                        if (deriv > 2) {
                                v3rho3         += offset * 4 ;
                                v3rho2sigma    += offset * 9 ;
                                v3rhosigma2    += offset * 12;
                                v3sigma3       += offset * 10;
                                v3rho2tau      += offset * 6 ;
                                v3rhosigmatau  += offset * 12;
                                v3rhotau2      += offset * 6 ;
                                v3sigma2tau    += offset * 12;
                                v3sigmatau2    += offset * 9 ;
                                v3tau3         += offset * 4 ;
                        }
                        if (deriv > 3) {
                                v4rho4         += offset * 5    ;
                                v4rho3sigma    += offset * 4*3  ;
                                v4rho2sigma2   += offset * 3*6  ;
                                v4rhosigma3    += offset * 2*10 ;
                                v4sigma4       += offset * 15   ;
                                v4rho3tau      += offset * 4*2  ;
                                v4rho2sigmatau += offset * 3*3*2;
                                v4rho2tau2     += offset * 3*3  ;
                                v4rhosigma2tau += offset * 2*6*2;
                                v4rhosigmatau2 += offset * 2*3*3;
                                v4rhotau3      += offset * 2*4  ;
                                v4sigma3tau    += offset * 10*2 ;
                                v4sigma2tau2   += offset * 6*3  ;
                                v4sigmatau3    += offset * 3*4  ;
                                v4tau4         += offset * 5    ;
                        }
                } else {
                        sigma = rho + blksize;
                        tau = sigma + blksize;
                        if (deriv > 0) {
                                vrho = exc + np;
                                vsigma = vrho + np;
                                vtau = vsigma + np;
                        }
                        if (deriv > 1) {
                                v2rho2      = vtau        + np;
                                v2rhosigma  = v2rho2      + np;
                                v2sigma2    = v2rhosigma  + np;
                                v2rhotau    = v2sigma2    + np;
                                v2sigmatau  = v2rhotau    + np;
                                v2tau2      = v2sigmatau  + np;
                        }
                        if (deriv > 2) {
                                v3rho3         = v2tau2         + np;
                                v3rho2sigma    = v3rho3         + np;
                                v3rhosigma2    = v3rho2sigma    + np;
                                v3sigma3       = v3rhosigma2    + np;
                                v3rho2tau      = v3sigma3       + np;
                                v3rhosigmatau  = v3rho2tau      + np;
                                v3rhotau2      = v3rhosigmatau  + np;
                                v3sigma2tau    = v3rhotau2      + np;
                                v3sigmatau2    = v3sigma2tau    + np;
                                v3tau3         = v3sigmatau2    + np;
                        }
                        if (deriv > 3) {
                                v4rho4         = v3tau3         + np;
                                v4rho3sigma    = v4rho4         + np;
                                v4rho2sigma2   = v4rho3sigma    + np;
                                v4rhosigma3    = v4rho2sigma2   + np;
                                v4sigma4       = v4rhosigma3    + np;
                                v4rho3tau      = v4sigma4       + np;
                                v4rho2sigmatau = v4rho3tau      + np;
                                v4rho2tau2     = v4rho2sigmatau + np;
                                v4rhosigma2tau = v4rho2tau2     + np;
                                v4rhosigmatau2 = v4rhosigma2tau + np;
                                v4rhotau3      = v4rhosigmatau2 + np;
                                v4sigma3tau    = v4rhotau3      + np;
                                v4sigma2tau2   = v4sigma3tau    + np;
                                v4sigmatau3    = v4sigma2tau2   + np;
                                v4tau4         = v4sigmatau3    + np;
                        }

                        // set offset
                        exc += offset;
                        if (deriv > 0) {
                                vrho   += offset;
                                vsigma += offset;
                                vtau   += offset;
                        }
                        if (deriv > 1) {
                                v2rho2      += offset;
                                v2rhosigma  += offset;
                                v2sigma2    += offset;
                                v2rhotau    += offset;
                                v2sigmatau  += offset;
                                v2tau2      += offset;
                        }
                        if (deriv > 2) {
                                v3rho3         += offset;
                                v3rho2sigma    += offset;
                                v3rhosigma2    += offset;
                                v3sigma3       += offset;
                                v3rho2tau      += offset;
                                v3rhosigmatau  += offset;
                                v3rhotau2      += offset;
                                v3sigma2tau    += offset;
                                v3sigmatau2    += offset;
                                v3tau3         += offset;
                        }
                        if (deriv > 3) {
                                v4rho4         += offset;
                                v4rho3sigma    += offset;
                                v4rho2sigma2   += offset;
                                v4rhosigma3    += offset;
                                v4sigma4       += offset;
                                v4rho3tau      += offset;
                                v4rho2sigmatau += offset;
                                v4rho2tau2     += offset;
                                v4rhosigma2tau += offset;
                                v4rhosigmatau2 += offset;
                                v4rhotau3      += offset;
                                v4sigma3tau    += offset;
                                v4sigma2tau2   += offset;
                                v4sigmatau3    += offset;
                                v4tau4         += offset;
                        }
                }
                xc_mgga(func_x, blksize, rho, sigma, lapl, tau,
                     exc, vrho, vsigma, vlapl, vtau,
                     v2rho2, v2rhosigma, v2rholapl, v2rhotau, v2sigma2,
                     v2sigmalapl, v2sigmatau, v2lapl2, v2lapltau, v2tau2,
                     v3rho3, v3rho2sigma, v3rho2lapl, v3rho2tau, v3rhosigma2,
                     v3rhosigmalapl, v3rhosigmatau, v3rholapl2, v3rholapltau,
                     v3rhotau2, v3sigma3, v3sigma2lapl, v3sigma2tau,
                     v3sigmalapl2, v3sigmalapltau, v3sigmatau2, v3lapl3,
                     v3lapl2tau, v3lapltau2, v3tau3,
                     v4rho4, v4rho3sigma, v4rho3lapl, v4rho3tau, v4rho2sigma2,
                     v4rho2sigmalapl, v4rho2sigmatau, v4rho2lapl2, v4rho2lapltau,
                     v4rho2tau2, v4rhosigma3, v4rhosigma2lapl, v4rhosigma2tau,
                     v4rhosigmalapl2, v4rhosigmalapltau, v4rhosigmatau2,
                     v4rholapl3, v4rholapl2tau, v4rholapltau2, v4rhotau3,
                     v4sigma4, v4sigma3lapl, v4sigma3tau, v4sigma2lapl2,
                     v4sigma2lapltau, v4sigma2tau2, v4sigmalapl3, v4sigmalapl2tau,
                     v4sigmalapltau2, v4sigmatau3, v4lapl4, v4lapl3tau,
                     v4lapl2tau2, v4lapltau3, v4tau4);
                break;
        default:
                fprintf(stderr, "functional %d '%s' is not implemented\n",
                        func_x->info->number, func_x->info->name);
                raise_error;
        }
}

int LIBXC_is_lda(int xc_id)
{
        xc_func_type func;
        int lda;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                raise_error -1;
        }
        switch(func.info->family)
        {
                case XC_FAMILY_LDA:
#ifdef XC_FAMILY_HYB_LDA
                case XC_FAMILY_HYB_LDA:
#endif
                        lda = 1;
                        break;
                default:
                        lda = 0;
        }

        xc_func_end(&func);
        return lda;
}

int LIBXC_is_gga(int xc_id)
{
        xc_func_type func;
        int gga;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                raise_error -1;
        }
        switch(func.info->family)
        {
                case XC_FAMILY_GGA:
#ifdef XC_FAMILY_HYB_GGA
                case XC_FAMILY_HYB_GGA:
#endif
                        gga = 1;
                        break;
                default:
                        gga = 0;
        }

        xc_func_end(&func);
        return gga;
}

int LIBXC_is_meta_gga(int xc_id)
{
        xc_func_type func;
        int mgga;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                raise_error -1;
        }
        switch(func.info->family)
        {
                case XC_FAMILY_MGGA:
#ifdef XC_FAMILY_HYB_MGGA
                case XC_FAMILY_HYB_MGGA:
#endif
                        mgga = 1;
                        break;
                default:
                        mgga = 0;
        }

        xc_func_end(&func);
        return mgga;
}

int LIBXC_needs_laplacian(int xc_id)
{
        xc_func_type func;
        int lapl;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                raise_error -1;
        }
        lapl = func.info->flags & XC_FLAGS_NEEDS_LAPLACIAN ? 1 : 0;
        xc_func_end(&func);
        return lapl;
}

int LIBXC_is_hybrid(int xc_id)
{
        xc_func_type func;
        int hyb;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                raise_error -1;
        }

#if XC_MAJOR_VERSION <= 7
        switch(func.info->family)
        {
#ifdef XC_FAMILY_HYB_LDA
                case XC_FAMILY_HYB_LDA:
#endif
                case XC_FAMILY_HYB_GGA:
                case XC_FAMILY_HYB_MGGA:
                        hyb = 1;
                        break;
                default:
                        hyb = 0;
        }
#else
        hyb = (xc_hyb_type(&func) == XC_HYB_HYBRID);
#endif

        xc_func_end(&func);
        return hyb;
}

double LIBXC_hybrid_coeff(int xc_id)
{
        xc_func_type func;
        double factor;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                raise_error 0.0;
        }

#if XC_MAJOR_VERSION <= 7
        switch(func.info->family)
        {
#ifdef XC_FAMILY_HYB_LDA
                case XC_FAMILY_HYB_LDA:
#endif
                case XC_FAMILY_HYB_GGA:
                case XC_FAMILY_HYB_MGGA:
                        factor = xc_hyb_exx_coef(&func);
                        break;
                default:
                        factor = 0;
        }

#else
        if(xc_hyb_type(&func) == XC_HYB_HYBRID)
          factor = xc_hyb_exx_coef(&func);
        else
          factor = 0.0;
#endif

        xc_func_end(&func);
        return factor;
}

void LIBXC_nlc_coeff(int xc_id, double *nlc_pars) {

        xc_func_type func;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                raise_error;
        }
        XC(nlc_coef)(&func, &nlc_pars[0], &nlc_pars[1]);
        xc_func_end(&func);
}

void LIBXC_rsh_coeff(int xc_id, double *rsh_pars) {

        xc_func_type func;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                raise_error;
        }
        rsh_pars[0] = 0.0;
        rsh_pars[1] = 0.0;
        rsh_pars[2] = 0.0;

#if XC_MAJOR_VERSION <= 7
        XC(hyb_cam_coef)(&func, &rsh_pars[0], &rsh_pars[1], &rsh_pars[2]);
#else
        switch(xc_hyb_type(&func)) {
        case(XC_HYB_HYBRID):
        case(XC_HYB_CAM):
          XC(hyb_cam_coef)(&func, &rsh_pars[0], &rsh_pars[1], &rsh_pars[2]);
        }
#endif
        xc_func_end(&func);
}

int LIBXC_is_cam_rsh(int xc_id) {
        xc_func_type func;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                raise_error -1;
        }
#if XC_MAJOR_VERSION <= 7
        int is_cam = func.info->flags & XC_FLAGS_HYB_CAM;
#else
        int is_cam = (xc_hyb_type(&func) == XC_HYB_CAM);
#endif
        xc_func_end(&func);
        return is_cam;
}

/*
 * XC_FAMILY_LDA           1
 * XC_FAMILY_GGA           2
 * XC_FAMILY_MGGA          4
 * XC_FAMILY_LCA           8
 * XC_FAMILY_OEP          16
 * XC_FAMILY_HYB_GGA      32
 * XC_FAMILY_HYB_MGGA     64
 * XC_FAMILY_HYB_LDA     128
 */
int LIBXC_xc_type(int fn_id)
{
        xc_func_type func;
        if (xc_func_init(&func, fn_id, 1) != 0) {
                fprintf(stderr, "XC functional %d not found\n", fn_id);
                raise_error -1;
        }
        int type = func.info->family;
        xc_func_end(&func);
        return type;
}

//static int xc_output_length(int nvar, int deriv)
//{
//        int i;
//        int len = 1;
//        for (i = 1; i <= nvar; i++) {
//                len *= deriv + i;
//                len /= i;
//        }
//        return len;
//}
// offsets = [xc_output_length(nvar, i) for i in range(deriv+1)
//            for nvar in [1,2,3,5,7]]
static int xc_nvar1_offsets[] = {0, 1, 2, 3, 4, 5};
static int xc_nvar2_offsets[] = {0, 1, 3, 6, 10, 15};
static int xc_nvar3_offsets[] = {0, 1, 4, 10, 20, 35};
static int xc_nvar5_offsets[] = {0, 1, 6, 21, 56, 126};
static int xc_nvar7_offsets[] = {0, 1, 8, 36, 120, 330};

static void axpy(double *dst, double *src, double fac,
                 int np, int nsrc)
{
        int i, j;
        for (j = 0; j < nsrc; j++) {
                #pragma omp parallel for schedule(static)
                for (i = 0; i < np; i++) {
                        dst[j*np+i] += fac * src[i*nsrc+j];
                }
        }
}
static int vseg1[] = {2, 3, 2};
static int fseg1[] = {3, 6, 6, 4, 6, 3};
static int kseg1[] = {4, 9, 12, 10, 6, 12, 6, 12, 9, 4};
static int lseg1[] = {5, 12, 18, 20, 15, 8, 18, 9, 24, 18, 8, 20, 18, 12, 5};
static int *seg1[] = {NULL, vseg1, fseg1, kseg1, lseg1};

static void merge_xc(double *dst, double *ebuf, double fac,
                     int spin, int deriv, int nvar, int np, int outlen, int type)
{
        int order, nsrc, i;
        for (i = 0; i < np; i++) {
                dst[i] += fac * ebuf[i];
        }

        int *offsets0, *offsets1;
        double *pout, *pin;
        switch (type) {
        case XC_FAMILY_GGA:
#ifdef XC_FAMILY_HYB_GGA
        case XC_FAMILY_HYB_GGA:
#endif
                offsets0 = xc_nvar2_offsets;
                break;
        case XC_FAMILY_MGGA:
#ifdef XC_FAMILY_HYB_MGGA
        case XC_FAMILY_HYB_MGGA:
#endif
                offsets0 = xc_nvar3_offsets;
                break;
        default: //case XC_FAMILY_LDA:
                offsets0 = xc_nvar1_offsets;
        }

        if (spin == 0) {
                switch (nvar) {
                case LDA_NVAR:
                        offsets1 = xc_nvar1_offsets;
                        break;
                case GGA_NVAR:
                        offsets1 = xc_nvar2_offsets;
                        break;
                default: // MGGA_NVAR
                        offsets1 = xc_nvar3_offsets;
                        break;
                }

                for (order = 1; order <= deriv; order++) {
                        pout = dst + offsets1[order] * np;
                        pin = ebuf + offsets0[order] * np;
                        nsrc = offsets0[order+1] - offsets0[order];
                        #pragma omp parallel for schedule(static)
                        for (i = 0; i < np * nsrc; i++) {
                                pout[i] += fac * pin[i];
                        }
                }
                return;
        }

        switch (nvar) {
        case LDA_NVAR:
                offsets1 = xc_nvar2_offsets;
                break;
        case GGA_NVAR:
                offsets1 = xc_nvar5_offsets;
                break;
        default: // MGGA_NVAR
                offsets1 = xc_nvar7_offsets;
                break;
        }

        int terms;
        int *pseg1;
        pin = ebuf + np;
        for (order = 1; order <= deriv; order++) {
                pseg1 = seg1[order];
                pout = dst + offsets1[order] * np;
                terms = offsets0[order+1] - offsets0[order];
                for (i = 0; i < terms; i++) {
                        nsrc = pseg1[i];
                        axpy(pout, pin, fac, np, nsrc);
                        pin += np * nsrc;
                        pout += np * nsrc;
                }
        }
}

// omega is the range separation parameter mu in xcfun
void LIBXC_eval_xc(int nfn, int *fn_id, double *fac, double *omega,
                   int spin, int deriv, int nvar, int np, int outlen,
                   double *rho_u, double *output, double dens_threshold)
{
        assert(deriv <= 4);
        double *ebuf = malloc(sizeof(double) * np * outlen);

        double *rhobufs[MAX_THREADS];
        int offsets[MAX_THREADS+1];
#pragma omp parallel
{
        int iblk = omp_get_thread_num();
        int nblk = omp_get_num_threads();
        assert(nblk <= MAX_THREADS);

        int blksize = np / nblk;
        int ioff = iblk * blksize;
        int np_mod = np % nblk;
        if (iblk < np_mod) {
            blksize += 1;
        }
        if (np_mod > 0) {
            ioff += MIN(iblk, np_mod);
        }
        offsets[iblk] = ioff;
        if (iblk == nblk-1) {
            offsets[nblk] = np;
            assert(ioff + blksize == np);
        }

        double *rho_priv = malloc(sizeof(double) * blksize * 7);
        rhobufs[iblk] = rho_priv;
        _eval_rho(rho_priv, rho_u+ioff, spin, nvar, blksize, np);
}

        int nspin = spin + 1;
        int i, j;
        xc_func_type func;
        for (i = 0; i < nfn; i++) {
                if (xc_func_init(&func, fn_id[i], nspin) != 0) {
                        fprintf(stderr, "XC functional %d not found\n",
                                fn_id[i]);
                        raise_error;
                }
                if (dens_threshold > 0) {
                        xc_func_set_dens_threshold(&func, dens_threshold);
                }

                // set the range-separated parameter
                if (omega[i] != 0) {
                        // skip if func is not a RSH functional
                        if ( xc_func_find_ext_params_name(&func, "_omega") >= 0 ) {
                                xc_func_set_ext_params_name(&func, "_omega", omega[i]);
                        }
                        // Recursively set the sub-functionals if they are RSH
                        // functionals
                        for (j = 0; j < func.n_func_aux; j++) {
                                if ( xc_func_find_ext_params_name(func.func_aux[j], "_omega") >= 0 ) {
                                        xc_func_set_ext_params_name(func.func_aux[j], "_omega", omega[i]);
                                }
                        }
                }

                // alpha and beta are hardcoded in many functionals in the libxc
                // code, e.g. the coefficients of B88 (=1-alpha) and
                // ITYH (=-beta) in cam-b3lyp.  Overwriting func->cam_alpha and
                // func->cam_beta does not update the coefficients accordingly.
                //func->cam_alpha = alpha;
                //func->cam_beta  = beta;
                // However, the parameters can be set with the libxc function
                //void xc_func_set_ext_params_name(xc_func_type *p, const char *name, double par);
                // since libxc 5.1.0
#if defined XC_SET_RELATIVITY
                xc_lda_x_set_params(&func, relativity);
#endif

#pragma omp parallel
{
                int iblk = omp_get_thread_num();
                int offset = offsets[iblk];
                int blksize = offsets[iblk+1] - offset;
                _eval_xc(&func, spin, deriv, np, rhobufs[iblk], ebuf, offset, blksize);
}

                merge_xc(output, ebuf, fac[i],
                         spin, deriv, nvar, np, outlen, func.info->family);
                xc_func_end(&func);
        }
        free(ebuf);
#pragma omp parallel
{
        int iblk = omp_get_thread_num();
        free(rhobufs[iblk]);
}
}

int LIBXC_max_deriv_order(int xc_id)
{
        xc_func_type func;
        int ord;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                raise_error -1;
        }

        if (func.info->flags & XC_FLAGS_HAVE_LXC) {
                ord = 4;
        } else if(func.info->flags & XC_FLAGS_HAVE_KXC) {
                ord = 3;
        } else if(func.info->flags & XC_FLAGS_HAVE_FXC) {
                ord = 2;
        } else if(func.info->flags & XC_FLAGS_HAVE_VXC) {
                ord = 1;
        } else if(func.info->flags & XC_FLAGS_HAVE_EXC) {
                ord = 0;
        } else {
                ord = -1;
        }

        xc_func_end(&func);
        return ord;
}

int LIBXC_number_of_functionals()
{
  return xc_number_of_functionals();
}

void LIBXC_functional_numbers(int *list)
{
  return xc_available_functional_numbers(list);
}

char * LIBXC_functional_name(int ifunc)
{
  return xc_functional_get_name(ifunc);
}

const char * LIBXC_version()
{
  return xc_version_string();
}

const char * LIBXC_reference()
{
  return xc_reference();
}

const char * LIBXC_reference_doi()
{
  return xc_reference_doi();
}

void LIBXC_xc_reference(int xc_id, const char **refs)
{
        xc_func_type func;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                raise_error;
        }

        int i;
        for (i = 0; i < XC_MAX_REFERENCES; i++) {
                if (func.info->refs[i] == NULL || func.info->refs[i]->ref == NULL) {
                        refs[i] = NULL;
                        break;
                }
                refs[i] = func.info->refs[i]->ref;
        }
	xc_func_end(&func);
}

int LIBXC_is_nlc(int xc_id)
{
        xc_func_type func;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                raise_error -1;
        }
	int is_nlc = func.info->flags & XC_FLAGS_VV10;
	xc_func_end(&func);
        return is_nlc; 
}
