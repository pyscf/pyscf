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
 *
 * libxc from
 * http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:manual
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <xc.h>
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

// TODO: register python signal
#define raise_error     return

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
/*
 * rho_u/rho_d = (den,grad_x,grad_y,grad_z,laplacian,tau)
 * In spin restricted case (spin == 1), rho_u is assumed to be the
 * spin-free quantities, rho_d is not used.
 */
static void _eval_xc(xc_func_type *func_x, int spin, int np,
                     double *rho_u, double *rho_d,
                     double *ex, double *vxc, double *fxc, double *kxc)
{
        int i;
        double *rho, *sigma, *lapl, *tau;
        double *gxu, *gyu, *gzu, *gxd, *gyd, *gzd;
        double *lapl_u, *lapl_d, *tau_u, *tau_d;
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

        switch (func_x->info->family) {
        case XC_FAMILY_LDA:
#ifdef XC_FAMILY_HYB_LDA
        case XC_FAMILY_HYB_LDA:
#endif
                // ex is the energy density
                // NOTE libxc library added ex/ec into vrho/vcrho
                // vrho = rho d ex/d rho + ex, see work_lda.c:L73
                if (spin == XC_POLARIZED) {
                        rho = malloc(sizeof(double) * np*2);
                        for (i = 0; i < np; i++) {
                                rho[i*2+0] = rho_u[i];
                                rho[i*2+1] = rho_d[i];
                        }
                        xc_lda_exc_vxc_fxc_kxc(func_x, np, rho, ex, vxc, fxc, kxc);
                        free(rho);
                } else {
                        rho = rho_u;
                        xc_lda_exc_vxc_fxc_kxc(func_x, np, rho, ex, vxc, fxc, kxc);
                }
                break;
        case XC_FAMILY_GGA:
#ifdef XC_FAMILY_HYB_GGA
        case XC_FAMILY_HYB_GGA:
#endif
                if (spin == XC_POLARIZED) {
                        rho = malloc(sizeof(double) * np * 5);
                        sigma = rho + np * 2;
                        gxu = rho_u + np;
                        gyu = rho_u + np * 2;
                        gzu = rho_u + np * 3;
                        gxd = rho_d + np;
                        gyd = rho_d + np * 2;
                        gzd = rho_d + np * 3;
                        for (i = 0; i < np; i++) {
                                rho[i*2+0] = rho_u[i];
                                rho[i*2+1] = rho_d[i];
                                sigma[i*3+0] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                                sigma[i*3+1] = gxu[i]*gxd[i] + gyu[i]*gyd[i] + gzu[i]*gzd[i];
                                sigma[i*3+2] = gxd[i]*gxd[i] + gyd[i]*gyd[i] + gzd[i]*gzd[i];
                        }
                        if (vxc != NULL) {
                                vrho = vxc;
                                vsigma = vxc + np * 2;
                        }
                        if (fxc != NULL) {
                                v2rho2 = fxc;
                                v2rhosigma = fxc + np * 3;
                                v2sigma2 = v2rhosigma + np * 6; // np*6
                        }
                        if (kxc != NULL) {
                                v3rho3 = kxc;
                                v3rho2sigma = kxc + np * 4;
                                v3rhosigma2 = v3rho2sigma + np * 9;
                                v3sigma3 = v3rhosigma2 + np * 12; // np*10
                        }
                        xc_gga_exc_vxc_fxc_kxc(func_x, np, rho, sigma, ex,
                               vrho, vsigma, v2rho2, v2rhosigma, v2sigma2,
                               v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3);
                        free(rho);
                } else {
                        rho = rho_u;
                        sigma = malloc(sizeof(double) * np);
                        gxu = rho_u + np;
                        gyu = rho_u + np * 2;
                        gzu = rho_u + np * 3;
                        for (i = 0; i < np; i++) {
                                sigma[i] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                        }
                        if (vxc != NULL) {
                                vrho = vxc;
                                vsigma = vxc + np;
                        }
                        if (fxc != NULL) {
                                v2rho2 = fxc;
                                v2rhosigma = fxc + np;
                                v2sigma2 = v2rhosigma + np;
                        }
                        if (kxc != NULL) {
                                v3rho3 = kxc;
                                v3rho2sigma = v3rho3 + np;
                                v3rhosigma2 = v3rho2sigma + np;
                                v3sigma3 = v3rhosigma2 + np;
                        }
                        xc_gga_exc_vxc_fxc_kxc(func_x, np, rho, sigma, ex,
                               vrho, vsigma, v2rho2, v2rhosigma, v2sigma2,
                               v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3);
                        free(sigma);
                }
                break;
        case XC_FAMILY_MGGA:
#ifdef XC_FAMILY_HYB_MGGA
        case XC_FAMILY_HYB_MGGA:
#endif
                if (spin == XC_POLARIZED) {
                        rho = malloc(sizeof(double) * np * 9);
                        sigma = rho + np * 2;
                        lapl = sigma + np * 3;
                        tau = lapl + np * 2;
                        gxu = rho_u + np;
                        gyu = rho_u + np * 2;
                        gzu = rho_u + np * 3;
                        gxd = rho_d + np;
                        gyd = rho_d + np * 2;
                        gzd = rho_d + np * 3;
                        lapl_u = rho_u + np * 4;
                        tau_u  = rho_u + np * 5;
                        lapl_d = rho_d + np * 4;
                        tau_d  = rho_d + np * 5;
                        for (i = 0; i < np; i++) {
                                rho[i*2+0] = rho_u[i];
                                rho[i*2+1] = rho_d[i];
                                lapl[i*2+0] = lapl_u[i];
                                lapl[i*2+1] = lapl_d[i];
                                tau[i*2+0] = tau_u[i];
                                tau[i*2+1] = tau_d[i];
                        }
                        for (i = 0; i < np; i++) {
                                sigma[i*3+0] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                                sigma[i*3+1] = gxu[i]*gxd[i] + gyu[i]*gyd[i] + gzu[i]*gzd[i];
                                sigma[i*3+2] = gxd[i]*gxd[i] + gyd[i]*gyd[i] + gzd[i]*gzd[i];
                        }
                        if (vxc != NULL) {
                                vrho = vxc;
                                vsigma = vxc + np * 2;
                                vlapl = vsigma + np * 3;
                                vtau = vlapl + np * 2; // np*2
                        }
                        if (fxc != NULL) {
                                v2rho2      = fxc;
                                v2rhosigma  = v2rho2      + np * 3;
                                v2sigma2    = v2rhosigma  + np * 6;
                                v2lapl2     = v2sigma2    + np * 6;
                                v2tau2      = v2lapl2     + np * 3;
                                v2rholapl   = v2tau2      + np * 3;
                                v2rhotau    = v2rholapl   + np * 4;
                                v2lapltau   = v2rhotau    + np * 4;
                                v2sigmalapl = v2lapltau   + np * 4;
                                v2sigmatau  = v2sigmalapl + np * 6;
                        }
                        if (kxc != NULL) {
                                v3rho3         = kxc;
                                v3rho2sigma    = v3rho3         + np * 4 ;
                                v3rhosigma2    = v3rho2sigma    + np * 9 ;
                                v3sigma3       = v3rhosigma2    + np * 12;
                                v3rho2lapl     = v3sigma3       + np * 10;
                                v3rho2tau      = v3rho2lapl     + np * 6 ;
                                v3rhosigmalapl = v3rho2tau      + np * 6 ;
                                v3rhosigmatau  = v3rhosigmalapl + np * 12;
                                v3rholapl2     = v3rhosigmatau  + np * 12;
                                v3rholapltau   = v3rholapl2     + np * 6 ;
                                v3rhotau2      = v3rholapltau   + np * 8 ;
                                v3sigma2lapl   = v3rhotau2      + np * 6 ;
                                v3sigma2tau    = v3sigma2lapl   + np * 12;
                                v3sigmalapl2   = v3sigma2tau    + np * 12;
                                v3sigmalapltau = v3sigmalapl2   + np * 9 ;
                                v3sigmatau2    = v3sigmalapltau + np * 12;
                                v3lapl3        = v3sigmatau2    + np * 9 ;
                                v3lapl2tau     = v3lapl3        + np * 4 ;
                                v3lapltau2     = v3lapl2tau     + np * 6 ;
                                v3tau3         = v3lapltau2     + np * 6 ;
                        }
                        xc_mgga_exc_vxc_fxc_kxc(func_x, np, rho, sigma, lapl, tau, ex,
                                vrho, vsigma, vlapl, vtau,
                                v2rho2, v2rhosigma, v2rholapl, v2rhotau, v2sigma2,
                                v2sigmalapl, v2sigmatau, v2lapl2, v2lapltau, v2tau2,
                                v3rho3, v3rho2sigma, v3rho2lapl, v3rho2tau, v3rhosigma2,
                                v3rhosigmalapl, v3rhosigmatau, v3rholapl2, v3rholapltau,
                                v3rhotau2, v3sigma3, v3sigma2lapl, v3sigma2tau,
                                v3sigmalapl2, v3sigmalapltau, v3sigmatau2, v3lapl3,
                                v3lapl2tau, v3lapltau2, v3tau3);
                        free(rho);
                } else {
                        rho = rho_u;
                        sigma = malloc(sizeof(double) * np);
                        lapl = rho_u + np * 4;
                        tau  = rho_u + np * 5;
                        gxu = rho_u + np;
                        gyu = rho_u + np * 2;
                        gzu = rho_u + np * 3;
                        for (i = 0; i < np; i++) {
                                sigma[i] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                        }
                        if (vxc != NULL) {
                                vsigma = vxc + np;
                                vlapl = vsigma + np;
                                vtau = vlapl + np;
                        }
                        if (fxc != NULL) {
                                v2rho2      = fxc;
                                v2rhosigma  = v2rho2      + np;
                                v2sigma2    = v2rhosigma  + np;
                                v2lapl2     = v2sigma2    + np;
                                v2tau2      = v2lapl2     + np;
                                v2rholapl   = v2tau2      + np;
                                v2rhotau    = v2rholapl   + np;
                                v2lapltau   = v2rhotau    + np;
                                v2sigmalapl = v2lapltau   + np;
                                v2sigmatau  = v2sigmalapl + np;
                        }
                        if (kxc != NULL) {
                                v3rho3         = kxc;
                                v3rho2sigma    = v3rho3         + np;
                                v3rhosigma2    = v3rho2sigma    + np;
                                v3sigma3       = v3rhosigma2    + np;
                                v3rho2lapl     = v3sigma3       + np;
                                v3rho2tau      = v3rho2lapl     + np;
                                v3rhosigmalapl = v3rho2tau      + np;
                                v3rhosigmatau  = v3rhosigmalapl + np;
                                v3rholapl2     = v3rhosigmatau  + np;
                                v3rholapltau   = v3rholapl2     + np;
                                v3rhotau2      = v3rholapltau   + np;
                                v3sigma2lapl   = v3rhotau2      + np;
                                v3sigma2tau    = v3sigma2lapl   + np;
                                v3sigmalapl2   = v3sigma2tau    + np;
                                v3sigmalapltau = v3sigmalapl2   + np;
                                v3sigmatau2    = v3sigmalapltau + np;
                                v3lapl3        = v3sigmatau2    + np;
                                v3lapl2tau     = v3lapl3        + np;
                                v3lapltau2     = v3lapl2tau     + np;
                                v3tau3         = v3lapltau2     + np;
                        }
                        xc_mgga_exc_vxc_fxc_kxc(func_x, np, rho, sigma, lapl, tau, ex,
                                vxc, vsigma, vlapl, vtau,
                                v2rho2, v2rhosigma, v2rholapl, v2rhotau, v2sigma2,
                                v2sigmalapl, v2sigmatau, v2lapl2, v2lapltau, v2tau2,
                                v3rho3, v3rho2sigma, v3rho2lapl, v3rho2tau, v3rhosigma2,
                                v3rhosigmalapl, v3rhosigmatau, v3rholapl2, v3rholapltau,
                                v3rhotau2, v3sigma3, v3sigma2lapl, v3sigma2tau,
                                v3sigmalapl2, v3sigmalapltau, v3sigmatau2, v3lapl3,
                                v3lapl2tau, v3lapltau2, v3tau3);
                        free(sigma);
                }
                break;
        default:
                fprintf(stderr, "functional %d '%s' is not implmented\n",
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

#if XC_MAJOR_VERSION <= 6
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

#if XC_MAJOR_VERSION <= 6
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

#if XC_MAJOR_VERSION <= 6
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
#if XC_MAJOR_VERSION <= 6
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

static int xc_output_length(int nvar, int deriv)
{
        int i;
        int len = 1.;
        for (i = 1; i <= nvar; i++) {
                len *= deriv + i;
                len /= i;
        }
        return len;
}

// return value 0 means no functional needs to be evaluated.
int LIBXC_input_length(int nfn, int *fn_id, double *fac, int spin)
{
        int i;
        int nvar = 0;
        xc_func_type func;
        for (i = 0; i < nfn; i++) {
                if (xc_func_init(&func, fn_id[i], spin) != 0) {
                        fprintf(stderr, "XC functional %d not found\n",
                                fn_id[i]);
                        raise_error -1;
                }
                if (spin == XC_POLARIZED) {
                        switch (func.info->family) {
                        case XC_FAMILY_LDA:
#ifdef XC_FAMILY_HYB_LDA
                        case XC_FAMILY_HYB_LDA:
#endif
                                nvar = MAX(nvar, 2);
                                break;
                        case XC_FAMILY_GGA:
#ifdef XC_FAMILY_HYB_GGA
                        case XC_FAMILY_HYB_GGA:
#endif
                                nvar = MAX(nvar, 5);
                                break;
                        case XC_FAMILY_MGGA:
#ifdef XC_FAMILY_HYB_MGGA
                        case XC_FAMILY_HYB_MGGA:
#endif
                                nvar = MAX(nvar, 9);
                        }
                } else {
                        switch (func.info->family) {
                        case XC_FAMILY_LDA:
#ifdef XC_FAMILY_HYB_LDA
                        case XC_FAMILY_HYB_LDA:
#endif
                                nvar = MAX(nvar, 1);
                                break;
                        case XC_FAMILY_GGA:
#ifdef XC_FAMILY_HYB_GGA
                        case XC_FAMILY_HYB_GGA:
#endif
                                nvar = MAX(nvar, 2);
                                break;
                        case XC_FAMILY_MGGA:
#ifdef XC_FAMILY_HYB_MGGA
                        case XC_FAMILY_HYB_MGGA:
#endif
                                nvar = MAX(nvar, 4);
                        }
                }
                xc_func_end(&func);
        }
        return nvar;
}

static void axpy(double *dst, double *src, double fac,
                 int np, int ndst, int nsrc)
{
        int i, j;
        for (j = 0; j < nsrc; j++) {
                for (i = 0; i < np; i++) {
                        dst[j*np+i] += fac * src[i*nsrc+j];
                }
        }
}

static void merge_xc(double *dst, double *ebuf, double *vbuf,
                     double *fbuf, double *kbuf, double fac,
                     int np, int ndst, int nvar, int spin, int type)
{
        int seg0 [] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        // LDA         |  |
        // GGA         |     |
        // MGGA        |        |
        int vseg1[] = {2, 3, 2, 2};
        // LDA         |  |
        // GGA         |        |
        // MGGA        |                          |
        int fseg1[] = {3, 6, 6, 3, 3, 4, 4, 4, 6, 6};
        // LDA         |  |
        // GGA         |        |
        // MGGA        |                                                        |
        int kseg1[] = {4, 9,12,10, 6, 6,12,12, 6, 8, 6,12,12, 9,12, 9, 4, 6, 6, 4};
        int vsegtot, fsegtot, ksegtot;
        int *vseg, *fseg, *kseg;
        if (spin == XC_POLARIZED) {
                vseg = vseg1;
                fseg = fseg1;
                kseg = kseg1;
        } else {
                vseg = seg0;
                fseg = seg0;
                kseg = seg0;
        }

        switch (type) {
        case XC_FAMILY_GGA:
#ifdef XC_FAMILY_HYB_GGA
        case XC_FAMILY_HYB_GGA:
#endif
                vsegtot = 2;
                fsegtot = 3;
                ksegtot = 4;
                break;
        case XC_FAMILY_MGGA:
#ifdef XC_FAMILY_HYB_MGGA
        case XC_FAMILY_HYB_MGGA:
#endif
                vsegtot = 4;
                fsegtot = 10;
                ksegtot = 20;  // not supported
                break;
        default: //case XC_FAMILY_LDA:
                vsegtot = 1;
                fsegtot = 1;
                ksegtot = 1;
        }

        int i;
        size_t offset;
        axpy(dst, ebuf, fac, np, ndst, 1);

        if (vbuf != NULL) {
                offset = np;
                for (i = 0; i < vsegtot; i++) {
                        axpy(dst+offset, vbuf, fac, np, ndst, vseg[i]);
                        offset += np * vseg[i];
                        vbuf += np * vseg[i];
                }
        }

        if (fbuf != NULL) {
                offset = np * xc_output_length(nvar, 1);
                for (i = 0; i < fsegtot; i++) {
                        axpy(dst+offset, fbuf, fac, np, ndst, fseg[i]);
                        offset += np * fseg[i];
                        fbuf += np * fseg[i];
                }
        }

        if (kbuf != NULL) {
                offset = np * xc_output_length(nvar, 2);
                for (i = 0; i < ksegtot; i++) {
                        axpy(dst+offset, kbuf, fac, np, ndst, kseg[i]);
                        offset += np * kseg[i];
                        kbuf += np * kseg[i];
                }
        }
}

// omega is the range separation parameter mu in xcfun
void LIBXC_eval_xc(int nfn, int *fn_id, double *fac, double *omega,
                   int spin, int deriv, int np,
                   double *rho_u, double *rho_d, double *output,
                   double dens_threshold)
{
        assert(deriv <= 3);
        int nvar = LIBXC_input_length(nfn, fn_id, fac, spin);
        if (nvar == 0) { // No functional needs to be evaluated.
                return;
        }

        int outlen = xc_output_length(nvar, deriv);
        // output buffer is zeroed in the Python caller
        //NPdset0(output, np*outlen);

        double *ebuf = malloc(sizeof(double) * np);
        double *vbuf = NULL;
        double *fbuf = NULL;
        double *kbuf = NULL;
        if (deriv > 0) {
                vbuf = malloc(sizeof(double) * np*9);
        }
        if (deriv > 1) {
                fbuf = malloc(sizeof(double) * np*45);
        }
        if (deriv > 2) {
                if (spin == XC_POLARIZED) {  // spin-unresctricted MGGA
                        // FIXME *220 in xcfun
                        kbuf = malloc(sizeof(double) * np*165);
                } else {  // spin-resctricted MGGA
                        kbuf = malloc(sizeof(double) * np*20);
                }
        }

        int i, j;
        xc_func_type func;
        for (i = 0; i < nfn; i++) {
                if (xc_func_init(&func, fn_id[i], spin) != 0) {
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
#if XC_MAJOR_VERSION <= 6
                        if (func.cam_omega != 0) {
                                func.cam_omega = omega[i];
                        }
#else
                        if (func.hyb_omega != NULL && func.hyb_omega[0] != 0) {
                                func.hyb_omega[0] = omega[i];
                        }
#endif
                        // Recursively set the sub-functionals if they are RSH
                        // functionals
                        for (j = 0; j < func.n_func_aux; j++) {
#if XC_MAJOR_VERSION <= 6
                                if (func.func_aux[j]->cam_omega != 0) {
                                        func.func_aux[j]->cam_omega = omega[i];
                                }
#else
                                if (func.func_aux[j]->hyb_omega != NULL && func.func_aux[j]->hyb_omega[0] != 0) {
                                        func.func_aux[j]->hyb_omega[0] = omega[i];
                                }
#endif
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
                _eval_xc(&func, spin, np, rho_u, rho_d, ebuf, vbuf, fbuf, kbuf);
                merge_xc(output, ebuf, vbuf, fbuf, kbuf, fac[i],
                         np, outlen, nvar, spin, func.info->family);
                xc_func_end(&func);
        }

        free(ebuf);
        if (deriv > 0) {
                free(vbuf);
        }
        if (deriv > 1) {
                free(fbuf);
        }
        if (deriv > 2) {
                free(kbuf);
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
}

int LIBXC_is_nlc(int xc_id)
{
        xc_func_type func;
        if(xc_func_init(&func, xc_id, XC_UNPOLARIZED) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                raise_error -1;
        }
        return func.info->flags & XC_FLAGS_VV10;
}
