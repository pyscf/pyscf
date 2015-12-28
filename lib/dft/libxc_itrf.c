/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 * libxc from
 * http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:manual
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xc.h>

//double xc_hyb_exx_coef(xc_func_type *);

double VXChybrid_coeff(int xc_id, int spin)
{
        xc_func_type func;
        double factor;
        if(xc_func_init(&func, xc_id, spin) != 0){
                fprintf(stderr, "XC functional %d not found\n", xc_id);
                exit(1);
        }
        switch(func.info->family)
        {
                case XC_FAMILY_HYB_GGA:
                case XC_FAMILY_HYB_MGGA:
                        factor = xc_hyb_exx_coef(&func);
                        break;
                default:
                        factor = 0;
        }

        xc_func_end(&func);
        return factor;
}

int VXCinit_libxc(xc_func_type *func_x, xc_func_type *func_c,
                  int x_id, int c_id, int spin, int relativity)
{
        if (!func_x) {
                func_x->info = NULL;
        }
        if (!func_c) {
                func_c->info = NULL;
        }
        if (xc_func_init(func_x, x_id, spin) != 0) {
                fprintf(stderr, "X functional %d not found\n", x_id);
                exit(1);
        }
        if (func_x->info->kind == XC_EXCHANGE &&
            xc_func_init(func_c, c_id, spin) != 0) {
                fprintf(stderr, "C functional %d not found\n", c_id);
                exit(1);
        }

#if defined XC_SET_RELATIVITY
        xc_lda_x_set_params(func_x, relativity);
#endif
        return 0;
}

int VXCdel_libxc(xc_func_type *func_x, xc_func_type *func_c)
{
        if (func_x->info->kind == XC_EXCHANGE) {
                xc_func_end(func_c);
        }
        xc_func_end(func_x);
        return 0;
}

// y += x
static void addvec(double *y, double *x, int n)
{
        int i;
        for (i = 0; i < n; i++) {
                y[i] += x[i];
        }
}

/* Extracted from comments of libxc:gga.c
 
    sigma_st       = grad rho_s . grad rho_t
    zk             = energy density per unit particle
 
    vrho_s         = d zk / d rho_s
    vsigma_st      = d n*zk / d sigma_st
    
    v2rho2_st      = d^2 n*zk / d rho_s d rho_t
    v2rhosigma_svx = d^2 n*zk / d rho_s d sigma_tv
    v2sigma2_stvx  = d^2 n*zk / d sigma_st d sigma_vx
 
 if nspin == 2
    rho(2)        = (u, d)
    sigma(3)      = (uu, du, dd)
 
    vrho(2)       = (u, d)
    vsigma(3)     = (uu, du, dd)
 
    v2rho2(3)     = (uu, du, dd)
    v2rhosigma(6) = (u_uu, u_ud, u_dd, d_uu, d_ud, d_dd)
    v2sigma2(6)   = (uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd)
 */
// e ~ energy,  n ~ num electrons,  v ~ XC potential matrix
void VXCnr_eval_xc(int x_id, int c_id, int spin, int relativity, int np,
                   double *rho, double *sigma,
                   double *exc, double *vrho, double *vsigma)
{
        xc_func_type func_x = {};
        xc_func_type func_c = {};
        VXCinit_libxc(&func_x, &func_c, x_id, c_id, spin, relativity);

        double *buf = malloc(sizeof(double) * np*6);
        double *vcrho = buf;
        double *vcsigma = vcrho+2*np;
        double *ec = vcsigma+3*np;

        switch (func_x.info->family) {
        case XC_FAMILY_LDA:
                // exc is the energy density
                // note libxc have added exc/ec to vrho/vcrho
                xc_lda_exc_vxc(&func_x, np, rho, exc, vrho);
                //memset(vsigma, 0, sizeof(double)*np);
                break;
        case XC_FAMILY_GGA:
        case XC_FAMILY_HYB_GGA:
                xc_gga_exc_vxc(&func_x, np, rho, sigma, exc,
                               vrho, vsigma);
                break;
        default:
                fprintf(stderr, "X functional %d '%s' is not implmented\n",
                        func_x.info->number, func_x.info->name);
                exit(1);
        }

        if (func_x.info->kind == XC_EXCHANGE) {
                switch (func_c.info->family) {
                case XC_FAMILY_LDA:
                        xc_lda_exc_vxc(&func_c, np, rho, ec, vcrho);
                        break;
                case XC_FAMILY_GGA:
                        xc_gga_exc_vxc(&func_c, np, rho, sigma, ec,
                                       vcrho, vcsigma);
                        if (spin == XC_POLARIZED) {
                                addvec(vsigma, vcsigma, np*3);
                        } else {
                                addvec(vsigma, vcsigma, np);
                        }
                        break;
                default:
                        fprintf(stderr, "C functional %d '%s' is not implmented\n",
                                func_c.info->number,
                                func_c.info->name);
                        exit(1);
                }
                addvec(exc, ec, np);
                if (spin == XC_POLARIZED) {
                        addvec(vrho, vcrho, np*2);
                } else {
                        addvec(vrho, vcrho, np);
                }
        }

        free(buf);
        VXCdel_libxc(&func_x, &func_c);
}

void VXCnr_eval_x(int x_id, int spin, int relativity, int np,
                  double *rho, double *sigma,
                  double *ex, double *vrho, double *vsigma)
{
        xc_func_type func_x = {};
        if (xc_func_init(&func_x, x_id, spin) != 0) {
                fprintf(stderr, "X functional %d not found\n", x_id);
                exit(1);
        }

        switch (func_x.info->family) {
        case XC_FAMILY_LDA:
                // ex is the energy density
                // note libxc have added ex/ec to vrho/vcrho
                xc_lda_exc_vxc(&func_x, np, rho, ex, vrho);
                //memset(vsigma, 0, sizeof(double)*np);
                break;
        case XC_FAMILY_GGA:
        case XC_FAMILY_HYB_GGA:
                xc_gga_exc_vxc(&func_x, np, rho, sigma, ex,
                               vrho, vsigma);
                break;
        default:
                fprintf(stderr, "X functional %d '%s' is not implmented\n",
                        func_x.info->number, func_x.info->name);
                exit(1);
        }

        xc_func_end(&func_x);
}

void VXCnr_eval_c(int c_id, int spin, int relativity, int np,
                  double *rho, double *sigma,
                  double *ec, double *vrho, double *vsigma)
{
        xc_func_type func_c = {};
        if (xc_func_init(&func_c, c_id, spin) != 0) {
                fprintf(stderr, "C functional %d not found\n", c_id);
                exit(1);
        }

        switch (func_c.info->family) {
        case XC_FAMILY_LDA:
                xc_lda_exc_vxc(&func_c, np, rho, ec, vrho);
                break;
        case XC_FAMILY_GGA:
                xc_gga_exc_vxc(&func_c, np, rho, sigma, ec,
                               vrho, vsigma);
                break;
        default:
                fprintf(stderr, "C functional %d '%s' is not implmented\n",
                        func_c.info->number,
                        func_c.info->name);
                exit(1);
        }

        xc_func_end(&func_c);
}

