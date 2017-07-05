/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 * xcfun Library from
 * https://github.com/dftlibs/xcfun
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <xcfun.h>

static enum xcfun_parameters CONVERT2ENUM[] = {
        XC_SLATERX,
        XC_VWN5C,
        XC_BECKEX,
        XC_BECKECORRX,
        XC_BECKESRX,
        XC_OPTX,
        XC_LYPC,
        XC_PBEX,
        XC_REVPBEX,
        XC_RPBEX,
        XC_PBEC,
        XC_SPBEC,
        XC_VWN_PBEC,
        XC_LDAERFX,
        XC_LDAERFC,
        XC_LDAERFC_JT,
        XC_RANGESEP_MU,
        XC_KTX,
        XC_TFK,
        XC_PW91X,
        XC_PW91K,
        XC_PW92C,
        XC_M05X,
        XC_M05X2X,
        XC_M06X,
        XC_M06X2X,
        XC_M06LX,
        XC_M06HFX,
        XC_BRX,
        XC_M05X2C,
        XC_M05C,
        XC_M06C,
        XC_M06LC,
        XC_M06X2C,
        XC_TPSSC,
        XC_TPSSX,
        XC_REVTPSSC,
        XC_REVTPSSX,
};

/*
 * XC_LDA      0 // Local density
 * XC_GGA      1 // Local density & gradient
 * XC_MGGA     2 // Local density, gradient and kinetic energy density
 * XC_MLGGA    3 // Local density, gradient, laplacian and kinetic energy density
 */
int XCFUN_xc_type(int fn_id)
{
        xc_functional fun = xc_new_functional();
        assert(fn_id < XC_NR_PARAMS);
        xc_set_param(fun, CONVERT2ENUM[fn_id], 1);
        int type = xc_get_type(fun);
        xc_free_functional(fun);
        return type;
}

int XCFUN_input_length(int nfn, int *fn_id, double *fac)
{
        int i;
        xc_functional fun = xc_new_functional();
        for (i = 0; i < nfn; i++) {
                assert(fn_id[i] < XC_NR_PARAMS);
                xc_set_param(fun, CONVERT2ENUM[fn_id[i]], fac[i]);
        }
        int nvar = xc_input_length(fun);
        xc_free_functional(fun);
        return nvar;
}

void XCFUN_eval_xc(int nfn, int *fn_id, double *fac,
                   int spin, int deriv, int np,
                   double *rho_u, double *rho_d, double *output)
{
        int i, outlen;
        double *rho;
        double *gxu, *gyu, *gzu, *gxd, *gyd, *gzd, *tau_u, *tau_d;
        xc_functional fun = xc_new_functional();
        for (i = 0; i < nfn; i++) {
                assert(fn_id[i] < XC_NR_PARAMS);
                xc_set_param(fun, CONVERT2ENUM[fn_id[i]], fac[i]);
        }

        if (spin == 0) {
                xc_set_mode(fun, XC_VARS_N);
                outlen = xc_output_length(fun, deriv);
                switch (xc_get_type(fun)) {
                case XC_LDA:
                        xc_eval_vec(fun, deriv, np, rho_u, 1, output, outlen);
                        break;
                case XC_GGA:
                        rho = malloc(sizeof(double) * np*2);
                        gxu = rho_u + np;
                        gyu = rho_u + np * 2;
                        gzu = rho_u + np * 3;
                        for (i = 0; i < np; i++) {
                                rho[i*2+0] = rho_u[i];
                                rho[i*2+1] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                        }
                        xc_eval_vec(fun, deriv, np, rho, 2, output, outlen);
                        free(rho);
                        break;
                case XC_MGGA:
                        rho = malloc(sizeof(double) * np*3);
                        gxu = rho_u + np;
                        gyu = rho_u + np * 2;
                        gzu = rho_u + np * 3;
                        tau_u = rho_u + np * 5;
                        for (i = 0; i < np; i++) {
                                rho[i*3+0] = rho_u[i];
                                rho[i*3+1] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                                rho[i*3+2] = tau_u[i];
                        }
                        xc_eval_vec(fun, deriv, np, rho, 3, output, outlen);
                        free(rho);
                        break;
                default:  // XC_MLGGA:
                        fprintf(stderr, "MLGGA not implemented in xcfun\n");
                        exit(1);
                }
// xcfun computed rho*Exc[rho] for zeroth order deriviative instead of Exc[rho]
                for (i = 0; i < np; i++) {
                        output[i*outlen] /= rho_u[i] + 1e-150;
                }
        } else {
                xc_set_mode(fun, XC_VARS_AB);
                outlen = xc_output_length(fun, deriv);
                switch (xc_get_type(fun)) {
                case XC_LDA:
                        rho = malloc(sizeof(double) * np*2);
                        for (i = 0; i < np; i++) {
                                rho[i*2+0] = rho_u[i];
                                rho[i*2+1] = rho_d[i];
                        }
                        xc_eval_vec(fun, deriv, np, rho, 2, output, outlen);
                        free(rho);
                        break;
                case XC_GGA:
                        rho = malloc(sizeof(double) * np*5);
                        gxu = rho_u + np;
                        gyu = rho_u + np * 2;
                        gzu = rho_u + np * 3;
                        gxd = rho_d + np;
                        gyd = rho_d + np * 2;
                        gzd = rho_d + np * 3;
                        for (i = 0; i < np; i++) {
                                rho[i*5+0] = rho_u[i];
                                rho[i*5+1] = rho_d[i];
                                rho[i*5+2] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                                rho[i*5+3] = gxu[i]*gxd[i] + gyu[i]*gyd[i] + gzu[i]*gzd[i];
                                rho[i*5+4] = gxd[i]*gxd[i] + gyd[i]*gyd[i] + gzd[i]*gzd[i];
                        }
                        xc_eval_vec(fun, deriv, np, rho, 5, output, outlen);
                        free(rho);
                        break;
                case XC_MGGA:
                        rho = malloc(sizeof(double) * np*7);
                        gxu = rho_u + np;
                        gyu = rho_u + np * 2;
                        gzu = rho_u + np * 3;
                        gxd = rho_d + np;
                        gyd = rho_d + np * 2;
                        gzd = rho_d + np * 3;
                        tau_u = rho_u + np * 5;
                        tau_d = rho_d + np * 5;
                        for (i = 0; i < np; i++) {
                                rho[i*7+0] = rho_u[i];
                                rho[i*7+1] = rho_d[i];
                                rho[i*7+2] = gxu[i]*gxu[i] + gyu[i]*gyu[i] + gzu[i]*gzu[i];
                                rho[i*7+3] = gxu[i]*gxd[i] + gyu[i]*gyd[i] + gzu[i]*gzd[i];
                                rho[i*7+4] = gxd[i]*gxd[i] + gyd[i]*gyd[i] + gzd[i]*gzd[i];
                                rho[i*7+5] = tau_u[i];
                                rho[i*7+6] = tau_d[i];
                        }
                        xc_eval_vec(fun, deriv, np, rho, 7, output, outlen);
                        free(rho);
                        break;
                default:  // XC_MLGGA:
                        fprintf(stderr, "MLGGA not implemented in xcfun\n");
                        exit(1);
                }
                for (i = 0; i < np; i++) {
                        output[i*outlen] /= rho_u[i] + rho_d[i] + 1e-150;
                }
        }
        xc_free_functional(fun);
}

