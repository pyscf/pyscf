/*
 * File: libxc_itrf.c
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 * libxc from
 * http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:manual
 */

#include <stdio.h>
#include <stdlib.h>
#include <xc.h>

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
                        factor = xc_hyb_gga_exx_coef(func.gga);
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
