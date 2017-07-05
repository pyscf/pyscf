/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <xc.h>

double VXChybrid_coeff(int xc_id, int xc_polarized);

int VXCinit_libxc(xc_func_type *func_x, xc_func_type *func_c,
                  int x_id, int c_id, int spin, int relativity);
int VXCdel_libxc(xc_func_type *func_x, xc_func_type *func_c);

double VXCnr_vxc(int x_id, int c_id, int spin, int relativity,
                 double *dm, double *exc, double *v,
                 int num_grids, double *coords, double *weights,
                 int *atm, int natm, int *bas, int nbas, double *env);
