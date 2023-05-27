#include <xc.h>
#include <string.h>

/* This is an example init_callback library to reparameterize B97-2
 * functional.
 *
 * Compile with
 *   gcc -shared -fPIC -O3 -I ../../pyscf/lib/deps/include -o 24-reparameterize_xc_functional.so 24-reparameterize_xc_functional.c
 *
 * This library does not call Libxc. It only uses its data structure.
 * We don't need to link to Libxc. */


/* Copy param struct definition from Libxc source `src/gga_xc_b97.c`.
 * It is not defined in a .h file, so we need to copy and paste it here. */
typedef struct {
  double c_x[5], c_ss[5], c_ab[5];
} gga_xc_b97_params;

/* Variable to store the new parameter */
static gga_xc_b97_params my_param;

/* Interface to set new parameters from Python */
void set_param(double *param) {
    memcpy(my_param.c_x,  param,      5 * sizeof(double));
    memcpy(my_param.c_ss, param + 5,  5 * sizeof(double));
    memcpy(my_param.c_ab, param + 10, 5 * sizeof(double));
}

/* This is the callback function to be called from `pyscf/lib/dft/libxc_itrf.c`.
 * It sets the functional parameters to `my_param` every time PySCF fully
 * initializes the xc_func_type struct, before the functional is evaluated. */
void init_callback(xc_func_type *func) {
    if (func->info->number == XC_HYB_GGA_XC_B97_2) {
        memcpy(func->params, &my_param, sizeof(gga_xc_b97_params));
    }
}
