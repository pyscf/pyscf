#ifndef HAVE_DEFINED_PBC_FILL_INTS_H
#define HAVE_DEFINED_PBC_FILL_INTS_H

void sort2c_gs1(double *out, double *in, int *shls_slice, int *ao_loc,
                int comp, int ish, int jsh);
void sort2c_gs2_igtj(double *out, double *in, int *shls_slice, int *ao_loc,
                     int comp, int ish, int jsh);
void sort2c_gs2_ieqj(double *out, double *in, int *shls_slice, int *ao_loc,
                     int comp, int ish, int jsh);
#endif
