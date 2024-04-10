/* Copyright 2014-2024 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 */

#ifndef HAVE_DEFINED_PBC_FILL_INTS_H
#define HAVE_DEFINED_PBC_FILL_INTS_H

void sort2c_gs1(double *out, double *in, int *shls_slice, int *ao_loc,
                int comp, int ish, int jsh);
void sort2c_gs2_igtj(double *out, double *in, int *shls_slice, int *ao_loc,
                     int comp, int ish, int jsh);
void sort2c_gs2_ieqj(double *out, double *in, int *shls_slice, int *ao_loc,
                     int comp, int ish, int jsh);
void sort2c_ks1(double complex *out, double *bufr, double *bufi,
                int *shls_slice, int *ao_loc, int nkpts, int comp,
                int jsh, int msh0, int msh1);
#endif
