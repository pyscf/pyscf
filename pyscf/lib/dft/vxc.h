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
