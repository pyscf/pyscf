/* Copyright 2021- The PySCF Developers. All Rights Reserved.

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
 * Author: Xing Zhang <zhangxing.nju@gmail.com>
 */

#ifndef HAVE_DEFINED_CELL_H
#define HAVE_DEFINED_CELL_H

#define RCUT_MAX_CYCLE 10
#define RCUT_EPS 1e-3

double pgf_rcut(int l, double alpha, double coeff, double precision, double r0);
void rcut_by_shells(double* shell_radius, double** ptr_pgf_rcut,
                    int* bas, double* env, int nbas,
                    double r0, double precision);
#endif
