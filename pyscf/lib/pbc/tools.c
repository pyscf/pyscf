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

#include <stdlib.h>
#include <complex.h>

void gradient_gs(double complex* out, double complex* f_gs, double* Gv,
                 int n, size_t ng)
{
    int i;
    double complex *outx, *outy, *outz;
    for (i = 0; i < n; i++) {
        outx = out;
        outy = outx + ng;
        outz = outy + ng;
        #pragma omp parallel
        {
            size_t igrid;
            double *pGv;
            #pragma omp for schedule(static)
            for (igrid = 0; igrid < ng; igrid++) {
                pGv = Gv + igrid * 3;
                outx[igrid] = pGv[0] * creal(f_gs[igrid]) * _Complex_I - pGv[0] * cimag(f_gs[igrid]);
                outy[igrid] = pGv[1] * creal(f_gs[igrid]) * _Complex_I - pGv[1] * cimag(f_gs[igrid]);
                outz[igrid] = pGv[2] * creal(f_gs[igrid]) * _Complex_I - pGv[2] * cimag(f_gs[igrid]);
            }
        }
        f_gs += ng;
        out += 3 * ng;
    }
}
