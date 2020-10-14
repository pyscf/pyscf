/*  Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
   
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
 *  Author: Oliver J. Backhouse <olbackhouse@gmail.com>
 *          Alejandro Santana-Bonilla <alejandro.santana_bonilla@kcl.ac.uk>
 *          George H. Booth <george.booth@kcl.ac.uk>
 */

#include<stdlib.h>
#include<assert.h>
#include<math.h>

//#include "omp.h"
#include "config.h"
#include "vhf/fblas.h"


void AGF2sum_inplace(double *a, double *b, int x, double alpha, double beta);
void AGF2prod_inplace(double *a, double *b, int x);
void AGF2prod_outplace(double *a, double *b, int x, double *c);
void AGF2slice_0i2(double *a, int x, int y, int z, int idx, double *b);
void AGF2slice_01i(double *a, int x, int y, int z, int idx, double *b);
void AGF2sum_inplace_ener(double a, double *b, double *c, int x, int y, double *d);
void AGF2prod_inplace_ener(double *a, double *b, int x, int y);
void AGF2prod_outplace_ener(double *a, double *b, int x, int y, double *c);
