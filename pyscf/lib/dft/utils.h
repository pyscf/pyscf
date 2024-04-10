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

#ifndef HAVE_DEFINED_GRID_UTILS_H
#define HAVE_DEFINED_GRID_UTILS_H

extern void dgemm_wrapper(const char transa, const char transb,
                   const int m, const int n, const int k,
                   const double alpha, const double* a, const int lda,
                   const double* b, const int ldb,
                   const double beta, double* c, const int ldc);
#endif
