/* Copyright 2014-2021 The PySCF Developers. All Rights Reserved.

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
 * Author: Hong-Zhou Ye <hzyechem@gmail.com>
 */

typedef struct {
        size_t i;
        size_t j;
        double fac;
} CacheJob;

const double **_gen_ptr_arr(const double *, const size_t, const size_t);
size_t _MP2_gen_jobs(CacheJob *, const int, const size_t, const size_t, const size_t, const size_t);
void MP2_contract_d(double *, double *, const int,
                    const double *, const double *,
                    const int, const int, const int, const int,
                    const int, const int, const int,
                    const double *, const double *,
                    double *, const int);
void MP2_contract_c(double *, double *, const int,
                    const double *, const double *,
                    const double *, const double *,
                    const int, const int, const int, const int,
                    const int, const int,
                    const double *, const double *);
void MP2_OS_contract_c(double *,
                       const double *, const double *,
                       const double *, const double *,
                       const int, const int, const int, const int,
                       const int, const int, const int,
                       const double *, const double *);
