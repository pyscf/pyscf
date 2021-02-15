/* Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
  
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

#include <stdlib.h>
#include <complex.h>
#include "config.h"

#define MIN(x, y)       ((x) < (y) ? (x) : (y))

//void NPomp_dsum_reduce_inplace(double **vec, size_t count)
//{
//        unsigned int nthreads = omp_get_num_threads();
//        unsigned int thread_id = omp_get_thread_num();
//        unsigned int bit, thread_src;
//        unsigned int mask = 0;
//        double *dst = vec[thread_id];
//        double *src;
//        size_t i;
//#pragma omp barrier
//        for (bit = 0; (1<<bit) < nthreads; bit++) {
//                mask |= 1 << bit;
//                if (!(thread_id & mask)) {
//                        thread_src = thread_id | (1<<bit);
//                        if (thread_src < nthreads) {
//                                src = vec[thread_src];
//                                for (i = 0; i < count; i++) {
//                                        dst[i] += src[i];
//                                }
//                        }
//                }
//#pragma omp barrier
//        }
//
//}

void NPomp_dsum_reduce_inplace(double **vec, size_t count)
{
        unsigned int nthreads = omp_get_num_threads();
        unsigned int thread_id = omp_get_thread_num();
        size_t blksize = (count + nthreads - 1) / nthreads;
        size_t start = thread_id * blksize;
        size_t end = MIN(start + blksize, count);
        double *dst = vec[0];
        double *src;
        size_t it, i;
#pragma omp barrier
        for (it = 1; it < nthreads; it++) {
                src = vec[it];
                for (i = start; i < end; i++) {
                        dst[i] += src[i];
                }
        }
#pragma omp barrier
}

void NPomp_dprod_reduce_inplace(double **vec, size_t count)
{
        unsigned int nthreads = omp_get_num_threads();
        unsigned int thread_id = omp_get_thread_num();
        size_t blksize = (count + nthreads - 1) / nthreads;
        size_t start = thread_id * blksize;
        size_t end = MIN(start + blksize, count);
        double *dst = vec[0];
        double *src;
        size_t it, i;
#pragma omp barrier
        for (it = 1; it < nthreads; it++) {
                src = vec[it];
                for (i = start; i < end; i++) {
                        dst[i] *= src[i];
                }
        }
#pragma omp barrier
}

void NPomp_zsum_reduce_inplace(double complex **vec, size_t count)
{
        unsigned int nthreads = omp_get_num_threads();
        unsigned int thread_id = omp_get_thread_num();
        size_t blksize = (count + nthreads - 1) / nthreads;
        size_t start = thread_id * blksize;
        size_t end = MIN(start + blksize, count);
        double complex *dst = vec[0];
        double complex *src;
        size_t it, i;
#pragma omp barrier
        for (it = 1; it < nthreads; it++) {
                src = vec[it];
                for (i = start; i < end; i++) {
                        dst[i] += src[i];
                }
        }
#pragma omp barrier
}

void NPomp_zprod_reduce_inplace(double complex **vec, size_t count)
{
        unsigned int nthreads = omp_get_num_threads();
        unsigned int thread_id = omp_get_thread_num();
        size_t blksize = (count + nthreads - 1) / nthreads;
        size_t start = thread_id * blksize;
        size_t end = MIN(start + blksize, count);
        double complex *dst = vec[0];
        double complex *src;
        size_t it, i;
#pragma omp barrier
        for (it = 1; it < nthreads; it++) {
                src = vec[it];
                for (i = start; i < end; i++) {
                        dst[i] *= src[i];
                }
        }
#pragma omp barrier
}

#ifdef _OPENMP
int get_omp_threads() {
        return omp_get_max_threads();
}
int set_omp_threads(int n) {
        omp_set_num_threads(n);
        return n;
}
#else
// mimic omp_get_max_threads omp_set_num_threads function of libgomp
int get_omp_threads() { return 1; }
int set_omp_threads(int n) { return 0; }
#endif
