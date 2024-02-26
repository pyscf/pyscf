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

#include <fftw3.h>

#define FFT_PLAN fftw_plan

FFT_PLAN fft_create_r2c_plan(double* in, complex double* out, int rank, int* mesh);
FFT_PLAN fft_create_c2r_plan(complex double* in, double* out, int rank, int* mesh);
void fft_execute(FFT_PLAN p);
void fft_destroy_plan(FFT_PLAN p);
