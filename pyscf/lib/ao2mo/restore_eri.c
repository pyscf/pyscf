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

#include <stdlib.h>
#include <complex.h>
#include "np_helper/np_helper.h"


void AO2MOrestore_nr8to1(double *eri8, double *eri1, int norb)
{
        size_t npair = norb*(norb+1)/2;
        size_t i, j, ij;
        size_t d2 = norb * norb;
        size_t d3 = norb * norb * norb;
        double *buf = malloc(sizeof(double)*npair);

        for (ij = 0, i = 0; i < norb; i++) {
        for (j = 0; j < i+1; j++, ij++) {
                NPdunpack_row(npair, ij, eri8, buf);
                NPdunpack_tril(norb, buf, eri1+i*d3+j*d2, HERMITIAN);
                if (i > j) {
                        NPdcopy(eri1+j*d3+i*d2, eri1+i*d3+j*d2, norb*norb);
                }
        } }
        free(buf);
}

void AO2MOrestore_nr4to1(double *eri4, double *eri1, int norb)
{
        size_t npair = norb*(norb+1)/2;
        size_t i, j, ij;
        size_t d2 = norb * norb;
        size_t d3 = norb * norb * norb;

        for (ij = 0, i = 0; i < norb; i++) {
        for (j = 0; j <= i; j++, ij++) {
                NPdunpack_tril(norb, eri4+ij*npair, eri1+i*d3+j*d2, HERMITIAN);
                if (i > j) {
                        NPdcopy(eri1+j*d3+i*d2, eri1+i*d3+j*d2, norb*norb);
                }
        } }
}

void AO2MOrestore_nr1to4(double *eri1, double *eri4, int norb)
{
        size_t npair = norb*(norb+1)/2;
        size_t i, j, k, l, ij, kl;
        size_t d1 = norb;
        size_t d2 = norb * norb;
        size_t d3 = norb * norb * norb;

        for (ij = 0, i = 0; i < norb; i++) {
        for (j = 0; j <= i; j++, ij++) {
                for (kl = 0, k = 0; k < norb; k++) {
                for (l = 0; l <= k; l++, kl++) {
                        eri4[ij*npair+kl] = eri1[i*d3+j*d2+k*d1+l];
                } }
        } }
}

void AO2MOrestore_nr1to8(double *eri1, double *eri8, int norb)
{
        size_t i, j, k, l, ij, kl, ijkl;
        size_t d1 = norb;
        size_t d2 = norb * norb;
        size_t d3 = norb * norb * norb;

        ijkl = 0;
        for (ij = 0, i = 0; i < norb; i++) {
        for (j = 0; j <= i; j++, ij++) {
                for (kl = 0, k = 0; k <= i; k++) {
                for (l = 0; l <= k; l++, kl++) {
                        if (ij >= kl) {
                                eri8[ijkl] = eri1[i*d3+j*d2+k*d1+l];
                                ijkl++;
                        }
                } }
        } }
}

