/* Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
  
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
 * Author: Yu Jin <jinyuchem@uchicago.edu>
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"

/*
 * A unified function for the symmetrization of T3-like object
 * A = beta * A + alpha * op(A)
 */
void t3_symm_ip(double *A, int64_t nocc3, int64_t nvir, char *pattern, double alpha, double beta) {
    int64_t bl = 16;
    int64_t nvv = nvir * nvir;
    double p0 = 0.0, p1 = 0.0, p2 = 0.0, p3 = 0.0, p4 = 0.0;

    if (strcmp(pattern, "111111") == 0) {
        // abc + acb + bac + bca + cab + cba
        // (1 + P_a^b) (1 + P_c^b + P_c^a) abc
        p0 = 1.0;
        p1 = 1.0;
        p2 = 1.0;
        p3 = 1.0;
        p4 = 1.0;

    } else if (strcmp(pattern, "4-2-211-2") == 0) {
        // 4 abc - 2 acb - 2 bac + bca + cab - 2 cba
        // (2 - P_a^b) (2 - P_c^b - P_c^a) abc
        p0 = 2.0;
        p1 = -1.0;
        p2 = -1.0;
        p3 = 2.0;
        p4 = -1.0;

    } else if (strcmp(pattern, "2-1000-1") == 0) {
        // 2 abc - acb + 0 bac + 0 bca + 0 cab - cba
        // (1 - 0 P_a^b) (2 - P_c^b - P_c^a) abc
        p0 = 2.0;
        p1 = -1.0;
        p2 = -1.0;
        p3 = 1.0;
        p4 = 0.0;

    } else {
        fprintf(stderr, "Error: unrecognized pattern \"%s\"\n", pattern);
        return;
    }

#pragma omp parallel for schedule(static)
    for (int64_t ijk = 0; ijk < nocc3; ijk++) {
        int64_t h = ijk * nvir * nvir * nvir;
        for (int64_t a0 = 0; a0 < nvir; a0 += bl) {
            for (int64_t b0 = 0; b0 <= a0; b0 += bl) {
                for (int64_t c0 = 0; c0 <= b0; c0 += bl) {
                    for (int64_t a = a0; a < a0 + bl && a < nvir; a++) {
                        for (int64_t b = b0; b < b0 + bl && b <= a; b++) {
                            for (int64_t c = c0; c < c0 + bl && c <= b; c++) {

                                if (a > b && b > c) {
                                    double T_local[6];

                                    int64_t idx1 = a * nvv + b * nvir + c; // abc
                                    int64_t idx2 = c * nvv + b * nvir + a; // cba
                                    int64_t idx3 = a * nvv + c * nvir + b; // acb
                                    int64_t idx4 = b * nvv + a * nvir + c; // bac
                                    int64_t idx5 = b * nvv + c * nvir + a; // bca
                                    int64_t idx6 = c * nvv + a * nvir + b; // cab

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx2] + p2 * A[h + idx3]; // abc -> cba -> acb
                                    T_local[1] = p0 * A[h + idx2] + p1 * A[h + idx1] + p2 * A[h + idx6]; // cba -> abc -> cab
                                    T_local[2] = p0 * A[h + idx3] + p1 * A[h + idx5] + p2 * A[h + idx1]; // acb -> bca -> abc
                                    T_local[3] = p0 * A[h + idx4] + p1 * A[h + idx6] + p2 * A[h + idx5]; // bac -> cab -> bca
                                    T_local[4] = p0 * A[h + idx5] + p1 * A[h + idx3] + p2 * A[h + idx4]; // bca -> acb -> bac
                                    T_local[5] = p0 * A[h + idx6] + p1 * A[h + idx4] + p2 * A[h + idx2]; // cab -> bac -> cba

                                    A[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[3]); // abc -> bac
                                    A[h + idx2] = beta * A[h + idx2] + alpha * (p3 * T_local[1] + p4 * T_local[4]); // cba -> bca
                                    A[h + idx3] = beta * A[h + idx3] + alpha * (p3 * T_local[2] + p4 * T_local[5]); // acb -> cab
                                    A[h + idx4] = beta * A[h + idx4] + alpha * (p3 * T_local[3] + p4 * T_local[0]); // bac -> abc
                                    A[h + idx5] = beta * A[h + idx5] + alpha * (p3 * T_local[4] + p4 * T_local[1]); // bca -> cba
                                    A[h + idx6] = beta * A[h + idx6] + alpha * (p3 * T_local[5] + p4 * T_local[2]); // cab -> acb
                                }

                                else if (a > b && b == c) {
                                    double T_local[3];

                                    int64_t idx1 = a * nvv + b * nvir + b; // abb
                                    int64_t idx2 = b * nvv + b * nvir + a; // bba
                                    int64_t idx4 = b * nvv + a * nvir + b; // bab

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx2] + p2 * A[h + idx1]; // abb -> bba -> abb
                                    T_local[1] = p0 * A[h + idx2] + p1 * A[h + idx1] + p2 * A[h + idx4]; // bba -> abb -> bab
                                    T_local[2] = p0 * A[h + idx4] + p1 * A[h + idx4] + p2 * A[h + idx2]; // bab -> bab -> bba

                                    A[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[2]); // abb -> bab
                                    A[h + idx2] = beta * A[h + idx2] + alpha * (p3 * T_local[1] + p4 * T_local[1]); // bba -> bba
                                    A[h + idx4] = beta * A[h + idx4] + alpha * (p3 * T_local[2] + p4 * T_local[0]); // bab -> abb
                                }

                                else if (a == b && b > c) {

                                    double T_local[3];

                                    int64_t idx1 = a * nvv + a * nvir + c; // aac
                                    int64_t idx2 = c * nvv + a * nvir + a; // caa
                                    int64_t idx3 = a * nvv + c * nvir + a; // aca

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx2] + p2 * A[h + idx3]; // aac -> caa -> aca
                                    T_local[1] = p0 * A[h + idx2] + p1 * A[h + idx1] + p2 * A[h + idx2]; // caa -> aac -> caa
                                    T_local[2] = p0 * A[h + idx3] + p1 * A[h + idx3] + p2 * A[h + idx1]; // aca -> aca -> aac

                                    A[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[0]); // aac -> aac
                                    A[h + idx2] = beta * A[h + idx2] + alpha * (p3 * T_local[1] + p4 * T_local[2]); // caa -> aca
                                    A[h + idx3] = beta * A[h + idx3] + alpha * (p3 * T_local[2] + p4 * T_local[1]); // aca -> caa

                                } else if (a == b && b == c) {
                                    double T_local[1];

                                    int64_t idx1 = a * nvv + a * nvir + a; // aaa

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx1] + p2 * A[h + idx1]; // abc -> cba -> acb

                                    A[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[0]); // abc -> bac
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
