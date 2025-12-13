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
 * Author: Yu Jin <yjin@flatironinstitute.org>
 *         Huanchen Zhai <hczhai.ok@gmail.com>
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Apply spin summation projection to T4 amplitudes in place.
// A: pointer to T4 tensor (size nocc4 * nvir**4)
// pattern: "P4_full" : P(A) = (1 + P_c^d) (1 + P_b^c + P_b^d) (1 + P_a^b + P_a^c + P_a^d) A
//          "P4_422"  : P(A) = (1 + 0 * P_c^d) (1 + 0 * P_b^c + 0 * P_b^d) (2 - P_a^b - P_a^c - P_a^d) A
//          "P4_201"  : P(A) = (1 + 0 * P_c^d) (2 - P_b^c - P_b^d) (2 - P_a^b - P_a^c - P_a^d) A
// alpha, beta: A = beta * A + alpha * P(A)
void t4_spin_summation_inplace_(double *A, int64_t nocc4, int64_t nvir, char *pattern, double alpha, double beta)
{
    int64_t ijkl;
    const int64_t bl = 8;
    int64_t nvv = nvir * nvir;
    int64_t nvvv = nvir * nvv;
    int64_t nvvvv = nvir * nvvv;

    double p[9];

    if (strcmp(pattern, "P4_full") == 0)
    {
        for (int i = 0; i < 9; i++)
            p[i] = 1.0;
    }
    else if (strcmp(pattern, "P4_201") == 0)
    {
        p[0] = 2.0;
        p[1] = -1.0;
        p[2] = -1.0;
        p[3] = -1.0;
        p[4] = 1.0;
        p[5] = 0.0;
        p[6] = 0.0;
        p[7] = 1.0;
        p[8] = 0.0;
    }
    else if (strcmp(pattern, "P4_442") == 0)
    {
        p[0] = 2.0;
        p[1] = -1.0;
        p[2] = -1.0;
        p[3] = -1.0;
        p[4] = 2.0;
        p[5] = -1.0;
        p[6] = -1.0;
        p[7] = 1.0;
        p[8] = 0.0;
    }
    else
    {
        fprintf(stderr, "Error: unrecognized pattern \"%s\"\n", pattern);
        return;
    }

#pragma omp parallel for schedule(static)
    for (ijkl = 0; ijkl < nocc4; ijkl++)
    {
        int64_t h = ijkl * nvvvv;
        for (int64_t a0 = 0; a0 < nvir; a0 += bl)
        {
            for (int64_t b0 = 0; b0 <= a0; b0 += bl)
            {
                for (int64_t c0 = 0; c0 <= b0; c0 += bl)
                {
                    for (int64_t d0 = 0; d0 <= c0; d0 += bl)
                    {
                        for (int64_t a = a0; a < a0 + bl && a < nvir; a++)
                        {
                            for (int64_t b = b0; b < b0 + bl && b <= a; b++)
                            {
                                for (int64_t c = c0; c < c0 + bl && c <= b; c++)
                                {
                                    for (int64_t d = d0; d < d0 + bl && d <= c; d++)
                                    {
                                        if (a > b && b > c && c > d)
                                        {
                                            double T1_local[24];
                                            double T2_local[24];

                                            int64_t indices[24];
                                            indices[0] = a * nvvv + b * nvv + c * nvir + d;
                                            indices[1] = a * nvvv + b * nvv + d * nvir + c;
                                            indices[2] = a * nvvv + c * nvv + b * nvir + d;
                                            indices[3] = a * nvvv + c * nvv + d * nvir + b;
                                            indices[4] = a * nvvv + d * nvv + b * nvir + c;
                                            indices[5] = a * nvvv + d * nvv + c * nvir + b;
                                            indices[6] = b * nvvv + a * nvv + c * nvir + d;
                                            indices[7] = b * nvvv + a * nvv + d * nvir + c;
                                            indices[8] = b * nvvv + c * nvv + a * nvir + d;
                                            indices[9] = b * nvvv + c * nvv + d * nvir + a;
                                            indices[10] = b * nvvv + d * nvv + a * nvir + c;
                                            indices[11] = b * nvvv + d * nvv + c * nvir + a;
                                            indices[12] = c * nvvv + a * nvv + b * nvir + d;
                                            indices[13] = c * nvvv + a * nvv + d * nvir + b;
                                            indices[14] = c * nvvv + b * nvv + a * nvir + d;
                                            indices[15] = c * nvvv + b * nvv + d * nvir + a;
                                            indices[16] = c * nvvv + d * nvv + a * nvir + b;
                                            indices[17] = c * nvvv + d * nvv + b * nvir + a;
                                            indices[18] = d * nvvv + a * nvv + b * nvir + c;
                                            indices[19] = d * nvvv + a * nvv + c * nvir + b;
                                            indices[20] = d * nvvv + b * nvv + a * nvir + c;
                                            indices[21] = d * nvvv + b * nvv + c * nvir + a;
                                            indices[22] = d * nvvv + c * nvv + a * nvir + b;
                                            indices[23] = d * nvvv + c * nvv + b * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[14]] + p[3] * A[h + indices[21]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[20]] + p[3] * A[h + indices[15]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[12]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[23]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[13]] + p[2] * A[h + indices[22]] + p[3] * A[h + indices[9]];
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[18]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[17]];
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[19]] + p[2] * A[h + indices[16]] + p[3] * A[h + indices[11]];
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[12]] + p[3] * A[h + indices[19]];
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[18]] + p[3] * A[h + indices[13]];
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[14]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[22]];
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[15]] + p[2] * A[h + indices[23]] + p[3] * A[h + indices[3]];
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[20]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[16]];
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[21]] + p[2] * A[h + indices[17]] + p[3] * A[h + indices[5]];
                                            T1_local[12] = p[0] * A[h + indices[12]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[18]];
                                            T1_local[13] = p[0] * A[h + indices[13]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[19]] + p[3] * A[h + indices[7]];
                                            T1_local[14] = p[0] * A[h + indices[14]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[20]];
                                            T1_local[15] = p[0] * A[h + indices[15]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[21]] + p[3] * A[h + indices[1]];
                                            T1_local[16] = p[0] * A[h + indices[16]] + p[1] * A[h + indices[22]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[10]];
                                            T1_local[17] = p[0] * A[h + indices[17]] + p[1] * A[h + indices[23]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[4]];
                                            T1_local[18] = p[0] * A[h + indices[18]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[12]];
                                            T1_local[19] = p[0] * A[h + indices[19]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[13]] + p[3] * A[h + indices[6]];
                                            T1_local[20] = p[0] * A[h + indices[20]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[14]];
                                            T1_local[21] = p[0] * A[h + indices[21]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[15]] + p[3] * A[h + indices[0]];
                                            T1_local[22] = p[0] * A[h + indices[22]] + p[1] * A[h + indices[16]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[8]];
                                            T1_local[23] = p[0] * A[h + indices[23]] + p[1] * A[h + indices[17]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[2]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[2] + p[6] * T1_local[5];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[4] + p[6] * T1_local[3];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[0] + p[6] * T1_local[4];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[5] + p[6] * T1_local[1];
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[1] + p[6] * T1_local[2];
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[3] + p[6] * T1_local[0];
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[8] + p[6] * T1_local[11];
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[10] + p[6] * T1_local[9];
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[6] + p[6] * T1_local[10];
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[11] + p[6] * T1_local[7];
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[7] + p[6] * T1_local[8];
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[9] + p[6] * T1_local[6];
                                            T2_local[12] = p[4] * T1_local[12] + p[5] * T1_local[14] + p[6] * T1_local[17];
                                            T2_local[13] = p[4] * T1_local[13] + p[5] * T1_local[16] + p[6] * T1_local[15];
                                            T2_local[14] = p[4] * T1_local[14] + p[5] * T1_local[12] + p[6] * T1_local[16];
                                            T2_local[15] = p[4] * T1_local[15] + p[5] * T1_local[17] + p[6] * T1_local[13];
                                            T2_local[16] = p[4] * T1_local[16] + p[5] * T1_local[13] + p[6] * T1_local[14];
                                            T2_local[17] = p[4] * T1_local[17] + p[5] * T1_local[15] + p[6] * T1_local[12];
                                            T2_local[18] = p[4] * T1_local[18] + p[5] * T1_local[20] + p[6] * T1_local[23];
                                            T2_local[19] = p[4] * T1_local[19] + p[5] * T1_local[22] + p[6] * T1_local[21];
                                            T2_local[20] = p[4] * T1_local[20] + p[5] * T1_local[18] + p[6] * T1_local[22];
                                            T2_local[21] = p[4] * T1_local[21] + p[5] * T1_local[23] + p[6] * T1_local[19];
                                            T2_local[22] = p[4] * T1_local[22] + p[5] * T1_local[19] + p[6] * T1_local[20];
                                            T2_local[23] = p[4] * T1_local[23] + p[5] * T1_local[21] + p[6] * T1_local[18];

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];
                                            A[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];
                                            A[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];
                                            A[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];
                                            A[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];
                                            A[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[9]) + beta * A[h + indices[8]];
                                            A[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[8]) + beta * A[h + indices[9]];
                                            A[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]];
                                            A[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]];
                                            A[h + indices[12]] = alpha * (p[7] * T2_local[12] + p[8] * T2_local[13]) + beta * A[h + indices[12]];
                                            A[h + indices[13]] = alpha * (p[7] * T2_local[13] + p[8] * T2_local[12]) + beta * A[h + indices[13]];
                                            A[h + indices[14]] = alpha * (p[7] * T2_local[14] + p[8] * T2_local[15]) + beta * A[h + indices[14]];
                                            A[h + indices[15]] = alpha * (p[7] * T2_local[15] + p[8] * T2_local[14]) + beta * A[h + indices[15]];
                                            A[h + indices[16]] = alpha * (p[7] * T2_local[16] + p[8] * T2_local[17]) + beta * A[h + indices[16]];
                                            A[h + indices[17]] = alpha * (p[7] * T2_local[17] + p[8] * T2_local[16]) + beta * A[h + indices[17]];
                                            A[h + indices[18]] = alpha * (p[7] * T2_local[18] + p[8] * T2_local[19]) + beta * A[h + indices[18]];
                                            A[h + indices[19]] = alpha * (p[7] * T2_local[19] + p[8] * T2_local[18]) + beta * A[h + indices[19]];
                                            A[h + indices[20]] = alpha * (p[7] * T2_local[20] + p[8] * T2_local[21]) + beta * A[h + indices[20]];
                                            A[h + indices[21]] = alpha * (p[7] * T2_local[21] + p[8] * T2_local[20]) + beta * A[h + indices[21]];
                                            A[h + indices[22]] = alpha * (p[7] * T2_local[22] + p[8] * T2_local[23]) + beta * A[h + indices[22]];
                                            A[h + indices[23]] = alpha * (p[7] * T2_local[23] + p[8] * T2_local[22]) + beta * A[h + indices[23]];
                                        }
                                        else if (a > b && b > c && c == d)
                                        {
                                            double T1_local[12];
                                            double T2_local[12];

                                            int64_t indices[12];
                                            indices[0] = a * nvvv + b * nvv + c * nvir + c;
                                            indices[1] = a * nvvv + c * nvv + b * nvir + c;
                                            indices[2] = a * nvvv + c * nvv + c * nvir + b;
                                            indices[3] = b * nvvv + a * nvv + c * nvir + c;
                                            indices[4] = b * nvvv + c * nvv + a * nvir + c;
                                            indices[5] = b * nvvv + c * nvv + c * nvir + a;
                                            indices[6] = c * nvvv + a * nvv + b * nvir + c;
                                            indices[7] = c * nvvv + a * nvv + c * nvir + b;
                                            indices[8] = c * nvvv + b * nvv + a * nvir + c;
                                            indices[9] = c * nvvv + b * nvv + c * nvir + a;
                                            indices[10] = c * nvvv + c * nvv + a * nvir + b;
                                            indices[11] = c * nvvv + c * nvv + b * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[9]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[11]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[5]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[7]];
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[10]];
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[2]];
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[6]];
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[3]];
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[8]];
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[0]];
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[4]];
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[1]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[1] + p[6] * T1_local[2];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[0] + p[6] * T1_local[1];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[2] + p[6] * T1_local[0];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[4] + p[6] * T1_local[5];
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[3] + p[6] * T1_local[4];
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[5] + p[6] * T1_local[3];
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[8] + p[6] * T1_local[11];
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[10] + p[6] * T1_local[9];
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[6] + p[6] * T1_local[10];
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[11] + p[6] * T1_local[7];
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[7] + p[6] * T1_local[8];
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[9] + p[6] * T1_local[6];

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[2]) + beta * A[h + indices[1]];
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[1]) + beta * A[h + indices[2]];
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[3]) + beta * A[h + indices[3]];
                                            A[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];
                                            A[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];
                                            A[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];
                                            A[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];
                                            A[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[9]) + beta * A[h + indices[8]];
                                            A[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[8]) + beta * A[h + indices[9]];
                                            A[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]];
                                            A[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]];
                                        }
                                        else if (a > b && b == c && c > d)
                                        {
                                            double T1_local[12];
                                            double T2_local[12];

                                            int64_t indices[12];
                                            indices[0] = a * nvvv + b * nvv + b * nvir + d;
                                            indices[1] = a * nvvv + b * nvv + d * nvir + b;
                                            indices[2] = a * nvvv + d * nvv + b * nvir + b;
                                            indices[3] = b * nvvv + a * nvv + b * nvir + d;
                                            indices[4] = b * nvvv + a * nvv + d * nvir + b;
                                            indices[5] = b * nvvv + b * nvv + a * nvir + d;
                                            indices[6] = b * nvvv + b * nvv + d * nvir + a;
                                            indices[7] = b * nvvv + d * nvv + a * nvir + b;
                                            indices[8] = b * nvvv + d * nvv + b * nvir + a;
                                            indices[9] = d * nvvv + a * nvv + b * nvir + b;
                                            indices[10] = d * nvvv + b * nvv + a * nvir + b;
                                            indices[11] = d * nvvv + b * nvv + b * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[11]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[6]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[8]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[9]];
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[4]];
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[10]];
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[1]];
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[7]];
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[2]];
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[3]];
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[5]];
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[0]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[2];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[1];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[0];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[5] + p[6] * T1_local[8];
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[7] + p[6] * T1_local[6];
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[3] + p[6] * T1_local[7];
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[8] + p[6] * T1_local[4];
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[4] + p[6] * T1_local[5];
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[6] + p[6] * T1_local[3];
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[10] + p[6] * T1_local[11];
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[9] + p[6] * T1_local[10];
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[11] + p[6] * T1_local[9];

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[2]) + beta * A[h + indices[2]];
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[4]) + beta * A[h + indices[3]];
                                            A[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[3]) + beta * A[h + indices[4]];
                                            A[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[6]) + beta * A[h + indices[5]];
                                            A[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[5]) + beta * A[h + indices[6]];
                                            A[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[8]) + beta * A[h + indices[7]];
                                            A[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[7]) + beta * A[h + indices[8]];
                                            A[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[9]) + beta * A[h + indices[9]];
                                            A[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]];
                                            A[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]];
                                        }
                                        else if (a == b && b > c && c > d)
                                        {
                                            double T1_local[12];
                                            double T2_local[12];

                                            int64_t indices[12];
                                            indices[0] = a * nvvv + a * nvv + c * nvir + d;
                                            indices[1] = a * nvvv + a * nvv + d * nvir + c;
                                            indices[2] = a * nvvv + c * nvv + a * nvir + d;
                                            indices[3] = a * nvvv + c * nvv + d * nvir + a;
                                            indices[4] = a * nvvv + d * nvv + a * nvir + c;
                                            indices[5] = a * nvvv + d * nvv + c * nvir + a;
                                            indices[6] = c * nvvv + a * nvv + a * nvir + d;
                                            indices[7] = c * nvvv + a * nvv + d * nvir + a;
                                            indices[8] = c * nvvv + d * nvv + a * nvir + a;
                                            indices[9] = d * nvvv + a * nvv + a * nvir + c;
                                            indices[10] = d * nvvv + a * nvv + c * nvir + a;
                                            indices[11] = d * nvvv + c * nvv + a * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[10]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[7]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[11]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[3]];
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[8]];
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[5]];
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[9]];
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[1]];
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[4]];
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[6]];
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[0]];
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[2]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[2] + p[6] * T1_local[5];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[4] + p[6] * T1_local[3];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[0] + p[6] * T1_local[4];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[5] + p[6] * T1_local[1];
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[1] + p[6] * T1_local[2];
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[3] + p[6] * T1_local[0];
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[6] + p[6] * T1_local[8];
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[8] + p[6] * T1_local[7];
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[7] + p[6] * T1_local[6];
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[9] + p[6] * T1_local[11];
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[11] + p[6] * T1_local[10];
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[10] + p[6] * T1_local[9];

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];
                                            A[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];
                                            A[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];
                                            A[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];
                                            A[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];
                                            A[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[8]) + beta * A[h + indices[8]];
                                            A[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[10]) + beta * A[h + indices[9]];
                                            A[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[9]) + beta * A[h + indices[10]];
                                            A[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[11]) + beta * A[h + indices[11]];
                                        }
                                        else if (a > b && b == c && c == d)
                                        {
                                            double T1_local[4];
                                            double T2_local[4];

                                            int64_t indices[4];
                                            indices[0] = a * nvvv + b * nvv + b * nvir + b;
                                            indices[1] = b * nvvv + a * nvv + b * nvir + b;
                                            indices[2] = b * nvvv + b * nvv + a * nvir + b;
                                            indices[3] = b * nvvv + b * nvv + b * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[3]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[1]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[2]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[0]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[0];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[3];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[2];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[1];

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[1]) + beta * A[h + indices[1]];
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];
                                        }
                                        else if (a == b && b == c && c > d)
                                        {
                                            double T1_local[4];
                                            double T2_local[4];

                                            int64_t indices[4];
                                            indices[0] = a * nvvv + a * nvv + a * nvir + d;
                                            indices[1] = a * nvvv + a * nvv + d * nvir + a;
                                            indices[2] = a * nvvv + d * nvv + a * nvir + a;
                                            indices[3] = d * nvvv + a * nvv + a * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[3]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[1]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[2]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[0]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[2];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[1];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[0];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[3];

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[2]) + beta * A[h + indices[2]];
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[3]) + beta * A[h + indices[3]];
                                        }
                                        else if (a == b && b > c && c == d)
                                        {
                                            double T1_local[6];
                                            double T2_local[6];

                                            int64_t indices[6];
                                            indices[0] = b * nvvv + b * nvv + c * nvir + c;
                                            indices[1] = b * nvvv + c * nvv + b * nvir + c;
                                            indices[2] = b * nvvv + c * nvv + c * nvir + b;
                                            indices[3] = c * nvvv + b * nvv + b * nvir + c;
                                            indices[4] = c * nvvv + b * nvv + c * nvir + b;
                                            indices[5] = c * nvvv + c * nvv + b * nvir + b;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[4]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[5]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[2]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[3]];
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[0]];
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[1]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[1] + p[6] * T1_local[2];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[0] + p[6] * T1_local[1];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[2] + p[6] * T1_local[0];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[5];
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[5] + p[6] * T1_local[4];
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[4] + p[6] * T1_local[3];

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                                            A[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[2]) + beta * A[h + indices[1]];
                                            A[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[1]) + beta * A[h + indices[2]];
                                            A[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[4]) + beta * A[h + indices[3]];
                                            A[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[3]) + beta * A[h + indices[4]];
                                            A[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[5]) + beta * A[h + indices[5]];
                                        }
                                        else if (a == b && b == c && c == d)
                                        {
                                            double T1_local[1];
                                            double T2_local[1];

                                            int64_t indices[1];
                                            indices[0] = a * nvvv + a * nvv + a * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[0]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[0];

                                            A[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Apply spin summation projection to T4 amplitudes in place.
// A: pointer to T4 tensor (size nocc4 * nvir**4)
// B: pointer to transformed T4 tensor
// alpha, beta: B = beta * A + alpha * P(A)
void t4_spin_summation(const double *A, double *B, int64_t nocc4, int64_t nvir,
                       char *pattern, double alpha, double beta)
{
    int64_t ijkl;
    const int64_t bl = 8;
    int64_t nvv = nvir * nvir;
    int64_t nvvv = nvir * nvv;
    int64_t nvvvv = nvir * nvvv;

    double p[9];

    if (strcmp(pattern, "P4_full") == 0)
    {
        for (int i = 0; i < 9; i++)
            p[i] = 1.0;
    }
    else if (strcmp(pattern, "P4_201") == 0)
    {
        p[0] = 2.0;
        p[1] = -1.0;
        p[2] = -1.0;
        p[3] = -1.0;
        p[4] = 1.0;
        p[5] = 0.0;
        p[6] = 0.0;
        p[7] = 1.0;
        p[8] = 0.0;
    }
    else if (strcmp(pattern, "P4_442") == 0)
    {
        p[0] = 2.0;
        p[1] = -1.0;
        p[2] = -1.0;
        p[3] = -1.0;
        p[4] = 2.0;
        p[5] = -1.0;
        p[6] = -1.0;
        p[7] = 1.0;
        p[8] = 0.0;
    }
    else
    {
        fprintf(stderr, "Error: unrecognized pattern \"%s\"\n", pattern);
        return;
    }

#pragma omp parallel for schedule(static)
    for (ijkl = 0; ijkl < nocc4; ijkl++)
    {
        int64_t h = ijkl * nvvvv;
        for (int64_t a0 = 0; a0 < nvir; a0 += bl)
        {
            for (int64_t b0 = 0; b0 <= a0; b0 += bl)
            {
                for (int64_t c0 = 0; c0 <= b0; c0 += bl)
                {
                    for (int64_t d0 = 0; d0 <= c0; d0 += bl)
                    {
                        for (int64_t a = a0; a < a0 + bl && a < nvir; a++)
                        {
                            for (int64_t b = b0; b < b0 + bl && b <= a; b++)
                            {
                                for (int64_t c = c0; c < c0 + bl && c <= b; c++)
                                {
                                    for (int64_t d = d0; d < d0 + bl && d <= c; d++)
                                    {
                                        if (a > b && b > c && c > d)
                                        {
                                            double T1_local[24];
                                            double T2_local[24];

                                            int64_t indices[24];
                                            indices[0] = a * nvvv + b * nvv + c * nvir + d;
                                            indices[1] = a * nvvv + b * nvv + d * nvir + c;
                                            indices[2] = a * nvvv + c * nvv + b * nvir + d;
                                            indices[3] = a * nvvv + c * nvv + d * nvir + b;
                                            indices[4] = a * nvvv + d * nvv + b * nvir + c;
                                            indices[5] = a * nvvv + d * nvv + c * nvir + b;
                                            indices[6] = b * nvvv + a * nvv + c * nvir + d;
                                            indices[7] = b * nvvv + a * nvv + d * nvir + c;
                                            indices[8] = b * nvvv + c * nvv + a * nvir + d;
                                            indices[9] = b * nvvv + c * nvv + d * nvir + a;
                                            indices[10] = b * nvvv + d * nvv + a * nvir + c;
                                            indices[11] = b * nvvv + d * nvv + c * nvir + a;
                                            indices[12] = c * nvvv + a * nvv + b * nvir + d;
                                            indices[13] = c * nvvv + a * nvv + d * nvir + b;
                                            indices[14] = c * nvvv + b * nvv + a * nvir + d;
                                            indices[15] = c * nvvv + b * nvv + d * nvir + a;
                                            indices[16] = c * nvvv + d * nvv + a * nvir + b;
                                            indices[17] = c * nvvv + d * nvv + b * nvir + a;
                                            indices[18] = d * nvvv + a * nvv + b * nvir + c;
                                            indices[19] = d * nvvv + a * nvv + c * nvir + b;
                                            indices[20] = d * nvvv + b * nvv + a * nvir + c;
                                            indices[21] = d * nvvv + b * nvv + c * nvir + a;
                                            indices[22] = d * nvvv + c * nvv + a * nvir + b;
                                            indices[23] = d * nvvv + c * nvv + b * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[14]] + p[3] * A[h + indices[21]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[20]] + p[3] * A[h + indices[15]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[12]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[23]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[13]] + p[2] * A[h + indices[22]] + p[3] * A[h + indices[9]];
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[18]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[17]];
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[19]] + p[2] * A[h + indices[16]] + p[3] * A[h + indices[11]];
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[12]] + p[3] * A[h + indices[19]];
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[18]] + p[3] * A[h + indices[13]];
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[14]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[22]];
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[15]] + p[2] * A[h + indices[23]] + p[3] * A[h + indices[3]];
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[20]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[16]];
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[21]] + p[2] * A[h + indices[17]] + p[3] * A[h + indices[5]];
                                            T1_local[12] = p[0] * A[h + indices[12]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[18]];
                                            T1_local[13] = p[0] * A[h + indices[13]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[19]] + p[3] * A[h + indices[7]];
                                            T1_local[14] = p[0] * A[h + indices[14]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[20]];
                                            T1_local[15] = p[0] * A[h + indices[15]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[21]] + p[3] * A[h + indices[1]];
                                            T1_local[16] = p[0] * A[h + indices[16]] + p[1] * A[h + indices[22]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[10]];
                                            T1_local[17] = p[0] * A[h + indices[17]] + p[1] * A[h + indices[23]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[4]];
                                            T1_local[18] = p[0] * A[h + indices[18]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[12]];
                                            T1_local[19] = p[0] * A[h + indices[19]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[13]] + p[3] * A[h + indices[6]];
                                            T1_local[20] = p[0] * A[h + indices[20]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[14]];
                                            T1_local[21] = p[0] * A[h + indices[21]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[15]] + p[3] * A[h + indices[0]];
                                            T1_local[22] = p[0] * A[h + indices[22]] + p[1] * A[h + indices[16]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[8]];
                                            T1_local[23] = p[0] * A[h + indices[23]] + p[1] * A[h + indices[17]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[2]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[2] + p[6] * T1_local[5];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[4] + p[6] * T1_local[3];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[0] + p[6] * T1_local[4];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[5] + p[6] * T1_local[1];
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[1] + p[6] * T1_local[2];
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[3] + p[6] * T1_local[0];
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[8] + p[6] * T1_local[11];
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[10] + p[6] * T1_local[9];
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[6] + p[6] * T1_local[10];
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[11] + p[6] * T1_local[7];
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[7] + p[6] * T1_local[8];
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[9] + p[6] * T1_local[6];
                                            T2_local[12] = p[4] * T1_local[12] + p[5] * T1_local[14] + p[6] * T1_local[17];
                                            T2_local[13] = p[4] * T1_local[13] + p[5] * T1_local[16] + p[6] * T1_local[15];
                                            T2_local[14] = p[4] * T1_local[14] + p[5] * T1_local[12] + p[6] * T1_local[16];
                                            T2_local[15] = p[4] * T1_local[15] + p[5] * T1_local[17] + p[6] * T1_local[13];
                                            T2_local[16] = p[4] * T1_local[16] + p[5] * T1_local[13] + p[6] * T1_local[14];
                                            T2_local[17] = p[4] * T1_local[17] + p[5] * T1_local[15] + p[6] * T1_local[12];
                                            T2_local[18] = p[4] * T1_local[18] + p[5] * T1_local[20] + p[6] * T1_local[23];
                                            T2_local[19] = p[4] * T1_local[19] + p[5] * T1_local[22] + p[6] * T1_local[21];
                                            T2_local[20] = p[4] * T1_local[20] + p[5] * T1_local[18] + p[6] * T1_local[22];
                                            T2_local[21] = p[4] * T1_local[21] + p[5] * T1_local[23] + p[6] * T1_local[19];
                                            T2_local[22] = p[4] * T1_local[22] + p[5] * T1_local[19] + p[6] * T1_local[20];
                                            T2_local[23] = p[4] * T1_local[23] + p[5] * T1_local[21] + p[6] * T1_local[18];

                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                                            B[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                                            B[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];
                                            B[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];
                                            B[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];
                                            B[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];
                                            B[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];
                                            B[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];
                                            B[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[9]) + beta * A[h + indices[8]];
                                            B[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[8]) + beta * A[h + indices[9]];
                                            B[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]];
                                            B[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]];
                                            B[h + indices[12]] = alpha * (p[7] * T2_local[12] + p[8] * T2_local[13]) + beta * A[h + indices[12]];
                                            B[h + indices[13]] = alpha * (p[7] * T2_local[13] + p[8] * T2_local[12]) + beta * A[h + indices[13]];
                                            B[h + indices[14]] = alpha * (p[7] * T2_local[14] + p[8] * T2_local[15]) + beta * A[h + indices[14]];
                                            B[h + indices[15]] = alpha * (p[7] * T2_local[15] + p[8] * T2_local[14]) + beta * A[h + indices[15]];
                                            B[h + indices[16]] = alpha * (p[7] * T2_local[16] + p[8] * T2_local[17]) + beta * A[h + indices[16]];
                                            B[h + indices[17]] = alpha * (p[7] * T2_local[17] + p[8] * T2_local[16]) + beta * A[h + indices[17]];
                                            B[h + indices[18]] = alpha * (p[7] * T2_local[18] + p[8] * T2_local[19]) + beta * A[h + indices[18]];
                                            B[h + indices[19]] = alpha * (p[7] * T2_local[19] + p[8] * T2_local[18]) + beta * A[h + indices[19]];
                                            B[h + indices[20]] = alpha * (p[7] * T2_local[20] + p[8] * T2_local[21]) + beta * A[h + indices[20]];
                                            B[h + indices[21]] = alpha * (p[7] * T2_local[21] + p[8] * T2_local[20]) + beta * A[h + indices[21]];
                                            B[h + indices[22]] = alpha * (p[7] * T2_local[22] + p[8] * T2_local[23]) + beta * A[h + indices[22]];
                                            B[h + indices[23]] = alpha * (p[7] * T2_local[23] + p[8] * T2_local[22]) + beta * A[h + indices[23]];
                                        }
                                        else if (a > b && b > c && c == d)
                                        {
                                            double T1_local[12];
                                            double T2_local[12];

                                            int64_t indices[12];
                                            indices[0] = a * nvvv + b * nvv + c * nvir + c;
                                            indices[1] = a * nvvv + c * nvv + b * nvir + c;
                                            indices[2] = a * nvvv + c * nvv + c * nvir + b;
                                            indices[3] = b * nvvv + a * nvv + c * nvir + c;
                                            indices[4] = b * nvvv + c * nvv + a * nvir + c;
                                            indices[5] = b * nvvv + c * nvv + c * nvir + a;
                                            indices[6] = c * nvvv + a * nvv + b * nvir + c;
                                            indices[7] = c * nvvv + a * nvv + c * nvir + b;
                                            indices[8] = c * nvvv + b * nvv + a * nvir + c;
                                            indices[9] = c * nvvv + b * nvv + c * nvir + a;
                                            indices[10] = c * nvvv + c * nvv + a * nvir + b;
                                            indices[11] = c * nvvv + c * nvv + b * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[9]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[11]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[5]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[7]];
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[10]];
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[2]];
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[6]];
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[3]];
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[8]];
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[0]];
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[4]];
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[1]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[1] + p[6] * T1_local[2];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[0] + p[6] * T1_local[1];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[2] + p[6] * T1_local[0];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[4] + p[6] * T1_local[5];
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[3] + p[6] * T1_local[4];
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[5] + p[6] * T1_local[3];
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[8] + p[6] * T1_local[11];
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[10] + p[6] * T1_local[9];
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[6] + p[6] * T1_local[10];
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[11] + p[6] * T1_local[7];
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[7] + p[6] * T1_local[8];
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[9] + p[6] * T1_local[6];

                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                                            B[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[2]) + beta * A[h + indices[1]];
                                            B[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[1]) + beta * A[h + indices[2]];
                                            B[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[3]) + beta * A[h + indices[3]];
                                            B[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];
                                            B[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];
                                            B[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];
                                            B[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];
                                            B[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[9]) + beta * A[h + indices[8]];
                                            B[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[8]) + beta * A[h + indices[9]];
                                            B[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]];
                                            B[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]];
                                        }
                                        else if (a > b && b == c && c > d)
                                        {
                                            double T1_local[12];
                                            double T2_local[12];

                                            int64_t indices[12];
                                            indices[0] = a * nvvv + b * nvv + b * nvir + d;
                                            indices[1] = a * nvvv + b * nvv + d * nvir + b;
                                            indices[2] = a * nvvv + d * nvv + b * nvir + b;
                                            indices[3] = b * nvvv + a * nvv + b * nvir + d;
                                            indices[4] = b * nvvv + a * nvv + d * nvir + b;
                                            indices[5] = b * nvvv + b * nvv + a * nvir + d;
                                            indices[6] = b * nvvv + b * nvv + d * nvir + a;
                                            indices[7] = b * nvvv + d * nvv + a * nvir + b;
                                            indices[8] = b * nvvv + d * nvv + b * nvir + a;
                                            indices[9] = d * nvvv + a * nvv + b * nvir + b;
                                            indices[10] = d * nvvv + b * nvv + a * nvir + b;
                                            indices[11] = d * nvvv + b * nvv + b * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[11]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[6]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[8]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[9]];
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[4]];
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[10]];
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[1]];
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[7]];
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[2]];
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[3]];
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[5]];
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[0]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[2];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[1];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[0];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[5] + p[6] * T1_local[8];
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[7] + p[6] * T1_local[6];
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[3] + p[6] * T1_local[7];
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[8] + p[6] * T1_local[4];
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[4] + p[6] * T1_local[5];
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[6] + p[6] * T1_local[3];
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[10] + p[6] * T1_local[11];
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[9] + p[6] * T1_local[10];
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[11] + p[6] * T1_local[9];

                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                                            B[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                                            B[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[2]) + beta * A[h + indices[2]];
                                            B[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[4]) + beta * A[h + indices[3]];
                                            B[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[3]) + beta * A[h + indices[4]];
                                            B[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[6]) + beta * A[h + indices[5]];
                                            B[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[5]) + beta * A[h + indices[6]];
                                            B[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[8]) + beta * A[h + indices[7]];
                                            B[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[7]) + beta * A[h + indices[8]];
                                            B[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[9]) + beta * A[h + indices[9]];
                                            B[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[11]) + beta * A[h + indices[10]];
                                            B[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[10]) + beta * A[h + indices[11]];
                                        }
                                        else if (a == b && b > c && c > d)
                                        {
                                            double T1_local[12];
                                            double T2_local[12];

                                            int64_t indices[12];
                                            indices[0] = a * nvvv + a * nvv + c * nvir + d;
                                            indices[1] = a * nvvv + a * nvv + d * nvir + c;
                                            indices[2] = a * nvvv + c * nvv + a * nvir + d;
                                            indices[3] = a * nvvv + c * nvv + d * nvir + a;
                                            indices[4] = a * nvvv + d * nvv + a * nvir + c;
                                            indices[5] = a * nvvv + d * nvv + c * nvir + a;
                                            indices[6] = c * nvvv + a * nvv + a * nvir + d;
                                            indices[7] = c * nvvv + a * nvv + d * nvir + a;
                                            indices[8] = c * nvvv + d * nvv + a * nvir + a;
                                            indices[9] = d * nvvv + a * nvv + a * nvir + c;
                                            indices[10] = d * nvvv + a * nvv + c * nvir + a;
                                            indices[11] = d * nvvv + c * nvv + a * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[6]] + p[3] * A[h + indices[10]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[9]] + p[3] * A[h + indices[7]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[6]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[11]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[7]] + p[2] * A[h + indices[11]] + p[3] * A[h + indices[3]];
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[9]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[8]];
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[10]] + p[2] * A[h + indices[8]] + p[3] * A[h + indices[5]];
                                            T1_local[6] = p[0] * A[h + indices[6]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[9]];
                                            T1_local[7] = p[0] * A[h + indices[7]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[10]] + p[3] * A[h + indices[1]];
                                            T1_local[8] = p[0] * A[h + indices[8]] + p[1] * A[h + indices[11]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[4]];
                                            T1_local[9] = p[0] * A[h + indices[9]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[6]];
                                            T1_local[10] = p[0] * A[h + indices[10]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[7]] + p[3] * A[h + indices[0]];
                                            T1_local[11] = p[0] * A[h + indices[11]] + p[1] * A[h + indices[8]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[2]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[2] + p[6] * T1_local[5];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[4] + p[6] * T1_local[3];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[0] + p[6] * T1_local[4];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[5] + p[6] * T1_local[1];
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[1] + p[6] * T1_local[2];
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[3] + p[6] * T1_local[0];
                                            T2_local[6] = p[4] * T1_local[6] + p[5] * T1_local[6] + p[6] * T1_local[8];
                                            T2_local[7] = p[4] * T1_local[7] + p[5] * T1_local[8] + p[6] * T1_local[7];
                                            T2_local[8] = p[4] * T1_local[8] + p[5] * T1_local[7] + p[6] * T1_local[6];
                                            T2_local[9] = p[4] * T1_local[9] + p[5] * T1_local[9] + p[6] * T1_local[11];
                                            T2_local[10] = p[4] * T1_local[10] + p[5] * T1_local[11] + p[6] * T1_local[10];
                                            T2_local[11] = p[4] * T1_local[11] + p[5] * T1_local[10] + p[6] * T1_local[9];

                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                                            B[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                                            B[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];
                                            B[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];
                                            B[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[5]) + beta * A[h + indices[4]];
                                            B[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[4]) + beta * A[h + indices[5]];
                                            B[h + indices[6]] = alpha * (p[7] * T2_local[6] + p[8] * T2_local[7]) + beta * A[h + indices[6]];
                                            B[h + indices[7]] = alpha * (p[7] * T2_local[7] + p[8] * T2_local[6]) + beta * A[h + indices[7]];
                                            B[h + indices[8]] = alpha * (p[7] * T2_local[8] + p[8] * T2_local[8]) + beta * A[h + indices[8]];
                                            B[h + indices[9]] = alpha * (p[7] * T2_local[9] + p[8] * T2_local[10]) + beta * A[h + indices[9]];
                                            B[h + indices[10]] = alpha * (p[7] * T2_local[10] + p[8] * T2_local[9]) + beta * A[h + indices[10]];
                                            B[h + indices[11]] = alpha * (p[7] * T2_local[11] + p[8] * T2_local[11]) + beta * A[h + indices[11]];
                                        }
                                        else if (a > b && b == c && c == d)
                                        {
                                            double T1_local[4];
                                            double T2_local[4];

                                            int64_t indices[4];
                                            indices[0] = a * nvvv + b * nvv + b * nvir + b;
                                            indices[1] = b * nvvv + a * nvv + b * nvir + b;
                                            indices[2] = b * nvvv + b * nvv + a * nvir + b;
                                            indices[3] = b * nvvv + b * nvv + b * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[3]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[1]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[2]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[0]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[0];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[3];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[2];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[1];

                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                                            B[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[1]) + beta * A[h + indices[1]];
                                            B[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[3]) + beta * A[h + indices[2]];
                                            B[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[2]) + beta * A[h + indices[3]];
                                        }
                                        else if (a == b && b == c && c > d)
                                        {
                                            double T1_local[4];
                                            double T2_local[4];

                                            int64_t indices[4];
                                            indices[0] = a * nvvv + a * nvv + a * nvir + d;
                                            indices[1] = a * nvvv + a * nvv + d * nvir + a;
                                            indices[2] = a * nvvv + d * nvv + a * nvir + a;
                                            indices[3] = d * nvvv + a * nvv + a * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[3]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[1]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[2]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[0]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[2];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[2] + p[6] * T1_local[1];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[1] + p[6] * T1_local[0];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[3];

                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[1]) + beta * A[h + indices[0]];
                                            B[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[0]) + beta * A[h + indices[1]];
                                            B[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[2]) + beta * A[h + indices[2]];
                                            B[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[3]) + beta * A[h + indices[3]];
                                        }
                                        else if (a == b && b > c && c == d)
                                        {
                                            double T1_local[6];
                                            double T2_local[6];

                                            int64_t indices[6];
                                            indices[0] = b * nvvv + b * nvv + c * nvir + c;
                                            indices[1] = b * nvvv + c * nvv + b * nvir + c;
                                            indices[2] = b * nvvv + c * nvv + c * nvir + b;
                                            indices[3] = c * nvvv + b * nvv + b * nvir + c;
                                            indices[4] = c * nvvv + b * nvv + c * nvir + b;
                                            indices[5] = c * nvvv + c * nvv + b * nvir + b;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[3]] + p[3] * A[h + indices[4]];
                                            T1_local[1] = p[0] * A[h + indices[1]] + p[1] * A[h + indices[3]] + p[2] * A[h + indices[1]] + p[3] * A[h + indices[5]];
                                            T1_local[2] = p[0] * A[h + indices[2]] + p[1] * A[h + indices[4]] + p[2] * A[h + indices[5]] + p[3] * A[h + indices[2]];
                                            T1_local[3] = p[0] * A[h + indices[3]] + p[1] * A[h + indices[1]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[3]];
                                            T1_local[4] = p[0] * A[h + indices[4]] + p[1] * A[h + indices[2]] + p[2] * A[h + indices[4]] + p[3] * A[h + indices[0]];
                                            T1_local[5] = p[0] * A[h + indices[5]] + p[1] * A[h + indices[5]] + p[2] * A[h + indices[2]] + p[3] * A[h + indices[1]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[1] + p[6] * T1_local[2];
                                            T2_local[1] = p[4] * T1_local[1] + p[5] * T1_local[0] + p[6] * T1_local[1];
                                            T2_local[2] = p[4] * T1_local[2] + p[5] * T1_local[2] + p[6] * T1_local[0];
                                            T2_local[3] = p[4] * T1_local[3] + p[5] * T1_local[3] + p[6] * T1_local[5];
                                            T2_local[4] = p[4] * T1_local[4] + p[5] * T1_local[5] + p[6] * T1_local[4];
                                            T2_local[5] = p[4] * T1_local[5] + p[5] * T1_local[4] + p[6] * T1_local[3];

                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                                            B[h + indices[1]] = alpha * (p[7] * T2_local[1] + p[8] * T2_local[2]) + beta * A[h + indices[1]];
                                            B[h + indices[2]] = alpha * (p[7] * T2_local[2] + p[8] * T2_local[1]) + beta * A[h + indices[2]];
                                            B[h + indices[3]] = alpha * (p[7] * T2_local[3] + p[8] * T2_local[4]) + beta * A[h + indices[3]];
                                            B[h + indices[4]] = alpha * (p[7] * T2_local[4] + p[8] * T2_local[3]) + beta * A[h + indices[4]];
                                            B[h + indices[5]] = alpha * (p[7] * T2_local[5] + p[8] * T2_local[5]) + beta * A[h + indices[5]];
                                        }
                                        else if (a == b && b == c && c == d)
                                        {
                                            double T1_local[1];
                                            double T2_local[1];

                                            int64_t indices[1];
                                            indices[0] = a * nvvv + a * nvv + a * nvir + a;

                                            T1_local[0] = p[0] * A[h + indices[0]] + p[1] * A[h + indices[0]] + p[2] * A[h + indices[0]] + p[3] * A[h + indices[0]];

                                            T2_local[0] = p[4] * T1_local[0] + p[5] * T1_local[0] + p[6] * T1_local[0];

                                            B[h + indices[0]] = alpha * (p[7] * T2_local[0] + p[8] * T2_local[0]) + beta * A[h + indices[0]];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Apply permutation-symmetry projection to T4 amplitudes in place.
// A = beta * A + alpha * P(A)
// where P(A) ijklabcd = ijklabcd + ijlkabdc + ...
//
// Input
//   A     : pointer to T4 tensor in full i,j,k,l,a,b,c,d layout
//   nocc  : number of occupied orbitals
//   nvir  : number of virtual orbitals
//   alpha : scaling factor for projected contribution
//   beta  : scaling factor for original tensor
void t4_perm_symmetrize_inplace_(double *A, int64_t nocc, int64_t nvir, double alpha, double beta)
{
    const int64_t bl = 8;
    int64_t nvv = nvir * nvir;
    int64_t nvvv = nvir * nvv;
    int64_t nvvvv = nvir * nvvv;
    int64_t novvvv = nocc * nvvvv;
    int64_t noovvvv = nocc * novvvv;
    int64_t nooovvvv = nocc * noovvvv;

    int64_t ntriplets = 0;
    for (int i = 0; i < nocc; i++)
        for (int j = 0; j <= i; j++)
            for (int k = 0; k <= j; k++)
                for (int l = 0; l <= k; l++)
                    ntriplets++;

    const int64_t map1[24][4] = {
        {0, 6, 14, 21},
        {1, 7, 20, 15},
        {2, 12, 8, 23},
        {3, 13, 22, 9},
        {4, 18, 10, 17},
        {5, 19, 16, 11},
        {6, 0, 12, 19},
        {7, 1, 18, 13},
        {8, 14, 2, 22},
        {9, 15, 23, 3},
        {10, 20, 4, 16},
        {11, 21, 17, 5},
        {12, 2, 6, 18},
        {13, 3, 19, 7},
        {14, 8, 0, 20},
        {15, 9, 21, 1},
        {16, 22, 5, 10},
        {17, 23, 11, 4},
        {18, 4, 7, 12},
        {19, 5, 13, 6},
        {20, 10, 1, 14},
        {21, 11, 15, 0},
        {22, 16, 3, 8},
        {23, 17, 9, 2},
    };
    const int64_t map2[24][3] = {
        {0, 2, 5},
        {1, 4, 3},
        {2, 0, 4},
        {3, 5, 1},
        {4, 1, 2},
        {5, 3, 0},
        {6, 8, 11},
        {7, 10, 9},
        {8, 6, 10},
        {9, 11, 7},
        {10, 7, 8},
        {11, 9, 6},
        {12, 14, 17},
        {13, 16, 15},
        {14, 12, 16},
        {15, 17, 13},
        {16, 13, 14},
        {17, 15, 12},
        {18, 20, 23},
        {19, 22, 21},
        {20, 18, 22},
        {21, 23, 19},
        {22, 19, 20},
        {23, 21, 18},
    };
    const int64_t map3[24][2] = {
        {0, 1},
        {1, 0},
        {2, 3},
        {3, 2},
        {4, 5},
        {5, 4},
        {6, 7},
        {7, 6},
        {8, 9},
        {9, 8},
        {10, 11},
        {11, 10},
        {12, 13},
        {13, 12},
        {14, 15},
        {15, 14},
        {16, 17},
        {17, 16},
        {18, 19},
        {19, 18},
        {20, 21},
        {21, 20},
        {22, 23},
        {23, 22},
    };

#pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < ntriplets; idx++)
    {
        int64_t i, j, k, l = 0;
        int64_t tmp = idx;

        for (i = 0; i < nocc; i++)
        {
            int64_t count_i = (int64_t)(i + 1) * (i + 2) * (i + 3) / 6;
            if (tmp < count_i)
                break;
            tmp -= count_i;
        }

        for (j = 0; j <= i; j++)
        {
            int64_t count_j = (int64_t)(j + 1) * (j + 2) / 2;
            if (tmp < count_j)
                break;
            tmp -= count_j;
        }

        for (k = 0; k <= j; k++)
        {
            int64_t count_k = k + 1;
            if (tmp < count_k)
            {
                l = (int)tmp;
                break;
            }
            tmp -= count_k;
        }

        int64_t occ_perms[24][4] = {
            {i, j, k, l},
            {i, j, l, k},
            {i, k, j, l},
            {i, k, l, j},
            {i, l, j, k},
            {i, l, k, j},
            {j, i, k, l},
            {j, i, l, k},
            {j, k, i, l},
            {j, k, l, i},
            {j, l, i, k},
            {j, l, k, i},
            {k, i, j, l},
            {k, i, l, j},
            {k, j, i, l},
            {k, j, l, i},
            {k, l, i, j},
            {k, l, j, i},
            {l, i, j, k},
            {l, i, k, j},
            {l, j, i, k},
            {l, j, k, i},
            {l, k, i, j},
            {l, k, j, i},
        };

        double T1_local[24][24];
        double T2_local[24][24];
        for (int64_t a0 = 0; a0 < nvir; a0 += bl)
        {
            for (int64_t b0 = 0; b0 <= a0; b0 += bl)
            {
                for (int64_t c0 = 0; c0 <= b0; c0 += bl)
                {
                    for (int64_t d0 = 0; d0 <= c0; d0 += bl)
                    {
                        for (int64_t a = a0; a < a0 + bl && a < nvir; a++)
                        {
                            for (int64_t b = b0; b < b0 + bl && b <= a; b++)
                            {
                                for (int64_t c = c0; c < c0 + bl && c <= b; c++)
                                {
                                    for (int64_t d = d0; d < d0 + bl && d <= c; d++)
                                    {
                                        int64_t vir_perms[24][4] = {
                                            {a, b, c, d},
                                            {a, b, d, c},
                                            {a, c, b, d},
                                            {a, c, d, b},
                                            {a, d, b, c},
                                            {a, d, c, b},
                                            {b, a, c, d},
                                            {b, a, d, c},
                                            {b, c, a, d},
                                            {b, c, d, a},
                                            {b, d, a, c},
                                            {b, d, c, a},
                                            {c, a, b, d},
                                            {c, a, d, b},
                                            {c, b, a, d},
                                            {c, b, d, a},
                                            {c, d, a, b},
                                            {c, d, b, a},
                                            {d, a, b, c},
                                            {d, a, c, b},
                                            {d, b, a, c},
                                            {d, b, c, a},
                                            {d, c, a, b},
                                            {d, c, b, a},
                                        };

                                        int64_t indices[24][24];
                                        for (int perm_occ = 0; perm_occ < 24; perm_occ++)
                                        {
                                            for (int perm_vir = 0; perm_vir < 24; perm_vir++)
                                            {
                                                indices[perm_occ][perm_vir] =
                                                    occ_perms[perm_occ][0] * nooovvvv +
                                                    occ_perms[perm_occ][1] * noovvvv +
                                                    occ_perms[perm_occ][2] * novvvv +
                                                    occ_perms[perm_occ][3] * nvvvv +
                                                    vir_perms[perm_vir][0] * nvvv +
                                                    vir_perms[perm_vir][1] * nvv +
                                                    vir_perms[perm_vir][2] * nvir +
                                                    vir_perms[perm_vir][3];
                                            }
                                        }

                                        for (int perm_occ = 0; perm_occ < 24; perm_occ++)
                                        {
                                            for (int perm_vir = 0; perm_vir < 24; perm_vir++)
                                            {
                                                T1_local[perm_occ][perm_vir] = A[indices[map1[perm_occ][0]][map1[perm_vir][0]]] + A[indices[map1[perm_occ][1]][map1[perm_vir][1]]] + A[indices[map1[perm_occ][2]][map1[perm_vir][2]]] + A[indices[map1[perm_occ][3]][map1[perm_vir][3]]];
                                            }
                                        }

                                        for (int perm_occ = 0; perm_occ < 24; perm_occ++)
                                        {
                                            for (int perm_vir = 0; perm_vir < 24; perm_vir++)
                                            {
                                                T2_local[perm_occ][perm_vir] = T1_local[map2[perm_occ][0]][map2[perm_vir][0]] + T1_local[map2[perm_occ][1]][map2[perm_vir][1]] + T1_local[map2[perm_occ][2]][map2[perm_vir][2]];
                                            }
                                        }

                                        for (int perm_occ = 0; perm_occ < 24; perm_occ++)
                                        {
                                            for (int perm_vir = 0; perm_vir < 24; perm_vir++)
                                            {
                                                A[indices[map3[perm_occ][0]][map3[perm_vir][0]]] = beta * A[indices[map3[perm_occ][0]][map3[perm_vir][0]]] + alpha * (T2_local[map3[perm_occ][0]][map3[perm_vir][0]] + T2_local[map3[perm_occ][1]][map3[perm_vir][1]]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void eijkl_division_(double *r4, const double *eia, const int64_t nocc, const int64_t nvir)
{
#pragma omp parallel for collapse(4) schedule(static)
    for (int64_t i = 0; i < nocc; i++)
    {
        for (int64_t j = 0; j < nocc; j++)
        {
            for (int64_t k = 0; k < nocc; k++)
            {
                for (int64_t l = 0; l < nocc; l++)
                {
                    size_t base_size = ((((size_t)i * nocc + j) * nocc + k) * nocc + l) * nvir * nvir * nvir * nvir;
                    for (int64_t a = 0; a < nvir; a++)
                    {
                        for (int64_t b = 0; b < nvir; b++)
                        {
                            for (int64_t c = 0; c < nvir; c++)
                            {
                                for (int64_t d = 0; d < nvir; d++)
                                {
                                    size_t r4_idx = (size_t)base_size + ((a * nvir + b) * nvir + c) * nvir + d;

                                    size_t eia_ia_idx = (size_t)i * nvir + a;
                                    size_t eia_jb_idx = (size_t)j * nvir + b;
                                    size_t eia_kc_idx = (size_t)k * nvir + c;
                                    size_t eia_ld_idx = (size_t)l * nvir + d;

                                    double eijklabcd = eia[eia_ia_idx] + eia[eia_jb_idx] + eia[eia_kc_idx] + eia[eia_ld_idx];

                                    if (fabs(eijklabcd) > 1e-15)
                                    {
                                        r4[r4_idx] /= eijklabcd;
                                    }
                                    else
                                    {
                                        r4[r4_idx] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void t4_add_(double *t4, const double *r4, const int64_t nocc4, const int64_t nvir)
{
    const int64_t total_size = nocc4 * nvir * nvir * nvir * nvir;

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < total_size; i++)
    {
        t4[i] += r4[i];
    }
}

const int64_t tp_t4[24][4] = {
    {0, 1, 2, 3},
    {0, 1, 3, 2},
    {0, 2, 1, 3},
    {0, 2, 3, 1},
    {0, 3, 1, 2},
    {0, 3, 2, 1},
    {1, 0, 2, 3},
    {1, 0, 3, 2},
    {1, 2, 0, 3},
    {1, 2, 3, 0},
    {1, 3, 0, 2},
    {1, 3, 2, 0},
    {2, 0, 1, 3},
    {2, 0, 3, 1},
    {2, 1, 0, 3},
    {2, 1, 3, 0},
    {2, 3, 0, 1},
    {2, 3, 1, 0},
    {3, 0, 1, 2},
    {3, 0, 2, 1},
    {3, 1, 0, 2},
    {3, 1, 2, 0},
    {3, 2, 0, 1},
    {3, 2, 1, 0},
};

// Unpack triangular-stored T4 amplitudes into a full T4 block.
//
// This kernel reconstructs the full permutation-expanded T4 tensor block from the compressed triangular
// representation without forming the full tensor in memory.
//
// Input:
//   t4_tri                             : triangular-stored T4 amplitudes
//   t4_blk                             : output buffer [blk_i * blk_j * blk_k * nvir**3]
//   map                                : mapping index table for (i, j, k) -> tri index
//   mask                               : mask indicating which (i, j, k) indices are stored (triangular domain)
//   [i0:i1), [j0:j1), [k0:k1), [l0:l1) : occupied index block ranges
//   nocc, nvir                         : number of occupied / virtual orbitals
//   blk_i, blk_j, blk_k, blk_l         : block sizes for the destination tensor
void unpack_t4_tri2block_(const double *restrict t4_tri,
                          double *restrict t4_blk,
                          const int64_t *restrict map,
                          const bool *restrict mask,
                          int64_t i0, int64_t i1,
                          int64_t j0, int64_t j1,
                          int64_t k0, int64_t k1,
                          int64_t l0, int64_t l1,
                          int64_t nocc, int64_t nvir,
                          int64_t blk_i, int64_t blk_j, int64_t blk_k, int64_t blk_l)
{
#define MAP(sym, w, x, y, z) map[((((sym) * nocc + (w)) * nocc + (x)) * nocc + (y)) * nocc + (z)]
#define MASK(sym, w, x, y, z) mask[((((sym) * nocc + (w)) * nocc + (x)) * nocc + (y)) * nocc + (z)]

#pragma omp parallel for collapse(5) schedule(dynamic)
    for (int64_t sym = 0; sym < 24; ++sym)
    {
        for (int64_t i = i0; i < i1; ++i)
        {
            for (int64_t j = j0; j < j1; ++j)
            {
                for (int64_t k = k0; k < k1; ++k)
                {
                    for (int64_t l = l0; l < l1; ++l)
                    {
                        if (!MASK(sym, i, j, k, l))
                            continue;

                        const int64_t *perm = tp_t4[sym];

                        int64_t loc_i = i - i0;
                        int64_t loc_j = j - j0;
                        int64_t loc_k = k - k0;
                        int64_t loc_l = l - l0;

                        int64_t src_base = MAP(sym, i, j, k, l) * nvir * nvir * nvir * nvir;
                        int64_t dest_base = (((loc_i * blk_j + loc_j) * blk_k + loc_k) * blk_l + loc_l) * nvir * nvir * nvir * nvir;

                        for (int64_t a = 0; a < nvir; ++a)
                        {
                            for (int64_t b = 0; b < nvir; ++b)
                            {
                                for (int64_t c = 0; c < nvir; ++c)
                                {
                                    for (int64_t d = 0; d < nvir; ++d)
                                    {
                                        int64_t abcd[4] = {a, b, c, d};
                                        int64_t aa = abcd[perm[0]];
                                        int64_t bb = abcd[perm[1]];
                                        int64_t cc = abcd[perm[2]];
                                        int64_t dd = abcd[perm[3]];

                                        int64_t src_idx = src_base + ((a * nvir + b) * nvir + c) * nvir + d;
                                        int64_t dest_idx = dest_base + ((aa * nvir + bb) * nvir + cc) * nvir + dd;

                                        t4_blk[dest_idx] = t4_tri[src_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#undef MAP
#undef MASK
}

// Accumulate a full T4 block into the triangular-stored T4 amplitudes.
//
// This routine performs the reverse of the unpacking:
// given a (i0:i1, j0:j1, k0:k1, l0:l1) block of the full T4 tensor in t4_blk, it updates the
// corresponding triangular-stored T4 buffer t4_tri using
//
//     t4_tri = beta * t4_tri + alpha * t4_blk
//
// Only unique (i <= j <= k <= l) indices are processed. The mapping table `map`
// provides the location of the triangular representative for each index triplet.
//
// Inputs:
//     t4_tri                             : triangular-stored T4 amplitudes
//     t4_blk                             : full T4 block [blk_i * blk_j * blk_k * nvir**3]
//     map                                : maps (sym, i, j, k, l) -> triangular index
//     [i0:i1), [j0:j1), [k0:k1), [l0:l1) : occupied index block ranges
//     nocc, nvir                         : number of occupied / virtual orbitals
//     blk_i, blk_j, blk_k, blk_l         : block dimensions for t4_blk
//     alpha, beta                        : scaling factors for update
void accumulate_t4_block2tri_(double *restrict t4_tri,
                              const double *restrict t4_blk,
                              const int64_t *restrict map,
                              int64_t i0, int64_t i1,
                              int64_t j0, int64_t j1,
                              int64_t k0, int64_t k1,
                              int64_t l0, int64_t l1,
                              int64_t nocc, int64_t nvir,
                              int64_t blk_i, int64_t blk_j, int64_t blk_k, int64_t blk_l,
                              double alpha, double beta)
{
#define MAP(sym, w, x, y, z) map[((((sym) * nocc + (w)) * nocc + (x)) * nocc + (y)) * nocc + (z)]

    if (j1 < i0 || k1 < j0 || l1 < k0)
        return;

#pragma omp parallel for collapse(4) schedule(dynamic)
    for (int64_t l = l0; l < l1; ++l)
    {
        for (int64_t k = k0; k < k1; ++k)
        {
            for (int64_t j = j0; j < j1; ++j)
            {
                for (int64_t i = i0; i < i1; ++i)
                {
                    if (k > l || j > k || i > j)
                        continue;

                    int64_t p = MAP(0, i, j, k, l);
                    int64_t tri_base = p * nvir * nvir * nvir * nvir;

                    int64_t loc_i = i - i0;
                    int64_t loc_j = j - j0;
                    int64_t loc_k = k - k0;
                    int64_t loc_l = l - l0;
                    int64_t blk_base = (((loc_i * blk_j + loc_j) * blk_k + loc_k) * blk_l + loc_l) * nvir * nvir * nvir * nvir;

                    for (int64_t a = 0; a < nvir; ++a)
                    {
                        for (int64_t b = 0; b < nvir; ++b)
                        {
                            for (int64_t c = 0; c < nvir; ++c)
                            {
                                for (int64_t d = 0; d < nvir; ++d)
                                {
                                    int64_t idx = (((a * nvir + b) * nvir + c) * nvir + d);
                                    t4_tri[tri_base + idx] = beta * t4_tri[tri_base + idx] + alpha * t4_blk[blk_base + idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#undef MAP
}
