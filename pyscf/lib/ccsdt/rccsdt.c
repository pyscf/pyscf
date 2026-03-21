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

// Apply spin summation projection to T3 amplitudes in place.
// A: pointer to T3 tensor (size nocc3 * nvir**3)
// pattern: "P3_full" :  ijkabc + ijkacb + ijkbac + ijkbca + ijkcab + ijkcba
//                       P(A) = (1 + P_b^c + P_a^b + P_b^c P_a^b + P_b^c P_a^c + P_a^c) A
//                            = (1 + P_b^c) (1 + P_a^b + P_a^c) A
//          "P3_422" :   4 ijkabc - 2 ijkacb - 2 ijkbac + ijkbca + ijkcab - 2 ijkcba
//                       P(A) = (4 - 2 P_b^c - 2 P_a^b + P_b^c P_a^b + P_b^c P_a^c - 2 P_a^c) A
//                            = (2 - P_b^c) (2 - P_a^b - P_a^c) A
//          "P3_201" :   2 ijkabc - ijkbac - ijkcba
//                       P(A) = (2 - 0 P_b^c - 1 P_a^b + 0 P_b^c P_a^b + 0 P_b^c P_a^c - 1 P_a^c) A
//                            = (1 - 0 P_b^c) (2 - P_a^b - P_a^c) A
// alpha, beta: A = beta * A + alpha * P(A)
void t3_spin_summation_inplace_(double *A, int64_t nocc3, int64_t nvir, char *pattern, double alpha, double beta)
{
    int64_t ijk;
    const int64_t bl = 16;
    int64_t nvv = nvir * nvir;
    double p0 = 0.0, p1 = 0.0, p2 = 0.0, p3 = 0.0, p4 = 0.0;

    if (strcmp(pattern, "P3_full") == 0)
    {
        p0 = 1.0;
        p1 = 1.0;
        p2 = 1.0;
        p3 = 1.0;
        p4 = 1.0;
    }
    else if (strcmp(pattern, "P3_422") == 0)
    {
        p0 = 2.0;
        p1 = -1.0;
        p2 = -1.0;
        p3 = 2.0;
        p4 = -1.0;
    }
    else if (strcmp(pattern, "P3_201") == 0)
    {
        p0 = 2.0;
        p1 = -1.0;
        p2 = -1.0;
        p3 = 1.0;
        p4 = 0.0;
    }
    else
    {
        fprintf(stderr, "Error: unrecognized pattern \"%s\"\n", pattern);
        return;
    }

#pragma omp parallel for schedule(static)
    for (ijk = 0; ijk < nocc3; ijk++)
    {
        int64_t h = ijk * nvir * nvir * nvir;
        for (int64_t a0 = 0; a0 < nvir; a0 += bl)
        {
            for (int64_t b0 = 0; b0 <= a0; b0 += bl)
            {
                for (int64_t c0 = 0; c0 <= b0; c0 += bl)
                {
                    for (int64_t a = a0; a < a0 + bl && a < nvir; a++)
                    {
                        for (int64_t b = b0; b < b0 + bl && b <= a; b++)
                        {
                            for (int64_t c = c0; c < c0 + bl && c <= b; c++)
                            {

                                if (a > b && b > c)
                                {
                                    double T_local[6];

                                    int64_t idx1 = a * nvv + b * nvir + c;
                                    int64_t idx2 = c * nvv + b * nvir + a;
                                    int64_t idx3 = a * nvv + c * nvir + b;
                                    int64_t idx4 = b * nvv + a * nvir + c;
                                    int64_t idx5 = b * nvv + c * nvir + a;
                                    int64_t idx6 = c * nvv + a * nvir + b;

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx2] + p2 * A[h + idx4];
                                    T_local[1] = p0 * A[h + idx2] + p1 * A[h + idx1] + p2 * A[h + idx5];
                                    T_local[2] = p0 * A[h + idx3] + p1 * A[h + idx5] + p2 * A[h + idx6];
                                    T_local[3] = p0 * A[h + idx4] + p1 * A[h + idx6] + p2 * A[h + idx1];
                                    T_local[4] = p0 * A[h + idx5] + p1 * A[h + idx3] + p2 * A[h + idx2];
                                    T_local[5] = p0 * A[h + idx6] + p1 * A[h + idx4] + p2 * A[h + idx3];

                                    A[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[2]);
                                    A[h + idx2] = beta * A[h + idx2] + alpha * (p3 * T_local[1] + p4 * T_local[5]);
                                    A[h + idx3] = beta * A[h + idx3] + alpha * (p3 * T_local[2] + p4 * T_local[0]);
                                    A[h + idx4] = beta * A[h + idx4] + alpha * (p3 * T_local[3] + p4 * T_local[4]);
                                    A[h + idx5] = beta * A[h + idx5] + alpha * (p3 * T_local[4] + p4 * T_local[3]);
                                    A[h + idx6] = beta * A[h + idx6] + alpha * (p3 * T_local[5] + p4 * T_local[1]);
                                }

                                else if (a > b && b == c)
                                {
                                    double T_local[3];

                                    int64_t idx1 = a * nvv + b * nvir + b;
                                    int64_t idx2 = b * nvv + b * nvir + a;
                                    int64_t idx4 = b * nvv + a * nvir + b;

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx2] + p2 * A[h + idx4];
                                    T_local[1] = p0 * A[h + idx2] + p1 * A[h + idx1] + p2 * A[h + idx2];
                                    T_local[2] = p0 * A[h + idx4] + p1 * A[h + idx4] + p2 * A[h + idx1];

                                    A[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[0]);
                                    A[h + idx2] = beta * A[h + idx2] + alpha * (p3 * T_local[1] + p4 * T_local[2]);
                                    A[h + idx4] = beta * A[h + idx4] + alpha * (p3 * T_local[2] + p4 * T_local[1]);
                                }

                                else if (a == b && b > c)
                                {

                                    double T_local[3];

                                    int64_t idx1 = a * nvv + a * nvir + c;
                                    int64_t idx2 = c * nvv + a * nvir + a;
                                    int64_t idx3 = a * nvv + c * nvir + a;

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx2] + p2 * A[h + idx1];
                                    T_local[1] = p0 * A[h + idx2] + p1 * A[h + idx1] + p2 * A[h + idx3];
                                    T_local[2] = p0 * A[h + idx3] + p1 * A[h + idx3] + p2 * A[h + idx2];

                                    A[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[2]);
                                    A[h + idx2] = beta * A[h + idx2] + alpha * (p3 * T_local[1] + p4 * T_local[1]);
                                    A[h + idx3] = beta * A[h + idx3] + alpha * (p3 * T_local[2] + p4 * T_local[0]);
                                }
                                else if (a == b && b == c)
                                {
                                    double T_local[1];

                                    int64_t idx1 = a * nvv + a * nvir + a;

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx1] + p2 * A[h + idx1];

                                    A[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[0]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Apply spin summation projection to T3 amplitudes.
// A: pointer to T3 tensor (size nocc3 * nvir**3)
// B: pointer to transformed T3 tensor
// alpha, beta: B = beta * A + alpha * P(A)
void t3_spin_summation(const double *A, double *B, int64_t nocc3, int64_t nvir,
                       char *pattern, double alpha, double beta)
{
    int64_t ijk;
    const int64_t bl = 16;
    int64_t nvv = nvir * nvir;
    double p0 = 0.0, p1 = 0.0, p2 = 0.0, p3 = 0.0, p4 = 0.0;

    if (strcmp(pattern, "P3_full") == 0)
    {
        p0 = 1.0;
        p1 = 1.0;
        p2 = 1.0;
        p3 = 1.0;
        p4 = 1.0;
    }
    else if (strcmp(pattern, "P3_422") == 0)
    {
        p0 = 2.0;
        p1 = -1.0;
        p2 = -1.0;
        p3 = 2.0;
        p4 = -1.0;
    }
    else if (strcmp(pattern, "P3_201") == 0)
    {
        p0 = 2.0;
        p1 = -1.0;
        p2 = -1.0;
        p3 = 1.0;
        p4 = 0.0;
    }
    else
    {
        fprintf(stderr, "Error: unrecognized pattern \"%s\"\n", pattern);
        return;
    }

#pragma omp parallel for schedule(static)
    for (ijk = 0; ijk < nocc3; ijk++)
    {
        int64_t h = ijk * nvir * nvir * nvir;
        for (int64_t a0 = 0; a0 < nvir; a0 += bl)
        {
            for (int64_t b0 = 0; b0 <= a0; b0 += bl)
            {
                for (int64_t c0 = 0; c0 <= b0; c0 += bl)
                {
                    for (int64_t a = a0; a < a0 + bl && a < nvir; a++)
                    {
                        for (int64_t b = b0; b < b0 + bl && b <= a; b++)
                        {
                            for (int64_t c = c0; c < c0 + bl && c <= b; c++)
                            {

                                if (a > b && b > c)
                                {
                                    double T_local[6];

                                    int64_t idx1 = a * nvv + b * nvir + c;
                                    int64_t idx2 = c * nvv + b * nvir + a;
                                    int64_t idx3 = a * nvv + c * nvir + b;
                                    int64_t idx4 = b * nvv + a * nvir + c;
                                    int64_t idx5 = b * nvv + c * nvir + a;
                                    int64_t idx6 = c * nvv + a * nvir + b;

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx2] + p2 * A[h + idx4];
                                    T_local[1] = p0 * A[h + idx2] + p1 * A[h + idx1] + p2 * A[h + idx5];
                                    T_local[2] = p0 * A[h + idx3] + p1 * A[h + idx5] + p2 * A[h + idx6];
                                    T_local[3] = p0 * A[h + idx4] + p1 * A[h + idx6] + p2 * A[h + idx1];
                                    T_local[4] = p0 * A[h + idx5] + p1 * A[h + idx3] + p2 * A[h + idx2];
                                    T_local[5] = p0 * A[h + idx6] + p1 * A[h + idx4] + p2 * A[h + idx3];

                                    B[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[2]);
                                    B[h + idx2] = beta * A[h + idx2] + alpha * (p3 * T_local[1] + p4 * T_local[5]);
                                    B[h + idx3] = beta * A[h + idx3] + alpha * (p3 * T_local[2] + p4 * T_local[0]);
                                    B[h + idx4] = beta * A[h + idx4] + alpha * (p3 * T_local[3] + p4 * T_local[4]);
                                    B[h + idx5] = beta * A[h + idx5] + alpha * (p3 * T_local[4] + p4 * T_local[3]);
                                    B[h + idx6] = beta * A[h + idx6] + alpha * (p3 * T_local[5] + p4 * T_local[1]);
                                }

                                else if (a > b && b == c)
                                {
                                    double T_local[3];

                                    int64_t idx1 = a * nvv + b * nvir + b;
                                    int64_t idx2 = b * nvv + b * nvir + a;
                                    int64_t idx4 = b * nvv + a * nvir + b;

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx2] + p2 * A[h + idx4];
                                    T_local[1] = p0 * A[h + idx2] + p1 * A[h + idx1] + p2 * A[h + idx2];
                                    T_local[2] = p0 * A[h + idx4] + p1 * A[h + idx4] + p2 * A[h + idx1];

                                    B[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[0]);
                                    B[h + idx2] = beta * A[h + idx2] + alpha * (p3 * T_local[1] + p4 * T_local[2]);
                                    B[h + idx4] = beta * A[h + idx4] + alpha * (p3 * T_local[2] + p4 * T_local[1]);
                                }

                                else if (a == b && b > c)
                                {

                                    double T_local[3];

                                    int64_t idx1 = a * nvv + a * nvir + c;
                                    int64_t idx2 = c * nvv + a * nvir + a;
                                    int64_t idx3 = a * nvv + c * nvir + a;

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx2] + p2 * A[h + idx1];
                                    T_local[1] = p0 * A[h + idx2] + p1 * A[h + idx1] + p2 * A[h + idx3];
                                    T_local[2] = p0 * A[h + idx3] + p1 * A[h + idx3] + p2 * A[h + idx2];

                                    B[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[2]);
                                    B[h + idx2] = beta * A[h + idx2] + alpha * (p3 * T_local[1] + p4 * T_local[1]);
                                    B[h + idx3] = beta * A[h + idx3] + alpha * (p3 * T_local[2] + p4 * T_local[0]);
                                }
                                else if (a == b && b == c)
                                {
                                    double T_local[1];

                                    int64_t idx1 = a * nvv + a * nvir + a;

                                    T_local[0] = p0 * A[h + idx1] + p1 * A[h + idx1] + p2 * A[h + idx1];

                                    B[h + idx1] = beta * A[h + idx1] + alpha * (p3 * T_local[0] + p4 * T_local[0]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Apply permutation-symmetry projection to T3 amplitudes in place.
// A = beta * A + alpha * P(A)
// where P(A) ijkabc = ijkabc + ikjacb + jikbac + jkibca + kijcab + kjicba
//
// Input
//   A     : pointer to T3 tensor in full i,j,k,a,b,c layout
//   nocc  : number of occupied orbitals
//   nvir  : number of virtual orbitals
//   alpha : scaling factor for projected contribution
//   beta  : scaling factor for original tensor
void t3_perm_symmetrize_inplace_(double *A, int64_t nocc, int64_t nvir, double alpha, double beta)
{
    const int64_t bl = 16;
    int64_t nvv = nvir * nvir;
    int64_t nvvv = nvir * nvv;
    int64_t novvv = nocc * nvvv;
    int64_t noovvv = nocc * novvv;

    int64_t ntriplets = 0;
    for (int i = 0; i < nocc; i++)
        for (int j = 0; j <= i; j++)
            for (int k = 0; k <= j; k++)
                ntriplets++;

#pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < ntriplets; idx++)
    {
        int i, j, k = 0;
        int64_t tmp = idx;

        for (i = 0; i < nocc; i++)
        {
            int64_t count_ij = (i + 1) * (i + 2) / 2;
            if (tmp < count_ij)
                break;
            tmp -= count_ij;
        }

        for (j = 0; j <= i; j++)
        {
            if (tmp < (j + 1))
            {
                k = tmp;
                break;
            }
            tmp -= (j + 1);
        }

        double T_local[36];
        for (int64_t a0 = 0; a0 < nvir; a0 += bl)
        {
            for (int64_t b0 = 0; b0 <= a0; b0 += bl)
            {
                for (int64_t c0 = 0; c0 <= b0; c0 += bl)
                {
                    for (int64_t a = a0; a < a0 + bl && a < nvir; a++)
                    {
                        for (int64_t b = b0; b < b0 + bl && b <= a; b++)
                        {
                            for (int64_t c = c0; c < c0 + bl && c <= b; c++)
                            {

                                int64_t idx11 = i * noovvv + j * novvv + k * nvvv + a * nvv + b * nvir + c;
                                int64_t idx12 = i * noovvv + j * novvv + k * nvvv + c * nvv + b * nvir + a;
                                int64_t idx13 = i * noovvv + j * novvv + k * nvvv + a * nvv + c * nvir + b;
                                int64_t idx14 = i * noovvv + j * novvv + k * nvvv + b * nvv + a * nvir + c;
                                int64_t idx15 = i * noovvv + j * novvv + k * nvvv + b * nvv + c * nvir + a;
                                int64_t idx16 = i * noovvv + j * novvv + k * nvvv + c * nvv + a * nvir + b;
                                int64_t idx21 = k * noovvv + j * novvv + i * nvvv + a * nvv + b * nvir + c;
                                int64_t idx22 = k * noovvv + j * novvv + i * nvvv + c * nvv + b * nvir + a;
                                int64_t idx23 = k * noovvv + j * novvv + i * nvvv + a * nvv + c * nvir + b;
                                int64_t idx24 = k * noovvv + j * novvv + i * nvvv + b * nvv + a * nvir + c;
                                int64_t idx25 = k * noovvv + j * novvv + i * nvvv + b * nvv + c * nvir + a;
                                int64_t idx26 = k * noovvv + j * novvv + i * nvvv + c * nvv + a * nvir + b;
                                int64_t idx31 = i * noovvv + k * novvv + j * nvvv + a * nvv + b * nvir + c;
                                int64_t idx32 = i * noovvv + k * novvv + j * nvvv + c * nvv + b * nvir + a;
                                int64_t idx33 = i * noovvv + k * novvv + j * nvvv + a * nvv + c * nvir + b;
                                int64_t idx34 = i * noovvv + k * novvv + j * nvvv + b * nvv + a * nvir + c;
                                int64_t idx35 = i * noovvv + k * novvv + j * nvvv + b * nvv + c * nvir + a;
                                int64_t idx36 = i * noovvv + k * novvv + j * nvvv + c * nvv + a * nvir + b;
                                int64_t idx41 = j * noovvv + i * novvv + k * nvvv + a * nvv + b * nvir + c;
                                int64_t idx42 = j * noovvv + i * novvv + k * nvvv + c * nvv + b * nvir + a;
                                int64_t idx43 = j * noovvv + i * novvv + k * nvvv + a * nvv + c * nvir + b;
                                int64_t idx44 = j * noovvv + i * novvv + k * nvvv + b * nvv + a * nvir + c;
                                int64_t idx45 = j * noovvv + i * novvv + k * nvvv + b * nvv + c * nvir + a;
                                int64_t idx46 = j * noovvv + i * novvv + k * nvvv + c * nvv + a * nvir + b;
                                int64_t idx51 = j * noovvv + k * novvv + i * nvvv + a * nvv + b * nvir + c;
                                int64_t idx52 = j * noovvv + k * novvv + i * nvvv + c * nvv + b * nvir + a;
                                int64_t idx53 = j * noovvv + k * novvv + i * nvvv + a * nvv + c * nvir + b;
                                int64_t idx54 = j * noovvv + k * novvv + i * nvvv + b * nvv + a * nvir + c;
                                int64_t idx55 = j * noovvv + k * novvv + i * nvvv + b * nvv + c * nvir + a;
                                int64_t idx56 = j * noovvv + k * novvv + i * nvvv + c * nvv + a * nvir + b;
                                int64_t idx61 = k * noovvv + i * novvv + j * nvvv + a * nvv + b * nvir + c;
                                int64_t idx62 = k * noovvv + i * novvv + j * nvvv + c * nvv + b * nvir + a;
                                int64_t idx63 = k * noovvv + i * novvv + j * nvvv + a * nvv + c * nvir + b;
                                int64_t idx64 = k * noovvv + i * novvv + j * nvvv + b * nvv + a * nvir + c;
                                int64_t idx65 = k * noovvv + i * novvv + j * nvvv + b * nvv + c * nvir + a;
                                int64_t idx66 = k * noovvv + i * novvv + j * nvvv + c * nvv + a * nvir + b;

                                T_local[0 * 6 + 0] = A[idx11] + A[idx22] + A[idx33];
                                T_local[0 * 6 + 1] = A[idx12] + A[idx21] + A[idx36];
                                T_local[0 * 6 + 2] = A[idx13] + A[idx25] + A[idx31];
                                T_local[0 * 6 + 3] = A[idx14] + A[idx26] + A[idx35];
                                T_local[0 * 6 + 4] = A[idx15] + A[idx23] + A[idx34];
                                T_local[0 * 6 + 5] = A[idx16] + A[idx24] + A[idx32];
                                T_local[1 * 6 + 0] = A[idx21] + A[idx12] + A[idx63];
                                T_local[1 * 6 + 1] = A[idx22] + A[idx11] + A[idx66];
                                T_local[1 * 6 + 2] = A[idx23] + A[idx15] + A[idx61];
                                T_local[1 * 6 + 3] = A[idx24] + A[idx16] + A[idx65];
                                T_local[1 * 6 + 4] = A[idx25] + A[idx13] + A[idx64];
                                T_local[1 * 6 + 5] = A[idx26] + A[idx14] + A[idx62];
                                T_local[2 * 6 + 0] = A[idx31] + A[idx52] + A[idx13];
                                T_local[2 * 6 + 1] = A[idx32] + A[idx51] + A[idx16];
                                T_local[2 * 6 + 2] = A[idx33] + A[idx55] + A[idx11];
                                T_local[2 * 6 + 3] = A[idx34] + A[idx56] + A[idx15];
                                T_local[2 * 6 + 4] = A[idx35] + A[idx53] + A[idx14];
                                T_local[2 * 6 + 5] = A[idx36] + A[idx54] + A[idx12];
                                T_local[3 * 6 + 0] = A[idx41] + A[idx62] + A[idx53];
                                T_local[3 * 6 + 1] = A[idx42] + A[idx61] + A[idx56];
                                T_local[3 * 6 + 2] = A[idx43] + A[idx65] + A[idx51];
                                T_local[3 * 6 + 3] = A[idx44] + A[idx66] + A[idx55];
                                T_local[3 * 6 + 4] = A[idx45] + A[idx63] + A[idx54];
                                T_local[3 * 6 + 5] = A[idx46] + A[idx64] + A[idx52];
                                T_local[4 * 6 + 0] = A[idx51] + A[idx32] + A[idx43];
                                T_local[4 * 6 + 1] = A[idx52] + A[idx31] + A[idx46];
                                T_local[4 * 6 + 2] = A[idx53] + A[idx35] + A[idx41];
                                T_local[4 * 6 + 3] = A[idx54] + A[idx36] + A[idx45];
                                T_local[4 * 6 + 4] = A[idx55] + A[idx33] + A[idx44];
                                T_local[4 * 6 + 5] = A[idx56] + A[idx34] + A[idx42];
                                T_local[5 * 6 + 0] = A[idx61] + A[idx42] + A[idx23];
                                T_local[5 * 6 + 1] = A[idx62] + A[idx41] + A[idx26];
                                T_local[5 * 6 + 2] = A[idx63] + A[idx45] + A[idx21];
                                T_local[5 * 6 + 3] = A[idx64] + A[idx46] + A[idx25];
                                T_local[5 * 6 + 4] = A[idx65] + A[idx43] + A[idx24];
                                T_local[5 * 6 + 5] = A[idx66] + A[idx44] + A[idx22];

                                A[idx11] = beta * A[idx11] + alpha * (T_local[0 * 6 + 0] + T_local[3 * 6 + 3]);
                                A[idx12] = beta * A[idx12] + alpha * (T_local[0 * 6 + 1] + T_local[3 * 6 + 4]);
                                A[idx13] = beta * A[idx13] + alpha * (T_local[0 * 6 + 2] + T_local[3 * 6 + 5]);
                                A[idx14] = beta * A[idx14] + alpha * (T_local[0 * 6 + 3] + T_local[3 * 6 + 0]);
                                A[idx15] = beta * A[idx15] + alpha * (T_local[0 * 6 + 4] + T_local[3 * 6 + 1]);
                                A[idx16] = beta * A[idx16] + alpha * (T_local[0 * 6 + 5] + T_local[3 * 6 + 2]);
                                A[idx21] = beta * A[idx21] + alpha * (T_local[1 * 6 + 0] + T_local[4 * 6 + 3]);
                                A[idx22] = beta * A[idx22] + alpha * (T_local[1 * 6 + 1] + T_local[4 * 6 + 4]);
                                A[idx23] = beta * A[idx23] + alpha * (T_local[1 * 6 + 2] + T_local[4 * 6 + 5]);
                                A[idx24] = beta * A[idx24] + alpha * (T_local[1 * 6 + 3] + T_local[4 * 6 + 0]);
                                A[idx25] = beta * A[idx25] + alpha * (T_local[1 * 6 + 4] + T_local[4 * 6 + 1]);
                                A[idx26] = beta * A[idx26] + alpha * (T_local[1 * 6 + 5] + T_local[4 * 6 + 2]);
                                A[idx31] = beta * A[idx31] + alpha * (T_local[2 * 6 + 0] + T_local[5 * 6 + 3]);
                                A[idx32] = beta * A[idx32] + alpha * (T_local[2 * 6 + 1] + T_local[5 * 6 + 4]);
                                A[idx33] = beta * A[idx33] + alpha * (T_local[2 * 6 + 2] + T_local[5 * 6 + 5]);
                                A[idx34] = beta * A[idx34] + alpha * (T_local[2 * 6 + 3] + T_local[5 * 6 + 0]);
                                A[idx35] = beta * A[idx35] + alpha * (T_local[2 * 6 + 4] + T_local[5 * 6 + 1]);
                                A[idx36] = beta * A[idx36] + alpha * (T_local[2 * 6 + 5] + T_local[5 * 6 + 2]);
                                A[idx41] = beta * A[idx41] + alpha * (T_local[3 * 6 + 0] + T_local[0 * 6 + 3]);
                                A[idx42] = beta * A[idx42] + alpha * (T_local[3 * 6 + 1] + T_local[0 * 6 + 4]);
                                A[idx43] = beta * A[idx43] + alpha * (T_local[3 * 6 + 2] + T_local[0 * 6 + 5]);
                                A[idx44] = beta * A[idx44] + alpha * (T_local[3 * 6 + 3] + T_local[0 * 6 + 0]);
                                A[idx45] = beta * A[idx45] + alpha * (T_local[3 * 6 + 4] + T_local[0 * 6 + 1]);
                                A[idx46] = beta * A[idx46] + alpha * (T_local[3 * 6 + 5] + T_local[0 * 6 + 2]);
                                A[idx51] = beta * A[idx51] + alpha * (T_local[4 * 6 + 0] + T_local[1 * 6 + 3]);
                                A[idx52] = beta * A[idx52] + alpha * (T_local[4 * 6 + 1] + T_local[1 * 6 + 4]);
                                A[idx53] = beta * A[idx53] + alpha * (T_local[4 * 6 + 2] + T_local[1 * 6 + 5]);
                                A[idx54] = beta * A[idx54] + alpha * (T_local[4 * 6 + 3] + T_local[1 * 6 + 0]);
                                A[idx55] = beta * A[idx55] + alpha * (T_local[4 * 6 + 4] + T_local[1 * 6 + 1]);
                                A[idx56] = beta * A[idx56] + alpha * (T_local[4 * 6 + 5] + T_local[1 * 6 + 2]);
                                A[idx61] = beta * A[idx61] + alpha * (T_local[5 * 6 + 0] + T_local[2 * 6 + 3]);
                                A[idx62] = beta * A[idx62] + alpha * (T_local[5 * 6 + 1] + T_local[2 * 6 + 4]);
                                A[idx63] = beta * A[idx63] + alpha * (T_local[5 * 6 + 2] + T_local[2 * 6 + 5]);
                                A[idx64] = beta * A[idx64] + alpha * (T_local[5 * 6 + 3] + T_local[2 * 6 + 0]);
                                A[idx65] = beta * A[idx65] + alpha * (T_local[5 * 6 + 4] + T_local[2 * 6 + 1]);
                                A[idx66] = beta * A[idx66] + alpha * (T_local[5 * 6 + 5] + T_local[2 * 6 + 2]);
                            }
                        }
                    }
                }
            }
        }
    }
}

const int64_t tp_t3[6][3] = {
    {0, 1, 2}, // no permutation
    {0, 2, 1}, // swap b <-> c
    {1, 0, 2}, // swap a <-> b
    {1, 2, 0}, // a->b, b->c, c->a
    {2, 0, 1}, // a->c, b->a, c->b
    {2, 1, 0}, // reverse
};

// Unpack triangular-stored T3 amplitudes into a full T3 block.
//
// This kernel reconstructs the full permutation-expanded T3 tensor block from the compressed triangular
// representation without forming the full tensor in memory.
//
// Input:
//   t3_tri                    : triangular-stored T3 amplitudes
//   t3_blk                    : output buffer [blk_i * blk_j * blk_k * nvir**3]
//   map                       : mapping index table for (i, j, k) -> tri index
//   mask                      : mask indicating which (i, j, k) indices are stored (triangular domain)
//   [i0:i1), [j0:j1), [k0:k1) : occupied index block ranges
//   nocc, nvir                : number of occupied / virtual orbitals
//   blk_i, blk_j, blk_k       : block sizes for the destination tensor
void unpack_t3_tri2block_(const double *restrict t3_tri,
                          double *restrict t3_blk,
                          const int64_t *restrict map,
                          const bool *restrict mask,
                          int64_t i0, int64_t i1,
                          int64_t j0, int64_t j1,
                          int64_t k0, int64_t k1,
                          int64_t nocc, int64_t nvir,
                          int64_t blk_i, int64_t blk_j, int64_t blk_k)
{
#define MAP(sym, x, y, z) map[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]
#define MASK(sym, x, y, z) mask[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]

#pragma omp parallel for collapse(4) schedule(dynamic)
    for (int64_t sym = 0; sym < 6; ++sym)
    {
        for (int64_t i = i0; i < i1; ++i)
        {
            for (int64_t j = j0; j < j1; ++j)
            {
                for (int64_t k = k0; k < k1; ++k)
                {
                    if (!MASK(sym, i, j, k))
                        continue;

                    const int64_t *perm = tp_t3[sym];

                    int64_t loc_i = i - i0;
                    int64_t loc_j = j - j0;
                    int64_t loc_k = k - k0;

                    int64_t src_base = MAP(sym, i, j, k) * nvir * nvir * nvir;
                    int64_t dest_base = ((loc_i * blk_j + loc_j) * blk_k + loc_k) * nvir * nvir * nvir;

                    for (int64_t a = 0; a < nvir; ++a)
                    {
                        for (int64_t b = 0; b < nvir; ++b)
                        {
                            for (int64_t c = 0; c < nvir; ++c)
                            {
                                int64_t abc[3] = {a, b, c};
                                int64_t aa = abc[perm[0]];
                                int64_t bb = abc[perm[1]];
                                int64_t cc = abc[perm[2]];

                                int64_t src_idx = src_base + (a * nvir + b) * nvir + c;
                                int64_t dest_idx = dest_base + (aa * nvir + bb) * nvir + cc;

                                t3_blk[dest_idx] = t3_tri[src_idx];
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

// Unpack a triangular-stored T3 (i, j, k) element into its 6-fold
// permutation representation for a single occupied triplet.
//
// This routine identifies the symmetry representative of (i0, j0, k0) in the triangular (i <= j <= k) index domain,
// applies the corresponding (a, b, c) permutation, and scatters the resulting amplitudes into `t3_blk`.
// In addition, a second symmetry partner (selected via `tmp_indices`) is accumulated to complete the required
// two-term contribution.  Conceptually, this corresponds to reconstructing:
//
//     t3_full[i0, j0, k0, :, :, :] + t3_full[j0, i0, k0, :, :, :].transpose(1, 0, 2)
//
// Input
//   t3_tri     : triangular-stored T3 amplitudes
//   t3_blk     : output buffer [nvir**3]
//   map        : mapping (sym, i, j, k) -> tri index
//   mask       : triangular-domain mask for valid (i, j, k)
//   i0, j0, k0 : occupied indices for this element
//   nocc       : number of occupied orbitals
//   nvir       : number of virtual orbitals
void unpack_t3_tri2single_pair_(const double *restrict t3_tri,
                                double *restrict t3_blk,
                                const int64_t *restrict map,
                                const bool *restrict mask,
                                int64_t i0, int64_t j0, int64_t k0,
                                int64_t nocc, int64_t nvir)
{

#define MAP(sym, x, y, z) map[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]
#define MASK(sym, x, y, z) mask[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]

    int64_t sym;
    for (sym = 0; sym < 6; ++sym)
    {
        if (MASK(sym, i0, j0, k0))
            break;
    }

    const int64_t *perm = tp_t3[sym];
    int64_t idx = MAP(sym, i0, j0, k0);

#pragma omp parallel for collapse(3) schedule(static)
    for (int64_t a = 0; a < nvir; ++a)
    {
        for (int64_t b = 0; b < nvir; ++b)
        {
            for (int64_t c = 0; c < nvir; ++c)
            {
                int64_t abc[3] = {a, b, c};
                int64_t aa = abc[perm[0]];
                int64_t bb = abc[perm[1]];
                int64_t cc = abc[perm[2]];

                int64_t src_idx = ((idx * nvir + a) * nvir + b) * nvir + c;
                int64_t dest_idx = (aa * nvir + bb) * nvir + cc;

                t3_blk[dest_idx] = t3_tri[src_idx];
            }
        }
    }

    const int64_t tmp_indices[6] = {2, 4, 0, 5, 1, 3};

    for (sym = 0; sym < 6; ++sym)
    {
        if (MASK(tmp_indices[sym], i0, j0, k0))
            break;
    }

    const int64_t *perm2 = tp_t3[tmp_indices[sym]];
    idx = MAP(tmp_indices[sym], i0, j0, k0);

#pragma omp parallel for collapse(3) schedule(static)
    for (int64_t a = 0; a < nvir; ++a)
    {
        for (int64_t b = 0; b < nvir; ++b)
        {
            for (int64_t c = 0; c < nvir; ++c)
            {
                int64_t abc[3] = {a, b, c};
                int64_t aa = abc[perm2[0]];
                int64_t bb = abc[perm2[1]];
                int64_t cc = abc[perm2[2]];

                int64_t src_idx = ((idx * nvir + a) * nvir + b) * nvir + c;
                int64_t dest_idx = (aa * nvir + bb) * nvir + cc;

                t3_blk[dest_idx] += t3_tri[src_idx];
            }
        }
    }
#undef MAP
#undef MASK
}

// Unpack triangular-stored T3 amplitudes into a full T3 block.
//
// This kernel reconstructs the full permutation-expanded T3 tensor block from the compressed triangular
// representation without forming the full tensor in memory.
//
// Input:
//   t3_tri                    : triangular-stored T3 amplitudes
//   t3_blk                    : output buffer [blk_i * blk_j * blk_k * nvir**3]
//   map                       : mapping index table for (i, j, k) -> tri index
//   mask                      : mask indicating which (i, j, k) indices are stored (triangular domain)
//   [i0:i1), [j0:j1), [k0:k1) : occupied index block ranges
//   nocc, nvir                : number of occupied / virtual orbitals
//   blk_i, blk_j, blk_k       : block sizes for the destination tensor
void unpack_t3_tri2block_pair_(const double *restrict t3_tri,
                               double *restrict t3_blk,
                               const int64_t *restrict map,
                               const bool *restrict mask,
                               int64_t i0, int64_t i1,
                               int64_t j0, int64_t j1,
                               int64_t k0, int64_t k1,
                               int64_t nocc, int64_t nvir,
                               int64_t blk_i, int64_t blk_j, int64_t blk_k)
{

#define MAP(sym, x, y, z) map[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]
#define MASK(sym, x, y, z) mask[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]

    const int64_t tmp_indices[6] = {5, 3, 4, 1, 2, 0};
    const int64_t trans_indices[6] = {1, 0, 3, 2, 5, 4};

#pragma omp parallel for collapse(4) schedule(dynamic)
    for (int64_t sym = 0; sym < 6; ++sym)
    {
        for (int64_t i = i0; i < i1; ++i)
        {
            for (int64_t j = j0; j < j1; ++j)
            {
                for (int64_t k = k0; k < k1; ++k)
                {
                    if (!MASK(sym, i, j, k))
                        continue;

                    const int64_t *perm = tp_t3[sym];

                    int64_t loc_i = i - i0;
                    int64_t loc_j = j - j0;
                    int64_t loc_k = k - k0;

                    int64_t src_base = MAP(sym, i, j, k) * nvir * nvir * nvir;
                    int64_t dest_base = ((loc_i * blk_j + loc_j) * blk_k + loc_k) * nvir * nvir * nvir;

                    for (int64_t a = 0; a < nvir; ++a)
                    {
                        for (int64_t b = 0; b < nvir; ++b)
                        {
                            for (int64_t c = 0; c < nvir; ++c)
                            {
                                int64_t abc[3] = {a, b, c};
                                int64_t aa = abc[perm[0]];
                                int64_t bb = abc[perm[1]];
                                int64_t cc = abc[perm[2]];

                                int64_t src_idx = src_base + (a * nvir + b) * nvir + c;
                                int64_t dest_idx = dest_base + (aa * nvir + bb) * nvir + cc;

                                t3_blk[dest_idx] = t3_tri[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }

#pragma omp parallel for collapse(4) schedule(dynamic)
    for (int64_t sym = 0; sym < 6; ++sym)
    {
        for (int64_t i = i0; i < i1; ++i)
        {
            for (int64_t j = j0; j < j1; ++j)
            {
                for (int64_t k = k0; k < k1; ++k)
                {
                    if (!MASK(tmp_indices[sym], i, j, k))
                        continue;

                    const int64_t *perm2 = tp_t3[trans_indices[sym]];

                    int64_t loc_i = i - i0;
                    int64_t loc_j = j - j0;
                    int64_t loc_k = k - k0;

                    int64_t src_base = MAP(tmp_indices[sym], i, j, k) * nvir * nvir * nvir;
                    int64_t dest_base = ((loc_i * blk_j + loc_j) * blk_k + loc_k) * nvir * nvir * nvir;

                    for (int64_t a = 0; a < nvir; ++a)
                    {
                        for (int64_t b = 0; b < nvir; ++b)
                        {
                            for (int64_t c = 0; c < nvir; ++c)
                            {
                                int64_t abc[3] = {a, b, c};
                                int64_t aa = abc[perm2[0]];
                                int64_t bb = abc[perm2[1]];
                                int64_t cc = abc[perm2[2]];

                                int64_t src_idx = src_base + (a * nvir + b) * nvir + c;
                                int64_t dest_idx = dest_base + (aa * nvir + bb) * nvir + cc;

                                t3_blk[dest_idx] += t3_tri[src_idx];
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

// Accumulate a full T3 block into the triangular-stored T3 amplitudes.
//
// This routine performs the reverse of the unpacking:
// given a (i0:i1, j0:j1, k0:k1) block of the full T3 tensor in t3_blk, it updates the
// corresponding triangular-stored T3 buffer t3_tri using
//
//     t3_tri = beta * t3_tri + alpha * t3_blk
//
// Only unique (i <= j <= k) indices are processed. The mapping table `map`
// provides the location of the triangular representative for each index triplet.
//
// Inputs:
//     t3_tri                    : triangular-stored T3 amplitudes
//     t3_blk                    : full T3 block [blk_i * blk_j * blk_k * nvir**3]
//     map                       : maps (sym, i, j, k) -> triangular index
//     [i0:i1), [j0:j1), [k0:k1) : occupied index block ranges
//     nocc, nvir                : number of occupied / virtual orbitals
//     blk_i, blk_j, blk_k       : block dimensions for t3_blk
//     alpha, beta               : scaling factors for update
void accumulate_t3_block2tri_(double *restrict t3_tri,
                              const double *restrict t3_blk,
                              const int64_t *restrict map,
                              int64_t i0, int64_t i1,
                              int64_t j0, int64_t j1,
                              int64_t k0, int64_t k1,
                              int64_t nocc, int64_t nvir,
                              int64_t blk_i, int64_t blk_j, int64_t blk_k,
                              double alpha, double beta)
{
#define MAP(sym, x, y, z) map[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]

    if (j1 < i0 || k1 < j0)
        return;

#pragma omp parallel for collapse(3) schedule(dynamic)
    for (int64_t k = k0; k < k1; ++k)
    {
        for (int64_t j = j0; j < j1; ++j)
        {
            for (int64_t i = i0; i < i1; ++i)
            {
                if (j > k || i > j)
                    continue;

                int64_t p = MAP(0, i, j, k);
                int64_t tri_base = p * nvir * nvir * nvir;

                int64_t loc_i = i - i0;
                int64_t loc_j = j - j0;
                int64_t loc_k = k - k0;
                int64_t blk_base = ((loc_i * blk_j + loc_j) * blk_k + loc_k) * nvir * nvir * nvir;

                for (int64_t a = 0; a < nvir; ++a)
                {
                    for (int64_t b = 0; b < nvir; ++b)
                    {
                        for (int64_t c = 0; c < nvir; ++c)
                        {
                            int64_t idx = ((a * nvir + b) * nvir + c);
                            t3_tri[tri_base + idx] = beta * t3_tri[tri_base + idx] + alpha * t3_blk[blk_base + idx];
                        }
                    }
                }
            }
        }
    }
#undef MAP
}

// Accumulate a single (i0, j0, k0) full T3 slice into the triangular 6-fold compressed T3 storage.
//
// Inputs
//   t3_tri      : triangular-stored T3 amplitudes
//   t3_blk      : full T3 slice [nvir**3] for (i0, j0, k0)
//   map         : mapping (sym, i, j, k) -> triangular index (sym = 0 used here)
//   i0, j0, k0  : occupied indices
//   nocc        : number of occupied orbitals
//   nvir        : number of virtual orbitals
//   alpha, beta : scaling coefficients for accumulation
void accumulate_t3_single2tri_(double *restrict t3_tri,
                               const double *restrict t3_blk,
                               const int64_t *restrict map,
                               int64_t i0, int64_t j0, int64_t k0,
                               int64_t nocc, int64_t nvir,
                               double alpha, double beta)
{
#define MAP(sym, x, y, z) map[(((sym) * nocc + (x)) * nocc + (y)) * nocc + (z)]

    int64_t p = MAP(0, i0, j0, k0);
    int64_t tri_base = p * nvir * nvir * nvir;

#pragma omp parallel for collapse(3) schedule(static)
    for (int64_t a = 0; a < nvir; ++a)
    {
        for (int64_t b = 0; b < nvir; ++b)
        {
            for (int64_t c = 0; c < nvir; ++c)
            {
                int64_t idx = ((a * nvir + b) * nvir + c);
                t3_tri[tri_base + idx] = beta * t3_tri[tri_base + idx] + alpha * t3_blk[idx];
            }
        }
    }
#undef MAP
}
