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
#include <stdint.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "cint.h"
#include "vhf/fblas.h"

#define ALIGNMENT       8
#define UNROLL_SIZE     (ALIGNMENT*7)
#define BLKSIZE         (ALIGNMENT*7)
#define BOXSIZE1_M      (ALIGNMENT*14)
#define BOXSIZE1_N      96
#define BOXSIZE1_K      BOXSIZE1_N
#define NCTR_CART       80
#define MAX_THREADS     256

int CVHFshls_block_partition(int *block_loc, int *shls_slice, int *ao_loc,
                             int block_size);

static void _dot_ao_dm_l1(double *out, double *ao, double *dm,
                          int nao, size_t ngrids, int nbas, int ig0, int ig1,
                          int ish0, int ish1, int jsh0, int jsh1, int nbins,
                          uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc)
{
        size_t Nao = nao;
        int ig, ish, jsh, i0, i1, i, j, box_id, n;
        uint8_t si_i, si_j, nbins_i;  // be careful with overflow
        size_t i_addr, j_addr;
        double dm_val;
        double s8[UNROLL_SIZE];
        double *dm_j;

        for (ig = ig0; ig < ig1; ig+=UNROLL_SIZE) {
                box_id = ig / BLKSIZE;
                for (jsh = jsh0; jsh < jsh1; jsh++) {
                        si_j = screen_index[box_id * nbas + jsh];
                        if (si_j != 0) {
                                if (nbins > si_j) {
                                        nbins_i = nbins - si_j;
                                } else {
                                        nbins_i = 1;
                                }
                                for (j = ao_loc[jsh]; j < ao_loc[jsh+1]; j++) {
                                        dm_j = dm + j;
                                        j_addr = j * ngrids + ig;
                                        for (n = 0; n < UNROLL_SIZE; n++) {
                                                s8[n] = out[j_addr+n];
                                        }
                                        for (ish = ish0; ish < ish1; ish++) {
                                                si_i = screen_index[box_id * nbas + ish];
                                                if (si_i >= nbins_i &&
                                                    pair_mask[ish*nbas+jsh]) {
i0 = ao_loc[ish];
i1 = ao_loc[ish+1];
switch (i1 - i0) {
case 1:
        for (i = i0; i < i0 + 1; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        } break;
case 2:
        for (i = i0; i < i0 + 2; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        } break;
case 3:
        for (i = i0; i < i0 + 3; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        } break;
case 5:
        for (i = i0; i < i0 + 5; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        } break;
case 6:
        for (i = i0; i < i0 + 6; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        } break;
case 7:
        for (i = i0; i < i0 + 7; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        } break;
case 10:
        for (i = i0; i < i0 + 10; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        } break;
default:
        for (i = i0; i < i1; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        }
}
                                                }
                                        }
                                        for (n = 0; n < UNROLL_SIZE; n++) {
                                                out[j_addr+n] = s8[n];
                                        }
                                }
                        }
                }
        }
}

static void _dot_ao_dm_frac(double *out, double *ao, double *dm,
                            int nao, size_t ngrids, int nbas, int ig0, int nbins,
                            uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc)
{
        size_t Nao = nao;
        int ngrids_rest = ngrids - ig0;
        int ish, jsh, i0, i1, i, j, n;
        uint8_t si_i, si_j, nbins_i;
        size_t i_addr, j_addr;
        double dm_val;
        double s8[UNROLL_SIZE];
        double *dm_j;

        int box_id = ig0 / BLKSIZE;
        for (jsh = 0; jsh < nbas; jsh++) {
                si_j = screen_index[box_id * nbas + jsh];
                if (si_j != 0) {
                        if (nbins > si_j) {
                                nbins_i = nbins - si_j;
                        } else {
                                nbins_i = 1;
                        }
                        for (j = ao_loc[jsh]; j < ao_loc[jsh+1]; j++) {
                                dm_j = dm + j;
                                j_addr = j * ngrids + ig0;
                                for (n = 0; n < ngrids_rest; n++) {
                                        s8[n] = 0;
                                }
                                for (ish = 0; ish < nbas; ish++) {
                                        si_i = screen_index[box_id * nbas + ish];
                                        if (si_i >= nbins_i && pair_mask[ish*nbas+jsh]) {
                                                i0 = ao_loc[ish];
                                                i1 = ao_loc[ish+1];
                                                for (i = i0; i < i1; i++) {
                                                        dm_val = dm_j[i * Nao];
                                                        i_addr = i * ngrids + ig0;
                                                        for (n = 0; n < ngrids_rest; n++) {
                                                                s8[n] += ao[i_addr+n] * dm_val;
                                                        }
                                                }
                                        }
                                }
                                for (n = 0; n < ngrids_rest; n++) {
                                        out[j_addr+n] = s8[n];
                                }
                        }
                }
        }
}

// return number of non-zero values in mask array
static void mask_l1_abstract(uint8_t *out, uint8_t *mask, int *box_loc,
                             int nbox, int ngrids, int nbas)
{
        int i, m, ig, box_id, ig0, ig1, i0, i1, n;
        int with_value;
        for (n = 0, ig0 = 0; ig0 < ngrids; ig0 += BOXSIZE1_M) {
        for (box_id = 0; box_id < nbox; box_id++, n++) {
                i0 = box_loc[box_id];
                i1 = box_loc[box_id+1];
                ig1 = MIN(ig0+BOXSIZE1_M, ngrids);
                with_value = 0;
                for (i = i0; i < i1; i++) {
                for (ig = ig0; ig < ig1; ig+=BLKSIZE) {
                        m = ig / BLKSIZE;
                        if (mask[m*nbas+i] != 0) {
                                with_value = 1;
                                goto next_l1_box;
                        }
                } }
next_l1_box:
                out[n] = with_value;
        } }
}


void VXCdot_ao_dm_sparse(double *out, double *ao, double *dm,
                         int nao, int ngrids, int nbas, int nbins,
                         uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc)
{
        size_t Ngrids = ngrids;
        int shls_slice[2] = {0, nbas};
        int *box_l1_loc = malloc(sizeof(int) * (nbas+1));
        int nbox_l1 = CVHFshls_block_partition(box_l1_loc, shls_slice, ao_loc, BOXSIZE1_N);
        int mask_l1_size = (ngrids + BOXSIZE1_M - 1)/BOXSIZE1_M * nbox_l1;
        uint8_t *mask_l1 = malloc(sizeof(uint8_t) * mask_l1_size);
        mask_l1_abstract(mask_l1, screen_index, box_l1_loc, nbox_l1, ngrids, nbas);
        int ngrids_align_down = (ngrids / BLKSIZE) * BLKSIZE;

        if (nao * 2 < ngrids) {
#pragma omp parallel
{
                int ig, j, j0, j1, jsh0, jsh1, ib, jb, ig0, ig1, ig_box2;
#pragma omp for schedule(dynamic)
                for (ig0 = 0; ig0 < ngrids_align_down; ig0+=BOXSIZE1_M) {
                        ig1 = MIN(ig0 + BOXSIZE1_M, ngrids_align_down);
                        ig_box2 = ig0 / BOXSIZE1_M;
                        for (jb = 0; jb < nbox_l1; jb++) {
                                if (mask_l1[ig_box2 * nbox_l1 + jb]) {
                                        jsh0 = box_l1_loc[jb];
                                        jsh1 = box_l1_loc[jb+1];
                                        j0 = ao_loc[jsh0];
                                        j1 = ao_loc[jsh1];
                                        for (j = j0; j < j1; j++) {
                                        for (ig = ig0; ig < ig1; ig++) {
                                                out[j*Ngrids+ig] = 0;
                                        } }
                                        for (ib = 0; ib < nbox_l1; ib++) {
                                                if (mask_l1[ig_box2 * nbox_l1 + ib]) {
_dot_ao_dm_l1(out, ao, dm, nao, ngrids, nbas, ig0, ig1,
              box_l1_loc[ib], box_l1_loc[ib+1], jsh0, jsh1,
              nbins, screen_index, pair_mask, ao_loc);
                                                }
                                        }
                                }
                        }
                }
}
        } else {
#pragma omp parallel
{
                int ig, j, j0, j1, jsh0, jsh1, ib, jb, ig0, ig1, ig_box2;
#pragma omp for schedule(dynamic)
                for (jb = 0; jb < nbox_l1; jb++) {
                        jsh0 = box_l1_loc[jb];
                        jsh1 = box_l1_loc[jb+1];
                        j0 = ao_loc[jsh0];
                        j1 = ao_loc[jsh1];
                        for (ig0 = 0; ig0 < ngrids_align_down; ig0+=BOXSIZE1_M) {
                                ig1 = MIN(ig0 + BOXSIZE1_M, ngrids_align_down);
                                ig_box2 = ig0 / BOXSIZE1_M;
                                if (mask_l1[ig_box2 * nbox_l1 + jb]) {
                                        for (j = j0; j < j1; j++) {
                                        for (ig = ig0; ig < ig1; ig++) {
                                                out[j*Ngrids+ig] = 0;
                                        } }
                                        for (ib = 0; ib < nbox_l1; ib++) {
                                                if (mask_l1[ig_box2 * nbox_l1 + ib]) {
_dot_ao_dm_l1(out, ao, dm, nao, ngrids, nbas, ig0, ig1,
              box_l1_loc[ib], box_l1_loc[ib+1], jsh0, jsh1,
              nbins, screen_index, pair_mask, ao_loc);
                                                }
                                        }
                                }
                        }
                }
}
        }
        if (ngrids_align_down < ngrids) {
                _dot_ao_dm_frac(out, ao, dm, nao, ngrids, nbas, ngrids_align_down,
                                nbins, screen_index, pair_mask, ao_loc);
        }
        free(box_l1_loc);
        free(mask_l1);
}

static void _dot_aow_ao_l1(double *out, double *bra, double *ket, double *wv,
                           int nao, size_t ngrids, int nbas, int ig0, int ig1,
                           int ish0, int ish1, int jsh0, int jsh1, int nj,
                           int ioff, int joff, int nbins, uint8_t *screen_index,
                           uint8_t *pair_mask, int *ao_loc)
{
        int ish, jsh, i0, i1, j0, j1, i, j, ij, n;
        uint8_t si_i0, si_i1, si_j0, si_j1, nbins_j0, nbins_j1;
        size_t i_addr, j_addr;
        double braw[BOXSIZE1_M];
        double s8[ALIGNMENT];
        int i_pattern, j_pattern;
        int gblk0 = ig0 / BLKSIZE;

        for (ish = ish0; ish < ish1; ish++) {
                si_i0 = screen_index[(gblk0+0) * nbas + ish];
                si_i1 = screen_index[(gblk0+1) * nbas + ish];
                i_pattern = (si_i0 != 0) | ((si_i1 != 0) << 1);
                if (!i_pattern) {
                        goto next_i;
                }
                if (nbins > si_i0) {
                        nbins_j0 = nbins - si_i0;
                } else {
                        nbins_j0 = 1;
                }
                if (nbins > si_i1) {
                        nbins_j1 = nbins - si_i1;
                } else {
                        nbins_j1 = 1;
                }
                i0 = ao_loc[ish];
                i1 = ao_loc[ish+1];
                for (i = i0; i < i1; i++) {
                        i_addr = i * ngrids + ig0;
                        switch (i_pattern) {
                        case 0:
                                goto next_i;
                        case 1 :
                                for (n = 0; n < ALIGNMENT; n++) {
                                        braw[n            ] = bra[i_addr+n            ] * wv[ig0+n            ];
                                        braw[n+ALIGNMENT*1] = bra[i_addr+n+ALIGNMENT*1] * wv[ig0+n+ALIGNMENT*1];
                                        braw[n+ALIGNMENT*2] = bra[i_addr+n+ALIGNMENT*2] * wv[ig0+n+ALIGNMENT*2];
                                        braw[n+ALIGNMENT*3] = bra[i_addr+n+ALIGNMENT*3] * wv[ig0+n+ALIGNMENT*3];
                                        braw[n+ALIGNMENT*4] = bra[i_addr+n+ALIGNMENT*4] * wv[ig0+n+ALIGNMENT*4];
                                        braw[n+ALIGNMENT*5] = bra[i_addr+n+ALIGNMENT*5] * wv[ig0+n+ALIGNMENT*5];
                                        braw[n+ALIGNMENT*6] = bra[i_addr+n+ALIGNMENT*6] * wv[ig0+n+ALIGNMENT*6];
                                }
                                break;
                        case 2 :
                                for (n = 0; n < ALIGNMENT; n++) {
                                        braw[n+ALIGNMENT*7 ] = bra[i_addr+n+ALIGNMENT*7 ] * wv[ig0+n+ALIGNMENT*7 ];
                                        braw[n+ALIGNMENT*8 ] = bra[i_addr+n+ALIGNMENT*8 ] * wv[ig0+n+ALIGNMENT*8 ];
                                        braw[n+ALIGNMENT*9 ] = bra[i_addr+n+ALIGNMENT*9 ] * wv[ig0+n+ALIGNMENT*9 ];
                                        braw[n+ALIGNMENT*10] = bra[i_addr+n+ALIGNMENT*10] * wv[ig0+n+ALIGNMENT*10];
                                        braw[n+ALIGNMENT*11] = bra[i_addr+n+ALIGNMENT*11] * wv[ig0+n+ALIGNMENT*11];
                                        braw[n+ALIGNMENT*12] = bra[i_addr+n+ALIGNMENT*12] * wv[ig0+n+ALIGNMENT*12];
                                        braw[n+ALIGNMENT*13] = bra[i_addr+n+ALIGNMENT*13] * wv[ig0+n+ALIGNMENT*13];
                                }
                                break;
                        case 3 :
                                for (n = 0; n < ALIGNMENT; n++) {
                                        braw[n             ] = bra[i_addr+n             ] * wv[ig0+n             ];
                                        braw[n+ALIGNMENT*1 ] = bra[i_addr+n+ALIGNMENT*1 ] * wv[ig0+n+ALIGNMENT*1 ];
                                        braw[n+ALIGNMENT*2 ] = bra[i_addr+n+ALIGNMENT*2 ] * wv[ig0+n+ALIGNMENT*2 ];
                                        braw[n+ALIGNMENT*3 ] = bra[i_addr+n+ALIGNMENT*3 ] * wv[ig0+n+ALIGNMENT*3 ];
                                        braw[n+ALIGNMENT*4 ] = bra[i_addr+n+ALIGNMENT*4 ] * wv[ig0+n+ALIGNMENT*4 ];
                                        braw[n+ALIGNMENT*5 ] = bra[i_addr+n+ALIGNMENT*5 ] * wv[ig0+n+ALIGNMENT*5 ];
                                        braw[n+ALIGNMENT*6 ] = bra[i_addr+n+ALIGNMENT*6 ] * wv[ig0+n+ALIGNMENT*6 ];
                                        braw[n+ALIGNMENT*7 ] = bra[i_addr+n+ALIGNMENT*7 ] * wv[ig0+n+ALIGNMENT*7 ];
                                        braw[n+ALIGNMENT*8 ] = bra[i_addr+n+ALIGNMENT*8 ] * wv[ig0+n+ALIGNMENT*8 ];
                                        braw[n+ALIGNMENT*9 ] = bra[i_addr+n+ALIGNMENT*9 ] * wv[ig0+n+ALIGNMENT*9 ];
                                        braw[n+ALIGNMENT*10] = bra[i_addr+n+ALIGNMENT*10] * wv[ig0+n+ALIGNMENT*10];
                                        braw[n+ALIGNMENT*11] = bra[i_addr+n+ALIGNMENT*11] * wv[ig0+n+ALIGNMENT*11];
                                        braw[n+ALIGNMENT*12] = bra[i_addr+n+ALIGNMENT*12] * wv[ig0+n+ALIGNMENT*12];
                                        braw[n+ALIGNMENT*13] = bra[i_addr+n+ALIGNMENT*13] * wv[ig0+n+ALIGNMENT*13];
                                }
                        }

                        for (jsh = jsh0; jsh < jsh1; jsh++) {
                                if (!pair_mask[ish*nbas+jsh]) {
                                        goto next_j;
                                }
                                si_j0 = screen_index[(gblk0+0) * nbas + jsh];
                                si_j1 = screen_index[(gblk0+1) * nbas + jsh];
                                j_pattern = (si_j0 >= nbins_j0) | ((si_j1 >= nbins_j1) << 1);
                                switch (j_pattern & i_pattern) {
                                case 0 :
                                        goto next_j;
                                case 1 :
                                        j0 = ao_loc[jsh];
                                        j1 = ao_loc[jsh+1];
                                        ij = (i - ioff) * nj - joff;
                                        for (j = j0; j < j1; j++) {
                                                j_addr = j * ngrids + ig0;
                                                for (n = 0; n < ALIGNMENT; n++) {
                                                        s8[n] = out[(ij+j)*ALIGNMENT+n];
                                                        s8[n] += braw[n             ] * ket[j_addr+n             ];
                                                        s8[n] += braw[n+ALIGNMENT*1 ] * ket[j_addr+n+ALIGNMENT*1 ];
                                                        s8[n] += braw[n+ALIGNMENT*2 ] * ket[j_addr+n+ALIGNMENT*2 ];
                                                        s8[n] += braw[n+ALIGNMENT*3 ] * ket[j_addr+n+ALIGNMENT*3 ];
                                                        s8[n] += braw[n+ALIGNMENT*4 ] * ket[j_addr+n+ALIGNMENT*4 ];
                                                        s8[n] += braw[n+ALIGNMENT*5 ] * ket[j_addr+n+ALIGNMENT*5 ];
                                                        s8[n] += braw[n+ALIGNMENT*6 ] * ket[j_addr+n+ALIGNMENT*6 ];
                                                        out[(ij+j)*ALIGNMENT+n] = s8[n];
                                                }
                                        }
                                        break;
                                case 2 :
                                        j0 = ao_loc[jsh];
                                        j1 = ao_loc[jsh+1];
                                        ij = (i - ioff) * nj - joff;
                                        for (j = j0; j < j1; j++) {
                                                j_addr = j * ngrids + ig0;
                                                for (n = 0; n < ALIGNMENT; n++) {
                                                        s8[n] = out[(ij+j)*ALIGNMENT+n];
                                                        s8[n] += braw[n+ALIGNMENT*7 ] * ket[j_addr+n+ALIGNMENT*7 ];
                                                        s8[n] += braw[n+ALIGNMENT*8 ] * ket[j_addr+n+ALIGNMENT*8 ];
                                                        s8[n] += braw[n+ALIGNMENT*9 ] * ket[j_addr+n+ALIGNMENT*9 ];
                                                        s8[n] += braw[n+ALIGNMENT*10] * ket[j_addr+n+ALIGNMENT*10];
                                                        s8[n] += braw[n+ALIGNMENT*11] * ket[j_addr+n+ALIGNMENT*11];
                                                        s8[n] += braw[n+ALIGNMENT*12] * ket[j_addr+n+ALIGNMENT*12];
                                                        s8[n] += braw[n+ALIGNMENT*13] * ket[j_addr+n+ALIGNMENT*13];
                                                        out[(ij+j)*ALIGNMENT+n] = s8[n];
                                                }
                                        }
                                        break;
                                case 3 :
                                        j0 = ao_loc[jsh];
                                        j1 = ao_loc[jsh+1];
                                        ij = (i - ioff) * nj - joff;
                                        for (j = j0; j < j1; j++) {
                                                j_addr = j * ngrids + ig0;
                                                for (n = 0; n < ALIGNMENT; n++) {
                                                        s8[n] = out[(ij+j)*ALIGNMENT+n];
                                                        s8[n] += braw[n             ] * ket[j_addr+n             ];
                                                        s8[n] += braw[n+ALIGNMENT*1 ] * ket[j_addr+n+ALIGNMENT*1 ];
                                                        s8[n] += braw[n+ALIGNMENT*2 ] * ket[j_addr+n+ALIGNMENT*2 ];
                                                        s8[n] += braw[n+ALIGNMENT*3 ] * ket[j_addr+n+ALIGNMENT*3 ];
                                                        s8[n] += braw[n+ALIGNMENT*4 ] * ket[j_addr+n+ALIGNMENT*4 ];
                                                        s8[n] += braw[n+ALIGNMENT*5 ] * ket[j_addr+n+ALIGNMENT*5 ];
                                                        s8[n] += braw[n+ALIGNMENT*6 ] * ket[j_addr+n+ALIGNMENT*6 ];
                                                        s8[n] += braw[n+ALIGNMENT*7 ] * ket[j_addr+n+ALIGNMENT*7 ];
                                                        s8[n] += braw[n+ALIGNMENT*8 ] * ket[j_addr+n+ALIGNMENT*8 ];
                                                        s8[n] += braw[n+ALIGNMENT*9 ] * ket[j_addr+n+ALIGNMENT*9 ];
                                                        s8[n] += braw[n+ALIGNMENT*10] * ket[j_addr+n+ALIGNMENT*10];
                                                        s8[n] += braw[n+ALIGNMENT*11] * ket[j_addr+n+ALIGNMENT*11];
                                                        s8[n] += braw[n+ALIGNMENT*12] * ket[j_addr+n+ALIGNMENT*12];
                                                        s8[n] += braw[n+ALIGNMENT*13] * ket[j_addr+n+ALIGNMENT*13];
                                                        out[(ij+j)*ALIGNMENT+n] = s8[n];
                                                }
                                        }
                                }
next_j:;
                        }
                }
next_i:;
        }
}

static void _dot_aow_ao_frac(double *out, double *bra, double *ket, double *wv,
                             int nao, size_t ngrids, int nbas, int ig0,
                             int ish0, int ish1, int jsh0, int jsh1, int nj,
                             int ioff, int joff, int nbins, uint8_t *screen_index,
                             uint8_t *pair_mask, int *ao_loc)
{
        int ish, jsh, i0, i1, j0, j1, i, j, ij, k, n, ig, gblk0;
        int ng1, ng1_aligned, ng1_rest;
        uint8_t si_i, si_j, nbins_j;
        size_t i_addr, j_addr;
        double braw[BLKSIZE];
        double s8[ALIGNMENT];

        for (ig = ig0; ig < ngrids; ig+=BLKSIZE) {
                gblk0 = ig / BLKSIZE;
                ng1 = MIN(BLKSIZE, ngrids - ig);
                ng1_aligned = ng1 & (-(uint32_t)ALIGNMENT);
                for (ish = ish0; ish < ish1; ish++) {
                        si_i = screen_index[gblk0 * nbas + ish];
                        if (si_i != 0) {
                                if (nbins > si_i) {
                                        nbins_j = nbins - si_i;
                                } else {
                                        nbins_j = 1;
                                }
                                i0 = ao_loc[ish];
                                i1 = ao_loc[ish+1];
                                for (i = i0; i < i1; i++) {
i_addr = i * ngrids + ig;
for (k = 0; k < ng1_aligned; k+=ALIGNMENT) {
for (n = 0; n < ALIGNMENT; n++) {
         braw[n+k] = bra[i_addr+n+k] * wv[ig+n+k];
} }

for (jsh = jsh0; jsh < jsh1; jsh++) {
        si_j = screen_index[gblk0 * nbas + jsh];
        if (si_j >= nbins_j && pair_mask[ish*nbas+jsh]) {
                j0 = ao_loc[jsh];
                j1 = ao_loc[jsh+1];
                ij = (i - ioff) * nj - joff;
                for (j = j0; j < j1; j++) {
                        j_addr = j * ngrids + ig;
                        for (n = 0; n < ALIGNMENT; n++) {
                                s8[n] = out[(ij+j)*ALIGNMENT+n];
                        }
                        for (k = 0; k < ng1_aligned; k+=ALIGNMENT) {
                        for (n = 0; n < ALIGNMENT; n++) {
                                s8[n] += braw[n+k] * ket[j_addr+n+k];
                        } }
                        for (n = 0; n < ALIGNMENT; n++) {
                                out[(ij+j)*ALIGNMENT+n] = s8[n];
                        }
                }
        }
}
                                }
                        }
                }

                if (ng1_aligned < ng1) {
                        ng1_rest = ng1 - ng1_aligned;
                        for (ish = ish0; ish < ish1; ish++) {
                                si_i = screen_index[gblk0 * nbas + ish];
                                if (si_i != 0) {
                                        if (nbins > si_i) {
                                                nbins_j = nbins - si_i;
                                        } else {
                                                nbins_j = 1;
                                        }
                                        i0 = ao_loc[ish];
                                        i1 = ao_loc[ish+1];
                                        for (i = i0; i < i1; i++) {
i_addr = i * ngrids + ng1_aligned;
for (k = 0; k < ng1_rest; k++) {
         braw[k] = bra[i_addr+k] * wv[ng1_aligned+k];
}

for (jsh = jsh0; jsh < jsh1; jsh++) {
        si_j = screen_index[gblk0 * nbas + jsh];
        if (si_j >= nbins_j && pair_mask[ish*nbas+jsh]) {
                j0 = ao_loc[jsh];
                j1 = ao_loc[jsh+1];
                ij = (i - ioff) * nj - joff;
                for (j = j0; j < j1; j++) {
                        j_addr = j * ngrids + ng1_aligned;
                        s8[0] = 0.;
                        for (k = 0; k < ng1_rest; k++) {
                                s8[0] += braw[k] * ket[j_addr+k];
                        }
                        out[(ij+j)*ALIGNMENT] += s8[0];
                }
        }
}
                                        }
                                }
                        }
                }
        }
}

void VXCdot_aow_ao_dense(double *out, double *bra, double *ket, double *wv,
                         int nao, int ngrids)
{
        const size_t Nao = nao;
        const size_t Ngrids = ngrids;
        const int nao_blksize = 64;
        const int ngrids_blksize = 256;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D1 = 1;
#pragma omp parallel
{
        int i0, ig, i, di, ig0, ig1, dg;
        double *buf = malloc(sizeof(double) * (ngrids_blksize * nao_blksize + ALIGNMENT));
        double *braw = (double *)((uintptr_t)(buf + ALIGNMENT - 1) & (-(uintptr_t)(ALIGNMENT*8)));
        double *pbra;
#pragma omp for schedule(dynamic) nowait
        for (i0 = 0; i0 < nao; i0+=nao_blksize) {
                di = MIN(nao_blksize, nao - i0);
                for (ig0 = 0; ig0 < ngrids; ig0+=ngrids_blksize) {
                        ig1 = MIN(ig0+ngrids_blksize, ngrids);
                        pbra = bra + i0 * Ngrids + ig0;
                        dg = ig1 - ig0;
                        for (i = 0; i < di; i++) {
                        for (ig = 0; ig < dg; ig++) {
                                braw[i*ngrids_blksize+ig] = pbra[i*Ngrids+ig] * wv[ig0+ig];
                        } }
                        dgemm_(&TRANS_T, &TRANS_N, &nao, &di, &dg,
                               &D1, ket+ig0, &ngrids, braw, &ngrids_blksize,
                               &D1, out+i0*Nao, &nao);
                }
        }
        free(buf);
}
}

/* vv[nao,nao] = bra[i,nao] * ket[i,nao] */
void VXCdot_aow_ao_sparse(double *out, double *bra, double *ket, double *wv,
                          int nao, int ngrids, int nbas, int hermi, int nbins,
                          uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc)
{
        size_t Nao = nao;
        int shls_slice[2] = {0, nbas};
        int *box_l1_loc = malloc(sizeof(int) * (nbas+1));
        int nbox_l1 = CVHFshls_block_partition(box_l1_loc, shls_slice, ao_loc, BOXSIZE1_N);
        int mask_l1_size = (ngrids + BOXSIZE1_M - 1)/BOXSIZE1_M * nbox_l1;
        uint8_t *mask_l1 = malloc(sizeof(uint8_t) * mask_l1_size);
        mask_l1_abstract(mask_l1, screen_index, box_l1_loc, nbox_l1, ngrids, nbas);
        int ngrids_align_down = (ngrids / BOXSIZE1_M) * BOXSIZE1_M;

#pragma omp parallel
{
        int ijb, ib, jb, ib0, jb0, ib1, jb1, ig0, ig1, ig_box2;
        int ish0, ish1, jsh0, jsh1, i0, i1, j0, j1, ni, nj, i, j, n;
        double s;
        double *pout;
        double *buf = malloc(sizeof(double) * (BOXSIZE1_N*BOXSIZE1_N*ALIGNMENT+ALIGNMENT));
        double *outbuf = (double *)((uintptr_t)(buf+ALIGNMENT-1) & (-(uintptr_t)(ALIGNMENT*sizeof(double))));
#pragma omp for schedule(dynamic, 4) nowait
        for (ijb = 0; ijb < nbox_l1*nbox_l1; ijb++) {
                ib = ijb / nbox_l1;
                jb = ijb % nbox_l1;
                if (hermi && ib < jb) {
                        continue;
                }
                ib0 = ib;
                jb0 = jb;
                ib1 = ib + 1;
                jb1 = jb + 1;
                ish0 = box_l1_loc[ib0];
                jsh0 = box_l1_loc[jb0];
                ish1 = box_l1_loc[ib1];
                jsh1 = box_l1_loc[jb1];
                i0 = ao_loc[ish0];
                i1 = ao_loc[ish1];
                j0 = ao_loc[jsh0];
                j1 = ao_loc[jsh1];
                ni = i1 - i0;
                nj = j1 - j0;
                for (n = 0; n < ni * nj * ALIGNMENT; n++) {
                        outbuf[n] = 0;
                }

                for (ig0 = 0; ig0 < ngrids_align_down; ig0+=BOXSIZE1_M) {
                        ig1 = MIN(ig0 + BOXSIZE1_M, ngrids_align_down);
                        ig_box2 = ig0 / BOXSIZE1_M;
                        if (mask_l1[ig_box2 * nbox_l1 + jb] &&
                            mask_l1[ig_box2 * nbox_l1 + ib]) {
_dot_aow_ao_l1(outbuf, bra, ket, wv, nao, ngrids, nbas, ig0, ig1,
               ish0, ish1, jsh0, jsh1, nj, i0, j0, nbins, screen_index,
               pair_mask, ao_loc);
                        }
                }
                if (ngrids_align_down < ngrids) {
                        _dot_aow_ao_frac(outbuf, bra, ket, wv, nao, ngrids, nbas,
                                         ngrids_align_down, ish0, ish1, jsh0, jsh1,
                                         nj, i0, j0, nbins, screen_index,
                                         pair_mask, ao_loc);
                }

                for (i = 0; i < ni; i++) {
                for (j = 0; j < nj; j++) {
                        s = 0;
                        pout = outbuf + (i*nj+j)*ALIGNMENT;
                        for (n = 0; n < ALIGNMENT; n++) {
                                s += pout[n];
                        }
                        out[(i0+i)*Nao+j0+j] += s;
                } }
        }
        free(buf);
}
        free(box_l1_loc);
        free(mask_l1);

        if (hermi != 0) {
                NPdsymm_triu(nao, out, hermi);
        }
}

static void _dot_ao_ao_l1(double *out, double *bra, double *ket,
                          int nao, size_t ngrids, int nbas, int ig0, int ig1,
                          int ish0, int ish1, int jsh0, int jsh1, int nj,
                          int ioff, int joff, int nbins, uint8_t *screen_index,
                          uint8_t *pair_mask, int *ao_loc)
{
        int ish, jsh, i0, i1, j0, j1, i, j, ij, n;
        uint8_t si_i0, si_i1, si_j0, si_j1, nbins_j0, nbins_j1;
        size_t i_addr, j_addr;
        double s8[ALIGNMENT];
        int i_pattern, j_pattern;
        int gblk0 = ig0 / BLKSIZE;

        for (ish = ish0; ish < ish1; ish++) {
                si_i0 = screen_index[(gblk0+0) * nbas + ish];
                si_i1 = screen_index[(gblk0+1) * nbas + ish];
                i_pattern = (si_i0 != 0) | ((si_i1 != 0) << 1);
                if (i_pattern) {
                        if (nbins > si_i0) {
                                nbins_j0 = nbins - si_i0;
                        } else {
                                nbins_j0 = 1;
                        }
                        if (nbins > si_i1) {
                                nbins_j1 = nbins - si_i1;
                        } else {
                                nbins_j1 = 1;
                        }
                        i0 = ao_loc[ish];
                        i1 = ao_loc[ish+1];
                        for (i = i0; i < i1; i++) {
                                i_addr = i * ngrids + ig0;
for (jsh = jsh0; jsh < jsh1; jsh++) {
        if (!pair_mask[ish*nbas+jsh]) {
                goto next_j;
        }
        si_j0 = screen_index[(gblk0+0) * nbas + jsh];
        si_j1 = screen_index[(gblk0+1) * nbas + jsh];
        j_pattern = (si_j0 >= nbins_j0) | ((si_j1 >= nbins_j1) << 1);
        switch (j_pattern & i_pattern) {
        case 0 :
                goto next_j;
        case 1 :
                j0 = ao_loc[jsh];
                j1 = ao_loc[jsh+1];
                ij = (i - ioff) * nj - joff;
                for (j = j0; j < j1; j++) {
                        j_addr = j * ngrids + ig0;
                        for (n = 0; n < ALIGNMENT; n++) {
                                s8[n] = out[(ij+j)*ALIGNMENT+n];
                                s8[n] += bra[i_addr+n             ] * ket[j_addr+n             ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*1 ] * ket[j_addr+n+ALIGNMENT*1 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*2 ] * ket[j_addr+n+ALIGNMENT*2 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*3 ] * ket[j_addr+n+ALIGNMENT*3 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*4 ] * ket[j_addr+n+ALIGNMENT*4 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*5 ] * ket[j_addr+n+ALIGNMENT*5 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*6 ] * ket[j_addr+n+ALIGNMENT*6 ];
                                out[(ij+j)*ALIGNMENT+n] = s8[n];
                        }
                }
                break;
        case 2 :
                j0 = ao_loc[jsh];
                j1 = ao_loc[jsh+1];
                ij = (i - ioff) * nj - joff;
                for (j = j0; j < j1; j++) {
                        j_addr = j * ngrids + ig0;
                        for (n = 0; n < ALIGNMENT; n++) {
                                s8[n] = out[(ij+j)*ALIGNMENT+n];
                                s8[n] += bra[i_addr+n+ALIGNMENT*7 ] * ket[j_addr+n+ALIGNMENT*7 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*8 ] * ket[j_addr+n+ALIGNMENT*8 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*9 ] * ket[j_addr+n+ALIGNMENT*9 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*10] * ket[j_addr+n+ALIGNMENT*10];
                                s8[n] += bra[i_addr+n+ALIGNMENT*11] * ket[j_addr+n+ALIGNMENT*11];
                                s8[n] += bra[i_addr+n+ALIGNMENT*12] * ket[j_addr+n+ALIGNMENT*12];
                                s8[n] += bra[i_addr+n+ALIGNMENT*13] * ket[j_addr+n+ALIGNMENT*13];
                                out[(ij+j)*ALIGNMENT+n] = s8[n];
                        }
                }
                break;
        case 3 :
                j0 = ao_loc[jsh];
                j1 = ao_loc[jsh+1];
                ij = (i - ioff) * nj - joff;
                for (j = j0; j < j1; j++) {
                        j_addr = j * ngrids + ig0;
                        for (n = 0; n < ALIGNMENT; n++) {
                                s8[n] = out[(ij+j)*ALIGNMENT+n];
                                s8[n] += bra[i_addr+n             ] * ket[j_addr+n             ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*1 ] * ket[j_addr+n+ALIGNMENT*1 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*2 ] * ket[j_addr+n+ALIGNMENT*2 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*3 ] * ket[j_addr+n+ALIGNMENT*3 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*4 ] * ket[j_addr+n+ALIGNMENT*4 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*5 ] * ket[j_addr+n+ALIGNMENT*5 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*6 ] * ket[j_addr+n+ALIGNMENT*6 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*7 ] * ket[j_addr+n+ALIGNMENT*7 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*8 ] * ket[j_addr+n+ALIGNMENT*8 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*9 ] * ket[j_addr+n+ALIGNMENT*9 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*10] * ket[j_addr+n+ALIGNMENT*10];
                                s8[n] += bra[i_addr+n+ALIGNMENT*11] * ket[j_addr+n+ALIGNMENT*11];
                                s8[n] += bra[i_addr+n+ALIGNMENT*12] * ket[j_addr+n+ALIGNMENT*12];
                                s8[n] += bra[i_addr+n+ALIGNMENT*13] * ket[j_addr+n+ALIGNMENT*13];
                                out[(ij+j)*ALIGNMENT+n] = s8[n];
                        }
                }
        }
next_j:;
                                }
                        }
                }
        }
}

static void _dot_ao_ao_frac(double *out, double *bra, double *ket,
                            int nao, size_t ngrids, int nbas, int ig0,
                            int ish0, int ish1, int jsh0, int jsh1, int nj,
                            int ioff, int joff, int nbins, uint8_t *screen_index,
                            uint8_t *pair_mask, int *ao_loc)
{
        int ish, jsh, i0, i1, j0, j1, i, j, ij, k, n, ig, gblk0;
        int ng1, ng1_aligned, ng1_rest;
        uint8_t si_i, si_j, nbins_j;
        size_t i_addr, j_addr;
        double s8[ALIGNMENT];

        for (ig = ig0; ig < ngrids; ig+=BLKSIZE) {
                gblk0 = ig / BLKSIZE;
                ng1 = MIN(BLKSIZE, ngrids - ig);
                ng1_aligned = ng1 & (-(uint32_t)ALIGNMENT);
                for (ish = ish0; ish < ish1; ish++) {
                        si_i = screen_index[gblk0 * nbas + ish];
                        if (si_i != 0) {
                                if (nbins > si_i) {
                                        nbins_j = nbins - si_i;
                                } else {
                                        nbins_j = 1;
                                }
                                i0 = ao_loc[ish];
                                i1 = ao_loc[ish+1];
                                for (i = i0; i < i1; i++) {
i_addr = i * ngrids + ig;
for (jsh = jsh0; jsh < jsh1; jsh++) {
        si_j = screen_index[gblk0 * nbas + jsh];
        if (si_j >= nbins_j && pair_mask[ish*nbas+jsh]) {
                j0 = ao_loc[jsh];
                j1 = ao_loc[jsh+1];
                ij = (i - ioff) * nj - joff;
                for (j = j0; j < j1; j++) {
                        j_addr = j * ngrids + ig;
                        for (n = 0; n < ALIGNMENT; n++) {
                                s8[n] = out[(ij+j)*ALIGNMENT+n];
                        }
                        for (k = 0; k < ng1_aligned; k+=ALIGNMENT) {
                        for (n = 0; n < ALIGNMENT; n++) {
                                s8[n] += bra[i_addr+n+k] * ket[j_addr+n+k];
                        } }
                        for (n = 0; n < ALIGNMENT; n++) {
                                out[(ij+j)*ALIGNMENT+n] = s8[n];
                        }
                }
        }
}
                                }
                        }
                }

                if (ng1_aligned < ng1) {
                        ng1_rest = ng1 - ng1_aligned;
                        for (ish = ish0; ish < ish1; ish++) {
                                si_i = screen_index[gblk0 * nbas + ish];
                                if (si_i != 0) {
                                        if (nbins > si_i) {
                                                nbins_j = nbins - si_i;
                                        } else {
                                                nbins_j = 1;
                                        }
                                        i0 = ao_loc[ish];
                                        i1 = ao_loc[ish+1];
                                        for (i = i0; i < i1; i++) {
i_addr = i * ngrids + ng1_aligned;
for (jsh = jsh0; jsh < jsh1; jsh++) {
        si_j = screen_index[gblk0 * nbas + jsh];
        if (si_j >= nbins_j && pair_mask[ish*nbas+jsh]) {
                j0 = ao_loc[jsh];
                j1 = ao_loc[jsh+1];
                ij = (i - ioff) * nj - joff;
                for (j = j0; j < j1; j++) {
                        j_addr = j * ngrids + ng1_aligned;
                        s8[0] = 0.;
                        for (k = 0; k < ng1_rest; k++) {
                                s8[0] += bra[i_addr+k] * ket[j_addr+k];
                        }
                        out[(ij+j)*ALIGNMENT] += s8[0];
                }
        }
}
                                        }
                                }
                        }
                }
        }
}

/* out = bra.T.dot(ket) */
void VXCdot_ao_ao_sparse(double *out, double *bra, double *ket,
                         int nao, int ngrids, int nbas, int hermi, int nbins,
                         uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc)
{
        size_t Nao = nao;
        int shls_slice[2] = {0, nbas};
        int *box_l1_loc = malloc(sizeof(int) * (nbas+1));
        int nbox_l1 = CVHFshls_block_partition(box_l1_loc, shls_slice, ao_loc, BOXSIZE1_N);
        int mask_l1_size = (ngrids + BOXSIZE1_M - 1)/BOXSIZE1_M * nbox_l1;
        uint8_t *mask_l1 = malloc(sizeof(uint8_t) * mask_l1_size);
        mask_l1_abstract(mask_l1, screen_index, box_l1_loc,
                         nbox_l1, ngrids, nbas);
        int ngrids_align_down = (ngrids / BOXSIZE1_M) * BOXSIZE1_M;

#pragma omp parallel
{
        int ijb, ib, jb, ib0, jb0, ib1, jb1, ig0, ig1, ig_box2;
        int ish0, ish1, jsh0, jsh1, i0, i1, j0, j1, ni, nj, i, j, n;
        double s;
        double *pout;
        double *buf = malloc(sizeof(double) * (BOXSIZE1_N*BOXSIZE1_N*ALIGNMENT+ALIGNMENT));
        double *outbuf = (double *)((uintptr_t)(buf+ALIGNMENT-1) & (-(uintptr_t)(ALIGNMENT*sizeof(double))));
#pragma omp for schedule(dynamic, 4) nowait
        for (ijb = 0; ijb < nbox_l1*nbox_l1; ijb++) {
                ib = ijb / nbox_l1;
                jb = ijb % nbox_l1;
                if (hermi && ib < jb) {
                        continue;
                }
                ib0 = ib;
                jb0 = jb;
                ib1 = ib + 1;
                jb1 = jb + 1;
                ish0 = box_l1_loc[ib0];
                jsh0 = box_l1_loc[jb0];
                ish1 = box_l1_loc[ib1];
                jsh1 = box_l1_loc[jb1];
                i0 = ao_loc[ish0];
                i1 = ao_loc[ish1];
                j0 = ao_loc[jsh0];
                j1 = ao_loc[jsh1];
                ni = i1 - i0;
                nj = j1 - j0;
                for (n = 0; n < ni * nj * ALIGNMENT; n++) {
                        outbuf[n] = 0;
                }

                for (ig0 = 0; ig0 < ngrids_align_down; ig0+=BOXSIZE1_M) {
                        ig1 = MIN(ig0 + BOXSIZE1_M, ngrids_align_down);
                        ig_box2 = ig0 / BOXSIZE1_M;
                        if (mask_l1[ig_box2 * nbox_l1 + jb] &&
                            mask_l1[ig_box2 * nbox_l1 + ib]) {
_dot_ao_ao_l1(outbuf, bra, ket, nao, ngrids, nbas, ig0, ig1,
              ish0, ish1, jsh0, jsh1, nj, i0, j0, nbins, screen_index,
              pair_mask, ao_loc);
                        }
                }
                if (ngrids_align_down < ngrids) {
                        _dot_ao_ao_frac(outbuf, bra, ket, nao, ngrids, nbas,
                                        ngrids_align_down, ish0, ish1, jsh0, jsh1,
                                        nj, i0, j0, nbins, screen_index,
                                        pair_mask, ao_loc);
                }

                for (i = 0; i < ni; i++) {
                for (j = 0; j < nj; j++) {
                        s = 0;
                        pout = outbuf + (i*nj+j)*ALIGNMENT;
                        for (n = 0; n < ALIGNMENT; n++) {
                                s += pout[n];
                        }
                        out[(i0+i)*Nao+j0+j] += s;
                } }
        }
        free(buf);
}
        free(box_l1_loc);
        free(mask_l1);

        if (hermi != 0) {
                NPdsymm_triu(nao, out, hermi);
        }
}

// 'ip,ip->p'
void VXCdcontract_rho_sparse(double *rho, double *bra, double *ket,
                             int nao, int ngrids, int nbas,
                             uint8_t *screen_index, int *ao_loc)
{
        int mask_last_row = ngrids / BLKSIZE;
        int ngrids_align_down = mask_last_row * BLKSIZE;
        int ngrids_rest = ngrids - ngrids_align_down;

        if (nao * 2 < ngrids) {

#pragma omp parallel
{
                size_t Ngrids = ngrids;
                size_t i_addr;
                int ig0, row, ish, i0, i1, n, i;
                double s8[BLKSIZE];
#pragma omp for schedule(dynamic, 4) nowait
                for (ig0 = 0; ig0 < ngrids_align_down; ig0+=BLKSIZE) {
                        for (n = 0; n < BLKSIZE; n++) {
                                s8[n] = 0;
                        }
                        row = ig0 / BLKSIZE;
                        for (ish = 0; ish < nbas; ish++) {
                                if (screen_index[row * nbas + ish]) {
                                        i0 = ao_loc[ish];
                                        i1 = ao_loc[ish+1];
                                        for (i = i0; i < i1; i++) {
                                                i_addr = i * Ngrids + ig0;
                                                for (n = 0; n < BLKSIZE; n++) {
                                                        s8[n] += bra[i_addr+n] * ket[i_addr+n];
                                                }
                                        }
                                }
                        }
                        for (n = 0; n < BLKSIZE; n++) {
                                rho[ig0+n] = s8[n];
                        }
                }
}
                if (ngrids_align_down < ngrids) {
                        size_t Ngrids = ngrids;
                        size_t i_addr;
                        int ish, i0, i1, n, i;
                        double s8[BLKSIZE];
                        int ngrids_rest = ngrids - ngrids_align_down;
                        for (n = 0; n < ngrids_rest; n++) {
                                s8[n] = 0;
                        }
                        for (ish = 0; ish < nbas; ish++) {
                                if (screen_index[mask_last_row * nbas + ish]) {
                                        i0 = ao_loc[ish];
                                        i1 = ao_loc[ish+1];
                                        for (i = i0; i < i1; i++) {
                                                i_addr = i * Ngrids + ngrids_align_down;
                                                for (n = 0; n < ngrids_rest; n++) {
                                                        s8[n] += bra[i_addr+n] * ket[i_addr+n];
                                                }
                                        }
                                }
                        }
                        for (n = 0; n < ngrids_rest; n++) {
                                rho[ngrids_align_down+n] = s8[n];
                        }
                }

        } else {  // if (nao * 2 < ngrids)

                double *rhobufs[MAX_THREADS];
                rhobufs[0] = rho;
#pragma omp parallel
{
                size_t Ngrids = ngrids;
                size_t i_addr;
                int ig0, row, ish, i0, i1, n, i;
                double s8[BLKSIZE];
                int thread_id = omp_get_thread_num();
                if (thread_id > 0) {
                        rhobufs[thread_id] = malloc(sizeof(double) * ngrids);
                }
                double *rho_priv = rhobufs[thread_id];
                NPdset0(rho_priv, ngrids);
#pragma omp for schedule(dynamic, 4)
                for (ish = 0; ish < nbas; ish++) {
                        i0 = ao_loc[ish];
                        i1 = ao_loc[ish+1];
                        for (ig0 = 0; ig0 < ngrids_align_down; ig0+=BLKSIZE) {
                                for (n = 0; n < BLKSIZE; n++) {
                                        s8[n] = rho_priv[ig0+n];
                                }
                                row = ig0 / BLKSIZE;
                                if (screen_index[row * nbas + ish]) {
                                        for (i = i0; i < i1; i++) {
                                                i_addr = i * Ngrids + ig0;
                                                for (n = 0; n < BLKSIZE; n++) {
                                                        s8[n] += bra[i_addr+n] * ket[i_addr+n];
                                                }
                                        }
                                }
                                for (n = 0; n < BLKSIZE; n++) {
                                        rho_priv[ig0+n] = s8[n];
                                }
                        }
                        if (ngrids_rest) {
                                for (n = 0; n < ngrids_rest; n++) {
                                        s8[n] = rho_priv[ngrids_align_down+n];
                                }
                                if (screen_index[mask_last_row * nbas + ish]) {
                                        for (i = i0; i < i1; i++) {
                                                i_addr = i * Ngrids + ngrids_align_down;
                                                for (n = 0; n < ngrids_rest; n++) {
                                                        s8[n] += bra[i_addr+n] * ket[i_addr+n];
                                                }
                                        }
                                }
                                for (n = 0; n < ngrids_rest; n++) {
                                        rho_priv[ngrids_align_down+n] = s8[n];
                                }
                        }
                }
                NPomp_dsum_reduce_inplace(rhobufs, ngrids);
                if (thread_id > 0) {
                        free(rho_priv);
                }
}
        }  // if (nao * 2 < ngrids)
}

// 'nip,np->ip'
void VXCdscale_ao_sparse(double *aow, double *ao, double *wv,
                         int comp, int nao, int ngrids, int nbas,
                         uint8_t *screen_index, int *ao_loc)
{
#pragma omp parallel
{
        size_t Ngrids = ngrids;
        size_t ao_size = nao * Ngrids;
        int ish, i, j, ic, i0, i1, ig0, ig1, row;
        double *pao = ao;
#pragma omp for schedule(static, 2)
        for (ish = 0; ish < nbas; ish++) {
                i0 = ao_loc[ish];
                i1 = ao_loc[ish+1];
                for (ig0 = 0; ig0 < ngrids; ig0 += BLKSIZE) {
                        ig1 = MIN(ig0+BLKSIZE, ngrids);
                        row = ig0 / BLKSIZE;
                        if (screen_index[row * nbas + ish]) {
for (i = i0; i < i1; i++) {
        pao = ao + i * Ngrids;
        for (j = ig0; j < ig1; j++) {
                aow[i*Ngrids+j] = pao[j] * wv[j];
        }
        for (ic = 1; ic < comp; ic++) {
        for (j = ig0; j < ig1; j++) {
                aow[i*Ngrids+j] += pao[ic*ao_size+j] * wv[ic*Ngrids+j];
        } }
}
                        }
                }
        }
}
}
