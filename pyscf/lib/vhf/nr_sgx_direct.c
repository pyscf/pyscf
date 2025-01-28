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
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "cint.h"
#include "nr_direct.h"
#include "gto/gto.h"
#include "fblas.h"

#define MAX(I,J)        ((I) > (J) ? (I) : (J))
#define MIN(I,J)        ((I) < (J) ? (I) : (J))
#define ALIGNMENT       8
#define NI_BLKSIZE      56
#define UNROLL_SIZE     NI_BLKSIZE
#define BOXSIZE         96
#define SGX_BLKSIZE     224

typedef struct {
        int ncomp;
        int v_dims[3];
        double *data;
} SGXJKArray;

typedef struct {
        SGXJKArray *(*allocate)(int *shls_slice, int *ao_loc, int ncomp, int ngrids);
        //void (*contract)(double *eri, double *dm, SGXJKArray *vjk,
        //                 int i0, int i1, int j0, int j1);
        void (*contract)(double *eri, double *dm, SGXJKArray *vjk,
                         int i0, int i1, int j0, int j1,
                         int grid_offset, int ngrids);
        void (*set0)(SGXJKArray *, int);
        void (*send)(SGXJKArray *, int, int, double *);
        void (*finalize)(SGXJKArray *, double *);
        void (*sanity_check)(int *shls_slice);
} SGXJKOperator;

// for grids integrals only
size_t _max_cache_size_sgx(int (*intor)(), int *shls_slice, int ncenter,
                           int *atm, int natm, int *bas, int nbas, double *env)
{
        int i;
        int i0 = shls_slice[0];
        int i1 = shls_slice[1];
        for (i = 1; i < ncenter; i++) {
                i0 = MIN(i0, shls_slice[i*2  ]);
                i1 = MAX(i1, shls_slice[i*2+1]);
        }
        size_t (*f)() = (size_t (*)())intor;
        size_t cache_size = 0;
        size_t n;
        int shls[4];
        for (i = i0; i < i1; i++) {
                shls[0] = i;
                shls[1] = i;
                shls[2] = 0;
                shls[3] = SGX_BLKSIZE;
                n = (*f)(NULL, NULL, shls, atm, natm, bas, nbas, env, NULL, NULL);
                cache_size = MAX(cache_size, n);
        }
        return cache_size;
}

void SGXreduce_dft_mask(uint8_t *sgx_mask, uint8_t *dft_mask,
                        int dft_nblk, int nbas, int ratio) {
        const int sgx_nblk = (dft_nblk + ratio - 1) / ratio;
        const size_t Nbas = nbas;
#pragma omp parallel
{
        int sgx_ib, dft_ib, dft_ib0, dft_ib1, ish;
#pragma omp parallel for
        for (sgx_ib = 0; sgx_ib < sgx_nblk; sgx_ib++) {
                dft_ib0 = sgx_ib * ratio;
                dft_ib1 = MIN(dft_ib0 + ratio, dft_nblk);
                for (ish = 0; ish < nbas; ish++) {
                        sgx_mask[sgx_ib * nbas + ish] = 0;
                }
                for (dft_ib = dft_ib0; dft_ib < dft_ib1; dft_ib++) {
                        for (ish = 0; ish < nbas; ish++) {
                                sgx_mask[sgx_ib * Nbas + ish] |= 
                                        dft_mask[dft_ib * Nbas + ish];
                        }
                }
        }
}
}

void SGXmake_shl_dm(double *dm, double *shl_dm, int nao,
                    int nbas, int *ao_loc) {
#pragma omp parallel
{
        int ish, jsh, i, j;
#pragma omp for
        for (ish = 0; ish < nbas; ish++) {
        for (jsh = 0; jsh < nbas; jsh++) {
                shl_dm[ish * nbas + jsh] = 0;
                for (i = ao_loc[ish]; i < ao_loc[ish + 1]; i++) {
                for (j = ao_loc[jsh]; j < ao_loc[jsh + 1]; j++) {
                        shl_dm[ish * nbas + jsh] += fabs(dm[i * nao + j]);
                }
                }
        }
        }
}
}

void SGXmake_shl_op(double *op, double *shl_op, int nao,
                    int nbas, int *ao_loc) {
#pragma omp parallel
{
        int ish, jsh, i, j;
#pragma omp for
        for (ish = 0; ish < nbas; ish++) {
        for (jsh = 0; jsh < nbas; jsh++) {
                shl_op[ish * nbas + jsh] = 0;
                for (i = ao_loc[ish]; i < ao_loc[ish + 1]; i++) {
                for (j = ao_loc[jsh]; j < ao_loc[jsh + 1]; j++) {
                        shl_op[ish * nbas + jsh] = MAX(fabs(op[i * nao + j]),
                                                       shl_op[ish * nbas + jsh]);
                }
                }
        }
        }
}
}

int CVHFshls_block_partition(int *block_loc, int *shls_slice, int *ao_loc,
                             int block_size);

void SGXdot_ao_ao_sparse(double *out, double *bra, double *ket, double *wv,
                         int nao, int ngrids, int nbas, int hermi,
                         uint8_t *bra_mask, uint8_t *ket_mask, int *ao_loc)
{
        size_t Nao = nao;
        size_t Ngrids = ngrids;
        size_t Ngrids_blksize = NI_BLKSIZE;
        int shls_slice[2] = {0, nbas};
        int *box_l1_loc = malloc(sizeof(int) * (nbas+1));
        int nbox_l1 = CVHFshls_block_partition(box_l1_loc, shls_slice, ao_loc, BOXSIZE);
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D1 = 1;
        const double D0 = 0;
        const int I1 = 1;

#pragma omp parallel
{
        int ijb, ib, jb, ib0, jb0, ib1, jb1, ig0, ig1, ig_box2;
        int ish0, ish1, jsh0, jsh1, i0, i1, j0, j1, ni, nj, i, j, n;
        int ng, g, gblk, screened_ni, ish, screened_nj;
        double s;
        double *pout;
        int *aolist_i = malloc(sizeof(int) * BOXSIZE);
        int *aolist_j = malloc(sizeof(int) * BOXSIZE);
        double *buf1 = malloc(sizeof(double) * (Ngrids_blksize * BOXSIZE + ALIGNMENT));
        double *buf2 = malloc(sizeof(double) * (Ngrids_blksize * BOXSIZE + ALIGNMENT));
        double *buf3 = malloc(sizeof(double) * (BOXSIZE * BOXSIZE + ALIGNMENT));
        double *braw = (double *)((uintptr_t)(buf1+ALIGNMENT-1) & (-(uintptr_t)(ALIGNMENT*sizeof(double))));
        double *ketw = (double *)((uintptr_t)(buf2+ALIGNMENT-1) & (-(uintptr_t)(ALIGNMENT*sizeof(double))));
        double *outbuf = (double *)((uintptr_t)(buf3+ALIGNMENT-1) & (-(uintptr_t)(ALIGNMENT*sizeof(double))));
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

                for (i = 0; i < BOXSIZE * BOXSIZE; i++) {
                        outbuf[i] = 0;
                }
                for (ig0 = 0; ig0 < ngrids; ig0+=Ngrids_blksize) {
                        ig1 = MIN(ig0 + Ngrids_blksize, ngrids);
                        ng = ig1 - ig0;
                        gblk = ig0 / Ngrids_blksize;
                        int ip = 0;
                        for(ish = ish0; ish < ish1; ish++) {
                        for (i = ao_loc[ish]; i < ao_loc[ish+1]; i++) {
                        if (bra_mask[gblk * nbas + ish]) {
                                aolist_i[ip] = i;
                                for (g = ig0; g < ig1; g++) {
                                        braw[ip * ng + g - ig0] = bra[i * Ngrids + g] * wv[g];
                                }
                                ip++;
                        } } }
                        screened_ni = ip;
                        ip = 0;
                        for(ish = jsh0; ish < jsh1; ish++) {
                        for (i = ao_loc[ish]; i < ao_loc[ish+1]; i++) {
                        if (ket_mask[gblk * nbas + ish]) {
                                aolist_j[ip] = i;
                                for (g = ig0; g < ig1; g++) {
                                        ketw[ip * ng + g - ig0] = ket[i * Ngrids + g];
                                }
                                ip++;
                        } } }
                        screened_nj = ip;
                        if (screened_ni == 0 || screened_nj == 0) {
                                continue;
                        }
                        dgemm_(&TRANS_T, &TRANS_N, &screened_nj, &screened_ni, &ng,
                               &D1, ketw, &ng, braw, &ng,
                               &D0, outbuf, &screened_nj);
                        for (i = 0; i < screened_ni; i++) {
                        for (j = 0; j < screened_nj; j++) {
                                out[aolist_i[i] * Nao + aolist_j[j]] +=
                                        outbuf[i * screened_nj + j];
                        } }
                }
        }
        free(buf1);
        free(buf2);
        free(buf3);
        free(aolist_i);
        free(aolist_j);
}
        free(box_l1_loc);

        //if (hermi != 0) {
        //        NPdsymm_triu(nao, out, hermi);
        //}
}

static void _dot_ao_dm_l1(double *out, double *ao, double *dm,
                          int nao, size_t ngrids, int nbas, int ig0, int ig1,
                          int ish0, int ish1, int jsh0, int jsh1,
                          uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc)
{
        size_t Nao = nao;
        int ig, ish, jsh, i0, i1, i, j, box_id, n;
        size_t i_addr, j_addr;
        double dm_val;
        double s8[UNROLL_SIZE];
        double *dm_j;

        for (ig = ig0; ig < ig1; ig+=UNROLL_SIZE) {
                box_id = ig / NI_BLKSIZE;
                for (jsh = jsh0; jsh < jsh1; jsh++) {
                        for (j = ao_loc[jsh]; j < ao_loc[jsh+1]; j++) {
                                dm_j = dm + j;
                                j_addr = j * ngrids + ig;
                                for (n = 0; n < UNROLL_SIZE; n++) {
                                        s8[n] = out[j_addr+n];
                                }
                                for (ish = ish0; ish < ish1; ish++) {
                                        if (screen_index[box_id * nbas + ish] &&
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

static void _dot_ao_dm_frac(double *out, double *ao, double *dm,
                            int nao, size_t ngrids, int nbas, int ig0,
                            uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc)
{
        size_t Nao = nao;
        int ngrids_rest = ngrids - ig0;
        int ish, jsh, i0, i1, i, j, n;
        size_t i_addr, j_addr;
        double dm_val;
        double s8[UNROLL_SIZE];
        double *dm_j;

        int box_id = ig0 / NI_BLKSIZE;
        for (jsh = 0; jsh < nbas; jsh++) {
                for (j = ao_loc[jsh]; j < ao_loc[jsh+1]; j++) {
                        dm_j = dm + j;
                        j_addr = j * ngrids + ig0;
                        for (n = 0; n < ngrids_rest; n++) {
                                s8[n] = 0;
                        }
                        for (ish = 0; ish < nbas; ish++) {
                                if (screen_index[box_id * nbas + ish]
                                    && pair_mask[ish*nbas+jsh]) {
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

// return number of non-zero values in mask array
static void mask_l1_abstract(uint8_t *out, uint8_t *mask, int *box_loc,
                             int nbox, int ngrids, int nbas)
{
        int i, m, ig, box_id, ig0, ig1, i0, i1, n;
        int with_value;
        for (n = 0, ig0 = 0; ig0 < ngrids; ig0 += NI_BLKSIZE) {
        for (box_id = 0; box_id < nbox; box_id++, n++) {
                i0 = box_loc[box_id];
                i1 = box_loc[box_id+1];
                ig1 = MIN(ig0+NI_BLKSIZE, ngrids);
                with_value = 0;
                for (i = i0; i < i1; i++) {
                for (ig = ig0; ig < ig1; ig+=NI_BLKSIZE) {
                        m = ig / NI_BLKSIZE;
                        if (mask[m*nbas+i] != 0) {
                                with_value = 1;
                                goto next_l1_box;
                        }
                } }
next_l1_box:
                out[n] = with_value;
        } }
}


void SGXdot_ao_dm_sparse(double *out, double *ao, double *dm,
                         int nao, int ngrids, int nbas,
                         uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc)
{
        size_t Ngrids = ngrids;
        int shls_slice[2] = {0, nbas};
        int *box_l1_loc = malloc(sizeof(int) * (nbas+1));
        int nbox_l1 = CVHFshls_block_partition(box_l1_loc, shls_slice, ao_loc, BOXSIZE);
        int mask_l1_size = (ngrids + NI_BLKSIZE - 1) / NI_BLKSIZE * nbox_l1;
        uint8_t *mask_l1 = malloc(sizeof(uint8_t) * mask_l1_size);
        mask_l1_abstract(mask_l1, screen_index, box_l1_loc, nbox_l1, ngrids, nbas);
        int ngrids_align_down = (ngrids / NI_BLKSIZE) * NI_BLKSIZE;

        if (nao * 2 < ngrids) {
#pragma omp parallel
{
                int ig, j, j0, j1, jsh0, jsh1, ib, jb, ig0, ig1, ig_box2;
#pragma omp for schedule(dynamic)
                for (ig0 = 0; ig0 < ngrids_align_down; ig0+=NI_BLKSIZE) {
                        ig1 = MIN(ig0 + NI_BLKSIZE, ngrids_align_down);
                        ig_box2 = ig0 / NI_BLKSIZE;
                        for (jb = 0; jb < nbox_l1; jb++) {
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
              screen_index, pair_mask, ao_loc);
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
                        for (ig0 = 0; ig0 < ngrids_align_down; ig0+=NI_BLKSIZE) {
                                ig1 = MIN(ig0 + NI_BLKSIZE, ngrids_align_down);
                                ig_box2 = ig0 / NI_BLKSIZE;
                                for (j = j0; j < j1; j++) {
                                for (ig = ig0; ig < ig1; ig++) {
                                        out[j*Ngrids+ig] = 0;
                                } }
                                for (ib = 0; ib < nbox_l1; ib++) {
                                        if (mask_l1[ig_box2 * nbox_l1 + ib]) {
_dot_ao_dm_l1(out, ao, dm, nao, ngrids, nbas, ig0, ig1,
              box_l1_loc[ib], box_l1_loc[ib+1], jsh0, jsh1,
              screen_index, pair_mask, ao_loc);
                                        }
                                }
                        }
                }
}
        }
        if (ngrids_align_down < ngrids) {
                _dot_ao_dm_frac(out, ao, dm, nao, ngrids, nbas, ngrids_align_down,
                                screen_index, pair_mask, ao_loc);
        }
        free(box_l1_loc);
        free(mask_l1);
}

int SGXreturn_blksize() { return SGX_BLKSIZE; }

void SGXmake_screen_norm(double *norms, double *aao, int ngrids, int nbas, int *ao_loc)
{
#pragma omp parallel
{
        int ish, iblk, g, i, g0, g1;
        double t1, t2;
        int bblk = (ngrids + SGX_BLKSIZE - 1) / SGX_BLKSIZE;
        const size_t Ngrids = ngrids;
#pragma omp for
        for (iblk = 0; iblk < bblk; iblk++) {
                g0 = iblk * SGX_BLKSIZE;
                g1 = MIN(g0 + SGX_BLKSIZE, ngrids);
                for (ish = 0; ish < nbas; ish++) {
                        t1 = 0;
                        for (i = ao_loc[ish]; i < ao_loc[ish + 1]; i++) {
                                t2 = 0;
                                for (g = g0; g < g1; g++) {
                                        aao[i*Ngrids+g] = fabs(aao[i*Ngrids+g]);
                                        t2 += aao[i*Ngrids+g] * aao[i*Ngrids+g];
                                }
                                t1 = (MAX(t1, t2) * SGX_BLKSIZE) / (g1 - g0);
                        }
                        norms[iblk * nbas + ish] = sqrt(t1);
                }
        }
}
}

void SGXmake_screen_q_cond(double *norms, double *ao, int ngrids, int nbas, int *ao_loc) {
#pragma omp parallel
{
        int ish, iblk, g, i, g0, g1;
        double t1, t2;
        int bblk = (ngrids + SGX_BLKSIZE - 1) / SGX_BLKSIZE;
        const size_t Ngrids = ngrids;
#pragma omp for
        for (iblk = 0; iblk < bblk; iblk++) {
                g0 = iblk * SGX_BLKSIZE;
                g1 = MIN(g0 + SGX_BLKSIZE, ngrids);
                for (ish = 0; ish < nbas; ish++) {
                        t1 = 0;
                        for (i = ao_loc[ish]; i < ao_loc[ish + 1]; i++) {
                                t2 = 0;
                                for (g = g0; g < g1; g++) {
                                        t2 = MAX(t2, fabs(ao[i*Ngrids+g]));
                                }
                                t1 = MAX(t1, t2);
                        }
                        norms[iblk * nbas + ish] = t1;
                }
        }
}
}

#define DECLARE_ALL \
        int *atm = envs->atm; \
        int *bas = envs->bas; \
        double *env = envs->env; \
        int natm = envs->natm; \
        int nbas = envs->nbas; \
        int *ao_loc = envs->ao_loc; \
        int *shls_slice = envs->shls_slice; \
        CINTOpt *cintopt = envs->cintopt; \
        int ioff = ao_loc[shls_slice[0]]; \
        int joff = ao_loc[shls_slice[2]]; \
        int i0, j0, i1, j1, ish, jsh, idm; \
        ish = shls[0]; \
        jsh = shls[1];

int SGXnr_pj_prescreen(int *shls, CVHFOpt *opt,
                       int *atm, int *bas, double *env)
{
        if (opt == NULL) {
                return 1;
        }
        int i = shls[0];
        int j = shls[1];
        int k = shls[2];
        int n = opt->nbas;
        int nk = opt->ngrids;
        assert(opt->q_cond);
        assert(opt->dm_cond);
        assert(i < n);
        assert(j < n);
        assert(k < nk);
        return opt->q_cond[i*n+j]
               * MAX(fabs(opt->dm_cond[j*nk+k]), fabs(opt->dm_cond[i*nk+k]))
               > opt->direct_scf_cutoff;
}

int SGXnr_pj_screen_otf(double **dms, CVHFOpt *opt, SGXJKArray **vjk,
                        int *shls, int i0, int i1, int j0, int j1,
                        int grid_offset, int blksize, int n_dm)
{
        if (opt == NULL) {
                return 1;
        }
        int ngrids;
        int idm, i, j, k, icomp;
        double max_i = 0;
        double max_j = 0;
        double max_dm = 0;
        double *dm;

        for (idm = 0; idm < n_dm; idm++) {
                dm = dms[idm] + grid_offset;
                ngrids = vjk[idm]->v_dims[2];
                for (icomp = 0; icomp < vjk[idm]->ncomp; icomp++) {
                        for (i = i0; i < i1; i++) {
                        for (k = 0; k < blksize; k++) {
                                max_i = MAX(fabs(dm[i*ngrids+k]), max_i);
                        } }
                        for (i = j0; i < j1; i++) {
                        for (k = 0; k < blksize; k++) {
                                max_j = MAX(fabs(dm[i*ngrids+k]), max_j);
                        } }
                }
        }
        max_dm = MAX(max_i, max_j);

        i = shls[0];
        j = shls[1];
        return opt->q_cond[i * opt->nbas + j] * max_i * max_j > opt->direct_scf_cutoff;
}

static int SGXnr_pj_screen_v2(int *shls, CVHFOpt *opt,
                              int *atm, int *bas, double *env,
                              double cutoff, double *shl_maxs)
{
        if (opt == NULL) {
                return 1;
        }
        int i = shls[0];
        int j = shls[1];
        int n = opt->nbas;
        assert(opt->q_cond);
        assert(i < n);
        assert(j < n);
        return opt->q_cond[i*n+j] * MAX(shl_maxs[i], shl_maxs[j]) > cutoff;
}

void SGXdot_nrk_no_pscreen(int (*intor)(), SGXJKOperator **jkop, SGXJKArray **vjk,
                double **dms, double *buf, double *cache, int n_dm, int* shls,
                CVHFOpt *vhfopt, IntorEnvs *envs,
                double* all_grids, int ngrids)
{
        DECLARE_ALL;

        i0 = ao_loc[ish  ] - ioff;
        j0 = ao_loc[jsh  ] - joff;
        i1 = ao_loc[ish+1] - ioff;
        j1 = ao_loc[jsh+1] - joff;

        int grid0, grid1;
        int dims[] = {ao_loc[ish+1]-ao_loc[ish], ao_loc[jsh+1]-ao_loc[jsh], ngrids};
        for (grid0 = 0; grid0 < ngrids; grid0 += SGX_BLKSIZE) {
                grid1 = MIN(grid0 + SGX_BLKSIZE, ngrids);
                shls[2] = grid0;
                shls[3] = grid1;
                (*intor)(buf+grid0, dims, shls, atm, natm, bas, nbas, env, cintopt, cache);
        }
        for (idm = 0; idm < n_dm; idm++) {
                jkop[idm]->contract(buf, dms[idm], vjk[idm],
                                    i0, i1, j0, j1, 0, ngrids);
        }
}

void SGXdot_nrk_pscreen(int (*intor)(), SGXJKOperator **jkop, SGXJKArray **vjk,
                double **dms, double *buf, double *cache, int n_dm, int* shls,
                CVHFOpt *vhfopt, IntorEnvs *envs,
                double* all_grids, int ngrids)
{
        if (vhfopt == NULL || vhfopt->dm_cond == NULL) {
                SGXdot_nrk_no_pscreen(intor, jkop, vjk, dms, buf, cache, n_dm, shls,
                                      vhfopt, envs, all_grids, ngrids);
                return;
        }

        DECLARE_ALL;

        i0 = ao_loc[ish  ] - ioff;
        j0 = ao_loc[jsh  ] - joff;
        i1 = ao_loc[ish+1] - ioff;
        j1 = ao_loc[jsh+1] - joff;

        int grid0, grid1;
        int dims[] = {ao_loc[ish+1]-ao_loc[ish], ao_loc[jsh+1]-ao_loc[jsh], 0};
        for (grid0 = 0; grid0 < ngrids; grid0 += SGX_BLKSIZE) {
                grid1 = MIN(grid0 + SGX_BLKSIZE, ngrids);
                shls[2] = grid0;
                shls[3] = grid1;
                dims[2] = grid1 - grid0;
                if (SGXnr_pj_screen_otf(
                        dms, vhfopt, vjk, shls, i0, i1, j0, j1,
                        grid0, grid1 - grid0, n_dm
                )) {
                        (*intor)(buf, dims, shls, atm, natm, bas, nbas, env, cintopt, cache);
                        for (idm = 0; idm < n_dm; idm++) {
                                jkop[idm]->contract(buf, dms[idm], vjk[idm],
                                                    i0, i1, j0, j1, grid0, grid1 - grid0);
                        }
                }
        }
}

static inline void _get_shell_norms(double **dms, double *shell_norms,
                                    double *weights, int *ao_loc, int n_dm,
                                    int ig0, int ig1, int nbas, size_t Ngrids) {
        double tot, val;
        int ish, i0, i1, idm, i, k;
        for (ish = 0; ish < nbas; ish++) {
                tot = 0;
                i0 = ao_loc[ish];
                i1 = ao_loc[ish + 1];
                for (idm = 0; idm < n_dm; idm++) {
                for (i = i0; i < i1; i++) {
#pragma omp simd reduction(+:tot)
                for (k = ig0; k < ig1; k++) {
                        val = dms[idm][i*Ngrids+k];
                        tot += val * val * weights[k];
                } } }
                shell_norms[ish] = sqrt(tot);
        }
}

void SGXget_shell_norms(double **dms, double *shell_norms, int n_dm,
                        int *ao_loc, int nbas, int ngrids, double *weights) {
        const int nblk = (ngrids + NI_BLKSIZE - 1) / NI_BLKSIZE;
        const size_t Ngrids = ngrids;
        const size_t Nbas = nbas;
#pragma omp parallel
{
        int ib;
        int ish, idm, i, k, i0, i1, ig0, ig1;
        double tot, val;
#pragma omp for schedule(static)
        for (ib = 0; ib < nblk; ib++) {
                ig0 = ib * NI_BLKSIZE;
                ig1 = MIN(ig0 + NI_BLKSIZE, ngrids);
                /*for (ish = 0; ish < nbas; ish++) {
                        tot = 0;
                        i0 = ao_loc[ish];
                        i1 = ao_loc[ish + 1];
                        for (idm = 0; idm < n_dm; idm++) {
                        for (i = i0; i < i1; i++) {
#pragma omp simd reduction(+:tot)
                        for (k = ig0; k < ig1; k++) {
                                val = dms[idm][i*Ngrids+k];
                                tot += val * val * weights[k];
                        } } }
                        shell_norms[ib * nbas + ish] = sqrt(tot);
                }*/
               _get_shell_norms(dms, shell_norms + ib * Nbas, weights,
                                ao_loc, n_dm, ig0, ig1, nbas, Ngrids);
        }
#pragma omp for schedule(static)
        for (ish = 0; ish < nbas; ish++) {
                tot = 0;
                for (ib = 0; ib < nblk; ib++) {
                        tot += pow(shell_norms[ib * nbas + ish], 0.1);
                }
                for (ib = 0; ib < nblk; ib++) {
                        shell_norms[ib * nbas + ish] =
                                pow(shell_norms[ib * nbas + ish], 0.9) * tot;
                }
        }
}
}

void SGXnr_direct_drv(int (*intor)(), SGXJKOperator **jkop,
                      double **dms, double **vjk, int n_dm, int ncomp,
                      int *shls_slice, int *ao_loc,
                      CINTOpt *cintopt, CVHFOpt *vhfopt,
                      int *atm, int natm, int *bas, int nbas, double *env,
                      int env_size, int aosym, double *ncond,
                      double *weights)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        int nish = ish1 - ish0;
        int di = GTOmax_shell_dim(ao_loc, shls_slice, 2);
        int cache_size = _max_cache_size_sgx(intor, shls_slice, 2,
                                             atm, natm, bas, nbas, env);
        int npair;
        if (aosym == 2) {
            npair = nish * (nish+1) / 2;
        } else {
            npair = nish * nish;
        }

        const int ioff = ao_loc[ish0];
        const int joff = ao_loc[jsh0];

        int (*fprescreen)();
        if (vhfopt != NULL) {
                fprescreen = vhfopt->fprescreen;
        } else {
                fprescreen = CVHFnoscreen;
        }
        const int tot_ngrids = (int) env[NGRIDS];
        const size_t Tot_Ngrids = tot_ngrids;
        const double* all_grids = env+(size_t)env[PTR_GRIDS];
        const int nbatch = (tot_ngrids + SGX_BLKSIZE - 1) / SGX_BLKSIZE;
        double usc = 0;
#pragma omp parallel
{
        int ig0, ig1, dg;
        int ish, jsh, ij, jmax;
        int i0, i1, j0, j1, idm, i, k;
        int shls[4];
        IntorEnvs envs = {natm, nbas, atm, bas, env, shls_slice, ao_loc, NULL,
                          cintopt, ncomp};
        SGXJKArray *v_priv[n_dm];
        for (idm = 0; idm < n_dm; idm++) {
                v_priv[idm] = jkop[idm]->allocate(shls_slice, ao_loc, ncomp, tot_ngrids);
        }
        double *buf = calloc(sizeof(double), SGX_BLKSIZE*di*di*ncomp);
        double *cache = malloc(sizeof(double) * cache_size);
        double *shl_maxs;
        double shl_max_sum;
        double shl_sum;
        if (ncond == NULL) {
                shl_maxs = NULL;
        } else {
                shl_maxs = malloc(sizeof(double) * nbas);
        }
        int *sj_shells = malloc(sizeof(int) * nish);
        int num_sj_shells;
        int sj_index;
        double _usc = 0;
        double tmp;
#pragma omp for nowait schedule(dynamic, 1)
        for (int ibatch = 0; ibatch < nbatch; ibatch++) {
                ig0 = ibatch * SGX_BLKSIZE;
                ig1 = MIN(ig0 + SGX_BLKSIZE, tot_ngrids);
                dg = ig1 - ig0;
                for (idm = 0; idm < n_dm; idm++) {
                        jkop[idm]->set0(v_priv[idm], dg);
                }
                if (shl_maxs != NULL) {
                        /*shl_max_sum = 0;
                        for (ish = 0; ish < nbas; ish++) {
                                shl_maxs[ish] = 0;
                                i0 = ao_loc[ish] - ioff;
                                i1 = ao_loc[ish + 1] - ioff;
                                for (idm = 0; idm < n_dm; idm++) {
                                for (i = i0; i < i1; i++) {
                                        shl_sum = 0;
                                        for (k = ig0; k < ig1; k++) {
                                                shl_maxs[ish] = MAX(
                                                        fabs(dms[idm][i*Tot_Ngrids+k]),
                                                        shl_maxs[ish]
                                                );
                                        }
                                        shl_maxs[ish] = MAX(shl_maxs[ish], shl_sum);
                                } }
                                shl_max_sum += (i1 - i0) * pow(shl_maxs[ish], 0.1);
                        }
                        for (ish = 0; ish < nbas; ish++) {
                                shl_maxs[ish] = pow(shl_maxs[ish], 0.9) * shl_max_sum;
                        }
                        */
                        _get_shell_norms(dms, shl_maxs, weights, ao_loc, n_dm,
                                         ig0, ig1, nbas, Tot_Ngrids);
                        shl_max_sum = 0;
                        for (ish = 0; ish < nbas; ish++) {
                                shl_max_sum += pow(shl_maxs[ish], 0.1);
                        }
                        for (ish = 0; ish < nbas; ish++) {
                                shl_maxs[ish] = pow(shl_maxs[ish], 0.9) * shl_max_sum;
                        }
                }
                for (ish = 0; ish < nbas; ish++) {
                        if (aosym == 2) {
                                jmax = ish + 1;
                        } else {
                                jmax = nish;
                        }
                        num_sj_shells = 0;
                        for (jsh = 0; jsh < jmax; jsh++) {
                                shls[0] = ish + ish0;
                                shls[1] = jsh + jsh0;
                                if ((*fprescreen)(shls, vhfopt, atm, bas, env)) {
                                if (ncond == NULL || SGXnr_pj_screen_v2(
                                        shls, vhfopt, atm, bas, env, ncond[ibatch], shl_maxs
                                )) {
                                        sj_shells[num_sj_shells] = jsh;
                                        num_sj_shells++;
                                        _usc += 1;
                                } }
                                /*if ((*fprescreen)(shls, vhfopt, atm, bas, env)) {
                                        sj_shells[num_sj_shells] = jsh;
                                        num_sj_shells++;
                                        _usc++;
                                }*/
                        }
                        for (sj_index = 0; sj_index < num_sj_shells; sj_index++) {
                                jsh = sj_shells[sj_index];
                                shls[0] = ish + ish0;
                                shls[1] = jsh + jsh0;
                                i0 = ao_loc[ish  ] - ioff;
                                j0 = ao_loc[jsh  ] - joff;
                                i1 = ao_loc[ish+1] - ioff;
                                j1 = ao_loc[jsh+1] - joff;
                                const int dims[] = {ao_loc[ish+1]-ao_loc[ish],
                                                    ao_loc[jsh+1]-ao_loc[jsh], dg};
                                shls[2] = ig0;
                                shls[3] = ig1;
                                (*intor)(buf, dims, shls, atm, natm, bas, nbas,
                                         env, cintopt, cache);
                                for (idm = 0; idm < n_dm; idm++) {
                                        jkop[idm]->contract(buf, dms[idm], v_priv[idm],
                                                            i0, i1, j0, j1, ig0, dg);
                                }
                        }
                }
                for (idm = 0; idm < n_dm; idm++) {
                        jkop[idm]->send(v_priv[idm], ig0, dg, vjk[idm]);
                }
        }
#pragma omp critical
{
        for (idm = 0; idm < n_dm; idm++) {
                jkop[idm]->finalize(v_priv[idm], vjk[idm]);
        }
}
        free(buf);
        free(cache);
        free(sj_shells);
        if (shl_maxs != NULL) {
                free(shl_maxs);
        }
#pragma omp critical
{
        usc += _usc;
}
}
        printf("unscreened: %lf\n", usc);
}

void SGXnr_q_cond(int (*intor)(), CINTOpt *cintopt, double *q_cond,
                  int *ao_loc, int *atm, int natm,
                  int *bas, int nbas, double *env)
{
        int shls_slice[] = {0, nbas};
        int cache_size = GTOmax_cache_size(intor, shls_slice, 1,
                                           atm, natm, bas, nbas, env);
#pragma omp parallel default(none) \
        shared(intor, q_cond, ao_loc, atm, natm, bas, nbas, env, cache_size)
{
        double qtmp, tmp;
        int ij, i, j, di, dj, ish, jsh;
        int shls[2];
        di = 0;
        for (ish = 0; ish < nbas; ish++) {
                dj = ao_loc[ish+1] - ao_loc[ish];
                di = MAX(di, dj);
        }
        double *cache = malloc(sizeof(double) * (di*di + cache_size));
        double *buf = cache + cache_size;
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nbas*(nbas+1)/2; ij++) {
                ish = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                jsh = ij - ish*(ish+1)/2;
                if (bas(ATOM_OF,ish) == bas(ATOM_OF,jsh)) {
                        // If two shells are on the same center, their
                        // overlap integrals may be zero due to symmetry.
                        // But their contributions to sgX integrals should
                        // be recognized.
                        q_cond[ish*nbas+jsh] = 1;
                        q_cond[jsh*nbas+ish] = 1;
                        continue;
                }

                shls[0] = ish;
                shls[1] = jsh;
                qtmp = 1e-100;
                if (0 != (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env,
                                  NULL, cache)) {
                        di = ao_loc[ish+1] - ao_loc[ish];
                        dj = ao_loc[jsh+1] - ao_loc[jsh];
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                tmp = fabs(buf[i+di*j]);
                                qtmp = MAX(qtmp, tmp);
                        } }
                }
                q_cond[ish*nbas+jsh] = qtmp;
                q_cond[jsh*nbas+ish] = qtmp;
        }
        free(cache);
}
}

void SGXmake_rinv_ubound(double *q_cond, double *ovlp_abs, double *bas_max,
                         int *ao_loc, int nbas, int nao, int nblk) {
        int shls_slice[] = {0, nbas};
        // 2pi(3/4pi)^(2/3)
        double fac = 2.4179879310247046;
#pragma omp parallel
{
        int ij, i, j, ish, jsh;
        double maxnorm, maxprod;
        double tmp;
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nbas*(nbas+1)/2; ij++) {
                ish = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                jsh = ij - ish*(ish+1)/2;
                maxprod = 0;
                for (int iblk = 0; iblk < nblk; iblk++) {
                        tmp = bas_max[iblk * nbas + ish] * bas_max[iblk * nbas + jsh];
                        maxprod = MAX(tmp, maxprod);
                }
                maxnorm = 0;
                for (i = ao_loc[ish]; i < ao_loc[ish + 1]; i++) {
                for (j = ao_loc[jsh]; j < ao_loc[jsh + 1]; j++) {
                        maxnorm = MAX(maxnorm, ovlp_abs[i * nao + j]);
                } }
                maxnorm = fac * pow(maxnorm, 2.0 / 3) * pow(maxprod, 1.0 / 3);
                q_cond[ish*nbas+jsh] = maxnorm;
                q_cond[jsh*nbas+ish] = maxnorm;
        }
}
}

void SGXsetnr_direct_scf(CVHFOpt *opt, int (*intor)(), CINTOpt *cintopt,
                         int *ao_loc, int *atm, int natm,
                         int *bas, int nbas, double *env)
{
        if (opt->q_cond != NULL) {
                free(opt->q_cond);
        }
        nbas = opt->nbas;
        double *q_cond = (double *)malloc(sizeof(double) * nbas*nbas);
        opt->q_cond = q_cond;
        SGXnr_q_cond(intor, cintopt, q_cond, ao_loc, atm, natm, bas, nbas, env);
}

void SGXnr_dm_cond(double *dm_cond, double *dm, int nset, int *ao_loc,
                   int *atm, int natm, int *bas, int nbas, double *env,
                   int ngrids)
{
        size_t nao = ao_loc[nbas] - ao_loc[0];
        double dmax;
        size_t i, j, jsh, iset;
        double *pdm;
        for (i = 0; i < ngrids; i++) {
        for (jsh = 0; jsh < nbas; jsh++) {
                dmax = 0;
                for (iset = 0; iset < nset; iset++) {
                        pdm = dm + nao*ngrids*iset;
                        for (j = ao_loc[jsh]; j < ao_loc[jsh+1]; j++) {
                                dmax = MAX(dmax, fabs(pdm[i*nao+j]));
                        }
                }
                dm_cond[jsh*ngrids+i] = dmax;
        } }
}

void SGXsetnr_direct_scf_dm(CVHFOpt *opt, double *dm, int nset, int *ao_loc,
                            int *atm, int natm, int *bas, int nbas, double *env,
                            int ngrids)
{
        nbas = opt->nbas;
        if (opt->dm_cond != NULL) {
                free(opt->dm_cond);
        }
        opt->dm_cond = (double *)malloc(sizeof(double) * nbas*ngrids);
        if (opt->dm_cond == NULL) {
                fprintf(stderr, "malloc(%zu) failed in SGXsetnr_direct_scf_dm\n",
                        sizeof(double) * nbas*ngrids);
                exit(1);
        }
        // nbas in the input arguments may different to opt->nbas.
        // Use opt->nbas because it is used in the prescreen function
        memset(opt->dm_cond, 0, sizeof(double)*nbas*ngrids);
        opt->ngrids = ngrids;

        SGXnr_dm_cond(opt->dm_cond, dm, nset, ao_loc,
                      atm, natm, bas, nbas, env, ngrids);
}

int SGXnr_ovlp_prescreen(int *shls, CVHFOpt *opt,
                         int *atm, int *bas, double *env)
{
        if (opt == NULL) {
                return 1;
        }
        int i = shls[0];
        int j = shls[1];
        int n = opt->nbas;
        assert(opt->q_cond);
        assert(i < n);
        assert(j < n);
        return opt->q_cond[i*n+j] > opt->direct_scf_cutoff;
}


#define JTYPE1  1
#define JTYPE2  2
#define KTYPE1  3

#define ALLOCATE(label, task) \
static SGXJKArray *SGXJKOperator_allocate_##label(int *shls_slice, int *ao_loc, \
                                                  int ncomp, int ngrids) \
{ \
        SGXJKArray *jkarray = malloc(sizeof(SGXJKArray)); \
        jkarray->v_dims[0]  = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]; \
        jkarray->v_dims[1]  = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]; \
        jkarray->v_dims[2]  = ngrids; \
        if (task == JTYPE1) { \
                jkarray->data = malloc(ncomp * SGX_BLKSIZE * sizeof(double)); \
        } else if (task == JTYPE2) { \
                jkarray->data = calloc(ncomp * jkarray->v_dims[0] \
                                       * jkarray->v_dims[1], sizeof(double)); \
        } else { \
                jkarray->data = malloc(ncomp * jkarray->v_dims[0] \
                                       * SGX_BLKSIZE * sizeof(double)); \
        } \
        jkarray->ncomp = ncomp; \
        return jkarray; \
} \
static void SGXJKOperator_set0_##label(SGXJKArray *jkarray, int dk) \
{ \
        int ncomp = jkarray->ncomp; \
        int i, k; \
        double *data = jkarray->data; \
        if (task == JTYPE1) { \
                for (i = 0; i < ncomp; i++) { \
                for (k = 0; k < dk; k++) { \
                        data[i*dk+k] = 0; \
                } } \
        } else if (task == KTYPE1) { \
                for (i = 0; i < ncomp * jkarray->v_dims[0]; i++) { \
                for (k = 0; k < dk; k++) { \
                        data[i*dk+k] = 0; \
                } } \
        } \
} \
static void SGXJKOperator_send_##label(SGXJKArray *jkarray, int k0, int dk, double *out) \
{ \
        int ncomp = jkarray->ncomp; \
        size_t i, k, icomp; \
        double *data = jkarray->data; \
        const size_t ni = jkarray->v_dims[0]; \
        const size_t nk_global = jkarray->v_dims[2]; \
        out = out + k0; \
        if (task == JTYPE1) { \
                for (i = 0; i < ncomp; i++) { \
                for (k = 0; k < dk; k++) { \
                        out[i*nk_global+k] = data[i*dk+k]; \
                } } \
        } else if (task == KTYPE1) { \
                for (icomp = 0; icomp < ncomp; icomp++) { \
                        for (i = 0; i < ni; i++) { \
                        for (k = 0; k < dk; k++) { \
                                out[i*nk_global+k] = data[i*dk+k]; \
                        } } \
                        out += nk_global * ni; \
                        data += dk * ni; \
                } \
        } \
} \
static void SGXJKOperator_final_##label(SGXJKArray *jkarray, double *out) \
{ \
        int i; \
        double *data = jkarray->data; \
        if (task == JTYPE2) { \
                for (i = 0; i < jkarray->ncomp * jkarray->v_dims[0] * jkarray->v_dims[1]; i++) { \
                        out[i] += data[i]; \
                } \
        } \
        SGXJKOperator_deallocate(jkarray); \
}

#define ADD_OP(fname, task, type) \
        ALLOCATE(fname, task) \
SGXJKOperator SGX##fname = {SGXJKOperator_allocate_##fname, fname, \
        SGXJKOperator_set0_##fname, SGXJKOperator_send_##fname, \
        SGXJKOperator_final_##fname, \
        SGXJKOperator_sanity_check_##type}

static void SGXJKOperator_deallocate(SGXJKArray *jkarray)
{
        free(jkarray->data);
        free(jkarray);
}

static void SGXJKOperator_sanity_check_s1(int *shls_slice)
{
}
static void SGXJKOperator_sanity_check_s2(int *shls_slice)
{
        if (!((shls_slice[0] == shls_slice[2]) &&
              (shls_slice[1] == shls_slice[3]))) {
                fprintf(stderr, "Fail at s2\n");
                exit(1);
        };
}

static void nrs1_ijg_ji_g(double *eri, double *dm, SGXJKArray *out,
                          int i0, int i1, int j0, int j1,
                          const int k0, const int dk)
{
        const size_t ncol = out->v_dims[0];
        int i, j, k, icomp;
        double *data = out->data;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                for (k = 0; k < dk; k++) {
                        data[k] += eri[ij*dk+k] * dm[j*ncol+i];
                } } }
                data += dk;
        }
}
ADD_OP(nrs1_ijg_ji_g, JTYPE1, s1);

static void nrs2_ijg_ji_g(double *eri, double *dm, SGXJKArray *out,
                          int i0, int i1, int j0, int j1,
                          const int k0, const int dk)
{
        if (i0 == j0) {
                return nrs1_ijg_ji_g(eri, dm, out, i0, i1, j0, j1, k0, dk);
        }

        const size_t ncol = out->v_dims[0];
        int i, j, k, icomp;
        double *data = out->data;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                for (k = 0; k < dk; k++) {
                        data[k] += eri[ij*dk+k] * (dm[j*ncol+i] + dm[i*ncol+j]);
                } } }
                data += dk;
        }
}
ADD_OP(nrs2_ijg_ji_g, JTYPE1, s2);

static void nrs1_ijg_g_ij(double *eri, double *dm, SGXJKArray *out,
                          int i0, int i1, int j0, int j1,
                          const int k0, const int dk)
{
        const size_t ni = out->v_dims[0];
        const size_t nj = out->v_dims[1];
        int i, j, k, icomp;
        double *data = out->data;
        dm = dm + k0;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                for (k = 0; k < dk; k++) {
                        data[i*nj+j] += eri[ij*dk+k] * dm[k];
                } } }
                data += ni * nj;
        }
}
ADD_OP(nrs1_ijg_g_ij, JTYPE2, s1);

SGXJKOperator SGXnrs2_ijg_g_ij = {SGXJKOperator_allocate_nrs1_ijg_g_ij,
        nrs1_ijg_g_ij, SGXJKOperator_set0_nrs1_ijg_g_ij,
        SGXJKOperator_send_nrs1_ijg_g_ij, SGXJKOperator_final_nrs1_ijg_g_ij,
        SGXJKOperator_sanity_check_s2};

static void nrs1_ijg_gj_gi(double *eri, double *dm, SGXJKArray *out,
                           int i0, int i1, int j0, int j1,
                           const int k0, const int dk)
{
        double *data = out->data;
        dm = dm + k0;
        int i, j, k, icomp;
        const size_t nk_global = out->v_dims[2];

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                for (k = 0; k < dk; k++) {
                        data[i*dk+k] += eri[ij*dk+k] * dm[j*nk_global+k];
                } } }
                data += out->v_dims[0] * dk;
        }
}
ADD_OP(nrs1_ijg_gj_gi, KTYPE1, s1);

static void nrs2_ijg_gj_gi(double *eri, double *dm, SGXJKArray *out,
                           int i0, int i1, int j0, int j1,
                           const int k0, const int dk)
{
        if (i0 == j0) {
                return nrs1_ijg_gj_gi(eri, dm, out, i0, i1, j0, j1, k0, dk);
        }

        double *data = out->data;
        dm = dm + k0;
        const size_t nk_global = out->v_dims[2];
        int i, j, k, icomp;

        int ij = 0;
        for (icomp = 0; icomp < out->ncomp; icomp++) {
                for (j = j0; j < j1; j++) {
                for (i = i0; i < i1; i++, ij++) {
                for (k = 0; k < dk; k++) {
                        data[i*dk+k] += eri[ij*dk+k] * dm[j*nk_global+k];
                }
                for (k = 0; k < dk; k++) {
                        data[j*dk+k] += eri[ij*dk+k] * dm[i*nk_global+k];
                }
                } }
                data += out->v_dims[0] * dk;
        }
}
ADD_OP(nrs2_ijg_gj_gi, KTYPE1, s2);
