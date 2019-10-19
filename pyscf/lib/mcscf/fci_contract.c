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
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"
#include "fci.h"
// for (16e,16o) ~ 11 MB buffer = 120 * 12870 * 8
#define STRB_BLKSIZE    112

/*
 * CPU timing of single thread can be estimated:
 *      na*nb*nnorb*8(bytes)*5 / (mem_freq*64 (*2 if dual-channel mem))
 *      + na*nb*nnorb**2 (*2 for spin1, *1 for spin0)
 *        / (CPU_freq (*4 for SSE3 blas, or *6-8 for AVX blas))
 * where the 5 times memory accesses are 3 in prog_a_t1, prog0_b_t1,
 * spread_b_t1 and 2 in spread_a_t1
 *
 * multi threads
 *      na*nb*nnorb*8(bytes)*2 / (mem_freq*64 (*2 if dual-channel mem)) due to single thread
 *      + na*nb*nnorb*8(bytes)*3 / max_mem_bandwidth                    due to N-thread
 *      + na*nb*nnorb**2 (*2 for spin1, *1 for spin0)
 *        / (CPU_freq (*4 for SSE3 blas, or *6-8 for AVX blas)) / num_threads
 */

/*
 ***********************************************************
 *
 * Need the permutation symmetry 
 * h2e[i,j,k,l] = h2e[j,i,k,l] = h2e[i,j,l,k] = h2e[j,i,l,k]
 *
 ***********************************************************
 */

/*
 * optimize for OpenMP, to reduce memory/CPU data transfer
 * add software prefetch, it's especially important for OpenMP
 */

/* 
 * For given stra_id, spread alpah-strings (which can propagate to stra_id)
 * into t1[:nstrb,nnorb] 
 *    str1-of-alpha -> create/annihilate -> str0-of-alpha
 * ci0[:nstra,:nstrb] is contiguous in beta-strings
 * bcount control the number of beta strings to be calculated.
 * for spin=0 system, only lower triangle of the intermediate ci vector
 * needs to be calculated
 */
void FCIprog_a_t1(double *ci0, double *t1,
                  int bcount, int stra_id, int strb_id,
                  int norb, int nstrb, int nlinka, _LinkTrilT *clink_indexa)
{
        ci0 += strb_id;
        int j, k, ia, sign;
        size_t str1;
        const _LinkTrilT *tab = clink_indexa + stra_id * nlinka;
        double *pt1, *pci;

        for (j = 0; j < nlinka; j++) {
                ia   = EXTRACT_IA  (tab[j]);
                str1 = EXTRACT_ADDR(tab[j]);
                sign = EXTRACT_SIGN(tab[j]);
                pt1 = t1 + ia*bcount;
                pci = ci0 + str1*nstrb;
                if (sign == 0) {
                        break;
                } else if (sign > 0) {
                        for (k = 0; k < bcount; k++) {
                                pt1[k] += pci[k];
                        }
                } else if (sign < 0) {
                        for (k = 0; k < bcount; k++) {
                                pt1[k] -= pci[k];
                        }
                }
        }
}
/* 
 * For given stra_id, spread all beta-strings into t1[:nstrb,nnorb] 
 *    all str0-of-beta -> create/annihilate -> str1-of-beta
 * ci0[:nstra,:nstrb] is contiguous in beta-strings
 * bcount control the number of beta strings to be calculated.
 * for spin=0 system, only lower triangle of the intermediate ci vector
 * needs to be calculated
 */
void FCIprog_b_t1(double *ci0, double *t1,
                  int bcount, int stra_id, int strb_id,
                  int norb, int nstrb, int nlinkb, _LinkTrilT *clink_indexb)
{
        int j, ia, str0, str1, sign;
        const _LinkTrilT *tab = clink_indexb + strb_id * nlinkb;
        double *pci = ci0 + stra_id*(size_t)nstrb;

        for (str0 = 0; str0 < bcount; str0++) {
                for (j = 0; j < nlinkb; j++) {
                        ia   = EXTRACT_IA  (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        if (sign == 0) {
                                break;
                        } else {
                                t1[ia*bcount+str0] += sign * pci[str1];
                        }
                }
                tab += nlinkb;
        }
}


/*
 * spread t1 into ci1
 */
void FCIspread_a_t1(double *ci1, double *t1,
                    int bcount, int stra_id, int strb_id,
                    int norb, int nstrb, int nlinka, _LinkTrilT *clink_indexa)
{
        ci1 += strb_id;
        int j, k, ia, sign;
        size_t str1;
        const _LinkTrilT *tab = clink_indexa + stra_id * nlinka;
        double *cp0, *cp1;

        for (j = 0; j < nlinka; j++) {
                ia   = EXTRACT_IA  (tab[j]);
                str1 = EXTRACT_ADDR(tab[j]);
                sign = EXTRACT_SIGN(tab[j]);
                cp0 = t1 + ia*bcount;
                cp1 = ci1 + str1*nstrb;
                if (sign == 0) {
                        break;
                } else if (sign > 0) {
                        for (k = 0; k < bcount; k++) {
                                cp1[k] += cp0[k];
                        }
                } else {
                        for (k = 0; k < bcount; k++) {
                                cp1[k] -= cp0[k];
                        }
                }
        }
}

void FCIspread_b_t1(double *ci1, double *t1,
                    int bcount, int stra_id, int strb_id,
                    int norb, int nstrb, int nlinkb, _LinkTrilT *clink_indexb)
{
        int j, ia, str0, str1, sign;
        const _LinkTrilT *tab = clink_indexb + strb_id * nlinkb;
        double *pci = ci1 + stra_id * (size_t)nstrb;

        for (str0 = 0; str0 < bcount; str0++) {
                for (j = 0; j < nlinkb; j++) {
                        ia   = EXTRACT_IA  (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        if (sign == 0) {
                                break;
                        } else {
                                pci[str1] += sign * t1[ia*bcount+str0];
                        }
                }
                tab += nlinkb;
        }
}

/*
 * f1e_tril is the 1e hamiltonian for spin alpha
 */
void FCIcontract_a_1e(double *f1e_tril, double *ci0, double *ci1,
                      int norb, int nstra, int nstrb, int nlinka, int nlinkb,
                      int *link_indexa, int *link_indexb)
{
        int j, k, ia, sign;
        size_t str0, str1;
        double *pci0, *pci1;
        double tmp;
        _LinkTrilT *tab;
        _LinkTrilT *clink = malloc(sizeof(_LinkTrilT) * nlinka * nstra);
        FCIcompress_link_tril(clink, link_indexa, nstra, nlinka);

        for (str0 = 0; str0 < nstra; str0++) {
                tab = clink + str0 * nlinka;
                for (j = 0; j < nlinka; j++) {
                        ia   = EXTRACT_IA  (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        pci0 = ci0 + str0 * nstrb;
                        pci1 = ci1 + str1 * nstrb;
                        tmp = sign * f1e_tril[ia];
                        for (k = 0; k < nstrb; k++) {
                                pci1[k] += tmp * pci0[k];
                        }
                }
        }
        free(clink);
}

/*
 * f1e_tril is the 1e hamiltonian for spin beta
 */
void FCIcontract_b_1e(double *f1e_tril, double *ci0, double *ci1,
                      int norb, int nstra, int nstrb, int nlinka, int nlinkb,
                      int *link_indexa, int *link_indexb)
{
        int j, k, ia, sign;
        size_t str0, str1;
        double *pci1;
        double tmp;
        _LinkTrilT *tab;
        _LinkTrilT *clink = malloc(sizeof(_LinkTrilT) * nlinkb * nstrb);
        FCIcompress_link_tril(clink, link_indexb, nstrb, nlinkb);

        for (str0 = 0; str0 < nstra; str0++) {
                pci1 = ci1 + str0 * nstrb;
                for (k = 0; k < nstrb; k++) {
                        tab = clink + k * nlinkb;
                        tmp = ci0[str0*nstrb+k];
                        for (j = 0; j < nlinkb; j++) {
                                ia   = EXTRACT_IA  (tab[j]);
                                str1 = EXTRACT_ADDR(tab[j]);
                                sign = EXTRACT_SIGN(tab[j]);
                                pci1[str1] += sign * tmp * f1e_tril[ia];
                        }
                }
        }
        free(clink);
}

void FCIcontract_1e_spin0(double *f1e_tril, double *ci0, double *ci1,
                          int norb, int na, int nlink, int *link_index)
{
        memset(ci1, 0, sizeof(double)*na*na);
        FCIcontract_a_1e(f1e_tril, ci0, ci1, norb, na, na, nlink, nlink,
                         link_index, link_index);
}

/*
 * spread t1 into ci1buf
 */
static void spread_bufa_t1(double *ci1, double *t1, int nrow_t1,
                           int bcount, int stra_id, int strb_id,
                           int norb, int nstrb, int nlinka, _LinkTrilT *clink_indexa)
{
        int j, k, ia, sign;
        size_t str1;
        const _LinkTrilT *tab = clink_indexa + stra_id * nlinka;
        double *cp0, *cp1;

        for (j = 0; j < nlinka; j++) {
                ia   = EXTRACT_IA  (tab[j]);
                str1 = EXTRACT_ADDR(tab[j]);
                sign = EXTRACT_SIGN(tab[j]);
                cp0 = t1 + ia*nrow_t1;
                cp1 = ci1 + str1*nstrb;
                if (sign == 0) {
                        break;
                } else if (sign > 0) {
                        for (k = 0; k < bcount; k++) {
                                cp1[k] += cp0[k];
                        }
                } else {
                        for (k = 0; k < bcount; k++) {
                                cp1[k] -= cp0[k];
                        }
                }
        }
}

/*
 * bcount_for_spread_a is different for spin1 and spin0
 */
static void ctr_rhf2e_kern(double *eri, double *ci0, double *ci1,
                           double *ci1buf, double *t1buf,
                           int bcount_for_spread_a, int ncol_ci1buf,
                           int bcount, int stra_id, int strb_id,
                           int norb, int na, int nb, int nlinka, int nlinkb,
                           _LinkTrilT *clink_indexa, _LinkTrilT *clink_indexb)
{
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * (norb+1)/2;
        double *t1 = t1buf;
        double *vt1 = t1buf + nnorb*bcount;

        memset(t1, 0, sizeof(double)*nnorb*bcount);
        FCIprog_a_t1(ci0, t1, bcount, stra_id, strb_id,
                     norb, nb, nlinka, clink_indexa);
        FCIprog_b_t1(ci0, t1, bcount, stra_id, strb_id,
                     norb, nb, nlinkb, clink_indexb);

        dgemm_(&TRANS_N, &TRANS_N, &bcount, &nnorb, &nnorb,
               &D1, t1, &bcount, eri, &nnorb, &D0, vt1, &bcount);
        FCIspread_b_t1(ci1, vt1, bcount, stra_id, strb_id,
                       norb, nb, nlinkb, clink_indexb);
        //FCIspread_a_t1(ci1buf, vt1, bcount_for_spread_a, stra_id, 0,
        //               norb, ncol_ci1buf, nlinka, clink_indexa);
        spread_bufa_t1(ci1buf, vt1, bcount, bcount_for_spread_a, stra_id, 0,
                       norb, ncol_ci1buf, nlinka, clink_indexa);
}

void FCIaxpy2d(double *out, double *in, size_t count, size_t no, size_t ni)
{
        int i, j;
        for (i = 0; i < count; i++) {
                for (j = 0; j < ni; j++) {
                        out[i*no+j] += in[i*ni+j];
                }
        }
}

/*
 * nlink = nocc*nvir, num. all possible strings that a string can link to
 * link_index[str0] == linking map between str0 and other strings
 * link_index[str0][ith-linking-string] ==
 *     [tril(creation_op,annihilation_op),0,linking-string-id,sign]
 * FCIcontract_2e_spin0 only compute half of the contraction, due to the
 * symmetry between alpha and beta spin.  The right contracted ci vector
 * is (ci1+ci1.T)
 */
void FCIcontract_2e_spin0(double *eri, double *ci0, double *ci1,
                          int norb, int na, int nlink, int *link_index)
{
        _LinkTrilT *clink = malloc(sizeof(_LinkTrilT) * nlink * na);
        FCIcompress_link_tril(clink, link_index, na, nlink);

        memset(ci1, 0, sizeof(double)*na*na);
        double *ci1bufs[MAX_THREADS];
#pragma omp parallel
{
        int strk, ib;
        size_t blen;
        double *t1buf = malloc(sizeof(double) * (STRB_BLKSIZE*norb*(norb+1)+2));
        double *ci1buf = malloc(sizeof(double) * (na*STRB_BLKSIZE+2));
        ci1bufs[omp_get_thread_num()] = ci1buf;
        for (ib = 0; ib < na; ib += STRB_BLKSIZE) {
                blen = MIN(STRB_BLKSIZE, na-ib);
                memset(ci1buf, 0, sizeof(double) * na*blen);
#pragma omp for schedule(static, 112)
/* strk starts from MAX(strk0, ib), because [0:ib,0:ib] have been evaluated */
                for (strk = ib; strk < na; strk++) {
                        ctr_rhf2e_kern(eri, ci0, ci1, ci1buf, t1buf,
                                       MIN(STRB_BLKSIZE, strk-ib), blen,
                                       MIN(STRB_BLKSIZE, strk+1-ib),
                                       strk, ib, norb, na, na, nlink, nlink,
                                       clink, clink);
                }
                NPomp_dsum_reduce_inplace(ci1bufs, blen*na);
#pragma omp master
                FCIaxpy2d(ci1+ib, ci1buf, na, na, blen);
// An explicit barrier to ensure ci1 is updated. Without barrier, there may
// occur race condition between FCIaxpy2d and ctr_rhf2e_kern
#pragma omp barrier
        }
        free(ci1buf);
        free(t1buf);
}
        free(clink);
}


void FCIcontract_2e_spin1(double *eri, double *ci0, double *ci1,
                          int norb, int na, int nb, int nlinka, int nlinkb,
                          int *link_indexa, int *link_indexb)
{
        _LinkTrilT *clinka = malloc(sizeof(_LinkTrilT) * nlinka * na);
        _LinkTrilT *clinkb = malloc(sizeof(_LinkTrilT) * nlinkb * nb);
        FCIcompress_link_tril(clinka, link_indexa, na, nlinka);
        FCIcompress_link_tril(clinkb, link_indexb, nb, nlinkb);

        memset(ci1, 0, sizeof(double)*na*nb);
        double *ci1bufs[MAX_THREADS];
#pragma omp parallel
{
        int strk, ib;
        size_t blen;
        double *t1buf = malloc(sizeof(double) * (STRB_BLKSIZE*norb*(norb+1)+2));
        double *ci1buf = malloc(sizeof(double) * (na*STRB_BLKSIZE+2));
        ci1bufs[omp_get_thread_num()] = ci1buf;
        for (ib = 0; ib < nb; ib += STRB_BLKSIZE) {
                blen = MIN(STRB_BLKSIZE, nb-ib);
                memset(ci1buf, 0, sizeof(double) * na*blen);
#pragma omp for schedule(static)
                for (strk = 0; strk < na; strk++) {
                        ctr_rhf2e_kern(eri, ci0, ci1, ci1buf, t1buf,
                                       blen, blen, blen, strk, ib,
                                       norb, na, nb, nlinka, nlinkb,
                                       clinka, clinkb);
                }
                NPomp_dsum_reduce_inplace(ci1bufs, blen*na);
#pragma omp master
                FCIaxpy2d(ci1+ib, ci1buf, na, nb, blen);
// An explicit barrier to ensure ci1 is updated. Without barrier, there may
// occur race condition between FCIaxpy2d and ctr_rhf2e_kern
#pragma omp barrier
        }
        free(ci1buf);
        free(t1buf);
}
        free(clinka);
        free(clinkb);
}


/*
 * eri_ab is mixed integrals (alpha,alpha|beta,beta), |beta,beta) in small strides
 */
static void ctr_uhf2e_kern(double *eri_aa, double *eri_ab, double *eri_bb,
                           double *ci0, double *ci1, double *ci1buf, double *t1buf,
                           int bcount, int stra_id, int strb_id,
                           int norb, int na, int nb, int nlinka, int nlinkb,
                           _LinkTrilT *clink_indexa, _LinkTrilT *clink_indexb)
{
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * (norb+1)/2;
        double *t1a = t1buf;
        double *t1b = t1a + nnorb*bcount;
        double *vt1 = t1b + nnorb*bcount;

        memset(t1a, 0, sizeof(double)*nnorb*bcount);
        memset(t1b, 0, sizeof(double)*nnorb*bcount);
        FCIprog_a_t1(ci0, t1a, bcount, stra_id, strb_id,
                     norb, nb, nlinka, clink_indexa);
        FCIprog_b_t1(ci0, t1b, bcount, stra_id, strb_id,
                     norb, nb, nlinkb, clink_indexb);

        dgemm_(&TRANS_N, &TRANS_T, &bcount, &nnorb, &nnorb,
               &D1, t1a, &bcount, eri_ab, &nnorb, &D0, vt1, &bcount);
        dgemm_(&TRANS_N, &TRANS_N, &bcount, &nnorb, &nnorb,
               &D1, t1b, &bcount, eri_bb, &nnorb, &D1, vt1, &bcount);
        FCIspread_b_t1(ci1, vt1, bcount, stra_id, strb_id,
                       norb, nb, nlinkb, clink_indexb);

        dgemm_(&TRANS_N, &TRANS_N, &bcount, &nnorb, &nnorb,
               &D1, t1a, &bcount, eri_aa, &nnorb, &D0, vt1, &bcount);
        dgemm_(&TRANS_N, &TRANS_N, &bcount, &nnorb, &nnorb,
               &D1, t1b, &bcount, eri_ab, &nnorb, &D1, vt1, &bcount);
        FCIspread_a_t1(ci1buf, vt1, bcount, stra_id, 0,
                       norb, bcount, nlinka, clink_indexa);
}

void FCIcontract_uhf2e(double *eri_aa, double *eri_ab, double *eri_bb,
                       double *ci0, double *ci1,
                       int norb, int na, int nb, int nlinka, int nlinkb,
                       int *link_indexa, int *link_indexb)
{
        _LinkTrilT *clinka = malloc(sizeof(_LinkTrilT) * nlinka * na);
        _LinkTrilT *clinkb = malloc(sizeof(_LinkTrilT) * nlinkb * nb);
        FCIcompress_link_tril(clinka, link_indexa, na, nlinka);
        FCIcompress_link_tril(clinkb, link_indexb, nb, nlinkb);

        memset(ci1, 0, sizeof(double)*na*nb);
        double *ci1bufs[MAX_THREADS];
#pragma omp parallel
{
        int strk, ib;
        size_t blen;
        double *t1buf = malloc(sizeof(double) * (STRB_BLKSIZE*norb*(norb+1)*2+2));
        double *ci1buf = malloc(sizeof(double) * (na*STRB_BLKSIZE+2));
        ci1bufs[omp_get_thread_num()] = ci1buf;
        for (ib = 0; ib < nb; ib += STRB_BLKSIZE) {
                blen = MIN(STRB_BLKSIZE, nb-ib);
                memset(ci1buf, 0, sizeof(double) * na*blen);
#pragma omp for schedule(static)
                for (strk = 0; strk < na; strk++) {
                        ctr_uhf2e_kern(eri_aa, eri_ab, eri_bb, ci0, ci1,
                                       ci1buf, t1buf, blen, strk, ib,
                                       norb, na, nb, nlinka, nlinkb,
                                       clinka, clinkb);
                }
                NPomp_dsum_reduce_inplace(ci1bufs, blen*na);
#pragma omp master
                FCIaxpy2d(ci1+ib, ci1buf, na, nb, blen);
// An explicit barrier to ensure ci1 is updated. Without barrier, there may
// occur race condition between FCIaxpy2d and ctr_uhf2e_kern
#pragma omp barrier
        }
        free(t1buf);
        free(ci1buf);
}
        free(clinka);
        free(clinkb);
}



/*************************************************
 * hdiag
 *************************************************/

void FCImake_hdiag_uhf(double *hdiag, double *h1e_a, double *h1e_b,
                       double *jdiag_aa, double *jdiag_ab, double *jdiag_bb,
                       double *kdiag_aa, double *kdiag_bb,
                       int norb, int nstra, int nstrb, int nocca, int noccb,
                       int *occslista, int *occslistb)
{
#pragma omp parallel
{
        int j, j0, k0, jk, jk0;
        size_t ia, ib;
        double e1, e2;
        int *paocc, *pbocc;
#pragma omp for schedule(static)
        for (ia = 0; ia < nstra; ia++) {
                paocc = occslista + ia * nocca;
                for (ib = 0; ib < nstrb; ib++) {
                        e1 = 0;
                        e2 = 0;
                        pbocc = occslistb + ib * noccb;
                        for (j0 = 0; j0 < nocca; j0++) {
                                j = paocc[j0];
                                jk0 = j * norb;
                                e1 += h1e_a[j*norb+j];
                                for (k0 = 0; k0 < nocca; k0++) { // (alpha|alpha)
                                        jk = jk0 + paocc[k0];
                                        e2 += jdiag_aa[jk] - kdiag_aa[jk];
                                }
                                for (k0 = 0; k0 < noccb; k0++) { // (alpha|beta)
                                        jk = jk0 + pbocc[k0];
                                        e2 += jdiag_ab[jk] * 2;
                                }
                        }
                        for (j0 = 0; j0 < noccb; j0++) {
                                j = pbocc[j0];
                                jk0 = j * norb;
                                e1 += h1e_b[j*norb+j];
                                for (k0 = 0; k0 < noccb; k0++) { // (beta|beta)
                                        jk = jk0 + pbocc[k0];
                                        e2 += jdiag_bb[jk] - kdiag_bb[jk];
                                }
                        }
                        hdiag[ia*nstrb+ib] = e1 + e2 * .5;
                }
        }
}
}

void FCImake_hdiag(double *hdiag, double *h1e, double *jdiag, double *kdiag,
                   int norb, int na, int nocc, int *occslst)
{
        FCImake_hdiag_uhf(hdiag, h1e, h1e, jdiag, jdiag, jdiag, kdiag, kdiag,
                          norb, na, na, nocc, nocc, occslst, occslst);
}

static int first1(uint64_t r)
{
#ifdef HAVE_FFS
        return ffsll(r) - 1;
#else
        int n = 0;
        if (r >> (n + 32)) n += 32;
        if (r >> (n + 16)) n += 16;
        if (r >> (n +  8)) n +=  8;
        if (r >> (n +  4)) n +=  4;
        if (r >> (n +  2)) n +=  2;
        if (r >> (n +  1)) n +=  1;
        return n;
#endif
}


/*************************************************
 * pspace Hamiltonian, ref CPL, 169, 463
 *************************************************/
/*
 * sub-space Hamiltonian (tril part) of the determinants (stra,strb)
 */

void FCIpspace_h0tril_uhf(double *h0, double *h1e_a, double *h1e_b,
                          double *g2e_aa, double *g2e_ab, double *g2e_bb,
                          uint64_t *stra, uint64_t *strb,
                          int norb, int np)
{
        const int d2 = norb * norb;
        const int d3 = norb * norb * norb;
#pragma omp parallel
{
        int i, j, k, pi, pj, pk, pl;
        int n1da, n1db;
        uint64_t da, db, str1;
        double tmp;
#pragma omp for schedule(dynamic)
        for (i = 0; i < np; i++) {
        for (j = 0; j < i; j++) {
                da = stra[i] ^ stra[j];
                db = strb[i] ^ strb[j];
                n1da = FCIpopcount_1(da);
                n1db = FCIpopcount_1(db);
                switch (n1da) {
                case 0: switch (n1db) {
                        case 2:
                        pi = first1(db & strb[i]);
                        pj = first1(db & strb[j]);
                        tmp = h1e_b[pi*norb+pj];
                        for (k = 0; k < norb; k++) {
                                if (stra[i] & (1ULL<<k)) {
                                        tmp += g2e_ab[pi*norb+pj+k*d3+k*d2];
                                }
                                if (strb[i] & (1ULL<<k)) {
                                        tmp += g2e_bb[pi*d3+pj*d2+k*norb+k]
                                             - g2e_bb[pi*d3+k*d2+k*norb+pj];
                                }
                        }
                        if (FCIcre_des_sign(pi, pj, strb[j]) > 0) {
                                h0[i*np+j] = tmp;
                        } else {
                                h0[i*np+j] = -tmp;
                        } break;

                        case 4:
                        pi = first1(db & strb[i]);
                        pj = first1(db & strb[j]);
                        pk = first1((db & strb[i]) ^ (1ULL<<pi));
                        pl = first1((db & strb[j]) ^ (1ULL<<pj));
                        str1 = strb[j] ^ (1ULL<<pi) ^ (1ULL<<pj);
                        if (FCIcre_des_sign(pi, pj, strb[j])
                           *FCIcre_des_sign(pk, pl, str1) > 0) {
                                h0[i*np+j] = g2e_bb[pi*d3+pj*d2+pk*norb+pl]
                                           - g2e_bb[pi*d3+pl*d2+pk*norb+pj];
                        } else {
                                h0[i*np+j] =-g2e_bb[pi*d3+pj*d2+pk*norb+pl]
                                           + g2e_bb[pi*d3+pl*d2+pk*norb+pj];
                        } } break;
                case 2: switch (n1db) {
                        case 0:
                        pi = first1(da & stra[i]);
                        pj = first1(da & stra[j]);
                        tmp = h1e_a[pi*norb+pj];
                        for (k = 0; k < norb; k++) {
                                if (strb[i] & (1ULL<<k)) {
                                        tmp += g2e_ab[pi*d3+pj*d2+k*norb+k];
                                }
                                if (stra[i] & (1ULL<<k)) {
                                        tmp += g2e_aa[pi*d3+pj*d2+k*norb+k]
                                             - g2e_aa[pi*d3+k*d2+k*norb+pj];
                                }
                        }
                        if (FCIcre_des_sign(pi, pj, stra[j]) > 0) {
                                h0[i*np+j] = tmp;
                        } else {
                                h0[i*np+j] = -tmp;
                        } break;

                        case 2:
                        pi = first1(da & stra[i]);
                        pj = first1(da & stra[j]);
                        pk = first1(db & strb[i]);
                        pl = first1(db & strb[j]);
                        if (FCIcre_des_sign(pi, pj, stra[j])
                           *FCIcre_des_sign(pk, pl, strb[j]) > 0) {
                                h0[i*np+j] = g2e_ab[pi*d3+pj*d2+pk*norb+pl];
                        } else {
                                h0[i*np+j] =-g2e_ab[pi*d3+pj*d2+pk*norb+pl];
                        } } break;
                case 4: switch (n1db) {
                        case 0:
                        pi = first1(da & stra[i]);
                        pj = first1(da & stra[j]);
                        pk = first1((da & stra[i]) ^ (1ULL<<pi));
                        pl = first1((da & stra[j]) ^ (1ULL<<pj));
                        str1 = stra[j] ^ (1ULL<<pi) ^ (1ULL<<pj);
                        if (FCIcre_des_sign(pi, pj, stra[j])
                           *FCIcre_des_sign(pk, pl, str1) > 0) {
                                h0[i*np+j] = g2e_aa[pi*d3+pj*d2+pk*norb+pl]
                                           - g2e_aa[pi*d3+pl*d2+pk*norb+pj];
                        } else {
                                h0[i*np+j] =-g2e_aa[pi*d3+pj*d2+pk*norb+pl]
                                           + g2e_aa[pi*d3+pl*d2+pk*norb+pj];
                        }
                        } break;
                }
        } }
}
}

void FCIpspace_h0tril(double *h0, double *h1e, double *g2e,
                      uint64_t *stra, uint64_t *strb, int norb, int np)
{
        FCIpspace_h0tril_uhf(h0, h1e, h1e, g2e, g2e, g2e, stra, strb, norb, np);
}



/***********************************************************************
 *
 * With symmetry
 *
 * Note the ordering in eri and the index in link_index
 * eri is a tril matrix, it should be reordered wrt the irrep of the
 * direct product E_i^j.  The 2D array eri(ij,kl) is a diagonal block
 * matrix.  Each block is associated with an irrep.
 * link_index[str_id,pair_id,0] which is the index of pair_id, should be
 * reordered wrt the irreps accordingly
 *
 * dimirrep stores the number of occurence for each irrep
 *
 ***********************************************************************/
static void pick_link_by_irrep(_LinkTrilT *clink, int *link_index,
                               int nstr, int nlink, int eri_irrep)
{
        int i, j, k;
        for (i = 0; i < nstr; i++) {
                for (k = 0, j = 0; k < nlink; k++) {
                        if (link_index[k*4+1] == eri_irrep) {
                                clink[j].ia   = link_index[k*4+0];
                                clink[j].addr = link_index[k*4+2];
                                clink[j].sign = link_index[k*4+3];
                                j++;
                        }
                }
                if (j < nlink) {
                        clink[j].sign = 0;
                }
                clink += nlink;
                link_index += nlink * 4;
        }
}

static void ctr_rhf2esym_kern1(double *eri, double *ci0, double *ci1ab,
                              double *ci1buf, double *t1buf, int ncol_ci1buf,
                              int bcount, int stra_id, int strb_id,
                              int nnorb, int nb_intermediate,
                              int na, int nb, int nlinka, int nlinkb,
                              _LinkTrilT *clink_indexa, _LinkTrilT *clink_indexb)
{
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        double *t1 = t1buf;
        double *vt1 = t1buf + nnorb*bcount;

        memset(t1, 0, sizeof(double)*nnorb*bcount);
        FCIprog_a_t1(ci0, t1, bcount, stra_id, strb_id,
                     0, nb, nlinka, clink_indexa);
        dgemm_(&TRANS_N, &TRANS_N, &bcount, &nnorb, &nnorb,
               &D1, t1, &bcount, eri, &nnorb, &D0, vt1, &bcount);
        FCIspread_b_t1(ci1ab, vt1, bcount, stra_id, strb_id,
                       0, nb_intermediate, nlinkb, clink_indexb);
        spread_bufa_t1(ci1buf, vt1, bcount, bcount, stra_id, 0,
                       0, ncol_ci1buf, nlinka, clink_indexa);
}

static void loop_c2e_symm1(double *eri, double *ci0, double *ci1aa, double *ci1ab,
                           int nnorb, int na_intermediate, int nb_intermediate,
                           int na, int nb, int nlinka, int nlinkb,
                           _LinkTrilT *clinka, _LinkTrilT *clinkb)
{
        double *ci1bufs[MAX_THREADS];
#pragma omp parallel
{
        int strk, ib;
        size_t blen;
        double *t1buf = malloc(sizeof(double) * (STRB_BLKSIZE*nnorb*2+2));
        double *ci1buf = malloc(sizeof(double) * (na*STRB_BLKSIZE+2));
        ci1bufs[omp_get_thread_num()] = ci1buf;
        for (ib = 0; ib < nb; ib += STRB_BLKSIZE) {
                blen = MIN(STRB_BLKSIZE, nb-ib);
                memset(ci1buf, 0, sizeof(double) * na*blen);
#pragma omp for schedule(static)
                for (strk = 0; strk < na_intermediate; strk++) {
                        ctr_rhf2esym_kern1(eri, ci0, ci1ab, ci1buf, t1buf,
                                           blen, blen, strk, ib,
                                           nnorb, nb_intermediate, na, nb,
                                           nlinka, nlinkb, clinka, clinkb);
                }
                NPomp_dsum_reduce_inplace(ci1bufs, blen*na);
#pragma omp master
                FCIaxpy2d(ci1aa+ib, ci1buf, na, nb, blen);
// An explicit barrier to ensure ci1 is updated. Without barrier, there may
// occur race condition between FCIaxpy2d and ctr_rhf2esym_kern1
#pragma omp barrier
        }
        free(ci1buf);
        free(t1buf);
}
}

#define TOTIRREPS       8
void FCIcontract_2e_symm1(double **eris, double **ci0, double **ci1,
                          int norb, int *nas, int *nbs, int nlinka, int nlinkb,
                          int **linka, int **linkb, int *dimirrep, int wfnsym)
{
        int i;
        int na = 0;
        int nb = 0;
        for (i = 0; i < TOTIRREPS; i++) {
                na = MAX(nas[i], na);
                nb = MAX(nbs[i], nb);
        }
        _LinkTrilT *clinka = malloc(sizeof(_LinkTrilT) * nlinka * na);
        _LinkTrilT *clinkb = malloc(sizeof(_LinkTrilT) * nlinkb * nb);
        int ai_ir, stra_ir, strb_ir, intera_ir, interb_ir, ma, mb;
        for (stra_ir = 0; stra_ir < TOTIRREPS; stra_ir++) {
        for (ai_ir = 0; ai_ir < TOTIRREPS; ai_ir++) {
                strb_ir = wfnsym^stra_ir;
                ma = nas[stra_ir];
                mb = nbs[strb_ir];
                if (ma > 0 && mb > 0 && dimirrep[ai_ir] > 0) {
                        intera_ir = ai_ir^stra_ir;
                        interb_ir = ai_ir^strb_ir;
                        // clinka for inter_ir*ai_ir -> stra_ir
                        pick_link_by_irrep(clinka, linka[intera_ir],
                                           nas[intera_ir], nlinka, ai_ir);
                        // clinka for strb_ir*ai_ir -> inter_ir
                        pick_link_by_irrep(clinkb, linkb[strb_ir],
                                           nbs[strb_ir], nlinkb, ai_ir);
                        loop_c2e_symm1(eris[ai_ir], ci0[stra_ir],
                                       ci1[stra_ir], ci1[intera_ir],
                                       dimirrep[ai_ir], nas[intera_ir],
                                       nbs[interb_ir], ma, mb,
                                       nlinka, nlinkb, clinka, clinkb);
                }
        } }
        free(clinka);
        free(clinkb);
}

