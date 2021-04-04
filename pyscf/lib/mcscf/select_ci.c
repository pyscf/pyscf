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
 * Select CI
 */

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "config.h"
#include <assert.h>
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"
#include "fci.h"
#define BUFBASE         112
#define STRB_BLKSIZE    224


int SCIstr2addr(uint64_t str, uint64_t *strsbook, int nstrs)
{
        int head = 0;
        int tail = nstrs;
        int mid;
        int addr = -1;
        while (head < tail) {
                mid = (head + tail) / 2;
                if (str == strsbook[mid]) {
                        addr = mid;
                        break;
                } else if (str < strsbook[mid]) {
                        tail = mid;
                } else {
                        head = mid + 1;
                }
        }
        return addr;
}

static void make_occ_vir(int *occ, int *vir, uint64_t str1, int norb)
{
        int i, io, iv;
        for (i = 0, io = 0, iv = 0; i < norb; i++) {
                if (str1 & (1ULL<<i)) {
                        occ[io] = i;
                        io += 1;
                } else {
                        vir[iv] = i;
                        iv += 1;
                }
        }
}

void SCIcre_des_linkstr(int *link_index, int norb, int nstrs, int nocc,
                        uint64_t *strs, int store_trilidx)
{
        int ninter = nstrs;
        int occ[norb];
        int vir[norb];
        int nvir = norb - nocc;
        int nlink = nocc * nvir + nocc;
        int str_id, i, a, k, ai, addr;
        uint64_t str0, str1;
        int *tab;

        for (str_id = 0; str_id < ninter; str_id++) {
                str1 = strs[str_id];
                make_occ_vir(occ, vir, str1, norb);

                tab = link_index + str_id * nlink * 4;
                if (store_trilidx) {
                        for (k = 0; k < nocc; k++) {
                                tab[k*4+0] = occ[k]*(occ[k]+1)/2+occ[k];
                                tab[k*4+2] = str_id;
                                tab[k*4+3] = 1;
                        }
                        for (a = 0; a < nvir; a++) {
                        for (i = 0; i < nocc; i++) {
                                str0 = (str1^(1ULL<<occ[i])) | (1ULL<<vir[a]);
                                addr = SCIstr2addr(str0, strs, nstrs);
                                if (addr >= 0) {
                                        if (vir[a] > occ[i]) {
                                                ai = vir[a]*(vir[a]+1)/2+occ[i];
                                        } else {
                                                ai = occ[i]*(occ[i]+1)/2+vir[a];
                                        }
                                        tab[k*4+0] = ai;
                                        tab[k*4+2] = addr;
                                        tab[k*4+3] = FCIcre_des_sign(vir[a], occ[i], str1);
                                        k++;
                                }
                        } }

                } else {
                        for (k = 0; k < nocc; k++) {
                                tab[k*4+0] = occ[k];
                                tab[k*4+1] = occ[k];
                                tab[k*4+2] = str_id;
                                tab[k*4+3] = 1;
                        }
                        for (a = 0; a < nvir; a++) {
                        for (i = 0; i < nocc; i++) {
                                str0 = (str1^(1ULL<<occ[i])) | (1ULL<<vir[a]);
                                addr = SCIstr2addr(str0, strs, nstrs);
                                if (addr >= 0) {
                                        tab[k*4+0] = vir[a];
                                        tab[k*4+1] = occ[i];
                                        tab[k*4+2] = addr;
                                        tab[k*4+3] = FCIcre_des_sign(vir[a], occ[i], str1);
                                        k++;
                                }
                        } }
                }
        }
}

void SCIdes_des_linkstr(int *link_index, int norb, int nocc, int nstrs, int ninter,
                        uint64_t *strs, uint64_t *inter, int store_trilidx)
{
        int occ[norb];
        int vir[norb];
        int str_id, i, j, k, addr;
        uint64_t str0, str1;
        int sign;
        int nvir = norb - nocc + 2;
        int nlink = nvir * nvir;
        int *tab;
        for (str_id = 0; str_id < ninter; str_id++) {
                str1 = inter[str_id];
                make_occ_vir(occ, vir, str1, norb);

                tab = link_index + str_id * nlink * 4;
                if (store_trilidx) {
                        for (k = 0, i = 1; i < nvir; i++) {
                        for (j = 0; j < i; j++) {
                                str0 = str1 | (1ULL<<vir[i]) | (1ULL<<vir[j]);
                                addr = SCIstr2addr(str0, strs, nstrs);
                                if (addr >= 0) {
                                        sign = FCIcre_sign(vir[i], str1);
                                        sign*= FCIdes_sign(vir[j], str0);
                                        tab[k*4+0] = vir[i]*(vir[i]-1)/2+vir[j];;
                                        tab[k*4+2] = addr;
                                        tab[k*4+3] = sign;
                                        k++;
                                }
                        } }

                } else {
                        for (k = 0, i = 1; i < nvir; i++) {
                        for (j = 0; j < i; j++) {
                                str0 = str1 | (1ULL<<vir[i]) | (1ULL<<vir[j]);
                                addr = SCIstr2addr(str0, strs, nstrs);
                                if (addr >= 0) {
                                        sign = FCIcre_sign(vir[i], str1);
                                        sign*= FCIdes_sign(vir[j], str0);
                                        tab[k*4+0] = vir[i];
                                        tab[k*4+1] = vir[j];
                                        tab[k*4+2] = addr;
                                        tab[k*4+3] = sign;
                                        k++;
                                        tab[k*4+0] = vir[j];
                                        tab[k*4+1] = vir[i];
                                        tab[k*4+2] = addr;
                                        tab[k*4+3] =-sign;
                                        k++;
                                }
                        } }
                }
        }
}

int SCIdes_uniq_strs(uint64_t *uniq_strs, uint64_t *strs,
                     int norb, int nocc, int nstrs)
{
        int str_id, i;
        uint64_t str0, str1;
        int ninter = 0;

        for (str_id = 0; str_id < nstrs; str_id++) {
                str0 = strs[str_id];
                for (i = 0; i < norb; i++) {
                        if (str0 & (1ULL<<i)) {
                                str1 = str0 ^ (1ULL<<i);
                                uniq_strs[ninter] = str1;
                                ninter++;
                        }
                }
        }
        return ninter;
}

void SCIdes_linkstr(int *link_index, int norb, int nocc, int nstrs, int ninter,
                    uint64_t *strs, uint64_t *inter)
{
        int str_id, i, k, addr;
        uint64_t str0, str1;
        int nvir = norb - nocc + 1;
        int nlink = nvir;
        int *tab;
        for (str_id = 0; str_id < ninter; str_id++) {
                str1 = inter[str_id];
                tab = link_index + str_id * nlink * 4;
                for (k = 0, i = 0; i < norb; i++) {
                        if (!(str1 & (1ULL<<i))) {
                                str0 = str1 | (1ULL<<i);
                                addr = SCIstr2addr(str0, strs, nstrs);
                                if (addr >= 0) {
                                        tab[k*4+0] = 0;
                                        tab[k*4+1] = i;
                                        tab[k*4+2] = addr;
                                        tab[k*4+3] = FCIdes_sign(i, str0);
                                        k++;
                                }
                        }
                }
        }
}

int SCIcre_uniq_strs(uint64_t *uniq_strs, uint64_t *strs,
                     int norb, int nocc, int nstrs)
{
        int str_id, i;
        uint64_t str0, str1;
        int ninter = 0;

        for (str_id = 0; str_id < nstrs; str_id++) {
                str0 = strs[str_id];
                for (i = 0; i < norb; i++) {
                        if (!(str0 & (1ULL<<i))) {
                                str1 = str0 | (1ULL<<i);
                                uniq_strs[ninter] = str1;
                                ninter++;
                        }
                }
        }
        return ninter;
}

void SCIcre_linkstr(int *link_index, int norb, int nocc, int nstrs, int ninter,
                    uint64_t *strs, uint64_t *inter)
{
        int str_id, i, k, addr;
        uint64_t str0, str1;
        int nlink = nocc + 1;
        int *tab;
        for (str_id = 0; str_id < ninter; str_id++) {
                str1 = inter[str_id];
                tab = link_index + str_id * nlink * 4;
                for (k = 0, i = 0; i < norb; i++) {
                        if (str1 & (1ULL<<i)) {
                                str0 = str1 ^ (1ULL<<i);
                                addr = SCIstr2addr(str0, strs, nstrs);
                                if (addr >= 0) {
                                        tab[k*4+0] = i;
                                        tab[k*4+1] = 0;
                                        tab[k*4+2] = addr;
                                        tab[k*4+3] = FCIcre_sign(i, str0);
                                        k++;
                                }
                        }
                }
        }
}

int SCIselect_strs(uint64_t *inter, uint64_t *strs,
                   double *eri, double *eri_pq_max, double *civec_max,
                   double select_cutoff, int norb, int nocc, int nstrs)
{
        int nn = norb * norb;
        int n3 = norb * nn;
        int occ[norb];
        int vir[norb];
        int nvir = norb - nocc;
        int str_id, i, a, j, b;
        uint64_t str0, str1;
        double ca;
        double *peri;

        int ninter = 0;
        for (str_id = 0; str_id < nstrs; str_id++) {
                str0 = strs[str_id];
                make_occ_vir(occ, vir, str0, norb);

                ca = civec_max[str_id];
                for (i = 0; i < nocc; i++) {
                for (a = 0; a < nvir; a++) {
                if (eri_pq_max[vir[a]*norb+occ[i]]*ca > select_cutoff) {
                        str1 = (str0 ^ (1ULL<<occ[i])) | (1ULL<<vir[a]);
                        inter[ninter] = str1;
                        ninter++;

                        if (occ[i] < nocc && vir[a] >= nocc) {
                                peri = eri + n3 * vir[a] + nn * occ[i];
                                for (j = 0; j < i; j++) {
                                for (b = a+1; b < nvir; b++) {
                                if (fabs(peri[vir[b]*norb+occ[j]])*ca > select_cutoff) {
                                        inter[ninter] = (str1 ^ (1ULL<<occ[j])) | (1ULL<<vir[b]);
                                        ninter++;
                                } } }
                        }
                } } }
        }
        return ninter;
}


/*
 ***********************************************************
 *
 * Need the permutation symmetry 
 * h2e[i,j,k,l] = h2e[j,i,k,l] = h2e[i,j,l,k] = h2e[j,i,l,k]
 *
 ***********************************************************
 */

static void ctr_bbaa_kern(double *eri, double *ci0, double *ci1,
                          double *ci1buf, double *t1buf,
                          int bcount, int stra_id, int strb_id,
                          int norb, int na, int nb, int nlinka, int nlinkb,
                          _LinkTrilT *clink_indexa, _LinkTrilT *clink_indexb)
{
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * (norb+1) / 2;
        double *t1  = t1buf;
        double *vt1 = t1buf + nnorb*bcount;

        NPdset0(t1, nnorb*bcount);
        FCIprog_a_t1(ci0, t1, bcount, stra_id, strb_id,
                     norb, nb, nlinka, clink_indexa);
        dgemm_(&TRANS_N, &TRANS_N, &bcount, &nnorb, &nnorb,
               &D1, t1, &bcount, eri, &nnorb, &D0, vt1, &bcount);
        FCIspread_b_t1(ci1, vt1, bcount, stra_id, strb_id,
                       norb, nb, nlinkb, clink_indexb);
}

void SCIcontract_2e_bbaa(double *eri, double *ci0, double *ci1,
                         int norb, int na, int nb, int nlinka, int nlinkb,
                         int *link_indexa, int *link_indexb)
{
        _LinkTrilT *clinka = malloc(sizeof(_LinkTrilT) * nlinka * na);
        _LinkTrilT *clinkb = malloc(sizeof(_LinkTrilT) * nlinkb * nb);
        FCIcompress_link_tril(clinka, link_indexa, na, nlinka);
        FCIcompress_link_tril(clinkb, link_indexb, nb, nlinkb);

#pragma omp parallel
{
        int strk, ib, blen;
        double *t1buf = malloc(sizeof(double) * (STRB_BLKSIZE*norb*(norb+1)+2));
        double *ci1buf = NULL;
        for (ib = 0; ib < nb; ib += STRB_BLKSIZE) {
                blen = MIN(STRB_BLKSIZE, nb-ib);
#pragma omp for schedule(static)
                for (strk = 0; strk < na; strk++) {
                        ctr_bbaa_kern(eri, ci0, ci1, ci1buf, t1buf,
                                      blen, strk, ib, norb, na, nb,
                                      nlinka, nlinkb, clinka, clinkb);
                }
        }
        free(t1buf);
}
        free(clinka);
        free(clinkb);
}

static void ctr_aaaa_kern(double *eri, double *ci0, double *ci1,
                          double *ci1buf, double *t1buf,
                          int bcount, int stra_id, int strb_id,
                          int norb, int na, int nb, int nlinka, int nlinkb,
                          _LinkTrilT *clink_indexa, _LinkTrilT *clink_indexb)
{
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * (norb-1) / 2;
        double *t1  = t1buf;
        double *vt1 = t1buf + nnorb*bcount;

        NPdset0(t1, nnorb*bcount);
        FCIprog_a_t1(ci0, t1, bcount, stra_id, strb_id,
                     norb, nb, nlinka, clink_indexa);
        dgemm_(&TRANS_N, &TRANS_N, &bcount, &nnorb, &nnorb,
               &D1, t1, &bcount, eri, &nnorb, &D0, vt1, &bcount);
        FCIspread_a_t1(ci1buf, vt1, bcount, stra_id, 0,
                       norb, bcount, nlinka, clink_indexa);
}

void SCIcontract_2e_aaaa(double *eri, double *ci0, double *ci1,
                         int norb, int na, int nb,
                         int inter_na, int nlinka, int *link_indexa)
{
        _LinkTrilT *clinka = malloc(sizeof(_LinkTrilT) * nlinka * inter_na);
        FCIcompress_link_tril(clinka, link_indexa, inter_na, nlinka);
        _LinkTrilT *clinkb = NULL;

        double *ci1bufs[MAX_THREADS];
#pragma omp parallel
{
        int strk, ib, blen;
        double *t1buf = malloc(sizeof(double) * (STRB_BLKSIZE*norb*norb+2));
        double *ci1buf = malloc(sizeof(double) * (na*STRB_BLKSIZE+2));
        ci1bufs[omp_get_thread_num()] = ci1buf;
        for (ib = 0; ib < nb; ib += STRB_BLKSIZE) {
                blen = MIN(STRB_BLKSIZE, nb-ib);
                NPdset0(ci1buf, ((size_t)na) * blen);
#pragma omp for schedule(static)
                for (strk = 0; strk < inter_na; strk++) {
                        ctr_aaaa_kern(eri, ci0, ci1, ci1buf, t1buf,
                                      blen, strk, ib, norb, na, nb,
                                      nlinka, 0, clinka, clinkb);
                }
                NPomp_dsum_reduce_inplace(ci1bufs, blen*na);
#pragma omp master
                FCIaxpy2d(ci1+ib, ci1buf, na, nb, blen);
#pragma omp barrier
        }
        free(ci1buf);
        free(t1buf);
}
        free(clinka);
}



/*************************************************
 *
 * 2-particle DM
 *
 *************************************************/
void SCIrdm2_a_t1ci(double *ci0, double *t1,
                    int bcount, int stra_id, int strb_id,
                    int norb, int nstrb, int nlinka, _LinkT *clink_indexa)
{
        ci0 += strb_id;
        int i, j, k, a, sign;
        size_t str1;
        const _LinkT *tab = clink_indexa + stra_id * nlinka;
        double *pt1, *pci;

        for (j = 0; j < nlinka; j++) {
                a    = EXTRACT_CRE (tab[j]);
                i    = EXTRACT_DES (tab[j]);
                str1 = EXTRACT_ADDR(tab[j]);
                sign = EXTRACT_SIGN(tab[j]);
                pci = ci0 + str1*nstrb;
                pt1 = t1 + (i*norb+a) * bcount;
                if (sign == 0) {
                        break;
                } else if (sign > 0) {
                        for (k = 0; k < bcount; k++) {
                                pt1[k] += pci[k];
                        }
                } else {
                        for (k = 0; k < bcount; k++) {
                                pt1[k] -= pci[k];
                        }
                }
        }
}

void SCIrdm2kern_aaaa(double *rdm2, double *bra, double *ket, double *buf,
                      int bcount, int stra_id, int strb_id, int norb,
                      int na, int nb, int nlinka, _LinkT *clink_indexa)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const double DN1 = -1;
        const int nnorb = norb * norb;

        NPdset0(buf, nnorb*bcount);
        SCIrdm2_a_t1ci(ket, buf, bcount, stra_id, strb_id,
                       norb, nb, nlinka, clink_indexa);
        dgemm_(&TRANS_T, &TRANS_N, &nnorb, &nnorb, &bcount,
               &DN1, buf, &bcount, buf, &bcount, &D1, rdm2, &nnorb);
}

void SCIrdm2_aaaa(void (*dm2kernel)(), double *rdm2, double *bra, double *ket,
                  int norb, int na, int nb, int inter_na, int nlinka,
                  int *link_indexa)
{
        const int nnorb = norb * norb;
        double *pdm2;
        NPdset0(rdm2, nnorb*nnorb);

        _LinkT *clinka = malloc(sizeof(_LinkT) * nlinka * inter_na);
        FCIcompress_link(clinka, link_indexa, norb, inter_na, nlinka);

#pragma omp parallel private(pdm2)
{
        int strk, i, ib, blen;
        double *buf = malloc(sizeof(double) * (nnorb*BUFBASE*2+2));
        pdm2 = calloc(nnorb*nnorb, sizeof(double));
#pragma omp for schedule(dynamic, 40)
        for (strk = 0; strk < inter_na; strk++) {
                for (ib = 0; ib < nb; ib += BUFBASE) {
                        blen = MIN(BUFBASE, nb-ib);
                        (*dm2kernel)(pdm2, bra, ket, buf, blen, strk, ib,
                                     norb, na, nb, nlinka, clinka);
                }
        }
#pragma omp critical
{
        for (i = 0; i < nnorb*nnorb; i++) {
                rdm2[i] += pdm2[i];
        }
}
        free(pdm2);
        free(buf);
}
        free(clinka);

        int shape[] = {norb, nnorb, norb};
        pdm2 = malloc(sizeof(double) * nnorb*nnorb);
        NPdtranspose_021(shape, rdm2, pdm2);
        NPdcopy(rdm2, pdm2, nnorb*nnorb);
        free(pdm2);
}


/***********************************************************************
 *
 * With symmetry
 *
 ***********************************************************************/
static void ctr_bbaa_symm(double *eri, double *ci0, double *ci1,
                          double *ci1buf, double *t1buf,
                          int bcount, int stra_id, int strb_id,
                          int norb, int na, int nb, int nlinka, int nlinkb,
                          _LinkTrilT *clink_indexa, _LinkTrilT *clink_indexb,
                          int *dimirrep, int totirrep)
{
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * (norb+1) / 2;
        int ir, p0;
        double *t1  = t1buf;
        double *vt1 = t1buf + nnorb*bcount;

        NPdset0(t1, nnorb*bcount);
        FCIprog_a_t1(ci0, t1, bcount, stra_id, strb_id,
                     norb, nb, nlinka, clink_indexa);
        for (ir = 0, p0 = 0; ir < totirrep; ir++) {
                dgemm_(&TRANS_N, &TRANS_N, &bcount, dimirrep+ir, dimirrep+ir,
                       &D1,  t1+p0*bcount, &bcount, eri+p0*nnorb+p0, &nnorb,
                       &D0, vt1+p0*bcount, &bcount);
                p0 += dimirrep[ir];
        }
        FCIspread_b_t1(ci1, vt1, bcount, stra_id, strb_id,
                       norb, nb, nlinkb, clink_indexb);
}

void SCIcontract_2e_bbaa_symm(double *eri, double *ci0, double *ci1,
                              int norb, int na, int nb, int nlinka, int nlinkb,
                              int *link_indexa, int *link_indexb,
                              int *dimirrep, int totirrep)
{
        _LinkTrilT *clinka = malloc(sizeof(_LinkTrilT) * nlinka * na);
        _LinkTrilT *clinkb = malloc(sizeof(_LinkTrilT) * nlinkb * nb);
        FCIcompress_link_tril(clinka, link_indexa, na, nlinka);
        FCIcompress_link_tril(clinkb, link_indexb, nb, nlinkb);

#pragma omp parallel
{
        int strk, ib, blen;
        double *t1buf = malloc(sizeof(double) * (STRB_BLKSIZE*norb*(norb+1)+2));
        double *ci1buf = NULL;
        for (ib = 0; ib < nb; ib += STRB_BLKSIZE) {
                blen = MIN(STRB_BLKSIZE, nb-ib);
#pragma omp for schedule(static)
                for (strk = 0; strk < na; strk++) {
                        ctr_bbaa_symm(eri, ci0, ci1, ci1buf, t1buf,
                                      blen, strk, ib, norb, na, nb,
                                      nlinka, nlinkb, clinka, clinkb,
                                      dimirrep, totirrep);
                }
        }
        free(t1buf);
}
        free(clinka);
        free(clinkb);
}

static void ctr_aaaa_symm(double *eri, double *ci0, double *ci1,
                          double *ci1buf, double *t1buf,
                          int bcount, int stra_id, int strb_id,
                          int norb, int na, int nb, int nlinka, int nlinkb,
                          _LinkTrilT *clink_indexa, _LinkTrilT *clink_indexb,
                          int *dimirrep, int totirrep)
{
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * (norb-1) / 2;
        int ir, p0;
        double *t1  = t1buf;
        double *vt1 = t1buf + nnorb*bcount;

        NPdset0(t1, nnorb*bcount);
        FCIprog_a_t1(ci0, t1, bcount, stra_id, strb_id,
                     norb, nb, nlinka, clink_indexa);
        for (ir = 0, p0 = 0; ir < totirrep; ir++) {
                dgemm_(&TRANS_N, &TRANS_N, &bcount, dimirrep+ir, dimirrep+ir,
                       &D1,  t1+p0*bcount, &bcount, eri+p0*nnorb+p0, &nnorb,
                       &D0, vt1+p0*bcount, &bcount);
                p0 += dimirrep[ir];
        }
        FCIspread_a_t1(ci1buf, vt1, bcount, stra_id, 0,
                       norb, bcount, nlinka, clink_indexa);
}

void SCIcontract_2e_aaaa_symm(double *eri, double *ci0, double *ci1,
                              int norb, int na, int nb,
                              int inter_na, int nlinka, int *link_indexa,
                              int *dimirrep, int totirrep)
{
        _LinkTrilT *clinka = malloc(sizeof(_LinkTrilT) * nlinka * inter_na);
        FCIcompress_link_tril(clinka, link_indexa, inter_na, nlinka);
        _LinkTrilT *clinkb = NULL;

        double *ci1bufs[MAX_THREADS];
#pragma omp parallel
{
        int strk, ib, blen;
        double *t1buf = malloc(sizeof(double) * (STRB_BLKSIZE*norb*norb+2));
        double *ci1buf = malloc(sizeof(double) * (na*STRB_BLKSIZE+2));
        ci1bufs[omp_get_thread_num()] = ci1buf;
        for (ib = 0; ib < nb; ib += STRB_BLKSIZE) {
                blen = MIN(STRB_BLKSIZE, nb-ib);
                NPdset0(ci1buf, ((size_t)na) * blen);
#pragma omp for schedule(static)
                for (strk = 0; strk < inter_na; strk++) {
                        ctr_aaaa_symm(eri, ci0, ci1, ci1buf, t1buf,
                                      blen, strk, ib, norb, na, nb,
                                      nlinka, 0, clinka, clinkb,
                                      dimirrep, totirrep);
                }
                NPomp_dsum_reduce_inplace(ci1bufs, blen*na);
#pragma omp master
                FCIaxpy2d(ci1+ib, ci1buf, na, nb, blen);
#pragma omp barrier
        }
        free(ci1buf);
        free(t1buf);
}
        free(clinka);
}
