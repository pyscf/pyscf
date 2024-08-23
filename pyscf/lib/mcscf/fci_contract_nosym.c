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
 *
 * Paticle permutation symmetry for 2e Hamiltonian only
 * h2e[i,j,k,l] == h2e[k,l,i,j]
 * h2e[i,j,k,l] =/= h2e[j,i,k,l] =/= h2e[i,j,l,k] ...
 */

#include <stdlib.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"
#include "fci.h"
#define CSUMTHR         1e-28
// optimized for 1 MB L2 cache, (16e,16o)
#define STRB_BLKSIZE    120

double FCI_t1ci_sf(double *ci0, double *t1, int bcount,
                   int stra_id, int strb_id,
                   int norb, int na, int nb, int nlinka, int nlinkb,
                   _LinkT *clink_indexa, _LinkT *clink_indexb);


void FCIcontract_a_1e_nosym(double *h1e, double *ci0, double *ci1,
                            int norb, int nstra, int nstrb, int nlinka, int nlinkb,
                            int *link_indexa, int *link_indexb)
{
        int j, k, i, a, sign;
        size_t str0, str1;
        double *pci0, *pci1;
        double tmp;
        _LinkT *tab;
        _LinkT *clink = malloc(sizeof(_LinkT) * nlinka * nstra);
        FCIcompress_link(clink, link_indexa, norb, nstra, nlinka);

        for (str0 = 0; str0 < nstra; str0++) {
                tab = clink + str0 * nlinka;
                for (j = 0; j < nlinka; j++) {
                        a    = EXTRACT_CRE (tab[j]); // propagate from t1 to bra, through a^+ i
                        i    = EXTRACT_DES (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        pci0 = ci0 + str0 * nstrb;
                        pci1 = ci1 + str1 * nstrb;
                        tmp = sign * h1e[a*norb+i];
                        for (k = 0; k < nstrb; k++) {
                                pci1[k] += tmp * pci0[k];
                        }
                }
        }
        free(clink);
}

void FCIcontract_b_1e_nosym(double *h1e, double *ci0, double *ci1,
                            int norb, int nstra, int nstrb, int nlinka, int nlinkb,
                            int *link_indexa, int *link_indexb)
{
        int j, k, i, a, sign;
        size_t str0, str1;
        double *pci1;
        double tmp;
        _LinkT *tab;
        _LinkT *clink = malloc(sizeof(_LinkT) * nlinkb * nstrb);
        FCIcompress_link(clink, link_indexb, norb, nstrb, nlinkb);

        for (str0 = 0; str0 < nstra; str0++) {
                pci1 = ci1 + str0 * nstrb;
                for (k = 0; k < nstrb; k++) {
                        tab = clink + k * nlinkb;
                        tmp = ci0[str0*nstrb+k];
                        for (j = 0; j < nlinkb; j++) {
                                a    = EXTRACT_CRE (tab[j]);
                                i    = EXTRACT_DES (tab[j]);
                                str1 = EXTRACT_ADDR(tab[j]);
                                sign = EXTRACT_SIGN(tab[j]);
                                pci1[str1] += sign * tmp * h1e[a*norb+i];
                        }
                }
        }
        free(clink);
}

static void spread_a_t1(double *ci1, double *t1,
                        int bcount, int stra_id, int strb_id,
                        int norb, int nstrb, int nlinka, _LinkT *clink_indexa)
{
        ci1 += strb_id;
        const int nnorb = norb * norb;
        int j, k, i, a, str1, sign;
        const _LinkT *tab = clink_indexa + stra_id * nlinka;
        double *cp0, *cp1;

        for (j = 0; j < nlinka; j++) {
                a    = EXTRACT_CRE (tab[j]);
                i    = EXTRACT_DES (tab[j]);
                str1 = EXTRACT_ADDR(tab[j]);
                sign = EXTRACT_SIGN(tab[j]);
                cp0 = t1 + a*norb+i; // propagate from t1 to bra, through a^+ i
                cp1 = ci1 + str1*(size_t)nstrb;
                if (sign > 0) {
                        for (k = 0; k < bcount; k++) {
                                cp1[k] += cp0[k*nnorb];
                        }
                } else {
                        for (k = 0; k < bcount; k++) {
                                cp1[k] -= cp0[k*nnorb];
                        }
                }
        }
}

static void spread_b_t1(double *ci1, double *t1,
                        int bcount, int stra_id, int strb_id,
                        int norb, int nstrb, int nlinkb, _LinkT *clink_indexb)
{
        const int nnorb = norb * norb;
        int j, i, a, str0, str1, sign;
        const _LinkT *tab = clink_indexb + strb_id * nlinkb;
        double *pci = ci1 + stra_id * (size_t)nstrb;

        for (str0 = 0; str0 < bcount; str0++) {
                for (j = 0; j < nlinkb; j++) {
                        a    = EXTRACT_CRE (tab[j]);
                        i    = EXTRACT_DES (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        // propagate from t1 to bra, through a^+ i
                        pci[str1] += sign * t1[a*norb+i];
                }
                t1 += nnorb;
                tab += nlinkb;
        }
}

static void ctr_rhf2e_kern(double *eri, double *ci0, double *ci1,
                           double *ci1buf, double *t1, double *vt1,
                           int bcount_for_spread_a, int ncol_ci1buf,
                           int bcount, int stra_id, int strb_id,
                           int norb, int na, int nb, int nlinka, int nlinkb,
                           _LinkT *clink_indexa, _LinkT *clink_indexb)
{
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;

        csum = FCI_t1ci_sf(ci0, t1, bcount, stra_id, strb_id,
                           norb, na, nb, nlinka, nlinkb,
                           clink_indexa, clink_indexb);

        if (csum > CSUMTHR) {
                dgemm_(&TRANS_N, &TRANS_N, &nnorb, &bcount, &nnorb,
                       &D1, eri, &nnorb, t1, &nnorb,
                       &D0, vt1, &nnorb);
                spread_b_t1(ci1, vt1, bcount, stra_id, strb_id,
                            norb, nb, nlinkb, clink_indexb);
                spread_a_t1(ci1buf, vt1, bcount_for_spread_a, stra_id, 0,
                            norb, ncol_ci1buf, nlinka, clink_indexa);
        }
}

static void axpy2d(double *out, double *in, int count, int no, int ni)
{
        int i, j;
        for (i = 0; i < count; i++) {
                for (j = 0; j < ni; j++) {
                        out[i*no+j] += in[i*ni+j];
                }
        }
}

void FCIcontract_2es1(double *eri, double *ci0, double *ci1,
                      int norb, int na, int nb, int nlinka, int nlinkb,
                      int *link_indexa, int *link_indexb)
{
        _LinkT *clinka = malloc(sizeof(_LinkT) * nlinka * na);
        _LinkT *clinkb = malloc(sizeof(_LinkT) * nlinkb * nb);
        FCIcompress_link(clinka, link_indexa, norb, na, nlinka);
        FCIcompress_link(clinkb, link_indexb, norb, nb, nlinkb);

        NPdset0(ci1, ((size_t)na) * nb);

#pragma omp parallel default(none) \
        shared(eri, ci0, ci1, norb, na, nb, nlinka, nlinkb, \
               clinka, clinkb)
{
        int strk, ib, blen;
        double *t1buf = malloc(sizeof(double) * (STRB_BLKSIZE*norb*norb*2+2));
        double *tmp;
        double *t1 = t1buf;
        double *vt1 = t1buf + norb*norb*STRB_BLKSIZE;
        double *ci1buf = malloc(sizeof(double) * (na*STRB_BLKSIZE+2));
        for (ib = 0; ib < nb; ib += STRB_BLKSIZE) {
                blen = MIN(STRB_BLKSIZE, nb-ib);
                NPdset0(ci1buf, ((size_t)na) * blen);
#pragma omp for schedule(static)
                for (strk = 0; strk < na; strk++) {
                        ctr_rhf2e_kern(eri, ci0, ci1, ci1buf, t1, vt1,
                                       blen, blen, blen, strk, ib,
                                       norb, na, nb, nlinka, nlinkb,
                                       clinka, clinkb);
                        // swap buffer for better cache utilization in next task
                        tmp = t1;
                        t1 = vt1;
                        vt1 = tmp;
                }
#pragma omp critical
                axpy2d(ci1+ib, ci1buf, na, nb, blen);
#pragma omp barrier
        }
        free(ci1buf);
        free(t1buf);
}
        free(clinka);
        free(clinkb);
}

