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
#include <assert.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#include "fci.h"
#include "np_helper/np_helper.h"

#define BLK     48
#define BUFBASE 96

double FCI_t1ci_sf(double *ci0, double *t1, int bcount,
                   int stra_id, int strb_id,
                   int norb, int na, int nb, int nlinka, int nlinkb,
                   _LinkT *clink_indexa, _LinkT *clink_indexb);

/*
 * t2[:,i,j,k,l] = E^i_j E^k_l|ci0>
 */
static void rdm4_0b_t2(double *ci0, double *t2,
                       int bcount, int stra_id, int strb_id,
                       int norb, int na, int nb, int nlinka, int nlinkb,
                       _LinkT *clink_indexa, _LinkT *clink_indexb)
{
        const int nnorb = norb * norb;
        const size_t n4 = nnorb * nnorb;
        int i, j, k, l, a, sign, str1;
        double *t1 = malloc(sizeof(double) * nb * nnorb);
        double *pt1, *pt2;
        _LinkT *tab;

        // form t1 which has beta^+ beta |t1> => target stra_id
        FCI_t1ci_sf(ci0, t1, nb, stra_id, 0,
                    norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);

#pragma omp parallel private(i, j, k, l, a, str1, sign, pt1, pt2, tab)
{
#pragma omp for schedule(static, 1) nowait
        for (k = 0; k < bcount; k++) {
                NPdset0(t2+k*n4, n4);
                tab = clink_indexb + (strb_id+k) * nlinkb;
                for (j = 0; j < nlinkb; j++) {
                        a    = EXTRACT_CRE (tab[j]);
                        i    = EXTRACT_DES (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);

                        pt1 = t1 + str1 * nnorb;
                        pt2 = t2 + k * n4 + (i*norb+a)*nnorb;
                        if (sign > 0) {
                                for (l = 0; l < nnorb; l++) {
                                        pt2[l] += pt1[l];
                                }
                        } else {
                                for (l = 0; l < nnorb; l++) {
                                        pt2[l] -= pt1[l];
                                }
                        }
                }
        }
}
        free(t1);
}
/*
 * t2[:,i,j,k,l] = E^i_j E^k_l|ci0>
 */
static void rdm4_a_t2(double *ci0, double *t2,
                      int bcount, int stra_id, int strb_id,
                      int norb, int na, int nb, int nlinka, int nlinkb,
                      _LinkT *clink_indexa, _LinkT *clink_indexb)
{
        const int nnorb = norb * norb;
        const size_t n4 = nnorb * nnorb;
        int i, j, k, l, a, sign, str1;
        double *pt1, *pt2;
        _LinkT *tab = clink_indexa + stra_id * nlinka;

#pragma omp parallel private(i, j, k, l, a, str1, sign, pt1, pt2)
{
        double *t1 = malloc(sizeof(double) * bcount * nnorb);
#pragma omp for schedule(static, 40)
        for (j = 0; j < nlinka; j++) {
                a    = EXTRACT_CRE (tab[j]);
                i    = EXTRACT_DES (tab[j]);
                str1 = EXTRACT_ADDR(tab[j]);
                sign = EXTRACT_SIGN(tab[j]);

                // form t1 which has alpha^+ alpha |t1> => target stra_id (through str1)
                FCI_t1ci_sf(ci0, t1, bcount, str1, strb_id,
                            norb, na, nb, nlinka, nlinkb,
                            clink_indexa, clink_indexb);

                for (k = 0; k < bcount; k++) {
                        pt1 = t1 + k * nnorb;
                        pt2 = t2 + k * n4 + (i*norb+a)*nnorb;
                        if (sign > 0) {
                                for (l = 0; l < nnorb; l++) {
                                        pt2[l] += pt1[l];
                                }
                        } else {
                                for (l = 0; l < nnorb; l++) {
                                        pt2[l] -= pt1[l];
                                }
                        }
                }
        }
        free(t1);
}
}

void FCI_t2ci_sf(double *ci0, double *t2, int bcount,
                 int stra_id, int strb_id,
                 int norb, int na, int nb, int nlinka, int nlinkb,
                 _LinkT *clink_indexa, _LinkT *clink_indexb)
{
        rdm4_0b_t2(ci0, t2, bcount, stra_id, strb_id,
                   norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);
        rdm4_a_t2 (ci0, t2, bcount, stra_id, strb_id,
                   norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);
}

static void tril3pdm_particle_symm(double *rdm3, double *tbra, double *t2ket,
                                   int bcount, int ncre, int norb)
{
        assert(norb <= BLK);
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        int nnorb = norb * norb;
        int n4 = nnorb * nnorb;
        int i, j, k, m, n, blk1;
        int iblk = MIN(BLK/norb, norb);
        int blk = iblk * norb;

        //dgemm_(&TRANS_N, &TRANS_T, &n4, &nncre, &bcount,
        //       &D1, t2ket, &n4, tbra, &nnorb, &D1, rdm3, &n4);
// "upper triangle" F-array[k,j,i], k<=i<=j
        for (j = 0; j < ncre; j++) {
        for (n = 0; n < norb; n++) {
                for (k = 0; k < j+1-iblk; k+=iblk) {
                        m = k * norb;
                        i = m + blk;
                        dgemm_(&TRANS_N, &TRANS_T, &i, &blk, &bcount,
                               &D1, t2ket, &n4, tbra+m, &nnorb,
                               &D1, rdm3+m*n4, &n4);
                }

                m = k * norb;
                i = (j+1) * norb;
                blk1 = i - m;
                dgemm_(&TRANS_N, &TRANS_T, &i, &blk1, &bcount,
                       &D1, t2ket, &n4, tbra+m, &nnorb,
                       &D1, rdm3+m*n4, &n4);
                t2ket += nnorb;
                rdm3 += nnorb;
        } }
}

static void tril2pdm_particle_symm(double *rdm2, double *tbra, double *tket,
                                   int bcount, int ncre, int norb)
{
        assert(norb <= BLK);
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        int nnorb = norb * norb;
        int nncre = norb * ncre;
        int m, n;
        int blk = MIN(BLK/norb, norb) * norb;

        //dgemm_(&TRANS_N, &TRANS_T, &nncre, &nncre, &bcount,
        //       &D1, tket, &nnorb, tbra, &nnorb, &D1, rdm2, &nnorb);
// upper triangle part of F-array
        for (m = 0; m < nncre-blk; m+=blk) {
                n = m + blk;
                dgemm_(&TRANS_N, &TRANS_T, &n, &blk, &bcount,
                       &D1, tket, &nnorb, tbra+m, &nnorb,
                       &D1, rdm2+m*nnorb, &nnorb);
        }
        n = nncre - m;
        dgemm_(&TRANS_N, &TRANS_T, &nncre, &n, &bcount,
               &D1, tket, &nnorb, tbra+m, &nnorb,
               &D1, rdm2+m*nnorb, &nnorb);
}

static void make_rdm12_sf(double *rdm1, double *rdm2,
                          double *bra, double *ket, double *t1bra, double *t1ket,
                          int bcount, int stra_id, int strb_id,
                          int norb, int na, int nb)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int INC1 = 1;
        const double D1 = 1;
        const int nnorb = norb * norb;
        int k, l;
        size_t n;
        double *tbra = malloc(sizeof(double) * nnorb * bcount);
        double *pbra, *pt1;

        for (n = 0; n < bcount; n++) {
                pbra = tbra + n * nnorb;
                pt1 = t1bra + n * nnorb;
                for (k = 0; k < norb; k++) {
                        for (l = 0; l < norb; l++) {
                                pbra[k*norb+l] = pt1[l*norb+k];
                        }
                }
        }
        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &bcount,
               &D1, t1ket, &nnorb, tbra, &nnorb,
               &D1, rdm2, &nnorb);

        dgemv_(&TRANS_N, &nnorb, &bcount, &D1, t1ket, &nnorb,
               bra+stra_id*nb+strb_id, &INC1, &D1, rdm1, &INC1);

        free(tbra);
}

static void make_rdm12_spin0(double *rdm1, double *rdm2,
                             double *bra, double *ket, double *t1bra, double *t1ket,
                             int bcount, int stra_id, int strb_id,
                             int norb, int na, int nb)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const int INC1 = 1;
        const double D1 = 1;
        const int nnorb = norb * norb;
        int k, l;
        size_t n;
        double *tbra = malloc(sizeof(double) * nnorb * bcount);
        double *pbra, *pt1;
        double factor;

        for (n = 0; n < bcount; n++) {
                if (n+strb_id == stra_id) {
                        factor = 1;
                } else {
                        factor = 2;
                }
                pbra = tbra + n * nnorb;
                pt1 = t1bra + n * nnorb;
                for (k = 0; k < norb; k++) {
                        for (l = 0; l < norb; l++) {
                                pbra[k*norb+l] = pt1[l*norb+k] * factor;
                        }
                }
        }
        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &bcount,
               &D1, t1ket, &nnorb, tbra, &nnorb,
               &D1, rdm2, &nnorb);

        dgemv_(&TRANS_N, &nnorb, &bcount, &D1, tbra, &nnorb,
               bra+stra_id*na+strb_id, &INC1, &D1, rdm1, &INC1);

        free(tbra);
}

void FCI4pdm_kern_sf(double *rdm1, double *rdm2, double *rdm3, double *rdm4,
                     double *bra, double *ket,
                     int bcount, int stra_id, int strb_id,
                     int norb, int na, int nb, int nlinka, int nlinkb,
                     _LinkT *clink_indexa, _LinkT *clink_indexb)
{
        const int nnorb = norb * norb;
        const size_t n4 = nnorb * nnorb;
        const size_t n3 = nnorb * norb;
        const size_t n6 = nnorb * nnorb * nnorb;
        int i, j, k, l, ij;
        size_t n;
        double *tbra;
        double *t1bra = malloc(sizeof(double) * nnorb * bcount * 2);
        double *t2bra = malloc(sizeof(double) * n4 * bcount * 2);
        double *t1ket = t1bra + nnorb * bcount;
        double *t2ket = t2bra + n4 * bcount;
        double *pbra, *pt2;

        // t2[:,i,j,k,l] = E^i_j E^k_l|ket>
        FCI_t1ci_sf(bra, t1bra, bcount, stra_id, strb_id,
                    norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);
        FCI_t2ci_sf(bra, t2bra, bcount, stra_id, strb_id,
                    norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);
        if (bra == ket) {
                t1ket = t1bra;
                t2ket = t2bra;
        } else {
                FCI_t1ci_sf(ket, t1ket, bcount, stra_id, strb_id,
                            norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);
                FCI_t2ci_sf(ket, t2ket, bcount, stra_id, strb_id,
                            norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);
        }

#pragma omp parallel private(ij, i, j, k, l, n, tbra, pbra, pt2)
{
        tbra = malloc(sizeof(double) * nnorb * bcount);
#pragma omp for schedule(static, 1) nowait
        for (ij = 0; ij < nnorb; ij++) { // loop ij for (<ket| E^j_i E^l_k)
                for (n = 0; n < bcount; n++) {
                        for (k = 0; k < norb; k++) {
                                pbra = tbra + n * nnorb + k*norb;
                                pt2 = t2bra + n * n4 + k*nnorb + ij;
                                for (l = 0; l < norb; l++) {
                                        pbra[l] = pt2[l*n3];
                                }
                        }
                }

                i = ij / norb;
                j = ij - i * norb;
// contract <bra-of-Eij| with |E^k_l E^m_n ket>
                tril3pdm_particle_symm(rdm4+(j*norb+i)*n6, tbra, t2ket,
                                       bcount, j+1, norb);
// rdm3
                tril2pdm_particle_symm(rdm3+(j*norb+i)*n4, tbra, t1ket,
                                       bcount, j+1, norb);
        }
        free(tbra);
}

        make_rdm12_sf(rdm1, rdm2, bra, ket, t1bra, t1ket,
                      bcount, stra_id, strb_id, norb, na, nb);
        free(t1bra);
        free(t2bra);
}

/*
 * use symmetry ci0[a,b] == ci0[b,a], t2[a,b,...] == t2[b,a,...]
 */
void FCI4pdm_kern_spin0(double *rdm1, double *rdm2, double *rdm3, double *rdm4,
                        double *bra, double *ket,
                        int bcount, int stra_id, int strb_id,
                        int norb, int na, int nb, int nlinka, int nlinkb,
                        _LinkT *clink_indexa, _LinkT *clink_indexb)
{
        int fill1;
        if (strb_id+bcount <= stra_id) {
                fill1 = bcount;
        } else if (stra_id >= strb_id) {
                fill1 = stra_id - strb_id + 1;
        } else {
                return;
        }

        const int nnorb = norb * norb;
        const size_t n4 = nnorb * nnorb;
        const size_t n3 = nnorb * norb;
        const size_t n6 = nnorb * nnorb * nnorb;
        int i, j, k, l, ij;
        size_t n;
        double factor;
        double *tbra;
        double *t1bra = malloc(sizeof(double) * nnorb * fill1 * 2);
        double *t2bra = malloc(sizeof(double) * n4 * fill1 * 2);
        double *t1ket = t1bra + nnorb * fill1;
        double *t2ket = t2bra + n4 * fill1;
        double *pbra, *pt2;

        FCI_t1ci_sf(bra, t1bra, fill1, stra_id, strb_id,
                    norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);
        FCI_t2ci_sf(bra, t2bra, fill1, stra_id, strb_id,
                    norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);
        if (bra == ket) {
                t1ket = t1bra;
                t2ket = t2bra;
        } else {
                FCI_t1ci_sf(ket, t1ket, fill1, stra_id, strb_id,
                            norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);
                FCI_t2ci_sf(ket, t2ket, fill1, stra_id, strb_id,
                            norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);
        }

#pragma omp parallel private(ij, i, j, k, l, n, tbra, pbra, pt2, factor)
{
        tbra = malloc(sizeof(double) * nnorb * fill1);
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nnorb; ij++) { // loop ij for (<ket| E^j_i E^l_k)
                i = ij / norb;
                j = ij - i * norb;
                for (n = 0; n < fill1; n++) {
                        if (n+strb_id == stra_id) {
                                factor = 1;
                        } else {
                                factor = 2;
                        }
                        for (k = 0; k <= j; k++) {
                                pbra = tbra + n * nnorb + k*norb;
                                pt2 = t2bra + n * n4 + k*nnorb + ij;
                                for (l = 0; l < norb; l++) {
                                        pbra[l] = pt2[l*n3] * factor;
                                }
                        }
                }

// contract <bra-of-Eij| with |E^k_l E^m_n ket>
                tril3pdm_particle_symm(rdm4+(j*norb+i)*n6, tbra, t2ket,
                                       fill1, j+1, norb);
// rdm3
                tril2pdm_particle_symm(rdm3+(j*norb+i)*n4, tbra, t1ket,
                                       fill1, j+1, norb);
        }
        free(tbra);
}

        make_rdm12_spin0(rdm1, rdm2, bra, ket, t1bra, t1ket,
                         fill1, stra_id, strb_id, norb, na, nb);
        free(t1bra);
        free(t2bra);
}


/*
 * This function returns incomplete rdm3, rdm4, in which, particle
 * permutation symmetry is assumed.
 * kernel can be FCI4pdm_kern_sf, FCI4pdm_kern_spin0
 */
void FCIrdm4_drv(void (*kernel)(),
                 double *rdm1, double *rdm2, double *rdm3, double *rdm4,
                 double *bra, double *ket,
                 int norb, int na, int nb, int nlinka, int nlinkb,
                 int *link_indexa, int *link_indexb)
{
        const int nnorb = norb * norb;
        const size_t n4 = nnorb * nnorb;
        int ib, strk, bcount;

        _LinkT *clinka = malloc(sizeof(_LinkT) * nlinka * na);
        _LinkT *clinkb = malloc(sizeof(_LinkT) * nlinkb * nb);
        FCIcompress_link(clinka, link_indexa, norb, na, nlinka);
        FCIcompress_link(clinkb, link_indexb, norb, nb, nlinkb);
        NPdset0(rdm1, nnorb);
        NPdset0(rdm2, n4);
        NPdset0(rdm3, n4 * nnorb);
        NPdset0(rdm4, n4 * n4);

        for (strk = 0; strk < na; strk++) {
                for (ib = 0; ib < nb; ib += BUFBASE) {
                        bcount = MIN(BUFBASE, nb-ib);
                        (*kernel)(rdm1, rdm2, rdm3, rdm4,
                                  bra, ket, bcount, strk, ib,
                                  norb, na, nb, nlinka, nlinkb, clinka, clinkb);
                }
        }
        free(clinka);
        free(clinkb);
}


void FCI3pdm_kern_sf(double *rdm1, double *rdm2, double *rdm3,
                     double *bra, double *ket,
                     int bcount, int stra_id, int strb_id,
                     int norb, int na, int nb, int nlinka, int nlinkb,
                     _LinkT *clink_indexa, _LinkT *clink_indexb)
{
        const int nnorb = norb * norb;
        const size_t n4 = nnorb * nnorb;
        const size_t n3 = nnorb * norb;
        int i, j, k, l, ij;
        size_t n;
        double *tbra;
        double *t1bra = malloc(sizeof(double) * nnorb * bcount);
        double *t1ket = malloc(sizeof(double) * nnorb * bcount);
        double *t2bra = malloc(sizeof(double) * n4 * bcount);
        double *pbra, *pt2;

        // t2[:,i,j,k,l] = E^i_j E^k_l|ket>
        FCI_t1ci_sf(bra, t1bra, bcount, stra_id, strb_id,
                    norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);
        FCI_t2ci_sf(bra, t2bra, bcount, stra_id, strb_id,
                    norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);
        FCI_t1ci_sf(ket, t1ket, bcount, stra_id, strb_id,
                    norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);

#pragma omp parallel private(ij, i, j, k, l, n, tbra, pbra, pt2)
{
        tbra = malloc(sizeof(double) * nnorb * bcount);
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nnorb; ij++) { // loop ij for (<ket| E^j_i E^l_k)
                for (n = 0; n < bcount; n++) {
                        pbra = tbra + n * nnorb;
                        pt2 = t2bra + n * n4 + ij;
                        for (k = 0; k < norb; k++) {
                                for (l = 0; l < norb; l++) {
                                        pbra[k*norb+l] = pt2[l*n3+k*nnorb];
                                }
                        }
                }

                i = ij / norb;
                j = ij - i * norb;
                tril2pdm_particle_symm(rdm3+(j*norb+i)*n4, tbra, t1ket,
                                       bcount, j+1, norb);
        }
        free(tbra);
}

        make_rdm12_sf(rdm1, rdm2, bra, ket, t1bra, t1ket,
                      bcount, stra_id, strb_id, norb, na, nb);
        free(t1bra);
        free(t1ket);
        free(t2bra);
}

/*
 * use symmetry ci0[a,b] == ci0[b,a], t2[a,b,...] == t2[b,a,...]
 */
void FCI3pdm_kern_spin0(double *rdm1, double *rdm2, double *rdm3,
                        double *bra, double *ket,
                        int bcount, int stra_id, int strb_id,
                        int norb, int na, int nb, int nlinka, int nlinkb,
                        _LinkT *clink_indexa, _LinkT *clink_indexb)
{
        int fill1;
        if (strb_id+bcount <= stra_id) {
                fill1 = bcount;
        } else if (stra_id >= strb_id) {
                fill1 = stra_id - strb_id + 1;
        } else {
                return;
        }

        const int nnorb = norb * norb;
        const size_t n4 = nnorb * nnorb;
        const size_t n3 = nnorb * norb;
        int i, j, k, l, ij;
        size_t n;
        double factor;
        double *tbra;
        double *t1bra = malloc(sizeof(double) * nnorb * fill1);
        double *t1ket = malloc(sizeof(double) * nnorb * fill1);
        double *t2bra = malloc(sizeof(double) * n4 * fill1);
        double *pbra, *pt2;

        // t2[:,i,j,k,l] = E^i_j E^k_l|ket>
        FCI_t2ci_sf(bra, t2bra, fill1, stra_id, strb_id,
                    norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);
        FCI_t1ci_sf(bra, t1bra, fill1, stra_id, strb_id,
                    norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);
        FCI_t1ci_sf(ket, t1ket, fill1, stra_id, strb_id,
                    norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);

#pragma omp parallel private(ij, i, j, k, l, n, tbra, pbra, pt2, factor)
{
        tbra = malloc(sizeof(double) * nnorb * fill1);
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nnorb; ij++) { // loop ij for (<ket| E^j_i E^l_k)
                i = ij / norb;
                j = ij - i * norb;
                for (n = 0; n < fill1; n++) {
                        if (n+strb_id == stra_id) {
                                factor = 1;
                        } else {
                                factor = 2;
                        }
                        for (k = 0; k <= j; k++) {
                                pbra = tbra + n * nnorb + k*norb;
                                pt2 = t2bra + n * n4 + k*nnorb + ij;
                                for (l = 0; l < norb; l++) {
                                        pbra[l] = pt2[l*n3] * factor;
                                }
                        }
                }

                tril2pdm_particle_symm(rdm3+(j*norb+i)*n4, tbra, t1ket,
                                       fill1, j+1, norb);
        }
        free(tbra);
}
        make_rdm12_spin0(rdm1, rdm2, bra, ket, t1bra, t1ket,
                         fill1, stra_id, strb_id, norb, na, nb);
        free(t1bra);
        free(t1ket);
        free(t2bra);
}

/*
 * This function returns incomplete rdm3, in which, particle
 * permutation symmetry is assumed.
 * kernel can be FCI3pdm_kern_ms0, FCI3pdm_kern_spin0
 */
void FCIrdm3_drv(void (*kernel)(),
                 double *rdm1, double *rdm2, double *rdm3,
                 double *bra, double *ket,
                 int norb, int na, int nb, int nlinka, int nlinkb,
                 int *link_indexa, int *link_indexb)
{
        const int nnorb = norb * norb;
        const size_t n4 = nnorb * nnorb;
        int ib, strk, bcount;

        _LinkT *clinka = malloc(sizeof(_LinkT) * nlinka * na);
        _LinkT *clinkb = malloc(sizeof(_LinkT) * nlinkb * nb);
        FCIcompress_link(clinka, link_indexa, norb, na, nlinka);
        FCIcompress_link(clinkb, link_indexb, norb, nb, nlinkb);
        NPdset0(rdm1, nnorb);
        NPdset0(rdm2, n4);
        NPdset0(rdm3, n4 * nnorb);

        for (strk = 0; strk < na; strk++) {
                for (ib = 0; ib < nb; ib += BUFBASE) {
                        bcount = MIN(BUFBASE, nb-ib);
                        (*kernel)(rdm1, rdm2, rdm3,
                                  bra, ket, bcount, strk, ib,
                                  norb, na, nb, nlinka, nlinkb, clinka, clinkb);
                }
        }
        free(clinka);
        free(clinkb);
}

