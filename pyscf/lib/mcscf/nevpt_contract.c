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
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"
#include "fci.h"

#define BLK     48
#define BUFBASE 96

double FCI_t1ci_sf(double *ci0, double *t1, int bcount,
                   int stra_id, int strb_id,
                   int norb, int na, int nb, int nlinka, int nlinkb,
                   _LinkT *clink_indexa, _LinkT *clink_indexb);
double FCI_t2ci_sf(double *ci0, double *t1, int bcount,
                   int stra_id, int strb_id,
                   int norb, int na, int nb, int nlinka, int nlinkb,
                   _LinkT *clink_indexa, _LinkT *clink_indexb);

static void tril2pdm_particle_symm(double *rdm2, double *tbra, double *tket,
                                   int bcount, int ncre, int norb)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        int nnorb = norb * norb;
        int nncre = norb * ncre;

        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nncre, &bcount,
               &D1, tket, &nnorb, tbra, &nnorb, &D1, rdm2, &nnorb);
}


// (df|ce) E^d_f E^a_e|0> = t_ac
void NEVPTkern_dfec_dfae(double *gt2, double *eri, double *t2ket,
                         int bcount, int norb, int na, int nb)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * norb;
        const int n3 = nnorb * norb;
        const size_t n4 = nnorb * nnorb;

#pragma omp parallel
{
        double *cp0, *cp1;
        int i, k, m, n;
        double *t2t = malloc(sizeof(double) * n4); // E^d_fE^a_e with ae transposed
#pragma omp for schedule(dynamic, 4)
        for (k = 0; k < bcount; k++) {
                for (i = 0; i < nnorb; i++) {
                        cp0 = t2ket + k * n4 + i * nnorb;
                        cp1 = t2t + i * nnorb;
                        for (m = 0; m < norb; m++) {
                                for (n = 0; n < norb; n++) {
                                        cp1[n*norb+m] = cp0[m*norb+n];
                                }
                        }
                }
                dgemm_(&TRANS_N, &TRANS_T, &norb, &norb, &n3,
                       &D1, eri, &norb, t2t, &norb,
                       &D0, gt2+nnorb*k, &norb);
        }
        free(t2t);
}
}

// (df|ea) E^e_c E^d_f|0> = t_ac
void NEVPTkern_aedf_ecdf(double *gt2, double *eri, double *t2ket,
                         int bcount, int norb, int na, int nb)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * norb;
        const int n3 = nnorb * norb;
        const size_t n4 = nnorb * nnorb;

#pragma omp parallel
{
        int i, k, m, n;
        double *cp0, *cp1;
        double *t2t = malloc(sizeof(double) * n4);
#pragma omp for schedule(dynamic, 4)
        for (k = 0; k < bcount; k++) {
                for (m = 0; m < norb; m++) {
                        for (n = 0; n < norb; n++) {
                                cp0 = t2ket + k * n4 + (m*norb+n) * nnorb;
                                cp1 = t2t + (n*norb+m) * nnorb;
                                for (i = 0; i < nnorb; i++) {
                                        cp1[i] = cp0[i];
                                }
                        }
                }
                dgemm_(&TRANS_T, &TRANS_N, &norb, &norb, &n3,
                       &D1, t2t, &n3, eri, &n3,
                       &D0, gt2+nnorb*k, &norb);
        }
        free(t2t);
}
}

// (df|ce) E^a_e E^d_f|0> = t_ac
void NEVPTkern_cedf_aedf(double *gt2, double *eri, double *t2ket,
                         int bcount, int norb, int na, int nb)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * norb;
        const int n3 = nnorb * norb;
        const size_t n4 = nnorb * nnorb;

#pragma omp parallel
{
        int k, blen;
#pragma omp for schedule(dynamic, 1)
        for (k = 0; k < bcount; k+=8) {
                blen = MIN(bcount-k, 8) * norb;
                dgemm_(&TRANS_T, &TRANS_N, &norb, &blen, &n3,
                       &D1, eri, &n3, t2ket+n4*k, &n3,
                       &D0, gt2+nnorb*k, &norb);
        }
}
}

// (df|ea) E^d_f E^e_c|0> = t_ac
void NEVPTkern_dfea_dfec(double *gt2, double *eri, double *t2ket,
                         int bcount, int norb, int na, int nb)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * norb;
        const int n3 = nnorb * norb;
        const size_t n4 = nnorb * nnorb;

#pragma omp parallel
{
        int k;
#pragma omp for schedule(dynamic, 4)
        for (k = 0; k < bcount; k++) {
                dgemm_(&TRANS_N, &TRANS_T, &norb, &norb, &n3,
                       &D1, t2ket+n4*k, &norb, eri, &norb,
                       &D0, gt2+nnorb*k, &norb);
        }
}
}

// TODO: NEVPTkern_spin0 stra_id >= strb_id as FCI4pdm_kern_spin0

void NEVPTkern_sf(void (*contract_kernel)(),
                  double *rdm2, double *rdm3, double *eri, double *ci0,
                  int bcount, int stra_id, int strb_id,
                  int norb, int na, int nb, int nlinka, int nlinkb,
                  _LinkT *clink_indexa, _LinkT *clink_indexb)
{
        const int nnorb = norb * norb;
        const int n3 = nnorb * norb;
        const size_t n4 = nnorb * nnorb;
        double *t1ket = malloc(sizeof(double) * nnorb * bcount);
        double *t2ket = malloc(sizeof(double) * n4 * bcount);
        double *gt2 = malloc(sizeof(double) * nnorb * bcount);

        // t2[:,i,j,k,l] = E^i_j E^k_l|ket>
        FCI_t1ci_sf(ci0, t1ket, bcount, stra_id, strb_id,
                    norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);
        FCI_t2ci_sf(ci0, t2ket, bcount, stra_id, strb_id,
                    norb, na, nb, nlinka, nlinkb, clink_indexa, clink_indexb);

        (*contract_kernel)(gt2, eri, t2ket, bcount, norb, na, nb);

#pragma omp parallel
{
        int i, j, k, l, n, ij;
        double *pbra, *pt2;
        double *tbra = malloc(sizeof(double) * nnorb * bcount);
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < nnorb; ij++) { // loop ij for (<ket| E^j_i E^l_k)
                i = ij / norb;
                j = ij - i * norb;

                for (n = 0; n < bcount; n++) {
                        for (k = 0; k <= j; k++) {
                                pbra = tbra + n * nnorb + k*norb;
                                pt2 = t2ket + n * n4 + k*nnorb + ij;
                                for (l = 0; l < norb; l++) {
                                        pbra[l] = pt2[l*n3];
                                }
                        }
                }

                tril2pdm_particle_symm(rdm3+(j*norb+i)*n4, tbra, gt2,
                                       bcount, j+1, norb);
        }
        free(tbra);
}

        // reordering of rdm2 is needed: rdm2.transpose(1,0,2,3)
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &bcount,
               &D1, gt2, &nnorb, t1ket, &nnorb,
               &D1, rdm2, &nnorb);

        free(gt2);
        free(t1ket);
        free(t2ket);
}


void NEVPTcontract(void (*kernel)(),
                   double *rdm2, double *rdm3, double *eri, double *ci0,
                   int norb, int na, int nb, int nlinka, int nlinkb,
                   int *link_indexa, int *link_indexb)
{
        const int nnorb = norb * norb;
        const size_t n4 = nnorb * nnorb;
        int i, j, k, ib, strk, bcount;
        double *pdm2 = malloc(sizeof(double) * n4);
        double *cp1, *cp0;

        _LinkT *clinka = malloc(sizeof(_LinkT) * nlinka * na);
        _LinkT *clinkb = malloc(sizeof(_LinkT) * nlinkb * nb);
        FCIcompress_link(clinka, link_indexa, norb, na, nlinka);
        FCIcompress_link(clinkb, link_indexb, norb, nb, nlinkb);
        NPdset0(pdm2, n4);
        NPdset0(rdm3, n4 * nnorb);

        for (strk = 0; strk < na; strk++) {
                for (ib = 0; ib < nb; ib += BUFBASE) {
                        bcount = MIN(BUFBASE, nb-ib);
                        NEVPTkern_sf(kernel, pdm2, rdm3,
                                     eri, ci0, bcount, strk, ib,
                                     norb, na, nb, nlinka, nlinkb, clinka, clinkb);
                }
        }
        free(clinka);
        free(clinkb);

        for (i = 0; i < norb; i++) {
        for (j = 0; j < norb; j++) {
                cp1 = rdm2 + (i*norb+j) * nnorb;
                cp0 = pdm2 + (j*norb+i) * nnorb;
                for (k = 0; k < nnorb; k++) {
                        cp1[k] = cp0[k];
                }
        } }
        free(pdm2);
}

