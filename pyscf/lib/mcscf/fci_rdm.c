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
#include "fci.h"
#include "np_helper/np_helper.h"

#define CSUMTHR         1e-28
#define BUFBASE         96
#define SQRT2           1.4142135623730950488

#define BRAKETSYM       1
#define PARTICLESYM     2

/*
 * i is the index of the annihilation operator, a is the index of
 * creation operator.  t1[I,i*norb+a] because it represents that
 * starting from the intermediate I, removing i and creating a leads to
 * determinant of str1
 */

double FCIrdm2_a_t1ci(double *ci0, double *t1,
                      int bcount, int stra_id, int strb_id,
                      int norb, int nstrb, int nlinka, _LinkT *clink_indexa)
{
        ci0 += strb_id;
        const int nnorb = norb * norb;
        int i, j, k, a, sign;
        size_t str1;
        const _LinkT *tab = clink_indexa + stra_id * nlinka;
        double *pt1, *pci;
        double csum = 0;

        for (j = 0; j < nlinka; j++) {
                a    = EXTRACT_CRE (tab[j]);
                i    = EXTRACT_DES (tab[j]);
                str1 = EXTRACT_ADDR(tab[j]);
                sign = EXTRACT_SIGN(tab[j]);
                pci = ci0 + str1*nstrb;
                pt1 = t1 + i*norb+a;
                if (sign == 0) {
                        break;
                } else if (sign > 0) {
                        for (k = 0; k < bcount; k++) {
                                pt1[k*nnorb] += pci[k];
                                csum += pci[k] * pci[k];
                        }
                } else {
                        for (k = 0; k < bcount; k++) {
                                pt1[k*nnorb] -= pci[k];
                                csum += pci[k] * pci[k];
                        }
                }
        }
        return csum;
}

double FCIrdm2_b_t1ci(double *ci0, double *t1,
                      int bcount, int stra_id, int strb_id,
                      int norb, int nstrb, int nlinkb, _LinkT *clink_indexb)
{
        const int nnorb = norb * norb;
        int i, j, a, str0, str1, sign;
        const _LinkT *tab = clink_indexb + strb_id * nlinkb;
        double *pci = ci0 + stra_id*(size_t)nstrb;
        double csum = 0;

        for (str0 = 0; str0 < bcount; str0++) {
                for (j = 0; j < nlinkb; j++) {
                        a    = EXTRACT_CRE (tab[j]);
                        i    = EXTRACT_DES (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        if (sign == 0) {
                                break;
                        } else {
                                t1[i*norb+a] += sign * pci[str1];
                                csum += pci[str1] * pci[str1];
                        }
                }
                t1 += nnorb;
                tab += nlinkb;
        }
        return csum;
}
double FCIrdm2_0b_t1ci(double *ci0, double *t1,
                       int bcount, int stra_id, int strb_id,
                       int norb, int nstrb, int nlinkb, _LinkT *clink_indexb)
{
        const int nnorb = norb * norb;
        int i, j, a, str0, str1, sign;
        const _LinkT *tab = clink_indexb + strb_id * nlinkb;
        double *pci = ci0 + stra_id*(size_t)nstrb;
        double csum = 0;

        for (str0 = 0; str0 < bcount; str0++) {
                NPdset0(t1, nnorb);
                for (j = 0; j < nlinkb; j++) {
                        a    = EXTRACT_CRE (tab[j]);
                        i    = EXTRACT_DES (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        t1[i*norb+a] += sign * pci[str1];
                        csum += pci[str1] * pci[str1];
                }
                t1 += nnorb;
                tab += nlinkb;
        }
        return csum;
}

/* spin free E^i_j | ci0 > */
double FCI_t1ci_sf(double *ci0, double *t1, int bcount,
                   int stra_id, int strb_id,
                   int norb, int na, int nb, int nlinka, int nlinkb,
                   _LinkT *clink_indexa, _LinkT *clink_indexb)
{
        double csum;
        csum = FCIrdm2_0b_t1ci(ci0, t1, bcount, stra_id, strb_id,
                               norb, nb, nlinkb, clink_indexb)
             + FCIrdm2_a_t1ci (ci0, t1, bcount, stra_id, strb_id,
                               norb, nb, nlinka, clink_indexa);
        return csum;
}

static void tril_particle_symm(double *rdm2, double *tbra, double *tket,
                               int bcount, int norb,
                               double alpha, double beta)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        int nnorb = norb * norb;
        int i, j, k, m, n;
        int blk = MIN(((int)(48/norb))*norb, nnorb);
        double *buf = malloc(sizeof(double) * nnorb*bcount);
        double *p1;

        for (n = 0, k = 0; k < bcount; k++) {
                p1 = tbra + k * nnorb;
                for (i = 0; i < norb; i++) {
                for (j = 0; j < norb; j++, n++) {
                        buf[n] = p1[j*norb+i];
                } }
        }

//        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &bcount,
//               &alpha, tket, &nnorb, buf, &nnorb, &beta, rdm2, &nnorb);
        for (m = 0; m < nnorb-blk; m+=blk) {
                n = nnorb - m;
                dgemm_(&TRANS_N, &TRANS_T, &blk, &n, &bcount,
                       &alpha, tket+m, &nnorb, buf+m, &nnorb,
                       &beta, rdm2+m*nnorb+m, &nnorb);
        }
        n = nnorb - m;
        dgemm_(&TRANS_N, &TRANS_T, &n, &n, &bcount,
               &alpha, tket+m, &nnorb, buf+m, &nnorb,
               &beta, rdm2+m*nnorb+m, &nnorb);

        free(buf);
}

static void _transpose_jikl(double *dm2, int norb)
{
        int nnorb = norb * norb;
        int i, j;
        double *p0, *p1;
        double *tmp = malloc(sizeof(double)*nnorb*nnorb);
        NPdcopy(tmp, dm2, nnorb*nnorb);
        for (i = 0; i < norb; i++) {
                for (j = 0; j < norb; j++) {
                        p0 = tmp + (j*norb+i) * nnorb;
                        p1 = dm2 + (i*norb+j) * nnorb;
                        NPdcopy(p1, p0, nnorb);
                }
        }
        free(tmp);
}

/*
 * Note! The returned rdm2 from FCI*kern* function corresponds to
 *      [(p^+ q on <bra|) r^+ s] = [p q^+ r^+ s]
 * in FCIrdm12kern_sf, FCIrdm12kern_spin0, FCIrdm12kern_a, ...
 * t1 is calculated as |K> = i^+ j|0>. by doing dot(t1.T,t1) to get "rdm2",
 * The ket part (k^+ l|0>) will generate the correct order for the last
 * two indices kl of rdm2(i,j,k,l), But the bra part (i^+ j|0>)^dagger
 * will generate an order of (i,j), which is identical to call a bra of
 * (<0|i j^+).  The so-obtained rdm2(i,j,k,l) corresponds to the
 * operator sequence i j^+ k^+ l. 
 *
 * symm = 1: symmetrizes the 1pdm, and 2pdm.  This is true only if bra == ket,
 * and the operators on bra are equivalent to those on ket, like
 *      FCIrdm12kern_a, FCIrdm12kern_b, FCIrdm12kern_sf, FCIrdm12kern_spin0
 * sym = 2: consider the particle permutation symmetry:
 *      E^j_l E^i_k = E^i_k E^j_l - \delta_{il}E^j_k + \dleta_{jk}E^i_l
 */
void FCIrdm12_drv(void (*dm12kernel)(),
                  double *rdm1, double *rdm2, double *bra, double *ket,
                  int norb, int na, int nb, int nlinka, int nlinkb,
                  int *link_indexa, int *link_indexb, int symm)
{
        const int nnorb = norb * norb;
        int strk, i, j, k, l, ib, blen;
        double *pdm1, *pdm2;
        NPdset0(rdm1, nnorb);
        NPdset0(rdm2, nnorb*nnorb);

        _LinkT *clinka = malloc(sizeof(_LinkT) * nlinka * na);
        _LinkT *clinkb = malloc(sizeof(_LinkT) * nlinkb * nb);
        FCIcompress_link(clinka, link_indexa, norb, na, nlinka);
        FCIcompress_link(clinkb, link_indexb, norb, nb, nlinkb);

#pragma omp parallel private(strk, i, ib, blen, pdm1, pdm2)
{
        pdm1 = calloc(nnorb+2, sizeof(double));
        pdm2 = calloc(nnorb*nnorb+2, sizeof(double));
#pragma omp for schedule(dynamic, 40)
        for (strk = 0; strk < na; strk++) {
                for (ib = 0; ib < nb; ib += BUFBASE) {
                        blen = MIN(BUFBASE, nb-ib);
                        (*dm12kernel)(pdm1, pdm2, bra, ket, blen, strk, ib,
                                      norb, na, nb, nlinka, nlinkb,
                                      clinka, clinkb, symm);
                }
        }
#pragma omp critical
{
        for (i = 0; i < nnorb; i++) {
                rdm1[i] += pdm1[i];
        }
        for (i = 0; i < nnorb*nnorb; i++) {
                rdm2[i] += pdm2[i];
        }
}
        free(pdm1);
        free(pdm2);
}
        free(clinka);
        free(clinkb);
        switch (symm) {
        case BRAKETSYM:
                for (i = 0; i < norb; i++) {
                        for (j = 0; j < i; j++) {
                                rdm1[j*norb+i] = rdm1[i*norb+j];
                        }
                }
                for (i = 0; i < nnorb; i++) {
                        for (j = 0; j < i; j++) {
                                rdm2[j*nnorb+i] = rdm2[i*nnorb+j];
                        }
                }
                _transpose_jikl(rdm2, norb);
                break;
        case PARTICLESYM:
// right 2pdm order is required here,  which transposes the cre/des on bra
                for (i = 0; i < norb; i++) {
                for (j = 0; j < i; j++) {
                        pdm1 = rdm2 + (i*nnorb+j)*norb;
                        pdm2 = rdm2 + (j*nnorb+i)*norb;
                        for (k = 0; k < norb; k++) {
                        for (l = 0; l < norb; l++) {
                                pdm2[l*nnorb+k] = pdm1[k*nnorb+l];
                        } }
// E^j_lE^i_k = E^i_kE^j_l + \delta_{il}E^j_k - \dleta_{jk}E^i_l
                        for (k = 0; k < norb; k++) {
                                pdm2[i*nnorb+k] += rdm1[j*norb+k];
                                pdm2[k*nnorb+j] -= rdm1[i*norb+k];
                        }
                } }
                break;
        default:
                _transpose_jikl(rdm2, norb);
        }
}

void FCIrdm12kern_sf(double *rdm1, double *rdm2, double *bra, double *ket,
                     int bcount, int stra_id, int strb_id,
                     int norb, int na, int nb, int nlinka, int nlinkb,
                     _LinkT *clink_indexa, _LinkT *clink_indexb, int symm)
{
        const int INC1 = 1;
        const char UP = 'U';
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;
        double *buf = malloc(sizeof(double) * nnorb * bcount);

        csum = FCI_t1ci_sf(ket, buf, bcount, stra_id, strb_id,
                           norb, na, nb, nlinka, nlinkb,
                           clink_indexa, clink_indexb);
        if (csum > CSUMTHR) {
                dgemv_(&TRANS_N, &nnorb, &bcount, &D1, buf, &nnorb,
                       ket+stra_id*nb+strb_id, &INC1, &D1, rdm1, &INC1);
                switch (symm) {
                case BRAKETSYM:
                        dsyrk_(&UP, &TRANS_N, &nnorb, &bcount,
                               &D1, buf, &nnorb, &D1, rdm2, &nnorb);
                        break;
                case PARTICLESYM:
                        tril_particle_symm(rdm2, buf, buf, bcount, norb, 1, 1);
                        break;
                default:
                        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &bcount,
                               &D1, buf, &nnorb, buf, &nnorb,
                               &D1, rdm2, &nnorb);
                }
        }
        free(buf);
}

/*
 * _spin0 assumes the strict symmetry on alpha and beta electrons
 */
void FCIrdm12kern_spin0(double *rdm1, double *rdm2, double *bra, double *ket,
                        int bcount, int stra_id, int strb_id,
                        int norb, int na, int nb, int nlinka, int nlinkb,
                        _LinkT *clink_indexa, _LinkT *clink_indexb, int symm)
{
        if (stra_id < strb_id) {
                return;
        }
        const int INC1 = 1;
        const char UP = 'U';
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const double D2 = 2;
        const int nnorb = norb * norb;
        int fill0, fill1, i;
        double csum = 0;
        double *buf = calloc(nnorb * na, sizeof(double));

        if (strb_id+bcount <= stra_id) {
                fill0 = bcount;
                fill1 = bcount;
                csum = FCIrdm2_b_t1ci(ket, buf, fill0, stra_id, strb_id,
                                      norb, na, nlinka, clink_indexa)
                     + FCIrdm2_a_t1ci(ket, buf, fill1, stra_id, strb_id,
                                      norb, na, nlinka, clink_indexa);
        } else if (stra_id >= strb_id) {
                fill0 = stra_id - strb_id;
                fill1 = stra_id - strb_id + 1;
                csum = FCIrdm2_b_t1ci(ket, buf, fill0, stra_id, strb_id,
                                      norb, na, nlinka, clink_indexa)
                     + FCIrdm2_a_t1ci(ket, buf, fill1, stra_id, strb_id,
                                      norb, na, nlinka, clink_indexa);
        }
        if (csum > CSUMTHR) {
                dgemv_(&TRANS_N, &nnorb, &fill1, &D2, buf, &nnorb,
                       ket+stra_id*na+strb_id, &INC1, &D1, rdm1, &INC1);

                for (i = fill0*nnorb; i < fill1*nnorb; i++) {
                        buf[i] *= SQRT2;
                }
                switch (symm) {
                case BRAKETSYM:
                        dsyrk_(&UP, &TRANS_N, &nnorb, &fill1,
                               &D2, buf, &nnorb, &D1, rdm2, &nnorb);
                        break;
                case PARTICLESYM:
                        tril_particle_symm(rdm2, buf, buf, fill1, norb, D2, D1);
                        break;
                default:
                        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &fill1,
                               &D2, buf, &nnorb, buf, &nnorb,
                               &D1, rdm2, &nnorb);
                }
        }
        free(buf);
}



/*
 * ***********************************************
 * transition density matrix, spin free
 */
void FCItdm12kern_sf(double *tdm1, double *tdm2, double *bra, double *ket,
                     int bcount, int stra_id, int strb_id,
                     int norb, int na, int nb, int nlinka, int nlinkb,
                     _LinkT *clink_indexa, _LinkT *clink_indexb, int symm)
{
        const int INC1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;
        double *buf0 = malloc(sizeof(double) * nnorb*bcount);
        double *buf1 = malloc(sizeof(double) * nnorb*bcount);

        csum = FCI_t1ci_sf(bra, buf1, bcount, stra_id, strb_id,
                           norb, na, nb, nlinka, nlinkb,
                           clink_indexa, clink_indexb);
        if (csum < CSUMTHR) { goto _normal_end; }
        csum = FCI_t1ci_sf(ket, buf0, bcount, stra_id, strb_id,
                           norb, na, nb, nlinka, nlinkb,
                           clink_indexa, clink_indexb);
        if (csum < CSUMTHR) { goto _normal_end; }
        dgemv_(&TRANS_N, &nnorb, &bcount, &D1, buf0, &nnorb,
               bra+stra_id*nb+strb_id, &INC1, &D1, tdm1, &INC1);
        switch (symm) {
        case PARTICLESYM:
                tril_particle_symm(tdm2, buf1, buf0, bcount, norb, D1, D1);
                break;
        default:
                dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &bcount,
                       &D1, buf0, &nnorb, buf1, &nnorb,
                       &D1, tdm2, &nnorb);
        }
_normal_end:
        free(buf0);
        free(buf1);
}


/*
 * ***********************************************
 * 2pdm kernel for alpha^i alpha_j | ci0 >
 * ***********************************************
 */
void FCIrdm12kern_a(double *rdm1, double *rdm2, double *bra, double *ket,
                    int bcount, int stra_id, int strb_id,
                    int norb, int na, int nb, int nlinka, int nlinkb,
                    _LinkT *clink_indexa, _LinkT *clink_indexb, int symm)
{
        const int INC1 = 1;
        const char UP = 'U';
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;
        double *buf = calloc(nnorb*bcount, sizeof(double));

        csum = FCIrdm2_a_t1ci(ket, buf, bcount, stra_id, strb_id,
                              norb, nb, nlinka, clink_indexa);
        if (csum > CSUMTHR) {
                dgemv_(&TRANS_N, &nnorb, &bcount, &D1, buf, &nnorb,
                       ket+stra_id*nb+strb_id, &INC1, &D1, rdm1, &INC1);
                switch (symm) {
                case BRAKETSYM:
                        dsyrk_(&UP, &TRANS_N, &nnorb, &bcount,
                               &D1, buf, &nnorb, &D1, rdm2, &nnorb);
                        break;
                case PARTICLESYM:
                        tril_particle_symm(rdm2, buf, buf, bcount, norb, 1, 1);
                        break;
                default:
                        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &bcount,
                               &D1, buf, &nnorb, buf, &nnorb,
                               &D1, rdm2, &nnorb);
                }
        }
        free(buf);
}
/*
 * 2pdm kernel for  beta^i beta_j | ci0 >
 */
void FCIrdm12kern_b(double *rdm1, double *rdm2, double *bra, double *ket,
                    int bcount, int stra_id, int strb_id,
                    int norb, int na, int nb, int nlinka, int nlinkb,
                    _LinkT *clink_indexa, _LinkT *clink_indexb, int symm)
{
        const int INC1 = 1;
        const char UP = 'U';
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;
        double *buf = calloc(nnorb*bcount, sizeof(double));

        csum = FCIrdm2_b_t1ci(ket, buf, bcount, stra_id, strb_id,
                              norb, nb, nlinkb, clink_indexb);
        if (csum > CSUMTHR) {
                dgemv_(&TRANS_N, &nnorb, &bcount, &D1, buf, &nnorb,
                       ket+stra_id*nb+strb_id, &INC1, &D1, rdm1, &INC1);
                switch (symm) {
                case BRAKETSYM:
                        dsyrk_(&UP, &TRANS_N, &nnorb, &bcount,
                               &D1, buf, &nnorb, &D1, rdm2, &nnorb);
                        break;
                case PARTICLESYM:
                        tril_particle_symm(rdm2, buf, buf, bcount, norb, 1, 1);
                        break;
                default:
                        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &bcount,
                               &D1, buf, &nnorb, buf, &nnorb,
                               &D1, rdm2, &nnorb);
                }
        }
        free(buf);
}

void FCItdm12kern_a(double *tdm1, double *tdm2, double *bra, double *ket,
                    int bcount, int stra_id, int strb_id,
                    int norb, int na, int nb, int nlinka, int nlinkb,
                    _LinkT *clink_indexa, _LinkT *clink_indexb, int symm)
{
        const int INC1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;
        double *buf0 = calloc(nnorb*bcount, sizeof(double));
        double *buf1 = calloc(nnorb*bcount, sizeof(double));

        csum = FCIrdm2_a_t1ci(bra, buf1, bcount, stra_id, strb_id,
                              norb, nb, nlinka, clink_indexa);
        if (csum < CSUMTHR) { goto _normal_end; }
        csum = FCIrdm2_a_t1ci(ket, buf0, bcount, stra_id, strb_id,
                              norb, nb, nlinka, clink_indexa);
        if (csum < CSUMTHR) { goto _normal_end; }
        dgemv_(&TRANS_N, &nnorb, &bcount, &D1, buf0, &nnorb,
               bra+stra_id*nb+strb_id, &INC1, &D1, tdm1, &INC1);
        switch (symm) {
        case PARTICLESYM:
                tril_particle_symm(tdm2, buf1, buf0, bcount, norb, D1, D1);
                break;
        default:
                dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &bcount,
                       &D1, buf0, &nnorb, buf1, &nnorb, &D1, tdm2, &nnorb);
        }
_normal_end:
        free(buf0);
        free(buf1);
}

void FCItdm12kern_b(double *tdm1, double *tdm2, double *bra, double *ket,
                    int bcount, int stra_id, int strb_id,
                    int norb, int na, int nb, int nlinka, int nlinkb,
                    _LinkT *clink_indexa, _LinkT *clink_indexb, int symm)
{
        const int INC1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;
        double *buf0 = calloc(nnorb*bcount, sizeof(double));
        double *buf1 = calloc(nnorb*bcount, sizeof(double));

        csum = FCIrdm2_b_t1ci(bra, buf1, bcount, stra_id, strb_id,
                              norb, nb, nlinkb, clink_indexb);
        if (csum < CSUMTHR) { goto _normal_end; }
        csum = FCIrdm2_b_t1ci(ket, buf0, bcount, stra_id, strb_id,
                              norb, nb, nlinkb, clink_indexb);
        if (csum < CSUMTHR) { goto _normal_end; }
        dgemv_(&TRANS_N, &nnorb, &bcount, &D1, buf0, &nnorb,
               bra+stra_id*nb+strb_id, &INC1, &D1, tdm1, &INC1);
        switch (symm) {
        case PARTICLESYM:
                tril_particle_symm(tdm2, buf1, buf0, bcount, norb, D1, D1);
                break;
        default:
                dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &bcount,
                       &D1, buf0, &nnorb, buf1, &nnorb, &D1, tdm2, &nnorb);
        }
_normal_end:
        free(buf0);
        free(buf1);
}

void FCItdm12kern_ab(double *tdm1, double *tdm2, double *bra, double *ket,
                     int bcount, int stra_id, int strb_id,
                     int norb, int na, int nb, int nlinka, int nlinkb,
                     _LinkT *clink_indexa, _LinkT *clink_indexb, int symm)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;
        double *bufb = calloc(nnorb*bcount, sizeof(double));
        double *bufa = calloc(nnorb*bcount, sizeof(double));

        csum = FCIrdm2_a_t1ci(bra, bufa, bcount, stra_id, strb_id,
                              norb, nb, nlinka, clink_indexa);
        if (csum < CSUMTHR) { goto _normal_end; }
        csum = FCIrdm2_b_t1ci(ket, bufb, bcount, stra_id, strb_id,
                              norb, nb, nlinkb, clink_indexb);
        if (csum < CSUMTHR) { goto _normal_end; }
// no particle symmetry between alpha-alpha-beta-beta 2pdm
        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &bcount,
               &D1, bufb, &nnorb, bufa, &nnorb, &D1, tdm2, &nnorb);
_normal_end:
        free(bufb);
        free(bufa);
}

/*
 * ***********************************************
 * 1-pdm
 * ***********************************************
 */
void FCItrans_rdm1a(double *rdm1, double *bra, double *ket,
                    int norb, int na, int nb, int nlinka, int nlinkb,
                    int *link_indexa, int *link_indexb)
{
        int i, a, j, k, str0, str1, sign;
        double *pket, *pbra;
        _LinkT *tab;
        _LinkT *clink = malloc(sizeof(_LinkT) * nlinka * na);
        FCIcompress_link(clink, link_indexa, norb, na, nlinka);

        NPdset0(rdm1, norb*norb);

        for (str0 = 0; str0 < na; str0++) {
                tab = clink + str0 * nlinka;
                pket = ket + str0 * nb;
                for (j = 0; j < nlinka; j++) {
                        a    = EXTRACT_CRE (tab[j]);
                        i    = EXTRACT_DES (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        pbra = bra + str1 * nb;
                        if (sign == 0) {
                                break;
                        } else if (sign > 0) {
                                for (k = 0; k < nb; k++) {
                                        rdm1[a*norb+i] += pbra[k]*pket[k];
                                }
                        } else {
                                for (k = 0; k < nb; k++) {
                                        rdm1[a*norb+i] -= pbra[k]*pket[k];
                                }
                        }
                }
        }
        free(clink);
}

void FCItrans_rdm1b(double *rdm1, double *bra, double *ket,
                    int norb, int na, int nb, int nlinka, int nlinkb,
                    int *link_indexa, int *link_indexb)
{
        int i, a, j, k, str0, str1, sign;
        double *pket, *pbra;
        double tmp;
        _LinkT *tab;
        _LinkT *clink = malloc(sizeof(_LinkT) * nlinkb * nb);
        FCIcompress_link(clink, link_indexb, norb, nb, nlinkb);

        NPdset0(rdm1, norb*norb);

        for (str0 = 0; str0 < na; str0++) {
                pbra = bra + str0 * nb;
                pket = ket + str0 * nb;
                for (k = 0; k < nb; k++) {
                        tab = clink + k * nlinkb;
                        tmp = pket[k];
                        for (j = 0; j < nlinkb; j++) {
                                a    = EXTRACT_CRE (tab[j]);
                                i    = EXTRACT_DES (tab[j]);
                                str1 = EXTRACT_ADDR(tab[j]);
                                sign = EXTRACT_SIGN(tab[j]);
                                if (sign == 0) {
                                        break;
                                } else {
                                        rdm1[a*norb+i] += sign*pbra[str1]*tmp;
                                }
                        }
                }
        }
        free(clink);
}

/*
 * make_rdm1 assumed the hermitian of density matrix
 */
void FCImake_rdm1a(double *rdm1, double *cibra, double *ciket,
                   int norb, int na, int nb, int nlinka, int nlinkb,
                   int *link_indexa, int *link_indexb)
{
        int i, a, j, k, str0, str1, sign;
        double *pci0, *pci1;
        double *ci0 = ciket;
        _LinkT *tab;
        _LinkT *clink = malloc(sizeof(_LinkT) * nlinka * na);
        FCIcompress_link(clink, link_indexa, norb, na, nlinka);

        NPdset0(rdm1, norb*norb);

        for (str0 = 0; str0 < na; str0++) {
                tab = clink + str0 * nlinka;
                pci0 = ci0 + str0 * nb;
                for (j = 0; j < nlinka; j++) {
                        a    = EXTRACT_CRE (tab[j]);
                        i    = EXTRACT_DES (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        pci1 = ci0 + str1 * nb;
                        if (a >= i) {
                                if (sign == 0) {
                                        break;
                                } else if (sign > 0) {
                                        for (k = 0; k < nb; k++) {
                                                rdm1[a*norb+i] += pci0[k]*pci1[k];
                                        }
                                } else {
                                        for (k = 0; k < nb; k++) {
                                                rdm1[a*norb+i] -= pci0[k]*pci1[k];
                                        }
                                }
                        }
                }
        }
        for (j = 0; j < norb; j++) {
                for (k = 0; k < j; k++) {
                        rdm1[k*norb+j] = rdm1[j*norb+k];
                }
        }
        free(clink);
}

void FCImake_rdm1b(double *rdm1, double *cibra, double *ciket,
                   int norb, int na, int nb, int nlinka, int nlinkb,
                   int *link_indexa, int *link_indexb)
{
        int i, a, j, k, str0, str1, sign;
        double *pci0;
        double *ci0 = ciket;
        double tmp;
        _LinkT *tab;
        _LinkT *clink = malloc(sizeof(_LinkT) * nlinkb * nb);
        FCIcompress_link(clink, link_indexb, norb, nb, nlinkb);

        NPdset0(rdm1, norb*norb);

        for (str0 = 0; str0 < na; str0++) {
                pci0 = ci0 + str0 * nb;
                for (k = 0; k < nb; k++) {
                        tab = clink + k * nlinkb;
                        tmp = pci0[k];
                        for (j = 0; j < nlinkb; j++) {
                                a    = EXTRACT_CRE (tab[j]);
                                i    = EXTRACT_DES (tab[j]);
                                str1 = EXTRACT_ADDR(tab[j]);
                                sign = EXTRACT_SIGN(tab[j]);
                                if (a >= i) {
                                        if (sign == 0) {
                                                break;
                                        } else if (sign > 0) {
                                                rdm1[a*norb+i] += pci0[str1]*tmp;
                                        } else {
                                                rdm1[a*norb+i] -= pci0[str1]*tmp;
                                        }
                                }
                        }
                }
        }
        for (j = 0; j < norb; j++) {
                for (k = 0; k < j; k++) {
                        rdm1[k*norb+j] = rdm1[j*norb+k];
                }
        }
        free(clink);
}

