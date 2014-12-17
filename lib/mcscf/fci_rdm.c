/*
 *
 */

#include <stdlib.h>
#include <string.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#define CSUMTHR         1e-28


static double rdm2_a_t1(double *ci0, double *t1, int fillcnt, int stra_id,
                        int norb, int nstrb, int nlinka, short *link_indexa)
{
        const int nnorb = norb * norb;
        int i, j, k, a, str1, sign;
        const short *tab = link_indexa + stra_id * nlinka * 4;
        double *pt1, *pci;
        double csum = 0;

        for (j = 0; j < nlinka; j++) {
                a = tab[j*4+0];
                i = tab[j*4+1];
                str1 = tab[j*4+2];
                sign = tab[j*4+3];
                pci = ci0 + str1*nstrb;
                pt1 = t1 + i*norb+a;
                if (sign > 0) {
                        for (k = 0; k < fillcnt; k++) {
                                pt1[k*nnorb] += pci[k];
                                csum += pci[k] * pci[k];
                        }
                } else {
                        for (k = 0; k < fillcnt; k++) {
                                pt1[k*nnorb] -= pci[k];
                                csum += pci[k] * pci[k];
                        }
                }
        }
        return csum;
}

static double rdm2_0b_t1(double *ci0, double *t1, int fillcnt, int stra_id,
                         int norb, int nstrb, int nlinkb, short *link_indexb)
{
        const int nnorb = norb * norb;
        int i, j, a, str0, str1, sign;
        const short *tab = link_indexb;
        double *pci = ci0 + stra_id*nstrb;
        double csum = 0;

        for (str0 = 0; str0 < fillcnt; str0++) {
                memset(t1, 0, sizeof(double) * nnorb);
                for (j = 0; j < nlinkb; j++) {
                        a = tab[j*4+0];
                        i = tab[j*4+1];
                        str1 = tab[j*4+2];
                        sign = tab[j*4+3];
                        t1[i*norb+a] += sign * pci[str1];
                        csum += pci[str1] * pci[str1];
                }
                t1 += nnorb;
                tab += nlinkb * 4;
        }
        return csum;
}

static double kern_ms0_ab(double *ci0, double *t1, int fillcnt,
                          int stra_id, int strb_id,
                          int norb, int na, int nlink, short *link_index)
{
        double csum;
        csum = rdm2_0b_t1(ci0, t1, fillcnt, stra_id,
                          norb, na, nlink, link_index+strb_id*nlink*4)
             + rdm2_a_t1 (ci0+strb_id, t1, fillcnt, stra_id,
                          norb, na, nlink, link_index);
        return csum;
}


/*
 * If symm != 0, symmetrize rdm1 and rdm2
 * Note! The returned rdm2 corresponds to
 *      [(p^+ q on <bra|) r^+ s] = [p q^+ r^+ s]
 * in FCIrdm12kern_ms0, FCIrdm12kern_spin0, FCIrdm12kern_a, ...
 * t1 is calculated as |K> = i^+ j|0>. by doing dot(t1.T,t1) to get "rdm2",
 * The ket part (k^+ l|0>) will generate the correct order for the last
 * two indices kl of rdm2(i,j,k,l), But the bra part (i^+ j|0>)^dagger
 * will generate an order of (i,j), which is identical to call a bra of
 * (<0|i j^+).  The so-obtained rdm2(i,j,k,l) corresponds to the
 * operator sequence i j^+ k^+ l.  In these cases, be sure transpose i,j
 * for rdm2(i,j,k,l) after calling FCIrdm12_drv.
 */
void FCIrdm12_drv(void (*dm12kernel)(),
                  double *rdm1, double *rdm2, double *bra, double *ket,
                  int norb, int na, int nb, int nlinka, int nlinkb,
                  short *link_indexa, short *link_indexb, int symm)
{
        const int nnorb = norb * norb;
        const int bufbase = MIN(BUFBASE, nb);
        int strk, i, j, ib, blen;
        double *pdm1, *pdm2;
        memset(rdm1, 0, sizeof(double) * nnorb);
        memset(rdm2, 0, sizeof(double) * nnorb*nnorb);

#pragma omp parallel default(none) \
        shared(dm12kernel, bra, ket, norb, na, nb, nlinka, \
               nlinkb, link_indexa, link_indexb, rdm1, rdm2), \
        private(strk, i, ib, blen, pdm1, pdm2)
{
        pdm1 = (double *)malloc(sizeof(double) * nnorb);
        pdm2 = (double *)malloc(sizeof(double) * nnorb*nnorb);
        memset(pdm1, 0, sizeof(double) * nnorb);
        memset(pdm2, 0, sizeof(double) * nnorb*nnorb);
        for (ib = 0; ib < nb; ib += bufbase) {
                blen = MIN(bufbase, nb-ib);
#pragma omp for schedule(dynamic, 2) nowait
                for (strk = 0; strk < na; strk++) {
                        (*dm12kernel)(pdm1, pdm2, bra, ket, blen, strk, ib,
                                      norb, na, nb, nlinka, nlinkb,
                                      link_indexa, link_indexb);
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
        if (symm) {
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
        }
}

/*
 * for ms0 (which has same number of alpha and beta electrons),
 * not necessarily be singlet
 */
void FCIrdm12kern_ms0(double *rdm1, double *rdm2, double *bra, double *ket,
                      int fillcnt, int stra_id, int strb_id,
                      int norb, int na, int nb, int nlinka, int nlinkb,
                      short *link_indexa, short *link_indexb)
{
        const int INC1 = 1;
        const char UP = 'U';
        const char TRANS_N = 'N';
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;
        double *buf = malloc(sizeof(double) * nnorb * na);

        csum = kern_ms0_ab(ket, buf, fillcnt, stra_id, strb_id,
                           norb, na, nlinka, link_indexa);
        if (csum > CSUMTHR) {
                dgemv_(&TRANS_N, &nnorb, &fillcnt, &D1, buf, &nnorb,
                       ket+stra_id*na+strb_id, &INC1, &D1, rdm1, &INC1);
                dsyrk_(&UP, &TRANS_N, &nnorb, &fillcnt,
                       &D1, buf, &nnorb, &D1, rdm2, &nnorb);
        }
        free(buf);
}

void FCImake_rdm12_ms0(double *rdm1, double *rdm2, double *bra, double *ket,
                       int norb, int na, int nlink, short *link_index)
{
        FCIrdm12_drv(FCIrdm12kern_ms0,
                     rdm1, rdm2, ket, ket, norb, na, na, nlink, nlink,
                     link_index, link_index, 1);
}

/*
 * _spin0 assumes the strict symmetry on alpha and beta electrons
 */
void FCIrdm12kern_spin0(double *rdm1, double *rdm2, double *bra, double *ket,
                        int fillcnt, int stra_id, int strb_id,
                        int norb, int na, int nb, int nlinka, int nlinkb,
                        short *link_indexa, short *link_indexb)
{
        const int INC1 = 1;
        const char UP = 'U';
        const char TRANS_N = 'N';
        const double D1 = 1;
        const double D2 = 2;
        const double D4 = 4;
        const int nnorb = norb * norb;
        int fill0, fill1;
        double csum;
        double *buf = malloc(sizeof(double) * nnorb * na);

        if (strb_id+fillcnt < stra_id) {
                fill0 = fillcnt;
                fill1 = fillcnt;
                csum = rdm2_0b_t1(ket, buf, fill0, stra_id, norb, na,
                                  nlinka, link_indexa+strb_id*nlinka*4)
                     + rdm2_a_t1 (ket+strb_id, buf, fill1, stra_id, norb, na,
                                  nlinka, link_indexa);
        } else {
                fill0 = stra_id - strb_id;
                fill1 = stra_id - strb_id + 1;
                memset(buf+fill0*nnorb, 0, sizeof(double)*nnorb);
                csum = rdm2_0b_t1(ket, buf, fill0, stra_id, norb, na,
                                  nlinka, link_indexa+strb_id*nlinka*4)
                     + rdm2_a_t1 (ket+strb_id, buf, fill1, stra_id, norb, na,
                                  nlinka, link_indexa);
        }
        if (csum > CSUMTHR) {
                dgemv_(&TRANS_N, &nnorb, &fill1, &D2, buf, &nnorb,
                       ket+stra_id*na+strb_id, &INC1, &D1, rdm1, &INC1);
// dsyrk_ of debian-6 libf77blas.so.3gf causes NaN in pdm2, libblas bug?
//                        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &fill0,
//                               &D1, buf, &nnorb, buf, &nnorb,
//                               &D1, rdm2, &nnorb);
                dsyrk_(&UP, &TRANS_N, &nnorb, &fill0,
                       &D2, buf, &nnorb, &D1, rdm2, &nnorb);
                dsyr_(&UP, &nnorb, &D4, buf+nnorb*fill0, &INC1,
                      rdm2, &nnorb);
        }
        free(buf);
}

void FCImake_rdm12_spin0(double *rdm1, double *rdm2, double *bra, double *ket,
                         int norb, int na, int nlink, short *link_index)
{
        FCIrdm12_drv(FCIrdm12kern_spin0,
                     rdm1, rdm2, ket, ket, norb, na, na, nlink, nlink,
                     link_index, link_index, 1);
}




/*
 * ***********************************************
 * transition density matrix, for ms0, not necessarily be singlet
 */
void FCItdm12kern_ms0(double *tdm1, double *tdm2, double *bra, double *ket,
                      int fillcnt, int stra_id, int strb_id,
                      int norb, int na, int nb, int nlinka, int nlinkb,
                      short *link_indexa, short *link_indexb)
{
        const int INC1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;
        double *buf0 = malloc(sizeof(double) * nnorb*na);
        double *buf1 = malloc(sizeof(double) * nnorb*na);

        csum = kern_ms0_ab(bra, buf1, fillcnt, stra_id, strb_id,
                           norb, na, nlinka, link_indexa);
        if (csum < CSUMTHR) { goto end; }
        csum = kern_ms0_ab(ket, buf0, fillcnt, stra_id, strb_id,
                           norb, na, nlinka, link_indexa);
        if (csum < CSUMTHR) { goto end; }
        dgemv_(&TRANS_N, &nnorb, &fillcnt, &D1, buf0, &nnorb,
               bra+stra_id*na+strb_id, &INC1, &D1, tdm1, &INC1);
        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &fillcnt,
               &D1, buf0, &nnorb, buf1, &nnorb,
               &D1, tdm2, &nnorb);
end:
        free(buf0);
        free(buf1);
}

void FCItrans_rdm12_ms0(double *rdm1, double *rdm2,
                        double *bra, double *ket,
                        int norb, int na, int nlink, short *link_index)
{
        FCIrdm12_drv(FCItdm12kern_ms0,
                     rdm1, rdm2, bra, ket, norb, na, na, nlink, nlink,
                     link_index, link_index, 0);
}


/*
 * ***********************************************
 * 2pdm kernel for ms != 0 or spin-orbital
 * ***********************************************
 */
void FCIrdm12kern_a(double *rdm1, double *rdm2, double *bra, double *ket,
                    int fillcnt, int stra_id, int strb_id,
                    int norb, int na, int nb, int nlinka, int nlinkb,
                    short *link_indexa, short *link_indexb)
{
        const int INC1 = 1;
        const char UP = 'U';
        const char TRANS_N = 'N';
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;
        double *buf = malloc(sizeof(double) * nnorb*nb);

        memset(buf, 0, sizeof(double)*nnorb*nb);
        csum = rdm2_a_t1(ket+strb_id, buf, fillcnt, stra_id,
                         norb, nb, nlinka, link_indexa);
        if (csum > CSUMTHR) {
                dgemv_(&TRANS_N, &nnorb, &fillcnt, &D1, buf, &nnorb,
                       ket+stra_id*nb+strb_id, &INC1, &D1, rdm1, &INC1);
                dsyrk_(&UP, &TRANS_N, &nnorb, &fillcnt,
                       &D1, buf, &nnorb, &D1, rdm2, &nnorb);
        }
        free(buf);
}
void FCIrdm12kern_b(double *rdm1, double *rdm2, double *bra, double *ket,
                    int fillcnt, int stra_id, int strb_id,
                    int norb, int na, int nb, int nlinka, int nlinkb,
                    short *link_indexa, short *link_indexb)
{
        const int INC1 = 1;
        const char UP = 'U';
        const char TRANS_N = 'N';
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;
        double *buf = malloc(sizeof(double) * nnorb*nb);

        csum = rdm2_0b_t1(ket, buf, fillcnt, stra_id,
                          norb, nb, nlinkb, link_indexb+strb_id*nlinkb*4);
        if (csum > CSUMTHR) {
                dgemv_(&TRANS_N, &nnorb, &fillcnt, &D1, buf, &nnorb,
                       ket+stra_id*nb+strb_id, &INC1, &D1, rdm1, &INC1);
                dsyrk_(&UP, &TRANS_N, &nnorb, &fillcnt,
                       &D1, buf, &nnorb, &D1, rdm2, &nnorb);
        }
        free(buf);
}

void FCItdm12kern_a(double *tdm1, double *tdm2, double *bra, double *ket,
                    int fillcnt, int stra_id, int strb_id,
                    int norb, int na, int nb, int nlinka, int nlinkb,
                    short *link_indexa, short *link_indexb)
{
        const int INC1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;
        double *buf0 = malloc(sizeof(double) * nnorb*nb);
        double *buf1 = malloc(sizeof(double) * nnorb*nb);

        memset(buf1, 0, sizeof(double)*nnorb*nb);
        csum = rdm2_a_t1(bra+strb_id, buf1, fillcnt, stra_id,
                         norb, nb, nlinka, link_indexa);
        if (csum < CSUMTHR) { goto end; }
        memset(buf0, 0, sizeof(double)*nnorb*nb);
        csum = rdm2_a_t1(ket+strb_id, buf0, fillcnt, stra_id,
                         norb, nb, nlinka, link_indexa);
        if (csum < CSUMTHR) { goto end; }
        dgemv_(&TRANS_N, &nnorb, &fillcnt, &D1, buf0, &nnorb,
               bra+stra_id*nb+strb_id, &INC1, &D1, tdm1, &INC1);
        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &fillcnt,
               &D1, buf0, &nnorb, buf1, &nnorb,
               &D1, tdm2, &nnorb);
end:
        free(buf0);
        free(buf1);
}

void FCItdm12kern_b(double *tdm1, double *tdm2, double *bra, double *ket,
                    int fillcnt, int stra_id, int strb_id,
                    int norb, int na, int nb, int nlinka, int nlinkb,
                    short *link_indexa, short *link_indexb)
{
        const int INC1 = 1;
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;
        double *buf0 = malloc(sizeof(double) * nnorb*nb);
        double *buf1 = malloc(sizeof(double) * nnorb*nb);

        csum = rdm2_0b_t1(bra, buf1, fillcnt, stra_id,
                          norb, nb, nlinkb, link_indexb+strb_id*nlinkb*4);
        if (csum < CSUMTHR) { goto end; }
        csum = rdm2_0b_t1(ket, buf0, fillcnt, stra_id,
                          norb, nb, nlinkb, link_indexb+strb_id*nlinkb*4);
        if (csum < CSUMTHR) { goto end; }
        dgemv_(&TRANS_N, &nnorb, &fillcnt, &D1, buf0, &nnorb,
               bra+stra_id*nb+strb_id, &INC1, &D1, tdm1, &INC1);
        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &fillcnt,
               &D1, buf0, &nnorb, buf1, &nnorb,
               &D1, tdm2, &nnorb);
end:
        free(buf0);
        free(buf1);
}

void FCItdm12kern_ab(double *tdm1, double *tdm2, double *bra, double *ket,
                     int fillcnt, int stra_id, int strb_id,
                     int norb, int na, int nb, int nlinka, int nlinkb,
                     short *link_indexa, short *link_indexb)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;
        double *bufb = malloc(sizeof(double) * nnorb*nb);
        double *bufa = malloc(sizeof(double) * nnorb*nb);

        memset(bufa, 0, sizeof(double)*nnorb*nb);
        csum = rdm2_a_t1(bra+strb_id, bufa, fillcnt, stra_id,
                         norb, nb, nlinka, link_indexa);
        if (csum < CSUMTHR) { goto end; }
        csum = rdm2_0b_t1(ket, bufb, fillcnt, stra_id,
                          norb, nb, nlinkb, link_indexb+strb_id*nlinkb*4);
        if (csum < CSUMTHR) { goto end; }

        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &fillcnt,
               &D1, bufb, &nnorb, bufa, &nnorb,
               &D1, tdm2, &nnorb);
end:
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
                    short *link_indexa, short *link_indexb)
{
        int i, a, j, k, str0, str1, sign;
        short *tab;
        double *pket, *pbra;

        memset(rdm1, 0, sizeof(double) * norb*norb);

        for (str0 = 0; str0 < na; str0++) {
                tab = link_indexa + str0 * nlinka * 4;
                pket = ket + str0 * nb;
                for (j = 0; j < nlinka; j++) {
                        a = tab[j*4+0];
                        i = tab[j*4+1];
                        str1 = tab[j*4+2];
                        sign = tab[j*4+3];
                        pbra = bra + str1 * nb;
                        for (k = 0; k < nb; k++) {
                                rdm1[a*norb+i] += sign*pbra[k]*pket[k];
                        }
                }
        }
}

void FCItrans_rdm1b(double *rdm1, double *bra, double *ket,
                    int norb, int na, int nb, int nlinka, int nlinkb,
                    short *link_indexa, short *link_indexb)
{
        int i, a, j, k, str0, str1, sign;
        short *tab;
        double *pket, *pbra;
        double tmp;

        memset(rdm1, 0, sizeof(double) * norb*norb);

        for (str0 = 0; str0 < na; str0++) {
                pbra = bra + str0 * nb;
                pket = ket + str0 * nb;
                for (k = 0; k < nb; k++) {
                        tab = link_indexb + k * nlinkb * 4;
                        tmp = pket[k];
                        for (j = 0; j < nlinkb; j++) {
                                a = tab[j*4+0];
                                i = tab[j*4+1];
                                str1 = tab[j*4+2];
                                sign = tab[j*4+3];
                                rdm1[a*norb+i] += sign*pbra[str1]*tmp;
                        }
                }
        }
}

/*
 * make_rdm1 assumed the hermitian of density matrix
 */
void FCImake_rdm1a(double *rdm1, double *cibra, double *ciket,
                   int norb, int na, int nb, int nlinka, int nlinkb,
                   short *link_indexa, short *link_indexb)
{
        int i, a, j, k, str0, str1, sign;
        short *tab;
        double *pci0, *pci1;
        double *ci0 = ciket;

        memset(rdm1, 0, sizeof(double) * norb*norb);

        for (str0 = 0; str0 < na; str0++) {
                tab = link_indexa + str0 * nlinka * 4;
                pci0 = ci0 + str0 * nb;
                for (j = 0; j < nlinka; j++) {
                        a = tab[j*4+0];
                        i = tab[j*4+1];
                        str1 = tab[j*4+2];
                        sign = tab[j*4+3];
                        pci1 = ci0 + str1 * nb;
                        if (a >= i) {
                                if (sign > 0) {
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
}

void FCImake_rdm1b(double *rdm1, double *cibra, double *ciket,
                   int norb, int na, int nb, int nlinka, int nlinkb,
                   short *link_indexa, short *link_indexb)
{
        int i, a, j, k, str0, str1, sign;
        short *tab;
        double *pci0;
        double *ci0 = ciket;
        double tmp;

        memset(rdm1, 0, sizeof(double) * norb*norb);

        for (str0 = 0; str0 < na; str0++) {
                pci0 = ci0 + str0 * nb;
                for (k = 0; k < nb; k++) {
                        tab = link_indexb + k * nlinkb * 4;
                        tmp = pci0[k];
                        for (j = 0; j < nlinkb; j++) {
                                a = tab[j*4+0];
                                i = tab[j*4+1];
                                str1 = tab[j*4+2];
                                sign = tab[j*4+3];
                                if (a >= i) {
                                        if (sign > 0) {
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
}

