/*
 *
 */

#include <stdlib.h>
#include <string.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define BLK     48
#define BUFBASE 96

typedef struct {
        unsigned int addr;
        unsigned char a;
        unsigned char i;
        char sign;
        char _padding;
} _LinkT;
#define EXTRACT_I(I)    (I.i)
#define EXTRACT_A(I)    (I.a)
#define EXTRACT_SIGN(I) (I.sign)
#define EXTRACT_ADDR(I) (I.addr)

double FCI_t1ci_ms0(double *ci0, double *t1, int bcount,
                    int stra_id, int strb_id,
                    int norb, int na, int nlink, _LinkT *clink_index);

static void compress_link(_LinkT *clink, int *link_index,
                          int norb, int nstr, int nlink)
{
        int i, j, k, a, str1, sign;
        for (k = 0; k < nstr; k++) {
                for (j = 0; j < nlink; j++) {
                        a    = link_index[j*4+0];
                        i    = link_index[j*4+1];
                        str1 = link_index[j*4+2];
                        sign = link_index[j*4+3];
                        clink[j].a = a;
                        clink[j].i = i;
                        clink[j].sign = sign;
                        clink[j].addr = str1;
                }
                clink += nlink;
                link_index += nlink * 4;
        }
}

/*
 * t2[:,i,j,k,l] = E^i_j E^k_l|ci0>
 */
static void rdm4_0b_t2(double *ci0, double *t2,
                       int bcount, int stra_id, int strb_id,
                       int norb, int nstrb, int nlinkb, _LinkT *clink_indexb)
{
        const int nnorb = norb * norb;
        const int n4 = nnorb * nnorb;
        int i, j, k, l, a, sign, str1;
        double *t1 = malloc(sizeof(double) * nstrb * nnorb);
        double *pt1, *pt2;
        _LinkT *tab;

        // form t1 which has beta^+ beta |t1> => target stra_id
        FCI_t1ci_ms0(ci0, t1, nstrb, stra_id, 0,
                     norb, nstrb, nlinkb, clink_indexb);

#pragma omp parallel default(none) \
        shared(t1, t2, bcount, strb_id, norb, nlinkb, clink_indexb), \
        private(i, j, k, l, a, str1, sign, pt1, pt2, tab)
{
#pragma omp for schedule(dynamic, 1) nowait
        for (k = 0; k < bcount; k++) {
                memset(t2+k*n4, 0, sizeof(double)*n4);
                tab = clink_indexb + (strb_id+k) * nlinkb;
                for (j = 0; j < nlinkb; j++) {
                        i    = EXTRACT_I   (tab[j]);
                        a    = EXTRACT_A   (tab[j]);
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
                      int norb, int nstrb, int nlinka, _LinkT *clink_indexa)
{
        const int nnorb = norb * norb;
        const int n4 = nnorb * nnorb;
        int i, j, k, l, a, sign, str1;
        double *t1 = malloc(sizeof(double) * nstrb * nnorb);
        double *pt1, *pt2;
        _LinkT *tab = clink_indexa + stra_id * nlinka;

#pragma omp parallel default(none) \
        shared(ci0, t1, t2, bcount, strb_id, norb, nstrb, nlinka, \
               clink_indexa, tab), \
        private(i, j, k, l, a, str1, sign, pt1, pt2)
{
#pragma omp for schedule(dynamic, 1) nowait
        for (j = 0; j < nlinka; j++) {
                i    = EXTRACT_I   (tab[j]);
                a    = EXTRACT_A   (tab[j]);
                str1 = EXTRACT_ADDR(tab[j]);
                sign = EXTRACT_SIGN(tab[j]);

                // form t1 which has alpha^+ alpha |t1> => target stra_id (through str1)
                FCI_t1ci_ms0(ci0, t1, bcount, str1, strb_id,
                             norb, nstrb, nlinka, clink_indexa);

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
}
        free(t1);
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


void FCI4pdm_kern_ms0(double *rdm1, double *rdm2, double *rdm3, double *rdm4,
                      double *ci0, int bcount, int stra_id, int strb_id,
                      int norb, int nstrb, int nlink, _LinkT *clink)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nnorb = norb * norb;
        const int n4 = nnorb * nnorb;
        const int n3 = nnorb * norb;
        const unsigned long n6 = nnorb * nnorb * nnorb;
        int i, j, k, l, ij;
        unsigned long n;
        double *tbra;
        double *t1 = malloc(sizeof(double) * nnorb * bcount);
        double *t2 = malloc(sizeof(double) * n4 * bcount);
        double *pbra, *pt1, *pt2;

        // t2[:,i,j,k,l] = E^i_j E^k_l|ci0>
        rdm4_0b_t2(ci0, t2, bcount, stra_id, strb_id,
                   norb, nstrb, nlink, clink);
        rdm4_a_t2 (ci0, t2, bcount, stra_id, strb_id,
                   norb, nstrb, nlink, clink);
        FCI_t1ci_ms0(ci0, t1, bcount, stra_id, strb_id,
                     norb, nstrb, nlink, clink);

#pragma omp parallel default(none) \
        shared(rdm3, rdm4, t1, t2, norb, bcount), \
        private(ij, i, j, k, l, n, tbra, pbra, pt1, pt2)
{
        tbra = malloc(sizeof(double) * nnorb * bcount);
#pragma omp for schedule(dynamic, 1) nowait
        for (ij = 0; ij < nnorb; ij++) { // loop ij for (<ci0| E^j_i E^l_k)
                for (n = 0; n < bcount; n++) {
                        for (k = 0; k < norb; k++) {
                                pbra = tbra + n * nnorb + k*norb;
                                pt2 = t2 + n * n4 + k*nnorb + ij;
                                for (l = 0; l < norb; l++) {
                                        pbra[l] = pt2[l*n3];
                                }
                        }
                }

                i = ij / norb;
                j = ij - i * norb;
// contract <bra-of-Eij| with |E^k_l E^m_n ci0>
                tril3pdm_particle_symm(rdm4+(j*norb+i)*n6, tbra, t2,
                                       bcount, j+1, norb);
// rdm3
                tril2pdm_particle_symm(rdm3+(j*norb+i)*n4, tbra, t1,
                                       bcount, j+1, norb);
        }
        free(tbra);
}

// rdm1 and rdm2
        const int INC1 = 1;
        tbra = malloc(sizeof(double) * nnorb * bcount);
        for (n = 0; n < bcount; n++) {
                pbra = tbra + n * nnorb;
                pt1 = t1 + n * nnorb;
                for (k = 0; k < norb; k++) {
                        for (l = 0; l < norb; l++) {
                                pbra[k*norb+l] = pt1[l*norb+k];
                        }
                }
        }
        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &bcount,
               &D1, t1, &nnorb, tbra, &nnorb,
               &D1, rdm2, &nnorb);

        dgemv_(&TRANS_N, &nnorb, &bcount, &D1, t1, &nnorb,
               ci0+stra_id*nstrb+strb_id, &INC1, &D1, rdm1, &INC1);

        free(tbra);
        free(t1);
        free(t2);
}

/*
 * use symmetry ci0[a,b] == ci0[b,a], t2[a,b,...] == t2[b,a,...]
 */
void FCI4pdm_kern_spin0(double *rdm1, double *rdm2, double *rdm3, double *rdm4,
                        double *ci0, int bcount, int stra_id, int strb_id,
                        int norb, int nstrb, int nlink, _LinkT *clink)
{
        int fill1;
        if (strb_id+bcount <= stra_id) {
                fill1 = bcount;
        } else if (stra_id >= strb_id) {
                fill1 = stra_id - strb_id + 1;
        } else {
                return;
        }

        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nnorb = norb * norb;
        const int n4 = nnorb * nnorb;
        const int n3 = nnorb * norb;
        const unsigned long n6 = nnorb * nnorb * nnorb;
        int i, j, k, l, ij;
        unsigned long n;
        double factor;
        double *tbra;
        double *t1 = malloc(sizeof(double) * nnorb * fill1);
        double *t2 = malloc(sizeof(double) * n4 * fill1);
        double *pbra, *pt1, *pt2;

        FCI_t1ci_ms0(ci0, t1, fill1, stra_id, strb_id,
                     norb, nstrb, nlink, clink);

        rdm4_0b_t2(ci0, t2, fill1, stra_id, strb_id,
                   norb, nstrb, nlink, clink);
        rdm4_a_t2 (ci0, t2, fill1, stra_id, strb_id,
                   norb, nstrb, nlink, clink);

#pragma omp parallel default(none) \
        shared(rdm3, rdm4, t1, t2, norb, stra_id, strb_id, fill1), \
        private(ij, i, j, k, l, n, tbra, pbra, pt1, pt2, factor)
{
        tbra = malloc(sizeof(double) * nnorb * fill1);
#pragma omp for schedule(dynamic, 1) nowait
        for (ij = 0; ij < nnorb; ij++) { // loop ij for (<ci0| E^j_i E^l_k)
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
                                pt2 = t2 + n * n4 + k*nnorb + ij;
                                for (l = 0; l < norb; l++) {
                                        pbra[l] = pt2[l*n3] * factor;
                                }
                        }
                }

// contract <bra-of-Eij| with |E^k_l E^m_n ci0>
                tril3pdm_particle_symm(rdm4+(j*norb+i)*n6, tbra, t2,
                                       fill1, j+1, norb);
// rdm3
                tril2pdm_particle_symm(rdm3+(j*norb+i)*n4, tbra, t1,
                                       fill1, j+1, norb);
        }
        free(tbra);
}

// rdm1 and rdm2
        tbra = malloc(sizeof(double) * nnorb * fill1);
        for (n = 0; n < fill1; n++) {
                if (n+strb_id == stra_id) {
                        factor = 1;
                } else {
                        factor = 2;
                }
                pbra = tbra + n * nnorb;
                pt1 = t1 + n * nnorb;
                for (k = 0; k < norb; k++) {
                        for (l = 0; l < norb; l++) {
                                pbra[k*norb+l] = pt1[l*norb+k] * factor;
                        }
                }
        }
        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &fill1,
               &D1, t1, &nnorb, tbra, &nnorb,
               &D1, rdm2, &nnorb);

        const int INC1 = 1;
        dgemv_(&TRANS_N, &nnorb, &fill1, &D1, tbra, &nnorb,
               ci0+stra_id*nstrb+strb_id, &INC1, &D1, rdm1, &INC1);

        free(tbra);
        free(t1);
        free(t2);
}


/*
 * This function returns incomplete rdm3, rdm4, in which, particle
 * permutation symmetry is assumed.
 * kernel can be FCI4pdm_kern_ms0, FCI4pdm_kern_spin0
 */
void FCIrdm4_drv(void (*kernel)(),
                 double *rdm1, double *rdm2, double *rdm3, double *rdm4,
                 double *bra, double *ket,
                 int norb, int na, int nb, int nlinka, int nlinkb,
                 int *link_indexa, int *link_indexb)
{
        const unsigned long nnorb = norb * norb;
        const unsigned long n4 = nnorb * nnorb;
        int ib, strk, bcount;

        _LinkT *clinka = malloc(sizeof(_LinkT) * nlinka * na);
        compress_link(clinka, link_indexa, norb, na, nlinka);
        memset(rdm1, 0, sizeof(double) * nnorb);
        memset(rdm2, 0, sizeof(double) * n4);
        memset(rdm3, 0, sizeof(double) * n4 * nnorb);
        memset(rdm4, 0, sizeof(double) * n4 * n4);

        for (strk = 0; strk < na; strk++) {
                for (ib = 0; ib < na; ib += BUFBASE) {
                        bcount = MIN(BUFBASE, na-ib);
                        (*kernel)(rdm1, rdm2, rdm3, rdm4,
                                  ket, bcount, strk, ib,
                                  norb, na, nlinka, clinka);
                }
        }
        free(clinka);
}


void FCI3pdm_kern_ms0(double *rdm1, double *rdm2, double *rdm3,
                      double *ci0, int bcount, int stra_id, int strb_id,
                      int norb, int nstrb, int nlink, _LinkT *clink)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nnorb = norb * norb;
        const int n4 = nnorb * nnorb;
        const int n3 = nnorb * norb;
        int i, j, k, l, ij;
        unsigned long n;
        double *tbra = malloc(sizeof(double) * nnorb * bcount);
        double *t1 = malloc(sizeof(double) * nnorb * bcount);
        double *t2 = malloc(sizeof(double) * n4 * bcount);
        double *pbra, *pt1, *pt2;

        // t2[:,i,j,k,l] = E^i_j E^k_l|ci0>
        rdm4_0b_t2(ci0, t2, bcount, stra_id, strb_id,
                   norb, nstrb, nlink, clink);
        rdm4_a_t2 (ci0, t2, bcount, stra_id, strb_id,
                   norb, nstrb, nlink, clink);
        FCI_t1ci_ms0(ci0, t1, bcount, stra_id, strb_id,
                     norb, nstrb, nlink, clink);

#pragma omp parallel default(none) \
        shared(rdm3, t1, t2, norb, bcount), \
        private(ij, i, j, k, l, n, tbra, pbra, pt1, pt2)
{
        tbra = malloc(sizeof(double) * nnorb * bcount);
#pragma omp for schedule(dynamic, 1) nowait
        for (ij = 0; ij < nnorb; ij++) { // loop ij for (<ci0| E^j_i E^l_k)
                for (n = 0; n < bcount; n++) {
                        pbra = tbra + n * nnorb;
                        pt2 = t2 + n * n4 + ij;
                        for (k = 0; k < norb; k++) {
                                for (l = 0; l < norb; l++) {
                                        pbra[k*norb+l] = pt2[l*n3+k*nnorb];
                                }
                        }
                }

                i = ij / norb;
                j = ij - i * norb;
                tril2pdm_particle_symm(rdm3+(j*norb+i)*n4, tbra, t1,
                                       bcount, j+1, norb);
        }
        free(tbra);
}

// rdm1 and rdm2
        const int INC1 = 1;
        for (n = 0; n < bcount; n++) {
                pbra = tbra + n * nnorb;
                pt1 = t1 + n * nnorb;
                for (k = 0; k < norb; k++) {
                        for (l = 0; l < norb; l++) {
                                pbra[k*norb+l] = pt1[l*norb+k];
                        }
                }
        }
        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &bcount,
               &D1, t1, &nnorb, tbra, &nnorb,
               &D1, rdm2, &nnorb);

        dgemv_(&TRANS_N, &nnorb, &bcount, &D1, t1, &nnorb,
               ci0+stra_id*nstrb+strb_id, &INC1, &D1, rdm1, &INC1);

        free(tbra);
        free(t1);
        free(t2);
}

/*
 * use symmetry ci0[a,b] == ci0[b,a], t2[a,b,...] == t2[b,a,...]
 */
void FCI3pdm_kern_spin0(double *rdm1, double *rdm2, double *rdm3,
                        double *ci0, int bcount, int stra_id, int strb_id,
                        int norb, int nstrb, int nlink, _LinkT *clink)
{
        int fill1;
        if (strb_id+bcount <= stra_id) {
                fill1 = bcount;
        } else if (stra_id >= strb_id) {
                fill1 = stra_id - strb_id + 1;
        } else {
                return;
        }

        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int nnorb = norb * norb;
        const int n4 = nnorb * nnorb;
        const int n3 = nnorb * norb;
        int i, j, k, l, ij;
        unsigned long n;
        double factor;
        double *tbra;
        double *t1 = malloc(sizeof(double) * nnorb * fill1);
        double *t2 = malloc(sizeof(double) * n4 * fill1);
        double *pbra, *pt1, *pt2;

        FCI_t1ci_ms0(ci0, t1, fill1, stra_id, strb_id,
                     norb, nstrb, nlink, clink);

        // t2[:,i,j,k,l] = E^i_j E^k_l|ci0>
        rdm4_0b_t2(ci0, t2, fill1, stra_id, strb_id,
                   norb, nstrb, nlink, clink);
        rdm4_a_t2 (ci0, t2, fill1, stra_id, strb_id,
                   norb, nstrb, nlink, clink);

#pragma omp parallel default(none) \
        shared(rdm3, t1, t2, norb, stra_id, strb_id, fill1), \
        private(ij, i, j, k, l, n, tbra, pbra, pt1, pt2, factor)
{
        tbra = malloc(sizeof(double) * nnorb * fill1);
#pragma omp for schedule(dynamic, 1) nowait
        for (ij = 0; ij < nnorb; ij++) { // loop ij for (<ci0| E^j_i E^l_k)
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
                                pt2 = t2 + n * n4 + k*nnorb + ij;
                                for (l = 0; l < norb; l++) {
                                        pbra[l] = pt2[l*n3] * factor;
                                }
                        }
                }

                tril2pdm_particle_symm(rdm3+(j*norb+i)*n4, tbra, t1,
                                       fill1, j+1, norb);
        }
        free(tbra);
}

// rdm1 and rdm2
        tbra = malloc(sizeof(double) * nnorb * fill1);
        for (n = 0; n < fill1; n++) {
                if (n+strb_id == stra_id) {
                        factor = 1;
                } else {
                        factor = 2;
                }
                pbra = tbra + n * nnorb;
                pt1 = t1 + n * nnorb;
                for (k = 0; k < norb; k++) {
                        for (l = 0; l < norb; l++) {
                                pbra[k*norb+l] = pt1[l*norb+k] * factor;
                        }
                }
        }
        dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &fill1,
               &D1, t1, &nnorb, tbra, &nnorb,
               &D1, rdm2, &nnorb);

        const int INC1 = 1;
        dgemv_(&TRANS_N, &nnorb, &fill1, &D1, tbra, &nnorb,
               ci0+stra_id*nstrb+strb_id, &INC1, &D1, rdm1, &INC1);

        free(tbra);
        free(t1);
        free(t2);
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
        const unsigned long nnorb = norb * norb;
        const unsigned long n4 = nnorb * nnorb;
        int ib, strk, bcount;

        _LinkT *clinka = malloc(sizeof(_LinkT) * nlinka * na);
        compress_link(clinka, link_indexa, norb, na, nlinka);
        memset(rdm1, 0, sizeof(double) * nnorb);
        memset(rdm2, 0, sizeof(double) * n4);
        memset(rdm3, 0, sizeof(double) * n4 * nnorb);

        for (strk = 0; strk < na; strk++) {
                for (ib = 0; ib < na; ib += BUFBASE) {
                        bcount = MIN(BUFBASE, na-ib);
                        (*kernel)(rdm1, rdm2, rdm3,
                                  ket, bcount, strk, ib,
                                  norb, na, nlinka, clinka);
                }
        }
        free(clinka);
}

