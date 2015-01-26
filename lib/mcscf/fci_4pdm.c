/*
 *
 */

#include <stdlib.h>
#include <string.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define CSUMTHR         1e-28
#define BUFBASE         96

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

double FCI_t1ci_ms0(double *ci0, double *t1, int fillcnt,
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

static void fill_t1ci(double *t1ci, double *ci0, int fillcnt,
                      int stra_id, int strb_id,
                      int norb, int na, int nlink, _LinkT *clink_index)
{
        int nnorb = norb * norb;
        int i, k;
        double *p1;
        double *t1 = malloc(sizeof(double) * fillcnt * nnorb);

        FCI_t1ci_ms0(ci0, t1, fillcnt, stra_id, strb_id,
                     norb, na, nlink, clink_index);
        p1 = t1ci + stra_id * na + strb_id;
        for (i = 0; i < nnorb; i++) {
                for (k = 0; k < fillcnt; k++) {
                        p1[k] = t1[k*nnorb+i];
                }
                p1 += na * na;
        }
        free(t1);
}

static void rdm12_sub(void (*dm12kernel)(),
                      double *rdm1, double *rdm2, double *bra, double *ket,
                      int ncre, int norb, int na, int nlinka, _LinkT *clinka)
{
        const int nnorb = norb * norb;
        const int bufbase = MIN(BUFBASE, na);
        int strk, ib, blen;
        memset(rdm1, 0, sizeof(double) * nnorb);
        memset(rdm2, 0, sizeof(double) * nnorb*nnorb);

        for (ib = 0; ib < na; ib += bufbase) {
                blen = MIN(bufbase, na-ib);
                for (strk = 0; strk < na; strk++) {
                        (*dm12kernel)(rdm1, rdm2, bra, ket, blen, strk, ib,
                                      ncre, norb, na, na, nlinka, nlinka,
                                      clinka, clinka);
                }
        }

//        // particle symmetry is assumed
//
//        int i, j, k, l;
//        double *pdm1, *pdm2;
//        for (i = 0; i < ncre; i++) {
//        for (j = 0; j < i; j++) {
//                pdm1 = rdm2 + (i*nnorb+j)*norb;
//                pdm2 = rdm2 + (j*nnorb+i)*norb;
//                for (k = 0; k < norb; k++) {
//                for (l = 0; l < norb; l++) {
//                        pdm2[l*nnorb+k] = pdm1[k*nnorb+l];
//                } }
//// E^j_lE^i_k = E^i_kE^j_l + \delta_{il}E^j_k - \dleta_{jk}E^i_l
//                for (k = 0; k < ncre; k++) {
//                        pdm2[i*nnorb+k] += rdm1[j*norb+k];
//                        pdm2[k*nnorb+j] -= rdm1[i*norb+k];
//                }
//        } }
}


static void tril_particle_symm(double *rdm2, double *tbra, double *tket,
                               int fillcnt, int ncre, int norb,
                               double alpha, double beta)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        int nnorb = norb * norb;
        int nncre = norb * ncre;
        int i, j, k, m, n;
        int blk = MIN(((int)(48/norb))*norb, nnorb);
        double *buf = malloc(sizeof(double) * nncre*fillcnt);
        double *p1;

        for (n = 0, k = 0; k < fillcnt; k++) {
                p1 = tbra + k * nnorb;
                for (i = 0; i < ncre; i++) {
                for (j = 0; j < norb; j++, n++) {
                        buf[n] = p1[j*norb+i];
                } }
        }

        //dgemm_(&TRANS_N, &TRANS_T, &nncre, &nncre, &fillcnt,
        //       &alpha, tket, &nnorb, buf, &nncre, &beta, rdm2, &nnorb);
        for (m = 0; m < nncre-blk; m+=blk) {
                n = nncre - m;
                dgemm_(&TRANS_N, &TRANS_T, &blk, &n, &fillcnt,
                       &alpha, tket+m, &nnorb, buf+m, &nncre,
                       &beta, rdm2+m*nnorb+m, &nnorb);
        }
        n = nncre - m;
        dgemm_(&TRANS_N, &TRANS_T, &n, &n, &fillcnt,
               &alpha, tket+m, &nnorb, buf+m, &nncre,
               &beta, rdm2+m*nnorb+m, &nnorb);

        free(buf);
}

void FCI4pdm12kern_ms0(double *tdm1, double *tdm2, double *bra, double *ket,
                       int fillcnt, int stra_id, int strb_id,
                       int ncre, int norb,
                       int na, int nb, int nlinka, int nlinkb,
                       _LinkT *clink_indexa, _LinkT *clink_indexb)
{
        const int INC1 = 1;
        const char TRANS_N = 'N';
        const double D1 = 1;
        const int nnorb = norb * norb;
        double csum;
        double *buf0 = malloc(sizeof(double) * nnorb*fillcnt);
        double *buf1 = malloc(sizeof(double) * nnorb*fillcnt);

        csum = FCI_t1ci_ms0(bra, buf1, fillcnt, stra_id, strb_id,
                            norb, na, nlinka, clink_indexa);
        if (csum < CSUMTHR) { goto _normal_end; }
        csum = FCI_t1ci_ms0(ket, buf0, fillcnt, stra_id, strb_id,
                            norb, na, nlinka, clink_indexa);
        if (csum < CSUMTHR) { goto _normal_end; }
        dgemv_(&TRANS_N, &nnorb, &fillcnt, &D1, buf0, &nnorb,
               bra+stra_id*na+strb_id, &INC1, &D1, tdm1, &INC1);
        tril_particle_symm(tdm2, buf1, buf0, fillcnt, ncre, norb, D1, D1);
_normal_end:
        free(buf0);
        free(buf1);
}

static void fill_rdm2(double *dm4, double *dm2, int ncre, int norb, int stride)
{
        int nnorb = norb * norb;
        int nncre = norb * ncre;
        int i, j;
        for (i = 0; i < nncre; i++) {
                for (j = 0; j < nncre; j++) {
                        dm4[j*stride] = dm2[j];
                }
                dm2 += nnorb;
                dm4 += nnorb * stride;
        }
}

static void _transpose_jikl(double *dm2, int norb)
{
        int nnorb = norb * norb;
        int i, j;
        double *p0, *p1;
        double *tmp = malloc(sizeof(double)*nnorb*nnorb);

        for (i = 0; i < nnorb; i++) {
                for (j = 0; j < i; j++) {
                        dm2[j*nnorb+i] = dm2[i*nnorb+j];
                }
        }

        memcpy(tmp, dm2, sizeof(double)*nnorb*nnorb);
        for (i = 0; i < norb; i++) {
                for (j = 0; j < norb; j++) {
                        p0 = tmp + (j*norb+i) * nnorb;
                        p1 = dm2 + (i*norb+j) * nnorb;
                        memcpy(p1, p0, sizeof(double)*nnorb);
                }
        }
        free(tmp);
}


/*
 * This function returns incomplete rdm3, rdm4, in which, particle
 * permutation symmetry is assumed.
 */
void FCIrdm4_drv(double *rdm1, double *rdm2, double *rdm3, double *rdm4,
                 double *bra, double *ket,
                 int norb, int na, int nb, int nlinka, int nlinkb,
                 int *link_indexa, int *link_indexb)
{
        bra = ket;
        nb = na;
        nlinkb = nlinka;
        link_indexb = link_indexa;

        const int nnorb = norb * norb;
        const long n6 = nnorb * nnorb * nnorb;
        int ij, i, j, k, l;
        double *pdm1, *pdm2;
        double *t1ci = malloc(sizeof(double) * na*nb*nnorb);

        _LinkT *clinka = malloc(sizeof(_LinkT) * nlinka * na);
        compress_link(clinka, link_indexa, norb, na, nlinka);

#pragma omp parallel default(none) \
        shared(t1ci, ket, norb, na, nlinka, clinka), \
        private(i)
#pragma omp for schedule(dynamic, 2) nowait
        for (i = 0; i < na; i++) {
                fill_t1ci(t1ci, ket, na, i, 0, norb, na, nlinka, clinka);
        }

#pragma omp parallel default(none) \
        shared(rdm3, rdm4, t1ci, ket, norb, na, nlinka, clinka), \
        private(ij, i, j, k, l, pdm1, pdm2)
{
        pdm1 = (double *)malloc(sizeof(double) * nnorb);
        pdm2 = (double *)malloc(sizeof(double) * nnorb*nnorb);
#pragma omp for schedule(dynamic, 1) nowait
        for (ij = 0; ij < nnorb*norb; ij++) {
                i = ij / nnorb;
                j = ((int)(ij - i*nnorb)) / norb;
                l = ij % norb;
                for (k = 0; k <= i; k++) {
// the bra-side swap i,j indices
                        rdm12_sub(FCI4pdm12kern_ms0, pdm1, pdm2,
                                  t1ci+(j*norb+i)*na*na, t1ci+(k*norb+l)*na*na,
                                  k+1, norb, na, nlinka, clinka);
                        fill_rdm2(rdm4+(i*norb+j)*n6+k*norb+l, pdm2,
                                  k+1, norb, nnorb);
                }
                if (l == 0) {
                        rdm12_sub(FCI4pdm12kern_ms0, pdm1,
                                  rdm3+(i*norb+j)*nnorb*nnorb,
                                  t1ci+(j*norb+i)*na*na, ket,
                                  i+1, norb, na, nlinka, clinka);
                }
        }
        free(pdm1);
        free(pdm2);
}

        free(clinka);

        const int INC1 = 1;
        const char UP = 'U';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        int nna = na * na;
        dgemv_(&TRANS_T, &nna, &nnorb, &D1, t1ci, &nna,
               ket, &INC1, &D0, rdm1, &INC1);
        dsyrk_(&UP, &TRANS_T, &nnorb, &nna,
               &D1, t1ci, &nna, &D0, rdm2, &nnorb);
        free(t1ci);
        _transpose_jikl(rdm2, norb);
}

void FCIrdm3_drv(double *rdm1, double *rdm2, double *rdm3,
                 double *bra, double *ket,
                 int norb, int na, int nb, int nlinka, int nlinkb,
                 int *link_indexa, int *link_indexb)
{
        bra = ket;
        nb = na;
        nlinkb = nlinka;
        link_indexb = link_indexa;

        const int nnorb = norb * norb;
        const long n4 = nnorb * nnorb;
        int ij, i, j;
        double *pdm1;
        double *t1ci = malloc(sizeof(double) * na*nb*nnorb);

        _LinkT *clinka = malloc(sizeof(_LinkT) * nlinka * na);
        compress_link(clinka, link_indexa, norb, na, nlinka);

#pragma omp parallel default(none) \
        shared(t1ci, ket, norb, na, nlinka, clinka), \
        private(i)
#pragma omp for schedule(dynamic, 2) nowait
        for (i = 0; i < na; i++) {
                fill_t1ci(t1ci, ket, na, i, 0, norb, na, nlinka, clinka);
        }

#pragma omp parallel default(none) \
        shared(rdm3, t1ci, ket, norb, na, nlinka, clinka), \
        private(ij, i, j, pdm1)
{
        pdm1 = (double *)malloc(sizeof(double) * nnorb);
#pragma omp for schedule(dynamic, 1) nowait
        for (ij = 0; ij < nnorb; ij++) {
                i = ij / norb;
                j = ij - i * norb;
                rdm12_sub(FCI4pdm12kern_ms0, pdm1, rdm3+ij*n4,
                          t1ci+(j*norb+i)*na*na, ket,
                          i+1, norb, na, nlinka, clinka);
        }
        free(pdm1);
}

        free(clinka);

        const int INC1 = 1;
        const char UP = 'U';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        int nna = na * na;
        dgemv_(&TRANS_T, &nna, &nnorb, &D1, t1ci, &nna,
               ket, &INC1, &D0, rdm1, &INC1);
        dsyrk_(&UP, &TRANS_T, &nnorb, &nna,
               &D1, t1ci, &nna, &D0, rdm2, &nnorb);
        free(t1ci);
        _transpose_jikl(rdm2, norb);
}

