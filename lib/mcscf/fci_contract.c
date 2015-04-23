/*
 *
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))
#define CSUMTHR         1e-28
#define BUFBASE         320

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
typedef struct {
        unsigned int addr;
        unsigned short ia;
        char sign;
        char _padding;
} _LinkT;
#define EXTRACT_IA(I)   (I.ia)
#define EXTRACT_SIGN(I) (I.sign)
#define EXTRACT_ADDR(I) (I.addr)

static void compress_link(_LinkT *clink, int *link_index, int nstr, int nlink)
{
        int i, j, ia, str1, sign;
        for (i = 0; i < nstr; i++) {
                for (j = 0; j < nlink; j++) {
                        ia   = link_index[j*4+0];
                        str1 = link_index[j*4+2];
                        sign = link_index[j*4+3];
                        clink[j].ia = ia;
                        clink[j].sign = sign;
                        clink[j].addr = str1;
                }
                clink += nlink;
                link_index += nlink * 4;
        }
}

// for (16e,16o) ~ 39 MB buffer = 112 * 136 * 8 * 320 KB
static int strb_buflen(int nstr, int nnorb)
{
        return 112;
}


/* 
 * For given stra_id, spread alpah-strings (which can propagate to stra_id)
 * into t1[:nstrb,nnorb] 
 *    str1-of-alpha -> create/annihilate -> str0-of-alpha
 * ci0[:nstra,:nstrb] is contiguous in beta-strings
 * bcount control the number of beta strings to be calculated.
 * for spin=0 system, only lower triangle of the intermediate ci vector
 * needs to be calculated
 */
static double prog_a_t1(double *ci0, double *t1,
                        int bcount, int stra_id, int strb_id,
                        int norb, int nstrb, int nlinka, _LinkT *clink_indexa)
{
        ci0 += strb_id;
        const int nnorb = norb * (norb+1)/2;
        int j, k, ia, str1, sign;
        const _LinkT *tab = clink_indexa + stra_id * nlinka;
        double *pt1, *pci;
        double csum = 0;

        for (j = 0; j < nlinka; j++) {
                ia   = EXTRACT_IA  (tab[j]);
                str1 = EXTRACT_ADDR(tab[j]);
                sign = EXTRACT_SIGN(tab[j]);
                pt1 = t1 + ia;
                pci = ci0 + str1*(uint64_t)nstrb;
                if (sign > 0) {
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
/* 
 * For given stra_id, spread all beta-strings into t1[:nstrb,nnorb] 
 *    all str0-of-beta -> create/annihilate -> str1-of-beta
 * ci0[:nstra,:nstrb] is contiguous in beta-strings
 * bcount control the number of beta strings to be calculated.
 * for spin=0 system, only lower triangle of the intermediate ci vector
 * needs to be calculated
static double prog_b_t1(double *ci0, double *t1,
                        int bcount, int stra_id, int strb_id,
                        int norb, int nstrb, int nlinkb, int *clink_indexb)
{
        const int nnorb = norb * (norb+1)/2;
        int j, ia, str0, str1, sign;
        const _LinkT *tab = clink_indexb + strb_id * nlinkb;
        double *pci = ci0 + stra_id*(uint64_t)nstrb;
        double csum = 0;

        for (str0 = 0; str0 < bcount; str0++) {
                for (j = 0; j < nlinkb; j++) {
                        ia   = EXTRACT_IA  (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        t1[ia] += sign * pci[str1];
                        csum += pci[str1] * pci[str1];
                }
                t1 += nnorb;
                tab += nlinkb;
        }
        return csum;
}
 */

/*
 * prog0_b_t1 is the same to prog_b_t1, except that prog0_b_t1
 * initializes t1 with 0, to reduce data transfer between CPU
 * cache and memory
 */
static double prog0_b_t1(double *ci0, double *t1,
                         int bcount, int stra_id, int strb_id,
                         int norb, int nstrb, int nlinkb, _LinkT *clink_indexb)
{
        const int nnorb = norb * (norb+1)/2;
        int j, ia, str0, str1, sign;
        const _LinkT *tab = clink_indexb + strb_id * nlinkb;
        double *pci = ci0 + stra_id*(uint64_t)nstrb;
        double csum = 0;

        for (str0 = 0; str0 < bcount; str0++) {
                memset(t1, 0, sizeof(double)*nnorb);
                for (j = 0; j < nlinkb; j++) {
                        ia   = EXTRACT_IA  (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        t1[ia] += sign * pci[str1];
                        csum += pci[str1] * pci[str1];
                }
                t1 += nnorb;
                tab += nlinkb;
        }
        return csum;
}


/*
 * spread t1 into ci1
 */
static void spread_a_t1(double *ci1, double *t1,
                        int bcount, int stra_id, int strb_id,
                        int norb, int nstrb, int nlinka, _LinkT *clink_indexa)
{
        ci1 += strb_id;
        const int nnorb = norb * (norb+1)/2;
        int j, k, ia, str1, sign;
        const _LinkT *tab = clink_indexa + stra_id * nlinka;
        double *cp0, *cp1;

        for (j = 0; j < nlinka; j++) {
                ia   = EXTRACT_IA  (tab[j]);
                str1 = EXTRACT_ADDR(tab[j]);
                sign = EXTRACT_SIGN(tab[j]);
                cp0 = t1 + ia;
                cp1 = ci1 + str1*(uint64_t)nstrb;
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
        const int nnorb = norb * (norb+1)/2;
        int j, ia, str0, str1, sign;
        const _LinkT *tab = clink_indexb + strb_id * nlinkb;
        double *pci = ci1 + stra_id * (uint64_t)nstrb;

        for (str0 = 0; str0 < bcount; str0++) {
                for (j = 0; j < nlinkb; j++) {
                        ia   = EXTRACT_IA  (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        pci[str1] += sign * t1[ia];
                }
                t1 += nnorb;
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
        int j, k, ia, str0, str1, sign;
        double *pci0, *pci1;
        double tmp;
        _LinkT *tab;
        _LinkT *clink = malloc(sizeof(_LinkT) * nlinka * nstra);
        compress_link(clink, link_indexa, nstra, nlinka);

        for (str0 = 0; str0 < nstra; str0++) {
                tab = clink + str0 * nlinka;
                for (j = 0; j < nlinka; j++) {
                        ia   = EXTRACT_IA  (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        pci0 = ci0 + str0 * (uint64_t)nstrb;
                        pci1 = ci1 + str1 * (uint64_t)nstrb;
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
        int j, k, ia, str0, str1, sign;
        double *pci1;
        double tmp;
        _LinkT *tab;
        _LinkT *clink = malloc(sizeof(_LinkT) * nlinkb * nstrb);
        compress_link(clink, link_indexb, nstrb, nlinkb);

        for (str0 = 0; str0 < nstra; str0++) {
                pci1 = ci1 + str0 * (uint64_t)nstrb;
                for (k = 0; k < nstrb; k++) {
                        tab = clink + k * nlinkb;
                        tmp = ci0[str0*(uint64_t)nstrb+k];
                        for (j = 0; j < nlinkb; j++) {
                                ia   = EXTRACT_IA  (tab[j]);
                                str1 = EXTRACT_ADDR(tab[j]);
                                sign = EXTRACT_SIGN(tab[j]);
                                if (sign > 0) {
                                        pci1[str1] += tmp * f1e_tril[ia];
                                } else {
                                        pci1[str1] -= tmp * f1e_tril[ia];
                                }
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


static void ctr_rhf2e_kern(double *eri, double *ci0, double *ci1, double *tbuf,
                           int bcount, int stra_id, int strb_id,
                           int norb, int na, int nb, int nlinka, int nlinkb,
                           _LinkT *clink_indexa, _LinkT *clink_indexb)
{
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * (norb+1)/2;
        double *t1 = malloc(sizeof(double) * nnorb*bcount);
        double csum;

        csum = prog0_b_t1(ci0, t1, bcount, stra_id, strb_id,
                          norb, nb, nlinkb, clink_indexb)
             + prog_a_t1(ci0, t1, bcount, stra_id, strb_id,
                         norb, nb, nlinka, clink_indexa);

        if (csum > CSUMTHR) {
                dgemm_(&TRANS_N, &TRANS_N, &nnorb, &bcount, &nnorb,
                       &D1, eri, &nnorb, t1, &nnorb,
                       &D0, tbuf, &nnorb);
                spread_b_t1(ci1, tbuf, bcount, stra_id, strb_id,
                            norb, nb, nlinkb, clink_indexb);
        } else {
                memset(tbuf, 0, sizeof(double)*nnorb*bcount);
        }
        free(t1);
}

/*
 * for give n, m*(m+1)/2 - n*(n+1)/2 ~= base*(base+1)/2
 */
static int _square_pace(int n, int base, int minimal)
{
        float nd = n;
        float nb = base;
        return MAX((int)sqrt(nd*nd+nb*nb), n+minimal);
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
        const int nnorb = norb * (norb+1)/2;
        const int blklenb = strb_buflen(na, nnorb);

        int strk0, strk1, strk, ib, blen;
        int bufbas = MIN(BUFBASE, na);
        double *buf = malloc(sizeof(double)*bufbas*blklenb*nnorb);
        double *pbuf;
        _LinkT *clink = malloc(sizeof(_LinkT) * nlink * na);
        compress_link(clink, link_index, na, nlink);

        memset(ci1, 0, sizeof(double)*na*na);
        for (strk0 = 0, strk1 = na; strk0 < na; strk0 = strk1) {
                strk1 = _square_pace(strk0, bufbas, 1);
                strk1 = MIN(strk1, na);
                for (ib = 0; ib < strk1; ib += blklenb) {
                        blen = MIN(blklenb, strk1-ib);
#pragma omp parallel default(none) \
                shared(eri, ci0, ci1, norb, na, nlink, clink, \
                       strk0, strk1, bufbas, buf, ib, blen), \
                private(strk, pbuf)
#pragma omp for schedule(guided, 1)
/* strk starts from MAX(strk0, ib), because [0:ib,0:ib] have been evaluated */
                        for (strk = MAX(strk0, ib); strk < strk1; strk++) {
                                pbuf = buf + (strk-strk0)*blen*nnorb;
                                ctr_rhf2e_kern(eri, ci0, ci1, pbuf,
                                               MIN(blklenb, strk+1-ib), strk, ib,
                                               norb, na, na, nlink, nlink,
                                               clink, clink);
                        }

/* Note: the bcount diffs in ctr_rhf2e_kern and spread_a_t1.
 * ctr_rhf2e_kern needs strk+1 beta-strings, spread_a_t1 takes strk
 * beta-strings */
                        for (strk = MAX(strk0, ib); strk < strk1; strk++) {
                                pbuf = buf + (strk-strk0)*blen*nnorb;
                                spread_a_t1(ci1, pbuf,
                                            MIN(blklenb,strk-ib), strk, ib,
                                            norb, na, nlink, clink);
                        }
                }
        }
        free(clink);
        free(buf);
}


void FCIcontract_2e_spin1(double *eri, double *ci0, double *ci1,
                          int norb, int na, int nb, int nlinka, int nlinkb,
                          int *link_indexa, int *link_indexb)
{
        const int nnorb = norb * (norb+1)/2;
        const int blklenb = strb_buflen(nb, nnorb);

        int ic, strk1, strk0, strk, ib, blen;
        int bufbas = MIN(BUFBASE, nb);
        double *buf = (double *)malloc(sizeof(double) * bufbas*nnorb*blklenb);
        double *pbuf;
        _LinkT *clinka = malloc(sizeof(_LinkT) * nlinka * na);
        _LinkT *clinkb = malloc(sizeof(_LinkT) * nlinkb * nb);
        compress_link(clinka, link_indexa, na, nlinka);
        compress_link(clinkb, link_indexb, nb, nlinkb);

        memset(ci1, 0, sizeof(double)*na*nb);
        for (strk0 = 0; strk0 < na; strk0 += bufbas) {
                strk1 = MIN(na-strk0, bufbas);
                for (ib = 0; ib < nb; ib += blklenb) {
                        blen = MIN(blklenb, nb-ib);
#pragma omp parallel default(none) \
        shared(eri, ci0, ci1, norb, na, nb, nlinka, nlinkb, \
               clinka, clinkb, buf, strk0, strk1, ib, blen), \
        private(strk, ic, pbuf)
#pragma omp for schedule(static)
                        for (ic = 0; ic < strk1; ic++) {
                                strk = strk0 + ic;
                                pbuf = buf + ic * blen * nnorb;
                                ctr_rhf2e_kern(eri, ci0, ci1, pbuf,
                                               blen, strk, ib,
                                               norb, na, nb, nlinka, nlinkb,
                                               clinka, clinkb);
                        }
// spread alpha-strings in serial mode
                        for (ic = 0; ic < strk1; ic++) {
                                strk = strk0 + ic;
                                pbuf = buf + ic * blen * nnorb;
                                spread_a_t1(ci1, pbuf, blen, strk, ib,
                                            norb, nb, nlinka, clinka);
                        }
                }
        }
        free(clinka);
        free(clinkb);
        free(buf);
}

/*
 * eri_ab is mixed integrals (alpha,alpha|beta,beta), |beta,beta) in small strides
 */
static void ctr_uhf2e_kern(double *eri_aa, double *eri_ab, double *eri_bb,
                           double *ci0, double *ci1, double *tbuf,
                           int bcount, int stra_id, int strb_id,
                           int norb, int na, int nb, int nlinka, int nlinkb,
                           _LinkT *clink_indexa, _LinkT *clink_indexb)
{
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * (norb+1)/2;
        double *t1a = malloc(sizeof(double) * nnorb*bcount*3);
        double *t1b = t1a + nnorb*bcount;
        double *tmp = t1b + nnorb*bcount;
        double csum;

        memset(t1a, 0, sizeof(double)*nnorb*bcount);
        csum = prog0_b_t1(ci0, t1b, bcount, stra_id, strb_id,
                          norb, nb, nlinkb, clink_indexb)
             + prog_a_t1(ci0, t1a, bcount, stra_id, strb_id,
                         norb, nb, nlinka, clink_indexa);

        if (csum > CSUMTHR) {
                dgemm_(&TRANS_T, &TRANS_N, &nnorb, &bcount, &nnorb,
                       &D1, eri_aa, &nnorb, t1a, &nnorb, &D0, tbuf, &nnorb);
                dgemm_(&TRANS_T, &TRANS_N, &nnorb, &bcount, &nnorb,
                       &D1, eri_ab, &nnorb, t1b, &nnorb, &D1, tbuf, &nnorb);

                dgemm_(&TRANS_N, &TRANS_N, &nnorb, &bcount, &nnorb,
                       &D1, eri_ab, &nnorb, t1a, &nnorb, &D0, tmp, &nnorb);
                dgemm_(&TRANS_T, &TRANS_N, &nnorb, &bcount, &nnorb,
                       &D1, eri_bb, &nnorb, t1b, &nnorb, &D1, tmp, &nnorb);
                spread_b_t1(ci1, tmp, bcount, stra_id, strb_id,
                            norb, nb, nlinkb, clink_indexb);
        } else {
                memset(tbuf, 0, sizeof(double)*nnorb*bcount);
        }
        free(t1a);
}

void FCIcontract_uhf2e(double *eri_aa, double *eri_ab, double *eri_bb,
                       double *ci0, double *ci1,
                       int norb, int na, int nb, int nlinka, int nlinkb,
                       int *link_indexa, int *link_indexb)
{
        const int nnorb = norb * (norb+1)/2;
        const int blklenb = strb_buflen(nb, nnorb);

        int ic, strk1, strk0, strk, ib, blen;
        int bufbas = MIN(BUFBASE, nb);
        double *buf = (double *)malloc(sizeof(double) * bufbas*nnorb*blklenb);
        double *pbuf;
        _LinkT *clinka = malloc(sizeof(_LinkT) * nlinka * na);
        _LinkT *clinkb = malloc(sizeof(_LinkT) * nlinkb * nb);
        compress_link(clinka, link_indexa, na, nlinka);
        compress_link(clinkb, link_indexb, nb, nlinkb);

        memset(ci1, 0, sizeof(double)*na*nb);
        for (strk0 = 0; strk0 < na; strk0 += bufbas) {
                strk1 = MIN(na-strk0, bufbas);
                for (ib = 0; ib < nb; ib += blklenb) {
                        blen = MIN(blklenb, nb-ib);
#pragma omp parallel default(none) \
        shared(eri_aa, eri_ab, eri_bb, ci0, ci1, norb, na, nb, nlinka, nlinkb,\
               clinka, clinkb, buf, strk0, strk1, ib, blen), \
        private(strk, ic, pbuf)
#pragma omp for schedule(static)
                        for (ic = 0; ic < strk1; ic++) {
                                strk = strk0 + ic;
                                pbuf = buf + ic * blen * nnorb;
                                ctr_uhf2e_kern(eri_aa, eri_ab, eri_bb, ci0, ci1, pbuf,
                                               blen, strk, ib,
                                               norb, na, nb, nlinka, nlinkb,
                                               clinka, clinkb);
                        }
// spread alpha-strings in serial mode
                        for (ic = 0; ic < strk1; ic++) {
                                strk = strk0 + ic;
                                pbuf = buf + ic * blen * nnorb;
                                spread_a_t1(ci1, pbuf, blen, strk, ib,
                                            norb, nb, nlinka, clinka);
                        }
                }
        }
        free(clinka);
        free(clinkb);
        free(buf);
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
        int ia, ib, j, j0, k0, jk, jk0;
        double e1, e2;
        int *paocc, *pbocc;
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

void FCImake_hdiag(double *hdiag, double *h1e, double *jdiag, double *kdiag,
                   int norb, int na, int nocc, int *occslist)
{
        FCImake_hdiag_uhf(hdiag, h1e, h1e, jdiag, jdiag, jdiag, kdiag, kdiag,
                          norb, na, na, nocc, nocc, occslist, occslist);
}


int FCIpopcount_4(uint64_t x);
int FCIparity(uint64_t string0, uint64_t string1);
//see http://en.wikipedia.org/wiki/Find_first_set
static const int TABLE[] = {
        -1, // 0 
        0 , // 1 
        1 , // 2 
        1 , // 3 
        2 , // 4 
        2 , // 5 
        2 , // 6 
        2 , // 7 
        3 , // 8 
        3 , // 9 
        3 , // 10
        3 , // 11
        3 , // 12
        3 , // 13
        3 , // 14
        3 , // 15
};
// be carefull with (r >> 64), which is not defined in C99 standard
static int first1(uint64_t r)
{
        assert(r > 0);

        uint64_t n = 0;
        while (r != 0) {
                if (r & 0xf) {
                        return n + TABLE[r & 0xf];
                } else {
                        r >>= 4;
                        n += 4;
                }
        }
        return -1;
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
        int i, j, k, pi, pj, pk, pl;
        int n1da, n1db;
        int d2 = norb * norb;
        int d3 = norb * norb * norb;
        uint64_t da, db, str1;
        double tmp;

        for (i = 0; i < np; i++) {
                assert(stra[i] > 0 && strb[i] >> 0);
        for (j = 0; j < i; j++) {
                da = stra[i] ^ stra[j];
                db = strb[i] ^ strb[j];
                n1da = FCIpopcount_4(da);
                n1db = FCIpopcount_4(db);
                switch (n1da) {
                case 0: switch (n1db) {
                        case 2:
                        pi = first1(db & strb[i]);
                        pj = first1(db & strb[j]);
                        tmp = h1e_b[pi*norb+pj];
                        for (k = 0; k < norb; k++) {
                                if (stra[i] & (1UL<<k)) {
                                        tmp += g2e_ab[pi*norb+pj+k*d3+k*d2];
                                }
                                if (strb[i] & (1UL<<k)) {
                                        tmp += g2e_bb[pi*d3+pj*d2+k*norb+k]
                                             - g2e_bb[pi*d3+k*d2+k*norb+pj];
                                }
                        }
                        if (FCIparity(strb[j], strb[i]) > 0) {
                                h0[i*np+j] = tmp;
                        } else {
                                h0[i*np+j] = -tmp;
                        } break;

                        case 4:
                        pi = first1(db & strb[i]);
                        pj = first1(db & strb[j]);
                        pk = first1((db & strb[i]) ^ (1UL<<pi));
                        pl = first1((db & strb[j]) ^ (1UL<<pj));
                        str1 = strb[j] ^ (1UL<<pi) ^ (1UL<<pj);
                        if (FCIparity(strb[j], str1)
                           *FCIparity(str1, strb[i]) > 0) {
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
                                if (strb[i] & (1UL<<k)) {
                                        tmp += g2e_ab[pi*d3+pj*d2+k*norb+k];
                                }
                                if (stra[i] & (1UL<<k)) {
                                        tmp += g2e_aa[pi*d3+pj*d2+k*norb+k]
                                             - g2e_aa[pi*d3+k*d2+k*norb+pj];
                                }
                        }
                        if (FCIparity(stra[j], stra[i]) > 0) {
                                h0[i*np+j] = tmp;
                        } else {
                                h0[i*np+j] = -tmp;
                        } break;

                        case 2:
                        pi = first1(da & stra[i]);
                        pj = first1(da & stra[j]);
                        pk = first1(db & strb[i]);
                        pl = first1(db & strb[j]);
                        if (FCIparity(stra[j], stra[i])
                           *FCIparity(strb[j], strb[i]) > 0) {
                                h0[i*np+j] = g2e_ab[pi*d3+pj*d2+pk*norb+pl];
                        } else {
                                h0[i*np+j] =-g2e_ab[pi*d3+pj*d2+pk*norb+pl];
                        } } break;
                case 4: switch (n1db) {
                        case 0:
                        pi = first1(da & stra[i]);
                        pj = first1(da & stra[j]);
                        pk = first1((da & stra[i]) ^ (1UL<<pi));
                        pl = first1((da & stra[j]) ^ (1UL<<pj));
                        str1 = stra[j] ^ (1UL<<pi) ^ (1UL<<pj);
                        if (FCIparity(stra[j], str1)
                           *FCIparity(str1, stra[i]) > 0) {
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

void FCIpspace_h0tril(double *h0, double *h1e, double *g2e,
                      uint64_t *stra, uint64_t *strb,
                      int norb, int np)
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
static void ctr_rhf2esym_kern(double *eri, double *ci0, double *ci1, double *tbuf,
                              int bcount, int stra_id, int strb_id,
                              int norb, int na, int nb, int nlinka, int nlinkb,
                              _LinkT *clink_indexa, _LinkT *clink_indexb,
                              int *dimirrep, int totirrep)
{
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * (norb+1)/2;
        int ir, p0;
        double *t1 = malloc(sizeof(double) * nnorb*bcount);
        double csum;

        csum = prog0_b_t1(ci0, t1, bcount, stra_id, strb_id,
                          norb, nb, nlinkb, clink_indexb)
             + prog_a_t1(ci0, t1, bcount, stra_id, strb_id,
                         norb, nb, nlinka, clink_indexa);

        if (csum > CSUMTHR) {
                for (ir = 0, p0 = 0; ir < totirrep; ir++) {
                        dgemm_(&TRANS_N, &TRANS_N,
                               dimirrep+ir, &bcount, dimirrep+ir,
                               &D1, eri+p0*nnorb+p0, &nnorb, t1+p0, &nnorb,
                               &D0, tbuf+p0, &nnorb);
                        p0 += dimirrep[ir];
                }
                spread_b_t1(ci1, tbuf, bcount, stra_id, strb_id,
                            norb, nb, nlinkb, clink_indexb);
        } else {
                memset(tbuf, 0, sizeof(double)*nnorb*bcount);
        }
        free(t1);
}

void FCIcontract_2e_spin1_symm(double *eri, double *ci0, double *ci1,
                               int norb, int na, int nb, int nlinka, int nlinkb,
                               int *link_indexa, int *link_indexb,
                               int *dimirrep, int totirrep)
{
        const int nnorb = norb * (norb+1)/2;
        const int blklenb = strb_buflen(nb, nnorb);

        int ic, strk1, strk0, strk, ib, blen;
        int bufbas = MIN(BUFBASE, nb);
        double *buf = (double *)malloc(sizeof(double) * bufbas*nnorb*blklenb);
        double *pbuf;
        _LinkT *clinka = malloc(sizeof(_LinkT) * nlinka * na);
        _LinkT *clinkb = malloc(sizeof(_LinkT) * nlinkb * nb);
        compress_link(clinka, link_indexa, na, nlinka);
        compress_link(clinkb, link_indexb, nb, nlinkb);

        memset(ci1, 0, sizeof(double)*na*nb);
        for (strk0 = 0; strk0 < na; strk0 += bufbas) {
                strk1 = MIN(na-strk0, bufbas);
                for (ib = 0; ib < nb; ib += blklenb) {
                        blen = MIN(blklenb, nb-ib);
#pragma omp parallel default(none) \
                shared(eri, ci0, ci1, norb, na, nb, nlinka, nlinkb, \
                       clinka, clinkb, dimirrep, totirrep, \
                       buf, strk0, strk1, ib, blen), \
                private(strk, ic, pbuf)
#pragma omp for schedule(static)
                        for (ic = 0; ic < strk1; ic++) {
                                strk = strk0 + ic;
                                pbuf = buf + ic * blen * nnorb;
                                ctr_rhf2esym_kern(eri, ci0, ci1, pbuf,
                                                  blen, strk, ib,
                                                  norb, na, nb, nlinka, nlinkb,
                                                  clinka, clinkb,
                                                  dimirrep, totirrep);
                        }
// spread alpha-strings in serial mode
                        for (ic = 0; ic < strk1; ic++) {
                                strk = strk0 + ic;
                                pbuf = buf + ic * blen * nnorb;
                                spread_a_t1(ci1, pbuf, blen, strk, ib,
                                            norb, nb, nlinka, clinka);
                        }
                }
        }
        free(clinka);
        free(clinkb);
        free(buf);
}

void FCIcontract_2e_spin0_symm(double *eri, double *ci0, double *ci1,
                               int norb, int na, int nlink, int *link_index,
                               int *dimirrep, int totirrep)
{
        const int nnorb = norb * (norb+1)/2;
        const int blklenb = strb_buflen(na, nnorb);

        int strk0, strk1, strk, ib, blen;
        int bufbas = MIN(BUFBASE, na);
        double *buf = malloc(sizeof(double)*bufbas*blklenb*nnorb);
        double *pbuf;
        _LinkT *clink = malloc(sizeof(_LinkT) * nlink * na);
        compress_link(clink, link_index, na, nlink);

        memset(ci1, 0, sizeof(double)*na*na);
        for (strk0 = 0, strk1 = na; strk0 < na; strk0 = strk1) {
                strk1 = _square_pace(strk0, bufbas, 1);
                strk1 = MIN(strk1, na);
                for (ib = 0; ib < strk1; ib += blklenb) {
                        blen = MIN(blklenb, strk1-ib);
#pragma omp parallel default(none) \
        shared(eri, ci0, ci1, norb, na, nlink, clink, \
               dimirrep, totirrep, strk0, strk1, bufbas, buf, ib, blen), \
        private(strk, pbuf)
#pragma omp for schedule(guided, 1)
                        for (strk = MAX(strk0, ib); strk < strk1; strk++) {
                                pbuf = buf + (strk-strk0)*blen*nnorb;
                                ctr_rhf2esym_kern(eri, ci0, ci1, pbuf,
                                                  MIN(blklenb, strk+1-ib), strk, ib,
                                                  norb, na, na, nlink, nlink,
                                                  clink, clink,
                                                  dimirrep, totirrep);
                }

/* Note: the bcount diffs in ctr_rhf2e_kern and spread_a_t1.
 * ctr_rhf2e_kern needs strk+1 beta-strings, spread_a_t1 takes strk
 * beta-strings */
                        for (strk = MAX(strk0, ib); strk < strk1; strk++) {
                                pbuf = buf + (strk-strk0)*blen*nnorb;
                                spread_a_t1(ci1, pbuf,
                                            MIN(blklenb,strk-ib), strk, ib,
                                            norb, na, nlink, clink);
                        }
                }
        }
        free(clink);
        free(buf);
}

