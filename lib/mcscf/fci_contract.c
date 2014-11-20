/*
 *
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))
#define CSUMTHR         1e-14


/* 
 * For given stra_id, spread alpah-strings (which can propagate to stra_id)
 * into t1[:nstrb,nnorb] 
 *    str1-of-alpha -> create/annihilate -> str0-of-alpha
 * ci0[:nstra,:nstrb] is contiguous in beta-strings
 * fillcnt control the number of beta strings to be calculated.
 * for spin=0 system, only lower triangle of the intermediate ci vector
 * needs to be calculated
 */
static double prog_a_t1(double *ci0, double *t1, int fillcnt, int stra_id,
                        int norb, int nstrb, int nlinka, int *link_indexa)
{
        const int nnorb = norb * (norb+1)/2;
        int j, k, ia, str1, sign;
        const int *tab = link_indexa + stra_id * nlinka * 4;
        double *pt1, *pci;
        double csum = 0;

        for (j = 0; j < nlinka; j++) {
                ia   = tab[j*4+0];
                str1 = tab[j*4+2];
                sign = tab[j*4+3];
                pci = ci0 + str1*nstrb;
                pt1 = t1 + ia;
                if (sign > 0) {
                        for (k = 0; k < fillcnt-1; k+=2) {
                                pt1[k*nnorb] += pci[k];
                                pt1[k*nnorb+nnorb] += pci[k+1];
                                csum += fabs(pci[k]) + fabs(pci[k+1]);
                        }
                } else {
                        for (k = 0; k < fillcnt-1; k+=2) {
                                pt1[k*nnorb] -= pci[k];
                                pt1[k*nnorb+nnorb] -= pci[k+1];
                                csum += fabs(pci[k]) + fabs(pci[k+1]);
                        }
                }
                if (k < fillcnt) {
                        pt1[k*nnorb] += sign * pci[k];
                        csum += fabs(pci[k]);
                }
        }
        return csum;
}
/* 
 * For given stra_id, spread all beta-strings into t1[:nstrb,nnorb] 
 *    all str0-of-beta -> create/annihilate -> str1-of-beta
 * ci0[:nstra,:nstrb] is contiguous in beta-strings
 * fillcnt control the number of beta strings to be calculated.
 * for spin=0 system, only lower triangle of the intermediate ci vector
 * needs to be calculated
 */
static double prog_b_t1(double *ci0, double *t1, int fillcnt, int stra_id,
                        int norb, int nstrb, int nlinkb, int *link_indexb)
{
        const int nnorb = norb * (norb+1)/2;
        int j, ia, str0, str1, sign;
        const int *tab;
        double *pt1, *pci;
        double csum = 0;

        pci = ci0 + stra_id*nstrb;
        for (str0 = 0; str0 < fillcnt; str0++) {
                tab = link_indexb + str0 * nlinkb * 4;
                pt1 = t1 + str0*nnorb;
                for (j = 0; j < nlinkb; j++) {
                        ia   = tab[j*4+0];
                        str1 = tab[j*4+2];
                        sign = tab[j*4+3];
                        pt1[ia] += sign * pci[str1];
                        csum += fabs(pci[str1]);
                }
        }
        return csum;
}

/*
 * spread t1 into ci1
 */
static void spread_a_t1(double *ci1, double *t1, int fillcnt, int stra_id,
                        int norb, int nstrb, int nlinka, int *link_indexa)
{
        const int nnorb = norb * (norb+1)/2;
        int j, k, ia, str1, sign;
        int *tab = link_indexa + stra_id * nlinka * 4;
        double *cp0, *cp1;

        for (j = 0; j < nlinka; j++) {
                ia   = tab[j*4+0];
                str1 = tab[j*4+2];
                sign = tab[j*4+3];
                cp0 = t1 + ia;
                cp1 = ci1 + str1*nstrb;
                if (sign > 0) {
                        for (k = 0; k < fillcnt; k++) {
                                cp1[k] += cp0[k*nnorb];
                        }
                } else {
                        for (k = 0; k < fillcnt; k++) {
                                cp1[k] -= cp0[k*nnorb];
                        }
                }
        }
}

static void spread_b_t1(double *ci1, double *t1, int fillcnt, int stra_id,
                        int norb, int nstrb, int nlinkb, int *link_indexb)
{
        const int nnorb = norb * (norb+1)/2;
        int j, ia, str0, str1, sign;
        const int *tab;
        double *pt1, *pci;

        pci = ci1 + stra_id * nstrb;
        for (str0 = 0; str0 < fillcnt; str0++) {
                tab = link_indexb + str0 * nlinkb * 4;
                pt1 = t1 + str0 * nnorb;
                for (j = 0; j < nlinkb; j++) {
                        ia   = tab[j*4+0];
                        str1 = tab[j*4+2];
                        sign = tab[j*4+3];
                        pci[str1] += sign * pt1[ia];
                }
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
        int *tab;
        double *pci0, *pci1;
        double tmp;

        for (str0 = 0; str0 < nstra; str0++) {
                tab = link_indexa + str0 * nlinka * 4;
                for (j = 0; j < nlinka; j++) {
                        ia   = tab[j*4+0];
                        str1 = tab[j*4+2];
                        sign = tab[j*4+3];
                        pci0 = ci0 + str0 * nstrb;
                        pci1 = ci1 + str1 * nstrb;
                        tmp  = sign * f1e_tril[ia];
                        for (k = 0; k < nstrb; k++) {
                                pci1[k] += tmp * pci0[k];
                        }
                }
        }
}

/*
 * f1e_tril is the 1e hamiltonian for spin beta
 */
void FCIcontract_b_1e(double *f1e_tril, double *ci0, double *ci1,
                      int norb, int nstra, int nstrb, int nlinka, int nlinkb,
                      int *link_indexa, int *link_indexb)
{
        int j, k, ia, str0, str1, sign;
        int *tab;
        double *pci1;
        double tmp;

        for (str0 = 0; str0 < nstra; str0++) {
                pci1 = ci1 + str0 * nstrb;
                for (k = 0; k < nstrb; k++) {
                        tab = link_indexb + k * nlinkb * 4;
                        tmp = ci0[str0*nstrb+k];
                        for (j = 0; j < nlinkb; j++) {
                                ia   = tab[j*4+0];
                                str1 = tab[j*4+2];
                                sign = tab[j*4+3];
                                if (sign > 0) {
                                        pci1[str1] += tmp * f1e_tril[ia];
                                } else {
                                        pci1[str1] -= tmp * f1e_tril[ia];
                                }
                        }
                }
        }
}

void FCIcontract_1e_spin0(double *f1e_tril, double *ci0, double *ci1,
                          int norb, int na, int nlink, int *link_index)
{
        memset(ci1, 0, sizeof(double)*na*na);
        FCIcontract_a_1e(f1e_tril, ci0, ci1, norb, na, na, nlink, nlink,
                         link_index, link_index);
}

void FCIcontract_1e_ms0(double *f1e_tril, double *ci0, double *ci1,
                        int norb, int na, int nlink, int *link_index)
{
        memset(ci1, 0, sizeof(double)*na*na);
        FCIcontract_a_1e(f1e_tril, ci0, ci1, norb, na, na, nlink, nlink,
                         link_index, link_index);
        FCIcontract_b_1e(f1e_tril, ci0, ci1, norb, na, na, nlink, nlink,
                         link_index, link_index);
}


static void ctr_rhf2e_kern(double *eri, double *ci0, double *ci1, double *tbuf,
                           int fillcnt, int stra_id, int norb, int na, int nb,
                           int nlinka, int nlinkb,
                           int *link_indexa, int *link_indexb)
{
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * (norb+1)/2;
        double *t1 = malloc(sizeof(double) * nnorb*fillcnt);
        double csum;

        memset(t1, 0, sizeof(double)*nnorb*fillcnt);
        csum = prog_a_t1(ci0, t1, fillcnt, stra_id, norb, nb, nlinka, link_indexa)
             + prog_b_t1(ci0, t1, fillcnt, stra_id, norb, nb, nlinkb, link_indexb);

        if (csum > CSUMTHR) {
                dgemm_(&TRANS_T, &TRANS_N, &nnorb, &fillcnt, &nnorb,
                       &D1, eri, &nnorb, t1, &nnorb, &D0, tbuf, &nnorb);
                spread_b_t1(ci1, tbuf, fillcnt, stra_id, norb, nb, nlinkb,
                            link_indexb);
        } else {
                memset(tbuf, 0, sizeof(double)*nnorb*fillcnt);
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
 * buf_size in MB
 * FCIcontract_2e_spin0 only compute half of the contraction, due to the
 * symmetry between alpha and beta spin.  The right contracted ci vector
 * is (ci1+ci1.T)
 */
void FCIcontract_2e_spin0(double *eri, double *ci0, double *ci1,
                          int norb, int na, int nlink, int *link_index,
                          int buf_size)
{
        const int nnorb = norb * (norb+1)/2;
        int nthreads = 1;
#if defined HAVE_OPENMP
#pragma omp parallel shared(nthreads)
        nthreads = omp_get_num_threads();
#endif

        int strk0, strk1, strk;
        long blk_base = MAX(sqrt((((long)buf_size)<<20)/8/nnorb*2), nthreads);
        blk_base = MIN(blk_base, na);
        double *buf = malloc(sizeof(double)*blk_base*(blk_base+1)*nnorb/2);
        double *pbuf;
        long off;

        memset(ci1, 0, sizeof(double)*na*na);
        for (strk0 = 0, strk1 = na; strk0 < na; strk0 = strk1) {
                strk1 = _square_pace(strk0, blk_base, nthreads);
                strk1 = MIN(strk1, na);
#pragma omp parallel default(none) \
        shared(eri, ci0, ci1, norb, na, nlink, link_index, \
               nthreads, strk0, strk1, blk_base, buf), \
        private(strk, off, pbuf)
#pragma omp for schedule(guided, 1)
                for (strk = strk0; strk < strk1; strk++) {
                        //pbuf = buf; ; pbuf += nnorb*(strk+1);
                        off = ((long)(strk-strk0))*(strk+strk0+1)/2;
                        pbuf = buf + off*nnorb;
                        ctr_rhf2e_kern(eri, ci0, ci1, pbuf,
                                       strk+1, strk, norb, na, na, nlink, nlink,
                                       link_index, link_index);
                }

/* Note: the fillcnt diffs in ctr_rhf2e_kern and spread_a_t1.
 * ctr_rhf2e_kern needs strk+1 beta-strings, spread_a_t1 takes strk
 * beta-strings */
                for (strk = strk0; strk < strk1; strk++) {
                        off = ((long)(strk-strk0))*(strk+strk0+1)/2;
                        pbuf = buf + off*nnorb;
                        spread_a_t1(ci1, pbuf, strk, strk, norb, na, nlink,
                                    link_index);
                }
        }
        free(buf);
}


void FCIcontract_rhf2e_spin1(double *eri, double *ci0, double *ci1,
                             int norb, int na, int nb, int nlinka, int nlinkb,
                             int *link_indexa, int *link_indexb, int buf_size)
{
        const int nnorb = norb * (norb+1)/2;
        int len_blk = 1;
#if defined HAVE_OPENMP
#pragma omp parallel shared(len_blk)
        len_blk = omp_get_num_threads();
#endif

        int max_buflen = MAX((((long)buf_size)<<20)/8/nnorb/nb, len_blk);
        len_blk = (int)(max_buflen/len_blk) * len_blk;
        len_blk = MIN(len_blk, na);
        double *buf = (double *)malloc(sizeof(double) * len_blk*nnorb*nb);
        double *pbuf;

        int ic, ic1, strk0, strk;

        memset(ci1, 0, sizeof(double)*na*nb);
        for (strk0 = 0; strk0 < na; strk0 += len_blk) {
                ic1 = MIN(na-strk0, len_blk);
#pragma omp parallel default(none) \
        shared(eri, ci0, ci1, norb, na, nb, nlinka, nlinkb, \
               link_indexa, link_indexb, buf, strk0, ic1), \
        private(strk, ic)
#pragma omp for schedule(static)
                for (ic = 0; ic < ic1; ic++) {
                        strk = strk0 + ic;
                        ctr_rhf2e_kern(eri, ci0, ci1, buf+ic*nnorb*nb,
                                       nb, strk, norb, na, nb, nlinka, nlinkb,
                                       link_indexa, link_indexb);
                }
// spread alpha-strings in serial mode
                for (ic = 0; ic < ic1; ic++) {
                        strk = strk0 + ic;
                        pbuf = buf + ic * nnorb * nb;
                        spread_a_t1(ci1, pbuf, nb, strk, norb, nb, nlinka,
                                    link_indexa);
                }
        }
        free(buf);
}

void FCIcontract_2e_ms0(double *eri, double *ci0, double *ci1,
                        int norb, int na, int nlink, int *link_index,
                        int buf_size)
{
        FCIcontract_rhf2e_spin1(eri, ci0, ci1, norb, na, na, nlink, nlink,
                                link_index, link_index, buf_size);
}


/*
 * eri_ab is mixed integrals (alpha,alpha|beta,beta), |beta,beta) in small strides
 */
static void ctr_uhf2e_kern(double *eri_aa, double *eri_ab, double *eri_bb,
                           double *ci0, double *ci1, double *tbuf,
                           int stra_id, int norb, int na, int nb,
                           int nlinka, int nlinkb,
                           int *link_indexa, int *link_indexb)
{
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * (norb+1)/2;
        double *t1a = malloc(sizeof(double) * nnorb*nb*3);
        double *t1b = t1a + nnorb*nb;
        double *tmp = t1b + nnorb*nb;
        double csum;

        memset(t1a, 0, sizeof(double)*nnorb*nb*2);
        csum = prog_a_t1(ci0, t1a, nb, stra_id, norb, nb, nlinka, link_indexa)
             + prog_b_t1(ci0, t1b, nb, stra_id, norb, nb, nlinkb, link_indexb);

        if (csum > CSUMTHR) {
                dgemm_(&TRANS_T, &TRANS_N, &nnorb, &nb, &nnorb,
                       &D1, eri_aa, &nnorb, t1a, &nnorb, &D0, tbuf, &nnorb);
                dgemm_(&TRANS_T, &TRANS_N, &nnorb, &nb, &nnorb,
                       &D1, eri_ab, &nnorb, t1b, &nnorb, &D1, tbuf, &nnorb);

                dgemm_(&TRANS_N, &TRANS_N, &nnorb, &nb, &nnorb,
                       &D1, eri_ab, &nnorb, t1a, &nnorb, &D0, tmp, &nnorb);
                dgemm_(&TRANS_T, &TRANS_N, &nnorb, &nb, &nnorb,
                       &D1, eri_bb, &nnorb, t1b, &nnorb, &D1, tmp, &nnorb);
                spread_b_t1(ci1, tmp, nb, stra_id, norb, nb, nlinkb,
                            link_indexb);
        } else {
                memset(tbuf, 0, sizeof(double)*nnorb*nb);
        }
        free(t1a);
}

void FCIcontract_uhf2e(double *eri_aa, double *eri_ab, double *eri_bb,
                       double *ci0, double *ci1,
                       int norb, int na, int nb, int nlinka, int nlinkb,
                       int *link_indexa, int *link_indexb, int buf_size)
{
        const int nnorb = norb * (norb+1)/2;
        int len_blk = 1;
#if defined HAVE_OPENMP
#pragma omp parallel shared(len_blk)
        len_blk = omp_get_num_threads();
#endif

        int max_buflen = MAX((((long)buf_size)<<20)/8/nnorb/nb, len_blk);
        len_blk = (int)(max_buflen/len_blk) * len_blk;
        len_blk = MIN(len_blk, na);
        double *buf = (double *)malloc(sizeof(double) * len_blk*nnorb*nb);
        double *pbuf;

        int ic, ic1, strk0, strk;

        memset(ci1, 0, sizeof(double)*na*nb);
        for (strk0 = 0; strk0 < na; strk0 += len_blk) {
                ic1 = MIN(na-strk0, len_blk);
#pragma omp parallel default(none) \
        shared(eri_aa, eri_ab, eri_bb, ci0, ci1, norb, na, nb, nlinka, nlinkb,\
               link_indexa, link_indexb, buf, strk0, ic1), \
        private(strk, ic)
#pragma omp for schedule(static)
                for (ic = 0; ic < ic1; ic++) {
                        strk = strk0 + ic;
                        ctr_uhf2e_kern(eri_aa, eri_ab, eri_bb,
                                       ci0, ci1, buf+ic*nnorb*nb,
                                       strk, norb, na, nb, nlinka, nlinkb,
                                       link_indexa, link_indexb);
                }
// spread alpha-strings in serial mode
                for (ic = 0; ic < ic1; ic++) {
                        strk = strk0 + ic;
                        pbuf = buf + ic * nnorb * nb;
                        spread_a_t1(ci1, pbuf, nb, strk, norb, nb, nlinka,
                                    link_indexa);
                }
        }
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


int FCIpopcount_4(unsigned long x);
int FCIparity(unsigned long string0, unsigned long string1);
//see http://en.wikipedia.org/wiki/Find_first_set
static int first1(unsigned long r)
{
        int n = 1;
        while (r >> n) {
            n++;
        }
        return n-1;
}


/*************************************************
 * pspace Hamiltonian, ref CPL, 169, 463
 *************************************************/
/*
 * sub-space Hamiltonian (tril part) of the determinants (stra,strb)
 */

void FCIpspace_h0tril_uhf(double *h0, double *h1e_a, double *h1e_b,
                          double *g2e_aa, double *g2e_ab, double *g2e_bb,
                          unsigned long *stra, unsigned long *strb,
                          int norb, int np)
{
        int i, j, k, pi, pj, pk, pl;
        int n1da, n1db;
        int d2 = norb * norb;
        int d3 = norb * norb * norb;
        unsigned long da, db, str1;
        double tmp;

        for (i = 0; i < np; i++) {
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
                                if (stra[i] & (1<<k)) {
                                        tmp += g2e_ab[pi*norb+pj+k*d3+k*d2];
                                }
                                if (strb[i] & (1<<k)) {
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
                        pk = first1((db & strb[i]) ^ (1<<pi));
                        pl = first1((db & strb[j]) ^ (1<<pj));
                        str1 = strb[j] ^ (1<<pi) ^ (1<<pj);
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
                                if (strb[i] & (1<<k)) {
                                        tmp += g2e_ab[pi*d3+pj*d2+k*norb+k];
                                }
                                if (stra[i] & (1<<k)) {
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
                        pk = first1((da & stra[i]) ^ (1<<pi));
                        pl = first1((da & stra[j]) ^ (1<<pj));
                        str1 = stra[j] ^ (1<<pi) ^ (1<<pj);
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

/*
void FCIpspace_h0tril(double *h0, double *h1e, double *g2e,
                      unsigned long *stra, unsigned long *strb,
                      int norb, int np)
{
        int i, j, k, pi, pj, pk, pl;
        int n1da, n1db;
        int d2 = norb * norb;
        int d3 = norb * norb * norb;
        unsigned long da, db, str1;
        double tmp;

        for (i = 0; i < np; i++) {
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
                        tmp = h1e[pi*norb+pj];
                        for (k = 0; k < norb; k++) {
                                if (stra[i] & (1<<k)) {
                                        tmp += g2e[pi*d3+pj*d2+k*norb+k];
                                }
                                if (strb[i] & (1<<k)) {
                                        tmp += g2e[pi*d3+pj*d2+k*norb+k]
                                             - g2e[pi*d3+k*d2+k*norb+pj];
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
                        pk = first1((db & strb[i]) ^ (1<<pi));
                        pl = first1((db & strb[j]) ^ (1<<pj));
                        str1 = strb[j] ^ (1<<pi) ^ (1<<pj);
                        if (FCIparity(strb[j], str1)
                           *FCIparity(str1, strb[i]) > 0) {
                                h0[i*np+j] = g2e[pi*d3+pj*d2+pk*norb+pl]
                                           - g2e[pi*d3+pl*d2+pk*norb+pj];
                        } else {
                                h0[i*np+j] =-g2e[pi*d3+pj*d2+pk*norb+pl]
                                           + g2e[pi*d3+pl*d2+pk*norb+pj];
                        } } break;
                case 2: switch (n1db) {
                        case 0:
                        pi = first1(da & stra[i]);
                        pj = first1(da & stra[j]);
                        tmp = h1e[pi*norb+pj];
                        for (k = 0; k < norb; k++) {
                                if (strb[i] & (1<<k)) {
                                        tmp += g2e[pi*d3+pj*d2+k*norb+k];
                                }
                                if (stra[i] & (1<<k)) {
                                        tmp += g2e[pi*d3+pj*d2+k*norb+k]
                                             - g2e[pi*d3+k*d2+k*norb+pj];
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
                                h0[i*np+j] = g2e[pi*d3+pj*d2+pk*norb+pl];
                        } else {
                                h0[i*np+j] =-g2e[pi*d3+pj*d2+pk*norb+pl];
                        } } break;
                case 4: switch (n1db) {
                        case 0:
                        pi = first1(da & stra[i]);
                        pj = first1(da & stra[j]);
                        pk = first1((da & stra[i]) ^ (1<<pi));
                        pl = first1((da & stra[j]) ^ (1<<pj));
                        str1 = stra[j] ^ (1<<pi) ^ (1<<pj);
                        if (FCIparity(stra[j], str1)
                           *FCIparity(str1, stra[i]) > 0) {
                                h0[i*np+j] = g2e[pi*d3+pj*d2+pk*norb+pl]
                                           - g2e[pi*d3+pl*d2+pk*norb+pj];
                        } else {
                                h0[i*np+j] =-g2e[pi*d3+pj*d2+pk*norb+pl]
                                           + g2e[pi*d3+pl*d2+pk*norb+pj];
                        }
                        } break;
                }
        } }
}
*/

void FCIpspace_h0tril(double *h0, double *h1e, double *g2e,
                      unsigned long *stra, unsigned long *strb,
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
                              int fillcnt, int stra_id, int norb, int na, int nb,
                              int nlinka, int nlinkb,
                              int *link_indexa, int *link_indexb,
                              int *dimirrep, int totirrep)
{
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * (norb+1)/2;
        int ir, p0;
        double *t1 = malloc(sizeof(double) * nnorb*fillcnt);
        double csum;

        memset(t1, 0, sizeof(double)*nnorb*fillcnt);
        csum = prog_a_t1(ci0, t1, fillcnt, stra_id, norb, nb, nlinka, link_indexa)
             + prog_b_t1(ci0, t1, fillcnt, stra_id, norb, nb, nlinkb, link_indexb);

        if (csum > CSUMTHR) {
                for (ir = 0, p0 = 0; ir < totirrep; ir++) {
                        dgemm_(&TRANS_T, &TRANS_N,
                               dimirrep+ir, &fillcnt, dimirrep+ir,
                               &D1, eri+p0*nnorb+p0, &nnorb, t1+p0, &nnorb,
                               &D0, tbuf+p0, &nnorb);
                        p0 += dimirrep[ir];
                }
                spread_b_t1(ci1, tbuf, fillcnt, stra_id, norb, nb, nlinkb,
                            link_indexb);
        } else {
                memset(tbuf, 0, sizeof(double)*nnorb*fillcnt);
        }
        free(t1);
}

void FCIcontract_rhf2e_spin1_symm(double *eri, double *ci0, double *ci1,
                                  int norb, int na, int nb, int nlinka, int nlinkb,
                                  int *link_indexa, int *link_indexb,
                                  int *dimirrep, int totirrep, int buf_size)
{
        const int nnorb = norb * (norb+1)/2;
        int len_blk = 1;
#if defined HAVE_OPENMP
#pragma omp parallel shared(len_blk)
        len_blk = omp_get_num_threads();
#endif

        int max_buflen = MAX((((long)buf_size)<<20)/8/nnorb/nb, len_blk);
        len_blk = (int)(max_buflen/len_blk) * len_blk;
        len_blk = MIN(len_blk, na);
        double *buf = (double *)malloc(sizeof(double) * len_blk*nnorb*nb);
        double *pbuf;

        int ic, ic1, strk0, strk;

        memset(ci1, 0, sizeof(double)*na*nb);
        for (strk0 = 0; strk0 < na; strk0 += len_blk) {
                ic1 = MIN(na-strk0, len_blk);
#pragma omp parallel default(none) \
        shared(eri, ci0, ci1, norb, na, nb, nlinka, nlinkb, \
               link_indexa, link_indexb, dimirrep, totirrep, \
               buf, strk0, ic1), \
        private(strk, ic)
#pragma omp for schedule(static)
                for (ic = 0; ic < ic1; ic++) {
                        strk = strk0 + ic;
                        ctr_rhf2esym_kern(eri, ci0, ci1, buf+ic*nnorb*nb,
                                          nb, strk, norb, na, nb, nlinka, nlinkb,
                                          link_indexa, link_indexb,
                                          dimirrep, totirrep);
                }
// spread alpha-strings in serial mode
                for (ic = 0; ic < ic1; ic++) {
                        strk = strk0 + ic;
                        pbuf = buf + ic * nnorb * nb;
                        spread_a_t1(ci1, pbuf, nb, strk, norb, nb, nlinka,
                                    link_indexa);
                }
        }
        free(buf);
}

void FCIcontract_2e_ms0_symm(double *eri, double *ci0, double *ci1,
                             int norb, int na, int nlink, int *link_index,
                             int *dimirrep, int totirrep, int buf_size)
{
        FCIcontract_rhf2e_spin1_symm(eri, ci0, ci1, norb, na, na, nlink, nlink,
                                     link_index, link_index, dimirrep, totirrep,
                                     buf_size);
}

void FCIcontract_2e_spin0_symm(double *eri, double *ci0, double *ci1,
                               int norb, int na, int nlink, int *link_index,
                               int *dimirrep, int totirrep, int buf_size)
{
        const int nnorb = norb * (norb+1)/2;
        int nthreads = 1;
#if defined HAVE_OPENMP
#pragma omp parallel shared(nthreads)
        nthreads = omp_get_num_threads();
#endif

        int strk0, strk1, strk;
        long blk_base = MAX(sqrt((((long)buf_size)<<20)/8/nnorb*2), nthreads);
        blk_base = MIN(blk_base, na);
        double *buf = malloc(sizeof(double)*blk_base*(blk_base+1)*nnorb/2);
        double *pbuf;
        long off;

        memset(ci1, 0, sizeof(double)*na*na);
        for (strk0 = 0, strk1 = na; strk0 < na; strk0 = strk1) {
                strk1 = _square_pace(strk0, blk_base, nthreads);
                strk1 = MIN(strk1, na);
#pragma omp parallel default(none) \
        shared(eri, ci0, ci1, norb, na, nlink, link_index, \
               dimirrep, totirrep, nthreads, strk0, strk1, blk_base, buf), \
        private(strk, off, pbuf)
#pragma omp for schedule(guided, 1)
                for (strk = strk0; strk < strk1; strk++) {
                        //pbuf = buf; ; pbuf += nnorb*(strk+1);
                        off = ((long)(strk-strk0))*(strk+strk0+1)/2;
                        pbuf = buf + off*nnorb;
                        ctr_rhf2esym_kern(eri, ci0, ci1, pbuf,
                                          strk+1, strk, norb, na, na, nlink, nlink,
                                          link_index, link_index,
                                          dimirrep, totirrep);
                }

/* Note: the fillcnt diffs in ctr_rhf2e_kern and spread_a_t1.
 * ctr_rhf2e_kern needs strk+1 beta-strings, spread_a_t1 takes strk
 * beta-strings */
                for (strk = strk0; strk < strk1; strk++) {
                        off = ((long)(strk-strk0))*(strk+strk0+1)/2;
                        pbuf = buf + off*nnorb;
                        spread_a_t1(ci1, pbuf, strk, strk, norb, na, nlink,
                                    link_index);
                }
        }
        free(buf);
}

