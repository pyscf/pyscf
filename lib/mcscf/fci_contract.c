/*
 *
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "vhf/fblas.h"
#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))


void FCImake_hdiag(double *hdiag, double *h1e, double *jdiag, double *kdiag,
                   int norb, int na, int nocc, int *occslist)
{
        int ia, ib, j, j0, k0, jk, jk0;
        double e1, e2;
        int *paocc, *pbocc;
        for (ia = 0; ia < na; ia++) {
                paocc = occslist + ia * nocc;
                for (ib = 0; ib < na; ib++) {
                        e1 = 0;
                        e2 = 0;
                        pbocc = occslist + ib * nocc;
                        for (j0 = 0; j0 < nocc; j0++) {
                                j = paocc[j0];
                                jk0 = j * norb;
                                e1 += h1e[j*norb+j];
                                for (k0 = 0; k0 < nocc; k0++) { // (alpha|alpha)
                                        jk = jk0 + paocc[k0];
                                        e2 += jdiag[jk] - kdiag[jk];
                                }
                                for (k0 = 0; k0 < nocc; k0++) { // (alpha|beta)
                                        jk = jk0 + pbocc[k0];
                                        e2 += jdiag[jk] *2;
                                }
                        }
                        for (j0 = 0; j0 < nocc; j0++) {
                                j = pbocc[j0];
                                jk0 = j * norb;
                                e1 += h1e[j*norb+j];
                                for (k0 = 0; k0 < nocc; k0++) { // (beta|beta)
                                        jk = jk0 + pbocc[k0];
                                        e2 += jdiag[jk] - kdiag[jk];
                                }
                        }
                        hdiag[ia*na+ib] = e1 + e2 * .5;
                }
        }
}

/* strcnt control the number of beta strings to be calculated.
 * for spin=0 system, only lower triangle of the intermediate ci vector
 * needs to be calculated */
static void contract_2e_o3iter(double *eri, double *ci0, double *ci1,
                               double *t2, int ldt2, int strcnt, int strk,
                               int norb, int na, int nlink, int *link_index)
{
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * (norb+1)/2;
        int j, k, ia, str0, str1, sign;
        const int *tab;
        double *pt1, *pci;
        double *t1 = malloc(sizeof(double) * nnorb*strcnt);
        double csum = 0;

        memset(t1, 0, sizeof(double)*nnorb*strcnt);
        pci = ci0 + strk*na;
        for (str0 = 0; str0 < strcnt; str0++) {
                tab = link_index + str0 * nlink * 4;
                pt1 = t1 + str0*nnorb;
                for (j = 0; j < nlink; j++) {
                        ia   = tab[j*4+0];
                        str1 = tab[j*4+2];
                        sign = tab[j*4+3];
                        pt1[ia] += sign * pci[str1];
                        csum += fabs(pci[str1]);
                }
        }
        tab = link_index + strk * nlink * 4;
        for (j = 0; j < nlink; j++) {
                ia   = tab[j*4+0];
                str1 = tab[j*4+2];
                sign = tab[j*4+3];
                pci = ci0 + str1*na;
                pt1 = t1 + ia;
                if (sign > 0) {
                        for (k = 0; k < strcnt-1; k+=2) {
                                pt1[k*nnorb] += pci[k];
                                pt1[k*nnorb+nnorb] += pci[k+1];
                                csum += fabs(pci[k]) + fabs(pci[k+1]);
                        }
                } else {
                        for (k = 0; k < strcnt-1; k+=2) {
                                pt1[k*nnorb] -= pci[k];
                                pt1[k*nnorb+nnorb] -= pci[k+1];
                                csum += fabs(pci[k]) + fabs(pci[k+1]);
                        }
                }
                if (k < strcnt) {
                        pt1[k*nnorb] += sign * pci[k];
                        csum += fabs(pci[k]);
                }
        }
        if (csum < 1e-14) {
                memset(t2, 0, sizeof(double)*nnorb*strcnt);
                goto end;
        }

        dgemm_(&TRANS_T, &TRANS_N, &strcnt, &nnorb, &nnorb,
               &D1, t1, &nnorb, eri, &nnorb, &D0, t2, &ldt2);

        pci = ci1 + strk*na;
        for (str0 = 0; str0 < strcnt; str0++) {
                tab = link_index + str0 * nlink * 4;
                for (j = 0; j < nlink; j++) {
                        ia   = tab[j*4+0];
                        str1 = tab[j*4+2];
                        sign = tab[j*4+3];
                        pci[str1] += sign * t2[ia*ldt2+str0];
                }
        }
end:
        free(t1);
}

static void contract_critical(double *eri, double *ci0, double *ci1,
                              double *t2, int ldt2, int strcnt, int strk,
                              int norb, int na, int nlink, int *link_index)
{
        int j, k, ia, str1, sign;
        int *tab = link_index + strk * nlink * 4;
        double *cp0, *cp1;
        for (j = 0; j < nlink; j++) {
                ia   = tab[j*4+0];
                str1 = tab[j*4+2];
                sign = tab[j*4+3];
                cp0 = t2 + ia*ldt2;
                cp1 = ci1 + str1*na;
                if (sign > 0) {
                        for (k = 0; k < strcnt-1; k+=2) {
                                cp1[k] += cp0[k];
                                cp1[k+1] += cp0[k+1];
                        }
                } else {
                        for (k = 0; k < strcnt-1; k+=2) {
                                cp1[k] -= cp0[k];
                                cp1[k+1] -= cp0[k+1];
                        }
                }
                if (k < strcnt) {
                        cp1[k] += sign * cp0[k];
                }
        }
}

/*
 * nlink = nocc*nvir, num. all possible strings that a string can link to
 * link_index[str0] == linking map between str0 and other strings
 * link_index[str0][ith-linking-string] ==
 *     [creation_op,annihilation_op,linking-string-id,sign]
 * buf_size in MB
 */
void FCIcontract_2e_o3(double *eri, double *ci0, double *ci1,
                       int norb, int na, int nlink, int *link_index,
                       int buf_size)
{
        const int nnorb = norb * (norb+1)/2;
        int len_blk;
#pragma omp parallel shared(len_blk)
        len_blk = omp_get_num_threads();

        int max_buflen = MAX((((long)buf_size)<<20)/8/nnorb/na, len_blk);
        len_blk = (int)(max_buflen/len_blk) * len_blk;
        len_blk = MIN(len_blk, na);
        double *buf = (double *)malloc(sizeof(double) * len_blk*nnorb*na);

        int ic, strk0, strk;
        double *pbuf;

        memset(ci1, 0, sizeof(double)*na*na);
        for (strk0 = 0; strk0 < na; strk0 += len_blk) {
#pragma omp parallel default(none) \
        shared(eri, ci0, ci1, norb, na, nlink, link_index, \
               buf, strk0, len_blk), \
        private(strk, ic)
#pragma omp for schedule(static)
                for (ic = 0; ic < MIN(na-strk0,len_blk); ic++) {
                        strk = strk0 + ic;
                        contract_2e_o3iter(eri, ci0, ci1, buf+ic*nnorb*na,
                                           na, na, strk,
                                           norb, na, nlink, link_index);
                }

                for (ic = 0; ic < MIN(na-strk0,len_blk); ic++) {
                        strk = strk0 + ic;
                        pbuf = buf + ic * nnorb * na;
                        contract_critical(eri, ci0, ci1, pbuf, na, na, strk,
                                          norb, na, nlink, link_index);
                }
        }
        free(buf);
}

void FCIcontract_1e_spin0(double *f1e_tril, double *ci0, double *ci1,
                          int norb, int na, int nlink, int *link_index);
void FCIcontract_1e_o3(double *f1e_tril, double *ci0, double *ci1,
                       int norb, int na, int nlink, int *link_index)
{
        FCIcontract_1e_spin0(f1e_tril, ci0, ci1, norb, na, nlink, link_index);

        int j, k, ia, str0, str1, sign;
        int *tab;
        double *pci1;
        double tmp;
        int nnorb = norb*(norb+1)/2;
        double *t1 = malloc(sizeof(double) * nnorb*na);

        for (str0 = 0; str0 < na; str0++) {
                memset(t1, 0, sizeof(double)*nnorb*na);
                pci1 = ci1 + str0 * na;
                for (k = 0; k < na; k++) {
                        tab = link_index + k * nlink * 4;
                        tmp = ci0[str0*na+k];
                        for (j = 0; j < nlink; j++) {
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
 * buf_size in MB
 */
void FCIcontract_2e_spin0(double *eri, double *ci0, double *ci1,
                          int norb, int na, int nlink, int *link_index,
                          int buf_size)
{
        const int nnorb = norb * (norb+1)/2;
        int nthreads = 16;
//#pragma omp parallel shared(nthreads)
//        nthreads = omp_get_num_threads();

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
                        contract_2e_o3iter(eri, ci0, ci1, pbuf,
                                           strk+1, strk+1, strk,
                                           norb, na, nlink, link_index);
                }

                for (strk = strk0; strk < strk1; strk++) {
                        off = ((long)(strk-strk0))*(strk+strk0+1)/2;
                        pbuf = buf + off*nnorb;
                        contract_critical(eri, ci0, ci1, pbuf,
                                          strk+1, strk, strk,
                                          norb, na, nlink, link_index);
                }
        }
        free(buf);
}

void FCIcontract_1e_spin0(double *f1e_tril, double *ci0, double *ci1,
                          int norb, int na, int nlink, int *link_index)
{
        int j, k, ia, str0, str1, sign;
        int *tab;
        double *pci0, *pci1;
        double tmp;

        memset(ci1, 0, sizeof(double)*na*na);

        for (str0 = 0; str0 < na; str0++) {
                tab = link_index + str0 * nlink * 4;
                for (j = 0; j < nlink; j++) {
                        ia   = tab[j*4+0];
                        str1 = tab[j*4+2];
                        sign = tab[j*4+3];
                        pci0 = ci0 + str0 * na;
                        pci1 = ci1 + str1 * na;
                        tmp  = sign * f1e_tril[ia];
                        for (k = 0; k < na; k++) {
                                pci1[k] += tmp * pci0[k];
                        }
                }
        }
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

