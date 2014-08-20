/*
 * File: fci_contract.c
 *
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#if defined SCIPY_MKL_H
typedef long FINT;
#else
typedef int FINT;
#endif

#include "vhf/fblas.h"
#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))


/* strcnt control the number of beta strings to be calculated.
 * for spin=0 system, only lower triangle of the intermediate ci vector
 * needs to be calculated */
static void contract_2e_o3iter(double *eri, double *ci0, double *ci1,
                               double *t2, FINT ldt2, FINT strcnt, int strk,
                               int norb, int na, int nov, int *link_index)
{
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const FINT nnorb = norb * (norb+1)/2;
        int j, k, ia, str0, str1, sign;
        const int *tab;
        double *pt1, *pci;
        double *t1 = malloc(sizeof(double) * nnorb*strcnt);
        double csum = 0;

        memset(t1, 0, sizeof(double)*nnorb*strcnt);
        pci = ci0 + strk*na;
        for (str0 = 0; str0 < strcnt; str0++) {
                tab = link_index + str0 * nov * 4;
                pt1 = t1 + str0*nnorb;
                for (j = 0; j < nov; j++) {
                        ia = tab[j*4+0];
                        str1 = tab[j*4+1];
                        sign = tab[j*4+2];
                        pt1[ia] += sign * pci[str1];
                        csum += fabs(pci[str1]);
                }
        }
        tab = link_index + strk * nov * 4;
        for (j = 0; j < nov; j++) {
                ia = tab[j*4+0];
                str1 = tab[j*4+1];
                sign = tab[j*4+2];
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
                tab = link_index + str0 * nov * 4;
                for (j = 0; j < nov; j++) {
                        ia = tab[j*4+0];
                        str1 = tab[j*4+1];
                        sign = tab[j*4+2];
                        pci[str1] += sign * t2[ia*ldt2+str0];
                }
        }
end:
        free(t1);
}

static void contract_critical(double *eri, double *ci0, double *ci1,
                              double *t2, int ldt2, int strcnt, int strk,
                              int norb, int na, int nov, int *link_index)
{
        int j, k, ia, str1, sign;
        int *tab = link_index + strk * nov * 4;
        double *cp0, *cp1;
        for (j = 0; j < nov; j++) {
                ia = tab[j*4+0];
                str1 = tab[j*4+1];
                sign = tab[j*4+2];
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
 * nov = nocc*nvir, num. all possible strings that a string can link to
 * link_index[str0] == linking map between str0 and other strings
 * link_index[str0][ith-linking-string] ==
 *     [creation_op,annihilation_op,linking-string-id,sign]
 * buf_size in MB
 */
void FCIcontract_2e_o3(double *eri, double *ci0, double *ci1,
                       int norb, int na, int nov, int *link_index,
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

        for (strk0 = 0; strk0 < na; strk0 += len_blk) {
#pragma omp parallel default(none) \
        shared(eri, ci0, ci1, norb, na, nov, link_index, \
               buf, strk0, len_blk), \
        private(strk, ic)
#pragma omp for schedule(static)
                for (ic = 0; ic < MIN(na-strk0,len_blk); ic++) {
                        strk = strk0 + ic;
                        contract_2e_o3iter(eri, ci0, ci1, buf+ic*nnorb*na,
                                           na, na, strk,
                                           norb, na, nov, link_index);
                }

                for (ic = 0; ic < MIN(na-strk0,len_blk); ic++) {
                        strk = strk0 + ic;
                        pbuf = buf + ic * nnorb * na;
                        contract_critical(eri, ci0, ci1, pbuf, na, na, strk,
                                          norb, na, nov, link_index);
                }
        }
        free(buf);
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
                          int norb, int na, int nov, int *link_index,
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

        for (strk0 = 0, strk1 = na; strk0 < na; strk0 = strk1) {
                strk1 = _square_pace(strk0, blk_base, nthreads);
                strk1 = MIN(strk1, na);
#pragma omp parallel default(none) \
        shared(eri, ci0, ci1, norb, na, nov, link_index, \
               nthreads, strk0, strk1, blk_base, buf), \
        private(strk, off, pbuf)
#pragma omp for schedule(guided, 1)
                for (strk = strk0; strk < strk1; strk++) {
                        //pbuf = buf; ; pbuf += nnorb*(strk+1);
                        off = ((long)(strk-strk0))*(strk+strk0+1)/2;
                        pbuf = buf + off*nnorb;
                        contract_2e_o3iter(eri, ci0, ci1, pbuf,
                                           strk+1, strk+1, strk,
                                           norb, na, nov, link_index);
                }

                for (strk = strk0; strk < strk1; strk++) {
                        off = ((long)(strk-strk0))*(strk+strk0+1)/2;
                        pbuf = buf + off*nnorb;
                        contract_critical(eri, ci0, ci1, pbuf,
                                          strk+1, strk, strk,
                                          norb, na, nov, link_index);
                }
        }
        free(buf);
}

/*
 * Hamming weight popcount
 */

int FCIpopcount_1(unsigned long x) {
        const unsigned long m1  = 0x5555555555555555; //binary: 0101...
        const unsigned long m2  = 0x3333333333333333; //binary: 00110011..
        const unsigned long m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
        const unsigned long m8  = 0x00ff00ff00ff00ff; //binary:  8 zeros,  8 ones ...
        const unsigned long m16 = 0x0000ffff0000ffff; //binary: 16 zeros, 16 ones ...
        const unsigned long m32 = 0x00000000ffffffff; //binary: 32 zeros, 32 ones
//        const unsigned long hff = 0xffffffffffffffff; //binary: all ones
//        const unsigned long h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...
        x = (x & m1 ) + ((x >>  1) & m1 ); //put count of each  2 bits into those  2 bits 
        x = (x & m2 ) + ((x >>  2) & m2 ); //put count of each  4 bits into those  4 bits 
        x = (x & m4 ) + ((x >>  4) & m4 ); //put count of each  8 bits into those  8 bits 
        x = (x & m8 ) + ((x >>  8) & m8 ); //put count of each 16 bits into those 16 bits 
        x = (x & m16) + ((x >> 16) & m16); //put count of each 32 bits into those 32 bits 
        x = (x & m32) + ((x >> 32) & m32); //put count of each 64 bits into those 64 bits 
        return x;
}

int FCIpopcount_4(unsigned long x) {
        int count;
        for (count = 0; x; count++) {
                x &= x-1;
        }
        return count;
}

