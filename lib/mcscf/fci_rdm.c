/*
 * File: fci_rdm.c
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


static double rdm2_o3iter(double *t1, double *ci0, int strk,
                          int norb, int na, int nov, int *link_index)
{
        const int nnorb = norb * norb;
        int i, j, k, a, str0, str1, sign;
        const int *tab;
        double *pt1, *pci;
        double csum = 0;

        memset(t1, 0, sizeof(double)*nnorb*na);
        pci = ci0 + strk*na;
        for (str0 = 0; str0 < na; str0++) {
                tab = link_index + str0 * nov * 4;
                pt1 = t1 + str0*nnorb;
                for (j = 0; j < nov; j++) {
                        a = tab[j*4+0];
                        i = tab[j*4+1];
                        str1 = tab[j*4+2];
                        sign = tab[j*4+3];
                        pt1[i*norb+a] += sign * pci[str1];
                        csum += fabs(pci[str1]);
                }
        }
        tab = link_index + strk * nov * 4;
        for (j = 0; j < nov; j++) {
                a = tab[j*4+0];
                i = tab[j*4+1];
                str1 = tab[j*4+2];
                sign = tab[j*4+3];
                pci = ci0 + str1*na;
                pt1 = t1 + i*norb+a;
                if (sign > 0) {
                        for (k = 0; k < na-1; k+=2) {
                                pt1[k*nnorb] += pci[k];
                                pt1[k*nnorb+nnorb] += pci[k+1];
                                csum += fabs(pci[k]) + fabs(pci[k+1]);
                        }
                } else {
                        for (k = 0; k < na-1; k+=2) {
                                pt1[k*nnorb] -= pci[k];
                                pt1[k*nnorb+nnorb] -= pci[k+1];
                                csum += fabs(pci[k]) + fabs(pci[k+1]);
                        }
                }
                if (k < na) {
                        pt1[k*nnorb] -= pci[k];
                        csum += fabs(pci[k]);
                }
        }
        return csum;
}

/*
 * incorrect order of rdm2[1,0,2,3]
 */
void FCImake_rdm12_o3(double *rdm1, double *rdm2, double *ci0,
                      int norb, int na, int nov, int *link_index)
{
        const FINT INC1 = 1;
        const char UP = 'U';
        const char TRANS_N = 'N';
        const double D1 = 1;
        const FINT nnorb = norb * norb;
        int strk, i, j;
        double csum;
        double *pdm1, *pdm2, *buf;
        memset(rdm1, 0, sizeof(double) * nnorb);
        memset(rdm2, 0, sizeof(double) * nnorb*nnorb);

#pragma omp parallel default(none) \
        shared(ci0, norb, na, nov, link_index, rdm1, rdm2), \
        private(strk, i, csum, pdm1, pdm2, buf)
{
        buf = (double *)malloc(sizeof(double) * nnorb*na);
        pdm1 = (double *)malloc(sizeof(double) * nnorb);
        pdm2 = (double *)malloc(sizeof(double) * nnorb*nnorb);
        memset(pdm1, 0, sizeof(double) * nnorb);
        memset(pdm2, 0, sizeof(double) * nnorb*nnorb);
#pragma omp for schedule(guided, 2)
        for (strk = 0; strk < na; strk++) {
                csum = rdm2_o3iter(buf, ci0, strk, norb, na, nov, link_index);
                if (csum > 1e-14) {
                        dgemv_(&TRANS_N, &nnorb, &na, &D1, buf, &nnorb,
                               ci0+strk*na, &INC1, &D1, pdm1, &INC1);
                        dsyrk_(&UP, &TRANS_N, &nnorb, &na,
                               &D1, buf, &nnorb, &D1, pdm2, &nnorb);
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
        free(buf);
}
        for (i = 0; i < nnorb; i++) {
                for (j = 0; j < i; j++) {
                        rdm2[j*nnorb+i] = rdm2[i*nnorb+j];
                }
        }
}


/*
 * ci0 is symmetric on alpha and beta electrons
 */
static double rdm2_spin0_o3iter(double *t1, double *ci0, int strk,
                                int norb, int na, int nov, int *link_index)
{
        const int nnorb = norb * norb;
        int i, j, k, a, str0, str1, sign;
        const int *tab;
        double *pt1, *pci;
        double csum = 0;

        memset(t1, 0, sizeof(double)*nnorb*(strk+1));
        pci = ci0 + strk*na;
        for (str0 = 0; str0 < strk; str0++) {
                tab = link_index + str0 * nov * 4;
                pt1 = t1 + str0*nnorb;
                for (j = 0; j < nov; j++) {
                        a = tab[j*4+0];
                        i = tab[j*4+1];
                        str1 = tab[j*4+2];
                        sign = tab[j*4+3];
                        pt1[i*norb+a] += sign * pci[str1];
                        csum += fabs(pci[str1]);
                }
        }
        tab = link_index + strk * nov * 4;
        for (j = 0; j < nov; j++) {
                a = tab[j*4+0];
                i = tab[j*4+1];
                str1 = tab[j*4+2];
                sign = tab[j*4+3];
                pci = ci0 + str1*na;
                pt1 = t1 + i*norb+a;
                if (sign > 0) {
                        for (k = 0; k < strk; k+=2) {
                                pt1[k*nnorb] += pci[k];
                                pt1[k*nnorb+nnorb] += pci[k+1];
                                csum += fabs(pci[k]) + fabs(pci[k+1]);
                        }
                } else {
                        for (k = 0; k < strk; k+=2) {
                                pt1[k*nnorb] -= pci[k];
                                pt1[k*nnorb+nnorb] -= pci[k+1];
                                csum += fabs(pci[k]) + fabs(pci[k+1]);
                        }
                }
                if (k < strk+1) {
                        pt1[k*nnorb] += sign * pci[k];
                        csum += fabs(pci[k]);
                }
        }
        return csum;
}


void FCImake_rdm12_spin0_o3(double *rdm1, double *rdm2, double *ci0,
                            int norb, int na, int nov, int *link_index)
{
        const FINT INC1 = 1;
        const char UP = 'U';
        const char TRANS_N = 'N';
        const double D1 = 1;
        const double D2 = 2;
        const FINT nnorb = norb * norb;
        int strk, strk1, i, j;
        double csum;
        double *pdm1, *pdm2, *buf;
        memset(rdm1, 0, sizeof(double) * nnorb);
        memset(rdm2, 0, sizeof(double) * nnorb*nnorb);

#pragma omp parallel default(none) \
        shared(ci0, norb, na, nov, link_index, rdm1, rdm2), \
        private(strk, strk1, i, csum, pdm1, pdm2, buf)
{
        buf = (double *)malloc(sizeof(double) * nnorb*na);
        pdm1 = (double *)malloc(sizeof(double) * nnorb);
        pdm2 = (double *)malloc(sizeof(double) * nnorb*nnorb);
        memset(pdm1, 0, sizeof(double) * nnorb);
        memset(pdm2, 0, sizeof(double) * nnorb*nnorb);
#pragma omp for schedule(guided, 2)
        for (strk = 0; strk < na; strk++) {
                csum = rdm2_spin0_o3iter(buf, ci0, strk, norb, na, nov, link_index);
                if (csum > 1e-14) {
                        strk1 = strk + 1;
                        dgemv_(&TRANS_N, &nnorb, &strk1, &D1, buf, &nnorb,
                               ci0+strk*na, &INC1, &D1, pdm1, &INC1);
                        dsyrk_(&UP, &TRANS_N, &nnorb, &strk,
                               &D1, buf, &nnorb, &D1, pdm2, &nnorb);
                        dsyr_(&UP, &nnorb, &D2, buf+nnorb*strk, &INC1,
                              pdm2, &nnorb);
                }
        }
#pragma omp critical
{
        for (i = 0; i < nnorb; i++) {
                rdm1[i] += pdm1[i] * 2;
        }
        for (i = 0; i < nnorb*nnorb; i++) {
                rdm2[i] += pdm2[i] * 2;
        }
}
        free(pdm1);
        free(pdm2);
        free(buf);
}
        for (i = 0; i < nnorb; i++) {
                for (j = 0; j < i; j++) {
                        rdm2[j*nnorb+i] = rdm2[i*nnorb+j];
                }
        }
}


/*
 * ***********************************************
 */
void FCItrans_rdm12_o3(double *rdm1, double *rdm2,
                       double *bra, double *ket,
                       int norb, int na, int nov, int *link_index)
{
        const FINT INC1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D1 = 1;
        const FINT nnorb = norb * norb;
        int strk, i;
        double csum;
        double *pdm1, *pdm2, *buf1, *buf0;
        memset(rdm1, 0, sizeof(double) * nnorb);
        memset(rdm2, 0, sizeof(double) * nnorb*nnorb);

#pragma omp parallel default(none) \
        shared(bra, ket, norb, na, nov, link_index, rdm1, rdm2), \
        private(strk, i, csum, pdm1, pdm2, buf1, buf0)
{
        buf0 = (double *)malloc(sizeof(double) * nnorb*na);
        buf1 = (double *)malloc(sizeof(double) * nnorb*na);
        pdm1 = (double *)malloc(sizeof(double) * nnorb);
        pdm2 = (double *)malloc(sizeof(double) * nnorb*nnorb);
        memset(pdm1, 0, sizeof(double) * nnorb);
        memset(pdm2, 0, sizeof(double) * nnorb*nnorb);
#pragma omp for schedule(guided, 2)
        for (strk = 0; strk < na; strk++) {
                csum = rdm2_o3iter(buf1, bra, strk, norb, na, nov, link_index);
                if (csum < 1e-14) { continue; }
                csum = rdm2_o3iter(buf0, ket, strk, norb, na, nov, link_index);
                if (csum < 1e-14) { continue; }
                dgemv_(&TRANS_N, &nnorb, &na, &D1, buf0, &nnorb,
                       bra+strk*na, &INC1, &D1, pdm1, &INC1);
                dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &na,
                       &D1, buf0, &nnorb, buf1, &nnorb,
                       &D1, pdm2, &nnorb);
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
        free(buf1);
        free(buf0);
}
}

void FCItrans_rdm12_spin0_o3(double *rdm1, double *rdm2,
                             double *bra, double *ket,
                             int norb, int na, int nov, int *link_index)
{
        const FINT INC1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D1 = 1;
        const double D2 = 2;
        const FINT nnorb = norb * norb;
        int strk, strk1, i;
        double csum;
        double *pdm1, *pdm2, *buf1, *buf0;
        memset(rdm1, 0, sizeof(double) * nnorb);
        memset(rdm2, 0, sizeof(double) * nnorb*nnorb);

#pragma omp parallel default(none) \
        shared(bra, ket, norb, na, nov, link_index, rdm1, rdm2), \
        private(strk, strk1, i, csum, pdm1, pdm2, buf1, buf0)
{
        buf0 = (double *)malloc(sizeof(double) * nnorb*na);
        buf1 = (double *)malloc(sizeof(double) * nnorb*na);
        pdm1 = (double *)malloc(sizeof(double) * nnorb);
        pdm2 = (double *)malloc(sizeof(double) * nnorb*nnorb);
        memset(pdm1, 0, sizeof(double) * nnorb);
        memset(pdm2, 0, sizeof(double) * nnorb*nnorb);
#pragma omp for schedule(guided, 2)
        for (strk = 0; strk < na; strk++) {
                csum = rdm2_spin0_o3iter(buf1, bra, strk, norb, na, nov, link_index);
                if (csum < 1e-14) { continue; }
                csum = rdm2_spin0_o3iter(buf0, ket, strk, norb, na, nov, link_index);
                if (csum < 1e-14) { continue; }
                strk1 = strk + 1;
                dgemv_(&TRANS_N, &nnorb, &strk1, &D1, buf0, &nnorb,
                       bra+strk*na, &INC1, &D1, pdm1, &INC1);
                dgemm_(&TRANS_N, &TRANS_T, &nnorb, &nnorb, &strk,
                       &D1, buf0, &nnorb, buf1, &nnorb,
                       &D1, pdm2, &nnorb);
                dger_(&nnorb, &nnorb, &D2, buf0+nnorb*strk, &INC1,
                      buf1+nnorb*strk, &INC1, pdm2, &nnorb);
        }
#pragma omp critical
{
        for (i = 0; i < nnorb; i++) {
                rdm1[i] += pdm1[i] * 2;
        }
        for (i = 0; i < nnorb*nnorb; i++) {
                rdm2[i] += pdm2[i] * 2;
        }
}
        free(pdm1);
        free(pdm2);
        free(buf1);
        free(buf0);
}
}
