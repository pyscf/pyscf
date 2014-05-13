/*
 *
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "misc.h"
#include "fblas.h"

void CVHFcompress_nr_dm(double *tri_dm, double *dm, int nao)
{
        int i, j, ij;
        for (i = 0, ij = 0; i < nao; i++) {
                for (j = 0; j < i; j++, ij++) {
                        tri_dm[ij] = dm[i*nao+j] + dm[j*nao+i];
                }
                tri_dm[ij] = dm[i*nao+i];
                ij++;
        }
}

void CVHFnr_k(int n, double *eri, double *dm, double *vk)
{
        const int INC1 = 1;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const int ln = n;
        double *tmp = malloc(sizeof(double)*n*n);
        int nao_pair = n * (n+1) / 2;
        int i, j;
        int l1, l2;
        for (i = 0; i < n; i++) {
                for (j = 0; j < i; j++, eri += nao_pair) {
                        CVHFunpack(n, eri, tmp);
                        l1 = j + 1;
                        l2 = i + 1;
                        dgemv_(&TRANS_T, &ln, &l1, &D1, tmp, &ln, dm+i*n, &INC1,
                               &D1, vk+j*n, &INC1);
                        dgemv_(&TRANS_T, &ln, &l2, &D1, tmp, &ln, dm+j*n, &INC1,
                               &D1, vk+i*n, &INC1);
                }
                CVHFunpack(n, eri, tmp);
                l1 = i + 1;
                dgemv_(&TRANS_T, &ln, &l1, &D1, tmp, &ln, dm+i*n, &INC1,
                       &D1, vk+i*n, &INC1);
                eri += nao_pair;
        }
        free(tmp);
}

/* eri uses 8-fold symmetry: i>=j,k>=ln,ij>=kl */
void CVHFnr_incore_o3(int n, double *eri, double *dm, double *vj, double *vk)
{
        const int INC1 = 1;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const int ln = n;
        double *row = malloc(sizeof(double)*n*n);
        double *tri_dm = malloc(sizeof(double)*n*n);
        double *tmp = malloc(sizeof(double)*n*n);
        const int nao_pair = n * (n+1) / 2;
        int i, j, ij;
        int l1, l2;

        ij = 0;
        for (i = 0; i < n; i++) {
                for (j = 0; j < i; j++, ij++) {
                        tri_dm[ij] = dm[i*n+j] + dm[j*n+i];
                }
                tri_dm[ij] = dm[i*n+i];
                ij++;
        }

        ij = 0;
        for (i = 0; i < n; i++) {
                for (j = 0; j < i; j++, ij++) {
                        extract_row_from_tri(row, ij, nao_pair, eri);
                        CVHFunpack(n, row, tmp);
                        l1 = j + 1;
                        l2 = i + 1;
                        vj[i*n+j] = ddot_(&nao_pair, row, &INC1, tri_dm, &INC1);

                        dgemv_(&TRANS_T, &ln, &l1, &D1, tmp, &ln, dm+i*n, &INC1,
                               &D1, vk+j*n, &INC1);
                        dgemv_(&TRANS_T, &ln, &l2, &D1, tmp, &ln, dm+j*n, &INC1,
                               &D1, vk+i*n, &INC1);
                }
                extract_row_from_tri(row, ij, nao_pair, eri);
                CVHFunpack(n, row, tmp);
                vj[i*n+i] = ddot_(&nao_pair, row, &INC1, tri_dm, &INC1);

                l1 = i + 1;
                dgemv_(&TRANS_T, &ln, &l1, &D1, tmp, &ln, dm+i*n, &INC1,
                       &D1, vk+i*n, &INC1);
                ij++;
        }
        free(tmp);
        free(tri_dm);
        free(row);
}

/*************************************************
 * eri uses 8-fold symmetry: i>=j,k>=ln,ij>=kl
 * eri is the address of the first element for pair ij
 * i.e. ~ &eri_ao[ij*(ij+1)/2] */
void CVHFnr_eri8fold_vj_o2(double *tri_vj, const int ij,
                           const double *eri, const double *tri_dm)
{
        int i;
        for (i = 0; i < ij; i++) {
                tri_vj[ij] += eri[i] * tri_dm[i];
                tri_vj[i] += eri[i] * tri_dm[ij];
        }
        tri_vj[ij] += eri[ij] * tri_dm[ij];
}
void CVHFnr_eri8fold_vj_o3(double *tri_vj, const int ij,
                           const double *eri, const double *tri_dm)
{
        const int INC1 = 1;
        int nij = ij;
        tri_vj[ij] += ddot_(&nij, eri, &INC1, tri_dm, &INC1);
        nij++;
        daxpy_(&nij, tri_dm+ij, eri, &INC1, tri_vj, &INC1);
}

/*
 * dm can be non-Hermitian
 */
void CVHFnr_eri8fold_vk_o0(double *vk, int i, int j, int n,
                           const double *eri, const double *dm)
{
        int k, l, kl;
        if (i > j) {
                for (k = 0, kl = 0; k < i; k++) {
                        for (l = 0; l < k; l++, kl++) {
                                vk[j*n+l] += eri[kl] * dm[i*n+k];
                                vk[i*n+l] += eri[kl] * dm[j*n+k];
                                vk[j*n+k] += eri[kl] * dm[i*n+l];
                                vk[i*n+k] += eri[kl] * dm[j*n+l];
                                vk[l*n+j] += eri[kl] * dm[k*n+i];
                                vk[k*n+j] += eri[kl] * dm[l*n+i];
                                vk[l*n+i] += eri[kl] * dm[k*n+j];
                                vk[k*n+i] += eri[kl] * dm[l*n+j];
                        }
                        vk[j*n+k] += eri[kl] * dm[i*n+k];
                        vk[i*n+k] += eri[kl] * dm[j*n+k];
                        vk[k*n+j] += eri[kl] * dm[k*n+i];
                        vk[k*n+i] += eri[kl] * dm[k*n+j];
                        kl++;
                }
                k = i;
                for (l = 0, kl = k*(k+1)/2; l < j; l++, kl++) { // l<k
                        vk[j*n+l] += eri[kl] * dm[i*n+i];
                        vk[i*n+l] += eri[kl] * dm[j*n+i];
                        vk[j*n+i] += eri[kl] * dm[i*n+l];
                        vk[i*n+i] += eri[kl] * dm[j*n+l];
                        vk[l*n+j] += eri[kl] * dm[i*n+i];
                        vk[i*n+j] += eri[kl] * dm[l*n+i];
                        vk[l*n+i] += eri[kl] * dm[i*n+j];
                        vk[i*n+i] += eri[kl] * dm[l*n+j];
                }
                // i = k, j = l;
                vk[j*n+j] += eri[kl] * dm[i*n+i];
                vk[i*n+j] += eri[kl] * dm[j*n+i];
                vk[j*n+i] += eri[kl] * dm[i*n+j];
                vk[i*n+i] += eri[kl] * dm[j*n+j];
        } else { // i = j
                for (k = 0, kl = 0; k < i; k++) {
                        for (l = 0; l < k; l++, kl++) {
                                vk[i*n+l] += eri[kl] * dm[i*n+k];
                                vk[i*n+k] += eri[kl] * dm[i*n+l];
                                vk[l*n+i] += eri[kl] * dm[k*n+i];
                                vk[k*n+i] += eri[kl] * dm[l*n+i];
                        }
                        vk[i*n+k] += eri[kl] * dm[i*n+k];
                        vk[k*n+i] += eri[kl] * dm[k*n+i];
                        kl++;
                }
                k = i;
                for (l = 0, kl = k*(k+1)/2; l < k; l++, kl++) { // l<k
                        vk[i*n+l] += eri[kl] * dm[i*n+i];
                        vk[i*n+i] += eri[kl] * dm[i*n+l];
                        vk[l*n+i] += eri[kl] * dm[i*n+i];
                        vk[i*n+i] += eri[kl] * dm[l*n+i];
                }
                // i = j = k = l
                vk[i*n+i] += eri[kl] * dm[i*n+i];
        }
}
void CVHFnr_eri8fold_vk_o1(double *vk, int i, int j, int n,
                           const double *eri, const double *dm)
{
        int k, l;
        if (i > j) {
                for (k = 0; k < i; k++) {
                        for (l = 0; l < k; l++, eri++) {
                                vk[j*n+l] += *eri * dm[i*n+k];
                                vk[i*n+l] += *eri * dm[j*n+k];
                                vk[j*n+k] += *eri * dm[i*n+l];
                                vk[i*n+k] += *eri * dm[j*n+l];
                                vk[l*n+j] += *eri * dm[k*n+i];
                                vk[k*n+j] += *eri * dm[l*n+i];
                        }
                        vk[j*n+k] += *eri * dm[i*n+k];
                        vk[i*n+k] += *eri * dm[j*n+k];
                        vk[k*n+j] += *eri * dm[k*n+i];
                        eri++;
                }
                for (l = 0; l < j; l++, eri++) { // l<k
                        vk[j*n+l] += *eri * dm[i*n+i];
                        vk[i*n+l] += *eri * dm[j*n+i];
                        vk[i*n+i] += *eri * dm[j*n+l];
                        vk[i*n+j] += *eri * dm[l*n+i];
                        vk[i*n+i] += *eri * dm[l*n+j];
                }
                // i = k, j = l;
                vk[j*n+j] += *eri * dm[i*n+i];
                vk[i*n+j] += *eri * dm[j*n+i];
                vk[i*n+i] += *eri * dm[j*n+j];
        } else { // i = j
                for (k = 0; k < i; k++) {
                        for (l = 0; l < k; l++, eri++) {
                                vk[i*n+l] += *eri * dm[i*n+k];
                                vk[i*n+k] += *eri * dm[i*n+l];
                        }
                        vk[i*n+k] += *eri * dm[i*n+k];
                        eri++;
                }
                for (l = 0; l < k; l++, eri++) { // l<k
                        vk[i*n+l] += *eri * dm[i*n+i];
                        vk[i*n+i] += *eri * dm[i*n+l];
                        vk[i*n+i] += *eri * dm[l*n+i];
                }
                // i = j = k = l
                vk[i*n+i] += *eri * dm[i*n+i];
        }
}
void CVHFnr_eri8fold_vk_o2(double *vk, int i, int j, int n,
                           const double *eri, const double *dm)
{
        int k, l;
        if (i > j) {
                // k < j
                for (k = 0; k < j; k++) {
                        for (l = 0; l < k; l++, eri++) {
                                vk[j*n+l] += *eri * dm[i*n+k];
                                vk[i*n+l] += *eri * dm[j*n+k];
                                vk[j*n+k] += *eri * dm[i*n+l];
                                vk[i*n+k] += *eri * dm[j*n+l];
                        }
                        // l = k
                        vk[j*n+k] += *eri * dm[i*n+k];
                        vk[i*n+k] += *eri * dm[j*n+k];
                        eri++;
                }
                // k = j
                for (l = 0; l < k; l++, eri++) {
                        vk[j*n+l] += *eri * dm[i*n+j];
                        vk[j*n+j] += *eri *(dm[i*n+l] + dm[l*n+i]);
                        vk[i*n+l] += *eri * dm[j*n+j];
                        vk[i*n+j] += *eri * dm[j*n+l];
                }
                // l = k = j
                vk[j*n+j] += *eri *(dm[i*n+j] + dm[j*n+i]);
                vk[i*n+j] += *eri * dm[j*n+j];
                eri++;
                // k > j
                for (k = j+1; k < i; k++) {
                        // l < j
                        for (l = 0; l < j; l++, eri++) {
                                vk[j*n+l] += *eri * dm[i*n+k];
                                vk[i*n+l] += *eri * dm[j*n+k];
                                vk[i*n+k] += *eri * dm[j*n+l];
                                vk[k*n+j] += *eri * dm[l*n+i];
                        }
                        // l = j
                        vk[j*n+j] += *eri *(dm[i*n+k] + dm[k*n+i]);
                        vk[i*n+j] += *eri * dm[j*n+k];
                        vk[i*n+k] += *eri * dm[j*n+j];
                        vk[k*n+j] += *eri * dm[j*n+i];
                        eri++;
                        // l > j
                        for (l = j+1; l < k; l++, eri++) {
                                vk[i*n+l] += *eri * dm[j*n+k];
                                vk[i*n+k] += *eri * dm[j*n+l];
                                vk[l*n+j] += *eri * dm[k*n+i];
                                vk[k*n+j] += *eri * dm[l*n+i];
                        }
                        // l = k
                        vk[j*n+k] += *eri * dm[i*n+k];
                        vk[i*n+k] += *eri * dm[j*n+k];
                        vk[k*n+j] += *eri * dm[k*n+i];
                        eri++;
                }
                // k = i
                for (l = 0; l < j; l++, eri++) {
                        vk[j*n+l] += *eri * dm[i*n+i];
                        vk[i*n+l] += *eri * dm[j*n+i];
                        vk[i*n+i] += *eri *(dm[j*n+l] + dm[l*n+j]);
                        vk[i*n+j] += *eri * dm[l*n+i];
                }
                // i = k, j = l;
                vk[j*n+j] += *eri * dm[i*n+i];
                vk[i*n+j] += *eri * dm[j*n+i];
                vk[i*n+i] += *eri * dm[j*n+j];
        } else { // i = j
                for (k = 0; k < i; k++) {
                        for (l = 0; l < k; l++, eri++) {
                                vk[i*n+l] += *eri * dm[i*n+k];
                                vk[i*n+k] += *eri * dm[i*n+l];
                        }
                        vk[i*n+k] += *eri * dm[i*n+k];
                        eri++;
                }
                for (l = 0; l < k; l++, eri++) { // l<k
                        vk[i*n+l] += *eri * dm[i*n+i];
                        vk[i*n+i] += *eri *(dm[i*n+l] + dm[l*n+i]);
                }
                // i = j = k = l
                vk[i*n+i] += *eri * dm[i*n+i];
        }
}
// daxpy and ddot is not faster because the effect of unrolling
void CVHFnr_eri8fold_vk_o3(double *vk, int i, int j, int n,
                           const double *eri, const double *dm)
{
        const int INC1 = 1;
        int k, n1;
        if (i > j) {
                for (k = 0; k < i; k++) {
                        daxpy_(&k, dm+i*n+k, eri, &INC1, vk+j*n, &INC1);
                        daxpy_(&k, dm+j*n+k, eri, &INC1, vk+i*n, &INC1);
                        daxpy_(&k, dm+k*n+i, eri, &INC1, vk+j, &n);
                        n1 = k + 1;
                        vk[j*n+k] += ddot_(&n1, eri, &INC1, dm+i*n, &INC1);
                        vk[i*n+k] += ddot_(&n1, eri, &INC1, dm+j*n, &INC1);
                        vk[k*n+j] += ddot_(&n1, eri, &INC1, dm+i, &n);
                        eri += n1;
                }
                n1 = j + 1;
                daxpy_(&n1, dm+i*n+i, eri, &INC1, vk+j*n, &INC1);
                daxpy_(&j, dm+j*n+i, eri, &INC1, vk+i*n, &INC1);
                vk[i*n+i] += ddot_(&j, eri, &INC1, dm+j*n, &INC1);
                vk[i*n+j] += ddot_(&n1, eri, &INC1, dm+i, &n);
                vk[i*n+i] += ddot_(&n1, eri, &INC1, dm+j, &n);
        } else { // i = j
                for (k = 0; k <= i; k++) {
                        n1 = k + 1;
                        daxpy_(&k, dm+i*n+k, eri, &INC1, vk+i*n, &INC1);
                        vk[i*n+k] += ddot_(&n1, eri, &INC1, dm+i*n, &INC1);
                        eri += n1;
                }
                vk[i*n+i] += ddot_(&i, eri-i-1, &INC1, dm+i, &n);
        }
}
void CVHFnr_eri8fold_vk_o4(double *vk, int i, int j, int n,
                           const double *eri, const double *dm)
{
        int k, l;
        if (i > j) {
                // k < j
                for (k=0; k < j; k++) {
                        for (l = 0; l < k; l++) {
                                vk[j*n+l] += eri[l] * dm[i*n+k];
                                vk[j*n+k] += eri[l] * dm[i*n+l];
                                vk[i*n+l] += eri[l] * dm[j*n+k];
                                vk[i*n+k] += eri[l] * dm[j*n+l];
                        }
                        // l = k
                        vk[j*n+k] += eri[k] * dm[i*n+k];
                        vk[i*n+k] += eri[k] * dm[j*n+k];
                        eri += k + 1;
                }
                // k = j
                for (l = 0; l < k; l++) {
                        vk[j*n+l] += eri[l] * dm[i*n+j];
                        vk[j*n+j] += eri[l] *(dm[i*n+l] + dm[l*n+i]);
                        vk[i*n+l] += eri[l] * dm[j*n+j];
                        vk[i*n+j] += eri[l] * dm[j*n+l];
                }
                eri += k;
                // l = k = j
                vk[j*n+j] += *eri *(dm[i*n+j] + dm[j*n+i]);
                vk[i*n+j] += *eri * dm[j*n+j];
                eri++;
                // k > j
                for (k=j+1; k < i; k++) {
                        // l < j
                        for (l = 0; l < j; l++) {
                                vk[j*n+l] += eri[l] * dm[i*n+k];
                                vk[i*n+l] += eri[l] * dm[j*n+k];
                                vk[i*n+k] += eri[l] * dm[j*n+l];
                                vk[k*n+j] += eri[l] * dm[l*n+i];
                        }
                        // l = j
                        vk[j*n+j] += eri[j] *(dm[i*n+k] + dm[k*n+i]);
                        vk[i*n+j] += eri[j] * dm[j*n+k];
                        vk[i*n+k] += eri[j] * dm[j*n+j];
                        vk[k*n+j] += eri[j] * dm[j*n+i];
                        eri += j+1;
                        // l > j
                        for (l = j+1; l < k; l++, eri++) {
                                vk[i*n+l] += eri[0] * dm[j*n+k];
                                vk[i*n+k] += eri[0] * dm[j*n+l];
                                vk[l*n+j] += eri[0] * dm[k*n+i];
                                vk[k*n+j] += eri[0] * dm[l*n+i];
                        }
                        // l = k
                        vk[j*n+k] += eri[0] * dm[i*n+k];
                        vk[i*n+k] += eri[0] * dm[j*n+k];
                        vk[k*n+j] += eri[0] * dm[k*n+i];
                        eri++;
                }
                // k = i
                for (l = 0; l < j; l++) {
                        vk[j*n+l] += eri[l] * dm[i*n+i];
                        vk[i*n+l] += eri[l] * dm[j*n+i];
                        vk[i*n+i] += eri[l] *(dm[j*n+l] + dm[l*n+j]);
                        vk[i*n+j] += eri[l] * dm[l*n+i];
                }
                eri += j;
                // i = k, j = l;
                vk[j*n+j] += *eri * dm[i*n+i];
                vk[i*n+j] += *eri * dm[j*n+i];
                vk[i*n+i] += *eri * dm[j*n+j];
        } else { // i = j
                for (k = 0; k < i-1; k+=2) {
                        for (l = 0; l < k; l++) {
                                vk[i*n+l] += eri[l] * dm[i*n+k];
                                vk[i*n+k] += eri[l] * dm[i*n+l];
                                vk[i*n+l  ] += eri[l+k+1] * dm[i*n+k+1];
                                vk[i*n+k+1] += eri[l+k+1] * dm[i*n+l  ];
                        }
                        vk[i*n+k] += eri[k] * dm[i*n+k];
                        eri += k+1;
                        vk[i*n+k  ] += eri[k] * dm[i*n+k+1];
                        vk[i*n+k+1] += eri[k] * dm[i*n+k  ];
                        vk[i*n+k+1] += eri[k+1] * dm[i*n+k+1];
                        eri += k+2;
                }
                for (; k < i; k++) {
                        for (l = 0; l < k; l++) {
                                vk[i*n+l] += eri[l] * dm[i*n+k];
                                vk[i*n+k] += eri[l] * dm[i*n+l];
                        }
                        vk[i*n+k] += eri[k] * dm[i*n+k];
                        eri += k+1;
                }
                for (l = 0; l < k; l++) { // l<k
                        vk[i*n+l] += eri[l] * dm[i*n+i];
                        vk[i*n+i] += eri[l] *(dm[i*n+l] + dm[l*n+i]);
                }
                eri += k;
                // i = j = k = l
                vk[i*n+i] += *eri * dm[i*n+i];
        }
}
void CVHFnr_incore_o4(int n, double *eri, double *dm, double *vj, double *vk)
{
        const int npair = n*(n+1)/2;
        double *tri_dm = malloc(sizeof(double)*npair);
        double *tri_vj = malloc(sizeof(double)*npair);
        double *vj_priv, *vk_priv;
        int i, j;
        int *ij2i = malloc(sizeof(int)*npair);
        unsigned long ij, off;

        CVHFcompress_nr_dm(tri_dm, dm, n);
        CVHFset_ij2i(ij2i, n);
        memset(tri_vj, 0, sizeof(double)*npair);
        memset(vk, 0, sizeof(double)*n*n);

#pragma omp parallel default(none) \
        shared(eri, tri_dm, dm, tri_vj, vk, ij2i, n) \
        private(ij, i, j, off, vj_priv, vk_priv)
        {
                vj_priv = malloc(sizeof(double)*npair);
                vk_priv = malloc(sizeof(double)*n*n);
                memset(vj_priv, 0, sizeof(double)*npair);
                memset(vk_priv, 0, sizeof(double)*n*n);
#pragma omp for nowait schedule(guided, 4)
                for (ij = 0; ij < npair; ij++) {
                        i = ij2i[ij];
                        j = ij - (i*(i+1)/2);
                        off = ij*(ij+1)/2;
                        CVHFnr_eri8fold_vj_o2(vj_priv, ij, eri+off, tri_dm);
                        CVHFnr_eri8fold_vk_o4(vk_priv, i, j, n, eri+off, dm);
                }
#pragma omp critical
                {
                        for (i = 0; i < npair; i++) {
                                tri_vj[i] += vj_priv[i];
                        }
                        for (i = 0; i < n*n; i++) {
                                vk[i] += vk_priv[i];
                        }
                }
                free(vj_priv);
                free(vk_priv);
        }

        for (i = 0, ij = 0; i < n; i++) {
                for (j = 0; j <= i; j++, ij++) {
                        vj[i*n+j] = tri_vj[ij];
                        vj[j*n+i] = tri_vj[ij];
                        vk[j*n+i] = vk[i*n+j];
                }
        }
        free(ij2i);
        free(tri_dm);
        free(tri_vj);
}

