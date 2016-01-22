/*
 * Paticle permutation symmetry for 2e Hamiltonian only
 * h2e[i,j,k,l] == h2e[k,l,i,j]
 * h2e[i,j,k,l] =/= h2e[j,i,k,l] =/= h2e[i,j,l,k] ...
 */

#include <stdlib.h>
#include <string.h>
//#include <omp.h>
#include "config.h"
#include "vhf/fblas.h"
#include "fci_string.h"
#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define CSUMTHR         1e-28
#define BUFBASE         96
#define BLKLEN          112

double FCI_t1ci_sf(double *ci0, double *t1, int bcount,
                   int stra_id, int strb_id,
                   int norb, int na, int nb, int nlinka, int nlinkb,
                   _LinkT *clink_indexa, _LinkT *clink_indexb);

static void spread_a_t1(double *ci1, double *t1,
                        int bcount, int stra_id, int strb_id,
                        int norb, int nstrb, int nlinka, _LinkT *clink_indexa)
{
        ci1 += strb_id;
        const int nnorb = norb * norb;
        int j, k, i, a, str1, sign;
        const _LinkT *tab = clink_indexa + stra_id * nlinka;
        double *cp0, *cp1;

        for (j = 0; j < nlinka; j++) {
                a    = EXTRACT_CRE (tab[j]);
                i    = EXTRACT_DES (tab[j]);
                str1 = EXTRACT_ADDR(tab[j]);
                sign = EXTRACT_SIGN(tab[j]);
                cp0 = t1 + a*norb+i; // propagate from t1 to bra, through a^+ i
                cp1 = ci1 + str1*(size_t)nstrb;
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
        const int nnorb = norb * norb;
        int j, i, a, str0, str1, sign;
        const _LinkT *tab = clink_indexb + strb_id * nlinkb;
        double *pci = ci1 + stra_id * (size_t)nstrb;

        for (str0 = 0; str0 < bcount; str0++) {
                for (j = 0; j < nlinkb; j++) {
                        a    = EXTRACT_CRE (tab[j]);
                        i    = EXTRACT_DES (tab[j]);
                        str1 = EXTRACT_ADDR(tab[j]);
                        sign = EXTRACT_SIGN(tab[j]);
                        // propagate from t1 to bra, through a^+ i
                        pci[str1] += sign * t1[a*norb+i];
                }
                t1 += nnorb;
                tab += nlinkb;
        }
}

static void ctr_rhf2e_kern(double *eri, double *ci0, double *ci1, double *tbuf,
                           int bcount, int stra_id, int strb_id,
                           int norb, int na, int nb, int nlinka, int nlinkb,
                           _LinkT *clink_indexa, _LinkT *clink_indexb)
{
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        const int nnorb = norb * norb;
        double *t1 = malloc(sizeof(double) * nnorb*bcount);
        double csum;

        csum = FCI_t1ci_sf(ci0, t1, bcount, stra_id, strb_id,
                           norb, na, nb, nlinka, nlinkb,
                           clink_indexa, clink_indexb);

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

void FCIcontract_2es1(double *eri, double *ci0, double *ci1,
                      int norb, int na, int nb, int nlinka, int nlinkb,
                      int *link_indexa, int *link_indexb)
{
        const int nnorb = norb * norb;

        int ic, strk1, strk0, strk, ib, blen;
        int bufbas = MIN(BUFBASE, nb);
        double *buf = (double *)malloc(sizeof(double) * bufbas*nnorb*BLKLEN);
        double *pbuf;
        _LinkT *clinka = malloc(sizeof(_LinkT) * nlinka * na);
        _LinkT *clinkb = malloc(sizeof(_LinkT) * nlinkb * nb);
        FCIcompress_link(clinka, link_indexa, norb, na, nlinka);
        FCIcompress_link(clinkb, link_indexb, norb, nb, nlinkb);

        memset(ci1, 0, sizeof(double)*na*nb);
        for (strk0 = 0; strk0 < na; strk0 += bufbas) {
                strk1 = MIN(na-strk0, bufbas);
                for (ib = 0; ib < nb; ib += BLKLEN) {
                        blen = MIN(BLKLEN, nb-ib);
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

