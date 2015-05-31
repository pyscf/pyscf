/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "cint.h"
#include "vhf/fblas.h"
#include <assert.h>

#define BOXSIZE         64
#define MIN(X,Y)        ((X)>(Y)?(Y):(X))

static void dot_ao_dm(double *vm, double *ao, double *dm,
                      int nao, int nocc, int ngrids, char *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;

        if (nao <= BOXSIZE) {
                dgemm_(&TRANS_N, &TRANS_N, &nocc, &ngrids, &nao,
                       &D1, dm, &nocc, ao, &nao, &D0, vm, &nocc);
                return;
        }

        char empty[nbas];
        int nbox = (int)((nao-1)/BOXSIZE) + 1;
        int box_id, bas_id, nd, b0, blen;
        int ao_id = 0;
        for (box_id = 0; box_id < nbox; box_id++) {
                empty[box_id] = 1;
        }

        box_id = 0;
        b0 = BOXSIZE;
        for (bas_id = 0; bas_id < nbas; bas_id++) {
                nd = (bas[ANG_OF] * 2 + 1) * bas[NCTR_OF];
                assert(nd < BOXSIZE);
                ao_id += nd;
                empty[box_id] &= !non0table[bas_id];
                if (ao_id == b0) {
                        box_id++;
                        b0 += BOXSIZE;
                } else if (ao_id > b0) {
                        box_id++;
                        b0 += BOXSIZE;
                        empty[box_id] = !non0table[bas_id];
                }
                bas += BAS_SLOTS;
        }

        memset(vm, 0, sizeof(double) * ngrids * nocc);

        for (box_id = 0; box_id < nbox; box_id++) {
                if (!empty[box_id]) {
                        b0 = box_id * BOXSIZE;
                        blen = MIN(nao-b0, BOXSIZE);
                        dgemm_(&TRANS_N, &TRANS_N, &nocc, &ngrids, &blen,
                               &D1, dm+b0*nocc, &nocc, ao+b0, &nao,
                               &D1, vm, &nocc);
                }
        }
}


/* vm[ngrids,nocc] = ao[ngrids,i] * dm[i,nocc] */
void VXCdot_ao_dm(double *vm, double *ao, double *dm,
                  int nao, int nocc, int ngrids, int blksize, char *non0table,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nblk = (ngrids+blksize-1) / blksize;
        int ip, ib;

#pragma omp parallel default(none) \
        shared(vm, ao, dm, nao, nocc, ngrids, blksize, non0table, \
               atm, natm, bas, nbas, env) \
        private(ip, ib)
#pragma omp for nowait schedule(static)
        for (ib = 0; ib < nblk; ib++) {
                ip = ib * blksize;
                dot_ao_dm(vm+ip*nocc, ao+ip*nao, dm,
                          nao, nocc, MIN(ngrids-ip, blksize),
                          non0table+ib*nbas,
                          atm, natm, bas, nbas, env);
        }
}



static void dot_ao_ao(double *vv, double *ao1, double *ao2,
                      int nao, int ngrids, char *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D1 = 1;

        if (nao <= BOXSIZE) {
                dgemm_(&TRANS_N, &TRANS_T, &nao, &nao, &ngrids,
                       &D1, ao2, &nao, ao1, &nao, &D1, vv, &nao);
                return;
        }

        char empty[nbas];
        int nbox = (int)((nao-1)/BOXSIZE) + 1;
        int box_id, bas_id, nd, b0;
        int ao_id = 0;
        for (box_id = 0; box_id < nbox; box_id++) {
                empty[box_id] = 1;
        }

        box_id = 0;
        b0 = BOXSIZE;
        for (bas_id = 0; bas_id < nbas; bas_id++) {
                nd = (bas[ANG_OF] * 2 + 1) * bas[NCTR_OF];
                assert(nd < BOXSIZE);
                ao_id += nd;
                empty[box_id] &= !non0table[bas_id];
                if (ao_id == b0) {
                        box_id++;
                        b0 += BOXSIZE;
                } else if (ao_id > b0) {
                        box_id++;
                        b0 += BOXSIZE;
                        empty[box_id] = !non0table[bas_id];
                }
                bas += BAS_SLOTS;
        }

        int ib, jb, b0i, b0j, leni, lenj;

        for (ib = 0; ib < nbox; ib++) {
                if (!empty[ib]) {
                        b0i = ib * BOXSIZE;
                        leni = MIN(nao-b0i, BOXSIZE);
                        for (jb = 0; jb < nbox; jb++) {
                                if (!empty[jb]) {
                                        b0j = jb * BOXSIZE;
                                        lenj = MIN(nao-b0j, BOXSIZE);
                                        dgemm_(&TRANS_N, &TRANS_T,
                                               &lenj, &leni, &ngrids,
                                               &D1, ao2+b0j, &nao, ao1+b0i, &nao,
                                               &D1, vv+b0i*nao+b0j, &nao);
                                }
                        }
                }
        }
}


/* vv[nao,nao] = ao1[i,nao] * ao2[i,nao] */
void VXCdot_ao_ao(double *vv, double *ao1, double *ao2,
                  int nao, int ngrids, int blksize, char *non0table,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nblk = (ngrids+blksize-1) / blksize;
        int ip, ib;
        double *v_priv;

        memset(vv, 0, sizeof(double) * nao * nao);

#pragma omp parallel default(none) \
        shared(vv, ao1, ao2, nao, ngrids, blksize, non0table, \
               atm, natm, bas, nbas, env) \
        private(ip, ib, v_priv)
        {
                v_priv = malloc(sizeof(double) * nao * nao);
                memset(v_priv, 0, sizeof(double) * nao * nao);
#pragma omp for nowait schedule(static)
                for (ib = 0; ib < nblk; ib++) {
                        ip = ib * blksize;
                        dot_ao_ao(v_priv, ao1+ip*nao, ao2+ip*nao,
                                  nao, MIN(ngrids-ip, blksize), non0table+ib*nbas,
                                  atm, natm, bas, nbas, env);
                }
#pragma omp critical
                {
                        for (ip = 0; ip < nao*nao; ip++) {
                                vv[ip] += v_priv[ip];
                        }
                }
                free(v_priv);
        }
}
