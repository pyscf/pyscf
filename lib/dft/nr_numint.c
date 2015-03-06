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

        int bas_id, nd;
        int empty = 1;
        int b0 = 0;
        int b1 = BOXSIZE;
        int blen = BOXSIZE;
        int ao_id = 0;

        memset(vm, 0, sizeof(double) * ngrids * nocc);

        for (bas_id = 0; bas_id < nbas; bas_id++) {
                nd = (bas[ANG_OF] * 2 + 1) * bas[NCTR_OF];
                if (non0table[bas_id] != 0) {
                        empty = 0;
                }
                ao_id += nd;

                while (ao_id >= b1) { // current AO out of box
                        if (!empty) {
                                dgemm_(&TRANS_N, &TRANS_N, &nocc, &ngrids, &blen,
                                       &D1, dm+b0*nocc, &nocc, ao+b0, &nao,
                                       &D1, vm, &nocc);
                        }

                        if (ao_id > b1 && non0table[bas_id] != 0) {
                                empty = 0;
                        } else {
                                empty = 1;
                        }
                        b0 += BOXSIZE;
                        b1 += BOXSIZE;
                }

                bas += BAS_SLOTS;
        }

        blen = nao - b0;
        if (!empty && blen > 0) {
                dgemm_(&TRANS_N, &TRANS_N, &nocc, &ngrids, &blen,
                       &D1, dm+b0*nocc, &nocc, ao+b0, &nao, &D1, vm, &nocc);
        }
}


/* vm[ngrids,nocc] = ao[ngrids,i] * dm[i,nocc] */
void VXCdot_ao_dm(double *vm, double *ao, double *dm,
                  int nao, int nocc, int ngrids, int blksize, char *non0table,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nblk = (ngrids+blksize-1) / blksize;
        int ip, ib;
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

        int ib, jb, ndi, ndj;
        int ia, ja;
        int b0i = 0;
        int b1i = BOXSIZE;
        int bleni = BOXSIZE;
        int b0j = 0;
        int b1j = BOXSIZE;
        int blenj = BOXSIZE;
        int iempty, jempty;

        iempty = 1;
        ia = 0;
        for (ib = 0; ib < nbas; ib++) {
                ndi = (bas[ib*BAS_SLOTS+ANG_OF] * 2 + 1)
                        * bas[ib*BAS_SLOTS+NCTR_OF];
                if (non0table[ib] != 0) {
                        iempty = 0;
                }
                ia += ndi;

                while (ia >= b1i) { // current AO out of box
                        if (!iempty) {

                                b0j = 0;
                                b1j = BOXSIZE;
                                blenj = BOXSIZE;
                                jempty = 1;
                                ja = 0;
                                for (jb = 0; jb < nbas; jb++) {
                                        ndj = (bas[jb*BAS_SLOTS+ANG_OF]*2+1)
                                                * bas[jb*BAS_SLOTS+NCTR_OF];
                                        if (non0table[jb] != 0) {
                                                jempty = 0;
                                        }
                                        ja += ndj;

                                        while (ja >= b1j) { // current AO out of box
                                                if (!jempty) {
                                                        dgemm_(&TRANS_N, &TRANS_T,
                                                               &blenj, &bleni, &ngrids,
                                                               &D1, ao2+b0j, &nao, ao1+b0i, &nao,
                                                               &D1, vv+b0i*nao+b0j, &nao);

                                                }

                                                if (ja > b1j && non0table[jb] != 0) {
                                                        jempty = 0;
                                                } else {
                                                        jempty = 1;
                                                }
                                                b0j += BOXSIZE;
                                                b1j += BOXSIZE;
                                        }
                                }

                                blenj = nao - b0j;
                                if (!jempty && blenj > 0) {
                                        dgemm_(&TRANS_N, &TRANS_T, &blenj, &bleni, &ngrids,
                                               &D1, ao2+b0j, &nao, ao1+b0i, &nao,
                                               &D1, vv+b0i*nao+b0j, &nao);
                                }
                        }

                        if (ia > b1i && non0table[ib] != 0) {
                                iempty = 0;
                        } else {
                                iempty = 1;
                        }
                        b0i += BOXSIZE;
                        b1i += BOXSIZE;
                }
        }

        bleni = nao - b0i;
        if (!iempty && bleni > 0) {
                dgemm_(&TRANS_N, &TRANS_T, &nao, &bleni, &ngrids,
                       &D1, ao2, &nao, ao1+b0i, &nao,
                       &D1, vv+b0i*nao, &nao);
        }
}


/* vv[nao,nao] = ao1[i,nao] * ao2[i,nao] */
void VXCdot_ao_ao(double *vv, double *ao1, double *ao2,
                  int nao, int ngrids, int blksize, char *non0table,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nblk = (ngrids+blksize-1) / blksize;
        int ip, ib;

        memset(vv, 0, sizeof(double) * nao * nao);

        for (ib = 0; ib < nblk; ib++) {
                ip = ib * blksize;
                dot_ao_ao(vv, ao1+ip*nao, ao2+ip*nao,
                          nao, MIN(ngrids-ip, blksize), non0table+ib*nbas,
                          atm, natm, bas, nbas, env);
        }
}
