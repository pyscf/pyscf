/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
 
        http://www.apache.org/licenses/LICENSE-2.0
 
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include "cint.h"
#include "optimizer.h"

#define NOVALUE 0xffffffff

typedef struct {
        int v_bra_sh0;
        int v_bra_sh1;
        int v_ket_sh0;
        int v_ket_sh1;
        int v_ket_nsh;  /* v_ket_sh1 - v_ket_sh0 */
        int offset0_outptr;  /* v_bra_sh0 * v_ket_nsh + v_ket_sh0 */
        int dm_bra_sh0;
        int dm_bra_sh1;
        int dm_ket_sh0;
        int dm_ket_sh1;
        int dm_dims[2];
        int v_dims[2];
        int *outptr;   /* Offset array to index the data which are stored in stack */
        double *data;  /* Stack to store data */
        int stack_size;  /* How many data have been used */
        int ncomp;
} JKArray;

typedef struct {
        JKArray *(*allocate)(int *shls_slice, int *ao_loc, int ncomp);
        void (*deallocate)(JKArray *);
        void (*contract)(double *eri, double *dm, JKArray *vjk, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1);
        size_t (*data_size)(int *shls_slice, int *ao_loc);
        void (*sanity_check)(int *shls_slice);
} JKOperator;

typedef struct {
        int natm;
        int nbas;
        const int *atm;
        const int *bas;
        const double *env;
        int *shls_slice;
        int *ao_loc;  /* size of nbas+1, last element = nao */
        int *tao;     /* time reversal mappings, index start from 1 */
        CINTOpt *cintopt;
        int ncomp;
} IntorEnvs;

struct _VHFEnvs {
        int natm;
        int nbas;
        const int *atm;
        const int *bas;
        const double *env;
        int nao;
        int *ao_loc; // size of nbas+1, last element = nao
        int *tao; // time reversal mappings, index start from 1
        CVHFOpt *vhfopt;
        CINTOpt *cintopt;
};

void CVHFnr_direct_drv(int (*intor)(), void (*fdot)(), JKOperator **jkop,
                       double **dms, double **vjk, int n_dm, int ncomp,
                       int *shls_slice, int *ao_loc,
                       CINTOpt *cintopt, CVHFOpt *vhfopt,
                       int *atm, int natm, int *bas, int nbas, double *env);
