/* Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
  
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

#define AO_BLOCK_SIZE   64

#define NOVALUE 0x7fffffff

#if !defined(HAVE_DEFINED_INTORENV_H)
#define HAVE_DEFINED_INTORENV_H
typedef struct {
        int v_ket_nsh;  /* v_ket_sh1 - v_ket_sh0 */
        int offset0_outptr;  /* v_bra_sh0 * v_ket_nsh + v_ket_sh0 */
        int dm_dims[2];
        int *outptr;   /* Offset array to index the data which are stored in stack */
        double *data;  /* Stack to store data */
        int stack_size;  /* How many data have been used */
        int ncomp;
        int nblock;
        int ao_off[4];  /* Offsets for the first AO in the block */
        int shape[4];  /* The shape of eri for the target block */
        int block_quartets[4]; /* 4 block Ids */
        int *keys_cache;  /* key (=i*nblock+j) to track allocated data */
        int key_counts; /* Number of allocated data block */
} JKArray;

typedef struct {
        int ibra_shl0;  // = 0, 2, 4, 6. The index in shls_slice
        int iket_shl0;
        int obra_shl0;
        int oket_shl0;
        void (*contract)(double *eri, double *dm, JKArray *vjk, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1);
        size_t (*data_size)(int *shls_slice, int *ao_loc);
        void (*sanity_check)(int *shls_slice);
        void (*write_back)(double *vjk, JKArray *jkarray,
                           int *shls_slice, int *ao_loc,
                           int *block_iloc, int *block_jloc,
                           int *block_kloc, int *block_lloc);
} JKOperator;

typedef struct {
        int natm;
        int nbas;
        int *atm;
        int *bas;
        double *env;
        int *shls_slice;
        int *ao_loc;  /* size of nbas+1, last element = nao */
        int *tao;     /* time reversal mappings, index start from 1 */
        CINTOpt *cintopt;
        int ncomp;
} IntorEnvs;
#endif

void CVHFnr_direct_drv(int (*intor)(), void (*fdot)(), JKOperator **jkop,
                       double **dms, double **vjk, int n_dm, int ncomp,
                       int *shls_slice, int *ao_loc,
                       CINTOpt *cintopt, CVHFOpt *vhfopt,
                       int *atm, int natm, int *bas, int nbas, double *env);

JKArray *CVHFallocate_JKArray(JKOperator *op, int *shls_slice, int *ao_loc,
                              int ncomp, int nblock, int size_limit);
void CVHFdeallocate_JKArray(JKArray *jkarray);
double *CVHFallocate_and_reorder_dm(JKOperator *op, double *dm,
                                    int *shls_slice, int *ao_loc);
void CVHFzero_out_vjk(double *vjk, JKOperator *op,
                      int *shls_slice, int *ao_loc, int ncomp);
void CVHFassemble_v(double *vjk, JKOperator *op, JKArray *jkarray,
                    int *shls_slice, int *ao_loc);
