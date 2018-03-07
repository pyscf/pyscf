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
 * Author: Alexander Sokolov <alexander.y.sokolov@gmail.com>
 *
 *  C functions for Heat-Bath CI implementation
 */

#include <stdint.h>
#define MAX_THREADS     256

void contract_h_c(double *h1, double *eri, int norb, int neleca, int nelecb, uint64_t *strs, double *civec, double *hdiag, uint64_t ndet, double *ci1);
int n_excitations(uint64_t *str1, uint64_t *str2, int nset);
int popcount(uint64_t bb);
int *get_single_excitation(uint64_t *str1, uint64_t *str2, int nset);
int *get_double_excitation(uint64_t *str1, uint64_t *str2, int nset);
int trailz(uint64_t v);
char *int2bin(uint64_t i);
double compute_cre_des_sign(int a, int i, uint64_t *stria, int nset);
int *compute_occ_list(uint64_t *string, int nset, int norb, int nelec);
int *compute_vir_list(uint64_t *string, int nset, int norb, int nelec);
void select_strs(double *h1, double *eri, double *jk, uint64_t *eri_sorted, uint64_t *jk_sorted, int norb, int neleca, int nelecb, uint64_t *strs, double *civec, uint64_t ndet_start, uint64_t ndet_finish, double select_cutoff, uint64_t *strs_add, uint64_t* strs_add_size);
uint64_t *toggle_bit(uint64_t *str, int nset, int p);
int order(uint64_t *strs_i, uint64_t *strs_j, int nset);
void qsort_idx(uint64_t *strs, uint64_t *idx, uint64_t *nstrs, int nset, uint64_t *new_idx);
void argunique(uint64_t *strs, uint64_t *sort_idx, uint64_t *nstrs, int nset);
void contract_ss_c(int norb, int neleca, int nelecb, uint64_t *strs, double *civec, uint64_t ndet, double *ci1);
void contract_h_c_ss_c(double *h1, double *eri, int norb, int neleca, int nelecb, uint64_t *strs, double *civec, double *hdiag, uint64_t ndet, double *ci1, double *ci2);
void compute_rdm12s(int norb, int neleca, int nelecb, uint64_t *strs, double *civec, uint64_t ndet, double *rdm1a, double *rdm1b, double *rdm2aa, double *rdm2ab, double *rdm2bb);
