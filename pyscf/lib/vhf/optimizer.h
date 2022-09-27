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

#if !defined(HAVE_DEFINED_CVHFOPT_H)
#define HAVE_DEFINED_CVHFOPT_H
typedef struct CVHFOpt_struct {
    int nbas;
    int ngrids;
    double direct_scf_cutoff;
    double *q_cond;
    double *dm_cond;
    int (*fprescreen)(int *shls, struct CVHFOpt_struct *opt,
                      int *atm, int *bas, double *env);
    int (*r_vkscreen)(int *shls, struct CVHFOpt_struct *opt,
                      double **dms_cond, int n_dm, double *dm_atleast,
                      int *atm, int *bas, double *env);
} CVHFOpt;
#endif

void CVHFinit_optimizer(CVHFOpt **opt, int *atm, int natm,
                        int *bas, int nbas, double *env);

void CVHFdel_optimizer(CVHFOpt **opt);

int CVHFnoscreen(int *shls, CVHFOpt *opt, int *atm, int *bas, double *env);
int CVHFnr_schwarz_cond(int *shls, CVHFOpt *opt, int *atm, int *bas, double *env);
int CVHFnrs8_prescreen(int *shls, CVHFOpt *opt, int *atm, int *bas, double *env);
int CVHFnrs8_vj_prescreen(int *shls, CVHFOpt *opt, int *atm, int *bas, double *env);
int CVHFnrs8_vk_prescreen(int *shls, CVHFOpt *opt, int *atm, int *bas, double *env);
int CVHFnrs8_prescreen_block(CVHFOpt *opt, int *ishls, int *jshls, int *kshls, int *lshls);
int CVHFnrs8_vj_prescreen_block(CVHFOpt *opt, int *ishls, int *jshls, int *kshls, int *lshls);
int CVHFnrs8_vk_prescreen_block(CVHFOpt *opt, int *ishls, int *jshls, int *kshls, int *lshls);

int CVHFr_vknoscreen(int *shls, CVHFOpt *opt,
                     double **dms_cond, int n_dm, double *dm_atleast,
                     int *atm, int *bas, double *env);

void CVHFsetnr_direct_scf(CVHFOpt *opt, int (*intor)(), CINTOpt *cintopt,
                          int *ao_loc, int *atm, int natm,
                          int *bas, int nbas, double *env);
void CVHFsetnr_direct_scf_dm(CVHFOpt *opt, double *dm, int nset, int *ao_loc,
                             int *atm, int natm, int *bas, int nbas, double *env);

void CVHFnr_optimizer(CVHFOpt **vhfopt, int (*intor)(), CINTOpt *cintopt,
                      int *ao_loc, int *atm, int natm,
                      int *bas, int nbas, double *env);

void CVHFset_int2e_q_cond(int (*intor)(), CINTOpt *cintopt, double *q_cond,
                          int *ao_loc, int *atm, int natm,
                          int *bas, int nbas, double *env);
