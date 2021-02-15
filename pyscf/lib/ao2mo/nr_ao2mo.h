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
#include "vhf/cvhf.h"

#if !defined HAVE_DEFINED_NR_AO2MOENVS_H
#define HAVE_DEFINED_NR_AO2MOENVS_H
struct _AO2MOEnvs {
        int natm;
        int nbas;
        int *atm;
        int *bas;
        double *env;
        int nao;
        int klsh_start;
        int klsh_count;
        int bra_start;
        int bra_count;
        int ket_start;
        int ket_count;
        int ncomp;
        int *ao_loc;
        double *mo_coeff;
        CINTOpt *cintopt;
        CVHFOpt *vhfopt;
};
#endif

void AO2MOnr_e1fill_drv(int (*intor)(), void (*fill)(), double *eri,
                        int klsh_start, int klsh_count, int nkl, int ncomp,
                        int *ao_loc, CINTOpt *cintopt, CVHFOpt *vhfopt,
                        int *atm, int natm, int *bas, int nbas, double *env);

void AO2MOnr_e1_drv(int (*intor)(), void (*fill)(), void (*ftrans)(), int (*fmmm)(),
                    double *eri, double *mo_coeff,
                    int klsh_start, int klsh_count, int nkl, int ncomp,
                    int *orbs_slice, int *ao_loc,
                    CINTOpt *cintopt, CVHFOpt *vhfopt,
                    int *atm, int natm, int *bas, int nbas, double *env);

void AO2MOnr_e2_drv(void (*ftrans)(), int (*fmmm)(),
                    double *vout, double *vin, double *mo_coeff,
                    int nij, int nao, int *orbs_slice, int *ao_loc, int nbas);

int AO2MOmmm_bra_nr_s1(double *vout, double *vin, double *buf,
                       struct _AO2MOEnvs *envs, int seekdim);
int AO2MOmmm_ket_nr_s1(double *vout, double *vin, double *buf,
                       struct _AO2MOEnvs *envs, int seekdim);
int AO2MOmmm_bra_nr_s2(double *vout, double *vin, double *buf,
                       struct _AO2MOEnvs *envs, int seekdim);
int AO2MOmmm_ket_nr_s2(double *vout, double *vin, double *buf,
                       struct _AO2MOEnvs *envs, int seekdim);
