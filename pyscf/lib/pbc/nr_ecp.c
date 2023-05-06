/* Copyright 2023 The PySCF Developers. All Rights Reserved.

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

#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include "cint.h"
#include "np_helper/np_helper.h"
#include "pbc/pbc.h"
#include "gto/nr_ecp.h"

int ECPtype_scalar_cart(double *gctr, int *shls, int *ecpbas, int necpbas,
                        int *atm, int natm, int *bas, int nbas, double *env,
                        ECPOpt *opt, double *cache);
void ECPscalar_distribute(double *out, double *gctr, const int *dims,
                          const int comp, const int di, const int dj);

int PBCECP_loop(Function_cart intor,
                double *gctr, int *cell0_shls, int *bvk_cells, int cutoff,
                float *rij_cond, CINTEnvVars *envs_cint, BVKEnvs *envs_bvk,
                double *cache)
{
        size_t Nbas = envs_cint->nbas;
        int nbasp = envs_bvk->nbasp;
        int *seg_loc = envs_bvk->seg_loc;
        int *seg2sh = envs_bvk->seg2sh;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        int cell_i = bvk_cells[0];
        int cell_j = bvk_cells[1];
        int ish_bvk = ish_cell0 + cell_i * nbasp;
        int jsh_bvk = jsh_cell0 + cell_j * nbasp;
        int iseg0 = seg_loc[ish_bvk];
        int jseg0 = seg_loc[jsh_bvk];
        int iseg1 = seg_loc[ish_bvk+1];
        int jseg1 = seg_loc[jsh_bvk+1];
        int nish = seg2sh[iseg1] - seg2sh[iseg0];
        int njsh = seg2sh[jseg1] - seg2sh[jseg0];
        int nij = nish * njsh;
        int rij_off = seg2sh[iseg0] * njsh + seg2sh[jseg0];

        // basis in remote bvk cell may be skipped
        if (iseg0 == iseg1 || jseg0 == jseg1) {
                return 0;
        }

        ECPOpt *opt = (ECPOpt *)envs_cint->opt;
        int *atm = envs_cint->atm;
        int *bas = envs_cint->bas;
        int natm = envs_cint->natm;
        int nbas = envs_cint->nbas;
        double *env = envs_cint->env;
        int *all_ecpbas = bas + (int)(env[AS_ECPBAS_OFFSET])*BAS_SLOTS;
        int necpbas = (int)(env[AS_NECPBAS]);
        if (necpbas == 0) {
                return 0;
        }

        int ish0 = seg2sh[iseg0];
        int jsh0 = seg2sh[jseg0];
        int li = bas[ANG_OF+ish0*BAS_SLOTS];
        int lj = bas[ANG_OF+jsh0*BAS_SLOTS];
        int nci = bas[NCTR_OF+ish0*BAS_SLOTS];
        int ncj = bas[NCTR_OF+jsh0*BAS_SLOTS];
        int nfi = (li+1)*(li+2)/2;
        int nfj = (lj+1)*(lj+2)/2;
        int n_comp = envs_cint->ncomp_e1 * envs_cint->ncomp_tensor;
        int dij = nfi * nfj * nci * ncj * n_comp;
        int has_value = 0;
        NPdset0(gctr, dij);

        int shls[3];
        int ish, jsh, ksh, esh, atm_id, nprim, ptr;
        int iseg, jseg;
        int ish1, jsh1;
        int ksh0, ksh1;
        int16_t *sindex = envs_bvk->qindex;
        float *xij_cond = rij_cond;
        float *yij_cond = rij_cond + nij;
        float *zij_cond = rij_cond + nij * 2;
        int16_t *sij_idx;
        float xk, yk, zk, dx, dy, dz, r2;

        float ai, aj, ak, aij;
        float eta, theta, theta_r2, fac;
        float ij_cutoff, sij;
        ECPOpt opt1;
        double *u_ecp = NULL;
        if (opt != NULL) {
                u_ecp = opt->u_ecp;
                opt = &opt1;
        }

        int atm_last = -1;
        int ecp_natm = 0;
        int *ecpbas;
        int *ecploc;
        MALLOC_INSTACK(ecploc, natm+1);
        for (esh = 0; esh < necpbas; esh++) {
                atm_id = all_ecpbas[ATOM_OF+esh*BAS_SLOTS];
                ecpbas = all_ecpbas + esh * BAS_SLOTS;
                if (atm_id != atm_last) {
                        ecploc[ecp_natm] = esh;
                        ecp_natm++;
                        atm_last = atm_id;
                }
        }
        ecploc[ecp_natm] = necpbas;

        for (esh = 0; esh < ecp_natm; esh++) {
                ksh0 = ecploc[esh];
                ksh1 = ecploc[esh+1];
                ecpbas = all_ecpbas + ksh0 * BAS_SLOTS;
                necpbas = ksh1 - ksh0;
                eta = 1.f;
                for (ksh = 0; ksh < necpbas; ksh++) {
                        nprim = ecpbas[ksh*BAS_SLOTS+NPRIM_OF];
                        ak = env[ecpbas[ksh*BAS_SLOTS+PTR_EXP]+nprim-1];
                        eta = MIN(ak, eta);
                }
                if (opt != NULL) {
                        // opt->u_ecp used by ECPrad_part was initialized for all atoms.
                        // shifts the u_ecp pointer to the u_ecp of atm_id
                        opt1.u_ecp = u_ecp + ksh0 * (1 << LEVEL_MAX);
                }

                ptr = atm(PTR_COORD, ecpbas[ATOM_OF]);
                xk = env[ptr];
                yk = env[ptr+1];
                zk = env[ptr+2];

                fac = logf(eta)/4;
                ij_cutoff = cutoff + fac * LOG_ADJUST;
                for (iseg = iseg0; iseg < iseg1; iseg++) {
                        ish0 = seg2sh[iseg];
                        ish1 = seg2sh[iseg+1];
                        ai = env[bas(PTR_EXP,ish0) + bas(NPRIM_OF,ish0)-1];
                        for (jseg = jseg0; jseg < jseg1; jseg++) {
                                jsh0 = seg2sh[jseg];
                                jsh1 = seg2sh[jseg+1];
                                aj = env[bas(PTR_EXP,jsh0) + bas(NPRIM_OF,jsh0)-1];
                                aij = ai + aj;
                                theta = eta * aij / (eta + aij);
for (ish = ish0; ish < ish1; ish++) {
        shls[0] = ish;
        sij_idx = sindex + ish * Nbas;
        for (jsh = jsh0; jsh < jsh1; jsh++) {
                sij = sij_idx[jsh];
                dx = xk - xij_cond[ish * njsh + jsh - rij_off];
                dy = yk - yij_cond[ish * njsh + jsh - rij_off];
                dz = zk - zij_cond[ish * njsh + jsh - rij_off];
                r2 = dx * dx + dy * dy + dz * dz;
                theta_r2 = theta * r2 + logf(r2 + 1e-30f);
                if (theta_r2*LOG_ADJUST + ij_cutoff < sij) {
                        shls[1] = jsh;
                        has_value = intor(gctr, shls, ecpbas, necpbas, atm, natm, bas,
                                          nbas, env, opt, cache) | has_value;
                }
        }
}
                        }
                }
        }
        return has_value;
}

int PBCECPscalar_cart(double *eri_buf, int *cell0_shls, int *bvk_cells, int cutoff,
                      float *rij_cond, CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int *bas = envs_cint->bas;
        int nbasp = envs_bvk->nbasp;
        int *seg_loc = envs_bvk->seg_loc;
        int *seg2sh = envs_bvk->seg2sh;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        int cell_i = bvk_cells[0];
        int cell_j = bvk_cells[1];
        int ish_bvk = ish_cell0 + cell_i * nbasp;
        int jsh_bvk = jsh_cell0 + cell_j * nbasp;
        int iseg0 = seg_loc[ish_bvk];
        int jseg0 = seg_loc[jsh_bvk];
        int ish0 = seg2sh[iseg0];
        int jsh0 = seg2sh[jseg0];
        int li = bas[ANG_OF+ish0*BAS_SLOTS];
        int lj = bas[ANG_OF+jsh0*BAS_SLOTS];
        int nci = bas[NCTR_OF+ish0*BAS_SLOTS];
        int ncj = bas[NCTR_OF+jsh0*BAS_SLOTS];
        int nfi = (li+1)*(li+2)/2;
        int nfj = (lj+1)*(lj+2)/2;
        int comp = 1;
        double *cache = eri_buf + nfi * nfj * nci * ncj * comp;
        int has_value = PBCECP_loop(ECPtype_scalar_cart,
                                    eri_buf, cell0_shls, bvk_cells, cutoff,
                                    rij_cond, envs_cint, envs_bvk, cache);
        return has_value;
}

int PBCECPscalar_sph(double *eri_buf, int *cell0_shls, int *bvk_cells, int cutoff,
                   float *rij_cond, CINTEnvVars *envs_cint, BVKEnvs *envs_bvk)
{
        int *bas = envs_cint->bas;
        int nbasp = envs_bvk->nbasp;
        int *seg_loc = envs_bvk->seg_loc;
        int *seg2sh = envs_bvk->seg2sh;
        int ish_cell0 = cell0_shls[0];
        int jsh_cell0 = cell0_shls[1];
        int cell_i = bvk_cells[0];
        int cell_j = bvk_cells[1];
        int ish_bvk = ish_cell0 + cell_i * nbasp;
        int jsh_bvk = jsh_cell0 + cell_j * nbasp;
        int iseg0 = seg_loc[ish_bvk];
        int jseg0 = seg_loc[jsh_bvk];
        int ish0 = seg2sh[iseg0];
        int jsh0 = seg2sh[jseg0];
        int li = bas[ANG_OF+ish0*BAS_SLOTS];
        int lj = bas[ANG_OF+jsh0*BAS_SLOTS];
        int nci = bas[NCTR_OF+ish0*BAS_SLOTS];
        int ncj = bas[NCTR_OF+jsh0*BAS_SLOTS];
        int nfi = (li+1)*(li+2)/2;
        int nfj = (lj+1)*(lj+2)/2;
        int di = (li*2+1);
        int dj = (lj*2+1);
        int comp = 1;
        double *gcart = eri_buf + di*dj*nci*ncj * comp;
        double *cache = gcart + nfi*nfj*nci*ncj * comp;
        int has_value = PBCECP_loop(ECPtype_scalar_cart,
                                    gcart, cell0_shls, bvk_cells, cutoff,
                                    rij_cond, envs_cint, envs_bvk, cache);
        if (!has_value) {
                NPdset0(eri_buf, di*dj*nci*ncj*comp);
                return has_value;
        }

        int j;
        if (li < 2) {
                for (j = 0; j < ncj * comp; j++) {
                        CINTc2s_ket_sph1(eri_buf+j*dj*nfi*nci, gcart+j*nfj*nfi*nci,
                                         nfi*nci, nfi*nci, lj);
                }
        } else {
                for (j = 0; j < ncj * comp; j++) {
                        CINTc2s_ket_sph1(cache+j*dj*nfi*nci, gcart+j*nfj*nfi*nci,
                                         nfi*nci, nfi*nci, lj);
                }
                CINTc2s_bra_sph(eri_buf, nci*dj*ncj * comp, cache, li);
        }
        return has_value;
}
