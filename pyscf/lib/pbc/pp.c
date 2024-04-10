/* Copyright 2021- The PySCF Developers. All Rights Reserved.

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
 * Author: Xing Zhang <zhangxing.nju@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include "config.h"
#include "cint.h"
#include "gto/gto.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"
#include "pbc/fill_ints.h"
#include "pbc/neighbor_list.h"

#define HL_TABLE_SLOTS  7
//#define ATOM_OF         0
//#define ANG_OF          1
#define HL_DIM_OF       2
#define HL_DATA_OF      3
#define HL_OFFSET0      4
#define HF_OFFSET1      5
#define HF_OFFSET2      6
#define MAX_THREADS     256


static void _ppnl_fill_g(void (*fsort)(), double* out, double** ints,
                         int comp, int ish, int jsh, double* buf,
                         int *shls_slice, int *ao_loc,
                         int* hl_table, double* hl_data, int nhl,
                         NeighborListOpt* nlopt)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];

    ish += ish0;
    jsh += jsh0;

    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di *dj;
    const int ioff = ao_loc[ish] - ao_loc[ish0];
    const int joff = ao_loc[jsh] - ao_loc[jsh0];
    const int naoi = ao_loc[ish1] - ao_loc[ish0];
    const int naoj = ao_loc[jsh1] - ao_loc[jsh0];

    int i, j, ij, pi, pj, ksh;
    int hl_dim, nd;
    int shls_ki[2], shls_kj[2];
    int *table, *offset;
    double *hl;
    for (ij = 0; ij < dij; ij++) {
        buf[ij] = 0;
    }

    int (*fprescreen)();
    if (nlopt != NULL) {
        fprescreen = nlopt->fprescreen;
    } else {
        fprescreen = NLOpt_noscreen;
    }

    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const double D1 = 1.;
    for (ksh = 0; ksh < nhl; ksh++) {
        shls_ki[0] = ksh;
        shls_ki[1] = ish;
        shls_kj[0] = ksh;
        shls_kj[1] = jsh;
        if ((*fprescreen)(shls_ki, nlopt) && (*fprescreen)(shls_kj, nlopt)) {
            table = hl_table + ksh * HL_TABLE_SLOTS;
            hl_dim = table[HL_DIM_OF];
            nd = table[ANG_OF] * 2 + 1;
            offset = table + HL_OFFSET0;
            hl = hl_data + table[HL_DATA_OF];
            for (i=0; i<hl_dim; i++) {
                pi = offset[i];
                for (j=0; j<hl_dim; j++) {
                    pj = offset[j];
                    dgemm_(&TRANS_N, &TRANS_T, &di, &dj, &nd,
                           hl+j+i*hl_dim, ints[i]+pi*naoi+ioff, &naoi,
                           ints[j]+pj*naoj+joff, &naoj, &D1, buf, &di);
                }
            }
        }
    }
    (*fsort)(out, buf, shls_slice, ao_loc, comp, ish, jsh);
}


void ppnl_fill_gs1(double* out, double** ints,
                   int comp, int ish, int jsh, double* buf,
                   int *shls_slice, int *ao_loc,
                   int* hl_table, double* hl_data, int nhl,
                   NeighborListOpt* nlopt)
{
    _ppnl_fill_g(&sort2c_gs1, out, ints, comp, ish, jsh, buf,
                 shls_slice, ao_loc, hl_table, hl_data, nhl, nlopt);
}


void ppnl_fill_gs2(double* out, double** ints,
                   int comp, int ish, int jsh, double* buf,
                   int *shls_slice, int *ao_loc,
                   int* hl_table, double* hl_data, int nhl,
                   NeighborListOpt* nlopt)
{
    int ip = ish + shls_slice[0];
    int jp = jsh + shls_slice[2];
    if (ip > jp) {
        _ppnl_fill_g(&sort2c_gs2_igtj, out, ints, comp, ish, jsh, buf,
                     shls_slice, ao_loc, hl_table, hl_data, nhl, nlopt);
    } else if (ip == jp) {
        _ppnl_fill_g(&sort2c_gs2_ieqj, out, ints, comp, ish, jsh, buf,
                     shls_slice, ao_loc, hl_table, hl_data, nhl, nlopt);
    }
}


void contract_ppnl(void (*fill)(), double* out,
                   double* ppnl_half0, double* ppnl_half1, double* ppnl_half2,
                   int comp, int* shls_slice, int *ao_loc,
                   int* hl_table, double* hl_data, int nhl,
                   NeighborListOpt* nlopt)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    const size_t nijsh = (size_t) nish * njsh;

    double *ints[3] = {ppnl_half0, ppnl_half1, ppnl_half2};

    int di = GTOmax_shell_dim(ao_loc, shls_slice+0, 1);
    int dj = GTOmax_shell_dim(ao_loc, shls_slice+2, 1);
    size_t buf_size = di*dj*comp;

    #pragma omp parallel
    {
        int ish, jsh;
        size_t ij;
        double *buf = (double*) malloc(sizeof(double) * buf_size);
        #pragma omp for schedule(dynamic)
        for (ij = 0; ij < nijsh; ij++) {
            ish = ij / njsh;
            jsh = ij % njsh;
            (*fill)(out, ints, comp, ish, jsh, buf,
                    shls_slice, ao_loc, hl_table, hl_data, nhl, nlopt);
        }
        free(buf);
    }
}


void contract_ppnl_ip1(double* out, int comp,
                       double* ppnl_half0, double* ppnl_half1, double* ppnl_half2,
                       double* ppnl_half_ip2_0, double* ppnl_half_ip2_1, double* ppnl_half_ip2_2,
                       int* hl_table, double* hl_data, int nhl, int nao, int* naux,
                       int* aux_id)
{
    const int One = 1;
    const char TRANS_N = 'N';
    //const char TRANS_T = 'T';
    const double D1 = 1.;
    const double D0 = 0.;

    size_t nao_pair = (size_t) nao * nao;
    memset(out, 0, nao_pair*comp*sizeof(double));

    size_t n2[3];
    n2[0] = (size_t) nao * naux[0];
    n2[1] = (size_t) nao * naux[1];
    n2[2] = (size_t) nao * naux[2];
    size_t buf_size = 54 * (size_t) nao + 27;

#pragma omp parallel
{
    size_t ib, id, i, p, ic;
    double *pout;
    double *buf = (double*) malloc(sizeof(double)*buf_size);

    #pragma omp for schedule(dynamic)
    for (p = 0; p < nao; p++){
        pout = out + (size_t)p*nao;
        for (id = 0; id < nhl; id++) {
            ib = aux_id[id];
            int *table = hl_table + ib * HL_TABLE_SLOTS;
            int hl_dim = table[HL_DIM_OF];
            int ptr = table[HL_DATA_OF];
            int nd = table[ANG_OF] * 2 + 1;
            int *offset = table + HL_OFFSET0;
            double *hl = hl_data + ptr;
            int lp_dim = nd * nao;
            int ilp_dim = hl_dim * lp_dim;
            int il_dim = hl_dim * nd;

            double *ilp = buf;
            double *ilp_ip2 = ilp + ilp_dim;
            double *hilp = ilp_ip2 + nd*3;
            for (ic = 0; ic < comp; ic++) {
                for (i=0; i<hl_dim; i++) {
                    int p0 = offset[i];
                    if (i == 0) {
                        dcopy_(&lp_dim, ppnl_half0+p0*nao, &One, ilp+i*lp_dim, &One);
                        dcopy_(&nd, ppnl_half_ip2_0+p+p0*nao+ic*n2[0], &nao, ilp_ip2+i*nd, &One);
                    }
                    else if (i == 1) {
                        dcopy_(&lp_dim, ppnl_half1+p0*nao, &One, ilp+i*lp_dim, &One);
                        dcopy_(&nd, ppnl_half_ip2_1+p+p0*nao+ic*n2[1], &nao, ilp_ip2+i*nd, &One);
                    }
                    else if (i == 2) {
                        dcopy_(&lp_dim, ppnl_half2+p0*nao, &One, ilp+i*lp_dim, &One);
                        dcopy_(&nd, ppnl_half_ip2_2+p+p0*nao+ic*n2[2], &nao, ilp_ip2+i*nd, &One);
                    }
                }
                dgemm_(&TRANS_N, &TRANS_N, &lp_dim, &hl_dim, &hl_dim, 
                       &D1, ilp, &lp_dim, hl, &hl_dim, &D0, hilp, &lp_dim);
                dgemm_(&TRANS_N, &TRANS_N, &nao, &One, &il_dim,
                       &D1, hilp, &nao, ilp_ip2, &il_dim, &D1, pout+ic*nao_pair, &nao);
            }
        }
    }
    free(buf);
}
}


static void _contract_vnuc_ip1_dm(double* out, double* in, double* dm, int comp,
                                  int* shls_slice, int* ao_loc, int* bas,
                                  int ish, int jsh, int naoi, int katm)
{
    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int iatm = bas[ATOM_OF+ish*BAS_SLOTS];

    const int One = 1;
    int ic, j;
    double buf[comp];
    double *pdm;
    for (ic = 0; ic < comp; ic++) {
        buf[ic] = 0;
        pdm = dm;
        for (j = 0; j < dj; j++) {
            buf[ic] += ddot_(&di, in, &One, pdm, &One);
            in += di;
            pdm += naoi;
        }
    }

    for (ic = 0; ic < comp; ic++) {
        out[iatm*comp+ic] += buf[ic];
        out[katm*comp+ic] -= buf[ic];
    }
}


void ppnl_nuc_grad_fill_gs1(double* out, double* dm, int comp,
                            double** ints, double** ints_ip2,
                            int* hl_table, double* hl_data, int nhl, int* naux,
                            int* shls_slice, int* ao_loc, int* bas, double* buf, int ish, int jsh,
                            NeighborListOpt* nlopt)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];

    ish += ish0;
    jsh += jsh0;

    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di * dj;
    const size_t dijm = (size_t)dij * comp;
    const int i0 = ao_loc[ish] - ao_loc[ish0];
    const int j0 = ao_loc[jsh] - ao_loc[jsh0];
    const int naoi = ao_loc[ish1] - ao_loc[ish0];
    const int naoj = ao_loc[jsh1] - ao_loc[jsh0];

    size_t n2[3];
    n2[0] = (size_t) naoi * naux[0];
    n2[1] = (size_t) naoi * naux[1];
    n2[2] = (size_t) naoi * naux[2];

    int (*fprescreen)();
    if (nlopt != NULL) {
        fprescreen = nlopt->fprescreen;
    } else {
        fprescreen = NLOpt_noscreen;
    }

    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const double D1 = 1.;

    int i, j, pi, pj, ksh, ic;
    int katm, l, hl_dim, nd;
    int shls_ki[2], shls_kj[2];
    int *table, *offset;
    double *hl;
    for (ksh = 0; ksh < nhl; ksh++) {
        shls_ki[0] = ksh;
        shls_ki[1] = ish;
        shls_kj[0] = ksh;
        shls_kj[1] = jsh;
        if ((*fprescreen)(shls_ki, nlopt) && (*fprescreen)(shls_kj, nlopt)) {
            table = hl_table + ksh * HL_TABLE_SLOTS;
            katm = table[ATOM_OF];
            l = table[ANG_OF];
            hl_dim = table[HL_DIM_OF];
            nd = 2 * l + 1;
            offset = table + HL_OFFSET0;
            hl = hl_data + table[HL_DATA_OF];

            memset(buf, 0, dijm*sizeof(double));
            for (ic = 0; ic < comp; ic++) {
                for (i=0; i<hl_dim; i++) {
                    pi = offset[i];
                    for (j=0; j<hl_dim; j++) {
                        pj = offset[j];
                        dgemm_(&TRANS_N, &TRANS_T, &di, &dj, &nd,
                               hl+j+i*hl_dim, ints_ip2[i]+ic*n2[i]+pi*naoi+i0, &naoi,
                               ints[j]+pj*naoj+j0, &naoj, &D1, buf+ic*dij, &di);
                    }
                }
            }
            _contract_vnuc_ip1_dm(out, buf, dm+j0*naoi+i0, comp,
                                  shls_slice, ao_loc, bas,
                                  ish, jsh, naoi, katm);
        }
    }
}


void contract_ppnl_nuc_grad(void (*fill)(), double* grad, double* dm, int comp,
                            double* ppnl_half0, double* ppnl_half1, double* ppnl_half2,
                            double* ppnl_half_ip2_0, double* ppnl_half_ip2_1, double* ppnl_half_ip2_2,
                            int* hl_table, double* hl_data, int nhl, int* naux,
                            int* shls_slice, int* ao_loc, int* bas, int natm,
                            NeighborListOpt* nlopt)
{
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    const size_t nijsh = (size_t)nish * njsh;

    int di = GTOmax_shell_dim(ao_loc, shls_slice+0, 1);
    int dj = GTOmax_shell_dim(ao_loc, shls_slice+2, 1);
    size_t buf_size = di*dj*comp;

    double *ints[3] = {ppnl_half0, ppnl_half1, ppnl_half2};
    double *ints_ip2[3] = {ppnl_half_ip2_0, ppnl_half_ip2_1, ppnl_half_ip2_2};

    double *gradbufs[MAX_THREADS];
    #pragma omp parallel
    {
        int ish, jsh;
        size_t ij;
        double *grad_loc;
        int thread_id = omp_get_thread_num();
        if (thread_id == 0) {
            grad_loc = grad;
        } else {
            grad_loc = calloc(natm*comp, sizeof(double));
        }
        gradbufs[thread_id] = grad_loc;
        double *buf = (double*) malloc(sizeof(double)*buf_size);

        #pragma omp for schedule(dynamic)
        for (ij = 0; ij < nijsh; ij++) {
            ish = ij / njsh;
            jsh = ij % njsh;

            (*fill)(grad_loc, dm, comp, ints, ints_ip2,
                    hl_table, hl_data, nhl, naux,
                    shls_slice, ao_loc, bas, buf, ish, jsh, nlopt);
        }
        free(buf);

        NPomp_dsum_reduce_inplace(gradbufs, natm*comp);
        if (thread_id != 0) {
            free(grad_loc);
        }
    }
}


void pp_loc_part1_gs(double complex* out, double* coulG,
                     double* Gv, double* G2, int G0idx, int ngrid,
                     double* Z, double* coords, double* rloc,
                     int natm)
{
#pragma omp parallel
{
    int ig, ia;
    double vlocG, r0, RG;
    double *Gv_loc, *coords_local;
    #pragma omp for schedule(static)
    for (ig = 0; ig < ngrid; ig++){
        out[ig] = 0;
        Gv_loc = Gv + ig*3;
        for (ia = 0; ia < natm; ia++)
        {
            coords_local = coords + ia*3;
            RG = (coords_local[0] * Gv_loc[0]
                  + coords_local[1] * Gv_loc[1]
                  + coords_local[2] * Gv_loc[2]);

            r0 = rloc[ia];
            if (r0 > 0) {
                if (ig == G0idx) {
                    vlocG = -2. * M_PI * Z[ia] * r0*r0;
                }
                else {
                    vlocG = Z[ia] * coulG[ig] * exp(-0.5*r0*r0 * G2[ig]);
                }
            }
            else { // Z/r
                vlocG = Z[ia] * coulG[ig];
            }
            out[ig] -= (vlocG * cos(RG)) - (vlocG * sin(RG)) * _Complex_I;
        }
    }
}
}
