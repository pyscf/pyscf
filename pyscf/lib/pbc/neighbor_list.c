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
#include <math.h>
#include "config.h"
#include "cint.h"
#include "pbc/neighbor_list.h"

#define SQUARE(r)       (r[0]*r[0]+r[1]*r[1]+r[2]*r[2])

void init_neighbor_pair(NeighborPair** np, int nimgs, int* Ls_list)
{
    NeighborPair *np0 = (NeighborPair*) malloc(sizeof(NeighborPair));
    np0->nimgs = nimgs;
    np0->q_cond = NULL;
    np0->center = NULL;
    if (nimgs > 0){
        np0->Ls_list = (int*) malloc(sizeof(int)*nimgs);
        int i;
        for (i=0; i<nimgs; i++) {
            np0->Ls_list[i] = Ls_list[i];
        }
    }
    else {
        np0->Ls_list = NULL;
    }
    *np = np0;
}

void del_neighbor_pair(NeighborPair** np)
{
    NeighborPair *np0 = *np;
    if (!np0) {
        return;
    }
    if (np0->Ls_list) {
        free(np0->Ls_list);
    }
    if (np0->q_cond) {
        free(np0->q_cond);
    }
    if (np0->center) {
        free(np0->center);
    }
    free(np0);
    *np = NULL;
}

void init_neighbor_list(NeighborList** nl, int nish, int njsh, int nimgs)
{
    NeighborList *nl0 = (NeighborList*) malloc(sizeof(NeighborList)); 
    nl0->nish = nish;
    nl0->njsh = njsh;
    nl0->nimgs = nimgs;
    nl0->pairs = (NeighborPair**) malloc(sizeof(NeighborPair*)*nish*njsh);
    int ish, jsh;
    for (ish=0; ish<nish; ish++)
        for (jsh=0; jsh<njsh; jsh++) {
            (nl0->pairs)[ish*njsh+jsh] = NULL;
        }
    *nl = nl0;
}

void build_neighbor_list(NeighborList** nl,
                         int* ish_atm, int* ish_bas, double* ish_env, double* ish_rcut, 
                         int* jsh_atm, int* jsh_bas, double* jsh_env, double* jsh_rcut,
                         int nish, int njsh, double* Ls, int nimgs, int hermi)
{
    init_neighbor_list(nl, nish, njsh, nimgs);
    NeighborList* nl0 = *nl;

#pragma omp parallel
{
    int *buf = (int*) malloc(sizeof(int)*nimgs);
    int ish, jsh, iL, nL;
    int ish_atm_id, jsh_atm_id;
    double ish_radius, jsh_radius, rmax, dij;
    double *ish_ratm, *jsh_ratm, *rL;
    double rij[3];
    NeighborPair **np = NULL;
#pragma omp for schedule(dynamic)
    for (ish=0; ish<nish; ish++) {
        ish_radius = ish_rcut[ish];
        ish_atm_id = ish_bas[ish*BAS_SLOTS+ATOM_OF];
        ish_ratm = ish_env + ish_atm[ish_atm_id*ATM_SLOTS+PTR_COORD];
        for (jsh=0; jsh<njsh; jsh++) {
            if (hermi == 1 && jsh < ish) {
                continue;
            }
            jsh_radius = jsh_rcut[jsh];
            jsh_atm_id = jsh_bas[jsh*BAS_SLOTS+ATOM_OF];
            jsh_ratm = jsh_env + jsh_atm[jsh_atm_id*ATM_SLOTS+PTR_COORD];
            rmax = ish_radius + jsh_radius;
            nL = 0;
            for (iL=0; iL<nimgs; iL++) {
                rL = Ls + iL*3;
                rij[0] = jsh_ratm[0] + rL[0] - ish_ratm[0];
                rij[1] = jsh_ratm[1] + rL[1] - ish_ratm[1];
                rij[2] = jsh_ratm[2] + rL[2] - ish_ratm[2];
                dij = sqrt(SQUARE(rij));
                if (dij < rmax) {
                    buf[nL] = iL;
                    nL += 1;
                }
            }
            np = nl0->pairs + ish*njsh+jsh;
            init_neighbor_pair(np, nL, buf);
        }
    }
    free(buf);
}
}

void del_neighbor_list(NeighborList** nl)
{
    NeighborList *nl0 = *nl;
    if (!nl0) {
        return;
    }
    int ish, jsh;
    int nish = nl0->nish;
    int njsh = nl0->njsh;
    if (nl0->pairs) {
        for (ish=0; ish<nish; ish++) {
            for (jsh=0; jsh<njsh; jsh++) {
                del_neighbor_pair(nl0->pairs + ish*njsh+jsh);
            }
        }
        free(nl0->pairs);
    }
    free(nl0);
    *nl = NULL;
}


int NLOpt_noscreen(int* shls, NeighborListOpt* opt)
{
    return 1;
}

int NLOpt_screen(int* shls, NeighborListOpt* opt)
{
    int ish = shls[0];
    int jsh = shls[1];
    NeighborList *nl = opt->nl;
    int njsh = nl->njsh;
    NeighborPair *np;
    np = (nl->pairs)[ish*njsh + jsh];
    return np->nimgs > 0;
}

void NLOpt_init(NeighborListOpt **opt)
{
    NeighborListOpt *opt0 = malloc(sizeof(NeighborListOpt));
    opt0->nl = NULL;
    opt0->fprescreen = &NLOpt_noscreen;
    *opt = opt0;
}

void NLOpt_del(NeighborListOpt **opt)
{
    NeighborListOpt *opt0 = *opt;
    if (!opt0) {
        return;
    }
    free(opt0);
    *opt = NULL;
}

void NLOpt_set_nl(NeighborListOpt *opt, NeighborList *nl)
{
    opt->nl = nl;
}

void NLOpt_reset(NeighborListOpt *opt)
{
    opt->nl = NULL;
    opt->fprescreen = &NLOpt_screen;
}

void NLOpt_set_optimizer(NeighborListOpt *opt)
{
    opt->fprescreen = &NLOpt_screen;
}

void NLOpt_del_optimizer(NeighborListOpt *opt)
{
    opt->fprescreen = &NLOpt_noscreen;
}

