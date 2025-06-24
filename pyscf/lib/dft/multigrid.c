/* Copyright 2021-2025 The PySCF Developers. All Rights Reserved.

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
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include "config.h"
#include "cint.h"
#include "pbc/neighbor_list.h"
#include "pbc/cell.h"
#include "dft/multigrid.h"

#define SQUARE(r)       (r[0]*r[0]+r[1]*r[1]+r[2]*r[2])
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define BUF_SIZE 2000
#define ADD_SIZE 1000
#define RZERO 1e-6

const int _LEN_CART[] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136
};

const int _LEN_CART0[] = {
    0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120
};

const int _BINOMIAL_COEF[] = {
    1,
    1,   1,
    1,   2,   1,
    1,   3,   3,   1,
    1,   4,   6,   4,   1,
    1,   5,  10,  10,   5,   1,
    1,   6,  15,  20,  15,   6,   1,
    1,   7,  21,  35,  35,  21,   7,   1,
    1,   8,  28,  56,  70,  56,  28,   8,   1,
    1,   9,  36,  84, 126, 126,  84,  36,   9,   1,
    1,  10,  45, 120, 210, 252, 210, 120,  45,  10,   1,
    1,  11,  55, 165, 330, 462, 462, 330, 165,  55,  11,   1,
    1,  12,  66, 220, 495, 792, 924, 792, 495, 220,  66,  12,   1,
    1,  13,  78, 286, 715,1287,1716,1716,1287, 715, 286,  78,  13,   1,
    1,  14,  91, 364,1001,2002,3003,3432,3003,2002,1001, 364,  91,  14,   1,
    1,  15, 105, 455,1365,3003,5005,6435,6435,5005,3003,1365, 455, 105,  15,   1,
};

double CINTsquare_dist(const double *r1, const double *r2);

void init_gridlevel_info(GridLevel_Info** gridlevel_info,
                         double* cutoff, int* mesh, int nlevels, double rel_cutoff)
{
    GridLevel_Info* gl_info = (GridLevel_Info*) malloc(sizeof(GridLevel_Info));
    gl_info->nlevels = nlevels;
    gl_info->rel_cutoff = rel_cutoff;
    gl_info->cutoff = (double*) malloc(sizeof(double) * nlevels);
    gl_info->mesh = (int*) malloc(sizeof(int) * nlevels * 3);
    int i;
    for (i = 0; i < nlevels; i++) {
        (gl_info->cutoff)[i] = cutoff[i];
        (gl_info->mesh)[i*3] = mesh[i*3];
        (gl_info->mesh)[i*3+1] = mesh[i*3+1];
        (gl_info->mesh)[i*3+2] = mesh[i*3+2];
    }
    *gridlevel_info = gl_info;
}


void init_rs_grid(RS_Grid** rs_grid, GridLevel_Info** gridlevel_info, int comp)
{
    RS_Grid* rg = (RS_Grid*) malloc(sizeof(RS_Grid));
    GridLevel_Info* gl_info = *gridlevel_info;
    int nlevels = gl_info->nlevels;
    rg->nlevels = nlevels;
    rg->gridlevel_info = gl_info;
    rg->comp = comp;

    int i;
    size_t ngrid;
    int *mesh = gl_info->mesh;
    rg->data = (double**)malloc(sizeof(double*) * nlevels);
    for (i = 0; i < nlevels; i++) {
        ngrid = mesh[i*3] * mesh[i*3+1] * mesh[i*3+2];
        (rg->data)[i] = calloc(comp*ngrid, sizeof(double));
    }
    *rs_grid = rg;
}


void del_rs_grid(RS_Grid** rs_grid)
{
    RS_Grid* rg = *rs_grid;
    if (!rg) {
        return;
    }
    if (rg->data) {
        int i;
        for (i = 0; i < rg->nlevels; i++) {
            if (rg->data[i]) {
                free(rg->data[i]);
            }
        }
        free(rg->data);
    }
    rg->gridlevel_info = NULL;
    free(rg);
    *rs_grid = NULL;
}


void del_gridlevel_info(GridLevel_Info** gridlevel_info)
{
    GridLevel_Info* gl_info = *gridlevel_info;
    if (!gl_info) {
        return;
    }
    if (gl_info->cutoff) {
        free(gl_info->cutoff);
    }
    if (gl_info->mesh) {
        free(gl_info->mesh);
    }
    free(gl_info);
    *gridlevel_info = NULL;
}


void init_pgfpair(PGFPair** pair_info,
                  int ish, int ipgf, int jsh, int jpgf, int iL, double radius)
{
    PGFPair *pair0 = (PGFPair*) malloc(sizeof(PGFPair));
    pair0->ish = ish;
    pair0->ipgf = ipgf;
    pair0->jsh = jsh;
    pair0->jpgf = jpgf;
    pair0->iL = iL;
    pair0->radius = radius;
    *pair_info = pair0;
}


bool pgfpairs_with_same_shells(PGFPair *pair1, PGFPair *pair2)
{
    if (!pair1 || !pair2) {
        return false;
    }
    if (pair1->ish == pair2->ish && pair1->jsh == pair2->jsh) {
        return true;
    }
    return false;
}


double pgfpair_radius(int la, int lb, double zeta, double zetb, double* ra, double* rab, double precision)
{
    double radius = 0;
    double zetp = zeta + zetb;
    double eps = precision * precision;

    if (rab[0] < RZERO && rab[1] < RZERO && rab[2] < RZERO) {
        radius = pgf_rcut(la+lb, zetp, 1., eps, radius);
        return radius;
    }

    double prefactor = exp(-zeta*zetb/zetp*SQUARE(rab));
    double rb[3], rp[3];
    rb[0] = ra[0] + rab[0];
    rb[1] = ra[1] + rab[1];
    rb[2] = ra[2] + rab[2];
    rp[0] = ra[0] + zetb/zetp*rab[0];
    rp[1] = ra[1] + zetb/zetp*rab[1];
    rp[2] = ra[2] + zetb/zetp*rab[2];

    double rad_a = sqrt(CINTsquare_dist(ra, rp));
    double rad_b = sqrt(CINTsquare_dist(rb, rp));

    int lmax = la + lb;
    double coef[lmax+1];
    double rap[la+1];
    double rbp[lb+1];

    int lxa, lxb, i;
    for (i = 0; i <= lmax; i++) {
        coef[i] = 0;
    }
    rap[0] = 1.;
    for (i = 1; i <= la; i++) {
        rap[i] = rap[i-1] * rad_a;
    }
    rbp[0] = 1.;
    for (i = 1; i <= lb; i++) {
        rbp[i] = rbp[i-1] * rad_b;
    }

    for (lxa = 0; lxa <= la; lxa++) {
        for (lxb = 0; lxb <= lb; lxb++) {
            coef[lxa+lxb] += BINOMIAL(la, lxa) * BINOMIAL(lb, lxb) * rap[la-lxa] * rbp[lb-lxb];
        }
    }

    for (i = 0; i <= lmax; i++){
        coef[i] *= prefactor;
        radius = MAX(radius, pgf_rcut(i, zetp, coef[i], eps, radius));
    }
    return radius;
}


void del_pgfpair(PGFPair** pair_info)
{
    PGFPair *pair0 = *pair_info;
    if (!pair0) {
        return;
    } else {
        free(pair0);
    }
    *pair_info = NULL;
}


//unlink the pgfpair data instead of deleting
void nullify_pgfpair(PGFPair** pair_info)
{
    *pair_info = NULL;
}


void init_task(Task** task)
{
    Task *t0 = *task = (Task*) malloc(sizeof(Task));
    t0->ntasks = 0;
    t0->buf_size = BUF_SIZE; 
    t0->pgfpairs = (PGFPair**) malloc(sizeof(PGFPair*) * t0->buf_size);
    int i;
    for (i = 0; i < t0->buf_size; i++) {
        (t0->pgfpairs)[i] = NULL;
    }
}


void del_task(Task** task)
{
    Task *t0 = *task;
    if (!t0) {
        return;
    }
    if (t0->pgfpairs) {
        size_t i, ntasks = t0->ntasks;
        for (i = 0; i < ntasks; i++) {
            del_pgfpair(t0->pgfpairs + i);
        }
        free(t0->pgfpairs);
    }
    free(t0);
    *task = NULL;
}


void nullify_task(Task** task)
{
    Task *t0 = *task;
    if (!t0) {
        return;
    }
    if (t0->pgfpairs) {
        size_t i, ntasks = t0->ntasks;
        for (i = 0; i < ntasks; i++) {
            nullify_pgfpair(t0->pgfpairs + i);
        }
        free(t0->pgfpairs);
    }
    free(t0);
    *task = NULL;
}


void init_task_list(TaskList** task_list, GridLevel_Info* gridlevel_info, int nlevels, int hermi)
{
    TaskList* tl = *task_list = (TaskList*) malloc(sizeof(TaskList));
    tl->nlevels = nlevels;
    tl->hermi = hermi;
    tl->gridlevel_info = gridlevel_info;
    tl->tasks = (Task**) malloc(sizeof(Task*)*nlevels);
    int i;
    for (i = 0; i < nlevels; i++) {
        init_task(tl->tasks + i);
    }
}


void del_task_list(TaskList** task_list)
{
    TaskList *tl = *task_list;
    if (!tl) {
        return;
    }
    if (tl->gridlevel_info) {
        //del_gridlevel_info(&(tl->gridlevel_info));
        tl->gridlevel_info = NULL;
    }
    if (tl->tasks) {
        int i;
        for (i = 0; i < tl->nlevels; i++) {
            if ((tl->tasks)[i]) {
                del_task(tl->tasks + i);
            }
        }
        free(tl->tasks);
    }
    free(tl);
    *task_list = NULL;
}


void nullify_task_list(TaskList** task_list)
{
    TaskList *tl = *task_list;
    if (!tl) {
        return;
    }
    if (tl->gridlevel_info) {
        tl->gridlevel_info = NULL;
    }
    if (tl->tasks) {
        int i;
        for (i = 0; i < tl->nlevels; i++) {
            if ((tl->tasks)[i]) {
                nullify_task(tl->tasks + i);
            }
        }
        free(tl->tasks);
    }
    free(tl);
    *task_list = NULL;
}


void update_task_list(TaskList** task_list, int grid_level, 
                      int ish, int ipgf, int jsh, int jpgf, int iL, double radius)
{
    TaskList* tl = *task_list;
    Task *t0 = (tl->tasks)[grid_level];
    t0->ntasks += 1;
    if (t0->ntasks > t0->buf_size) {
        t0->buf_size += ADD_SIZE;
        t0->pgfpairs = (PGFPair**) realloc(t0->pgfpairs, sizeof(PGFPair*) * t0->buf_size);
    }
    init_pgfpair(t0->pgfpairs + t0->ntasks - 1,
                 ish, ipgf, jsh, jpgf, iL, radius);
}


void merge_task_list(TaskList** task_list, TaskList** task_list_loc)
{
    TaskList* tl = *task_list;
    TaskList* tl_loc = *task_list_loc;
    int ilevel, itask;
    for (ilevel = 0; ilevel < tl->nlevels; ilevel++) {
        Task *t0 = (tl->tasks)[ilevel];
        Task *t1 = (tl_loc->tasks)[ilevel];
        int itask_off = t0->ntasks;
        int ntasks_loc = t1->ntasks;
        t0->ntasks += ntasks_loc;
        t0->buf_size = t0->ntasks;
        t0->pgfpairs = (PGFPair**) realloc(t0->pgfpairs, sizeof(PGFPair*) * t0->buf_size);
        PGFPair** ptr_pgfpairs = t0->pgfpairs + itask_off;
        PGFPair** ptr_pgfpairs_loc = t1->pgfpairs;
        for (itask = 0; itask < ntasks_loc; itask++) {
            ptr_pgfpairs[itask] = ptr_pgfpairs_loc[itask];
        }
    }
}


int get_grid_level_cp2k(GridLevel_Info* gridlevel_info,
                        double alpha, double beta, double precision)
{
    int i;
    int nlevels = gridlevel_info->nlevels;
    int grid_level = nlevels - 1; //default use the most dense grid
    double needed_cutoff = (alpha + beta) * gridlevel_info->rel_cutoff;
    for (i = 0; i < nlevels; i++) {
        if ((gridlevel_info->cutoff)[i] >= needed_cutoff) {
            grid_level = i;
            break;
        }
    }
    return grid_level;
}


int get_grid_level(GridLevel_Info* gridlevel_info,
                   double alpha, double beta, double precision)
{
    int i;
    int nlevels = gridlevel_info->nlevels;
    int grid_level = nlevels - 1; //default use the most dense grid

    double ap = alpha + beta;
    double tmp = alpha * beta / (ap * ap);
    double fac = 2.8284271247461903 * pow(tmp, 0.75) / precision;
    double needed_cutoff = log(fac) * 2 * ap / pow(gridlevel_info->rel_cutoff, 0.25);
    for (i = 0; i < nlevels; i++) {
        if ((gridlevel_info->cutoff)[i] >= needed_cutoff) {
            grid_level = i;
            break;
        }
    }
    return grid_level;
}


void build_task_list(TaskList** task_list, NeighborList** neighbor_list,
                     GridLevel_Info** gridlevel_info,
                     int (*fn_get_grid_level)(),
                     int* ish_atm, int* ish_bas, double* ish_env, 
                     double* ish_rcut, double** ipgf_rcut,
                     int* jsh_atm, int* jsh_bas, double* jsh_env, 
                     double* jsh_rcut, double** jpgf_rcut,
                     int nish, int njsh, double* Ls, double precision, int hermi)
{
    GridLevel_Info *gl_info = *gridlevel_info;
    int ilevel;
    int nlevels = gl_info->nlevels;
    init_task_list(task_list, gl_info, nlevels, hermi);
    double* max_radius = calloc(nlevels, sizeof(double));
    NeighborList *nl0 = *neighbor_list;

#pragma omp parallel private(ilevel)
{
    double* max_radius_loc = calloc(nlevels, sizeof(double));
    TaskList** task_list_loc = (TaskList**) malloc(sizeof(TaskList*));
    init_task_list(task_list_loc, gl_info, nlevels, hermi);
    NeighborPair *np0_ij;
    int ish, jsh;
    int li, lj;
    int ipgf, jpgf;
    int nipgf, njpgf;
    int iL, iL_idx;
    int ish_atm_id, jsh_atm_id;
    int ish_alpha_of, jsh_alpha_of;
    double ipgf_alpha, jpgf_alpha;
    double *ish_ratm, *jsh_ratm, *rL;
    double rij[3];
    double dij, radius;

    #pragma omp for schedule(dynamic)
    for (ish = 0; ish < nish; ish++) {
        li = ish_bas[ANG_OF+ish*BAS_SLOTS];
        nipgf = ish_bas[NPRIM_OF+ish*BAS_SLOTS];
        ish_atm_id = ish_bas[ish*BAS_SLOTS+ATOM_OF];
        ish_ratm = ish_env + ish_atm[ish_atm_id*ATM_SLOTS+PTR_COORD];
        ish_alpha_of = ish_bas[PTR_EXP+ish*BAS_SLOTS];
        for (jsh = 0; jsh < njsh; jsh++) {
            if (hermi == 1 && jsh < ish) {
                continue;
            }
            np0_ij = (nl0->pairs)[ish*njsh + jsh];
            if (np0_ij->nimgs > 0) {
                lj = jsh_bas[ANG_OF+jsh*BAS_SLOTS];
                njpgf = jsh_bas[NPRIM_OF+jsh*BAS_SLOTS];
                jsh_atm_id = jsh_bas[jsh*BAS_SLOTS+ATOM_OF];
                jsh_ratm = jsh_env + jsh_atm[jsh_atm_id*ATM_SLOTS+PTR_COORD];
                jsh_alpha_of = jsh_bas[PTR_EXP+jsh*BAS_SLOTS];

                for (iL_idx = 0; iL_idx < np0_ij->nimgs; iL_idx++){
                    iL = (np0_ij->Ls_list)[iL_idx];
                    rL = Ls + iL*3;
                    rij[0] = jsh_ratm[0] + rL[0] - ish_ratm[0];
                    rij[1] = jsh_ratm[1] + rL[1] - ish_ratm[1];
                    rij[2] = jsh_ratm[2] + rL[2] - ish_ratm[2];
                    dij = sqrt(SQUARE(rij));

                    for (ipgf = 0; ipgf < nipgf; ipgf++) {
                        if (ipgf_rcut[ish][ipgf] + jsh_rcut[jsh] < dij) {
                            continue;
                        }
                        ipgf_alpha = ish_env[ish_alpha_of+ipgf];
                        for (jpgf = 0; jpgf < njpgf; jpgf++) {
                            //if (hermi == 1 && ish == jsh && jpgf < ipgf) {
                            //    continue;
                            //}
                            if (ipgf_rcut[ish][ipgf] + jpgf_rcut[jsh][jpgf] < dij) {
                                continue;
                            }
                            jpgf_alpha = jsh_env[jsh_alpha_of+jpgf]; 
                            ilevel = fn_get_grid_level(gl_info, ipgf_alpha, jpgf_alpha, precision);
                            radius = pgfpair_radius(li, lj, ipgf_alpha, jpgf_alpha, ish_ratm, rij, precision);
                            if (radius < RZERO) {
                                continue;
                            }
                            max_radius_loc[ilevel] = MAX(radius, max_radius_loc[ilevel]);
                            update_task_list(task_list_loc, ilevel, ish, ipgf, jsh, jpgf, iL, radius);
                        }
                    }
                }
            }
        }
    }

    #pragma omp critical
    {
        merge_task_list(task_list, task_list_loc);
    }

    nullify_task_list(task_list_loc);
    free(task_list_loc);

    for (ilevel = 0; ilevel < nlevels; ilevel++) {
        #pragma omp critical
        {
            max_radius[ilevel] = MAX(max_radius[ilevel], max_radius_loc[ilevel]);
        }
    }
    free(max_radius_loc);
}

    for (ilevel = 0; ilevel < nlevels; ilevel++) {
        Task *t0 = ((*task_list)->tasks)[ilevel];
        t0->radius = max_radius[ilevel];
    }
    free(max_radius);
}


int get_task_loc(int** task_loc, PGFPair** pgfpairs, int ntasks,
                 int ish0, int ish1, int jsh0, int jsh1, int hermi)
{
    int n = -2;
    int ish_prev = -1;
    int jsh_prev = -1;
    int itask, ish, jsh;
    int *buf = (int*)malloc(sizeof(int) * ntasks*2);
    PGFPair *pgfpair;
    for(itask = 0; itask < ntasks; itask++){
        pgfpair = pgfpairs[itask];
        ish = pgfpair->ish;
        jsh = pgfpair->jsh;
        if (ish < ish0 || ish >= ish1) {
            continue;
        }
        if (jsh < jsh0 || jsh >= jsh1) {
            continue;
        }
        if (hermi == 1 && jsh < ish) {
            continue;
        }

        if (ish != ish_prev || jsh != jsh_prev) {
            n += 2;
            buf[n] = itask;
            buf[n+1] = itask+1;
            ish_prev = ish;
            jsh_prev = jsh;
        } else {
            buf[n+1] = itask+1;
        }
    }
    n += 2;
    *task_loc = (int*)realloc(buf, sizeof(int) * n);
    return n;
}


void gradient_gs(double complex* out, double complex* f_gs, double* Gv,
                 int n, size_t ng)
{
    int i;
    double complex *outx, *outy, *outz;
    for (i = 0; i < n; i++) {
        outx = out;
        outy = outx + ng;
        outz = outy + ng;
        #pragma omp parallel
        {
            size_t igrid;
            double *pGv;
            #pragma omp for schedule(static)
            for (igrid = 0; igrid < ng; igrid++) {
                pGv = Gv + igrid * 3;
                outx[igrid] = pGv[0] * creal(f_gs[igrid]) * _Complex_I - pGv[0] * cimag(f_gs[igrid]);
                outy[igrid] = pGv[1] * creal(f_gs[igrid]) * _Complex_I - pGv[1] * cimag(f_gs[igrid]);
                outz[igrid] = pGv[2] * creal(f_gs[igrid]) * _Complex_I - pGv[2] * cimag(f_gs[igrid]);
            }
        }
        f_gs += ng;
        out += 3 * ng;
    }
}

/*
int get_task_loc_diff_ish(int** task_loc, PGFPair** pgfpairs, int ntasks,
                          int ish0, int ish1)
{
    int n = -2;
    int ish_prev = -1;
    int itask, ish;
    int *buf = (int*)malloc(sizeof(int) * ntasks*2);
    PGFPair *pgfpair;
    for(itask = 0; itask < ntasks; itask++){
        pgfpair = pgfpairs[itask];
        ish = pgfpair->ish;
        if (ish < ish0 || ish >= ish1) {
            continue;
        }

        if (ish != ish_prev) {
            n += 2;
            buf[n] = itask;
            ish_prev = ish;
        }
        if (ish == ish_prev) {
            buf[n+1] = itask+1;
        }
    }
    n += 2;
    *task_loc = (int*)realloc(buf, sizeof(int) * n);
    return n;
}
*/

/*
typedef struct Task_Index_struct {
    int ntasks;
    int bufsize;
    int* task_index;
} Task_Index;


void init_task_index(Task_Index* task_idx)
{
    task_idx->ntasks = 0;
    task_idx->bufsize = 10;
    task_idx->task_index = (int*)malloc(sizeof(int) * task_idx->bufsize);
}


void update_task_index(Task_Index* task_idx, int itask)
{
    task_idx->ntasks += 1;
    if (task_idx->bufsize < task_idx->ntasks) {
        task_idx->bufsize += 10;
        task_idx->task_index = (int*)realloc(task_idx->task_index, sizeof(int) * task_idx->bufsize);
    }
    task_idx->task_index[task_idx->ntasks-1] = itask;
}


void del_task_index(Task_Index* task_idx)
{
    if (!task_idx) {
        return;
    }
    if (task_idx->task_index) {
        free(task_idx->task_index);
    }
    task_idx->ntasks = 0;
    task_idx->bufsize = 0;
}


typedef struct Shlpair_Task_Index_struct {
    int nish;
    int njsh;
    int ish0;
    int jsh0;
    Task_Index *task_index;
} Shlpair_Task_Index;


void init_shlpair_task_index(Shlpair_Task_Index* shlpair_task_idx,
                             int ish0, int jsh0, int nish, int njsh)
{
    shlpair_task_idx->ish0 = ish0;
    shlpair_task_idx->jsh0 = jsh0;
    shlpair_task_idx->nish = nish;
    shlpair_task_idx->njsh = njsh;
    shlpair_task_idx->task_index = (Task_Index*)malloc(sizeof(Task_Index)*nish*njsh);

    int ijsh;
    for (ijsh = 0; ijsh < nish*njsh; ijsh++) {
        init_task_index(shlpair_task_idx->task_index + ijsh);
    }
}


void update_shlpair_task_index(Shlpair_Task_Index* shlpair_task_idx,
                               int ish, int jsh, int itask)
{
    int ish0 = shlpair_task_idx->ish0;
    int jsh0 = shlpair_task_idx->jsh0;
    int njsh = shlpair_task_idx->njsh;
    int ioff = ish - ish0;
    int joff = jsh - jsh0;

    update_task_index(shlpair_task_idx->task_index + ioff*njsh+joff, itask);
}


int get_task_index(Shlpair_Task_Index* shlpair_task_idx, int** idx, int ish, int jsh)
{
    int ish0 = shlpair_task_idx->ish0;
    int jsh0 = shlpair_task_idx->jsh0;
    int njsh = shlpair_task_idx->njsh;
    int ioff = ish - ish0;
    int joff = jsh - jsh0;
    Task_Index *task_idx = shlpair_task_idx->task_index + ioff*njsh+joff;
    int ntasks = task_idx->ntasks;
    *idx = task_idx->task_index;
    return ntasks;
}


void del_shlpair_task_index(Shlpair_Task_Index* shlpair_task_idx)
{
    if (!shlpair_task_idx) {
        return;
    }

    int nish = shlpair_task_idx->nish;
    int njsh = shlpair_task_idx->njsh;
    int ijsh;
    for (ijsh = 0; ijsh < nish*njsh; ijsh++) {
        del_task_index(shlpair_task_idx->task_index + ijsh);
    }
    free(shlpair_task_idx->task_index);
}


Shlpair_Task_Index* get_shlpair_task_index(PGFPair** pgfpairs, int ntasks,
            int ish0, int ish1, int jsh0, int jsh1, int hermi)
{
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;

    Shlpair_Task_Index* shlpair_task_idx = (Shlpair_Task_Index*) malloc(sizeof(Shlpair_Task_Index));
    init_shlpair_task_index(shlpair_task_idx, ish0, jsh0, nish, njsh);

    int itask;
    int ish, jsh;
    PGFPair *pgfpair = NULL;
    for(itask = 0; itask < ntasks; itask++){
        pgfpair = pgfpairs[itask];
        ish = pgfpair->ish;
        if (ish < ish0 || ish >= ish1) {
            continue;
        }
        jsh = pgfpair->jsh;
        if (jsh < jsh0 || jsh >= jsh1) {
            continue;
        }
        if (hermi == 1 && jsh < ish) {
            continue;
        }
        update_shlpair_task_index(shlpair_task_idx, ish, jsh, itask);
    }
    return shlpair_task_idx;
}
*/
