#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
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
    int i;
    for (i = 0; i < rg->nlevels; i++) {
        if (rg->data[i]) {
            free(rg->data[i]);
        }
    }
    free(rg->data);
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
        radius = pgf_rcut(la+lb, zetp, 1., eps, radius, RCUT_MAX_CYCLE);
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
        radius = MAX(radius, pgf_rcut(i, zetp, coef[i], eps, radius, RCUT_MAX_CYCLE));
    }
    return radius;
}


void del_pgfpair(PGFPair** pair_info)
{
    PGFPair *pair0 = *pair_info;
    if (!pair0) {
        return;
    }
    *pair_info = NULL;
}


void init_task(Task** task)
{
    Task *t0 = (Task*) malloc(sizeof(Task));
    t0->ntasks = 0;
    t0->buf_size = BUF_SIZE; 
    t0->pgfpairs = (PGFPair**) malloc(sizeof(PGFPair*) * t0->buf_size);
    int i;
    for (i = 0; i < t0->buf_size; i++) {
        (t0->pgfpairs)[i] = NULL;
    }
    *task = t0;
}


void del_task(Task** task)
{
    Task *t0 = *task;
    if (!t0) {
        return;
    }
    size_t i, buf_size = t0->buf_size;
    for (i = 0; i < buf_size; i++) {
        if ((t0->pgfpairs)[i]) {
            del_pgfpair(t0->pgfpairs + i);
        }
    }
    free(t0->pgfpairs); 
    *task = NULL;
}


void init_task_list(TaskList** task_list, GridLevel_Info* gridlevel_info, int nlevels)
{
    TaskList* tl = (TaskList*) malloc(sizeof(TaskList));
    tl->nlevels = nlevels;
    tl->gridlevel_info = gridlevel_info;
    tl->tasks = (Task**) malloc(sizeof(Task*)*nlevels);
    int i;
    for (i = 0; i < nlevels; i++) {
        init_task(tl->tasks + i);
    }
    *task_list = tl;
}


void del_task_list(TaskList** task_list)
{
    TaskList* tl = *task_list;
    if (!tl) {
        return;
    }
    tl->gridlevel_info = NULL;
    int i;
    for (i = 0; i < tl->nlevels; i++) {
        if ((tl->tasks)[i]) {
            del_task(tl->tasks + i);
        }
    }
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


int get_grid_level(GridLevel_Info* gridlevel_info, double alpha)
{
    int i;
    int nlevels = gridlevel_info->nlevels;
    int grid_level = nlevels - 1; //default use the most dense grid
    double needed_cutoff = alpha * gridlevel_info->rel_cutoff;
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
                     int* ish_atm, int* ish_bas, double* ish_env, 
                     double* ish_rcut, double** ipgf_rcut,
                     int* jsh_atm, int* jsh_bas, double* jsh_env, 
                     double* jsh_rcut, double** jpgf_rcut,
                     int nish, int njsh, double* Ls, double precision, int hermi)
{
    GridLevel_Info *gl_info = *gridlevel_info;
    int nlevels = gl_info->nlevels;
    init_task_list(task_list, gl_info, nlevels);
    double max_radius[nlevels];

//#pragma omp parallel
//{
    NeighborList *nl0 = *neighbor_list;
    NeighborPair *np0_ij;
    int ish, jsh;
    int li, lj;
    int ipgf, jpgf;
    int nipgf, njpgf;
    int iL, iL_idx;
    int grid_level;
    int ish_atm_id, jsh_atm_id;
    int ish_alpha_of, jsh_alpha_of;
    double ipgf_alpha, jpgf_alpha;
    double *ish_ratm, *jsh_ratm, *rL;
    double rij[3];
    double dij, radius;

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
                            grid_level = get_grid_level(gl_info, ipgf_alpha+jpgf_alpha);
                            radius = pgfpair_radius(li, lj, ipgf_alpha, jpgf_alpha, ish_ratm, rij, precision);
                            if (radius < RZERO) {
                                continue;
                            }
                            max_radius[grid_level] = MAX(radius, max_radius[grid_level]);
                            update_task_list(task_list, grid_level, ish, ipgf, jsh, jpgf, iL, radius);
                        }
                    }
                }
            }
        }
    }
//}

    for (int i = 0; i < nlevels; i++) {
        Task *t0 = ((*task_list)->tasks)[i];
        t0->radius = max_radius[i];
    }
}
