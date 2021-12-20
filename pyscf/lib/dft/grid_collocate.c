//#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include "config.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"
#include "dft/multigrid.h"
#include "dft/grid_common.h"

#define MAX_THREADS     256
#define PTR_RADIUS        5

static void transform_dm(double* dm_cart, double* dm,
                         double* ish_contr_coeff, double* jsh_contr_coeff,
                         int* ish_ao_loc, int* jsh_ao_loc,
                         int* ish_bas, int* jsh_bas, int ish, int jsh,
                         int ish0, int jsh0, int naoj, double* cache)
{
    int i0 = ish_ao_loc[ish] - ish_ao_loc[ish0];
    int i1 = ish_ao_loc[ish+1] - ish_ao_loc[ish0];
    int j0 = jsh_ao_loc[jsh] - jsh_ao_loc[jsh0];
    int j1 = jsh_ao_loc[jsh+1] - jsh_ao_loc[jsh0];

    int nrow = i1 - i0;
    int ncol = j1 - j0;
    double* pdm = dm + ((size_t)naoj) * i0 + j0;

    int l_i = ish_bas[ANG_OF+ish*BAS_SLOTS];
    int ncart_i = _LEN_CART[l_i];
    int nprim_i = ish_bas[NPRIM_OF+ish*BAS_SLOTS];
    int nao_i = nprim_i*ncart_i;
    int l_j = jsh_bas[ANG_OF+jsh*BAS_SLOTS];
    int ncart_j = _LEN_CART[l_j];
    int nprim_j = jsh_bas[NPRIM_OF+jsh*BAS_SLOTS];
    int nao_j = nprim_j*ncart_j;

    const char TRANS_T = 'T';
    const char TRANS_N = 'N';
    const double D1 = 1;
    const double D0 = 0;
    //einsum("pi,ij,qj->pq", coeff_i, dm, coeff_j)
    dgemm_(&TRANS_T, &TRANS_N, &nao_j, &nrow, &ncol,
           &D1, jsh_contr_coeff, &ncol, pdm, &naoj, &D0, cache, &nao_j);
    dgemm_(&TRANS_N, &TRANS_N, &nao_j, &nao_i, &nrow,
           &D1, cache, &nao_j, ish_contr_coeff, &nrow, &D0, dm_cart, &nao_j);
}


static void _orth_rho(double *rho, double *dm_xyz,
                      double fac, int topl,
                      int *mesh, int *img_slice, int *grid_slice,
                      double *xs_exp, double *ys_exp, double *zs_exp,
                      double *cache)
{
        int l1 = topl + 1;
        int l1l1 = l1 * l1;
        int nimgx0 = img_slice[0];
        int nimgx1 = img_slice[1];
        int nimgy0 = img_slice[2];
        int nimgy1 = img_slice[3];
        int nimgz0 = img_slice[4];
        int nimgz1 = img_slice[5];
        int nimgx = nimgx1 - nimgx0;
        int nimgy = nimgy1 - nimgy0;
        int nimgz = nimgz1 - nimgz0;
        int nx0 = grid_slice[0];
        int nx1 = MIN(grid_slice[1], mesh[0]);
        int ny0 = grid_slice[2];
        int ny1 = MIN(grid_slice[3], mesh[1]);
        int nz0 = grid_slice[4];
        int nz1 = MIN(grid_slice[5], mesh[2]);
        int ngridx = _num_grids_on_x(nimgx, nx0, nx1, mesh[0]);
        int ngridy = _num_grids_on_x(nimgy, ny0, ny1, mesh[1]);
        int ngridz = _num_grids_on_x(nimgz, nz0, nz1, mesh[2]);
        if (ngridx == 0 || ngridy == 0 || ngridz == 0) {
                return;
        }

        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        size_t mesh_yz = ((size_t)mesh[1]) * mesh[2];
        int xcols = ngridy * ngridz;
        double *xyr = cache;
        double *xqr = xyr + l1l1 * ngridz;
        double *pqr = xqr + l1 * xcols;
        int l, ix, iy, iz, nx, ny, nz;
        int xmap[ngridx];
        int ymap[ngridy];
        int zmap[ngridz];

        if (nimgz == 1) {
                dgemm_(&TRANS_N, &TRANS_N, &ngridz, &l1l1, &l1,
                       &fac, zs_exp+nz0, mesh+2, dm_xyz, &l1,
                       &D0, xyr, &ngridz);
                for (iz = 0; iz < ngridz; iz++) {
                    zmap[iz] = iz + nz0;
                }
        } else if (nimgz == 2 && !_has_overlap(nz0, nz1, mesh[2])) {
                nz = mesh[2]-nz0;
                dgemm_(&TRANS_N, &TRANS_N, &nz, &l1l1, &l1,
                       &fac, zs_exp+nz0, mesh+2, dm_xyz, &l1,
                       &D0, xyr+nz1, &ngridz);
                dgemm_(&TRANS_N, &TRANS_N, &nz1, &l1l1, &l1,
                       &fac, zs_exp, mesh+2, dm_xyz, &l1,
                       &D0, xyr, &ngridz);
                for (iz = 0; iz < nz1; iz++) {
                    zmap[iz] = iz;
                }
                nz = nz0 - nz1;
                for (iz = nz1; iz < ngridz; iz++) {
                    zmap[iz] = iz + nz;
                }
        }
        else{
                dgemm_(&TRANS_N, &TRANS_N, mesh+2, &l1l1, &l1,
                       &fac, zs_exp, mesh+2, dm_xyz, &l1,
                       &D0, xyr, mesh+2);
                for (iz = 0; iz < mesh[2]; iz++) {
                    zmap[iz] = iz;
                }
        }

        if (nimgy == 1) {
                for (l = 0; l <= topl; l++) {
                        dgemm_(&TRANS_N, &TRANS_T, &ngridz, &ngridy, &l1,
                               &D1, xyr+l*l1*ngridz, &ngridz, ys_exp+ny0, mesh+1,
                               &D0, xqr+l*xcols, &ngridz);
                }
                for (iy = 0; iy < ngridy; iy++) {
                    ymap[iy] = iy + ny0;
                }
        } else if (nimgy == 2 && !_has_overlap(ny0, ny1, mesh[1])) {
                ny = mesh[1] - ny0;
                for (l = 0; l <= topl; l++) {
                        dgemm_(&TRANS_N, &TRANS_T, &ngridz, &ny1, &l1,
                               &D1, xyr+l*l1*ngridz, &ngridz, ys_exp, mesh+1,
                               &D0, xqr+l*xcols, &ngridz);
                        dgemm_(&TRANS_N, &TRANS_T, &ngridz, &ny, &l1,
                               &D1, xyr+l*l1*ngridz, &ngridz, ys_exp+ny0, mesh+1,
                               &D0, xqr+l*xcols+ny1*ngridz, &ngridz);
                }
                for (iy = 0; iy < ny1; iy++) {
                    ymap[iy] = iy;
                }
                ny = ny0 - ny1;
                for (iy = ny1; iy < ngridy; iy++) {
                    ymap[iy] = iy + ny;
                }
        } else {
                for (l = 0; l <= topl; l++) {
                        dgemm_(&TRANS_N, &TRANS_T, &ngridz, mesh+1, &l1,
                               &D1, xyr+l*l1*ngridz, &ngridz, ys_exp, mesh+1,
                               &D0, xqr+l*xcols, &ngridz);
                }
                for (iy = 0; iy < mesh[1]; iy++) {
                    ymap[iy] = iy;
                }
        }

        if (nimgx == 1) {
                dgemm_(&TRANS_N, &TRANS_T, &xcols, &ngridx, &l1,
                       &D1, xqr, &xcols, xs_exp+nx0, mesh,
                       &D0, pqr, &xcols);
                for (ix = 0; ix < ngridx; ix++) {
                    xmap[ix] = ix + nx0;
                }
        } else if (nimgx == 2 && !_has_overlap(nx0, nx1, mesh[0])) {
                dgemm_(&TRANS_N, &TRANS_T, &xcols, &nx1, &l1,
                       &D1, xqr, &xcols, xs_exp, mesh,
                       &D0, pqr, &xcols);
                nx = mesh[0] - nx0;
                dgemm_(&TRANS_N, &TRANS_T, &xcols, &nx, &l1,
                       &D1, xqr, &xcols, xs_exp+nx0, mesh,
                       &D0, pqr+nx1*xcols, &xcols);
                for (ix = 0; ix < nx1; ix++) {
                    xmap[ix] = ix;
                }
                nx = nx0 - nx1;
                for (ix = nx1; ix < ngridx; ix++) {
                    xmap[ix] = ix + nx;
                }
        } else {
                dgemm_(&TRANS_N, &TRANS_T, &xcols, mesh, &l1,
                       &D1, xqr, &xcols, xs_exp, mesh,
                       &D0, pqr, &xcols);
                for (ix = 0; ix < mesh[0]; ix++) {
                    xmap[ix] = ix;
                }
        }

        // TODO optimize the following loops using e.g. axpy
        for (ix = 0; ix < ngridx; ix++) {
            for (iy = 0; iy < ngridy; iy++) {
                for (iz = 0; iz < ngridz; iz++) {
                    rho[xmap[ix]*mesh_yz+ymap[iy]*mesh[2]+zmap[iz]] += pqr[ix*xcols+iy*ngridz+iz];
                }
            }
        }
}


void make_rho_lda_orth(double *rho, double *dm, int comp,
                       int li, int lj, double ai, double aj,
                       double *ri, double *rj, double fac, double cutoff,
                       int dimension, double *a, double *b,
                       int *mesh, double *cache)
{
        int topl = li + lj;
        int l1 = topl + 1;
        int l1l1l1 = l1 * l1 * l1;
        int img_slice[6];
        int grid_slice[6];
        double *xs_exp, *ys_exp, *zs_exp;
        int data_size = _init_orth_data(&xs_exp, &ys_exp, &zs_exp, img_slice,
                                        grid_slice, mesh,
                                        topl, dimension, cutoff,
                                        ai, aj, ri, rj, a, b, cache);
        if (data_size == 0) {
                return;
        }
        cache += data_size;

        double *dm_xyz = cache;
        cache += l1l1l1;
        memset(dm_xyz, 0, l1l1l1*sizeof(double));

        _dm_to_dm_xyz(dm_xyz, dm, li, lj, ri, rj, cache);

        _orth_rho(rho, dm_xyz, fac, topl, mesh,
                  img_slice, grid_slice, xs_exp, ys_exp, zs_exp, cache);
}


static void _apply_rho(void (*eval_rho)(), double *rho, double *dm,
                       PGFPair* pgfpair,
                       int comp,
                       int dimension, double *a, double *b,
                       int *mesh,
                       double* ish_gto_norm, double* jsh_gto_norm,
                       int *ish_atm, int *ish_bas, double *ish_env,
                       int *jsh_atm, int *jsh_bas, double *jsh_env,
                       double* Ls,
                       double *cache)
{
        int ish = pgfpair->ish;
        int jsh = pgfpair->jsh;
        int ipgf = pgfpair->ipgf;
        int jpgf = pgfpair->jpgf;
        int iL = pgfpair->iL;
        double cutoff = pgfpair->radius;

        double *ri = ish_env + ish_atm[PTR_COORD+ish_bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS];
        double *rj = jsh_env + jsh_atm[PTR_COORD+jsh_bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS];
        double *rL = Ls + iL*3;
        double rjL[3];
        rjL[0] = rj[0] + rL[0];
        rjL[1] = rj[1] + rL[1];
        rjL[2] = rj[2] + rL[2];

        const int li = ish_bas[ANG_OF+ish*BAS_SLOTS];
        const int lj = jsh_bas[ANG_OF+jsh*BAS_SLOTS];
        double ai = ish_env[ish_bas[PTR_EXP+ish*BAS_SLOTS]+ipgf];
        double aj = jsh_env[jsh_bas[PTR_EXP+jsh*BAS_SLOTS]+jpgf];
        double ci = ish_gto_norm[ipgf];
        double cj = jsh_gto_norm[jpgf];
        double aij = ai + aj;
        double rrij = CINTsquare_dist(ri, rjL);
        double eij = (ai * aj / aij) * rrij;
        if (eij > EIJCUTOFF) {
                return;
        }
        double fac = exp(-eij) * ci * cj * CINTcommon_fac_sp(li) * CINTcommon_fac_sp(lj);
        if (fac < ish_env[PTR_EXPDROP] && fac < jsh_env[PTR_EXPDROP]) {
                return;
        }

        (*eval_rho)(rho, dm, comp, li, lj, ai, aj, ri, rjL,
                    fac, cutoff, dimension, a, b, mesh, cache);
}


static size_t _rho_cache_size(int l, int nprim, int nctr, int* mesh, double radius, double* a)
{
    size_t size = 0;
    size_t mesh_size = ((size_t)mesh[0]) * mesh[1] * mesh[2];
    int l1 = 2 * l + 1;
    int l1l1 = l1 * l1;
    int nimgs = (int) ceil(MAX(MAX(radius/fabs(a[0]), radius/a[4]), radius/a[8])) + 1;
    int nmx = MAX(MAX(mesh[0], mesh[1]), mesh[2]) * nimgs;
    size += (nprim * _LEN_CART[l]) * (nprim * _LEN_CART[l]);
    size += _LEN_CART[l]*_LEN_CART[l];
    size += nctr * _LEN_CART[l] * nprim * _LEN_CART[l];

    size += l1 * (mesh[0] + mesh[1] + mesh[2]);
    size += l1 * nmx + nmx;
    size += l1l1 * l1;
    size += 3 * (_LEN_CART[l] + l1);
    size += l1l1 * mesh[2];
    size += l1 * mesh[1] * mesh[2];
    size += mesh_size; // usually don't need so much
    size += 1000000;
    //printf("Memory allocated per thread for make_rho: %ld MB.\n", (size+mesh_size)*sizeof(double) / 1000000);
    return size;
}


static size_t _rho_core_cache_size(int* mesh, double radius, double* a)
{
    size_t size = 0;
    size_t mesh_size = ((size_t)mesh[0]) * mesh[1] * mesh[2];
    int l1 = 1;
    int l1l1 = l1 * l1;
    int nimgs = (int) ceil(MAX(MAX(radius/fabs(a[0]), radius/a[4]), radius/a[8])) + 1;
    int nmx = MAX(MAX(mesh[0], mesh[1]), mesh[2]) * nimgs;
    size += l1 * (mesh[0] + mesh[1] + mesh[2]);
    size += l1 * nmx + nmx;
    size += l1l1 * l1;
    size += l1l1 * mesh[2];
    size += l1 * mesh[1] * mesh[2];
    size += mesh_size; // usually don't need so much
    size += 1000000;
    return size;
}


void grid_collocate_drv(void (*eval_rho)(), RS_Grid** rs_rho, double* dm, TaskList** task_list,
                        int comp, int hermi, int *shls_slice, int* ish_ao_loc, int* jsh_ao_loc,
                        int dimension, double* Ls, double* a, double* b,
                        int* ish_atm, int* ish_bas, double* ish_env,
                        int* jsh_atm, int* jsh_bas, double* jsh_env, int cart)
{
    TaskList* tl = *task_list;
    GridLevel_Info* gridlevel_info = tl->gridlevel_info;
    int nlevels = gridlevel_info->nlevels;

    assert (comp == (*rs_rho)->comp);

    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    //const int nijsh = nish * njsh;
    //const int naoi = ish_ao_loc[ish1] - ish_ao_loc[ish0];
    const int naoj = jsh_ao_loc[jsh1] - jsh_ao_loc[jsh0];

    double **gto_norm_i = (double**) malloc(sizeof(double*) * nish);
    double **cart2sph_coeff_i = (double**) malloc(sizeof(double*) * nish);
    get_cart2sph_coeff(cart2sph_coeff_i, gto_norm_i, ish0, ish1, ish_bas, ish_env, cart);
    double **gto_norm_j = gto_norm_i;
    double **cart2sph_coeff_j = cart2sph_coeff_i;
    if (hermi != 1) {
        gto_norm_j = (double**) malloc(sizeof(double*) * njsh);
        cart2sph_coeff_j = (double**) malloc(sizeof(double*) * njsh);
        get_cart2sph_coeff(cart2sph_coeff_j, gto_norm_j, jsh0, jsh1, jsh_bas, jsh_env, cart);
    }

    int ish_lmax = get_lmax(ish0, ish1, ish_bas);
    int jsh_lmax = ish_lmax;
    if (hermi != 1) {
        jsh_lmax = get_lmax(jsh0, jsh1, jsh_bas);
    }

    int ish_nprim_max = get_nprim_max(ish0, ish1, ish_bas);
    int jsh_nprim_max = ish_nprim_max;
    if (hermi != 1) {
        jsh_nprim_max = get_nprim_max(jsh0, jsh1, jsh_bas);
    }

    int ish_nctr_max = get_nctr_max(ish0, ish1, ish_bas);
    int jsh_nctr_max = ish_nctr_max;
    if (hermi != 1) {
        jsh_nctr_max = get_nctr_max(jsh0, jsh1, jsh_bas);
    } 

    int ilevel;
    int *mesh;
    double max_radius;
    double *rho, *rhobufs[MAX_THREADS];
    Task* task;
    size_t ntasks;
    PGFPair** pgfpairs;
    for (ilevel = 0; ilevel < nlevels; ilevel++) {
        task = (tl->tasks)[ilevel];
        ntasks = task->ntasks;
        if (ntasks <= 0) {
            continue;
        }
        pgfpairs = task->pgfpairs;
        max_radius = task->radius;

        rho = (*rs_rho)->data[ilevel];
        mesh = gridlevel_info->mesh + ilevel*3;

        int *task_loc;
        int nblock = get_task_loc(&task_loc, pgfpairs, ntasks, ish0, ish1, jsh0, jsh1, hermi);

        size_t cache_size = _rho_cache_size(MAX(ish_lmax,jsh_lmax), 
                                            MAX(ish_nprim_max, jsh_nprim_max),
                                            MAX(ish_nctr_max, jsh_nctr_max), mesh, max_radius, a);
        size_t ngrids = ((size_t)mesh[0]) * mesh[1] * mesh[2];

#pragma omp parallel
{
    PGFPair *pgfpair = NULL;
    int iblock, itask, ish, jsh;
    double *ptr_gto_norm_i, *ptr_gto_norm_j;
    double *cache0 = malloc(sizeof(double) * cache_size);
    double *dm_cart = cache0;
    double *dm_pgf = cache0 + ish_nprim_max*_LEN_CART[ish_lmax]*jsh_nprim_max*_LEN_CART[jsh_lmax];
    double *cache = dm_pgf + _LEN_CART[ish_lmax]*_LEN_CART[jsh_lmax]; 

    int thread_id = omp_get_thread_num();
    double *rho_priv;
    if (thread_id == 0) {
        rho_priv = rho;
    } else {
        rho_priv = calloc(comp*ngrids, sizeof(double));
    }
    rhobufs[thread_id] = rho_priv;

    #pragma omp for schedule(dynamic)
    for (iblock = 0; iblock < nblock; iblock+=2) {
        itask = task_loc[iblock];
        pgfpair = pgfpairs[itask];
        ish = pgfpair->ish;
        jsh = pgfpair->jsh;
        ptr_gto_norm_i = gto_norm_i[ish];
        ptr_gto_norm_j = gto_norm_j[jsh];
        transform_dm(dm_cart, dm, cart2sph_coeff_i[ish],
                     cart2sph_coeff_j[jsh], ish_ao_loc, jsh_ao_loc,
                     ish_bas, jsh_bas, ish, jsh, ish0, jsh0, naoj, cache);
        for (; itask < task_loc[iblock+1]; itask++) {
            pgfpair = pgfpairs[itask];
            get_dm_pgfpair(dm_pgf, dm_cart, pgfpair, ish_bas, jsh_bas, hermi);
            _apply_rho(eval_rho, rho_priv, dm_pgf, pgfpair, comp, dimension, a, b, mesh,
                       ptr_gto_norm_i, ptr_gto_norm_j, ish_atm, ish_bas, ish_env,
                       jsh_atm, jsh_bas, jsh_env, Ls, cache);
        }
    }

    free(cache0);
    NPomp_dsum_reduce_inplace(rhobufs, comp*ngrids);
    if (thread_id != 0) {
        free(rho_priv);
    }
}
    free(task_loc);
    } // loop ilevel

    del_cart2sph_coeff(cart2sph_coeff_i, gto_norm_i, ish0, ish1);
    if (hermi != 1) {
        del_cart2sph_coeff(cart2sph_coeff_j, gto_norm_j, jsh0, jsh1);
    }
}


void build_core_density(void (*eval_rho)(), double* rho,
                        int* atm, int* bas, int nbas, double* env,
                        int* mesh, int dimension, double* a, double* b, double max_radius)
{
    size_t ngrids;
    ngrids = ((size_t) mesh[0]) * mesh[1] * mesh[2];
    double *rhobufs[MAX_THREADS];
    size_t cache_size =  _rho_core_cache_size(mesh, max_radius, a);

#pragma omp parallel
{
    int ia, ib;
    double alpha, coeff, charge, rad, fac;
    double dm[] = {1.0};
    double *r0;
    double *cache = (double*) malloc(sizeof(double) * cache_size);

    int thread_id = omp_get_thread_num();
    double *rho_priv;
    if (thread_id == 0) {
        rho_priv = rho;
    } else {
        rho_priv = calloc(ngrids, sizeof(double));
    }
    rhobufs[thread_id] = rho_priv;

    #pragma omp for schedule(static)
    for (ib = 0; ib < nbas; ib++) {
        ia = bas[ib*BAS_SLOTS+ATOM_OF];
        alpha = env[bas[ib*BAS_SLOTS+PTR_EXP]];
        coeff = env[bas[ib*BAS_SLOTS+PTR_COEFF]];
        charge = (double)atm[ia*ATM_SLOTS+CHARGE_OF];
        r0 = env + atm[ia*ATM_SLOTS+PTR_COORD];
        fac = -charge * coeff;
        rad = env[atm[ia*ATM_SLOTS+PTR_RADIUS]];
        eval_rho(rho_priv, dm, 1, 0, 0, alpha, 0., r0, r0,
                 fac, rad, dimension, a, b, mesh, cache);
    }
    free(cache);

    NPomp_dsum_reduce_inplace(rhobufs, ngrids);
    if (thread_id != 0) {
        free(rho_priv);
    }
}
}
