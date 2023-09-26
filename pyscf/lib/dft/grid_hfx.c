#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include "config.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"
#include "dft/multigrid.h"
#include "pbc/fft.h"
#include "dft/grid_common.h"
#include "dft/utils.h"

void _apply_ints(int (*eval_ints)(), double *weights, double *mat,
                        PGFPair* pgfpair, int comp, double fac, int dimension,
                        double* dh, double *a, double *b, int *mesh,
                        double* ish_gto_norm, double* jsh_gto_norm,
                        int *ish_atm, int *ish_bas, double *ish_env,
                        int *jsh_atm, int *jsh_bas, double *jsh_env,
                        double* Ls, double *cache);

//contract cart2sph_coeff with one mo_coeff
void get_cart2mo_coeff(double** out, double** cart2sph_coeff, double* mo_coeff,
                       int ish0, int ish1, int* bas, double* env, int* ao_loc, int cart)
{
    const char TRANS_T = 'T';
    const char TRANS_N = 'N';
    const int One = 1;
    const double D1 = 1;
    const double D0 = 0;

    #pragma omp parallel
    {
        int ish, i0;
        int l, ncart, nsph, nprim, nctr;
        int nrow, ncol;

        #pragma omp for schedule(dynamic)
        for (ish = ish0; ish < ish1; ish++) {
            l = bas[ANG_OF+ish*BAS_SLOTS];
            ncart = _LEN_CART[l];
            nsph = cart == 1 ? ncart : 2*l+1;
            nprim = bas[NPRIM_OF+ish*BAS_SLOTS];
            nctr = bas[NCTR_OF+ish*BAS_SLOTS];
            nrow = nprim * ncart;
            ncol = nctr * nsph;
            i0 = ao_loc[ish] - ao_loc[ish0];

            out[ish] = (double*) malloc(sizeof(double) * nrow);
            dgemm_wrapper(TRANS_T, TRANS_N, nrow, One, ncol,
                          D1, cart2sph_coeff[ish], ncol,
                          mo_coeff+i0, ncol, D0, out[ish], nrow);

        }
    }
}


static void transform_dm_inverse(double* dm_cart, double* dm, int comp,
                          double* ish_cart2mo_coeff, double* jsh_cart2sph_coeff,
                          int* ish_ao_loc, int* jsh_ao_loc,
                          int* ish_bas, int* jsh_bas, int ish, int jsh,
                          int ish0, int jsh0, int naoi, int naoj, double* cache)
{
    int j0 = jsh_ao_loc[jsh] - jsh_ao_loc[jsh0];
    int j1 = jsh_ao_loc[jsh+1] - jsh_ao_loc[jsh0];

    int ncol = j1 - j0;
    double* pdm = dm + j0;

    int l_i = ish_bas[ANG_OF+ish*BAS_SLOTS];
    int ncart_i = _LEN_CART[l_i];
    int nprim_i = ish_bas[NPRIM_OF+ish*BAS_SLOTS];
    int nao_i = nprim_i*ncart_i;
    int l_j = jsh_bas[ANG_OF+jsh*BAS_SLOTS];
    int ncart_j = _LEN_CART[l_j];
    int nprim_j = jsh_bas[NPRIM_OF+jsh*BAS_SLOTS];
    int nao_j = nprim_j*ncart_j;

    //const char TRANS_T = 'T';
    const char TRANS_N = 'N';
    const int One = 1;
    const double D1 = 1;
    const double D0 = 0;
    double *buf = cache;

    //einsum("pi,pq,qj->ij", coeff_i, dm_cart, coeff_j)
    dgemm_(&TRANS_N, &TRANS_N, &ncol, &nao_i, &nao_j,
           &D1, jsh_cart2sph_coeff, &ncol, dm_cart, &nao_j, &D0, buf, &ncol);
    dgemm_(&TRANS_N, &TRANS_N, &ncol, &One, &nao_i,
           &D1, buf, &ncol, ish_cart2mo_coeff, &nao_i, &D0, pdm, &ncol);
}


static size_t _ints_cache_size(int l, int nprim, int nctr, int* mesh, double radius, double* dh, int comp)
{
    size_t size = 0;
    size_t nmx = get_max_num_grid_orth(dh, radius);
    int max_mesh = MAX(MAX(mesh[0], mesh[1]), mesh[2]);
    int l1 = 2 * l + 1;
    if (comp == 3) {
        l1 += 1;
    }
    int l1l1 = l1 * l1;
    int ncart = _LEN_CART[l1]; // use l1 to be safe

    size += comp * nprim * nprim * ncart * ncart; // dm_cart
    size += comp * ncart * ncart; // out
    size += l1 * (mesh[0] + mesh[1] + mesh[2]); // xs_exp, ys_exp, zs_exp

    size_t size_orth_components = l1 * nmx + nmx; // orth_components
    size += l1l1 * l1; // dm_xyz
    size += 3 * (ncart + l1); // _dm_xyz_to_dm

    size_t size_orth_ints = 0;
    if (nmx < max_mesh) {
        size_orth_ints = (l1 + l1l1) * nmx;
    } else {
        size_orth_ints = l1*mesh[2] + l1l1*mesh[0];
    }
    size += MAX(size_orth_components, size_orth_ints);
    size += nctr * ncart * nprim * ncart;
    //size += 1000000;
    //printf("Memory allocated per thread for make_mat: %ld MB.\n", size*sizeof(double) / 1000000);
    return size;
}


void grid_hfx_integrate(int (*eval_ints)(), double* mat, double* mo_coeff,
                        double* weights, TaskList** task_list,
                        int comp, int hermi, int grid_level, 
                        int *shls_slice, int* ish_ao_loc, int* jsh_ao_loc,
                        int dimension, double* Ls, double* a, double* b,
                        int* ish_atm, int* ish_bas, double* ish_env,
                        int* jsh_atm, int* jsh_bas, double* jsh_env, int cart)
{
    TaskList* tl = *task_list;
    GridLevel_Info* gridlevel_info = tl->gridlevel_info;
    Task *task = (tl->tasks)[grid_level];
    int ntasks = task->ntasks;
    if (ntasks <= 0) {
        return;
    }
    double max_radius = task->radius;
    PGFPair **pgfpairs = task->pgfpairs;
    int* mesh = gridlevel_info->mesh + grid_level*3;

    double dh[9];
    get_grid_spacing(dh, a, mesh);

    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    const int naoi = ish_ao_loc[ish1] - ish_ao_loc[ish0];
    const int naoj = jsh_ao_loc[jsh1] - jsh_ao_loc[jsh0];

    bool nosym = (ish0 != jsh0 || ish1 != jsh1);

    int ish_lmax = get_lmax(ish0, ish1, ish_bas);
    int jsh_lmax = ish_lmax;
    if (nosym) {
        jsh_lmax = get_lmax(jsh0, jsh1, jsh_bas);
    }

    int ish_nprim_max = get_nprim_max(ish0, ish1, ish_bas);
    int jsh_nprim_max = ish_nprim_max;
    if (nosym) {
        jsh_nprim_max = get_nprim_max(jsh0, jsh1, jsh_bas);
    }

    int ish_nctr_max = get_nctr_max(ish0, ish1, ish_bas);
    int jsh_nctr_max = ish_nctr_max;
    if (nosym) {
        jsh_nctr_max = get_nctr_max(jsh0, jsh1, jsh_bas);
    }

    double **gto_norm_i = (double**) malloc(sizeof(double*) * nish);
    double **cart2sph_coeff_i = (double**) malloc(sizeof(double*) * nish);
    get_cart2sph_coeff(cart2sph_coeff_i, gto_norm_i, ish0, ish1, ish_bas, ish_env, cart);

    double **cart2mo_coeff_i = (double**) malloc(sizeof(double*) * nish);
    get_cart2mo_coeff(cart2mo_coeff_i, cart2sph_coeff_i, mo_coeff,
                      ish0, ish1, ish_bas, ish_env, ish_ao_loc, cart);

    double **gto_norm_j = gto_norm_i;
    double **cart2sph_coeff_j = cart2sph_coeff_i;
    if (nosym) {
        gto_norm_j = (double**) malloc(sizeof(double*) * njsh);
        cart2sph_coeff_j = (double**) malloc(sizeof(double*) * njsh);
        get_cart2sph_coeff(cart2sph_coeff_j, gto_norm_j, jsh0, jsh1, jsh_bas, jsh_env, cart);
    }

    int *task_loc;
    int nblock = get_task_loc(&task_loc, pgfpairs, ntasks, ish0, ish1, jsh0, jsh1, hermi);

    size_t cache_size = _ints_cache_size(MAX(ish_lmax,jsh_lmax),
                                         MAX(ish_nprim_max, jsh_nprim_max),
                                         MAX(ish_nctr_max, jsh_nctr_max), 
                                         mesh, max_radius, dh, comp);

#pragma omp parallel
{
    int ish, jsh, itask, iblock;
    int li, lj, ish_nprim, jsh_nprim;
    PGFPair *pgfpair = NULL;
    double *ptr_gto_norm_i, *ptr_gto_norm_j;
    double *cache0 = malloc(sizeof(double) * cache_size);
    double *dm_cart = cache0;
    int len_dm_cart = comp*ish_nprim_max*_LEN_CART[ish_lmax]*jsh_nprim_max*_LEN_CART[jsh_lmax];
    double *cache = dm_cart + len_dm_cart;

    #pragma omp for schedule(dynamic)
    for (iblock = 0; iblock < nblock; iblock+=2) {
        itask = task_loc[iblock];
        pgfpair = pgfpairs[itask];
        ish = pgfpair->ish;
        jsh = pgfpair->jsh;
        ptr_gto_norm_i = gto_norm_i[ish];
        ptr_gto_norm_j = gto_norm_j[jsh];
        li = ish_bas[ANG_OF+ish*BAS_SLOTS];
        lj = jsh_bas[ANG_OF+jsh*BAS_SLOTS];
        ish_nprim = ish_bas[NPRIM_OF+ish*BAS_SLOTS];
        jsh_nprim = jsh_bas[NPRIM_OF+jsh*BAS_SLOTS];
        len_dm_cart = comp*ish_nprim*_LEN_CART[li]*jsh_nprim*_LEN_CART[lj];
        memset(dm_cart, 0, len_dm_cart * sizeof(double));
        for (; itask < task_loc[iblock+1]; itask++) {
            pgfpair = pgfpairs[itask];
            _apply_ints(eval_ints, weights, dm_cart, pgfpair, comp, 1.0, dimension, dh, a, b, mesh,
                        ptr_gto_norm_i, ptr_gto_norm_j, ish_atm, ish_bas, ish_env,
                        jsh_atm, jsh_bas, jsh_env, Ls, cache);
        }
        transform_dm_inverse(dm_cart, mat, comp,
                             cart2mo_coeff_i[ish], cart2sph_coeff_j[jsh],
                             ish_ao_loc, jsh_ao_loc, ish_bas, jsh_bas,
                             ish, jsh, ish0, jsh0, naoi, naoj, cache);
    }
    free(cache0);
}

    if (task_loc) {
        free(task_loc);
    }
    del_cart2sph_coeff(cart2sph_coeff_i, gto_norm_i, ish0, ish1);
    if (nosym) {
        del_cart2sph_coeff(cart2sph_coeff_j, gto_norm_j, jsh0, jsh1);
    }

    int ish;
    for (ish = ish0; ish < ish1; ish++) {
        if (cart2mo_coeff_i[ish]) {
            free(cart2mo_coeff_i[ish]);
        }
    }
    free(cart2mo_coeff_i);
}
