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


void c2s_outer_product(double* out,
                       double* ish_contr_coeff, double* jsh_contr_coeff,
                       int ni, int nj, int np, int nq)
{
    //einsum('pi,qj->ijpq', coeff_i, coeff_j)
    int n, i, j, p, q;
    double *ptr_coeff_i, *ptr_coeff_j;
    for (n = 0, i = 0; i < ni; i++) {
        ptr_coeff_i = ish_contr_coeff + i;
        for(j = 0; j < nj; j++) {
            ptr_coeff_j = jsh_contr_coeff + j;
            for (p = 0; p < np; p++) {
                for (q = 0; q < nq; q++, n++) {
                    out[n] = ptr_coeff_i[0] * ptr_coeff_j[0];
                    ptr_coeff_j += nj;
                }
                ptr_coeff_i += ni;
            }
        }
    }
}


void transform_dm_inverse_block(double* dm_cart, double* dm,
                                double* ish_contr_coeff, double* jsh_contr_coeff,
                                int ni, int nj, int np, int nq, double* cache)
{
    const char TRANS_T = 'T';
    const char TRANS_N = 'N';
    const double D1 = 1;
    const double D0 = 0;
    double *buf = cache;
    //einsum("pi,pq,qj->ij", coeff_i, dm_cart, coeff_j)
    dgemm_(&TRANS_N, &TRANS_N, &nj, &np, &nq,
           &D1, jsh_contr_coeff, &nj, dm_cart, &nq, &D0, buf, &nj);
    dgemm_(&TRANS_N, &TRANS_T, &nj, &ni, &np,
           &D1, buf, &nj, ish_contr_coeff, &ni, &D0, dm, &nj);
}


void transform_dm_ket(double* dm_cart_ket, double* dm,
                      double* contr_coeff, int* ao_loc, int* bas,
                      int ish, int jsh, int sh0, int naoj)
{
    //This function assumes that the density matrix is squre.
    int i0 = ao_loc[ish] - ao_loc[sh0];
    int j0 = ao_loc[jsh] - ao_loc[sh0];

    int nrow = ao_loc[ish+1] - ao_loc[ish];
    int ncol = ao_loc[jsh+1] - ao_loc[jsh];
    double* pdm = dm + i0*naoj + j0;

    int l_j = bas[ANG_OF+jsh*BAS_SLOTS];
    int ncart_j = _LEN_CART[l_j];
    int nprim_j = bas[NPRIM_OF+jsh*BAS_SLOTS];
    int nao_j = nprim_j*ncart_j;

    const char TRANS_T = 'T';
    const char TRANS_N = 'N';
    const double D1 = 1;
    const double D0 = 0;
    //einsum("ij,qj->iq", coeff_i, dm, coeff_j)
    dgemm_(&TRANS_T, &TRANS_N, &nao_j, &nrow, &ncol,
           &D1, jsh_contr_coeff, &ncol, pdm, &naoj, &D0, dm_cart_ket, &nao_j);
}


static void multiply_rhoG_coulG(complex double* rhoG, complex double* coulG, int* mesh)
{
    int x, y, z;
    int colx = mesh[1] * mesh[2];
    int coly = mesh[2];
    int nz = mesh[2] / 2 + 1;
    int colx1 = mesh[1] * nz;
    for (x = 0; x < mesh[0]; x++) {
        for (y = 0; y < mesh[1]; y++) {
            for (z = 0; z < nz; z++) {
                rhoG[x*colx1 + y*nz + z] *= coulG[x*colx + y*coly + z];
            }
        }
    }
}


static size_t _hfx_cache_size(int l, int nprim, int nctr, int* mesh)
{
    int l1 = 2 * l + 1;
    int l1l1 = l1 * l1;
    size_t size = 1000000;
    size_t mesh_size = ((size_t)mesh[0]) * mesh[1] * mesh[2];
    size += mesh_size;
    size += nctr*nctr*_LEN_CART[l]*_LEN_CART[l] * nprim*nprim*_LEN_CART[l]*_LEN_CART[l];
    size += nprim*nprim*_LEN_CART[l]*_LEN_CART[l];
    size += nctr*nctr*_LEN_CART[l]*_LEN_CART[l];
    size += l1l1 * mesh[2];
    size += l1 * mesh[1] * mesh[2];
    size += mesh[0] * mesh[1] * mesh[2] / MESH_BLK;
    printf("Memory allocated per thread for hfx: %ld MB.\n", (size)*sizeof(double) / 1000000);
    return size;
}


void grid_hfx_drv(double* dm, double* vout, TaskList** task_list, complex double* coulG,
                  int hermi, int grid_level, int *shls_slice, int* ish_ao_loc, int* jsh_ao_loc,
                  int dimension, double* Ls, double* a, double* b,
                  int* ish_atm, int* ish_bas, double* ish_env,
                  int* jsh_atm, int* jsh_bas, double* jsh_env, int cart)
{
    // dm is transposed
    // vout is transposed
    TaskList* tl = *task_list;
    GridLevel_Info* gridlevel_info = tl->gridlevel_info;
    //int nlevels = gridlevel_info->nlevels;

    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    //const int nijsh = nish * njsh;
    const int naoi = ish_ao_loc[ish1] - ish_ao_loc[ish0];
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

    int ilevel = grid_level;
    int *mesh;
    //double max_radius;
    Task* task;
    size_t ntasks;
    PGFPair** pgfpairs;
    //for (ilevel = 0; ilevel < nlevels; ilevel++) {
        task = (tl->tasks)[ilevel];
        ntasks = task->ntasks;
        if (ntasks <= 0) {
            //continue;
            return;
        }
        pgfpairs = task->pgfpairs;
        //max_radius = task->radius;

        mesh = gridlevel_info->mesh + ilevel*3;

        int *task_loc, *task_loc_ish;
        int nblock = get_task_loc(&task_loc, pgfpairs, ntasks, ish0, ish1, jsh0, jsh1, hermi);
        int nblock_ish = get_task_loc_diff_ish(&task_loc_ish, pgfpairs, ntasks, ish0, ish1);
        size_t cache_size = _hfx_cache_size(MAX(ish_lmax,jsh_lmax),
                                            MAX(ish_nprim_max, jsh_nprim_max),
                                            MAX(ish_nctr_max, jsh_nctr_max), mesh);
        size_t ngrids = ((size_t)mesh[0]) * mesh[1] * mesh[2];

        #pragma omp parallel
        {
            PGFPair *pgfpair, *pgfpair_ij, *pgfpair_kl;
            int iblock, itask, ish, jsh, ksh, lsh, k, l;
            int nk, nl, np, nq;
            int ni, nj, ni_cart, nj_cart;
            int mu0, nu0, sigma0, lambda0;
            int nu, sigma;
            const char TRANS_N = 'N';
            const char TRANS_T = 'T';
            const int I1 = 1;
            const double D1 = 1;
            double *ptr_gto_norm_i, *ptr_gto_norm_j;
            double *ptr_gto_norm_k, *ptr_gto_norm_l;
            double *cache0 = malloc(sizeof(double) * cache_size);
            double *dm_cart = cache0;
            int len_dm_cart = (ish_nctr_max*_LEN_CART[ish_lmax]*jsh_nctr_max*_LEN_CART[jsh_lmax] *
                               ish_nprim_max*_LEN_CART[ish_lmax]*jsh_nprim_max*_LEN_CART[jsh_lmax]);
            double *dm_pgf = dm_cart + len_dm_cart;
            double *mat_cart = dm_pgf + _LEN_CART[ish_lmax]*_LEN_CART[jsh_lmax];
            double *mat = mat_cart + ish_nprim_max*_LEN_CART[ish_lmax]*jsh_nprim_max*_LEN_CART[jsh_lmax];
            double *rho_priv = mat + ish_nctr_max*_LEN_CART[ish_lmax]*jsh_nctr_max*_LEN_CART[jsh_lmax];
            double *rhoG_priv = rho_priv + ngrids;
            double *cache = rhoG_priv + ((size_t)mesh[0])*mesh[1]*(mesh[2]/2+1)*2;
            double *pdm;

            FFT_PLAN p_r2c = fft_create_r2c_plan(rho_priv, (complex double*)rhoG_priv, 3, mesh);
            FFT_PLAN p_c2r = fft_create_c2r_plan((complex double*)rhoG_priv, rho_priv, 3, mesh);

            int kblock, ktask;
            #pragma omp for schedule(dynamic)
            for (kblock = 0; kblock < nblock_ish; kblock+=2) {
                // loop over \nu, \sigma
                // dm_{\lambda,\sigma}
                ktask = task_loc_ish[kblock];
                pgfpair_kl = pgfpairs[ktask];
                ksh = pgfpair_kl->ish;
                lsh = pgfpair_kl->jsh;

                if (lsh < jsh0 || lsh >= jsh1) {
                    continue;
                }
                //if (hermi == 1 && lsh < ksh) {
                //    continue;
                //}

                ptr_gto_norm_k = gto_norm_i[ksh];
                ptr_gto_norm_l = gto_norm_j[lsh];

                nk = ish_ao_loc[ksh+1] - ish_ao_loc[ksh];
                nl = jsh_ao_loc[lsh+1] - jsh_ao_loc[lsh];
                np = ish_bas[NPRIM_OF+ksh*BAS_SLOTS] * _LEN_CART[ish_bas[ANG_OF+ksh*BAS_SLOTS]];
                nq = jsh_bas[NPRIM_OF+lsh*BAS_SLOTS] * _LEN_CART[jsh_bas[ANG_OF+lsh*BAS_SLOTS]];

                c2s_outer_product(dm_cart, 
                                  cart2sph_coeff_i[ksh], cart2sph_coeff_j[lsh],
                                  nk, nl, np, nq);

                nu0 = ish_ao_loc[ksh] - ish_ao_loc[ish0];
                sigma0 = jsh_ao_loc[lsh] - jsh_ao_loc[jsh0];
                for (k = 0; k < nk; k++) {
                    nu = nu0 + k;
                for (l = 0; l < nl; l++) {
                    sigma = sigma0 + l;
                    pdm = dm_cart + (l+k*nl) * np*nq;
                    memset(rho_priv, 0, ngrids*sizeof(double));
                    for (; ktask < task_loc[kblock+1]; ktask++) {
                        pgfpair = pgfpairs[ktask];
                        get_dm_pgfpair(dm_pgf, pdm, pgfpair, ish_bas, jsh_bas, hermi);
                        _apply_rho(collocate_rho_lda_orth, rho_priv, dm_pgf, pgfpair, 1, dimension, a, b, mesh,
                                   ptr_gto_norm_k, ptr_gto_norm_l, ish_atm, ish_bas, ish_env,
                                   jsh_atm, jsh_bas, jsh_env, Ls, cache);
                    }
                    fft_execute(p_r2c);
                    multiply_rhoG_coulG((complex double*)rhoG_priv, coulG, mesh);
                    fft_execute(p_c2r);

                for (iblock = 0; iblock < nblock; iblock+=2) {
                    // loop over \mu, \lambda
                    itask = task_loc[iblock];
                    pgfpair_ij = pgfpairs[itask];
                    ish = pgfpair_ij->ish;
                    jsh = pgfpair_ij->jsh;
                    ptr_gto_norm_i = gto_norm_i[ish];
                    ptr_gto_norm_j = gto_norm_j[jsh];

                    ni = ish_ao_loc[ish+1] - ish_ao_loc[ish];
                    nj = jsh_ao_loc[jsh+1] - jsh_ao_loc[jsh];
                    ni_cart = ish_bas[NPRIM_OF+ish*BAS_SLOTS]*_LEN_CART[ish_bas[ANG_OF+ish*BAS_SLOTS]];
                    nj_cart = jsh_bas[NPRIM_OF+jsh*BAS_SLOTS]*_LEN_CART[jsh_bas[ANG_OF+jsh*BAS_SLOTS]];
                    memset(mat_cart, 0, ni_cart * nj_cart * sizeof(double));
                    for (; itask < task_loc[iblock+1]; itask++) {
                        pgfpair = pgfpairs[itask];
                        _apply_ints(eval_mat_lda_orth, rho_priv, mat_cart, pgfpair, 1, 1.0, dimension, a, b, mesh,
                                    ptr_gto_norm_i, ptr_gto_norm_j, ish_atm, ish_bas, ish_env,
                                    jsh_atm, jsh_bas, jsh_env, Ls, cache);
                    }
                    transform_dm_inverse_block(mat_cart, mat,
                                               cart2sph_coeff_i[ish], cart2sph_coeff_j[jsh],
                                               ni, nj, ni_cart, nj_cart, cache);

                    mu0 = ish_ao_loc[ish] - ish_ao_loc[ish0];
                    lambda0 = jsh_ao_loc[jsh] - jsh_ao_loc[jsh0];
                    dgemm_(&TRANS_T, &TRANS_N, &ni, &I1, &nj,
                           &D1, mat, &nj, dm+lambda0+sigma*naoj, &nj,
                           &D1, vout+mu0+nu*naoi, &ni); 
                }

                }}
            }
            fft_destroy_plan(p_r2c);
            fft_destroy_plan(p_c2r);
            free(cache0);
        }
    //}
}
