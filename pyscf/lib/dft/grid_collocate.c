#include <stdio.h>
//#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "config.h"
#include "cint.h"
#include "vhf/fblas.h"
#include "np_helper/np_helper.h"
#include "dft/multigrid.h"


#define EXPMIN         -700
#define EIJCUTOFF        60
#define PTR_EXPDROP      16
#define MAX_THREADS     256

double CINTsquare_dist(const double *r1, const double *r2);
double CINTcommon_fac_sp(int l);
int _has_overlap(int nx0, int nx1, int nx_per_cell);
int _num_grids_on_x(int nimgx, int nx0, int nx1, int nx_per_cell);
//void GTOreverse_vrr2d_ket(double *g00, double *g01, int li, int lj, double *ri, double *rj);
//void _cart_to_xyz(double *dm_xyz, double *dm_cart, int floorl, int topl, int l1);


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


int get_lmax(int ish0, int ish1, int* bas)
{
    int lmax = 0;
    int ish;
    for (ish = ish0; ish < ish1; ish++) {
        lmax = MAX(lmax, bas[ANG_OF+ish*BAS_SLOTS]);
    }
    return lmax;
}


int get_nprim_max(int ish0, int ish1, int* bas)
{
    int nprim_max = 1;
    int ish;
    for (ish = ish0; ish < ish1; ish++) {
        nprim_max = MAX(nprim_max, bas[NPRIM_OF+ish*BAS_SLOTS]); 
    }
    return nprim_max;
}


int get_nctr_max(int ish0, int ish1, int* bas)
{
    int nctr_max = 1;
    int ish;
    for (ish = ish0; ish < ish1; ish++) {
        nctr_max = MAX(nctr_max, bas[NCTR_OF+ish*BAS_SLOTS]);
    }
    return nctr_max;
}


void get_cart2sph_coeff(double** contr_coeff, double** gto_norm, 
                        int ish0, int ish1,
                        int* bas, double* env, int cart)
{
    int l;
    int lmax = get_lmax(ish0, ish1, bas);
    int nprim, ncart, nsph, nctr;
    int ptr_exp, ptr_coeff;
    int ish, ipgf, ic, i, j;

    double **c2s = (double**) malloc(sizeof(double*) * (lmax+1));
    for (l = 0; l <= lmax; l++) {
        ncart = _LEN_CART[l];
        if (l <= 1 || cart == 1) {
            c2s[l] = (double*) calloc(ncart*ncart, sizeof(double));
            for (i = 0; i < ncart; i++) {
                c2s[l][i*ncart + i] = 1;
            }
        }
        else {
            nsph = 2*l + 1;
            c2s[l] = (double*) calloc(nsph*ncart, sizeof(double));
            double* gcart = (double*) calloc(ncart*ncart, sizeof(double));
            for (i = 0; i < ncart; i++) {
                gcart[i*ncart + i] = 1;
            }
            CINTc2s_ket_sph(c2s[l], ncart, gcart, l);
            free(gcart);
        }
    }

#pragma omp parallel private (ish, ipgf, ic, i, j, l,\
                              ncart, nsph, nprim, nctr,\
                              ptr_exp, ptr_coeff)
{
    #pragma omp for schedule(dynamic) 
    for (ish = ish0; ish < ish1; ish++) {
        l = bas[ANG_OF+ish*BAS_SLOTS];
        ncart = _LEN_CART[l];
        nsph = cart == 1 ? ncart : 2*l+1;
        nprim = bas[NPRIM_OF+ish*BAS_SLOTS];
        nctr = bas[NCTR_OF+ish*BAS_SLOTS];

        ptr_exp = bas[PTR_EXP+ish*BAS_SLOTS];
        gto_norm[ish] = (double*) malloc(sizeof(double) * nprim);
        for (ipgf = 0; ipgf < nprim; ipgf++) {
            gto_norm[ish][ipgf] = CINTgto_norm(l, env[ptr_exp+ipgf]);
        }

        ptr_coeff = bas[PTR_COEFF+ish*BAS_SLOTS];
        double *buf = (double*) calloc(nctr*nprim, sizeof(double));
        for (ipgf = 0; ipgf < nprim; ipgf++) {
            double inv_norm = 1./gto_norm[ish][ipgf];
            daxpy_(&nctr, &inv_norm, env+ptr_coeff+ipgf, &nprim, buf+ipgf, &nprim);
        }

        contr_coeff[ish] = (double*) malloc(sizeof(double) * nprim*ncart*nctr*nsph);
        double* ptr_contr_coeff = contr_coeff[ish];
        for (ipgf = 0; ipgf < nprim; ipgf++) {
            for (i = 0; i < ncart; i++) {
                for (ic = 0; ic < nctr; ic++) {
                    for (j = 0; j < nsph; j++) {
                        *ptr_contr_coeff = buf[ic*nprim+ipgf] * c2s[l][j*ncart+i];
                        ptr_contr_coeff += 1;
                    }
                }
            }
        }
        free(buf);
    }
}

    for (l = 0; l <= lmax; l++) {
        free(c2s[l]);
    }
    free(c2s);
}


void del_cart2sph_coeff(double** contr_coeff, double** gto_norm, int ish0, int ish1)
{
    int ish;
    for (ish = ish0; ish < ish1; ish++) {
        if (contr_coeff[ish]) {
            free(contr_coeff[ish]);
        }
        if (gto_norm[ish]) {
            free(gto_norm[ish]);
        }
    }
    free(contr_coeff);
    free(gto_norm);
}


void transform_dm_inverse(double* dm_cart, double* dm, int comp,
                          double* ish_contr_coeff, double* jsh_contr_coeff,
                          int* ish_ao_loc, int* jsh_ao_loc,
                          int* ish_bas, int* jsh_bas, int ish, int jsh,
                          int ish0, int jsh0, int naoi, int naoj, double* cache)
{
    int i0 = ish_ao_loc[ish] - ish_ao_loc[ish0];
    int i1 = ish_ao_loc[ish+1] - ish_ao_loc[ish0];
    int j0 = jsh_ao_loc[jsh] - jsh_ao_loc[jsh0];
    int j1 = jsh_ao_loc[jsh+1] - jsh_ao_loc[jsh0];

    int nrow = i1 - i0;
    int ncol = j1 - j0;
    double* pdm = dm + i0*naoj + j0;

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
    //double* buf = (double*) malloc(sizeof(double) * ncol*nao_i);
    double *buf = cache;

    int ic;
    for (ic = 0; ic < comp; ic++) {
        //einsum("pi,pq,qj->ij", coeff_i, dm_cart, coeff_j)
        dgemm_(&TRANS_N, &TRANS_N, &ncol, &nao_i, &nao_j,
               &D1, jsh_contr_coeff, &ncol, dm_cart, &nao_j, &D0, buf, &ncol);
        dgemm_(&TRANS_N, &TRANS_T, &ncol, &nrow, &nao_i,
               &D1, buf, &ncol, ish_contr_coeff, &nrow, &D0, pdm, &naoj);
        pdm += naoi * naoj;
        dm_cart += nao_i * nao_j;
    }
    //free(buf);
}


void transform_dm(double* dm_cart, double* dm,
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
    double* pdm = dm + i0*naoj + j0;

    int l_i = ish_bas[ANG_OF+ish*BAS_SLOTS];
    int ncart_i = (l_i+1)*(l_i+2)/2;
    int nprim_i = ish_bas[NPRIM_OF+ish*BAS_SLOTS];
    int nao_i = nprim_i*ncart_i;
    int l_j = jsh_bas[ANG_OF+jsh*BAS_SLOTS];
    int ncart_j = (l_j+1)*(l_j+2)/2;
    int nprim_j = jsh_bas[NPRIM_OF+jsh*BAS_SLOTS];
    int nao_j = nprim_j*ncart_j;

    const char TRANS_T = 'T';
    const char TRANS_N = 'N';
    const double D1 = 1;
    double beta = 0;
    //double* buf = (double*) malloc(sizeof(double) * nrow*nao_j);
    double *buf = cache;
    //einsum("pi,ij,qj->pq", coeff_i, dm, coeff_j)
    dgemm_(&TRANS_T, &TRANS_N, &nao_j, &nrow, &ncol,
           &D1, jsh_contr_coeff, &ncol, pdm, &naoj, &beta, buf, &nao_j);
    dgemm_(&TRANS_N, &TRANS_N, &nao_j, &nao_i, &nrow,
           &D1, buf, &nao_j, ish_contr_coeff, &nrow, &beta, dm_cart, &nao_j);
    //free(buf);
}


void get_dm_pgf(double* dm_pgf, double* dm_cart, 
                PGFPair* pgfpair,
                int* ish_bas, int* jsh_bas, int hermi)
{
    int ish = pgfpair->ish;
    int jsh = pgfpair->jsh;
    int ipgf = pgfpair->ipgf;
    int jpgf = pgfpair->jpgf;

    int li = ish_bas[ANG_OF+ish*BAS_SLOTS];
    int lj = jsh_bas[ANG_OF+jsh*BAS_SLOTS];
    int di = _LEN_CART[li];
    int dj = _LEN_CART[lj];

    int nprim_j = jsh_bas[NPRIM_OF+jsh*BAS_SLOTS];
    int ncol = nprim_j * dj;
    double *pdm = dm_cart + (ipgf*di*ncol + jpgf*dj);
    int i, j;
    for (i = 0; i < di; i++) {
        for (j = 0; j < dj; j++) {
            dm_pgf[i*dj+j] = pdm[j];
        }
        pdm += ncol;
    }

    /*
    if (hermi == 1 && ish == jsh) {
        assert(di == dj);
        for (i = 0; i < di; i++) {
            for (j = i+1; j < dj; j++) {
                dm_pgf[i*dj+j] *= 2;
                dm_pgf[j*dj+i] = 0;
            }
        }
    }*/
    if (hermi == 1 && ish != jsh) {
        for (i = 0; i < di; i++) {
            for (j = 0; j < dj; j++) {
                dm_pgf[i*dj+j] *= 2;
            }
        }
    }
}


static int _orth_components(double *xs_exp, int *img_slice, int *grid_slice,
                            double a, double b, double cutoff,
                            double xi, double xj, double ai, double aj,
                            int periodic, int nx_per_cell, int topl,
                            double *cache)
{
    double aij = ai + aj;
    double xij = (ai * xi + aj * xj) / aij;
    double heights_inv = b;
    double xij_frac = xij * heights_inv;
    double edge0 = xij_frac - cutoff * heights_inv;
    double edge1 = xij_frac + cutoff * heights_inv;

    if (edge0 == edge1) {
        return 0;
    }

    int nimg0 = 0;
    int nimg1 = 1;
    if (periodic) {
        nimg0 = (int) floor(edge0);
        nimg1 = (int) ceil(edge1);
    }

    int nimg = nimg1 - nimg0;

    int nmx0 = nimg0 * nx_per_cell;
    int nmx1 = nimg1 * nx_per_cell;
    int nmx = nmx1 - nmx0;

    int nx0 = (int) floor(edge0 * nx_per_cell);
    int nx1 = (int) ceil(edge1 * nx_per_cell);
   
    int nx0_edge = nx0 - nmx0;
    int nx1_edge = nx1 - nmx0;

    // to ensure nx0, nx1 being inside the unit cell
    if (periodic) {
        nx0 = nx0_edge % nx_per_cell;
        nx1 = nx1_edge % nx_per_cell;
        if (nx1 == 0) {
            nx1 = nx_per_cell;
        }
    }
    assert(nx0 == nx0_edge);

    img_slice[0] = nimg0;
    img_slice[1] = nimg1;
    grid_slice[0] = nx0;
    grid_slice[1] = nx1;

    int ngridx = _num_grids_on_x(nimg, nx0, nx1, nx_per_cell);
    if (ngridx == 0) {
        return 0;
    }

    int i, m, l;
    double *px0;

    double *gridx = cache;
    double *xs_all = cache + nmx;
    if (nimg == 1) {
        xs_all = xs_exp;
    }

    int grid_close_to_xij = rint(xij_frac * nx_per_cell) - nmx0;
    grid_close_to_xij = MIN(grid_close_to_xij, nx1_edge);
    grid_close_to_xij = MAX(grid_close_to_xij, nx0_edge);

    double img0_x = a * nimg0;
    double dx = a / nx_per_cell;
    double base_x = img0_x + dx * grid_close_to_xij;
    double x0xij = base_x - xij;
    double _x0x0 = -aij * x0xij * x0xij;
    if (_x0x0 < EXPMIN) {
        return 0;
    }

    double _dxdx = -aij * dx * dx;
    double _x0dx = -2 * aij * x0xij * dx;
    double exp_dxdx = exp(_dxdx);
    double exp_2dxdx = exp_dxdx * exp_dxdx;
    double exp_x0dx = exp(_x0dx + _dxdx);
    double exp_x0x0 = exp(_x0x0);

    for (i = grid_close_to_xij; i < nx1_edge; i++) {
        xs_all[i] = exp_x0x0;
        exp_x0x0 *= exp_x0dx;
        exp_x0dx *= exp_2dxdx;
    }

    exp_x0dx = exp(_dxdx - _x0dx);
    exp_x0x0 = exp(_x0x0);
    for (i = grid_close_to_xij-1; i >= nx0_edge; i--) {
        exp_x0x0 *= exp_x0dx;
        exp_x0dx *= exp_2dxdx;
        xs_all[i] = exp_x0x0;
    }

    if (topl > 0) {
        double x0xi = img0_x - xi;
        for (i = nx0_edge; i < nx1_edge; i++) {
            gridx[i] = x0xi + i * dx;
        }
        for (l = 1; l <= topl; l++) {
            px0 = xs_all + (l-1) * nmx;
            for (i = nx0_edge; i < nx1_edge; i++) {
                px0[nmx+i] = px0[i] * gridx[i];
            }
        }
    }

    int idx1;
    if (nimg > 1) {
        for (l = 0; l <= topl; l++) {
            px0 = xs_all + l * nmx;
            for (i = nx0; i < nx_per_cell; i++) {
                xs_exp[l*nx_per_cell+i] = px0[i];
            }
            memset(xs_exp+l*nx_per_cell, 0, nx0*sizeof(double));
            for (m = 1; m < nimg; m++) {
                px0 = xs_all + l * nmx + m*nx_per_cell;
                idx1 = (m == nimg - 1) ? nx1 : nx_per_cell;
                for (i = 0; i < idx1; i++) {
                    xs_exp[l*nx_per_cell+i] += px0[i];
                }
            }
        }
    }
    return ngridx;
}


static int _init_orth_data(double **xs_exp, double **ys_exp, double **zs_exp,
                           int *img_slice, int *grid_slice,
                           int *mesh,
                           int topl, int dimension, double cutoff,
                           double ai, double aj, double *ri, double *rj,
                           double *a, double *b, double *cache)
{
        int l1 = topl + 1;
        *xs_exp = cache;
        *ys_exp = *xs_exp + l1 * mesh[0];
        *zs_exp = *ys_exp + l1 * mesh[1];
        int data_size = l1 * (mesh[0] + mesh[1] + mesh[2]);
        cache += data_size;

        int ngridx = _orth_components(*xs_exp, img_slice, grid_slice,
                                      a[0], b[0], cutoff, ri[0], rj[0], ai, aj,
                                      (dimension>=1), mesh[0], topl, cache);
        if (ngridx == 0) {
                return 0;
        }

        int ngridy = _orth_components(*ys_exp, img_slice+2, grid_slice+2,
                                      a[4], b[4], cutoff, ri[1], rj[1], ai, aj,
                                      (dimension>=2), mesh[1], topl, cache);
        if (ngridy == 0) {
                return 0;
        }

        int ngridz = _orth_components(*zs_exp, img_slice+4, grid_slice+4,
                                      a[8], b[8], cutoff, ri[2], rj[2], ai, aj,
                                      (dimension>=3), mesh[2], topl, cache);
        if (ngridz == 0) {
                return 0;
        }

        return data_size;
}


static void _orth_ints(double *out, double *weights,
                       int topl, double fac,
                       double *xs_exp, double *ys_exp, double *zs_exp,
                       int *img_slice, int *grid_slice,
                       int *mesh, double *cache)
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
        int nx1 = grid_slice[1];
        int ny0 = grid_slice[2];
        int ny1 = grid_slice[3];
        int nz0 = grid_slice[4];
        int nz1 = grid_slice[5];
        int ngridx = _num_grids_on_x(nimgx, nx0, nx1, mesh[0]);
        int ngridy = _num_grids_on_x(nimgy, ny0, ny1, mesh[1]);
        int ngridz = _num_grids_on_x(nimgz, nz0, nz1, mesh[2]);
        int lx;

        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        int xcols = mesh[1] * mesh[2];
        int ycols = mesh[2];
        double *weightyz = cache;
        double *weightz = weightyz + l1*xcols;

        if (nimgx == 1) {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &ngridx,
                       &fac, weights+nx0*xcols, &xcols, xs_exp+nx0, mesh,
                       &D0, weightyz, &xcols);
        } else if (nimgx == 2 && !_has_overlap(nx0, nx1, mesh[0])) {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &nx1,
                       &fac, weights, &xcols, xs_exp, mesh,
                       &D0, weightyz, &xcols);
                ngridx = mesh[0] - nx0;
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &ngridx,
                       &fac, weights+nx0*xcols, &xcols, xs_exp+nx0, mesh,
                       &D1, weightyz, &xcols);
        } else {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, mesh,
                       &fac, weights, &xcols, xs_exp, mesh,
                       &D0, weightyz, &xcols);
        }

        if (nimgy == 1) {
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ngridy,
                               &D1, weightyz+lx*xcols+ny0*ycols, &ycols, ys_exp+ny0, mesh+1,
                               &D0, weightz+lx*l1*ycols, &ycols);
                }
        } else if (nimgy == 2 && !_has_overlap(ny0, ny1, mesh[1])) {
                ngridy = mesh[1] - ny0;
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ny1,
                               &D1, weightyz+lx*xcols, &ycols, ys_exp, mesh+1,
                               &D0, weightz+lx*l1*ycols, &ycols);
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ngridy,
                               &D1, weightyz+lx*xcols+ny0*ycols, &ycols, ys_exp+ny0, mesh+1,
                               &D1, weightz+lx*l1*ycols, &ycols);
                }
        } else {
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, mesh+1,
                               &D1, weightyz+lx*xcols, &ycols, ys_exp, mesh+1,
                               &D0, weightz+lx*l1*ycols, &ycols);
                }
        }

        if (nimgz == 1) {
                dgemm_(&TRANS_T, &TRANS_N, &l1, &l1l1, &ngridz,
                       &D1, zs_exp+nz0, mesh+2, weightz+nz0, mesh+2,
                       &D0, out, &l1);
        } else if (nimgz == 2 && !_has_overlap(nz0, nz1, mesh[2])) {
                dgemm_(&TRANS_T, &TRANS_N, &l1, &l1l1, &nz1,
                       &D1, zs_exp, mesh+2, weightz, mesh+2,
                       &D0, out, &l1);
                ngridz = mesh[2] - nz0;;
                dgemm_(&TRANS_T, &TRANS_N, &l1, &l1l1, &ngridz,
                       &D1, zs_exp+nz0, mesh+2, weightz+nz0, mesh+2,
                       &D1, out, &l1);
        } else {
                dgemm_(&TRANS_T, &TRANS_N, &l1, &l1l1, mesh+2,
                       &D1, zs_exp, mesh+2, weightz, mesh+2,
                       &D0, out, &l1);
        }
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
        int xcols = mesh[1] * mesh[2];
        double *xyr = cache;
        double *xqr = xyr + l1l1 * mesh[2];
        int l;

        if (nimgz == 1) {
                memset(xyr, 0, (l1l1 * mesh[2])*sizeof(double));
                dgemm_(&TRANS_N, &TRANS_N, &ngridz, &l1l1, &l1,
                       &fac, zs_exp+nz0, mesh+2, dm_xyz, &l1,
                       &D0, xyr+nz0, mesh+2);
        } else if (nimgz == 2 && !_has_overlap(nz0, nz1, mesh[2])) {
                memset(xyr, 0, (l1l1 * mesh[2])*sizeof(double));
                ngridz = mesh[2]-nz0;
                dgemm_(&TRANS_N, &TRANS_N, &ngridz, &l1l1, &l1,
                       &fac, zs_exp+nz0, mesh+2, dm_xyz, &l1,
                       &D0, xyr+nz0, mesh+2);
                dgemm_(&TRANS_N, &TRANS_N, &nz1, &l1l1, &l1,
                       &fac, zs_exp, mesh+2, dm_xyz, &l1,
                       &D0, xyr, mesh+2);
        }
        else{
                dgemm_(&TRANS_N, &TRANS_N, mesh+2, &l1l1, &l1,
                       &fac, zs_exp, mesh+2, dm_xyz, &l1,
                       &D0, xyr, mesh+2);
        }

        if (nimgy == 1) {
                for (l = 0; l <= topl; l++) {
                        memset(xqr+l*xcols, 0, (ny0*mesh[2])*sizeof(double));
                        memset(xqr+l*xcols+ny1*mesh[2], 0, ((mesh[1]-ny1)*mesh[2])*sizeof(double));
                        dgemm_(&TRANS_N, &TRANS_T, mesh+2, &ngridy, &l1,
                               &D1, xyr+l*l1*mesh[2], mesh+2, ys_exp+ny0, mesh+1,
                               &D0, xqr+l*xcols+ny0*mesh[2], mesh+2);
                }
        } else if (nimgy == 2 && !_has_overlap(ny0, ny1, mesh[1])) {
                for (l = 0; l <= topl; l++) {
                        memset(xqr+l*xcols+ny1*mesh[2], 0, ((ny0-ny1)*mesh[2])*sizeof(double));
                        dgemm_(&TRANS_N, &TRANS_T, mesh+2, &ny1, &l1,
                               &D1, xyr+l*l1*mesh[2], mesh+2, ys_exp, mesh+1,
                               &D0, xqr+l*xcols, mesh+2);
                        ngridy = mesh[1] - ny0;
                        dgemm_(&TRANS_N, &TRANS_T, mesh+2, &ngridy, &l1,
                               &D1, xyr+l*l1*mesh[2], mesh+2, ys_exp+ny0, mesh+1,
                               &D0, xqr+l*xcols+ny0*mesh[2], mesh+2);
                }
        } else {
                for (l = 0; l <= topl; l++) {
                        dgemm_(&TRANS_N, &TRANS_T, mesh+2, mesh+1, &l1,
                               &D1, xyr+l*l1*mesh[2], mesh+2, ys_exp, mesh+1,
                               &D0, xqr+l*xcols, mesh+2);
                }
        }

        if (nimgx == 1) {
                dgemm_(&TRANS_N, &TRANS_T, &xcols, &ngridx, &l1,
                       &D1, xqr, &xcols, xs_exp+nx0, mesh,
                       &D1, rho+nx0*xcols, &xcols);
        } else if (nimgx == 2 && !_has_overlap(nx0, nx1, mesh[0])) {
                dgemm_(&TRANS_N, &TRANS_T, &xcols, &nx1, &l1,
                       &D1, xqr, &xcols, xs_exp, mesh,
                       &D1, rho, &xcols);
                ngridx = mesh[0] - nx0;
                dgemm_(&TRANS_N, &TRANS_T, &xcols, &ngridx, &l1,
                       &D1, xqr, &xcols, xs_exp+nx0, mesh,
                       &D1, rho+nx0*xcols, &xcols);
        } else {
                dgemm_(&TRANS_N, &TRANS_T, &xcols, mesh, &l1,
                       &D1, xqr, &xcols, xs_exp, mesh,
                       &D1, rho, &xcols);
        }
}


static void _get_dm_to_dm_xyz_coeff(double* coeff, double* rij, int lmax, double* cache)
{
    int l1 = lmax + 1;
    int l, lx;

    double *rx_pow = cache;
    double *ry_pow = rx_pow + l1;
    double *rz_pow = ry_pow + l1;

    rx_pow[0] = 1.0;
    ry_pow[0] = 1.0;
    rz_pow[0] = 1.0;
    for (lx = 1; lx <= lmax; lx++) {
        rx_pow[lx] = rx_pow[lx-1] * rij[0];
        ry_pow[lx] = ry_pow[lx-1] * rij[1];
        rz_pow[lx] = rz_pow[lx-1] * rij[2];
    } 

    int dj = _LEN_CART[lmax];
    double *pcx = coeff;
    double *pcy = pcx + dj;
    double *pcz = pcy + dj;
    for (l = 0; l <= lmax; l++){
        for (lx = 0; lx <= l; lx++) {
            pcx[lx] = BINOMIAL(l, lx) * rx_pow[l-lx];
            pcy[lx] = BINOMIAL(l, lx) * ry_pow[l-lx];
            pcz[lx] = BINOMIAL(l, lx) * rz_pow[l-lx];
        }
        pcx += l+1;
        pcy += l+1;
        pcz += l+1;
    }
}


static void _dm_xyz_to_dm(double* dm_xyz, double* dm, int comp, int li, int lj, double* ri, double* rj, double* cache)
{
    int lx, ly, lz;
    int lx_i, ly_i, lz_i;
    int lx_j, ly_j, lz_j;
    int ic, jx, jy, jz;
    double rij[3];

    rij[0] = ri[0] - rj[0];
    rij[1] = ri[1] - rj[1];
    rij[2] = ri[2] - rj[2];

    int l1 = li + lj + 1;
    int l1l1 = l1 * l1;
    double *coeff = cache;
    int dj = _LEN_CART[lj];
    cache += 3 * dj;

    _get_dm_to_dm_xyz_coeff(coeff, rij, lj, cache);

    double cx, cy, cz;
    double *pcx = coeff;
    double *pcy = pcx + dj;
    double *pcz = pcy + dj;
    double *pdm = dm;
    for (ic = 0; ic < comp; ic++) {
    for (lx_i = li; lx_i >= 0; lx_i--) {
        for (ly_i = li-lx_i; ly_i >= 0; ly_i--) {
            lz_i = li - lx_i - ly_i;
            for (lx_j = lj; lx_j >= 0; lx_j--) {
                for (ly_j = lj-lx_j; ly_j >= 0; ly_j--) {
                    lz_j = lj - lx_j - ly_j;
                    for (jx = 0; jx <= lx_j; jx++) {
                        cx = pcx[jx+_LEN_CART0[lx_j]];
                        lx = lx_i + jx;
                        for (jy = 0; jy <= ly_j; jy++) {
                            cy = pcy[jy+_LEN_CART0[ly_j]];
                            ly = ly_i + jy;
                            for (jz = 0; jz <= lz_j; jz++) {
                                cz = pcz[jz+_LEN_CART0[lz_j]];
                                lz = lz_i + jz;
                                pdm[0] += cx*cy*cz * dm_xyz[lx*l1l1+ly*l1+lz];
                            }
                        }
                    }
                    pdm += 1;
                }
            }
        }
    }}
}


static void _dm_to_dm_xyz(double* dm_xyz, double* dm, int li, int lj, double* ri, double* rj, double* cache)
{
    int lx, ly, lz;
    int lx_i, ly_i, lz_i;
    int lx_j, ly_j, lz_j;
    int jx, jy, jz;
    double rij[3];

    rij[0] = ri[0] - rj[0];
    rij[1] = ri[1] - rj[1];
    rij[2] = ri[2] - rj[2];

    int l1 = li + lj + 1;
    int l1l1 = l1 * l1;
    double *coeff = cache;
    int dj = _LEN_CART[lj];
    cache += 3 * dj;

    _get_dm_to_dm_xyz_coeff(coeff, rij, lj, cache);

    double cx, cy, cz;
    double *pcx = coeff;
    double *pcy = pcx + dj;
    double *pcz = pcy + dj; 
    double *pdm = dm;
    for (lx_i = li; lx_i >= 0; lx_i--) {
        for (ly_i = li-lx_i; ly_i >= 0; ly_i--) {
            lz_i = li - lx_i - ly_i;
            for (lx_j = lj; lx_j >= 0; lx_j--) {
                for (ly_j = lj-lx_j; ly_j >= 0; ly_j--) {
                    lz_j = lj - lx_j - ly_j;
                    for (jx = 0; jx <= lx_j; jx++) {
                        cx = pcx[jx+_LEN_CART0[lx_j]];
                        lx = lx_i + jx;
                        for (jy = 0; jy <= ly_j; jy++) {
                            cy = pcy[jy+_LEN_CART0[ly_j]];
                            ly = ly_i + jy;
                            for (jz = 0; jz <= lz_j; jz++) {
                                cz = pcz[jz+_LEN_CART0[lz_j]];
                                lz = lz_i + jz;
                                dm_xyz[lx*l1l1+ly*l1+lz] += cx*cy*cz * pdm[0];
                            }
                        }
                    }
                    pdm += 1;
                }
            }
        }
    }
}


int eval_mat_lda_orth(double *weights, double *out, int comp,
                        int li, int lj, double ai, double aj,
                        double *ri, double *rj, double fac, double cutoff,
                        int dimension, double *a, double *b,
                        int *mesh, double *cache)
{
        int topl = li + lj;
        int l1 = topl+1;
        int l1l1l1 = l1*l1*l1;
        int img_slice[6];
        int grid_slice[6];
        double *xs_exp, *ys_exp, *zs_exp;

        int data_size = _init_orth_data(&xs_exp, &ys_exp, &zs_exp, img_slice,
                                        grid_slice, mesh,
                                        topl, dimension, cutoff,
                                        ai, aj, ri, rj, a, b, cache);
        if (data_size == 0) {
                return 0;
        }
        cache += data_size;

        double *dm_xyz = cache;
        cache += l1l1l1;
        _orth_ints(dm_xyz, weights, topl, fac, xs_exp, ys_exp, zs_exp,
                   img_slice, grid_slice, mesh, cache);

        _dm_xyz_to_dm(dm_xyz, out, comp, li, lj, ri, rj, cache);
        return 1;
}


void collocate_rho_lda_orth(double *rho, double *dm, int comp,
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
        //double *dm_cart = cache;
        //GTOreverse_vrr2d_ket(dm_cart, dm, li, lj, ri, rj);
        memset(dm_xyz, 0, l1l1l1*sizeof(double));
        //_cart_to_xyz(dm_xyz, dm_cart, li, topl, l1);

        _dm_to_dm_xyz(dm_xyz, dm, li, lj, ri, rj, cache);

        _orth_rho(rho, dm_xyz, fac, topl, mesh,
                  img_slice, grid_slice, xs_exp, ys_exp, zs_exp, cache);
}


static void _apply_ints(int (*eval_ints)(), double *weights, double *mat,
                        PGFPair* pgfpair, int comp, double fac,
                        int dimension, double *a, double *b, int *mesh,
                        double* ish_gto_norm, double* jsh_gto_norm,
                        int *ish_atm, int *ish_bas, double *ish_env,
                        int *jsh_atm, int *jsh_bas, double *jsh_env,
                        double* Ls, double *cache)
{
        int i_sh = pgfpair->ish;
        int j_sh = pgfpair->jsh;
        int ipgf = pgfpair->ipgf;
        int jpgf = pgfpair->jpgf;
        int iL = pgfpair->iL;
        double cutoff = pgfpair->radius;

        int li = ish_bas[ANG_OF+i_sh*BAS_SLOTS];
        int lj = jsh_bas[ANG_OF+j_sh*BAS_SLOTS];
        int di = _LEN_CART[li];
        int dj = _LEN_CART[lj];

        int ish_nprim = ish_bas[NPRIM_OF+i_sh*BAS_SLOTS];
        int jsh_nprim = jsh_bas[NPRIM_OF+j_sh*BAS_SLOTS];
        int naoi = ish_nprim * di;
        int naoj = jsh_nprim * dj;

        double *ri = ish_env + ish_atm[PTR_COORD+ish_bas[ATOM_OF+i_sh*BAS_SLOTS]*ATM_SLOTS];
        double *rj = jsh_env + jsh_atm[PTR_COORD+jsh_bas[ATOM_OF+j_sh*BAS_SLOTS]*ATM_SLOTS];
        double *rL = Ls + iL*3;
        double rjL[3];
        rjL[0] = rj[0] + rL[0];
        rjL[1] = rj[1] + rL[1];
        rjL[2] = rj[2] + rL[2];

        double ai = ish_env[ish_bas[PTR_EXP+i_sh*BAS_SLOTS]+ipgf];
        double aj = jsh_env[jsh_bas[PTR_EXP+j_sh*BAS_SLOTS]+jpgf];
        double ci = ish_gto_norm[ipgf];
        double cj = jsh_gto_norm[jpgf];
        double aij = ai + aj;
        double rrij = CINTsquare_dist(ri, rjL);
        double eij = (ai * aj / aij) * rrij;
        if (eij > EIJCUTOFF) {
                return;
        }
        fac *= exp(-eij) * ci * cj * CINTcommon_fac_sp(li) * CINTcommon_fac_sp(lj);
        if (fac < ish_env[PTR_EXPDROP] && fac < jsh_env[PTR_EXPDROP]) {
                return;
        }

        double *out = cache;
        memset(out, 0, comp*di*dj*sizeof(double));
        cache += comp * di * dj;

        int value = (*eval_ints)(weights, out, comp, li, lj, ai, aj, ri, rjL,
                                 fac, cutoff, dimension, a, b, mesh, cache);

        double *pmat = mat + ipgf*di*naoj + jpgf*dj;
        if (value != 0) {
                int i, j, ic;
                for (ic = 0; ic < comp; ic++) {
                        for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                pmat[i*naoj+j] += out[i*dj+j];
                        } }
                        pmat += naoi * naoj;
                        out += di * dj;
                }
        }
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


static int _ints_cache_size(int l, int nprim, int nctr, int* mesh, double radius, double* a, int comp)
{
    int size = 0;
    int l1 = 2 * l + 1;
    int l1l1 = l1 * l1;
    int ncart = _LEN_CART[l];
    int nimgs = (int) ceil(MAX(MAX(radius/fabs(a[0]), radius/a[4]), radius/a[8])) + 1;
    int nmx = MAX(MAX(mesh[0], mesh[1]), mesh[2]) * nimgs;

    size += comp * nprim * nprim * ncart * ncart;
    size += comp * ncart * ncart;
    size += l1 * (mesh[0] + mesh[1] + mesh[2]);
    size += l1 * nmx + nmx;
    size += l1l1 * l1;
    size += 3 * (ncart + l1);
    size += l1 * mesh[1] * mesh[2];
    size += l1l1 * mesh[2];
    size += nctr * ncart * nprim * ncart;
    return size+1000000;
}


static int _rho_cache_size(int l, int nprim, int nctr, int* mesh, double radius, double* a)
{
    int size = 0;
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
    return size+1000000;
}


static void fill_tril(double* mat, int comp, int* ish_ao_loc, int* jsh_ao_loc,
                      int ish, int jsh, int ish0, int jsh0, int naoi, int naoj)
{
    int i0 = ish_ao_loc[ish] - ish_ao_loc[ish0];
    int i1 = ish_ao_loc[ish+1] - ish_ao_loc[ish0];
    int j0 = jsh_ao_loc[jsh] - jsh_ao_loc[jsh0];
    int j1 = jsh_ao_loc[jsh+1] - jsh_ao_loc[jsh0];
    int ni = i1 - i0;
    int nj = j1 - j0;

    double *pmat_up = mat + i0*naoj + j0;
    double *pmat_low = mat + j0*naoj + i0;
    int ic, i, j;
    for (ic = 0; ic < comp; ic++) {
        for (i = 0; i < ni; i++) {
            for (j = 0; j < nj; j++) {
                pmat_low[j*naoj+i] = pmat_up[i*naoj+j];
            }
        }
        pmat_up += naoi * naoj;
        pmat_low += naoi * naoj;
    }
}


static Shlpair_Task_Index* get_shlpair_task_index(PGFPair** pgfpairs, int ntasks, 
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


void grid_eval_drv(int (*eval_ints)(), double* mat, double* weights, TaskList** task_list,
                   int comp, int hermi, int grid_level, 
                   int *shls_slice, int* ish_ao_loc, int* jsh_ao_loc,
                   int dimension, double* Ls, double* a, double* b,
                   int* ish_atm, int* ish_bas, double* ish_env,
                   int* jsh_atm, int* jsh_bas, double* jsh_env, int cart)
{
    TaskList* tl = *task_list;
    GridLevel_Info* gridlevel_info = tl->gridlevel_info;
    //int nlevels = gridlevel_info->nlevels;

    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    const int nish = ish1 - ish0;
    const int njsh = jsh1 - jsh0;
    const int nijsh = nish * njsh;
    const int naoi = ish_ao_loc[ish1] - ish_ao_loc[ish0];
    const int naoj = jsh_ao_loc[jsh1] - jsh_ao_loc[jsh0];

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

    Task *task = (tl->tasks)[grid_level];
    int ntasks = task->ntasks;
    double max_radius = task->radius;
    PGFPair **pgfpairs = task->pgfpairs;
    int* mesh = gridlevel_info->mesh + grid_level*3;

    Shlpair_Task_Index *shlpair_task_idx = get_shlpair_task_index(pgfpairs, ntasks, ish0, ish1, jsh0, jsh1, hermi);

    int cache_size = _ints_cache_size(MAX(ish_lmax,jsh_lmax),
                                      MAX(ish_nprim_max, jsh_nprim_max),
                                      MAX(ish_nctr_max, jsh_nctr_max), 
                                      mesh, max_radius, a, comp);

#pragma omp parallel
{
    int i, ish, jsh, itask;
    int ijsh, ijsh_ntasks;
    int li, lj, ish_nprim, jsh_nprim;
    int *task_idx = NULL;
    PGFPair *pgfpair = NULL;
    double *cache0 = malloc(sizeof(double) * cache_size);
    double *dm_cart = cache0;
    int len_dm_cart = comp*ish_nprim_max*_LEN_CART[ish_lmax]*jsh_nprim_max*_LEN_CART[jsh_lmax];
    double *cache = dm_cart + len_dm_cart;

    #pragma omp for schedule(dynamic)
    for(ijsh = 0; ijsh < nijsh; ijsh++){
        ish = ijsh / njsh;
        jsh = ijsh % njsh; 

        if (hermi == 1 && jsh < ish) {
            continue;
        }
        
        ish += ish0;
        jsh += jsh0;
        ijsh_ntasks = get_task_index(shlpair_task_idx, &task_idx, ish, jsh);
        if (ijsh_ntasks <= 0) {
            continue;
        }

        li = ish_bas[ANG_OF+ish*BAS_SLOTS];
        lj = jsh_bas[ANG_OF+jsh*BAS_SLOTS];
        ish_nprim = ish_bas[NPRIM_OF+ish*BAS_SLOTS];
        jsh_nprim = jsh_bas[NPRIM_OF+jsh*BAS_SLOTS];
        len_dm_cart = comp*ish_nprim*_LEN_CART[li]*jsh_nprim*_LEN_CART[lj]; 
        memset(dm_cart, 0, len_dm_cart * sizeof(double));
        for (i = 0; i < ijsh_ntasks; i++){
            itask = task_idx[i];
            pgfpair = pgfpairs[itask];
            _apply_ints(eval_ints, weights, dm_cart, pgfpair, comp, 1.0, dimension, a, b, mesh,
                        gto_norm_i[ish], gto_norm_j[jsh], ish_atm, ish_bas, ish_env,
                        jsh_atm, jsh_bas, jsh_env, Ls, cache);
        }

        transform_dm_inverse(dm_cart, mat, comp,
                             cart2sph_coeff_i[ish], cart2sph_coeff_j[jsh],
                             ish_ao_loc, jsh_ao_loc, ish_bas, jsh_bas,
                             ish, jsh, ish0, jsh0, naoi, naoj, cache);

        if (hermi == 1 && ish != jsh) {
            fill_tril(mat, comp, ish_ao_loc, jsh_ao_loc,
                      ish, jsh, ish0, jsh0, naoi, naoj);
        }
    }
    free(cache0);
}

    del_cart2sph_coeff(cart2sph_coeff_i, gto_norm_i, ish0, ish1);
    if (hermi != 1) {
        del_cart2sph_coeff(cart2sph_coeff_j, gto_norm_j, jsh0, jsh1);
    }
    del_shlpair_task_index(shlpair_task_idx);
    free(shlpair_task_idx);
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
    const int nijsh = nish * njsh;
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
        pgfpairs = task->pgfpairs;
        max_radius = task->radius;

        rho = (*rs_rho)->data[ilevel];
        mesh = gridlevel_info->mesh + ilevel*3;

        Shlpair_Task_Index *shlpair_task_idx = get_shlpair_task_index(pgfpairs, ntasks, ish0, ish1, jsh0, jsh1, hermi);

        int cache_size = _rho_cache_size(MAX(ish_lmax,jsh_lmax), 
                                         MAX(ish_nprim_max, jsh_nprim_max),
                                         MAX(ish_nctr_max, jsh_nctr_max), mesh, max_radius, a);
        size_t ngrids = ((size_t)mesh[0]) * mesh[1] * mesh[2];

#pragma omp parallel
{
    PGFPair *pgfpair = NULL;
    //PGFPair *prev_pgfpair = NULL;
    int itask, i, ish, jsh, ijsh, ijsh_ntasks;
    int *task_idx = NULL;
    double *cache0 = malloc(sizeof(double) * cache_size);
    double *dm_cart = cache0;
    double *dm_pgf = dm_cart + ish_nprim_max*_LEN_CART[ish_lmax]*jsh_nprim_max*_LEN_CART[jsh_lmax];
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
    for(ijsh = 0; ijsh < nijsh; ijsh++){
        ish = ijsh / njsh;
        jsh = ijsh % njsh;

        if (hermi == 1 && jsh < ish) {
            continue;
        }

        ish += ish0;
        jsh += jsh0;
        ijsh_ntasks = get_task_index(shlpair_task_idx, &task_idx, ish, jsh);
        if (ijsh_ntasks <= 0) {
            continue;
        }

        transform_dm(dm_cart, dm, cart2sph_coeff_i[ish],
                     cart2sph_coeff_j[jsh], ish_ao_loc, jsh_ao_loc,
                     ish_bas, jsh_bas, ish, jsh, ish0, jsh0, naoj, cache);

        for (i = 0; i < ijsh_ntasks; i++){
            itask = task_idx[i];
            pgfpair = pgfpairs[itask];
            get_dm_pgf(dm_pgf, dm_cart, pgfpair, ish_bas, jsh_bas, hermi);
            _apply_rho(eval_rho, rho_priv, dm_pgf, pgfpair, comp, dimension, a, b, mesh,
                   gto_norm_i[ish], gto_norm_j[jsh], ish_atm, ish_bas, ish_env,
                   jsh_atm, jsh_bas, jsh_env, Ls, cache);
        }
    }

    free(cache0);
    NPomp_dsum_reduce_inplace(rhobufs, comp*ngrids);
    if (thread_id != 0) {
        free(rho_priv);
    }
}

    del_shlpair_task_index(shlpair_task_idx);
    free(shlpair_task_idx);
    } // loop ilevel

    del_cart2sph_coeff(cart2sph_coeff_i, gto_norm_i, ish0, ish1);
    if (hermi != 1) {
        del_cart2sph_coeff(cart2sph_coeff_j, gto_norm_j, jsh0, jsh1);
    }
}
