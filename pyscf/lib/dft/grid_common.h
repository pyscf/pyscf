#ifndef HAVE_DEFINED_GRID_COMMON_H
#define HAVE_DEFINED_GRID_COMMON_H

#include "cint.h"

#define EIJCUTOFF        60
#define PTR_EXPDROP      16

extern double CINTsquare_dist(const double *r1, const double *r2);
extern double CINTcommon_fac_sp(int l);

int get_lmax(int ish0, int ish1, int* bas);
int get_nprim_max(int ish0, int ish1, int* bas);
int get_nctr_max(int ish0, int ish1, int* bas);
void get_cart2sph_coeff(double** contr_coeff, double** gto_norm,
                        int ish0, int ish1, int* bas, double* env, int cart);
void del_cart2sph_coeff(double** contr_coeff, double** gto_norm, int ish0, int ish1);

static inline int _has_overlap(int nx0, int nx1, int nx_per_cell)
{
    return nx0 <= nx1;
}

static inline int _num_grids_on_x(int nimgx, int nx0, int nx1, int nx_per_cell)
{
    int ngridx;
    if (nimgx == 1) {
        ngridx = nx1 - nx0;
    } else if (nimgx == 2 && !_has_overlap(nx0, nx1, nx_per_cell)) {
        ngridx = nx1 - nx0 + nx_per_cell;
    } else {
        ngridx = nx_per_cell;
    }
    return ngridx;
}

int _orth_components(double *xs_exp, int *img_slice, int *grid_slice,
                     double a, double b, double cutoff,
                     double xi, double xj, double ai, double aj,
                     int periodic, int nx_per_cell, int topl, double *cache);
int _init_orth_data(double **xs_exp, double **ys_exp, double **zs_exp,
                    int *img_slice, int *grid_slice, int *mesh,
                    int topl, int dimension, double cutoff,
                    double ai, double aj, double *ri, double *rj,
                    double *a, double *b, double *cache);
void _get_dm_to_dm_xyz_coeff(double* coeff, double* rij, int lmax, double* cache);
void _dm_to_dm_xyz(double* dm_xyz, double* dm, int li, int lj, double* ri, double* rj, double* cache);
void _dm_xyz_to_dm(double* dm_xyz, double* dm, int li, int lj, double* ri, double* rj, double* cache);
void get_dm_pgfpair(double* dm_pgf, double* dm_cart,
                    PGFPair* pgfpair, int* ish_bas, int* jsh_bas, int hermi);
#endif