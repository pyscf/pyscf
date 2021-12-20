#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include "vhf/fblas.h"
#include "config.h"
#include "cint.h"
#include "np_helper/np_helper.h"

#define HL_TABLE_SLOTS  7
//#define ATOM_OF         0
//#define ANG_OF          1
#define HL_DIM_OF       2
#define HL_DATA_OF      3
#define HL_OFFSET0      4
#define HF_OFFSET1      5
#define HF_OFFSET2      6
#define MAX_THREADS     256

int GTOmax_shell_dim(int *ao_loc, int *shls_slice, int ncenter);

//#define MAX(x, y) (((x) > (y)) ? (x) : (y))

/*
static double maxval(double* a, int n)
{
    double amax = 0;
    for (size_t i=0; i<n; i++) {
        amax = MAX(amax, fabs(a[i]));
    }
    return amax;
}*/

void contract_ppnl(double* out,
                   double* ppnl_half0, 
                   double* ppnl_half1, 
                   double* ppnl_half2,
                   int* hl_table, double* hl_data,
                   int nhl, int nao)
{
    const int One = 1;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const double D1 = 1.;
    const double D0 = 0.;

    size_t nao_pair = (size_t) nao*nao;
    size_t i;
    for (i = 0; i < nao_pair; i++) {
        out[i] = 0;
    }

#pragma omp parallel
{
    size_t ib, i, p;
    double *pout;
    double *buf = (double*) malloc(sizeof(double)*54*nao);
    #pragma omp for schedule(dynamic)
    for (p = 0; p < nao; p++){
        pout = out + (size_t)p*nao;
    for (ib = 0; ib < nhl; ib++) {
        int *table = hl_table + ib * HL_TABLE_SLOTS;
        int hl_dim = table[HL_DIM_OF];
        int ptr = table[HL_DATA_OF];
        int nd = table[ANG_OF] * 2 + 1;
        int *offset = table + HL_OFFSET0;
        double *hl = hl_data + ptr;
        double *ilp = buf;
        int lp_dim = nd * nao;
        //int ilp_dim = hl_dim * lp_dim;
        int il_dim = hl_dim * nd;
        for (i=0; i<hl_dim; i++) {
            int p0 = offset[i];
            if (i == 0) {
                dcopy_(&lp_dim, ppnl_half0+p0*nao, &One, ilp+i*lp_dim, &One);
            }
            else if (i == 1) {
                dcopy_(&lp_dim, ppnl_half1+p0*nao, &One, ilp+i*lp_dim, &One);
            }
            else if (i == 2) {
                dcopy_(&lp_dim, ppnl_half2+p0*nao, &One, ilp+i*lp_dim, &One);
            }
        }
        double *hilp = ilp + hl_dim*lp_dim;
        dgemm_(&TRANS_N, &TRANS_N, &lp_dim, &hl_dim, &hl_dim, 
               &D1, ilp, &lp_dim, hl, &hl_dim, &D0, hilp, &lp_dim);
        dgemm_(&TRANS_N, &TRANS_T, &nao, &One, &il_dim,
               &D1, hilp, &nao, ilp+p, &nao, &D1, pout, &nao);
    }}
    free(buf);
}
}


void contract_ppnl_ip1(double* out, int comp,
                       double* ppnl_half0, double* ppnl_half1, double* ppnl_half2,
                       double* ppnl_half_ip2_0, double* ppnl_half_ip2_1, double* ppnl_half_ip2_2,
                       int* hl_table, double* hl_data, int nhl, int nao, int naux,
                       int* aux_id)
{
    const int One = 1;
    const char TRANS_N = 'N';
    //const char TRANS_T = 'T';
    const double D1 = 1.;
    const double D0 = 0.;

    size_t nao_pair = (size_t) nao * nao;
    memset(out, 0, nao_pair*comp*sizeof(double));

    size_t n2 = (size_t) nao * naux;
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
                        dcopy_(&nd, ppnl_half_ip2_0+p+p0*nao+ic*n2, &nao, ilp_ip2+i*nd, &One);
                    }
                    else if (i == 1) {
                        dcopy_(&lp_dim, ppnl_half1+p0*nao, &One, ilp+i*lp_dim, &One);
                        dcopy_(&nd, ppnl_half_ip2_1+p+p0*nao+ic*n2, &nao, ilp_ip2+i*nd, &One);
                    }
                    else if (i == 2) {
                        dcopy_(&lp_dim, ppnl_half2+p0*nao, &One, ilp+i*lp_dim, &One);
                        dcopy_(&nd, ppnl_half_ip2_2+p+p0*nao+ic*n2, &nao, ilp_ip2+i*nd, &One);
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


static void contract_vppnl_ipik_dm(double* grad, double* dm, double* eri, int comp,
                                   int* ao_loc, int* bas, int nao, int ish, int jsh, int katm)
{
    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di * dj;
    const int i0 = ao_loc[ish];
    const int j0 = ao_loc[jsh];
    const int iatm = bas[ATOM_OF+ish*BAS_SLOTS];

    const int One = 1;
    int i, ic;
    double *ptr_eri, *ptr_dm;
    double *dm0 = dm + (i0 * nao + j0);
    double ipi_dm[comp];
    for (ic = 0; ic < comp; ic++) {
        ipi_dm[ic] = 0;
        ptr_dm = dm0;
        ptr_eri = eri + dij * ic;
        for (i = 0; i < di; i++) {
            ipi_dm[ic] += ddot_(&dj, ptr_eri+i*dj, &One, ptr_dm, &One);
            ptr_dm += nao;
        }
    }

    for (ic = 0; ic < comp; ic++) {
        grad[iatm*comp+ic] += ipi_dm[ic];
        grad[katm*comp+ic] -= ipi_dm[ic];
    }
}


void vppnl_ip1_fill_gs1(double* grad, double* dm, int comp,
                               double* ppnl_half0, double* ppnl_half1, double* ppnl_half2,
                               double* ppnl_half_ip2_0, double* ppnl_half_ip2_1, double* ppnl_half_ip2_2,
                               int* hl_table, double* hl_data, int nhl, int nao, int naux,
                               int* shls_slice, int* ao_loc, int* bas, double* buf, int ish, int jsh)
{
    const int ish0 = shls_slice[0];
    const int jsh0 = shls_slice[2];

    ish += ish0;
    jsh += jsh0;

    const int di = ao_loc[ish+1] - ao_loc[ish];
    const int dj = ao_loc[jsh+1] - ao_loc[jsh];
    const int dij = di * dj;
    const int i0 = ao_loc[ish];
    const int j0 = ao_loc[jsh];

    size_t n2 = (size_t) nao * naux;

    const int One = 1;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';
    const double D1 = 1.;
    const double D0 = 0.;

    int i, ib, ic, li;
    double *ptr_ppnl_i, *ptr_ppnl_j;
    double *ptr_ilp_i, *ptr_ilp_j;
    for (ib = 0; ib < nhl; ib++) {
        int *table = hl_table + ib * HL_TABLE_SLOTS;
        int katm = table[ATOM_OF];
        int l = table[ANG_OF];
        int hl_dim = table[HL_DIM_OF];
        int ptr = table[HL_DATA_OF];
        int nd = 2 * l + 1;
        int *offset = table + HL_OFFSET0;
        double *hl = hl_data + ptr;
        int lp_dim_i = nd * di;
        int lp_dim_j = nd * dj;
        int ilp_dim_i = hl_dim * lp_dim_i;
        int ilp_dim_j = hl_dim * lp_dim_j;
        int il_dim = hl_dim * nd;

        double *ilp = buf + dij*comp;
        double *ilp_ip2 = ilp + ilp_dim_j;
        double *hilp = ilp_ip2 + ilp_dim_i;
        for (ic = 0; ic < comp; ic++) {
            for (i=0; i<hl_dim; i++) {
                int p0 = offset[i];
                if (i == 0) {
                    ptr_ppnl_i = ppnl_half_ip2_0 + i0+p0*nao+ic*n2;
                    ptr_ppnl_j = ppnl_half0 + p0*nao+j0;
                }
                else if (i == 1) {
                    ptr_ppnl_i = ppnl_half_ip2_1 + i0+p0*nao+ic*n2;
                    ptr_ppnl_j = ppnl_half1 + p0*nao+j0;
                }
                else if (i == 2) {
                    ptr_ppnl_i = ppnl_half_ip2_2 + i0+p0*nao+ic*n2;
                    ptr_ppnl_j = ppnl_half2 + p0*nao+j0;
                }

                ptr_ilp_i = ilp_ip2 + i*lp_dim_i;
                ptr_ilp_j = ilp + i*lp_dim_j;
                for (li = 0; li < nd; li++) {
                    dcopy_(&di, ptr_ppnl_i, &One, ptr_ilp_i, &One);
                    dcopy_(&dj, ptr_ppnl_j, &One, ptr_ilp_j, &One);
                    ptr_ppnl_i += nao;
                    ptr_ppnl_j += nao;
                    ptr_ilp_i += di;
                    ptr_ilp_j += dj;
                }
            }
            dgemm_(&TRANS_N, &TRANS_N, &lp_dim_j, &hl_dim, &hl_dim,
                   &D1, ilp, &lp_dim_j, hl, &hl_dim, &D0, hilp, &lp_dim_j);
            dgemm_(&TRANS_N, &TRANS_T, &dj, &di, &il_dim,
                   &D1, hilp, &dj, ilp_ip2, &di, &D0, buf+ic*dij, &dj);
        }
        contract_vppnl_ipik_dm(grad, dm, buf, comp, ao_loc, bas, nao, ish, jsh, katm);
    }
}



void contract_ppnl_nuc_grad(void (*fill)(), double* grad, double* dm, int comp,
                            double* ppnl_half0, double* ppnl_half1, double* ppnl_half2,
                            double* ppnl_half_ip2_0, double* ppnl_half_ip2_1, double* ppnl_half_ip2_2,
                            int* hl_table, double* hl_data, int nhl, int nao, int naux,
                            int* shls_slice, int* ao_loc, int* bas, int natm)
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
    size_t buf_size = di*dj*comp + 3*9*(di+dj*2);

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

        (*fill)(grad_loc, dm, comp,
                ppnl_half0, ppnl_half1, ppnl_half2,
                ppnl_half_ip2_0, ppnl_half_ip2_1, ppnl_half_ip2_2,
                hl_table, hl_data, nhl, nao, naux,
                shls_slice, ao_loc, bas, buf, ish, jsh);
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
