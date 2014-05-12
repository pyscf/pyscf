/*
 *
 */

#include "cint.h"

#if !defined HAVE_DEFINED_CVHFOPT_H
#define HAVE_DEFINED_CVHFOPT_H
typedef struct CVHFOpt_struct {
    int nbas;
    double direct_scf_cutoff;
    double *q_cond;
    double *dm_cond;
    int (*fprescreen)(int *shls, struct CVHFOpt_struct *opt);
} CVHFOpt;
#endif

void CVHFinit_optimizer(CVHFOpt **opt, const int *atm, const int natm,
                        const int *bas, const int nbas, const double *env);

void CVHFdel_optimizer(CVHFOpt **opt);


void CVHFnr_optimizer(CVHFOpt **vhfopt, const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env);

void CVHFnr_direct_o4(double *dm, double *vj, double *vk, const int nset,
                      CVHFOpt *vhfopt, const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env);

int CVHFnr8fold_eri_o2(double *eri, int ish, int jsh, int ksh_lim,
                       const int *atm, const int natm,
                       const int *bas, const int nbas, const double *env,
                       CINTOpt *opt, CVHFOpt *vhfopt);

int CVHFno_screen(int *shls, CVHFOpt *opt);
int CVHFnr_schwarz_cond(int *shls, CVHFOpt *opt);
int CVHFnr_vhf_prescreen(int *shls, CVHFOpt *opt);

void CVHFindex_blocks2tri(int *idx, int *ao_loc,
                          const int *bas, const int nbas);
void CVHFset_ij2i(int *ij2i, int n);
void CVHFunpack(int n, double *vec, double *mat);
void extract_row_from_tri(double *row, int row_id, int ndim, double *tri);

void CVHFnr_k(int n, double *eri, double *dm, double *vk);
void CVHFnr_incore_o3(int n, double *eri, double *dm, double *vj, double *vk);
void CVHFnr_incore_o4(int n, double *eri, double *dm, double *vj, double *vk);

