/*
 *
 */

#include "cint.h"

#define DM_PLAIN        0
#define DM_HERMITIAN    1
#define DM_ANTI         2

#if !defined(HAVE_DEFINED_CVHFOPT_H)
#define HAVE_DEFINED_CVHFOPT_H
typedef struct CVHFOpt_struct {
    int nbas;
    int _padding;
    double direct_scf_cutoff;
    double *q_cond;
    double *dm_cond;
    int (*fprescreen)(int *shls, struct CVHFOpt_struct *opt,
                      int *atm, int *bas, double *env);
} CVHFOpt;
#endif

void int2e_sph_o5(double *eri, int *atm, int natm, int *bas, int nbas, double *env);

void CVHFinit_optimizer(CVHFOpt **opt, int *atm, int natm,
                        int *bas, int nbas, double *env);

void CVHFdel_optimizer(CVHFOpt **opt);


int CVHFnoscreen(int *shls, CVHFOpt *opt,
                  int *atm, int *bas, double *env);
int CVHFnr_schwarz_cond(int *shls, CVHFOpt *opt,
                        int *atm, int *bas, double *env);
int CVHFnrs8_prescreen(int *shls, CVHFOpt *opt,
                       int *atm, int *bas, double *env);

