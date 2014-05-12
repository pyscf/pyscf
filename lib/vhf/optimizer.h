/*
 *
 */

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

int CVHFno_screen(int *shls, CVHFOpt *opt);
int CVHFnr_schwarz_cond(int *shls, CVHFOpt *opt);

void CVHFset_direct_scf(CVHFOpt *opt, const int *atm, const int natm,
                        const int *bas, const int nbas, const double *env);
void CVHFset_direct_scf_dm(CVHFOpt *opt, double *dm, const int nset,
                           const int *atm, const int natm,
                           const int *bas, const int nbas, const double *env);
