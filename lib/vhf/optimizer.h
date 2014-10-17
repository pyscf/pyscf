/*
 *
 */

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

void CVHFinit_optimizer(CVHFOpt **opt, int *atm, int natm,
                        int *bas, int nbas, double *env);

void CVHFdel_optimizer(CVHFOpt **opt);

int CVHFnoscreen(int *shls, CVHFOpt *opt,
                  int *atm, int *bas, double *env);
int CVHFnr_schwarz_cond(int *shls, CVHFOpt *opt,
                        int *atm, int *bas, double *env);
int CVHFnrs8_prescreen(int *shls, CVHFOpt *opt,
                       int *atm, int *bas, double *env);

void CVHFsetnr_direct_scf(CVHFOpt *opt, int *atm, int natm,
                          int *bas, int nbas, double *env);
void CVHFsetnr_direct_scf_dm(CVHFOpt *opt, double *dm, int nset,
                             int *atm, int natm, int *bas, int nbas, double *env);

void CVHFnr_optimizer(CVHFOpt **vhfopt, int *atm, int natm,
                      int *bas, int nbas, double *env);
