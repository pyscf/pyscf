/*
 *
 */

#if !defined HAVE_DEFINED_CVHFOPT_H
#define HAVE_DEFINED_CVHFOPT_H
typedef struct {
    int nbas;
    double direct_scf_cutoff;
    double *q_cond;
    double *dm_cond;
} CVHFOpt;
#endif

void CVHFinit_optimizer(CVHFOpt **opt, const int *atm, const int natm,
                        const int *bas, const int nbas, const double *env);

void CVHFdel_optimizer(CVHFOpt **opt);

void CVHFset_direct_scf(CVHFOpt *opt, const int *atm, const int natm,
                        const int *bas, const int nbas, const double *env);
void CVHFset_direct_scf_dm(CVHFOpt *opt, double *dm, const int nset,
                           const int *atm, const int natm,
                           const int *bas, const int nbas, const double *env);
