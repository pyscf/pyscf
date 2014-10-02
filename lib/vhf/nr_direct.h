/*
 *
 */

struct _VHFEnvs {
        int natm;
        int nbas;
        const int *atm;
        const int *bas;
        const double *env;
        int nao;
        int *ao_loc; // size of nbas+1, last element = nao
        int *tao; // time reversal mappings, index start from 1
};

void CVHFunpack_nrblock2rect (double *buf, double *eri,
                              int ish, int jsh, int dkl, struct _VHFEnvs *envs);
void CVHFunpack_nrblock2tril (double *buf, double *eri,
                              int ish, int jsh, int dkl, struct _VHFEnvs *envs);
void CVHFunpack_nrblock2trilu(double *buf, double *eri,
                              int ish, int jsh, int dkl, struct _VHFEnvs *envs);
int CVHFfill_nr_s1(int (*intor)(), void (*funpack)(), int (*fprescreen)(),
                   double *eri, int ksh, int lsh, int ncomp,
                   CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs);
int CVHFfill_nr_s2ij(int (*intor)(), void (*funpack)(), int (*fprescreen)(),
                     double *eri, int ksh, int lsh, int ncomp,
                     CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs);
int CVHFfill_nr_s2kl(int (*intor)(), void (*funpack)(), int (*fprescreen)(),
                     double *eri, int ksh, int lsh, int ncomp,
                     CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs);
int CVHFfill_nr_s4(int (*intor)(), void (*funpack)(), int (*fprescreen)(),
                   double *eri, int ksh, int lsh, int ncomp,
                   CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs);
int CVHFfill_nr_s8(int (*intor)(), void (*funpack)(), int (*fprescreen)(),
                   double *eri, int ksh, int lsh, int ncomp,
                   CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs);
void CVHFfill_dot_nrs8(int (*intor)(), void (*funpack)(), void (**fjk)(),
                       double **dms, double *vjk,
                       int n_dm, int ncomp, int ksh, int lsh,
                       CINTOpt *cintopt, CVHFOpt *vhfopt,
                       struct _VHFEnvs *envs);
void CVHFfill_dot_nrs4(int (*intor)(), void (*funpack)(), void (**fjk)(),
                       double **dms, double *vjk,
                       int n_dm, int ncomp, int ksh, int lsh,
                       CINTOpt *cintopt, CVHFOpt *vhfopt,
                       struct _VHFEnvs *envs);
void CVHFfill_dot_nrs2kl(int (*intor)(), void (*funpack)(), void (**fjk)(),
                         double **dms, double *vjk,
                         int n_dm, int ncomp, int ksh, int lsh,
                         CINTOpt *cintopt, CVHFOpt *vhfopt,
                         struct _VHFEnvs *envs);
void CVHFfill_dot_nrs2ij(int (*intor)(), void (*funpack)(), void (**fjk)(),
                         double **dms, double *vjk,
                         int n_dm, int ncomp, int ksh, int lsh,
                         CINTOpt *cintopt, CVHFOpt *vhfopt,
                         struct _VHFEnvs *envs);
void CVHFfill_dot_nrs1(int (*intor)(), void (*funpack)(), void (**fjk)(),
                       double **dms, double *vjk,
                       int n_dm, int ncomp, int ksh, int lsh,
                       CINTOpt *cintopt, CVHFOpt *vhfopt,
                       struct _VHFEnvs *envs);
void CVHFnr_direct_drv(int (*intor)(), void (*fdot)(), void (*funpack)(),
                       void (**fjk)(), double **dms, double *vjk,
                       int n_dm, int ncomp, CINTOpt *cintopt, CVHFOpt *vhfopt,
                       int *atm, int natm, int *bas, int nbas, double *env);
