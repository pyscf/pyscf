/*
 *
 */

#include "cint.h"
#include "optimizer.h"

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

void CVHFdot_nrs1(int (*intor)(), void (**fjk)(), double **dms, double *vjk,
                 int n_dm, int ncomp, int ish, int jsh,
                 CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs);

void CVHFdot_nrs2ij(int (*intor)(), void (**fjk)(), double **dms, double *vjk,
                    int n_dm, int ncomp, int ish, int jsh,
                    CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs);

void CVHFdot_nrs2kl(int (*intor)(), void (**fjk)(), double **dms, double *vjk,
                    int n_dm, int ncomp, int ish, int jsh,
                    CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs);

void CVHFdot_nrs4(int (*intor)(), void (**fjk)(), double **dms, double *vjk,
                  int n_dm, int ncomp, int ish, int jsh,
                  CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs);

void CVHFdot_nrs8(int (*intor)(), void (**fjk)(), double **dms, double *vjk,
                  int n_dm, int ncomp, int ish, int jsh,
                  CINTOpt *cintopt, CVHFOpt *vhfopt, struct _VHFEnvs *envs);

void CVHFnr_direct_drv(int (*intor)(), void (*fdot)(), void (**fjk)(),
                       double **dms, double *vjk,
                       int n_dm, int ncomp, CINTOpt *cintopt, CVHFOpt *vhfopt,
                       int *atm, int natm, int *bas, int nbas, double *env);
