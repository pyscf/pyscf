/*
 *
 */

void CVHFindex_blocks2tri(int *idx, int *ao_loc,
                          const int *bas, const int nbas);

void CVHFnr_optimizer(CVHFOpt **vhfopt, const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env);

int CVHFnr8fold_eri_o2(double *eri, int ish, int jsh, int ksh_lim,
                       const int *atm, const int natm,
                       const int *bas, const int nbas, const double *env,
                       CINTOpt *opt, CVHFOpt *vhfopt);

int CVHFnr_vhf_prescreen(int *shls, CVHFOpt *opt);

void CVHFnr_direct_o4(double *dm, double *vj, double *vk, const int nset,
                      CVHFOpt *vhfopt, const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env);

