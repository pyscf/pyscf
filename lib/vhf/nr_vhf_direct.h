/*
 *
 */

void CVHFindex_blocks2tri(int *idx, int *ao_loc,
                          const int *bas, const int nbas);

void CVHFnr_optimizer(CVHFOpt **vhfopt, const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env);

void CVHFnr_direct_o4(double *dm, double *vj, double *vk, const int nset,
                      CVHFOpt *vhfopt, const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env);

