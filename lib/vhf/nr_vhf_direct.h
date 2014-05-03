/*
 *
 */

void nr_vhf_optimizer(CVHFOpt **vhfopt, const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env);

void nr_vhf_direct_o4(double *dm, double *vj, double *vk, CVHFOpt *vhfopt,
                      const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env);

void nr_vhf_direct_m4(double *dm, double *vj, double *vk, const int nset,
                      CVHFOpt *vhfopt, const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env);
