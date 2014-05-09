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


void CVHFnr_optimizer(CVHFOpt **vhfopt, const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env);

void CVHFnr_direct_o4(double *dm, double *vj, double *vk, const int nset,
                      CVHFOpt *vhfopt, const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env);

void CVHFindex_blocks2tri(int *idx, int *ao_loc,
                          const int *bas, const int nbas);
void CVHFset_ij2i(int *ij2i, int n);
void CVHFunpack(int n, double *vec, double *mat);

void CVHFnr_k(int n, double *eri, double *dm, double *vk);
void CVHFnr_incore_o3(int n, double *eri, double *dm, double *vj, double *vk);
void CVHFnr_incore_o4(int n, double *eri, double *dm, double *vj, double *vk);

