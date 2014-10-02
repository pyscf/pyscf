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
        int *ao_loc;
        int *idx_tri;
};

void CVHFindex_blocks2tri(int *idx, int *ao_loc,
                          const int *bas, const int nbas);

int CVHFfill_nr_eri_o2(double *eri, int ish, int jsh, int ksh_lim,
                       int *atm, int natm, int *bas, int nbas, double *env,
                       CINTOpt *opt, CVHFOpt *vhfopt);

void CVHFnr_direct_o4(double *dm, double *vj, double *vk, const int nset,
                      CVHFOpt *vhfopt, int *atm, int natm,
                      int *bas, int nbas, double *env);

