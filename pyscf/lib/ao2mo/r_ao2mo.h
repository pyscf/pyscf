#include "cint.h"
#include "vhf/cvhf.h"

#if !defined HAVE_DEFINED_R_AO2MOENVS_H
#define HAVE_DEFINED_R_AO2MOENVS_H
struct _AO2MOEnvs {
        int natm;
        int nbas;
        int *atm;
        int *bas;
        double *env;
        int nao;
        int klsh_start;
        int klsh_count;
        int bra_start;
        int bra_count;
        int ket_start;
        int ket_count;
        int ncomp;
        int *tao;
        int *ao_loc;
        double complex *mo_coeff;
        double *mo_r;
        double *mo_i;
        CINTOpt *cintopt;
        CVHFOpt *vhfopt;
};
#endif
