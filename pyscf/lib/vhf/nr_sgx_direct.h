#ifndef _NR_SGX_DIRECT_H
#define _NR_SGX_DIRECT_H

#include <stdlib.h>

typedef struct CSGXOpt_struct {
    // information on the algorithm and system
    int mode;
    int nbas;
    int ngrids;
    int *ao_loc;
    double etol;
    double *vtol;
    double *wt;
    
    // outputs used in the process of screening
    // double *fbar_i;
    // double *bscreen_i;
    // double *shlscreen_i;

    // integral bounds, maxes, sums, etc.
    double *mbar_ij;
    double *mbar_bi;
    double *rbar_ij;
    double *mmax_i;
    double *msum_i;

    // buffer for computations and its size in bytes
    size_t buf_size_bytes;
    size_t shl_info_size_bytes;

    // integral cutoff tolerance as in CVHFOpt
    double direct_scf_cutoff;

    // screening functions
    // frscreen_grid sets up the screening process for all
    // shell pairs in a batch
    void (*fscreen_grid)(struct CSGXOpt_struct *opt, double *f_ug,
                         int ibatch, int n_dm, int nish, void *shl_info,
                         void *buf);
    int (*fscreen_shl)(struct CSGXOpt_struct *opt, int jmax, int ish,
                       int nish, int ish0, int jsh0, void *shl_info,
                       void *buf, int *shl_inds);

    // approximate block-norm of full density matrix, for use in
    // incremental Fock build
    double *full_f_bi;
} CSGXOpt;

#endif
