/// define functions

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef DEFINED_PBC_ISDF
#define DEFINED_PBC_ISDF
    typedef struct PBC_ISDF_struct
    {
        /// the following variables are input variables
        int nao;
        int natm;
        int ngrids;
        double cutoff_aoValue;
        double cutoff_QR;
        int naux; // number of auxiliary basis
        const int *ao2atomID;
        const double *aoG;
        /// the following variables are output variables
        int *voronoi_partition;
        int *ao_sparse_rep_row;
        int *ao_sparse_rep_col;
        double *ao_sparse_rep_val;
        int *IP_index;
        double *auxiliary_basis;
    } PBC_ISDF;
#endif

    void PBC_ISDF_init(PBC_ISDF **opt,
                       int nao,
                       int natm,
                       int ngrids,
                       double cutoff_aoValue,
                       const int *ao2atomID,
                       const double *aoG,
                       double cutoff_QR);

    void PBC_ISDF_del(PBC_ISDF **input);
    void PBC_ISDF_build(PBC_ISDF *input);
    void PBC_ISDF_build_onlyVoronoiPartition(PBC_ISDF *input);

#ifdef __cplusplus
}
#endif