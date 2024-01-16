/// define functions

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct PBC_ISDF_struct
    {
        /// the following variables are input variables
        int nao;
        int natm;
        int ngrids;
        double cutoff_aoValue;
        const int *ao2atomID;
        const double *aoG;
        double cutoff_QR;
        /// the following variables are output variables
        int *voronoi_partition;
        int *ao_sparse_rep_row;
        int *ao_sparse_rep_col;
        double *ao_sparse_rep_val;
        int naux;
        int *IP_index;
        double *auxiliary_basis;
    } PBC_ISDF;

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

#ifdef __cplusplus
}
#endif