#include "pbc_isdf_cpp.h"

template <bool getAOSparseRep>
void _VoronoiPartition_and_AOSparseRep(
    IN const double *AOonGrids, // Row-major, (nAO, nGrids)
    IN const int nAO,
    IN const int nGrids,
    IN const int *AO2AtomID,
    IN const double cutoff,
    OUT int **VoronoiPartition, // for each point on the grid, the atom ID of the Voronoi cell it belongs to
    OUT int **AOSparseRepRow,
    OUT int **AOSparseRepCol,
    OUT double **AOSparseRepVal)
{
    static const int BATCHSIZE = 32;
    static const int SPARSE_RESERVE = 128;

    auto nBatch = nGrids / BATCHSIZE + (nGrids % BATCHSIZE == 0 ? 0 : 1);

    assert(AOonGrids != NULL);
    assert(AO2AtomID != NULL);
    assert(*VoronoiPartition == NULL);
    assert(*AOSparseRepRow == NULL);
    assert(*AOSparseRepCol == NULL);
    assert(*AOSparseRepVal == NULL);

    *VoronoiPartition = (int *)malloc(sizeof(int) * nGrids);
    Clear(*VoronoiPartition, nGrids);

    auto nThreads = OMP_NUM_OF_THREADS;

    /// pre allocate buffer

    std::vector<double> Buffer_aoABS(nThreads * nGrids);
    std::vector<int> Buffer_atmID(nThreads * nGrids);

    Clear(Buffer_aoABS.data(), Buffer_aoABS.size());
    Clear(Buffer_atmID.data(), Buffer_atmID.size());

    std::vector<std::vector<double>> Buffer_AO_on_Grids_Value(nThreads);
    std::vector<std::vector<int>> Buffer_AO_on_Grids_GridID(nThreads);

    if constexpr (getAOSparseRep)
    {
        Buffer_AO_on_Grids_GridID.resize(nAO);
        Buffer_AO_on_Grids_Value.resize(nAO);
        for (auto &v : Buffer_AO_on_Grids_GridID)
            v.reserve(SPARSE_RESERVE);
        for (auto &v : Buffer_AO_on_Grids_Value)
            v.reserve(SPARSE_RESERVE);
    }

#pragma omp parallel num_threads(nThreads)
    {
        auto threadID = OMP_THREAD_LABEL;

        double *aoABS_ptr = Buffer_aoABS.data() + threadID * nGrids;
        int *atmID_ptr = Buffer_atmID.data() + threadID * nGrids;

#pragma omp for schedule(static)
        for (int i = 0; i < nAO; ++i)
        {
            const double *ao_ptr = AOonGrids + i * nGrids;
            int atmID = AO2AtomID[i];
            for (int j = 0; j < nGrids; ++j)
            {
                if (fabs(ao_ptr[j]) > cutoff)
                {
                    aoABS_ptr[j] = fabs(ao_ptr[j]);
                    atmID_ptr[j] = atmID;
                }

                if constexpr (getAOSparseRep)
                {
                    if (fabs(ao_ptr[j]) > cutoff)
                    {
                        Buffer_AO_on_Grids_GridID[i].push_back(j);
                        Buffer_AO_on_Grids_Value[i].push_back(ao_ptr[j]);
                    }
                }
            }
        }

        /// merge buffer

#pragma omp for schedule(static, BATCHSIZE)
        for (int i = 0; i < nBatch; ++i)
        {
            int start = i * BATCHSIZE;
            int end = std::min((i + 1) * BATCHSIZE, nGrids);

            for (int j = start; j < end; ++j)
            {

                double maxABS = 0.0;
                int maxID = -1;

                for (int k = 0; k < nThreads; ++k)
                {
                    if (Buffer_aoABS[k * nGrids + j] > maxABS)
                    {
                        maxABS = Buffer_aoABS[k * nGrids + j];
                        maxID = Buffer_atmID[k * nGrids + j];
                    }
                }

                (*VoronoiPartition)[j] = maxID;
            }
        }
    }

    if constexpr (getAOSparseRep)
    {
        /// pack sparse rep

        *AOSparseRepRow = (int *)malloc(sizeof(int) * (nAO + 1));
        (*AOSparseRepRow)[0] = 0;

        size_t nElmt = 0;

        for (int i = 0; i < nAO; ++i)
        {
            nElmt += Buffer_AO_on_Grids_GridID[i].size();
            (*AOSparseRepRow)[i + 1] = nElmt;
        }

        *AOSparseRepCol = (int *)malloc(sizeof(int) * nElmt);
        *AOSparseRepVal = (double *)malloc(sizeof(double) * nElmt);

        nElmt = 0;

        for (int i = 0; i < nAO; ++i)
        {
            auto &gridID = Buffer_AO_on_Grids_GridID[i];
            auto &value = Buffer_AO_on_Grids_Value[i];

            memcpy(*AOSparseRepCol + nElmt, gridID.data(), sizeof(int) * gridID.size());
            memcpy(*AOSparseRepVal + nElmt, value.data(), sizeof(double) * value.size());
            nElmt += gridID.size();
        }
    }
}

void _VoronoiPartition_Only(IN const double *AOonGrids, // Row-major, (nAO, nGrids)
                            IN const int nAO,
                            IN const int nGrids,
                            IN const int *AO2AtomID,
                            IN const double cutoff,
                            OUT int **VoronoiPartition)
{
    _VoronoiPartition_and_AOSparseRep<false>(AOonGrids, nAO, nGrids, AO2AtomID, cutoff, VoronoiPartition, NULL, NULL, NULL);
}

void _VoronoiPartition(IN const double *AOonGrids, // Row-major, (nAO, nGrids)
                       IN const int nAO,
                       IN const int nGrids,
                       IN const int *AO2AtomID,
                       IN const double cutoff,
                       OUT int **VoronoiPartition, // for each point on the grid, the atom ID of the Voronoi cell it belongs to
                       OUT int **AOSparseRepRow,
                       OUT int **AOSparseRepCol,
                       OUT double **AOSparseRepVal)
{
    _VoronoiPartition_and_AOSparseRep<true>(AOonGrids, nAO, nGrids, AO2AtomID, cutoff, VoronoiPartition, AOSparseRepRow, AOSparseRepCol, AOSparseRepVal);
}

/// Sandeep's method to determine IP , a multi-level QRCP based method

std::vector<std::vector<int>> _get_atm_2_non_vanishing_AO(
    IN const int *VoronoiPartition, // Row-major, (nAO, nGrids)
    IN const int nAO,
    IN const int nAtom,
    IN const int *AOSparseRepRow,
    IN const int *AOSparseRepCol)
{
    std::vector<std::vector<int>> atm_2_non_vanishing_AO(nAtom);

    std::vector<int> HasValueOnAtm(nAtom, 0);

    for (int i = 0; i < nAO; ++i)
    {
        int begin_indx = AOSparseRepRow[i];
        int end_indx = AOSparseRepRow[i + 1];

        Clear(HasValueOnAtm.data(), nAtom);

        for (int j = begin_indx; j < end_indx; ++j)
        {
            int gridID = AOSparseRepCol[j];
            int atmID = VoronoiPartition[gridID];
            HasValueOnAtm[atmID] = 1;
        }

        for (int j = 0; j < nAtom; ++j)
        {
            if (HasValueOnAtm[j])
            {
                atm_2_non_vanishing_AO[j].push_back(i);
            }
        }
    }

    return atm_2_non_vanishing_AO;
}

std::vector<int> _get_gridID_2_atmgridID(
    IN const int *VoronoiPartition,
    IN const int nGrids,
    IN const int nAtom)
{
    std::vector<int> AtmGridID(nAtom, 0);

    std::vector<int> gridID_2_atmgridID(nGrids, -1);

    for (int i = 0; i < nGrids; ++i)
    {
        int atmID = VoronoiPartition[i];
        gridID_2_atmgridID[i] = AtmGridID[atmID];
        AtmGridID[atmID] += 1;
    }

    return gridID_2_atmgridID;
}

std::vector<std::vector<int>> _get_atmgridID_2_grid_ID(IN const int *VoronoiPartition,
                                                       IN const int nGrids,
                                                       IN const int nAtom)
{
    std::vector<std::vector<int>> atmgridID_2_grid_ID;
    atmgridID_2_grid_ID.resize(nAtom);

    for (int i = 0; i < nGrids; ++i)
    {
        int atmID = VoronoiPartition[i];
        atmgridID_2_grid_ID[atmID].push_back(i);
    }

    return atmgridID_2_grid_ID;
}

/// given atmID get sparse rep of AO pairs
/// this is worth noted that the ordering of the AO pairs does not matter !

int _get_AO_pairs_SparseRep(
    IN const int i_nPoint,
    IN const int *i_Grid_ID,
    IN const double *i_Value_on_Grid,
    IN const int j_nPoint,
    IN const int *j_Grid_ID,
    IN const double *j_Value_on_Grid,
    IN const double cutoff,
    OUT int *AO_pairs_sparseRepCol,
    OUT double *AO_pairs_sparseRepVal)
{
    /// coefficientwise product

    int i_loc = 0;
    int j_loc = 0;

    int nElmt = 0;

    while (i_loc < i_nPoint && j_loc < j_nPoint)
    {
        int i_gridID = i_Grid_ID[i_loc];
        int j_gridID = j_Grid_ID[j_loc];

        if (i_gridID == j_gridID)
        {
            double val = i_Value_on_Grid[i_loc] * j_Value_on_Grid[j_loc];

            if (fabs(val) > cutoff)
            {
                AO_pairs_sparseRepCol[nElmt] = i_gridID;
                AO_pairs_sparseRepVal[nElmt] = val;
                nElmt += 1;
            }

            i_loc += 1;
            j_loc += 1;
        }
        else if (i_gridID < j_gridID)
        {
            i_loc += 1;
        }
        else
        {
            j_loc += 1;
        }
    }

    return nElmt;
}

int _get_AO_pairs_SparseRep_on_givan_Atm(
    IN const int atmID,
    IN const int nAO,    // number of AOs
    IN const int *AO_ID, // AO has non-zero value on atm atmID
    IN const int *AOSparseRepRow,
    IN const int *AOSparseRepCol,
    IN const double *AOSparseRepVal,
    IN const double cutoff,
    OUT std::vector<int> &AO_pairs_sparseRepRow,
    OUT std::vector<int> &AO_pairs_sparseRepCol,
    OUT std::vector<double> &AO_pairs_sparseRepVal)
{
    int nPair = 0;
    int nnz = 0;

    AO_pairs_sparseRepRow.resize(0);
    AO_pairs_sparseRepCol.resize(0);
    AO_pairs_sparseRepVal.resize(0);

    AO_pairs_sparseRepRow.reserve(nAO + 1);
    AO_pairs_sparseRepCol.reserve(65536);
    AO_pairs_sparseRepVal.reserve(65536);

    AO_pairs_sparseRepRow.push_back(0);

    for (int i = 0; i < nAO; ++i)
    {
        auto iAO_ID = AO_ID[i];

        int i_begin = AOSparseRepRow[iAO_ID];
        int i_end = AOSparseRepRow[iAO_ID + 1];

        for (int j = i_begin; j < i_end; ++j)
        {
            auto jAO_ID = AO_ID[j];

            int j_begin = AOSparseRepRow[jAO_ID];
            int j_end = AOSparseRepRow[jAO_ID + 1];

            auto max_nnz = std::max(i_end - i_begin, j_end - j_begin);
            AO_pairs_sparseRepCol.resize(nnz + max_nnz);
            AO_pairs_sparseRepVal.resize(nnz + max_nnz);

            auto nnz_tmp = _get_AO_pairs_SparseRep(
                i_end - i_begin,
                AOSparseRepCol + i_begin,
                AOSparseRepVal + i_begin,
                j_end - j_begin,
                AOSparseRepCol + j_begin,
                AOSparseRepVal + j_begin,
                cutoff,
                AO_pairs_sparseRepCol.data() + nnz,
                AO_pairs_sparseRepVal.data() + nnz);

            if (nnz_tmp > 0)
            {
                AO_pairs_sparseRepRow.push_back(AO_pairs_sparseRepRow[nPair] + nnz_tmp);
                nPair += 1;
                nnz += nnz_tmp;
            }
        }
    }

    return nPair;
}

void _fill_AO_pairs_on_Grid(
    IN const int nPair,
    IN const int nAtmGrid,
    IN const int *AO_pairs_sparseRepRow,
    IN const int *AO_pairs_sparseRepCol,
    IN const double *AO_pairs_sparseRepVal,
    IN const int *GridID_2_AtmGridID,
    OUT double *AO_pairs_on_Grid)
{
    Clear(AO_pairs_on_Grid, nPair * nAtmGrid);

    for (int i = 0; i < nPair; ++i)
    {
        int begin = AO_pairs_sparseRepRow[i];
        int end = AO_pairs_sparseRepRow[i + 1];

        for (int j = begin; j < end; ++j)
        {
            int gridID = AO_pairs_sparseRepCol[j];
            double val = AO_pairs_sparseRepVal[j];

            AO_pairs_on_Grid[i * nAtmGrid + GridID_2_AtmGridID[gridID]] = val;
        }
    }
}

/// Lu's Method

std::pair<std::vector<int>, std::vector<double>> _determine_IP_AuxiliaryBasis_via_QR(
    IN const int nPair,
    IN const int nAtmGrid,
    IN const double *AO_pairs_on_Grid,
    IN const double epsilon // 1e-4 ~ 1e-10 ?
)
{
    //// (1) create a Map to invoke eigen

    Eigen::Map<const Eigen::MatrixXd> AO_pairs_on_Grid_EigenMap((const double *)AO_pairs_on_Grid, nPair, nAtmGrid);

    //// (2) perform ColPivHouseholderQR

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(AO_pairs_on_Grid_EigenMap);
    Eigen::MatrixXd R;

    auto P_matrix = qr.colsPermutation();
    auto P = P_matrix.indices();
    R = qr.matrixQR().triangularView<Eigen::Upper>();

    //// (3) determine rank, find n, such that |R(n,n)| < epsilon * |R(0,0)|

    int nAux = 1;

    for (int i = 0; i < nPair; ++i)
    {
        if (fabs(R(i, i)) < epsilon * fabs(R(0, 0)))
        {
            nAux = i;
            break;
        }
    }

    //// (4) fill AuxiliaryBasis

    std::vector<int> IP_Index;
    for (int i = 0; i < nAux; ++i)
    {
        IP_Index.push_back(P(i));
    }

    std::vector<double> AuxiliaryBasis(nAux * nAtmGrid);

    /// create a map

    Eigen::MatrixXd tmp = R.block(0, 0, nAux, nAux).triangularView<Eigen::Upper>().solve(R.block(0, 0, nAux, nAtmGrid)) * P_matrix.inverse();

    return std::make_pair(IP_Index, AuxiliaryBasis);
}

void PBC_ISDF_init(PBC_ISDF **opt,
                   int nao,
                   int natm,
                   int ngrids,
                   double cutoff_aoValue,
                   const int *ao2atomID,
                   const double *aoG,
                   double cutoff_QR)
{
    PBC_ISDF *obj0 = (PBC_ISDF *)malloc(sizeof(PBC_ISDF));
    memset(obj0, 0, sizeof(PBC_ISDF));
    obj0->nao = nao;
    obj0->natm = natm;
    obj0->ngrids = ngrids;
    obj0->cutoff_aoValue = cutoff_aoValue;
    obj0->ao2atomID = ao2atomID;
    obj0->aoG = aoG;
    obj0->cutoff_QR = cutoff_QR;
    *opt = obj0;
}

void PBC_ISDF_del(PBC_ISDF **input)
{
    PBC_ISDF *obj = *input;
    if (obj == NULL)
    {
        return;
    }

    if (obj->voronoi_partition != NULL)
    {
        free(obj->voronoi_partition);
    }
    if (obj->ao_sparse_rep_row != NULL)
    {
        free(obj->ao_sparse_rep_row);
    }
    if (obj->ao_sparse_rep_col != NULL)
    {
        free(obj->ao_sparse_rep_col);
    }
    if (obj->ao_sparse_rep_val != NULL)
    {
        free(obj->ao_sparse_rep_val);
    }
    if (obj->IP_index != NULL)
    {
        free(obj->IP_index);
    }
    if (obj->auxiliary_basis != NULL)
    {
        free(obj->auxiliary_basis);
    }

    free(obj);
    *input = NULL;
}

void PBC_ISDF_build(PBC_ISDF *input)
{
    /// (1) build VoronoiPartition and AOSparseRep

    _VoronoiPartition(input->aoG, input->nao, input->ngrids, (const int *)input->ao2atomID, input->cutoff_aoValue,
                      &input->voronoi_partition, &input->ao_sparse_rep_row, &input->ao_sparse_rep_col, &input->ao_sparse_rep_val);

    /// (2) build buffer

    auto atm2AO = _get_atm_2_non_vanishing_AO(input->voronoi_partition, input->nao, input->natm, input->ao_sparse_rep_row, input->ao_sparse_rep_col);
    auto gridID_2_atmgridID = _get_gridID_2_atmgridID(input->voronoi_partition, input->ngrids, input->natm);
    auto atmgridID_2_grid_ID = _get_atmgridID_2_grid_ID(input->voronoi_partition, input->ngrids, input->natm);

    /// (3) determine IP

    /// (3.1) determine IP for each atom

    std::vector<std::vector<int>> IP_index_per_atm;
    std::vector<std::vector<double>> AuxiliaryBasis_per_atm;
    IP_index_per_atm.resize(input->natm);
    AuxiliaryBasis_per_atm.resize(input->natm);

#pragma omp parallel for schedule(dynamic)
    for (int atmID = 0; atmID < input->natm; ++atmID)
    {

        auto &AO_ID = atm2AO[atmID];
        auto &atm_gridID = atmgridID_2_grid_ID[atmID];
        auto natmGrid = atm_gridID.size();

        std::vector<int> AO_pairs_sparseRepRow;
        std::vector<int> AO_pairs_sparseRepCol;
        std::vector<double> AO_pairs_sparseRepVal;

        int nPair = _get_AO_pairs_SparseRep_on_givan_Atm(
            atmID,
            AO_ID.size(),
            AO_ID.data(),
            input->ao_sparse_rep_row,
            input->ao_sparse_rep_col,
            input->ao_sparse_rep_val,
            input->cutoff_aoValue,
            AO_pairs_sparseRepRow,
            AO_pairs_sparseRepCol,
            AO_pairs_sparseRepVal);

        std::vector<double> AO_pairs_on_Grid(nPair * natmGrid);

        _fill_AO_pairs_on_Grid(
            nPair,
            natmGrid,
            AO_pairs_sparseRepRow.data(),
            AO_pairs_sparseRepCol.data(),
            AO_pairs_sparseRepVal.data(),
            gridID_2_atmgridID.data(),
            AO_pairs_on_Grid.data());

        auto aux_info = _determine_IP_AuxiliaryBasis_via_QR(
            nPair,
            natmGrid,
            AO_pairs_on_Grid.data(),
            input->cutoff_QR);

        IP_index_per_atm[atmID] = aux_info.first; // this index is the index of IP with the given atom
        AuxiliaryBasis_per_atm[atmID] = aux_info.second;
    }

    /// pack IP_index and AuxiliaryBasis

    if (input->IP_index != NULL)
    {
        free(input->IP_index);
    }
    if (input->auxiliary_basis != NULL)
    {
        free(input->auxiliary_basis);
    }

    input->naux = 0;
    for (int i = 0; i < input->natm; ++i)
    {
        input->naux += IP_index_per_atm[i].size();
    }
    input->IP_index = (int *)malloc(sizeof(int) * input->naux);
    Clear(input->IP_index, input->naux);
    input->auxiliary_basis = (double *)malloc(sizeof(double) * input->naux * input->ngrids);
    Clear(input->auxiliary_basis, input->naux * input->ngrids);

    std::vector<int> atm_offset(input->natm, 0);
    for (int i = 1; i < input->natm; ++i)
    {
        atm_offset[i] = atm_offset[i - 1] + IP_index_per_atm[i - 1].size();
    }

#pragma omp parallel for schedule(dynamic)
    for (int atmID = 0; atmID < input->natm; ++atmID)
    {
        auto &IP_index = IP_index_per_atm[atmID];
        auto &AuxiliaryBasis = AuxiliaryBasis_per_atm[atmID];
        auto &map_atmgridID_2_grid_ID = atmgridID_2_grid_ID[atmID];
        auto nGridAtm = map_atmgridID_2_grid_ID.size();

        auto offset = atm_offset[atmID];

        auto naux_tmp = IP_index.size();

        for (int i = 0; i < naux_tmp; ++i)
        {
            input->IP_index[offset + i] = map_atmgridID_2_grid_ID[IP_index[i]];

            for (int j = 0; j < nGridAtm; ++j)
            {
                input->auxiliary_basis[(offset + i) * input->ngrids + map_atmgridID_2_grid_ID[j]] = AuxiliaryBasis[i * nGridAtm + j];
            }
        }
    }
}