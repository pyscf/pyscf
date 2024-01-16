#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include "config.h"

#include <Eigen/Dense>
#include <Eigen/QR>

#include "pbc_isdf.h"

#define IN
#define OUT
#define BUFFER

#define OMP_THREAD_LABEL omp_get_thread_num()
#define OMP_NUM_OF_THREADS std::atoi(getenv("OMP_NUM_THREADS"))

template <typename Scalar>
void Clear(Scalar *_Input, size_t _n)
{
    static_assert(std::is_pod<Scalar>::value, "In Clear typename Clear is not POD!");
    memset((void *)_Input, 0, _n * sizeof(Scalar));
}

template <>
inline void Clear(std::complex<double> *_Input, size_t _n)
{
    memset((void *)_Input, 0, _n * sizeof(std::complex<double>));
}

template <>
inline void Clear(std::complex<float> *_Input, size_t _n)
{
    memset((void *)_Input, 0, _n * sizeof(std::complex<float>));
}

/// declare functions

void _VoronoiPartition_Only(IN const double *AOonGrids, // Row-major, (nAO, nGrids)
                            IN const int nAO,
                            IN const int nGrids,
                            IN const int *AO2AtomID,
                            IN const double cutoff, // < 1e-12 recommended
                            OUT int **VoronoiPartition);

void _VoronoiPartition(IN const double *AOonGrids, // Row-major, (nAO, nGrids)
                       IN const int nAO,
                       IN const int nGrids,
                       IN const int *AO2AtomID,
                       IN const double cutoff,     // < 1e-12 recommended
                       OUT int **VoronoiPartition, // for each point on the grid, the atom ID of the Voronoi cell it belongs to
                       OUT int **AOSparseRepRow,
                       OUT int **AOSparseRepCol,
                       OUT double **AOSparseRepVal);

// #define PRINT_DEBUG