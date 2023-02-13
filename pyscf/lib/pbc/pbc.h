/* Copyright 2021 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdint.h>

#if !defined(HAVE_DEFINED_BVKENV_H)
#define HAVE_DEFINED_BVKENV_H
typedef struct {
        // number of primitive cells in bvk-cell
        int ncells;
        // number of repeated images associated to cell.rcut
        int nimgs;
        int nkpts;
        int nbands;
        // nbas of primitive cell
        int nbasp;
        int ncomp;
        // number of grids (or planewaves)
        int nGv;
        // length of kpt_ij_idx
        int kpt_ij_size;
        // indicates how to map basis in bvk-cell to supmol basis
        int *seg_loc;
        int *seg2sh;
        int *ao_loc;
        int *shls_slice;
        // index to get a sbuset of nkpts x nkpts output
        int *kpt_ij_idx;
        double *expLkR;
        double *expLkI;

        // Integral mask of SupMole based on s-function overlap
        int8_t *ovlp_mask;
        // Integral screening condition ~log((ij|ij))/2
        uint8_t *qindex;
        // cutoff for schwarz condtion
        uint32_t cutoff;
        float eta;

        // parameters for ft_ao
        double *Gv;
        double *b;
        int *gxyz;
        int *gs;

        int (*intor)();
} BVKEnvs;
#endif

// supports integrals value between exp(-115/2) - exp((127-115)/2)
// 1e-25 - 4e2
#define CUTOFF_OFFSET 115
