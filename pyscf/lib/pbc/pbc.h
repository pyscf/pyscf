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
        int *sh_loc;
        int *ao_loc;
        // Map from supmol._bas to [bvk_cell-id, rs_basis-id, image-id]
        int *bas_map;
        int *shls_slice;
        // index to get a sbuset of nkpts x nkpts output
        int *kpt_ij_idx;
        double *expLkR;
        double *expLkI;

        // Integral mask of SupMole based on s-function overlap
        int8_t *ovlp_mask;
        // Integral screening condition
        double *q_cond;
        // cutoff for schwarz condtion
        double cutoff;

        // parameters for ft_ao
        double *Gv;
        double *b;
        int *gxyz;
        int *gs;

        int (*intor)();
} BVKEnvs;
#endif
