/* Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
  
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

#include <stdlib.h>
#include <math.h>
#include "cint.h"
#include "config.h"
#include "gto/grid_ao_drv.h"
#include "np_helper/np_helper.h"

void VXC_screen_index(uint8_t *screen_index, int nbins, double cutoff,
                      double *coords, int ngrids,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        double scale = -nbins / log(MIN(cutoff, .1));
#pragma omp parallel
{
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
        int i, j;
        int np, nc, atm_id, ng0, ng1, dg;
        size_t bas_id, ib;
        double min_exp, log_coeff, maxc, arr, arr_min, si;
        double dx, dy, dz, atom_x, atom_y, atom_z;
        double *p_exp, *pcoeff, *ratm;
        double rr[BLKSIZE];
        double *coordx = coords;
        double *coordy = coords + ngrids;
        double *coordz = coords + ngrids * 2;

#pragma omp for nowait schedule(static)
        for (bas_id = 0; bas_id < nbas; bas_id++) {
                np = bas[NPRIM_OF+bas_id*BAS_SLOTS];
                nc = bas[NCTR_OF +bas_id*BAS_SLOTS];
                p_exp = env + bas[PTR_EXP+bas_id*BAS_SLOTS];
                pcoeff = env + bas[PTR_COEFF+bas_id*BAS_SLOTS];
                atm_id = bas[ATOM_OF+bas_id*BAS_SLOTS];
                ratm = env + atm[atm_id*ATM_SLOTS+PTR_COORD];
                atom_x = ratm[0];
                atom_y = ratm[1];
                atom_z = ratm[2];

                maxc = 0;
                min_exp = 1e9;
                for (j = 0; j < np; j++) {
                        min_exp = MIN(min_exp, p_exp[j]);
                        for (i = 0; i < nc; i++) {
                                maxc = MAX(maxc, fabs(pcoeff[i*np+j]));
                        }
                }
                log_coeff = log(maxc);

                for (ib = 0; ib < nblk; ib++) {
                        ng0 = ib * BLKSIZE;
                        ng1 = MIN(ngrids, (ib+1)*BLKSIZE);
                        dg = ng1 - ng0;
                        for (i = 0; i < dg; i++) {
                                dx = coordx[ng0+i] - atom_x;
                                dy = coordy[ng0+i] - atom_y;
                                dz = coordz[ng0+i] - atom_z;
                                rr[i] = dx*dx + dy*dy + dz*dz;
                        }
                        arr_min = 1e9;
                        for (i = 0; i < dg; i++) {
                                arr = min_exp * rr[i] - log_coeff;
                                arr_min = MIN(arr_min, arr);
                        }
                        si = nbins - arr_min * scale;
                        if (si <= 0) {
                                screen_index[ib*nbas+bas_id] = 0;
                        } else {
                                screen_index[ib*nbas+bas_id] = (int8_t)(si + 1);
                        }
                }
        }
}
}

// 1k grids per block
#define GRIDS_BLOCK     512

void VXCgen_grid(double *out, double *coords, double *atm_coords,
                 double *radii_table, int natm, int ngrids)
{
        const size_t Ngrids = ngrids;
        int i, j;
        double dx, dy, dz;
        double *atom_dist = malloc(sizeof(double) * natm*natm);
        for (i = 0; i < natm; i++) {
                for (j = 0; j < i; j++) {
                        dx = atm_coords[i*3+0] - atm_coords[j*3+0];
                        dy = atm_coords[i*3+1] - atm_coords[j*3+1];
                        dz = atm_coords[i*3+2] - atm_coords[j*3+2];
                        atom_dist[i*natm+j] = 1 / sqrt(dx*dx + dy*dy + dz*dz);
                }
        }

#pragma omp parallel private(i, j, dx, dy, dz)
{
        double *grid_dist = malloc(sizeof(double) * natm*GRIDS_BLOCK);
        double *buf = malloc(sizeof(double) * natm*GRIDS_BLOCK);
        double *g = malloc(sizeof(double) * GRIDS_BLOCK);
        size_t ig0, n, ngs;
        double fac;
#pragma omp for nowait schedule(static)
        for (ig0 = 0; ig0 < Ngrids; ig0 += GRIDS_BLOCK) {
                ngs = MIN(Ngrids-ig0, GRIDS_BLOCK);
                for (i = 0; i < natm; i++) {
                for (n = 0; n < ngs; n++) {
                        dx = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                        dy = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                        dz = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
                        grid_dist[i*GRIDS_BLOCK+n] = sqrt(dx*dx + dy*dy + dz*dz);
                        buf[i*GRIDS_BLOCK+n] = 1;
                } }

                for (i = 0; i < natm; i++) {
                for (j = 0; j < i; j++) {

                        fac = atom_dist[i*natm+j];
                        for (n = 0; n < ngs; n++) {
                                g[n] = (grid_dist[i*GRIDS_BLOCK+n] -
                                        grid_dist[j*GRIDS_BLOCK+n]) * fac;
                        }
                        if (radii_table != NULL) {
                                fac = radii_table[i*natm+j];
                                for (n = 0; n < ngs; n++) {
                                        g[n] += fac * (1 - g[n]*g[n]);
                                }
                        }
                        for (n = 0; n < ngs; n++) {
                                g[n] = (3 - g[n]*g[n]) * g[n] * .5;
                        }
                        for (n = 0; n < ngs; n++) {
                                g[n] = (3 - g[n]*g[n]) * g[n] * .5;
                        }
                        for (n = 0; n < ngs; n++) {
                                g[n] = ((3 - g[n]*g[n]) * g[n] * .5) * .5;
                        }
                        for (n = 0; n < ngs; n++) {
                                buf[i*GRIDS_BLOCK+n] *= .5 - g[n];
                                buf[j*GRIDS_BLOCK+n] *= .5 + g[n];
                        }
                } }

                for (i = 0; i < natm; i++) {
                        for (n = 0; n < ngs; n++) {
                                out[i*Ngrids+ig0+n] = buf[i*GRIDS_BLOCK+n];
                        }
                }
        }
        free(g);
        free(buf);
        free(grid_dist);
}
        free(atom_dist);
}

