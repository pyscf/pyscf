/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
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
#include <string.h>
#include <math.h>
#include "cint.h"
#include "config.h"
#include "gto/grid_ao_drv.h"
#include "np_helper/np_helper.h"

#define MAX_THREADS     256

void VXCnr_ao_screen(unsigned char *non0table, double *coords, int ngrids,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
        int i, j;
        int np, nc, atm_id;
        size_t bas_id, ib;
        double rr, arr, maxc;
        double logcoeff[NPRIMAX];
        double dr[3];
        double *p_exp, *pcoeff, *ratm;

        for (bas_id = 0; bas_id < nbas; bas_id++) {
                np = bas[NPRIM_OF];
                nc = bas[NCTR_OF ];
                p_exp = env + bas[PTR_EXP];
                pcoeff = env + bas[PTR_COEFF];
                atm_id = bas[ATOM_OF];
                ratm = env + atm[atm_id*ATM_SLOTS+PTR_COORD];

                for (j = 0; j < np; j++) {
                        maxc = 0;
                        for (i = 0; i < nc; i++) {
                                maxc = MAX(maxc, fabs(pcoeff[i*np+j]));
                        }
                        logcoeff[j] = log(maxc);
                }

                for (ib = 0; ib < nblk; ib++) {
                        for (i = ib*BLKSIZE; i < MIN(ngrids, (ib+1)*BLKSIZE); i++) {
                                dr[0] = coords[0*ngrids+i] - ratm[0];
                                dr[1] = coords[1*ngrids+i] - ratm[1];
                                dr[2] = coords[2*ngrids+i] - ratm[2];
                                rr = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
                                for (j = 0; j < np; j++) {
                                        arr = p_exp[j] * rr;
                                        if (arr-logcoeff[j] < EXPCUTOFF) {
                                                non0table[ib*nbas+bas_id] = 1;
                                                goto next_blk;
                                        }
                                }
                        }
                        non0table[ib*nbas+bas_id] = 0;
next_blk:;
                }
                bas += BAS_SLOTS;
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
                                g[n] = (3 - g[n]*g[n]) * g[n] * .5;
                                g[n] *= .5;
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

