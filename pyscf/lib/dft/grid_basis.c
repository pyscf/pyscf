/*
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
        int ib, i, j;
        int np, nc, atm_id, bas_id;
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

void VXCgen_grid(double *out, double *coords, double *atm_coords,
                 double *radii_table, int natm, int ngrids)
{
        const size_t Ngrids = ngrids;
        int i, j, n;
        double dx, dy, dz, dist;
        double *grid_dist = malloc(sizeof(double) * natm*Ngrids);
        for (i = 0; i < natm; i++) {
                for (n = 0; n < Ngrids; n++) {
                        dx = coords[0*Ngrids+n] - atm_coords[i*3+0];
                        dy = coords[1*Ngrids+n] - atm_coords[i*3+1];
                        dz = coords[2*Ngrids+n] - atm_coords[i*3+2];
                        grid_dist[i*Ngrids+n] = sqrt(dx*dx + dy*dy + dz*dz);
                }
        }

        double *bufs[MAX_THREADS];
#pragma omp parallel default(none) \
        shared(out, grid_dist, atm_coords, radii_table, natm, bufs) \
        private(i, j, n, dx, dy, dz)
{
        int thread_id = omp_get_thread_num();
        double *buf = out;
        if (thread_id != 0) {
                buf = malloc(sizeof(double) * natm*Ngrids);
        }
        bufs[thread_id] = buf;
        for (i = 0; i < natm*Ngrids; i++) {
                buf[i] = 1;
        }
        int ij;
        double fac;
        double *g = malloc(sizeof(double)*Ngrids);
#pragma omp for nowait schedule(static)
        for (ij = 0; ij < natm*natm; ij++) {
                i = ij / natm;
                j = ij % natm;
                if (i <= j) {
                        continue;
                }

                dx = atm_coords[i*3+0] - atm_coords[j*3+0];
                dy = atm_coords[i*3+1] - atm_coords[j*3+1];
                dz = atm_coords[i*3+2] - atm_coords[j*3+2];
                fac = 1 / sqrt(dx*dx + dy*dy + dz*dz);

                for (n = 0; n < Ngrids; n++) {
                        g[n] = grid_dist[i*Ngrids+n] - grid_dist[j*Ngrids+n];
                        g[n] *= fac;
                }
                if (radii_table != NULL) {
                        fac = radii_table[i*natm+j];
                        for (n = 0; n < Ngrids; n++) {
                                g[n] += fac * (1 - g[n]*g[n]);
                        }
                }
                for (n = 0; n < Ngrids; n++) {
                        g[n] = (3 - g[n]*g[n]) * g[n] * .5;
                }
                for (n = 0; n < Ngrids; n++) {
                        g[n] = (3 - g[n]*g[n]) * g[n] * .5;
                }
                for (n = 0; n < Ngrids; n++) {
                        g[n] = (3 - g[n]*g[n]) * g[n] * .5;
                        g[n] *= .5;
                }
                for (n = 0; n < Ngrids; n++) {
                        buf[i*Ngrids+n] *= .5 - g[n];
                        buf[j*Ngrids+n] *= .5 + g[n];
                }
        }
        NPomp_dprod_reduce_inplace(bufs, natm*Ngrids);
        if (thread_id != 0) {
                free(buf);
        }
        free(g);
}
        free(grid_dist);
}

