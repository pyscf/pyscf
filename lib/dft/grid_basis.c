/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cint.h"

// 128s42p21d12f8g6h4i3j 
#define NCTR_CART      128
//  72s24p14d10f8g6h5i4j 
#define NCTR_SPH        72
#define NPRIMAX         64
#define BLKSIZE         96
#define EXPCUTOFF       50  // 1e-22
#define MIN(X,Y)        ((X)<(Y)?(X):(Y))
#define MAX(X,Y)        ((X)>(Y)?(X):(Y))

void VXCnr_ao_screen(signed char *non0table, double *coord,
                     int ngrids, int blksize,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        const int nblk = (ngrids+blksize-1) / blksize;
        int ib, i, j;
        int np, nc, atm_id, bas_id;
        double rr, arr, maxc;
        double logcoeff[NPRIMAX];
        double dr[3];
        double *p_exp, *pcoeff, *pcoord, *ratm;

        memset(non0table, 0, sizeof(signed char) * nblk*nbas);

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

                pcoord = coord;
                for (ib = 0; ib < nblk; ib++) {
                        for (i = 0; i < MIN(ngrids-ib*blksize, blksize); i++) {
                                dr[0] = pcoord[i*3+0] - ratm[0];
                                dr[1] = pcoord[i*3+1] - ratm[1];
                                dr[2] = pcoord[i*3+2] - ratm[2];
                                rr = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
                                for (j = 0; j < np; j++) {
                                        arr = p_exp[j] * rr;
                                        if (arr-logcoeff[j] < EXPCUTOFF) {
                                                non0table[ib*nbas+bas_id] = 1;
                                                goto next_blk;
                                        }
                                }
                        }
next_blk:
                        pcoord += blksize*3;
                }
                bas += BAS_SLOTS;
        }
}

void VXCgen_grid(double *out, double *coords, double *atm_coords,
                 double *radii_table, int natm, int ngrids)
{
        int i, j, n;
        double dx, dy, dz, dist;
        double *grid_dist = malloc(sizeof(double) * natm*ngrids);
        for (i = 0; i < natm; i++) {
                for (n = 0; n < ngrids; n++) {
                        dx = coords[n*3+0] - atm_coords[i*3+0];
                        dy = coords[n*3+1] - atm_coords[i*3+1];
                        dz = coords[n*3+2] - atm_coords[i*3+2];
                        grid_dist[i*ngrids+n] = sqrt(dx*dx + dy*dy + dz*dz);
                }
        }

        for (n = 0; n < natm*ngrids; n++) {
                out[n] = 1;
        }

#pragma omp parallel default(none) \
        shared(out, grid_dist, atm_coords, radii_table, natm, ngrids) \
        private(i, j, n, dx, dy, dz)
{
        double *buf = malloc(sizeof(double) * natm*ngrids);
        for (i = 0; i < natm*ngrids; i++) {
                buf[i] = 1;
        }
        int ij;
        double fac;
        double g[ngrids];
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

                for (n = 0; n < ngrids; n++) {
                        g[n] = grid_dist[i*ngrids+n] - grid_dist[j*ngrids+n];
                        g[n] *= fac;
                }
                if (radii_table != NULL) {
                        fac = radii_table[i*natm+j];
                        for (n = 0; n < ngrids; n++) {
                                g[n] += fac * (1 - g[n]*g[n]);
                        }
                }
                for (n = 0; n < ngrids; n++) {
                        g[n] = (3 - g[n]*g[n]) * g[n] * .5;
                }
                for (n = 0; n < ngrids; n++) {
                        g[n] = (3 - g[n]*g[n]) * g[n] * .5;
                }
                for (n = 0; n < ngrids; n++) {
                        g[n] = (3 - g[n]*g[n]) * g[n] * .5;
                        g[n] *= .5;
                }
                for (n = 0; n < ngrids; n++) {
                        buf[i*ngrids+n] *= .5 - g[n];
                        buf[j*ngrids+n] *= .5 + g[n];
                }
        }
#pragma omp critical
        for (i = 0; i < natm*ngrids; i++) {
                out[i] *= buf[i];
        }
        free(buf);
}
        free(grid_dist);
}

