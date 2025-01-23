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

// 1k grids per block
#define GRIDS_BLOCK     512
#define RCUT_LKO        5.0
#define MC_LKO          12
#define ALIGNMENT       8

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
        double *_buf = malloc(sizeof(double) * ((natm*2+1)*GRIDS_BLOCK + ALIGNMENT));
        double *buf = (double *)((uintptr_t)(_buf + ALIGNMENT - 1) & (-(uintptr_t)(ALIGNMENT*8)));
        double *g = buf + natm * GRIDS_BLOCK;
        double *grid_dist = g + GRIDS_BLOCK;
        size_t ig0, n, ngs;
        double fac, s;
#pragma omp for nowait schedule(static)
        for (ig0 = 0; ig0 < Ngrids; ig0 += GRIDS_BLOCK) {
                ngs = MIN(Ngrids-ig0, GRIDS_BLOCK);
                for (i = 0; i < natm; i++) {
#pragma GCC ivdep
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
#pragma GCC ivdep
                        for (n = 0; n < ngs; n++) {
                                s = g[n];
                                s = (3 - s*s) * s * .5;
                                s = (3 - s*s) * s * .5;
                                s = ((3 - s*s) * s * .5) * .5;
                                buf[i*GRIDS_BLOCK+n] *= .5 - s;
                                buf[j*GRIDS_BLOCK+n] *= .5 + s;
                        }
                } }

                for (i = 0; i < natm; i++) {
                        for (n = 0; n < ngs; n++) {
                                out[i*Ngrids+ig0+n] = buf[i*GRIDS_BLOCK+n];
                        }
                }
        }
        free(_buf);
}
        free(atom_dist);
}


inline double _lko_sat_func(double x) {
    int m;
    double tot = 0;
    x = x / RCUT_LKO;
    double xm = 1;
    for (m = 1; m <= MC_LKO; m++) {
        // tot += pow(x, m) / m;
        xm *= x;
        tot += xm / m;
    }
    return RCUT_LKO * (1 - exp(-tot));
}

inline double _lko_sat_deriv(double x) {
    int m;
    double tot = 0;
    double dtot = 0;
    x = x / RCUT_LKO;
    double xm = 1;
    for (m = 1; m <= MC_LKO; m++) {
        // tot += pow(x, m) / m;
        // dtot += pow(x, m-1);
        dtot += xm;
        xm *= x;
        tot += xm / m;
    }
    return exp(-tot) * dtot;
}


void VXCgen_grid_lko(double *out, double *coords, double *atm_coords,
                     double *radii_table, int natm, int ngrids)
{
    const size_t Ngrids = ngrids;
    int i, j;
    double dx, dy, dz;
    double *atom_dist = malloc(sizeof(double) * natm*natm);
#pragma omp parallel private(i, j, dx, dy, dz)
{
    int ij;
    const int num_ij = natm * (natm + 1) / 2;
#pragma omp for nowait schedule(static)
    for (ij = 0; ij < num_ij; ij++) {

    //for (i = 0; i < natm; i++) {
        //for (j = 0; j < i; j++) {
            i = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
            j = ij - i*(i+1)/2;
            dx = atm_coords[i*3+0] - atm_coords[j*3+0];
            dy = atm_coords[i*3+1] - atm_coords[j*3+1];
            dz = atm_coords[i*3+2] - atm_coords[j*3+2];
            atom_dist[i*natm+j] = 1 / _lko_sat_func(sqrt(dx*dx + dy*dy + dz*dz));
        //}
    }
}

#pragma omp parallel private(i, j, dx, dy, dz)
{
    size_t ig_start, ig_end;
    NPomp_split(&ig_start, &ig_end, (size_t)ngrids);
    double *grid_dist = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *buf = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *min_dist_i = malloc(sizeof(double) * natm);
    double *min_dist_n = malloc(sizeof(double) * GRIDS_BLOCK);
    double *g = malloc(sizeof(double) * GRIDS_BLOCK);
    size_t ig0, n, ngs;
    double fac, s;
    double max_min_dist;
// #pragma omp for nowait schedule(static)
    for (ig0 = ig_start; ig0 < ig_end; ig0 += GRIDS_BLOCK) {
        ngs = MIN(ig0 + GRIDS_BLOCK, ig_end) - ig0;
        for (n = 0; n < ngs; n++) {
            min_dist_n[n] = 1e10;
        }
        for (i = 0; i < natm; i++) {
            min_dist_i[i] = 1e10;
            // this loop cannot be done with ivdep because of min_dist_i
            for (n = 0; n < ngs; n++) {
                dx = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                dy = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                dz = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
                grid_dist[i*GRIDS_BLOCK+n] = sqrt(dx*dx + dy*dy + dz*dz);
                buf[i*GRIDS_BLOCK+n] = 1;
                // for screening
                min_dist_i[i] = MIN(grid_dist[i*GRIDS_BLOCK+n], min_dist_i[i]);
                min_dist_n[n] = MIN(grid_dist[i*GRIDS_BLOCK+n], min_dist_n[n]);
            }
        }
        max_min_dist = 0.0;
        for (n = 0; n < ngs; n++) {
            max_min_dist = MAX(max_min_dist, min_dist_n[n]);
        }

        for (i = 0; i < natm; i++) {
        for (j = 0; j < i; j++) {
            // for screening
            if ((min_dist_i[i] > max_min_dist + RCUT_LKO) &&
                (min_dist_i[j] > max_min_dist + RCUT_LKO)) {
                continue;
            }
            fac = atom_dist[i*natm+j];
#pragma GCC ivdep
            for (n = 0; n < ngs; n++) {
                g[n] = (grid_dist[i*GRIDS_BLOCK+n] -
                    grid_dist[j*GRIDS_BLOCK+n]) * fac;
                g[n] = MAX(-1, g[n]);
                g[n] = MIN(1, g[n]);
            }
            if (radii_table != NULL) {
                fac = radii_table[i*natm+j];
#pragma GCC ivdep
                for (n = 0; n < ngs; n++) {
                    g[n] += fac * (1 - g[n]*g[n]);
                }
            }
            /*for (n = 0; n < ngs; n++) {
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
            }*/
#pragma GCC ivdep
            for (n = 0; n < ngs; n++) {
                s = g[n];
                s = (3 - s*s) * s * .5;
                s = (3 - s*s) * s * .5;
                s = ((3 - s*s) * s * .5) * .5;
                buf[i*GRIDS_BLOCK+n] *= .5 - s;
                buf[j*GRIDS_BLOCK+n] *= .5 + s;
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
    free(min_dist_i);
    free(min_dist_n);
}
    free(atom_dist);
}

void VXCgen_grid_lko_deriv(double *out, double *dw, double *coords, double *atm_coords,
                           double *radii_table, int natm, int ngrids, int ia_p)
{
    const size_t Ngrids = ngrids;
    int i, j;
    double dx, dy, dz, dr;
    double sat_deriv;
    double *atom_dist = malloc(sizeof(double) * natm*natm);
    double *dadx = malloc(sizeof(double) * natm*natm);
    double *dady = malloc(sizeof(double) * natm*natm);
    double *dadz = malloc(sizeof(double) * natm*natm);
    double *outx = out + 1*natm*ngrids;
    double *outy = out + 2*natm*ngrids;
    double *outz = out + 3*natm*ngrids;
#pragma omp parallel private(i, j, dx, dy, dz, dr, sat_deriv)
{   
#pragma omp for
    for (i = 0; i < natm; i++) {
        for (j = 0; j < i; j++) {
            dx = atm_coords[i*3+0] - atm_coords[j*3+0];
            dy = atm_coords[i*3+1] - atm_coords[j*3+1];
            dz = atm_coords[i*3+2] - atm_coords[j*3+2];
            dr = sqrt(dx*dx + dy*dy + dz*dz);
            atom_dist[i*natm+j] = 1 / _lko_sat_func(dr);
            sat_deriv = _lko_sat_deriv(dr) * pow(atom_dist[i*natm+j], 3);
            dadx[i*natm+j] = -dx * sat_deriv;
            dady[i*natm+j] = -dy * sat_deriv;
            dadz[i*natm+j] = -dz * sat_deriv;
        }
    }
}

#pragma omp parallel private(i, j, dx, dy, dz, dr)
{
    double *grid_dist = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *buf = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *bufx = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *bufy = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *bufz = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *min_dist_i = malloc(sizeof(double) * natm);
    double *min_dist_n = malloc(sizeof(double) * GRIDS_BLOCK);
    double *g = malloc(sizeof(double) * GRIDS_BLOCK);
    double *dg = malloc(sizeof(double) * GRIDS_BLOCK);
    size_t ig0, n, ngs;
    double fac, dfacx, dfacy, dfacz, tmp;
    double max_min_dist;
#pragma omp for nowait schedule(static)
    for (ig0 = 0; ig0 < Ngrids; ig0 += GRIDS_BLOCK) {
        ngs = MIN(Ngrids-ig0, GRIDS_BLOCK);
        for (n = 0; n < ngs; n++) {
            min_dist_n[n] = 1e10;
        }
        for (i = 0; i < natm; i++) {
            min_dist_i[i] = 1e10;
            for (n = 0; n < ngs; n++) {
                dx = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                dy = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                dz = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
                grid_dist[i*GRIDS_BLOCK+n] = sqrt(dx*dx + dy*dy + dz*dz);
                buf[i*GRIDS_BLOCK+n] = 1;
                bufx[i*GRIDS_BLOCK+n] = 0;
                bufy[i*GRIDS_BLOCK+n] = 0;
                bufz[i*GRIDS_BLOCK+n] = 0;
                // for screening
                min_dist_i[i] = MIN(grid_dist[i*GRIDS_BLOCK+n], min_dist_i[i]);
                min_dist_n[n] = MIN(grid_dist[i*GRIDS_BLOCK+n], min_dist_n[n]);
            }
        }
        max_min_dist = 0.0;
        for (n = 0; n < ngs; n++) {
            max_min_dist = MAX(max_min_dist, min_dist_n[n]);
        }

        for (i = 0; i < natm; i++) {
        for (j = 0; j < i; j++) {
            // for screening
            if ((min_dist_i[i] > max_min_dist + RCUT_LKO) &&
                (min_dist_i[j] > max_min_dist + RCUT_LKO)) {
                continue;
            }
            fac = atom_dist[i*natm+j];
            for (n = 0; n < ngs; n++) {
                g[n] = (grid_dist[i*GRIDS_BLOCK+n] -
                    grid_dist[j*GRIDS_BLOCK+n]) * fac;
                g[n] = MAX(-1, g[n]);
                g[n] = MIN(1, g[n]);
                dg[n] = 1.0;
                if (g[n] == -1 || g[n] == 1) {
                    dg[n] = 0.0;
                }
            }
            if (radii_table != NULL) {
                fac = radii_table[i*natm+j];
                for (n = 0; n < ngs; n++) {
                    dg[n] -= 2 * fac * g[n];
                    g[n] += fac * (1 - g[n]*g[n]);
                }
            }
            for (n = 0; n < ngs; n++) {
                dg[n] *= 1.5 * (1 - g[n]*g[n]);
                g[n] = (3 - g[n]*g[n]) * g[n] * .5;
            }
            for (n = 0; n < ngs; n++) {
                dg[n] *= 1.5 * (1 - g[n]*g[n]);
                g[n] = (3 - g[n]*g[n]) * g[n] * .5;
            }
            for (n = 0; n < ngs; n++) {
                dg[n] *= 0.75 * (1 - g[n]*g[n]);
                g[n] = ((3 - g[n]*g[n]) * g[n] * .5) * .5;
            }
            fac = atom_dist[i*natm+j];
            dfacx = dadx[i*natm+j];
            dfacy = dady[i*natm+j];
            dfacz = dadz[i*natm+j];
            for (n = 0; n < ngs; n++) {
                tmp = dw[j*Ngrids+ig0+n] * dg[n] / (.5 + g[n] + 1e-200);
                tmp-= dw[i*Ngrids+ig0+n] * dg[n] / (.5 - g[n] + 1e-200);
                tmp *= (grid_dist[i*GRIDS_BLOCK+n] -
                        grid_dist[j*GRIDS_BLOCK+n]);
                bufx[i*GRIDS_BLOCK+n] += tmp * dfacx;
                bufy[i*GRIDS_BLOCK+n] += tmp * dfacy;
                bufz[i*GRIDS_BLOCK+n] += tmp * dfacz;
                bufx[j*GRIDS_BLOCK+n] -= tmp * dfacx;
                bufy[j*GRIDS_BLOCK+n] -= tmp * dfacy;
                bufz[j*GRIDS_BLOCK+n] -= tmp * dfacz;

                tmp = dw[j*Ngrids+ig0+n] * dg[n] / (.5 + g[n] + 1e-200);
                tmp-= dw[i*Ngrids+ig0+n] * dg[n] / (.5 - g[n] + 1e-200);
                tmp *= fac;
                if (i != ia_p) {
                    dx = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                    dy = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                    dz = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
                    dr = grid_dist[i*GRIDS_BLOCK+n];
                    dx = tmp * dx / (dr + 1e-200);
                    dy = tmp * dy / (dr + 1e-200);
                    dz = tmp * dz / (dr + 1e-200);
                    bufx[i*GRIDS_BLOCK+n] -= dx;
                    bufy[i*GRIDS_BLOCK+n] -= dy;
                    bufz[i*GRIDS_BLOCK+n] -= dz;
                    bufx[ia_p*GRIDS_BLOCK+n] += dx;
                    bufy[ia_p*GRIDS_BLOCK+n] += dy;
                    bufz[ia_p*GRIDS_BLOCK+n] += dz;
                }
                if (j != ia_p) {
                    dx = coords[0*Ngrids+ig0+n] - atm_coords[j*3+0];
                    dy = coords[1*Ngrids+ig0+n] - atm_coords[j*3+1];
                    dz = coords[2*Ngrids+ig0+n] - atm_coords[j*3+2];
                    dr = grid_dist[j*GRIDS_BLOCK+n];
                    dx = tmp * dx / (dr + 1e-200);
                    dy = tmp * dy / (dr + 1e-200);
                    dz = tmp * dz / (dr + 1e-200);
                    bufx[j*GRIDS_BLOCK+n] += dx;
                    bufy[j*GRIDS_BLOCK+n] += dy;
                    bufz[j*GRIDS_BLOCK+n] += dz;
                    bufx[ia_p*GRIDS_BLOCK+n] -= dx;
                    bufy[ia_p*GRIDS_BLOCK+n] -= dy;
                    bufz[ia_p*GRIDS_BLOCK+n] -= dz;
                }
            }
        } }

        for (i = 0; i < natm; i++) {
            for (n = 0; n < ngs; n++) {
                out[i*Ngrids+ig0+n] = buf[i*GRIDS_BLOCK+n];
                outx[i*Ngrids+ig0+n] = bufx[i*GRIDS_BLOCK+n];
                outy[i*Ngrids+ig0+n] = bufy[i*GRIDS_BLOCK+n];
                outz[i*Ngrids+ig0+n] = bufz[i*GRIDS_BLOCK+n];
            }
        }
    }
    free(g);
    free(dg);
    free(buf);
    free(grid_dist);
    free(bufx);
    free(bufy);
    free(bufz);
    free(min_dist_i);
    free(min_dist_n);
}
    free(atom_dist);
    free(dadx);
    free(dady);
    free(dadz);
}

