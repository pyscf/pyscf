#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <config.h>

#define MODULO(x, n) ((x % n + n) % n)


static double smallest_dist(double x, double n)
{
    double x_frac;
    x_frac  = x / n;
    x_frac -= floor(x_frac);
    if (x_frac > .5) x_frac = 1. - x_frac;
    return x_frac * n;
}



void kmeans_orth(int* centroids, double* w, int naux,
                 double* coords, double* a, int* mesh,
                 double* Ls, int nL,
                 int max_cycle)
{
    const int nx = mesh[0];
    const int ny = mesh[1];
    const int nz = mesh[2];
    const int nyz = ny * nz;
    const int ng = nx * nyz;
    const double ax = a[0];
    const double ay = a[4];
    const double az = a[8];
    int* centroids1 = (int*) malloc(naux * sizeof(int));
    int* r2c = (int*) malloc(ng * sizeof(int));

    int iter, i, j, ix, iy, iz;
    int ns = (int) floor(pow(ng / naux, 1./3.));
    for (ix = 0, i = 0; ix < nx; ix += ns)
    for (iy = 0; iy < ny; iy += ns)
    for (iz = 0; iz < nz && i < naux; iz += ns, i++) {
        centroids[i] = ix * nyz + iy * nz + iz;
    }

    for (iter = 0; iter < max_cycle; iter++) {
        #pragma omp parallel private(i, j, ix, iy, iz)
        {
            double rmin, rr, rx, ry, rz;
            double r[3];
            double *ri, *rc;

            #pragma omp for schedule(static)
            for (i = 0; i < ng; i++) {
                ri = &coords[i*3];
                rmin = 1e16;
                for (j = 0; j < naux; j++) {
                    rc = &coords[centroids[j]*3];
                    rx = smallest_dist(rc[0] - ri[0], ax);
                    ry = smallest_dist(rc[1] - ri[1], ay);
                    rz = smallest_dist(rc[2] - ri[2], az);
                    rr = rx*rx + ry*ry + rz*rz;
                    if (rr < rmin) {
                        rmin = rr;
                        r2c[i] = j;
                    }
                }
            }

            #pragma omp for schedule(static)
            for (j = 0; j < naux; j++) {
                memset(r, 0, 3*sizeof(double));
                double w_sum = 0;
                for (i = 0; i < ng; i++) {
                    if (r2c[i] == j) {
                        ri = &coords[i*3];
                        r[0] += ri[0] * w[i];
                        r[1] += ri[1] * w[i];
                        r[2] += ri[2] * w[i];
                        w_sum += w[i];
                    }
                }
                r[0] /= w_sum;
                r[1] /= w_sum;
                r[2] /= w_sum;

                ix = MODULO((int) rint(r[0] / ax * nx), nx);
                iy = MODULO((int) rint(r[1] / ay * ny), ny);
                iz = MODULO((int) rint(r[2] / az * nz), nz);
                centroids1[j] = iz + iy * nz + ix * nyz;
            }
        } //omp parallel

        bool converged = true;
        for (j = 0; j < naux; j++) {
            if (centroids1[j] != centroids[j]) {
                converged = false;
                break;
            }
        }
        if (converged) {
            break;
        }
        memcpy(centroids, centroids1, naux*sizeof(int));
    }

    free(centroids1);
    free(r2c);
}
