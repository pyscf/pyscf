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
 * Author: Jan Hermann <dev@hermann.in>
 */

#include <stdlib.h>
#include <math.h>
#include <complex.h>

#define PI 3.14159265358979323846


void T_bare(double *dip, int ld, const double *r) {
    double rx2 = r[0]*r[0];
    double ry2 = r[1]*r[1];
    double rz2 = r[2]*r[2];
    double r2 = rx2+ry2+rz2;
    double r5 = pow(r2, 5./2);
    dip[0] = (r2-3*rx2)/r5;
    dip[ld+1] = (r2-3*ry2)/r5;
    dip[ld*2+2] = (r2-3*rz2)/r5;
    dip[1] = -3*r[0]*r[1]/r5;
    dip[2] = -3*r[0]*r[2]/r5;
    dip[ld+2] = -3*r[1]*r[2]/r5;
    dip[ld] = dip[1];
    dip[ld*2] = dip[2];
    dip[ld*2+1] = dip[2+ld];
}

void T_gg(double *dip, int ld, const double *r, double sigma) {
    double rx2 = r[0]*r[0];
    double ry2 = r[1]*r[1];
    double rz2 = r[2]*r[2];
    double r2 = rx2+ry2+rz2;
    double r5 = pow(r2, 5./2);
    double r_sigma = sqrt(r2)/sigma;
    double r_sigma2 = r_sigma*r_sigma;
    double a1 = -2/sqrt(PI)*r_sigma*exp(-r_sigma2);
    double zeta1 = erf(r_sigma)+a1;
    double zeta2 = a1*(2*r_sigma2);
    T_bare(dip, ld, r);
    int i, j;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            dip[ld*i+j] *= zeta1;
            dip[ld*i+j] -= zeta2*r[i]*r[j]/r5;
        }
    }
}

double damping_fermi(double r, double beta, double a) {
    return 1./(1.+exp(-a*(r/beta-1)));
}

double get_sigma_selfint(double alpha) {
    return pow(sqrt(2./PI)*alpha/3., 1./3);
}

typedef enum {
    BARE = 0,
    FERMI_DIP_GG = 1,
    FERMI_DIP = 2
} Version;

void add_dipole_matrix(Version version, int n, double *dip, const double *coords,
        const double *shift, double cutoff, const double *alpha, const double *R_vdw,
        double beta, double a) {
    double r[3], r_norm, *dip_ij, damping, R_vdw_ij, sigma_ij, sigma_i, sigma_j;
    int ld = 3*n;
    int p, q, i, j, k, l;
    for (i = 0; i < n; i++) {
        p = 3*i;
        for (j = 0; j <= i; j++) {
            if (i == j && !shift) continue;
            q = 3*j;
            for (k = 0; k < 3; k++) {
                r[k] = coords[p+k]-coords[q+k];
                if (shift) r[k] -= shift[k];
            }
            r_norm = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
            if (cutoff && r_norm > cutoff) continue;
            dip_ij = dip+ld*p+q;
            if (R_vdw) R_vdw_ij = R_vdw[i]+R_vdw[j];
            if (alpha) {
                sigma_i = get_sigma_selfint(alpha[i]);
                sigma_j = get_sigma_selfint(alpha[j]);
                sigma_ij = sqrt(sigma_i*sigma_i+sigma_j*sigma_j);
            }
            switch (version) {
                case BARE:
                    T_bare(dip_ij, ld, r);
                    break;
                case FERMI_DIP_GG:
                    T_gg(dip_ij, ld, r, sigma_ij);
                    damping = 1.-damping_fermi(r_norm, beta*R_vdw_ij, a);
                    for (k = 0; k < 3; k++) {
                        for (l = 0; l < 3; l++) {
                            dip_ij[ld*k+l] *= damping;
                        }
                    }
                    break;
                case FERMI_DIP:
                    T_bare(dip_ij, ld, r);
                    damping = damping_fermi(r_norm, beta*R_vdw_ij, a);
                    for (k = 0; k < 3; k++) {
                        for (l = 0; l < 3; l++) {
                            dip_ij[ld*k+l] *= damping;
                        }
                    }
                    break;
            }
            dip[ld*q+p] = dip[ld*p+q];
            dip[ld*q+p+1] = dip[ld*(p+1)+q];
            dip[ld*q+p+2] = dip[ld*(p+2)+q];
            dip[ld*(q+1)+p] = dip[ld*p+q+1];
            dip[ld*(q+1)+p+1] = dip[ld*(p+1)+q+1];
            dip[ld*(q+1)+p+2] = dip[ld*(p+2)+q+1];
            dip[ld*(q+2)+p] = dip[ld*p+q+2];
            dip[ld*(q+2)+p+1] = dip[ld*(p+1)+q+2];
            dip[ld*(q+2)+p+2] = dip[ld*(p+2)+q+2];
        }
    }
}
