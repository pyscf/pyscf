#include <stdlib.h>
#include <complex.h>
#define NCTRMAX         72

static void shift_bas(double *xyz, int *ptr_coords, double *L, int nxyz, double *env)
{
        int i, p;
        for (i = 0; i < nxyz; i++) {
                p = ptr_coords[i];
                env[p+0] = xyz[i*3+0] + L[0];
                env[p+1] = xyz[i*3+1] + L[1];
                env[p+2] = xyz[i*3+2] + L[2];
        }
}

static void axpy_s1(double complex **out, double complex *in,
                    double complex *exp_Lk, int nkpts, size_t off,
                    size_t nGv, int ni, int nj, int ip, int di, int dj)
{
        int i, j, n, ik;
        double complex *pin, *pout;
        for (ik = 0; ik < nkpts; ik++) {
        for (j = 0; j < dj; j++) {
        for (i = 0; i < di; i++) {
                pout = out[ik] + off + (j*ni+i) * nGv;
                pin  = in + (j*di+i) * nGv;
                for (n = 0; n < nGv; n++) {
                        pout[n] += pin[n] * exp_Lk[ik];
                }
        } } }
}
static void axpy_igtj(double complex **out, double complex *in,
                      double complex *exp_Lk, int nkpts, size_t off,
                      size_t nGv, int ni, int nj, int ip, int di, int dj)
{
        const size_t ip1 = ip + 1;
        int i, j, n, ik;
        double complex *pin, *pout;
        for (ik = 0; ik < nkpts; ik++) {
                pout = out[ik] + off;
                for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                                pin = in + (j*di+i) * nGv;
                                for (n = 0; n < nGv; n++) {
                                        pout[j*nGv+n] += pin[n] * exp_Lk[ik];
                                }
                        }
                        pout += (ip1 + i) * nGv;
                }
        }
}
static void axpy_ieqj(double complex **out, double complex *in,
                      double complex *exp_Lk, int nkpts, size_t off,
                      size_t nGv, int ni, int nj, int ip, int di, int dj)
{
        const size_t ip1 = ip + 1;
        int i, j, n, ik;
        double complex *pin, *pout;
        for (ik = 0; ik < nkpts; ik++) {
                pout = out[ik] + off;
                for (i = 0; i < di; i++) {
                        for (j = 0; j <= i; j++) {
                                pin = in + (j*di+i) * nGv;
                                for (n = 0; n < nGv; n++) {
                                        pout[j*nGv+n] += pin[n] * exp_Lk[ik];
                                }
                        }
                        pout += (ip1 + i) * nGv;
                }
        }
}

void PBC_ft_fill_s1(int (*intor)(), void (*eval_gz)(),
                    double complex **out, double complex *exp_Lk, int nkpts,
                    int ish, int jsh, double complex *buf,
                    int *shls_slice, int *ao_loc, double complex fac,
                    double *Gv, double *invh, double *gxyz, int *gs, int nGv,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        ish += ish0;
        jsh += jsh0;
        const int nrow = ao_loc[ish1] - ao_loc[ish0];
        const int ncol = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t off = ao_loc[ish] - ao_loc[ish0] + (ao_loc[jsh] - ao_loc[jsh0]) * nrow;
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int ip = ao_loc[ish] - ao_loc[ish0];
        int shls[2] = {ish, jsh};
        int dims[2] = {di, dj};
        if ((*intor)(buf, shls, dims, NULL, eval_gz, fac, Gv, invh, gxyz, gs, nGv,
                     atm, natm, bas, nbas, env)) {
                axpy_s1(out, buf, exp_Lk, nkpts, off*nGv, nGv, nrow, ncol, ip, di, dj);
        }
}

void PBC_ft_fill_s1hermi(int (*intor)(), void (*eval_gz)(),
                         double complex **out, double complex *exp_Lk, int nkpts,
                         int ish, int jsh, double complex *buf,
                         int *shls_slice, int *ao_loc, double complex fac,
                         double *Gv, double *invh, double *gxyz, int *gs, int nGv,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int jsh0 = shls_slice[2];
        const int ip = ao_loc[ish+ish0];
        const int jp = ao_loc[jsh+jsh0] - ao_loc[jsh0];
        if (ip >= jp) {
                PBC_ft_fill_s1(intor, eval_gz, out, exp_Lk, nkpts, ish, jsh, buf,
                               shls_slice, ao_loc, fac, Gv, invh, gxyz, gs, nGv,
                               atm, natm, bas, nbas, env);
        }
}

void PBC_ft_fill_s2(int (*intor)(), void (*eval_gz)(),
                    double complex **out, double complex *exp_Lk, int nkpts,
                    int ish, int jsh, double complex *buf,
                    int *shls_slice, int *ao_loc, double complex fac,
                    double *Gv, double *invh, double *gxyz, int *gs, int nGv,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        ish += ish0;
        jsh += jsh0;
        const int ip = ao_loc[ish];
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        if (ip < jp) {
                return;
        }

        const int nrow = ao_loc[ish1] - ao_loc[ish0];
        const int ncol = ao_loc[jsh1] - ao_loc[jsh0];
        const int di = ao_loc[ish+1] - ao_loc[ish];
        const int dj = ao_loc[jsh+1] - ao_loc[jsh];
        const int i0 = ao_loc[ish0];
        const size_t off = ip * (ip + 1) / 2 - i0 * (i0 + 1) / 2 + jp;
        int shls[2] = {ish, jsh};
        int dims[2] = {di, dj};
        if ((*intor)(buf, shls, dims, NULL, eval_gz, fac, Gv, invh, gxyz, gs, nGv,
                     atm, natm, bas, nbas, env)) {
                if (ip != jp) {
                        axpy_igtj(out, buf, exp_Lk, nkpts, off*nGv,
                                  nGv, nrow, ncol, ip, di, dj);
                } else {
                        axpy_ieqj(out, buf, exp_Lk, nkpts, off*nGv,
                                  nGv, nrow, ncol, ip, di, dj);
                }
        }
}


void ft_ovlp_kpts(int (*intor)(), void (*eval_gz)(), void (*fill)(),
                  double complex **out, double complex *exp_Lk, int nkpts,
                  int *shls_slice, int *ao_loc,
                  double *Gv, double *invh, double *gxyz, int *gs, int nGv,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const double complex fac = 1;

#pragma omp parallel default(none) \
        shared(intor, eval_gz, fill, out, exp_Lk, nkpts, Gv, invh, gxyz, gs, nGv, \
               shls_slice, ao_loc, atm, natm, bas, nbas, env)
{
        int i, j, ij;
        double complex *buf = malloc(sizeof(double complex) * nGv*NCTRMAX*NCTRMAX);
#pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                i = ij / njsh;
                j = ij % njsh;
                (*fill)(intor, eval_gz, out, exp_Lk, nkpts, i, j, buf,
                        shls_slice, ao_loc, fac, Gv, invh, gxyz, gs, nGv,
                        atm, natm, bas, nbas, env);
        }
        free(buf);
}
}

void PBC_ft_latsum_kpts(int (*intor)(), void (*eval_gz)(), void (*fill)(),
                        double complex **out, double *xyz, int *ptr_coords, int nxyz,
                        double *Ls, int nimgs, double complex *exp_Lk, int nkpts,
                        int *shls_slice, int *ao_loc,
                        double *Gv, double *invh, double *gxyz, int *gs, int nGv,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        int m;
        for (m = 0; m < nimgs; m++) {
                shift_bas(xyz, ptr_coords, Ls+m*3, nxyz, env);
                ft_ovlp_kpts(intor, eval_gz, fill, out, exp_Lk+m*nkpts, nkpts,
                             shls_slice, ao_loc, Gv, invh, gxyz, gs, nGv,
                             atm, natm, bas, nbas, env);
        }
}

