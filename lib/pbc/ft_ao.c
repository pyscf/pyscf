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

static void axpy(double complex **out, double complex *in,
                 double complex *exp_Lk, int nkpts, size_t off,
                 size_t nGv, size_t ni, size_t nj, size_t di, size_t dj)
{
        int i, j, k, ik;
        double complex *out_ik, *pout;
        double complex *pin;
        for (ik = 0; ik < nkpts; ik++) {
                out_ik = out[ik] + off;
                for (j = 0; j < dj; j++) {
                for (i = 0; i < di; i++) {
                        pout = out_ik + (j*ni+i) * nGv;
                        pin  = in     + (j*di+i) * nGv;
                        for (k = 0; k < nGv; k++) {
                                pout[k] += pin[k] * exp_Lk[ik];
                        }
                } }
        }
}

void ft_ovlp_kpts(int (*intor)(), void (*eval_gz)(), double complex **out,
                  double complex *exp_Lk, int nkpts,
                  int *shls_slice, int *ao_loc, int hermi,
                  double *Gv, double *invh, double *gxyz, int *gs, int nGv,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const int nrow = ao_loc[ish1] - ao_loc[ish0];
        const int ncol = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t off0 = ao_loc[ish0] + ao_loc[jsh0] * nrow;
        const double complex fac = 1;

#pragma omp parallel default(none) \
        shared(intor, out, exp_Lk, nkpts, Gv, invh, gxyz, gs, nGv, ao_loc, \
               eval_gz, hermi, atm, natm, bas, nbas, env)
{
        int i, j, ij, di, dj;
        int shls[2];
        size_t off;
        int dims[2];
        double complex *buf = malloc(sizeof(double complex) * nGv*NCTRMAX*NCTRMAX);
#pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                i = ij / njsh;
                j = ij % njsh;
                if (hermi && i < j) {
                        continue;
                }

                i += ish0;
                j += jsh0;
                shls[0] = i;
                shls[1] = j;
                di = ao_loc[i+1] - ao_loc[i];
                dj = ao_loc[j+1] - ao_loc[j];
                dims[0] = di;
                dims[1] = dj;
                off = ao_loc[i] + ao_loc[j] * nrow - off0;
                if ((*intor)(buf, shls, dims, NULL, eval_gz,
                             fac, Gv, invh, gxyz, gs, nGv,
                             atm, natm, bas, nbas, env)) {
                        axpy(out, buf, exp_Lk, nkpts, off*nGv,
                             nGv, nrow, ncol, di, dj);
                }
        }
        free(buf);
}
}

void PBC_ft_latsum_kpts(int (*intor)(), void (*eval_gz)(), double complex **out,
                        double *xyz, int *ptr_coords, int nxyz,
                        double *Ls, int nimgs, double complex *exp_Lk, int nkpts,
                        int *shls_slice, int *ao_loc, int hermi,
                        double *Gv, double *invh, double *gxyz, int *gs, int nGv,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        int m;
        for (m = 0; m < nimgs; m++) {
                shift_bas(xyz, ptr_coords, Ls+m*3, nxyz, env);
                ft_ovlp_kpts(intor, eval_gz, out, exp_Lk+m*nkpts, nkpts,
                             shls_slice, ao_loc, hermi, Gv, invh, gxyz, gs, nGv,
                             atm, natm, bas, nbas, env);
        }
}

