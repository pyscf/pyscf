#include <complex.h>

void symmetrize(double* rhoR_b, double* rhoR_a, int* op, int* mesh)
{
  const int nx = mesh[0];
  const int ny = mesh[1];
  const int nz = mesh[2];

#pragma omp parallel
{
  int x,y,z;

  #pragma omp for schedule(static)
    for (x = 0; x < nx; x++)
      for (y = 0; y < ny; y++)
        for (z = 0; z < nz; z++) {
            int x_p = ((op[0] * x + op[1] * y + op[2] * z) % nx + nx) % nx;
            int y_p = ((op[3] * x + op[4] * y + op[5] * z) % ny + ny) % ny;
            int z_p = ((op[6] * x + op[7] * y + op[8] * z) % nz + nz) % nz;
            rhoR_b[x*ny*nz + y*nz + z] += rhoR_a[x_p*ny*nz + y_p*nz + z_p];
        }
}
}

void symmetrize_ft(double* rhoR_b, double* rhoR_a, int* op, double* ft, int* mesh)
{
  const int nx = mesh[0];
  const int ny = mesh[1];
  const int nz = mesh[2];

  int ft_x = (int)(ft[0] * nx);
  int ft_y = (int)(ft[1] * ny);
  int ft_z = (int)(ft[2] * nz);

#pragma omp parallel
{
  int x,y,z;

  #pragma omp for schedule(static)
    for (x = 0; x < nx; x++)
      for (y = 0; y < ny; y++)
        for (z = 0; z < nz; z++) {
            int x_p = ((op[0] * x + op[1] * y + op[2] * z + ft_x) % nx + nx) % nx;
            int y_p = ((op[3] * x + op[4] * y + op[5] * z + ft_y) % ny + ny) % ny;
            int z_p = ((op[6] * x + op[7] * y + op[8] * z + ft_z) % nz + nz) % nz;
            rhoR_b[x*ny*nz + y*nz + z] += rhoR_a[x_p*ny*nz + y_p*nz + z_p];
        }
}
}

void symmetrize_complex(complex double* rhoR_b, complex double* rhoR_a, int* op, int* mesh)
{
  const int nx = mesh[0];
  const int ny = mesh[1];
  const int nz = mesh[2];

#pragma omp parallel
{
  int x,y,z;

  #pragma omp for schedule(static)
    for (x = 0; x < nx; x++)
      for (y = 0; y < ny; y++)
        for (z = 0; z < nz; z++) {
            int x_p = ((op[0] * x + op[1] * y + op[2] * z) % nx + nx) % nx;
            int y_p = ((op[3] * x + op[4] * y + op[5] * z) % ny + ny) % ny;
            int z_p = ((op[6] * x + op[7] * y + op[8] * z) % nz + nz) % nz;
            rhoR_b[x*ny*nz + y*nz + z] += rhoR_a[x_p*ny*nz + y_p*nz + z_p];
        }
}
}

void symmetrize_ft_complex(complex double* rhoR_b, complex double* rhoR_a, int* op, double* ft, int* mesh)
{
  const int nx = mesh[0];
  const int ny = mesh[1];
  const int nz = mesh[2];

  int ft_x = (int)(ft[0] * nx);
  int ft_y = (int)(ft[1] * ny);
  int ft_z = (int)(ft[2] * nz);

#pragma omp parallel
{
  int x,y,z;

  #pragma omp for schedule(static)
    for (x = 0; x < nx; x++)
      for (y = 0; y < ny; y++)
        for (z = 0; z < nz; z++) {
            int x_p = ((op[0] * x + op[1] * y + op[2] * z + ft_x) % nx + nx) % nx;
            int y_p = ((op[3] * x + op[4] * y + op[5] * z + ft_y) % ny + ny) % ny;
            int z_p = ((op[6] * x + op[7] * y + op[8] * z + ft_z) % nz + nz) % nz;
            rhoR_b[x*ny*nz + y*nz + z] += rhoR_a[x_p*ny*nz + y_p*nz + z_p];
        }
}
}
