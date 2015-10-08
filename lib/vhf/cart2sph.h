/*
 * Copyright (C) 2013  Qiming Sun <osirpt.sun@gmail.com>
 *
 * Cartisen GTO to spheric or spinor GTO transformation
 */

/*************************************************
 *
 * transform matrix
 *
 *************************************************/
#include <complex.h>
#include "g1e.h"

void c2s_sph_1e(double *opij, const double *gctr, CINTEnvVars *envs);
void c2s_sph_2e1(double *fijkl, const double *gctr, CINTEnvVars *envs);
void c2s_sph_2e2();

void c2s_cart_1e(double *opij, const double *gctr, CINTEnvVars *envs);
void c2s_cart_2e1(double *fijkl, const double *gctr, CINTEnvVars *envs);
void c2s_cart_2e2();

void c2s_sf_1e(double complex *opij, const double *gctr, CINTEnvVars *envs);
void c2s_sf_1ei(double complex *opij, const double *gctr, CINTEnvVars *envs);

void c2s_si_1e(double complex *opij, const double *gctr, CINTEnvVars *envs);
void c2s_si_1ei(double complex *opij, const double *gctr, CINTEnvVars *envs);

void c2s_sf_2e1(double complex *opij, const double *gctr, CINTEnvVars *envs);
void c2s_sf_2e1i(double complex *opij, const double *gctr, CINTEnvVars *envs);

void c2s_sf_2e2(double complex *fijkl, const double complex *opij, CINTEnvVars *envs);
void c2s_sf_2e2i(double complex *fijkl, const double complex *opij, CINTEnvVars *envs);

void c2s_si_2e1(double complex *opij, const double *gctr, CINTEnvVars *envs);
void c2s_si_2e1i(double complex *opij, const double *gctr, CINTEnvVars *envs);

void c2s_si_2e2(double complex *fijkl, const double complex *opij, CINTEnvVars *envs);
void c2s_si_2e2i(double complex *fijkl, const double complex *opij, CINTEnvVars *envs);

void c2s_sph_3c2e1(double *fijkl, const double *gctr, CINTEnvVars *envs);
void c2s_sph_3c2e2();

void c2s_cart_3c2e1(double *fijkl, const double *gctr, CINTEnvVars *envs);
void c2s_cart_3c2e2();

void c2s_sf_3c2e1(double complex *opijk, double *gctr, CINTEnvVars *envs);
void c2s_sf_3c2e1i(double complex *opijk, double *gctr, CINTEnvVars *envs);
void c2s_si_3c2e1(double complex *opijk, double *gctr, CINTEnvVars *envs);
void c2s_si_3c2e1i(double complex *opijk, double *gctr, CINTEnvVars *envs);
