/*
 * Copyright (C) 2013  Qiming Sun <osirpt.sun@gmail.com>
 *
 * basic functions
 */

#include "cint_const.h"
#include "fblas.h"

void CINTdcmplx_re(const FINT n, double complex *z, const double *re);
void CINTdcmplx_im(const FINT n, double complex *z, const double *im);
void CINTdcmplx_pp(const FINT n, double complex *z, const double *re, const double *im);
void CINTdcmplx_pn(const FINT n, double complex *z, const double *re, const double *im);
void CINTdcmplx_np(const FINT n, double complex *z, const double *re, const double *im);

double CINTsquare_dist(const double *r1, const double *r2);

void CINTrys_roots(const FINT nroots, double x, double *u, double *w);

double CINTgto_norm(FINT n, double a);

