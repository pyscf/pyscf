/*
 * Copyright (C) 2013  Qiming Sun <osirpt.sun@gmail.com>
 */

#include "config.h"

#if !defined HAVE_DEFINED_CINTOPT_H
#define HAVE_DEFINED_CINTOPT_H
typedef struct {
    FINT **index_xyz_array; // ANG_MAX**4 pointers to index_xyz
    FINT *prim_offset;
    FINT *non0ctr;
    FINT **non0idx;
    double **non0coeff;
    double **expij;
    double **rij;
    FINT **cceij;
    FINT tot_prim;
} CINTOpt;
#endif

void CINTinit_2e_optimizer(CINTOpt **opt, const FINT *atm, const FINT natm,
                           const FINT *bas, const FINT nbas, const double *env);
void CINTinit_optimizer(CINTOpt **opt, const FINT *atm, const FINT natm,
                        const FINT *bas, const FINT nbas, const double *env);
void CINTdel_2e_optimizer(CINTOpt **opt);
void CINTdel_optimizer(CINTOpt **opt);
void CINTOpt_set_index_xyz(CINTOpt *opt, FINT *ng,
                           const FINT *atm, const FINT natm,
                           const FINT *bas, const FINT nbas, const double *env);
void CINTOpt_setij(CINTOpt *opt, FINT *ng,
                   const FINT *atm, const FINT natm,
                   const FINT *bas, const FINT nbas, const double *env);
void CINTOpt_set_non0coeff(CINTOpt *opt, const FINT *atm, const FINT natm,
                           const FINT *bas, const FINT nbas, const double *env);

void CINTOpt_set_3cindex_xyz(CINTOpt *opt, FINT *ng,
                             const FINT *atm, const FINT natm,
                             const FINT *bas, const FINT nbas, const double *env);
void CINTOpt_set_2cindex_xyz(CINTOpt *opt, FINT *ng,
                             const FINT *atm, const FINT natm,
                             const FINT *bas, const FINT nbas, const double *env);

// optimizer examples
void CINTno_optimizer(CINTOpt **opt, const FINT *atm, const FINT natm,
                      const FINT *bas, const FINT nbas, const double *env);
void CINTuse_all_optimizer(CINTOpt **opt, FINT *ng,
                           const FINT *atm, const FINT natm,
                           const FINT *bas, const FINT nbas, const double *env);

