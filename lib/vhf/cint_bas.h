/*
 * Copyright (C) 2013  Qiming Sun <osirpt.sun@gmail.com>
 *
 * basic cGTO function
 */

#include "cint_const.h"

#define bas(SLOT,I)     bas[BAS_SLOTS * (I) + (SLOT)]
#define atm(SLOT,I)     atm[ATM_SLOTS * (I) + (SLOT)]

FINT CINTlen_cart(const FINT l);
FINT CINTlen_spinor(const FINT bas_id, const FINT *bas);

FINT CINTcgtos_cart(const FINT bas_id, const FINT *bas);
FINT CINTcgtos_spheric(const FINT bas_id, const FINT *bas);
FINT CINTcgtos_spinor(const FINT bas_id, const FINT *bas);
FINT CINTcgto_cart(const FINT bas_id, const FINT *bas);
FINT CINTcgto_spheric(const FINT bas_id, const FINT *bas);
FINT CINTcgto_spinor(const FINT bas_id, const FINT *bas);

FINT CINTtot_pgto_spheric(const FINT *bas, const FINT nbas);
FINT CINTtot_pgto_spinor(const FINT *bas, const FINT nbas);

FINT CINTtot_cgto_cart(const FINT *bas, const FINT nbas);
FINT CINTtot_cgto_spheric(const FINT *bas, const FINT nbas);
FINT CINTtot_cgto_spinor(const FINT *bas, const FINT nbas);

void CINTshells_cart_offset(FINT ao_loc[], const FINT *bas, const FINT nbas);
void CINTshells_spheric_offset(FINT ao_loc[], const FINT *bas, const FINT nbas);
void CINTshells_spinor_offset(FINT ao_loc[], const FINT *bas, const FINT nbas);

void CINTcart_comp(FINT *nx, FINT *ny, FINT *nz, const FINT lmax);

