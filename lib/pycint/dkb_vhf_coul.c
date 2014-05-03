/*
 * File: dkb_vhf_coul.c
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 * Relativistic HF potential (in AO representation)
 */

#include <stdio.h>
#include "vhf_drv.h"

void dkb_vhf_coul(double *dm, double *vj, double *vk,
                  int ndim, int nset, int nset_dm,
                  int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {dkb_vhf_pre_, dkb_vhf_after_,
                         dkb_vhf_coul_iter_, pass, no_screen_, CINTno_optimizer,
                         NULL};
        turnoff_direct_scf_();
        r_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm,
                  atm, natm, bas, nbas, env);
}
void dkb_vhf_coul_direct(double *dm, double *vj, double *vk,
                         int ndim, int nset, int nset_dm,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {dkb_vhf_pre_and_screen_, dkb_vhf_after_,
                         dkb_vhf_coul_iter_, pass, no_screen_, CINTno_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(dkb_vhf_coul)
VHF_C2F_(dkb_vhf_coul_direct)

void dkb_vhf_coul_o02(double *dm, double *vj, double *vk,
                      int ndim, int nset, int nset_dm,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        dkb_vhf_coul_o02_(dm, vj, vk, &ndim, atm, &natm, bas, &nbas, env);
}
