/*
 * File: nr_vhf.c
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 * Relativistic HF potential (in AO representation)
 */

#include <stdio.h>
#include "vhf_drv.h"

/*
 * non-relativistic HF coulomb and exchange potential (in AO representation)
 *       J = (ii|\mu \nu)
 *       K = (\mu i|i\nu)
 *  Density matrix *is* assumed to be *Hermitian*
 */
void nr_vhf_o0(double *dm, double *vj, double *vk,
               int ndim, int nset, int nset_dm,
               int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {pass, pass,
                         nr_hs_hs_dm2_o0_, cint2e_sph_, no_screen_, CINTno_optimizer,
                         NULL};
        nr_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm, 
                   atm, natm, bas, nbas, env);
}
void nr_vhf_o1(double *dm, double *vj, double *vk,
               int ndim, int nset, int nset_dm,
               int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {pass, pass,
                         nr_hs_hs_dm2_o1_, cint2e_sph_, no_screen_, CINTno_optimizer,
                         NULL};
        nr_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm, 
                   atm, natm, bas, nbas, env);
}
void nr_vhf_o2(double *dm, double *vj, double *vk,
               int ndim, int nset, int nset_dm,
               int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {pass, pass,
                         nr_hs_hs_dm2_o2_, cint2e_sph_, no_screen_, CINTno_optimizer,
                         NULL};
        nr_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm, 
                   atm, natm, bas, nbas, env);
}
void nr_vhf_o3(double *dm, double *vj, double *vk,
               int ndim, int nset, int nset_dm,
               int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {pass, pass,
                         nr_hs_hs_dm2_o3_, cint2e_sph_, no_screen_, cint2e_optimizer,
                         NULL};
        nr_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm, 
                   atm, natm, bas, nbas, env);
}
void nr_vhf_direct_o0(double *dm, double *vj, double *vk,
                      int ndim, int nset, int nset_dm,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {nr_vhf_init_screen_, nr_vhf_del_screen_,
                         nr_hs_hs_dm2_o0_, cint2e_sph_, nr_vhf_prescreen_, CINTno_optimizer,
                         NULL};
        nr_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm, 
                   atm, natm, bas, nbas, env);
}
void nr_vhf_direct_o3(double *dm, double *vj, double *vk,
                      int ndim, int nset, int nset_dm,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {nr_vhf_init_screen_, nr_vhf_del_screen_,
                         nr_hs_hs_dm2_o3_, cint2e_sph_, nr_vhf_prescreen_, cint2e_optimizer,
                         NULL};
        nr_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm, 
                   atm, natm, bas, nbas, env);
}
VHF_C2F_(nr_vhf_o0)
VHF_C2F_(nr_vhf_o1)
VHF_C2F_(nr_vhf_o2)
VHF_C2F_(nr_vhf_o3)
VHF_C2F_(nr_vhf_direct_o0)
VHF_C2F_(nr_vhf_direct_o3)

void nr_vhf_igiao_o0(double *dm, double *vj, double *vk,
                     int ndim, int nset, int nset_dm,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {pass, nr_vhf_igiao_after_,
                         nr_has_hs_dm2_o0_, cint2e_ig1_sph_, no_screen_, CINTno_optimizer,
                         NULL};
        nr_vhf_drv(filter, dm, vj, vk, ndim, 3, 1, 
                   atm, natm, bas, nbas, env);
}
void nr_vhf_igiao_o1(double *dm, double *vj, double *vk,
                     int ndim, int nset, int nset_dm,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {pass, nr_vhf_igiao_after_,
                         nr_has_hs_dm2_o1_, cint2e_ig1_sph_, no_screen_, CINTno_optimizer,
                         NULL};
        nr_vhf_drv(filter, dm, vj, vk, ndim, 3, 1, 
                   atm, natm, bas, nbas, env);
}
void nr_vhf_igiao_o2(double *dm, double *vj, double *vk,
                     int ndim, int nset, int nset_dm,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {pass, nr_vhf_igiao_after_,
                         nr_has_hs_dm2_o2_, cint2e_ig1_sph_, no_screen_, cint2e_ig1_sph_optimizer,
                         NULL};
        nr_vhf_drv(filter, dm, vj, vk, ndim, 3, 1, 
                   atm, natm, bas, nbas, env);
}
VHF_C2F_(nr_vhf_igiao_o0)
VHF_C2F_(nr_vhf_igiao_o1)
VHF_C2F_(nr_vhf_igiao_o2)

void nr_vhf_grad_o0(double *dm, double *vj, double *vk,
                    int ndim, int nset, int nset_dm,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {pass, pass,
                         nr_hs_hs_dm2_o0_, cint2e_ip1_sph_, no_screen_, CINTno_optimizer,
                         NULL};
        nr_vhf_drv(filter, dm, vj, vk, ndim, 3, 1, 
                   atm, natm, bas, nbas, env);
}
void nr_vhf_grad_o1(double *dm, double *vj, double *vk,
                    int ndim, int nset, int nset_dm,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {pass, pass,
                         nr_hs_hs_dm2_o1_, cint2e_ip1_sph_, no_screen_, cint2e_ip1_sph_optimizer,
                         NULL};
        nr_vhf_drv(filter, dm, vj, vk, ndim, 3, 1, 
                   atm, natm, bas, nbas, env);
}
VHF_C2F_(nr_vhf_grad_o0)
VHF_C2F_(nr_vhf_grad_o1)
