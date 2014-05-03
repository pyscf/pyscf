/*
 * File: rkb_vhf.c
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 *
 * Relativistic HF potential (in AO representation)
 */

#include <stdio.h>
#include "vhf_drv.h"

/*
 *  HF potential of Dirac-Coulomb Hamiltonian (in RKB basis)
 *       J = (ii|\mu \nu)
 *       K = (\mu i|i\nu)
 *  Density matrix *is* assumed to be *Hermitian*
 */
void rkb_vhf_coul_o0(double *dm, double *vj, double *vk,
                    int ndim, int nset, int nset_dm,
                    int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rkb_vhf_sl_after_,
                         r_tsll_tsll_dm2_o0_, cint2e_, no_screen_, CINTno_optimizer,
                         rkb_vhf_sl_o2_, cint2e_spsp1_, no_screen_, CINTno_optimizer,
                         r_tsss_tsss_dm2_o0_, cint2e_spsp1spsp2_, no_screen_, CINTno_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm,
                  atm, natm, bas, nbas, env);
}
void rkb_vhf_coul_o1(double *dm, double *vj, double *vk,
                     int ndim, int nset, int nset_dm,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rkb_vhf_sl_after_,
                         r_tsll_tsll_dm2_o1_, cint2e_, no_screen_, CINTno_optimizer,
                         rkb_vhf_sl_o2_, cint2e_spsp1_, no_screen_, CINTno_optimizer,
                         r_tsss_tsss_dm2_o1_, cint2e_spsp1spsp2_, no_screen_, CINTno_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm,
                  atm, natm, bas, nbas, env);
}
void rkb_vhf_coul_o2(double *dm, double *vj, double *vk,
                     int ndim, int nset, int nset_dm,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rkb_vhf_sl_after_,
                         r_tsll_tsll_dm2_o2_, cint2e_, no_screen_, CINTno_optimizer,
                         rkb_vhf_sl_o2_, cint2e_spsp1_, no_screen_, CINTno_optimizer,
                         r_tsss_tsss_dm2_o2_, cint2e_spsp1spsp2_, no_screen_, CINTno_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm,
                  atm, natm, bas, nbas, env);
}
void rkb_vhf_coul_o3(double *dm, double *vj, double *vk,
                     int ndim, int nset, int nset_dm,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rkb_vhf_sl_after_,
                         r_tsll_tsll_dm2_o3_, cint2e_, no_screen_, cint2e_optimizer,
                         rkb_vhf_sl_o2_, cint2e_spsp1_, no_screen_, cint2e_spsp1_optimizer,
                         r_tsss_tsss_dm2_o3_, cint2e_spsp1spsp2_, no_screen_, cint2e_spsp1spsp2_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm,
                  atm, natm, bas, nbas, env);
}

void rkb_vhf_coul_direct_o3(double *dm, double *vj, double *vk,
                            int ndim, int nset, int nset_dm,
                            int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_and_screen_, rkb_vhf_sl_after_,
                         r_tsll_tsll_dm2_o3_, cint2e_, rkb_vhf_ll_prescreen_, cint2e_optimizer,
                         rkb_vhf_sl_o2_, cint2e_spsp1_, rkb_vhf_sl_prescreen_, cint2e_spsp1_optimizer,
                         r_tsss_tsss_dm2_o3_, cint2e_spsp1spsp2_, rkb_vhf_ss_prescreen_, cint2e_spsp1spsp2_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(rkb_vhf_coul_o0);
VHF_C2F_(rkb_vhf_coul_o1);
VHF_C2F_(rkb_vhf_coul_o2);
VHF_C2F_(rkb_vhf_coul_o3);
VHF_C2F_(rkb_vhf_coul_direct_o3);

void rkb_vhf_ll_o3(double *dm, double *vj, double *vk,
                   int ndim, int nset, int nset_dm,
                   int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_ll_pre_, rkb_vhf_del_screen_,
                         r_tsll_tsll_dm2_o3_, cint2e_, no_screen_, cint2e_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm,
                  atm, natm, bas, nbas, env);
}
void rkb_vhf_ll_direct_o3(double *dm, double *vj, double *vk,
                          int ndim, int nset, int nset_dm,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_ll_pre_and_screen_, rkb_vhf_del_screen_,
                         r_tsll_tsll_dm2_o3_, cint2e_, rkb_vhf_ll_prescreen_, cint2e_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(rkb_vhf_ll_o3);
VHF_C2F_(rkb_vhf_ll_direct_o3);

void rkb_vhf_sl_o3(double *dm, double *vj, double *vk,
                   int ndim, int nset, int nset_dm,
                   int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rkb_vhf_sl_after_,
                         r_tsll_tsll_dm2_o3_, cint2e_, no_screen_, cint2e_optimizer,
                         rkb_vhf_sl_o2_, cint2e_spsp1_, no_screen_, cint2e_spsp1_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm,
                  atm, natm, bas, nbas, env);
}
void rkb_vhf_sl_direct_o3(double *dm, double *vj, double *vk,
                          int ndim, int nset, int nset_dm,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_and_screen_, rkb_vhf_sl_after_,
                         r_tsll_tsll_dm2_o3_, cint2e_, rkb_vhf_ll_prescreen_, cint2e_optimizer,
                         rkb_vhf_sl_o2_, cint2e_spsp1_, rkb_vhf_sl_prescreen_, cint2e_spsp1_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(rkb_vhf_sl_o3);
VHF_C2F_(rkb_vhf_sl_direct_o3);

/*
 * gradients
 */
void rkb_vhf_coul_grad_o0(double *dm, double *vj, double *vk,
                          int ndim, int nset, int nset_dm,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rkb_vhf_after_,
// Because of the particle permutation symm., use dm2 insteaf of dm12.
                         r_tsll_tsll_dm2_o0_, cint2e_ip1_, no_screen_, CINTno_optimizer,
                         r_tsll_tsss_dm2_o0_, cint2e_ip1spsp2_, no_screen_, CINTno_optimizer,
                         r_tsss_tsll_dm2_o0_, cint2e_ipspsp1_, no_screen_, CINTno_optimizer,
                         r_tsss_tsss_dm2_o0_, cint2e_ipspsp1spsp2_, no_screen_, CINTno_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 3, 1,
                  atm, natm, bas, nbas, env);
}
void rkb_vhf_coul_grad_o1(double *dm, double *vj, double *vk,
                          int ndim, int nset, int nset_dm,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rkb_vhf_after_,
                         r_tsll_tsll_dm2_o1_, cint2e_ip1_, no_screen_, cint2e_ip1_optimizer,
                         r_tsll_tsss_dm2_o1_, cint2e_ip1spsp2_, no_screen_, cint2e_ip1spsp2_optimizer,
                         r_tsss_tsll_dm2_o1_, cint2e_ipspsp1_, no_screen_, cint2e_ipspsp1_optimizer,
                         r_tsss_tsss_dm2_o1_, cint2e_ipspsp1spsp2_, no_screen_, cint2e_ipspsp1spsp2_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 3, 1,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(rkb_vhf_coul_grad_o0);
VHF_C2F_(rkb_vhf_coul_grad_o1);

void rkb_vhf_coul_grad_ll_o1(double *dm, double *vj, double *vk,
                             int ndim, int nset, int nset_dm,
                             int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rkb_vhf_after_,
                         r_tsll_tsll_dm2_o1_, cint2e_ip1_, no_screen_, cint2e_ip1_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 3, 1,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(rkb_vhf_coul_grad_ll_o1);

void rkb_vhf_coul_grad_ls2l_o1(double *dm, double *vj, double *vk,
                             int ndim, int nset, int nset_dm,
                             int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rkb_vhf_after_,
                         r_tsll_tsll_dm2_o1_, cint2e_ip1_, no_screen_, cint2e_ip1_optimizer,
                         r_tsll_tsss_dm2_o1_, cint2e_ip1spsp2_, no_screen_, cint2e_ip1spsp2_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 3, 1,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(rkb_vhf_coul_grad_ls2l_o1);

void rkb_vhf_coul_grad_l2sl_o1(double *dm, double *vj, double *vk,
                             int ndim, int nset, int nset_dm,
                             int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rkb_vhf_after_,
                         r_tsll_tsll_dm2_o1_, cint2e_ip1_, no_screen_, cint2e_ip1_optimizer,
                         r_tsss_tsll_dm2_o1_, cint2e_ipspsp1_, no_screen_, cint2e_ipspsp1_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 3, 1,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(rkb_vhf_coul_grad_l2sl_o1);

void rkb_vhf_coul_grad_xss_o1(double *dm, double *vj, double *vk,
                             int ndim, int nset, int nset_dm,
                             int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rkb_vhf_after_,
                         r_tsll_tsll_dm2_o1_, cint2e_ip1_, no_screen_, cint2e_ip1_optimizer,
                         r_tsll_tsss_dm2_o1_, cint2e_ip1spsp2_, no_screen_, cint2e_ip1spsp2_optimizer,
                         r_tsss_tsll_dm2_o1_, cint2e_ipspsp1_, no_screen_, cint2e_ipspsp1_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 3, 1,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(rkb_vhf_coul_grad_xss_o1);

/*
 * Magnetic balance
 */
// MB for GIAO
void rmb4giao_vhf_coul(double *dm, double *vj, double *vk,
                       int ndim, int nset, int nset_dm,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rmb_vhf_after_,
                         //rmb_vhf_cssll_o1_, cint2e_sa10sp1_, no_screen_, cint2e_sa10sp1_optimizer,
                         //rmb_vhf_cssss_o1_, cint2e_sa10sp1spsp2_, no_screen_, cint2e_sa10sp1spsp2_optimizer,
                         r_tasss_tsll_dm12_o1_, cint2e_giao_sa10sp1_, no_screen_, cint2e_giao_sa10sp1_optimizer,
                         r_tasss_tsss_dm12_o1_, cint2e_giao_sa10sp1spsp2_, no_screen_, cint2e_giao_sa10sp1spsp2_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 3, 1,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(rmb4giao_vhf_coul);

// MB for common gauge
void rmb4cg_vhf_coul(double *dm, double *vj, double *vk,
                     int ndim, int nset, int nset_dm,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rmb_vhf_after_,
                         //rmb_vhf_cssll_o1_, cint2e_sa10sp1_, no_screen_, cint2e_sa10sp1_optimizer,
                         //rmb_vhf_cssss_o1_, cint2e_sa10sp1spsp2_, no_screen_, cint2e_sa10sp1spsp2_optimizer,
                         r_tasss_tsll_dm12_o1_, cint2e_cg_sa10sp1_, no_screen_, cint2e_cg_sa10sp1_optimizer,
                         r_tasss_tsss_dm12_o1_, cint2e_cg_sa10sp1spsp2_, no_screen_, cint2e_cg_sa10sp1spsp2_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 3, 1,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(rmb4cg_vhf_coul);

// GIAO with RKB
void rkb_giao_vhf_coul(double *dm, double *vj, double *vk,
                       int ndim, int nset, int nset_dm,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
// Because of the particle permutation symm., use dm2 insteaf of dm12.
        FPtr filter[] = {rkb_vhf_pre_, rkb_giao_vhf_after_,
                         r_tasll_tsll_dm2_o2_, cint2e_g1_, no_screen_, cint2e_g1_optimizer,
                         r_tasll_tsss_dm2_o2_, cint2e_g1spsp2_, no_screen_, cint2e_g1spsp2_optimizer,
                         r_tasss_tsll_dm2_o2_, cint2e_spgsp1_, no_screen_, cint2e_spgsp1_optimizer,
                         r_tasss_tsss_dm2_o2_, cint2e_spgsp1spsp2_, no_screen_, cint2e_spgsp1spsp2_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 3, 1,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(rkb_giao_vhf_coul);


/*
 * Gaunt interactions
 */
void rkb_vhf_gaunt(double *dm, double *vj, double *vk,
                   int ndim, int nset, int nset_dm,
                   int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rkb_vhf_gaunt_after_,
                         rkb_vhf_gaunt_iter_, pass, no_screen_, CINTno_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm,
                  atm, natm, bas, nbas, env);
}
void rkb_vhf_gaunt_direct(double *dm, double *vj, double *vk,
                          int ndim, int nset, int nset_dm,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_gaunt_pre_and_screen_, rkb_vhf_gaunt_after_,
                         rkb_vhf_gaunt_iter_, pass, rkb_vhf_gaunt_prescreen_, CINTno_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 1, nset_dm,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(rkb_vhf_gaunt);
VHF_C2F_(rkb_vhf_gaunt_direct);

void rmb4cg_vhf_gaunt(double *dm, double *vj, double *vk,
                      int ndim, int nset, int nset_dm,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rmb_vhf_gaunt_after_,
                         rmb_vhf_gaunt_iter_, cint2e_cg_ssa10ssp2_, no_screen_, cint2e_cg_ssa10ssp2_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 3, 1,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(rmb4cg_vhf_gaunt);

void rmb4giao_vhf_gaunt(double *dm, double *vj, double *vk,
                        int ndim, int nset, int nset_dm,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rmb_vhf_gaunt_after_,
                         rmb_vhf_gaunt_iter_, cint2e_giao_ssa10ssp2_, no_screen_, cint2e_giao_ssa10ssp2_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 3, 1,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(rmb4giao_vhf_gaunt);

void rkb_giao_vhf_gaunt(double *dm, double *vj, double *vk,
                        int ndim, int nset, int nset_dm,
                        int *atm, int natm, int *bas, int nbas, double *env)
{
        FPtr filter[] = {rkb_vhf_pre_, rmb_vhf_gaunt_after_,
                         rmb_vhf_gaunt_iter_, cint2e_gssp1ssp2_, no_screen_, cint2e_gssp1ssp2_optimizer,
                         NULL};
        r_vhf_drv(filter, dm, vj, vk, ndim, 3, 1,
                  atm, natm, bas, nbas, env);
}
VHF_C2F_(rkb_giao_vhf_gaunt);
