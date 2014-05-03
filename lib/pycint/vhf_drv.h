/*
 * File: vhf_drv.h
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include "cint.h"

// poiter to function
// FPtr f[] == void (*f[])()
typedef int (*FPtr)();

#define DEF_VHF_(X)     void X##_(FPtr intor,\
        double *dm, double *vj, double *vk,\
        int ndim, int nset, int nset_dm, int idx, int *ao_loc,\
        int *atm, int natm, int *bas, int nbas, double *env)
#define DEF_NR_VHF_PRE_(X) void X##_(double *dm,\
        int *ndim, int *nset, int *nset_dm, int *ao_loc,\
        int *atm, int *natm, int *bas, int *nbas, double *env)
#define DEF_R_VHF_PRE_(X) void X##_(double *tdm, double *dm,\
        int *ndim, int *nset, int *nset_dm, int *ao_loc,\
        int *atm, int *natm, int *bas, int *nbas, double *env)
#define DEF_VHF_AFTER_(X)       void X##_(double *vj, double *vk,\
        int *n2c, int *nset, int *nset_dm,\
        int *atm, int *natm, int *bas, int *nbas, double *env)

#define DEF_INTOR(X)    int X(double *fijkl, const unsigned int *shls,\
        const int *atm, const int natm, const int *bas, const int nbas, \
        const double *env, const CINTOpt *opt);\
        int X##_(double *fkijl, const unsigned int *shls,\
        const int *atm, const int *natm, const int *bas, const int *nbas, \
        const double *env, const CINTOpt *opt);\
        void X##_optimizer(CINTOpt **opt, const int *atm, const int natm,\
                           const int *bas, const int nbas, const double *env)

#define VHF_C2F_(X)     void X##_(double *dm, double *vj, double *vk,\
        int *ndim, int *nset, int *nset_dm,\
        int *atm, int *natm, int *bas, int *nbas, double *env) {\
        X(dm, vj, vk, *ndim, *nset, *nset_dm, atm, *natm, bas, *nbas, env);}

#define DEF_PRESCREEN(X) int X##_(int shls[], int do_vj[], int do_vk[], int *nset);

int pass();

int CINTno_optimizer(CINTOpt **opt, const int *atm, const int natm,
                     const int *bas, const int nbas, const double *env);

int nr_vhf_drv(const FPtr* filter,
               const double *dm, double *vj, double *vk,
               const int ndim, const int nset_dm, const int nset,
               const int *atm, const int natm,
               const int *bas, const int nbas, const double *env);

int r_vhf_drv(const FPtr* filter,
              const double *dm, double *vj, double *vk,
              const int ndim, const int nset_dm, const int nset,
              const int *atm, const int natm,
              const int *bas, const int nbas, const double *env);

void set_direct_scf_cutoff_(double *);
void turnoff_direct_scf_();

DEF_VHF_(nr_hs_hs_dm2_o0);
DEF_VHF_(nr_hs_hs_dm2_o1);
DEF_VHF_(nr_hs_hs_dm2_o2);
DEF_VHF_(nr_hs_hs_dm2_o3);
DEF_VHF_(nr_has_hs_dm2_o0);
DEF_VHF_(nr_has_hs_dm2_o1);
DEF_VHF_(nr_has_hs_dm2_o2);
DEF_VHF_(nr_has_hs_dm2_o3);

DEF_VHF_(r_tsll_tsll_dm2_o0);
DEF_VHF_(r_tsll_tsll_dm2_o1);
DEF_VHF_(r_tsll_tsll_dm2_o2);
DEF_VHF_(r_tsll_tsll_dm2_o3);
DEF_VHF_(r_tsss_tsll_dm2_o0);
DEF_VHF_(r_tsss_tsll_dm2_o1);
DEF_VHF_(r_tsss_tsll_dm2_o2);
DEF_VHF_(r_tsll_tsss_dm2_o0);
DEF_VHF_(r_tsll_tsss_dm2_o1);
DEF_VHF_(r_tsll_tsss_dm2_o2);
DEF_VHF_(r_tsss_tsss_dm2_o0);
DEF_VHF_(r_tsss_tsss_dm2_o1);
DEF_VHF_(r_tsss_tsss_dm2_o2);
DEF_VHF_(r_tsss_tsss_dm2_o3);
DEF_VHF_(r_tsll_tsll_dm12_o0);
DEF_VHF_(r_tsll_tsll_dm12_o1);
DEF_VHF_(r_tsll_tsll_dm12_o2);
DEF_VHF_(r_tsll_tsll_dm12_o3);
DEF_VHF_(r_tsss_tsll_dm12_o0);
DEF_VHF_(r_tsss_tsll_dm12_o1);
DEF_VHF_(r_tsss_tsll_dm12_o2);
DEF_VHF_(r_tsll_tsss_dm12_o0);
DEF_VHF_(r_tsll_tsss_dm12_o1);
DEF_VHF_(r_tsll_tsss_dm12_o2);
DEF_VHF_(r_tsss_tsss_dm12_o0);
DEF_VHF_(r_tsss_tsss_dm12_o1);
DEF_VHF_(r_tsss_tsss_dm12_o2);
DEF_VHF_(r_tsss_tsss_dm12_o3);
DEF_VHF_(r_tasll_tsll_dm2_o0);
DEF_VHF_(r_tasll_tsll_dm2_o1);
DEF_VHF_(r_tasll_tsll_dm2_o2);
DEF_VHF_(r_tasll_tsll_dm2_o3);
DEF_VHF_(r_tasss_tsll_dm2_o0);
DEF_VHF_(r_tasss_tsll_dm2_o1);
DEF_VHF_(r_tasss_tsll_dm2_o2);
DEF_VHF_(r_tasll_tsss_dm2_o0);
DEF_VHF_(r_tasll_tsss_dm2_o1);
DEF_VHF_(r_tasll_tsss_dm2_o2);
DEF_VHF_(r_tasss_tsss_dm2_o0);
DEF_VHF_(r_tasss_tsss_dm2_o1);
DEF_VHF_(r_tasss_tsss_dm2_o2);
DEF_VHF_(r_tasss_tsss_dm2_o3);
DEF_VHF_(r_tasll_tsll_dm12_o0);
DEF_VHF_(r_tasll_tsll_dm12_o1);
DEF_VHF_(r_tasll_tsll_dm12_o2);
DEF_VHF_(r_tasll_tsll_dm12_o3);
DEF_VHF_(r_tasss_tsll_dm12_o0);
DEF_VHF_(r_tasss_tsll_dm12_o1);
DEF_VHF_(r_tasss_tsll_dm12_o2);
DEF_VHF_(r_tasll_tsss_dm12_o0);
DEF_VHF_(r_tasll_tsss_dm12_o1);
DEF_VHF_(r_tasll_tsss_dm12_o2);
DEF_VHF_(r_tasss_tsss_dm12_o0);
DEF_VHF_(r_tasss_tsss_dm12_o1);
DEF_VHF_(r_tasss_tsss_dm12_o2);
DEF_VHF_(r_tasss_tsss_dm12_o3);
DEF_VHF_(rkb_vhf_gaunt_iter);
DEF_VHF_(rmb_vhf_gaunt_iter);

DEF_PRESCREEN(no_screen);
DEF_NR_VHF_PRE_(nr_vhf_init_screen);
DEF_VHF_AFTER_(nr_vhf_del_screen);
int nr_vhf_prescreen_(int shls[]);

//DEF_INTOR(cint2e_sph);
int cint2e_sph_(double *fkijl, const unsigned int *shls,
                const int *atm, const int *natm, const int *bas, const int *nbas,
                const double *env, const CINTOpt *opt);
void cint2e_sph_optimizer(CINTOpt **opt, const int *atm, const int natm,
                           const int *bas, const int nbas, const double *env);

DEF_INTOR(cint2e_ip1_sph);

DEF_VHF_AFTER_(nr_vhf_igiao_after);
DEF_INTOR(cint2e_ig1_sph);


DEF_NR_VHF_PRE_(rkb_vhf_init_screen);
DEF_VHF_AFTER_(rkb_vhf_del_screen);
DEF_PRESCREEN(rkb_vhf_ll_prescreen);
DEF_PRESCREEN(rkb_vhf_sl_prescreen);
DEF_PRESCREEN(rkb_vhf_ss_prescreen);

DEF_R_VHF_PRE_(rkb_vhf_pre);
DEF_R_VHF_PRE_(rkb_vhf_pre_and_screen);
DEF_R_VHF_PRE_(rkb_vhf_ll_pre);
DEF_R_VHF_PRE_(rkb_vhf_ll_pre_and_screen);
DEF_R_VHF_PRE_(rkb_vhf_ss_pre);
DEF_R_VHF_PRE_(rkb_vhf_ss_pre_and_screen);
DEF_VHF_AFTER_(rkb_vhf_after);
DEF_VHF_AFTER_(rkb_vhf_ss_after);
DEF_VHF_AFTER_(rkb_vhf_sl_after);

DEF_VHF_(rkb_vhf_sl_o2);
//DEF_INTOR(cint2e);
int cint2e_(double *fkijl, const unsigned int *shls,
            const int *atm, const int *natm, const int *bas, const int *nbas,
            const double *env, const CINTOpt *opt);
void cint2e_optimizer(CINTOpt **opt, const int *atm, const int natm,
                      const int *bas, const int nbas, const double *env);
DEF_INTOR(cint2e_spsp1);
DEF_INTOR(cint2e_spsp1spsp2);

DEF_VHF_AFTER_(rmb_vhf_after);
DEF_INTOR(cint2e_giao_sa10sp1);
DEF_INTOR(cint2e_giao_sa10sp1spsp2);
DEF_INTOR(cint2e_cg_sa10sp1);
DEF_INTOR(cint2e_cg_sa10sp1spsp2);

DEF_VHF_AFTER_(rkb_giao_vhf_after);
DEF_VHF_AFTER_(rkb_giao_vhf_ll_after);
DEF_INTOR(cint2e_g1);
DEF_INTOR(cint2e_g1spsp2);
DEF_INTOR(cint2e_spgsp1);
DEF_INTOR(cint2e_spgsp1spsp2);

DEF_INTOR(cint2e_ip1);
DEF_INTOR(cint2e_ip1spsp2);
DEF_INTOR(cint2e_ipspsp1);
DEF_INTOR(cint2e_ipspsp1spsp2);



DEF_VHF_(dkb_vhf_coul_iter);
DEF_NR_VHF_PRE_(dkb_vhf_init_screen);
DEF_VHF_AFTER_(dkb_vhf_del_screen);
DEF_R_VHF_PRE_(dkb_vhf_pre);
DEF_R_VHF_PRE_(dkb_vhf_pre_and_screen);
DEF_VHF_AFTER_(dkb_vhf_after);

DEF_PRESCREEN(rkb_vhf_gaunt_prescreen);
DEF_R_VHF_PRE_(rkb_vhf_gaunt_pre_and_screen);
DEF_VHF_AFTER_(rkb_vhf_gaunt_after);
DEF_VHF_AFTER_(rmb_vhf_gaunt_after);
DEF_INTOR(cint2e_cg_ssa10ssp2);
DEF_INTOR(cint2e_giao_ssa10ssp2);
DEF_INTOR(cint2e_gssp1ssp2);
