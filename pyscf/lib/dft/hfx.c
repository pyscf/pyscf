#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "config.h"
#include "cint.h"
#include "cint_funcs.h"
#include "gto/gto.h"
#include "np_helper/np_helper.h"
#include "pbc/neighbor_list.h"
#include "vhf/nr_direct.h"
#include "dft/utils.h"


static void shift_bas(double *env_loc, double *env, double *Ls, int ptr, int iL)
{
    env_loc[ptr+0] = env[ptr+0] + Ls[iL*3+0];
    env_loc[ptr+1] = env[ptr+1] + Ls[iL*3+1];
    env_loc[ptr+2] = env[ptr+2] + Ls[iL*3+2];
}

static void shift_bas_r(double *env_loc, double *env, double *r, int ptr)
{
    env_loc[ptr+0] = env[ptr+0] + r[0];
    env_loc[ptr+1] = env[ptr+1] + r[1];
    env_loc[ptr+2] = env[ptr+2] + r[2];
}


static void get_cell_index(double *cell_id, double *r, double *b)
{
    double x, y, z;
    x = b[0] * r[0] + b[1] * r[1] + b[2] * r[2];
    y = b[3] * r[0] + b[4] * r[1] + b[5] * r[2];
    z = b[6] * r[0] + b[7] * r[1] + b[8] * r[2];
    cell_id[0] = floor(x);
    cell_id[1] = floor(y);
    cell_id[2] = floor(z);
}

static void get_nearest_shift_vector(double *ishift, double *a, int nvec)
{
    int i;
    for (i = 0; i < nvec; i++) {
        ishift[2] = (double)(nvec % 3 - 1);
        ishift[1] = (double)((nvec / 3) % 3 - 1);
        ishift[0] = (double)(nvec / 9 - 1);
        ishift += 3;
    }
}

static inline void _frac_to_xyz(double *xyz, double *frac, double *a)
{
    xyz[0] = a[0] * frac[0] + a[3] * frac[1] + a[6] * frac[2];
    xyz[1] = a[1] * frac[0] + a[4] * frac[1] + a[7] * frac[2];
    xyz[2] = a[2] * frac[0] + a[5] * frac[1] + a[8] * frac[2];
}

static inline void _r_copy(double *r, double *r1)
{
    r[0] = r1[0];
    r[1] = r1[1];
    r[2] = r1[2];
}

static inline void _rij(double* rij, double *ri, double *rj)
{
    rij[0] = ri[0] - rj[0];
    rij[1] = ri[1] - rj[1];
    rij[2] = ri[2] - rj[2];
}

static inline double _r_norm(double *r)
{
    return sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
}

static int _assemble_eris(int (*intor)(), double *buf,
                          int ish, int jsh, int ksh, int lsh,
                          NeighborPair *np0_ij, NeighborPair *np0_kl,
                          double dm_cond, double cutoff, double *Ls, double *a, double *b, double *ishift,
                          CVHFOpt *vhfopt, IntorEnvs *envs, double* env_loc)
{
    int *atm = envs->atm;
    int *bas = envs->bas;
    double *env = envs->env;
    int natm = envs->natm;
    int nbas = envs->nbas;
    CINTOpt *cintopt = envs->cintopt;
    const int *ao_loc = envs->ao_loc;
    const int i0 = ao_loc[ish];
    const int j0 = ao_loc[jsh];
    const int k0 = ao_loc[ksh];
    const int l0 = ao_loc[lsh];
    const int i1 = ao_loc[ish+1];
    const int j1 = ao_loc[jsh+1];
    const int k1 = ao_loc[ksh+1];
    const int l1 = ao_loc[lsh+1];
    const int di = i1 - i0;
    const int dj = j1 - j0;
    const int dk = k1 - k0;
    const int dl = l1 - l0;
    const int dijkl = di * dj * dk * dl;

    double *eri = buf;
    double *bufL = eri + dijkl;
    double *cache = bufL + dijkl;
    int empty = 1;
    int i, n;
    int jL_idx, lL_idx, jL, lL;
    int jsh_atm_id, ksh_atm_id, lsh_atm_id;
    int rj_of, rk_of, rl_of;
    int nimgs_ij = np0_ij->nimgs;
    int nimgs_kl = np0_kl->nimgs;
    double cell_id_ij[3], cell_id_kl[3];
    double q_cond_ij, q_cond_kl;
    double kl_cutoff;
    double *rij, *rkl;
    double rijkl[3], shift[3], shift_frac[3];
    double shift_min[3] = {0, 0, 0};
    //for (n = 0; n < dijkl; n++) {
    //    eri[n] = 0;
    //}
    memset(eri, 0, dijkl*sizeof(double));

    int shls[4] = {ish, jsh, ksh, lsh};
    double dist, dist_min;

    //const char TRANS_N = 'N';
    //const double D1 = 1;
    //const double D0 = 0;
    //const int I1 = 1;
    //const int dim = 3;
    const int nshift = 27;
    for (jL_idx = 0; jL_idx < nimgs_ij; jL_idx++) {
        q_cond_ij = (np0_ij->q_cond)[jL_idx];
        //if (q_cond_ij < cutoff) {
        //    continue;
        //}
        //kl_cutoff = cutoff / q_cond_ij;
        jL = (np0_ij->Ls_list)[jL_idx];
        rij = np0_ij->center + jL_idx * 3;
        get_cell_index(cell_id_ij, rij, b);

        jsh_atm_id = bas[ATOM_OF+jsh*BAS_SLOTS];
        rj_of = atm[PTR_COORD+jsh_atm_id*ATM_SLOTS];
        shift_bas(env_loc, env, Ls, rj_of, jL);

        for (lL_idx = 0; lL_idx < nimgs_kl; lL_idx++) {
            q_cond_kl = (np0_kl->q_cond)[lL_idx];
            //if (q_cond_kl < kl_cutoff) {
            //    continue;
            //}
            if (q_cond_ij*q_cond_kl*dm_cond < cutoff) {
                continue;
            }
            lL = (np0_kl->Ls_list)[lL_idx];
            lsh_atm_id = bas[ATOM_OF+lsh*BAS_SLOTS];
            rl_of = atm[PTR_COORD+lsh_atm_id*ATM_SLOTS];
            shift_bas(env_loc, env, Ls, rl_of, lL);

            rkl = np0_kl->center + lL_idx * 3;
            get_cell_index(cell_id_kl, rkl, b);
            _rij(rijkl, rkl, rij);
            dist_min = _r_norm(rijkl);
            for (i = 0; i < nshift; i++) {
                shift_frac[0] = cell_id_ij[0] + ishift[i*3+0] - cell_id_kl[0];
                shift_frac[1] = cell_id_ij[1] + ishift[i*3+1] - cell_id_kl[1];
                shift_frac[2] = cell_id_ij[2] + ishift[i*3+2] - cell_id_kl[2];
                //dgemm_wrapper(TRANS_N, TRANS_N, dim, I1, dim,
                //              D1, a, dim, shift_frac, dim, D0, shift, dim);
                _frac_to_xyz(shift, shift_frac, a);
                rijkl[0] = rkl[0] + shift[0] - rij[0];
                rijkl[1] = rkl[1] + shift[1] - rij[1];
                rijkl[2] = rkl[2] + shift[2] - rij[2];
                dist = _r_norm(rijkl);
                if (dist < dist_min) {
                    dist_min = dist;
                    _r_copy(shift_min, shift);
                }
            }
            ksh_atm_id = bas[ATOM_OF+ksh*BAS_SLOTS];
            rk_of = atm[PTR_COORD+ksh_atm_id*ATM_SLOTS];
            shift_bas_r(env_loc, env, shift_min, rk_of);
            shift_bas_r(env_loc, env_loc, shift_min, rl_of);

            if ((*intor)(bufL, NULL, shls, atm, natm,
                          bas, nbas, env, cintopt, cache)) {
                for (n = 0; n < dijkl; n++) {
                    printf("%.15f\n", bufL[n]);
                    eri[n] += bufL[n];
                }
                empty = 0;
            }
        }
    }
    return !empty;
}

void PBCDFT_contract_k_s1(int (*intor)(), double *vk, double *dms, int ndm, int nao,
                          double *buf, int ish, int jsh, int ksh, int lsh,
                          int jsh_ref, int ksh_ref, int lsh_ref, int nbas_ref,
                          NeighborPair *np0_ij, NeighborPair *np0_kl,
                          double *Ls, double *a, double *b, double *ishift,
                          CVHFOpt *vhfopt, IntorEnvs *envs, double* env_loc)
{
    double direct_scf_cutoff = vhfopt->direct_scf_cutoff;
    double dm_cond = vhfopt->dm_cond[jsh_ref*nbas_ref+ksh_ref];
    //if (dm_cond < direct_scf_cutoff) {
    //    return;
    //} else {
    //    direct_scf_cutoff /= dm_cond;
    //}
    if (!_assemble_eris(intor, buf, ish, jsh, ksh, lsh, np0_ij, np0_kl,
                        dm_cond, direct_scf_cutoff, Ls,
                        a, b, ishift, vhfopt, envs, env_loc)) {
        return;
    }

    const int *ao_loc = envs->ao_loc;
    const size_t i0 = ao_loc[ish];
    const size_t j0 = ao_loc[jsh];
    const size_t k0 = ao_loc[ksh];
    const size_t l0 = ao_loc[lsh];
    const size_t i1 = ao_loc[ish+1];
    const size_t j1 = ao_loc[jsh+1];
    const size_t k1 = ao_loc[ksh+1];
    const size_t l1 = ao_loc[lsh+1];

    int idm, n;
    int i, j, k, l;
    int j_ref, k_ref, l_ref; 
    const size_t nao2 = (size_t)nao * nao;
    double sjk, qijkl;
    double *pdm = dms;
    for (idm = 0; idm < ndm; idm++) {
        n = 0;
        for (l = l0; l < l1; l++) {
            l_ref = l % nao;
            for (k = k0; k < k1; k++) {
                k_ref = k % nao;
                for (j = j0; j < j1; j++) {
                    j_ref = j % nao;
                    sjk = pdm[j_ref*nao+k_ref];
                    for (i = i0; i < i1; i++, n++) {
                        qijkl = buf[n];
                        vk[i*nao+l_ref] += qijkl * sjk;
                    } 
                } 
            }
        }
        vk += nao2;
        pdm += nao2;
    }
}

void PBCDFT_direct_drv(void (*fdot)(), int (*intor)(), double *out,
                       double *dms, int ndm, int nao,
                       NeighborList** neighbor_list,
                       int *shls_slice, int *ao_loc,
                       CINTOpt *cintopt, CVHFOpt *vhfopt,
                       double *Ls, double *a, double *b,
                       int *atm, int natm, int *bas, int nbas, int nbas_ref,
                       double *env, int nenv)
{
    IntorEnvs envs = {natm, nbas, atm, bas, env, shls_slice, ao_loc,
                      NULL, cintopt, 1};

    const size_t ish0 = shls_slice[0];
    const size_t ish1 = shls_slice[1];
    const size_t jsh0 = shls_slice[2];
    const size_t jsh1 = shls_slice[3];
    const size_t ksh0 = shls_slice[4];
    const size_t ksh1 = shls_slice[5];
    const size_t lsh0 = shls_slice[6];
    const size_t lsh1 = shls_slice[7];
    const size_t nish = ish1 - ish0;
    const size_t njsh = jsh1 - jsh0;
    const size_t nksh = ksh1 - ksh0;
    const size_t nlsh = lsh1 - lsh0;

    const size_t nij = nish * njsh;
    const size_t nao2 = ((size_t)nao) * nao;
    const int di = GTOmax_shell_dim(ao_loc, shls_slice, 1);
    const int cache_size = GTOmax_cache_size(int2e_sph, shls_slice, 4,
                                             atm, natm, bas, nbas, env);
    const NeighborList *nl0 = *neighbor_list;
    double *ishift = malloc(sizeof(double) * 27 * 3);
    get_nearest_shift_vector(ishift, a, 27);
#pragma omp parallel
{
    size_t ij, n;
    int i, j, k, l, idm;
    int j_ref, k_ref, l_ref;
    double *v_priv = calloc(nao2*ndm, sizeof(double));
    double *buf = malloc(sizeof(double) * (di*di*di*di*2 + cache_size));
    double *env_loc = malloc(sizeof(double)*nenv);
    NPdcopy(env_loc, env, nenv);

    NeighborPair *np0_ij, *np0_kl;
    #pragma omp for schedule(dynamic)
    for (ij = 0; ij < nij; ij++) {
        i = ij / njsh + ish0;
        j = ij % njsh + jsh0;
        j_ref = j % nbas_ref;
        np0_ij = (nl0->pairs)[i*nbas_ref + j_ref];
        if (np0_ij->nimgs > 0) {
            for (k = ksh0; k < ksh1; k++) {
                k_ref = k % nbas_ref;
                for (l = lsh0; l < lsh1; l++) {
                    l_ref = l % nbas_ref;
                    np0_kl = (nl0->pairs)[k_ref*nbas_ref + l_ref];
                    if (np0_kl->nimgs > 0) {
                        (*fdot)(intor, v_priv, dms, ndm, nao,
                                buf, i, j, k, l, j_ref, k_ref, l_ref, nbas_ref,
                                np0_ij, np0_kl,
                                Ls, a, b, ishift, vhfopt, &envs, env_loc);
                    }
                }
            }
        }
    }

    double *pout;
    #pragma omp critical
    {
        pout = out;
        for (idm = 0; idm < ndm; idm++) {
            for (n = 0; n < nao2; n++) {
                pout[n] += v_priv[n];
            }
            pout += nao2;
        }
    }
    free(buf);
    free(v_priv);
}
}

void PBCDFT_set_int2e_q_cond(int (*intor)(), CINTOpt *cintopt,
                             NeighborList** neighbor_list, double* Ls, 
                             int *shls_slice, int *ao_loc, int *atm, int natm,
                             int *bas, int nbas, int nbas_ref, double *env, int nenv)
{
    // suppose atm, bas, env = conc_env(cell._atm, cell._bas, cell._env,
    //                                  cell._atm, cell._bas, cell._env)
    // nbas_ref = len(cell._bas)
    const int ish0 = shls_slice[0];
    const int ish1 = shls_slice[1];
    assert(ish1 <= nbas_ref);
    const int jsh0 = shls_slice[2];
    const int jsh1 = shls_slice[3];
    assert(jsh0 >= nbas_ref);
    const size_t nish = ish1 - ish0;
    const size_t njsh = jsh1 - jsh0;
    const size_t nij = nish * njsh;
    int shls_tmp[2];
    shls_tmp[0] = MIN(ish0, jsh0-nbas_ref);
    shls_tmp[1] = MAX(ish1, jsh1-nbas_ref);
    const int cache_size = GTOmax_cache_size(intor, shls_tmp, 1,
                                             atm, natm, bas, nbas, env);

    NeighborList *nl0 = *neighbor_list;
    double alpha_diffuse[nbas_ref];
#pragma omp parallel
{
    int ibas, ipgf, npgf, exp_of;
    double ai, aj, aij;
    #pragma omp for schedule(dynamic, 4)
    for (ibas = 0; ibas < nbas_ref; ibas++) {
        npgf = bas[NPRIM_OF+ibas*BAS_SLOTS];
        exp_of = bas[PTR_EXP+ibas*BAS_SLOTS];
        ai = env[exp_of];
        for (ipgf = 1; ipgf < npgf; ipgf++) {
            ai = MIN(ai, env[exp_of+ipgf]);
        }
        alpha_diffuse[ibas] = ai;
    }

    NeighborPair *np0_ij;
    double qtmp, tmp;
    size_t ij, i, j, di, dj, ish, jsh, jsh_ref;
    int iL, iL_idx, nimgs;
    int ish_atm_id, jsh_atm_id, ri_of, rj_of;
    int shls[4];
    double *ri, *rj;
    double *cache = malloc(sizeof(double) * cache_size);
    di = 0;
    for (ish = 0; ish < nbas_ref; ish++) {
        dj = ao_loc[ish+1] - ao_loc[ish];
        di = MAX(di, dj);
    }
    double *buf = malloc(sizeof(double) * di*di*di*di);
    double *env_loc = malloc(sizeof(double)*nenv);
    NPdcopy(env_loc, env, nenv);

    #pragma omp for schedule(dynamic)
    for (ij = 0; ij < nij; ij++) {
        ish = ij / njsh + ish0;
        jsh = ij % njsh + jsh0;
        jsh_ref = jsh % nbas_ref;
        np0_ij = (nl0->pairs)[ish*nbas_ref + jsh_ref];
        nimgs = np0_ij->nimgs;
        if (nimgs > 0) {
            if (np0_ij->q_cond) {
                free(np0_ij->q_cond);
            }
            np0_ij->q_cond = malloc(sizeof(double) * nimgs);
            if (np0_ij->center) {
                free(np0_ij->center);
            }
            np0_ij->center = malloc(sizeof(double) * nimgs*3);
            di = ao_loc[ish+1] - ao_loc[ish];
            dj = ao_loc[jsh+1] - ao_loc[jsh];
            shls[0] = ish;
            shls[1] = jsh;
            shls[2] = ish;
            shls[3] = jsh;

            ai = alpha_diffuse[ish];
            aj = alpha_diffuse[jsh_ref];
            aij = ai + aj;

            ish_atm_id = bas[ATOM_OF+ish*BAS_SLOTS];
            ri_of = atm[PTR_COORD+ish_atm_id*ATM_SLOTS];
            ri = env + ri_of;
            jsh_atm_id = bas[ATOM_OF+jsh*BAS_SLOTS];
            rj_of = atm[PTR_COORD+jsh_atm_id*ATM_SLOTS];
            for (iL_idx = 0; iL_idx < np0_ij->nimgs; iL_idx++){
                iL = (np0_ij->Ls_list)[iL_idx];
                shift_bas(env_loc, env, Ls, rj_of, iL);
                rj = env_loc + rj_of;
                (np0_ij->center)[iL_idx*3]   = (ai * ri[0] + aj * rj[0]) / aij;
                (np0_ij->center)[iL_idx*3+1] = (ai * ri[1] + aj * rj[1]) / aij;
                (np0_ij->center)[iL_idx*3+2] = (ai * ri[2] + aj * rj[2]) / aij;

                qtmp = 1e-100;
                if (0 != (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env_loc,
                                  cintopt, cache)) {
                    for (i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++) {
                            tmp = fabs(buf[i+di*j+di*dj*i+di*dj*di*j]);
                            qtmp = MAX(qtmp, tmp);
                        } 
                    }
                    qtmp = sqrt(qtmp);
                }
                (np0_ij->q_cond)[iL_idx] = qtmp;
            }
        }
    }
    free(buf);
    free(cache);
    free(env_loc);
}
}
