#!/usr/bin/env python
# constrained DFT
# written by Zhihao Cui. zcui@caltech.edu

r'''
Constrained DFT (cDFT) is a method to control the locality of electron density
during the SCF calculation. By constraining the electron density or occupation
on a particular orbital, atom, or functional group, cDFT provides a solution to
study the charge transfer problems. One type of constraints is to integrate the
electron density with certain real space weight function and make its
expectation value equal to a given value (see Wu. PRA, 72(2005), 024502)

    .. math::

        \int w(\mathbf{r}) \rho(\mathbf{r}) d\mathbf{r} = N_c

This example shows another type of constraints which controls the electron
population (Mulliken population with localized orbitals) during the SCF
iterations

    .. math::

        \sum_{p} \gamma_{pq} S_{qp} = N_c

When incorporating this constraint with the HF/KS method, a Lagrange multiplier
V_c for the constraint is used in the energy minimization procedure
:math:`E + V_c (\sum_{p}gamma_{pq} S_{qp} - N_c)`. The constraints lead to an
extra term in the Fock matrix. This can be achieved by modifying the
:func:`get_fock` method of SCF object as shown by the code below.

Since the constraints are based on population analysis, it has close relation to
the population method and quality of the localized orbitals. The code
demonstrated in this example supports four localization schemes: Lowdin
orthogonalization, meta-Lowdin orthogonalization, intrinsic atomic orbitals,
natural atomic orbitals.
'''

import numpy as np
import scipy.linalg as la
import copy
from functools import reduce
from pyscf import gto, scf, lo, dft, lib
from pyscf.pbc.scf import khf


def get_localized_orbitals(mf, lo_method, mo=None):
    if mo is None:
        mo = mf.mo_coeff

    if not isinstance(mf, khf.KSCF):
        mol = mf.mol
        s1e = mf.get_ovlp()

        if lo_method.lower() == 'lowdin' or lo_method.lower() == 'meta_lowdin':
            C = lo.orth_ao(mf, 'meta_lowdin', s=s1e)
            C_inv = np.dot(C.conj().T,s1e)
            if isinstance(mf, scf.hf.RHF):
                C_inv_spin = C_inv
            else:
                C_inv_spin = np.array([C_inv]*2)

        elif lo_method == 'iao':
            s1e = mf.get_ovlp()
            pmol = mf.mol.copy()
            pmol.build(False, False, basis='minao')
            if isinstance(mf, scf.hf.RHF):
                mo_coeff_occ = mf.mo_coeff[:,mf.mo_occ>0]
                C = lo.iao.iao(mf.mol, mo_coeff_occ)
                # Orthogonalize IAO
                C = lo.vec_lowdin(C, s1e)
                C_inv = np.dot(C.conj().T,s1e)
                C_inv_spin = C_inv
            else:
                mo_coeff_occ_a = mf.mo_coeff[0][:,mf.mo_occ[0]>0]
                mo_coeff_occ_b = mf.mo_coeff[1][:,mf.mo_occ[1]>0]
                C_a = lo.iao.iao(mf.mol, mo_coeff_occ_a)
                C_b = lo.iao.iao(mf.mol, mo_coeff_occ_b)
                C_a = lo.vec_lowdin(C_a, s1e)
                C_b = lo.vec_lowdin(C_b, s1e)
                C_inv_a = np.dot(C_a.T, s1e)
                C_inv_b = np.dot(C_b.T, s1e)
                C_inv_spin = np.array([C_inv_a, C_inv_b])

        elif lo_method == 'nao':
            C = lo.orth_ao(mf, 'nao')
            C_inv = np.dot(C.conj().T,s1e)
            if isinstance(mf, scf.hf.RHF):
                C_inv_spin = C_inv
            else:
                C_inv_spin = np.array([C_inv]*2)

        else:
            raise NotImplementedError("UNDEFINED LOCAL ORBITAL TYPE, EXIT...")

        mo_lo = np.einsum('...jk,...kl->...jl', C_inv_spin, mo)
        return C_inv_spin, mo_lo

    else:
        cell = mf.cell
        s1e = mf.get_ovlp()

        if lo_method.lower() == 'lowdin' or lo_method.lower() == 'meta_lowdin':
            nkpt = len(mf.kpts)
            C_arr = []
            C_inv_arr = []
            for i in range(nkpt):
                C_curr = lo.orth_ao(mf, 'meta_lowdin',s=s1e[i])
                C_inv_arr.append(np.dot(C_curr.conj().T,s1e[i]))
            C_inv_arr = np.array(C_inv_arr)
            if isinstance(mf, scf.hf.RHF):
                C_inv_spin = C_inv_arr
            else:
                C_inv_spin = np.array([C_inv_arr]*2)
        else:
            raise NotImplementedError("CONSTRUCTING...EXIT")

        mo_lo = np.einsum('...jk,...kl->...jl', C_inv_spin, mo)
        return C_inv_spin, mo_lo

def pop_analysis(mf, mo_on_loc_ao, disp=True, full_dm=False):
    '''
    population analysis for local orbitals.
    return dm_lo

    mf should be a converged object
    full_rdm = False: return the diagonal element of dm_lo
    disp = True: show all the population to screen
    '''
    dm_lo = mf.make_rdm1(mo_on_loc_ao, mf.mo_occ)

    if isinstance(mf, khf.KSCF):
        nkpt = len(mf.kpts)
        dm_lo_ave = np.einsum('...ijk->...jk', dm_lo)/float(nkpt)
        dm_lo = dm_lo_ave

    if disp:
        mf.mulliken_pop(mf.mol, dm_lo, np.eye(mf.mol.nao_nr()))

    if full_dm:
        return dm_lo
    else:
        return np.einsum('...ii->...i', dm_lo)


# get the matrix which should be added to the fock matrix, due to the lagrange multiplier V_lagr (in separate format)
def get_fock_add_cdft(constraints, V, C_ao2lo_inv):
    '''
    mf is a pre-converged mf object, with NO constraints.

    F_ao_new=F_ao_old + C^{-1}.T * V_diag_lo * C^{-1}
    F_add is defined as C^{-1}.T * V_diag_lo * C^{-1}

    C is the transformation matrix of BASIS, from ao to lo. |LO> = |AO> * C
    NOTE: C should be pre-orthogonalized, i.e. C^T S C = I
    and thus C^{-1} = C.T * S
    '''

    V_lagr = constraints.sum2separated(V)
    sites_a, sites_b = constraints.unique_sites()
    if isinstance(mf, scf.hf.RHF):
        C_ao2lo_a = C_ao2lo_b = C_ao2lo_inv
    else:
        C_ao2lo_a, C_ao2lo_b = C_ao2lo_inv

    if not isinstance(mf, khf.KSCF):
        V_a = np.einsum('ip,i,iq->pq', C_ao2lo_a[sites_a].conj(), V_lagr[0], C_ao2lo_a[sites_a])
        V_b = np.einsum('ip,i,iq->pq', C_ao2lo_b[sites_b].conj(), V_lagr[1], C_ao2lo_b[sites_b])
    else:
        V_a = np.einsum('kip,i,kiq->kpq', C_ao2lo_a[:,sites_a].conj(), V_lagr[0], C_ao2lo_a[:,sites_a])
        V_b = np.einsum('kip,i,kiq->kpq', C_ao2lo_b[:,sites_b].conj(), V_lagr[1], C_ao2lo_b[:,sites_b])

    if isinstance(mf, scf.hf.RHF):
        return V_a + V_b
    else:
        return np.array((V_a, V_b))


def W_cdft(mf, constraints, V_c, orb_pop):
    '''get value of functional W (= V * constraint)'''
    if isinstance(mf, scf.hf.RHF):
        pop_a = pop_b = orb_pop * .5
    else:
        pop_a, pop_b = orb_pop

    N_c = constraints.nelec_required
    sites_a, sites_b = constraints.unique_sites()
    N_cur = pop_a[sites_a], pop_b[sites_b]
    N_cur_sum = constraints.separated2sum(N_cur)[1]
    return np.einsum('i,i', V_c, N_cur_sum - N_c)

# get gradient of W, as well as return the current population of selected orbitals
def jac_cdft(mf, constraints, V_c, orb_pop):
    if isinstance(mf, scf.hf.RHF):
        pop_a = pop_b = orb_pop * .5
    else:
        pop_a, pop_b = orb_pop

    N_c = constraints.nelec_required
    sites_a, sites_b = constraints.unique_sites()
    N_cur = np.array([pop_a[sites_a],pop_b[sites_b]]).real
    N_cur_sum = constraints.separated2sum(N_cur)[1]
    return N_cur_sum - N_c, N_cur_sum

# get the hessian of W, w.r.t. V_lagr
def hess_cdft(mf, constraints, V_c, mo_on_loc_ao):
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    de_ov_a = mo_energy[0][mo_occ[0]>0][:,None] - mo_energy[0][mo_occ[0]==0]
    de_ov_b = mo_energy[1][mo_occ[1]>0][:,None] - mo_energy[1][mo_occ[1]==0]
    de_ov_a[de_ov_a == 0] = 1e-18
    de_ov_b[de_ov_b == 0] = 1e-18

    sites_a, sites_b = constraints.unique_sites()
    orb_o_a = mo_on_loc_ao[0][sites_a][:,mo_occ[0] > 0]  # Alpha occupied orbitals
    orb_v_a = mo_on_loc_ao[0][sites_a][:,mo_occ[0] ==0]  # Alpha virtual  orbitals
    orb_o_b = mo_on_loc_ao[1][sites_b][:,mo_occ[1] > 0]  # Beta  occupied orbitals
    orb_v_b = mo_on_loc_ao[1][sites_b][:,mo_occ[1] ==0]  # Beta  virtual  orbitals

    hess_a = np.einsum('pi,pa,qa,qi,ia->pq',
                       orb_o_a.conj(), orb_v_a,
                       orb_v_a.conj(), orb_o_a, 1./de_ov_a)
    hess_a = hess_a + hess_a.conj()

    hess_b = np.einsum('pi,pa,qa,qi,ia->pq',
                       orb_o_b.conj(), orb_v_b,
                       orb_v_b.conj(), orb_o_b, 1./de_ov_b)
    hess_b = hess_b + hess_b.conj()

    t_a, t_b = constraints.site_to_constraints_transform_matrix()
    hess_arr  = np.einsum('pq,pi,qj->ij', hess_a, t_a, t_a)
    hess_arr += np.einsum('pq,pi,qj->ij', hess_b, t_b, t_b)
    return hess_arr


# main function for cdft
# mf : pre-converged mf object
# V_0 : initial guess of lagrange multipliers
# orb_idx: orbital index for orbital to be constrained
# alpha : newton step
# lo_method: localization method, one of 'lowdin', 'meta-lowdin', 'iao', 'nao'
# diis_pos: 3 choices: post, pre, both
# diis_type: 3 choices: use gradient of error vectors, use subsequent diff as error vector, no DIIS
def cdft(mf, constraints, V_0=None, lo_method='lowdin', alpha=0.2, tol=1e-5,
         constraints_tol=1e-3, maxiter=200, C_inv=None, verbose=4,
         diis_pos='post', diis_type=1):

    mf.verbose = verbose
    mf.max_cycle = maxiter

    old_get_fock = mf.get_fock

    if V_0 is None:
        V_0 = np.zeros(constraints.get_n_constraints())
    constraints._final_V = V_0

    C_inv = get_localized_orbitals(mf, lo_method, mf.mo_coeff)[0]

    cdft_diis = lib.diis.DIIS()
    cdft_diis.space = 8

    def get_fock(h1e, s1e, vhf, dm, cycle=0, mf_diis=None):
        fock_0 = old_get_fock(h1e, s1e, vhf, dm, cycle, None)
        V_0 = constraints._final_V
        if mf_diis is None:
            fock_add = get_fock_add_cdft(constraints, V_0, C_inv)
            return fock_0 + fock_add

        cdft_conv_flag = False
        if cycle < 10:
            inner_max_cycle = 20
        else:
            inner_max_cycle = 50

        if verbose > 3:
            print("\nCDFT INNER LOOP:")

        fock_0 = old_get_fock(h1e, s1e, vhf, dm, cycle, None)
        fock_add = get_fock_add_cdft(constraints, V_0, C_inv)
        fock = fock_0 + fock_add #ZHC

        if diis_pos == 'pre' or diis_pos == 'both':
            for it in range(inner_max_cycle): # TO BE MODIFIED
                fock_add = get_fock_add_cdft(constraints, V_0, C_inv)
                fock = fock_0 + fock_add #ZHC

                mo_energy, mo_coeff = mf.eig(fock, s1e)
                mo_occ = mf.get_occ(mo_energy, mo_coeff)

                # Required by hess_cdft function
                mf.mo_energy = mo_energy
                mf.mo_coeff = mo_coeff
                mf.mo_occ = mo_occ

                if lo_method.lower() == 'iao':
                    mo_on_loc_ao = get_localized_orbitals(mf, lo_method, mo_coeff)[1]
                else:
                    mo_on_loc_ao = np.einsum('...jk,...kl->...jl', C_inv, mo_coeff)

                orb_pop = pop_analysis(mf, mo_on_loc_ao, disp=False)
                W_new = W_cdft(mf, constraints, V_0, orb_pop)
                jacob, N_cur = jac_cdft(mf, constraints, V_0, orb_pop)
                hess = hess_cdft(mf, constraints, V_0, mo_on_loc_ao)

                deltaV = get_newton_step_aug_hess(jacob,hess)
                #deltaV = np.linalg.solve (hess, -jacob)

                if it < 5 :
                    stp = min(0.05, alpha*0.1)
                else:
                    stp = alpha

                V = V_0 + deltaV * stp
                g_norm = np.linalg.norm(jacob)
                if verbose > 3:
                    print("  loop %4s : W: %.5e    V_c: %s     Nele: %s      g_norm: %.3e    "
                          % (it,W_new, V_0, N_cur, g_norm))
                if g_norm < tol and np.linalg.norm(V-V_0) < constraints_tol:
                    cdft_conv_flag = True
                    break
                V_0 = V

        if cycle > 1:
            if diis_type == 1:
                fock = cdft_diis.update(fock_0, scf.diis.get_err_vec(s1e, dm, fock)) + fock_add
            elif diis_type == 2:
                # TO DO difference < threshold...
                fock = cdft_diis.update(fock)
            elif diis_type == 3:
                fock = cdft_diis.update(fock, scf.diis.get_err_vec(s1e, dm, fock))
            else:
                print("\nWARN: Unknow CDFT DIIS type, NO DIIS IS USED!!!\n")

        if diis_pos == 'post' or diis_pos == 'both':
            cdft_conv_flag = False
            fock_0 = fock - fock_add
            for it in range(inner_max_cycle): # TO BE MODIFIED
                fock_add = get_fock_add_cdft(constraints, V_0, C_inv)
                fock = fock_0 + fock_add #ZHC

                mo_energy, mo_coeff = mf.eig(fock, s1e)
                mo_occ = mf.get_occ(mo_energy, mo_coeff)

                # Required by hess_cdft function
                mf.mo_energy = mo_energy
                mf.mo_coeff = mo_coeff
                mf.mo_occ = mo_occ

                if lo_method.lower() == 'iao':
                    mo_on_loc_ao = get_localized_orbitals(mf, lo_method, mo_coeff)[1]
                else:
                    mo_on_loc_ao = np.einsum('...jk,...kl->...jl', C_inv, mo_coeff)

                orb_pop = pop_analysis(mf, mo_on_loc_ao, disp=False)
                W_new = W_cdft(mf, constraints, V_0, orb_pop)
                jacob, N_cur = jac_cdft(mf, constraints, V_0, orb_pop)
                hess = hess_cdft(mf, constraints, V_0, mo_on_loc_ao)
                deltaV = np.linalg.solve (hess, -jacob)

                if it < 5 :
                    stp = min(0.05, alpha*0.1)
                else:
                    stp = alpha

                V = V_0 + deltaV * stp
                g_norm = np.linalg.norm(jacob)
                if verbose > 3:
                    print("  loop %4s : W: %.5e    V_c: %s     Nele: %s      g_norm: %.3e    "
                          % (it,W_new, V_0, N_cur, g_norm))
                if g_norm < tol and np.linalg.norm(V-V_0) < constraints_tol:
                    cdft_conv_flag = True
                    break
                V_0 = V

        if verbose > 0:
            print("CDFT W: %.5e   g_norm: %.3e    "%(W_new, g_norm))

        constraints._converged = cdft_conv_flag
        constraints._final_V = V_0
        return fock

    dm0 = mf.make_rdm1()
    mf.get_fock = get_fock
    mf.kernel(dm0)

    mo_on_loc_ao = get_localized_orbitals(mf, lo_method, mf.mo_coeff)[1]
    orb_pop = pop_analysis(mf, mo_on_loc_ao, disp=True)
    return mf, orb_pop


class Constraints(object):
    '''
    Attributes:
        site_indices: the orbital indices on which electron population to be
            constrained. Each element of site_indices is a list which has two
            items (first for spin alpha, second for spin beta). If the
            constraints are applied on alpha spin-density only, the second item
            should be set to None. For the constraints of beta spin-density, the
            first item should be None. If both items are specified, the
            population constraints will be applied to the spin-traced density.
        site_nelec: population constraints for each orbital. Each element is the
            number of electrons for the orbitals that are specified in site_indices.

    Examples:
        constraints.orbital_indices = [[2,2], [3]]
        constraints.spin_labels = [[0,1] , [1]]
        constraints.nelec_required = [1.5 , 0.5]

        correspond to two constraints:
        1. N_{alpha-MO_2} + N_{beta-MO_2} = 1.5
        2. N_{beta-MO_3} = 0.5
    '''
    def __init__(self, orbital_indices, spin_labels, nelec_required):
        self.orbital_indices = orbital_indices
        self.spin_labels = spin_labels
        self.nelec_required = np.asarray(nelec_required)
        assert(len(orbital_indices) == len(spin_labels) == len(nelec_required))

    def get_n_constraints(self):
        return len(self.nelec_required)

    def unique_sites(self):
        sites_a = []
        sites_b = []
        for group, spin_labels in zip(self.orbital_indices, self.spin_labels):
            for orbidx, spin in zip(group, spin_labels):
                if spin == 0:
                    sites_a.append(orbidx)
                else:
                    sites_b.append(orbidx)
        sites_a = np.sort(list(set(sites_a)))
        sites_b = np.sort(list(set(sites_b)))
        return sites_a, sites_b

    def site_to_constraints_transform_matrix(self):
        sites_a, sites_b = self.unique_sites()
        map_sites_a = dict(((v,k) for k,v in enumerate(sites_a)))
        map_sites_b = dict(((v,k) for k,v in enumerate(sites_b)))

        n_constraints = self.get_n_constraints()
        t_a = np.zeros((sites_a.size, n_constraints))
        t_b = np.zeros((sites_b.size, n_constraints))
        for k, group in enumerate(self.orbital_indices):
            for orbidx, spin in zip(group, self.spin_labels[k]):
                if spin == 0:
                    t_a[map_sites_a[orbidx],k] += 1
                else:
                    t_b[map_sites_b[orbidx],k] += 1
        return t_a, t_b

    def sum2separated(self, V_c):
        '''
        convert the format of constraint from a summation format (it allows several orbitals' linear combination)
        to the format each orbital is treated individually (also they are separated by spin)
        '''
        t_a, t_b = self.site_to_constraints_transform_matrix()
        V_c_a = np.einsum('pi,i->p', t_a, V_c)
        V_c_b = np.einsum('pi,i->p', t_b, V_c)
        return V_c_a, V_c_b

    def separated2sum(self, N_c):
        '''the inversion function for sum2separated'''
        t_a, t_b = self.site_to_constraints_transform_matrix()
        N_c_new = np.array([np.einsum('pi,p->i', t_a, N_c[0]),
                            np.einsum('pi,p->i', t_b, N_c[1])]).T

        N_c_sum = N_c_new[:,0] + N_c_new[:,1]

        # V_c on alpha-site if available, otherwise V_c on beta-site
        V_c_sum = [N_c_new[i,0] if 0 in spins else N_c_new[i,1]
                   for i,spins in enumerate(self.spin_labels)]
        return N_c_new, N_c_sum, V_c_sum


def get_newton_step_aug_hess(jac,hess):
    #lamb = 1.0 / alpha
    ah = np.zeros((hess.shape[0]+1,hess.shape[1]+1))
    ah[1:,0] = jac
    ah[0,1:] = jac.conj()
    ah[1:,1:] = hess

    eigval, eigvec = la.eigh(ah)
    idx = None
    for i in xrange(len(eigvec)):
        if abs(eigvec[0,i]) > 0.1 and eigval[i] > 0.0:
            idx = i
            break
    if idx is None:
        print("WARNING: ALL EIGENVALUESS in AUG-HESSIAN are NEGATIVE!!! ")
        return np.zeros_like(jac)
    deltax = eigvec[1:,idx] / eigvec[0,idx]
    return deltax


if __name__ == '__main__':
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = '''
    c   1.217739890298750 -0.703062453466927  0.000000000000000
    h   2.172991468538160 -1.254577209307266  0.000000000000000
    c   1.217739890298750  0.703062453466927  0.000000000000000
    h   2.172991468538160  1.254577209307266  0.000000000000000
    c   0.000000000000000  1.406124906933854  0.000000000000000
    h   0.000000000000000  2.509154418614532  0.000000000000000
    c  -1.217739890298750  0.703062453466927  0.000000000000000
    h  -2.172991468538160  1.254577209307266  0.000000000000000
    c  -1.217739890298750 -0.703062453466927  0.000000000000000
    h  -2.172991468538160 -1.254577209307266  0.000000000000000
    c   0.000000000000000 -1.406124906933854  0.000000000000000
    h   0.000000000000000 -2.509154418614532  0.000000000000000
    '''
    mol.basis = '631g'
    mol.spin=0
    mol.build()

    mf = scf.UHF(mol)
#    mf = dft.UKS(mol)
#    mf.xc = 'pbe,pbe'
    mf.conv_tol=1e-9
    mf.verbose=0
    mf.max_cycle=100
    mf.run()

    idx = mol.search_ao_label('C 2pz') # find all idx for carbon
    # there are 4 constraints:
    # 1. N_alpha_C0 + N_beta_C0 + N_beta_C1 = 1.5
    # 2. N_alpha_C2 = 0.5
    # 3. N_beta_C2 = 0.5
    orbital_indices = [[idx[0],idx[0],idx[1]], [idx[2]], [idx[2]]]
    spin_labels = [[0,1,1], [0], [1]]
    nelec_required = [1.5, .5, .5]
    constraints = Constraints(orbital_indices, spin_labels, nelec_required)
    mf, dm_pop = cdft(mf, constraints, lo_method='lowdin', verbose=4)
