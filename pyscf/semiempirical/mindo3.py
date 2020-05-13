#!/usr/bin/env python
#
# Modified based on MINDO/3 implementation in PyQuante-1.6
#

'''
MINDO/3

Ref:
[1] R. C. Bingham, M. J. Dewar, D. H. Lo, J. Am. Chem. Soc., 97, 1285 (1975)
[2] D. F. Lewis, Chem. Rev. 86, 1111 (1986).
'''

import copy
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf
from pyscf.data.elements import _symbol
from pyscf.semiempirical import mopac_param


@lib.with_doc(scf.hf.get_hcore.__doc__)
def get_hcore(mol):
    assert(not mol.has_ecp())
    nao = mol.nao

    atom_charges = mol.atom_charges()
    basis_atom_charges = atom_charges[mol._bas[:,gto.ATOM_OF]]

    basis_ip = []
    basis_u = []
    for i, z in enumerate(basis_atom_charges):
        l = mol.bas_angular(i)
        if l == 0:
            basis_ip.append(mopac_param.VS[z])
            basis_u.append(mopac_param.USS3[z])
        else:
            basis_ip.append(mopac_param.VP[z])
            basis_u.append(mopac_param.UPP3[z])

    ao_atom_charges = _to_ao_labels(mol, basis_atom_charges)
    ao_ip = _to_ao_labels(mol, basis_ip)

    # Off-diagonal terms
    hcore  = mol.intor('int1e_ovlp')
    hcore *= ao_ip[:,None] + ao_ip
    hcore *= _get_beta0(ao_atom_charges[:,None], ao_atom_charges)

    # U term 
    hcore[numpy.diag_indices(nao)] = _to_ao_labels(mol, basis_u)

    # Nuclear attraction
    gamma = _get_gamma(mol)
    z_eff = mopac_param.CORE[atom_charges]
    vnuc = numpy.einsum('ij,j->i', gamma, z_eff)

    aoslices = mol.aoslice_by_atom()
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        idx = numpy.arange(p0, p1)
        hcore[idx,idx] -= vnuc[ia]
    return hcore


@lib.with_doc(scf.hf.get_jk.__doc__)
def get_jk(mol, dm):
    dm = numpy.asarray(dm)
    dm_shape = dm.shape
    nao = dm_shape[-1]

    dm = dm.reshape(-1,nao,nao)
    vj = numpy.zeros_like(dm)
    vk = numpy.zeros_like(dm)

    # One-center contributions to the J/K matrices
    atom_charges = mol.atom_charges()
    jk_ints = {z: _get_jk_1c_ints(z) for z in set(atom_charges)}

    aoslices = mol.aoslice_by_atom()
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z = atom_charges[ia]
        j_ints, k_ints = jk_ints[z]

        dm_blk = dm[:,p0:p1,p0:p1]
        idx = numpy.arange(p0, p1)

        # J[i,i] = (ii|jj)*dm_jj
        vj[:,idx,idx] = numpy.einsum('ij,xjj->xi', j_ints, dm_blk)
        # J[i,j] = (ij|ij)*dm_ji +  (ij|ji)*dm_ij
        vj[:,p0:p1,p0:p1] += 2*k_ints * dm_blk

        # K[i,i] = (ij|ji)*dm_jj
        vk[:,idx,idx] = numpy.einsum('ij,xjj->xi', k_ints, dm_blk)
        # K[i,j] = (ii|jj)*dm_ij + (ij|ij)*dm_ji
        vk[:,p0:p1,p0:p1] += (j_ints+k_ints) * dm_blk

    # Two-center contributions to the J/K matrices
    gamma = _get_gamma(mol)
    pop_atom = [numpy.einsum('tii->t', dm[:,p0:p1,p0:p1])
                for p0, p1 in aoslices[:,2:]]
    vj_diag = numpy.einsum('ij,jt->ti', gamma, pop_atom)

    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        idx = numpy.arange(p0, p1)
        vj[:,idx,idx] += vj_diag[:,ia].reshape(-1,1)

        for ja, (q0, q1) in enumerate(aoslices[:,2:]):
            vk[:,p0:p1,q0:q1] += gamma[ia,ja] * dm[:,p0:p1,q0:q1]

    vj = vj.reshape(dm_shape)
    vk = vk.reshape(dm_shape)
    return vj, vk


def energy_nuc(mol):
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()

    distances = numpy.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    distances_in_AA = distances * lib.param.BOHR
    # numerically exclude atomic self-interaction terms
    distances_in_AA[numpy.diag_indices(mol.natm)] = 1e60

    # one atom is H, another atom is N or O
    where_NO = (atom_charges == 7) | (atom_charges == 8)
    mask = (atom_charges[:,None] == 1) & where_NO
    mask = mask | mask.T
    scale = alpha = _get_alpha(atom_charges[:,None], atom_charges)
    scale[mask] *= numpy.exp(-distances_in_AA[mask])
    scale[~mask] = numpy.exp(-alpha[~mask] * distances_in_AA[~mask])

    gamma = _get_gamma(mol)
    z_eff = mopac_param.CORE[atom_charges]
    e_nuc = .5 * numpy.einsum('i,ij,j->', z_eff, gamma, z_eff)
    e_nuc += .5 * numpy.einsum('i,j,ij,ij->', z_eff, z_eff, scale,
                               mopac_param.E2/distances_in_AA - gamma)
    return e_nuc


def get_init_guess(mol):
    '''Average occupation density matrix'''
    aoslices = mol.aoslice_by_atom()
    dm_diag = numpy.zeros(mol.nao)
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z_eff = mopac_param.CORE[mol.atom_charge(ia)]
        dm_diag[p0:p1] = float(z_eff) / (p1-p0)
    return numpy.diag(dm_diag)


def energy_tot(mf, dm=None, h1e=None, vhf=None):
    mol = mf._mindo_mol
    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + mf.energy_nuc()
    e_ref = _get_reference_energy(mol)

    mf.e_heat_formation = e_tot * mopac_param.HARTREE2KCAL + e_ref
    logger.debug(mf, 'E(ref) = %.15g  Heat of Formation = %.15g kcal/mol',
                 e_ref, mf.e_heat_formation)
    return e_tot.real


class RMINDO3(scf.hf.RHF):
    '''RHF-MINDO/3 for closed-shell systems'''
    def __init__(self, mol):
        scf.hf.RHF.__init__(self, mol)
        self.conv_tol = 1e-5
        self.e_heat_formation = None
        self._mindo_mol = _make_mindo_mol(mol)
        self._keys.update(['e_heat_formation'])

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
            self._mindo_mol = _make_mindo_mol(mol)
        return self

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self._mindo_mol = _make_mindo_mol(mol)
        return self

    def get_ovlp(self, mol=None):
        return numpy.eye(self._mindo_mol.nao)

    def get_hcore(self, mol=None):
        return get_hcore(self._mindo_mol)

    @lib.with_doc(get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None: dm = self.make_rdm1()
        return get_jk(self._mindo_mol, dm)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        with lib.temporary_env(self, mol=self._mindo_mol):
            return scf.hf.get_occ(self, mo_energy, mo_coeff)

    def get_init_guess(self, mol=None, key='minao'):
        return get_init_guess(self._mindo_mol)

    def energy_nuc(self):
        return energy_nuc(self._mindo_mol)

    energy_tot = energy_tot

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        scf.hf.RHF._finalize(self)

        # e_heat_formation was generated in SOSCF object.
        if (getattr(self, '_scf', None) and
            getattr(self._scf, 'e_heat_formation', None)):
            self.e_heat_formation = self._scf.e_heat_formation

        logger.note(self, 'Heat of formation = %.15g kcal/mol, %.15g Eh',
                    self.e_heat_formation,
                    self.e_heat_formation/mopac_param.HARTREE2KCAL)
        return self

    density_fit = None
    x2c = x2c1e = sfx2c1e = None

    def nuc_grad_method(self):
        from . import rmindo3_grad
        return rmindo3_grad.Gradients(self)


class UMINDO3(scf.uhf.UHF):
    '''UHF-MINDO/3 for open-shell systems'''
    def __init__(self, mol):
        scf.uhf.UHF.__init__(self, mol)
        self.conv_tol = 1e-5
        self.e_heat_formation = None
        self._mindo_mol = _make_mindo_mol(mol)
        self._keys.update(['e_heat_formation'])

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self._mindo_mol = _make_mindo_mol(mol)
        self.nelec = self._mindo_mol.nelec
        return self

    def get_ovlp(self, mol=None):
        return numpy.eye(self._mindo_mol.nao)

    def get_hcore(self, mol=None):
        return get_hcore(self._mindo_mol)

    @lib.with_doc(get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None: dm = self.make_rdm1()
        return get_jk(self._mindo_mol, dm)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        with lib.temporary_env(self, mol=self._mindo_mol):
            return scf.uhf.get_occ(self, mo_energy, mo_coeff)

    def get_init_guess(self, mol=None, key='minao'):
        dm = get_init_guess(self._mindo_mol) * .5
        return numpy.stack((dm,dm))

    def energy_nuc(self):
        return energy_nuc(self._mindo_mol)

    energy_tot = energy_tot

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        scf.uhf.UHF._finalize(self)

        # e_heat_formation was generated in SOSCF object.
        if (getattr(self, '_scf', None) and
            getattr(self._scf, 'e_heat_formation', None)):
            self.e_heat_formation = self._scf.e_heat_formation

        logger.note(self, 'Heat of formation = %.15g kcal/mol, %.15g Eh',
                    self.e_heat_formation,
                    self.e_heat_formation/mopac_param.HARTREE2KCAL)
        return self

    density_fit = None
    x2c = x2c1e = sfx2c1e = None

    def nuc_grad_method(self):
        from . import umindo3_grad
        return umindo3_grad.Gradients(self)


def _make_mindo_mol(mol):
    assert(not mol.has_ecp())
    def make_sto_6g(n, l, zeta):
        es = mopac_param.gexps[(n, l)]
        cs = mopac_param.gcoefs[(n, l)]
        return [l] + [(e*zeta**2, c) for e, c in zip(es, cs)]

    def principle_quantum_number(charge):
        if charge < 3:
            return 1
        elif charge < 10:
            return 2
        elif charge < 18:
            return 3
        else:
            return 4

    mindo_mol = copy.copy(mol)
    atom_charges = mindo_mol.atom_charges()
    atom_types = set(atom_charges)
    basis_set = {}
    for charge in atom_types:
        n = principle_quantum_number(charge)
        l = 0
        sto_6g_function = make_sto_6g(n, l, mopac_param.ZS3[charge])
        basis = [sto_6g_function]

        if charge > 2:  # include p functions
            l = 1
            sto_6g_function = make_sto_6g(n, l, mopac_param.ZP3[charge])
            basis.append(sto_6g_function)

        basis_set[_symbol(int(charge))] = basis
    mindo_mol.basis = basis_set

    z_eff = mopac_param.CORE[atom_charges]
    mindo_mol.nelectron = int(z_eff.sum() - mol.charge)

    mindo_mol.build(0, 0)
    return mindo_mol


def _to_ao_labels(mol, labels):
    ao_loc = mol.ao_loc
    degen = ao_loc[1:] - ao_loc[:-1]
    ao_labels = [[label]*n for label, n in zip(labels, degen)]
    return numpy.hstack(ao_labels)

def _get_beta0(atnoi,atnoj):
    "Resonanace integral for coupling between different atoms"
    return mopac_param.BETA3[atnoi-1,atnoj-1]

def _get_alpha(atnoi,atnoj):
    "Part of the scale factor for the nuclear repulsion"
    return mopac_param.ALP3[atnoi-1,atnoj-1]

def _get_jk_1c_ints(z):
    if z < 3:  # H, He
        j_ints = numpy.zeros((1,1))
        k_ints = numpy.zeros((1,1))
        j_ints[0,0] = mopac_param.GSSM[z]
    else:
        j_ints = numpy.zeros((4,4))
        k_ints = numpy.zeros((4,4))
        p_diag_idx = ((1, 2, 3), (1, 2, 3))
        # px,py,pz cross terms
        p_off_idx = ((1, 1, 2, 2, 3, 3), (2, 3, 1, 3, 1, 2))

        j_ints[0,0] = mopac_param.GSSM[z]
        j_ints[0,1:] = j_ints[1:,0] = mopac_param.GSPM[z]
        j_ints[p_off_idx] = mopac_param.GP2M[z]
        j_ints[p_diag_idx] = mopac_param.GPPM[z]

        k_ints[0,1:] = k_ints[1:,0] = mopac_param.HSPM[z]
        k_ints[p_off_idx] = mopac_param.HP2M[z]
    return j_ints, k_ints

def _get_gamma(mol, F03=mopac_param.F03):
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    distances = numpy.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    distances_in_AA = distances * lib.param.BOHR

    rho = numpy.array([mopac_param.E2/F03[z] for z in atom_charges])
    gamma = mopac_param.E2 / numpy.sqrt(distances_in_AA**2 +
                                        (rho[:,None] + rho)**2 * .25)
    gamma[numpy.diag_indices(mol.natm)] = 0  # remove self-interaction terms
    return gamma


def _get_reference_energy(mol):
    '''E(Ref) = heat of formation - energy of atomization (kcal/mol)'''
    atom_charges = mol.atom_charges()
    Hf =  mopac_param.EHEAT3[atom_charges].sum()
    Eat = mopac_param.EISOL3[atom_charges].sum()
    return Hf - Eat * mopac_param.EV2KCAL


if __name__ == '__main__':
    mol = gto.M(atom=[(8,(0,0,0)),(1,(1.,0,0)),(1,(0,1.,0))])
    mf = RMINDO3(mol).run(conv_tol=1e-6)
    print(mf.e_heat_formation - -48.82621264564841)

    mol = gto.M(atom=[(8,(0,0,0)),(1,(1.,0,0))], spin=1)
    mf = UMINDO3(mol).run(conv_tol=1e-6)
    print(mf.e_heat_formation - 18.08247965492137)

    mol = gto.M(atom=[(6,(0,0,0)),(1,(1.,0,0)),(1,(0,1.,0)),
                      (1,(0,0,1.)),(1,(0,0,-1.))])
    mf = RMINDO3(mol).run(conv_tol=1e-6)
    print(mf.e_heat_formation - 75.76019731515225)
