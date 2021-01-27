#!/usr/bin/env python

import numpy
from pyscf import lib
from pyscf import gto
from pyscf.grad import rhf as rhf_grad
from pyscf.semiempirical import mindo3
from pyscf.semiempirical import mopac_param


def grad_nuc(mol, atmlst=None):
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    natm = atom_charges.size

    dR = (atom_coords[:,None,:] - atom_coords) * lib.param.BOHR
    distances_in_AA = numpy.linalg.norm(dR, axis=2)
    # numerically exclude atomic self-interaction terms
    distances_in_AA[numpy.diag_indices(mol.natm)] = 1e60

    # one atom is H, another atom is N or O
    where_NO = (atom_charges == 7) | (atom_charges == 8)
    mask = (atom_charges[:,None] == 1) & where_NO
    mask = mask | mask.T
    scale = alpha = mindo3._get_alpha(atom_charges[:,None], atom_charges)
    scale[mask] *= numpy.exp(-distances_in_AA[mask])
    scale[~mask] = numpy.exp(-alpha[~mask] * distances_in_AA[~mask])

    z_eff = mopac_param.CORE[atom_charges]
    gamma = mindo3._get_gamma(mol)
    gamma1 = _get_gamma1_half(mol)

    #gs = .5 * numpy.einsum('i,sxij,j->sx', z_eff, _get_gamma1(mol), z_eff)
    gs = numpy.einsum('i,xij,j->ix', z_eff, gamma1, z_eff)

    alpha = mindo3._get_alpha(atom_charges[:,None], atom_charges)
    div = numpy.zeros((natm,natm))
    div[mask] = (-alpha[mask] * numpy.exp(-distances_in_AA[mask])
                 / distances_in_AA[mask])
    div[~mask] = (-alpha[~mask] * numpy.exp(-alpha[~mask] * distances_in_AA[~mask])
                  / distances_in_AA[~mask])
    #scale1 = numpy.zeros((natm,3,natm,natm))
    #for i in range(natm):
    #    v = dR[i] * div[i,:,None]
    #    scale1[i,:,i] = v.T
    #    scale1[i,:,:,i] = v.T
    #
    #gs += .5 * numpy.einsum('i,j,sxij,ij->sx', z_eff, z_eff, scale1,
    #                        mopac_param.E2/distances_in_AA - gamma)
    gs += numpy.einsum('i,j,ijx,ij,ij->ix', z_eff, z_eff, dR, div,
                       mopac_param.E2/distances_in_AA - gamma)

    div = -mopac_param.E2 / distances_in_AA**3
    div[numpy.diag_indices(natm)] = 0
    #t1 = numpy.zeros((natm,3,natm,natm))
    #for i in range(natm):
    #    v = dR[i] * div[i,:,None]
    #    t1[i,:,i] = v.T
    #    t1[i,:,:,i] = v.T
    #
    #gs += .5 * numpy.einsum('i,j,ij,sxij->sx', z_eff, z_eff, scale,
    #                        t1-_get_gamma1(mol))
    t1 = numpy.einsum('ijx,ij->xij', dR, div)
    gs += numpy.einsum('i,j,ij,xij->ix', z_eff, z_eff, scale, t1-gamma1)

    if atmlst is not None:
        gs = gs[atmlst]
    gs *= lib.param.BOHR
    return gs

def _get_gamma1(mol):
    natm = mol.natm
    gamma1_half = _get_gamma1_half(mol)
    gamma1 = numpy.zeros((natm,3,natm,natm))
    for i in range(natm):
        gamma1[i,:,i] = gamma1[i,:,:,i] = gamma1_half[:,i]
    return gamma1

def _get_gamma1_half(mol):
    '''gamma1_half[:,i,:] == gamma1[i,:,i,:] == gamma1[i,:,:,i]'''
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    natm = atom_charges.size

    dR = (atom_coords[:,None,:] - atom_coords) * lib.param.BOHR
    distances_in_AA = numpy.linalg.norm(dR, axis=2)

    rho = numpy.array([mopac_param.E2/mopac_param.F03[z] for z in atom_charges])
    div = -mopac_param.E2 / (distances_in_AA**2 + (rho[:,None] + rho)**2*.25)**1.5
    div[numpy.diag_indices(natm)] = 0  # remove self-interaction terms

    gamma1_dense = numpy.einsum('ijx,ij->xij', dR, div)
    return gamma1_dense


def hcore_generator(mf_grad, mol=None):
    mol = mf_grad.base._mindo_mol
    nao = mol.nao

    atom_charges = mol.atom_charges()
    basis_atom_charges = atom_charges[mol._bas[:,gto.ATOM_OF]]
    natm = atom_charges.size

    basis_ip = []
    for i, z in enumerate(basis_atom_charges):
        l = mol.bas_angular(i)
        if l == 0:
            basis_ip.append(mopac_param.VS[z])
        else:
            basis_ip.append(mopac_param.VP[z])

    ao_atom_charges = mindo3._to_ao_labels(mol, basis_atom_charges)
    ao_ip = mindo3._to_ao_labels(mol, basis_ip)

    # Off-diagonal terms
    hcore  = mol.intor('int1e_ipovlp', comp=3)
    hcore *= ao_ip[:,None] + ao_ip
    hcore *= mindo3._get_beta0(ao_atom_charges[:,None], ao_atom_charges)
    # int1e_ipovlp is computed in atomic unit. Scale it to AA
    hcore *= 1./ lib.param.BOHR

    # Nuclear attraction
    gamma1 = _get_gamma1_half(mol)
    z_eff = mopac_param.CORE[atom_charges]

    aoslices = mol.aoslice_by_atom()
    def hcore_deriv(atm_id):
        gamma1p = numpy.zeros((3,natm,natm))
        gamma1p[:,atm_id] = gamma1p[:,:,atm_id] = gamma1[:,atm_id]
        vnuc = numpy.einsum('xij,j->xi', gamma1p, z_eff)

        h1 = numpy.zeros((3,nao,nao))
        for ia, (p0, p1) in enumerate(aoslices[:,2:]):
            idx = numpy.arange(p0, p1)
            h1[:,idx,idx] -= vnuc[:,ia,None]

        p0, p1 = aoslices[atm_id,2:]
        h1[:,p0:p1,:] -= hcore[:,p0:p1]
        h1[:,:,p0:p1] -= hcore[:,p0:p1].transpose(0,2,1)
        return h1
    return hcore_deriv


def get_jk(mol, dm):
    dm = numpy.asarray(dm)
    dm_shape = dm.shape
    nao = dm_shape[-1]

    dm = dm.reshape(-1,nao,nao)
    ndm = dm.shape[0]
    vj = numpy.zeros((ndm,3,nao,nao))
    vk = numpy.zeros((ndm,3,nao,nao))

    # Two-center contributions to the J/K matrices
    gamma1 = _get_gamma1_half(mol)
    # Scaling integrals by .5 because rhf_grad.get_jk function only computes
    # the bra-derivatives. gamma is the diagonal part of the ERIs. gamma1 is
    # the full derivatives of the diagonal elements rho_ii, which include the
    # derivatives of ket functions.
    gamma1 *= .5

    aoslices = mol.aoslice_by_atom()
    pop_atom = [numpy.einsum('tii->t', dm[:,p0:p1,p0:p1])
                for p0, p1 in aoslices[:,2:]]
    vj_diag = numpy.einsum('xij,jt->txi', gamma1, pop_atom)

    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        idx = numpy.arange(p0, p1)
        vj[:,:,idx,idx] += vj_diag[:,:,ia,numpy.newaxis]

        for ja, (q0, q1) in enumerate(aoslices[:,2:]):
            ksub = numpy.einsum('x,tij->txij', gamma1[:,ia,ja], dm[:,p0:p1,q0:q1])
            vk[:,:,p0:p1,q0:q1] += ksub

    if ndm == 1:
        vj = vj[0]
        vk = vk[0]
    return vj, vk


class Gradients(rhf_grad.Gradients):
    get_hcore = None
    hcore_generator = hcore_generator

    def get_ovlp(self, mol=None):
        nao = self.base._mindo_mol.nao
        return numpy.zeros((3,nao,nao))

    def get_jk(self, mol=None, dm=None, hermi=0):
        if dm is None: dm = self.base.make_rdm1()
        vj, vk = get_jk(self.base._mindo_mol, dm)
        return vj, vk

    def grad_nuc(self, mol=None, atmlst=None):
        mol = self.base._mindo_mol
        return grad_nuc(mol, atmlst)

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        # grad_elec function use self.mol. However, self.mol points to the
        # input molecule with the input basis. MINDO/3 basis (in _mindo_mol)
        # should be passed to grad_elec.
        with lib.temporary_env(self, mol=self.base._mindo_mol):
            return rhf_grad.grad_elec(self, mo_energy, mo_coeff, mo_occ,
                                      atmlst) * lib.param.BOHR

Grad = Gradients


if __name__ == '__main__':
    from pyscf.data.nist import HARTREE2EV
    mol = gto.Mole()
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    mol.verbose = 0
    mol.build()
    mfs = mindo3.RMINDO3(mol).set(conv_tol=1e-8).as_scanner()
    mfs(mol)
    print(mfs.e_tot - -341.50046431149383/HARTREE2EV)

    mol1 = mol.copy()
    mol1.set_geom_([['O' , (0. , 0.     , 0.0001)],
                    [1   , (0. , -0.757 , 0.587)],
                    [1   , (0. , 0.757  , 0.587)]])
    mol2 = mol.copy()
    mindo_mol1 = mindo3._make_mindo_mol(mol1)
    mol2.set_geom_([['O' , (0. , 0.     ,-0.0001)],
                    [1   , (0. , -0.757 , 0.587)],
                    [1   , (0. , 0.757  , 0.587)]])
    mindo_mol2 = mindo3._make_mindo_mol(mol2)

    g1 = mfs.nuc_grad_method().kernel()
    e1 = mfs(mol1)
    e2 = mfs(mol2)
    print(abs((e1-e2)/0.0002*lib.param.BOHR - g1[0,2]))

    dgamma = _get_gamma1(mol)
    gamma1 = mindo3._get_gamma(mol1)
    gamma2 = mindo3._get_gamma(mol2)
    print(abs((gamma1 - gamma2)/0.0002 - dgamma[0,2,:,:]).max())
    dgd = _get_gamma1_half(mol)
    print(abs(dgamma[0,:,0] - dgd[:,0]).max())
    print(abs(dgamma[0,:,:,0] - dgd[:,0]).max())

    denuc = grad_nuc(mol)[0,2]
    enuc1 = mindo3.energy_nuc(mol1)
    enuc2 = mindo3.energy_nuc(mol2)
    print(abs((enuc1 - enuc2)/0.0002*lib.param.BOHR - denuc).max())

    fcore = hcore_generator(mindo3.RMINDO3(mol).nuc_grad_method())
    dh = fcore(0)[2]
    h1 = mindo3.get_hcore(mindo_mol1)
    h2 = mindo3.get_hcore(mindo_mol2)
    print(abs((h1 - h2)/0.0002 - dh).max())

    nao = mindo_mol1.nao
    numpy.random.seed(1)
    dm = numpy.random.random((nao,nao))
    dm = dm + dm.T
    djk = get_jk(mindo3.RMINDO3(mol)._mindo_mol, dm)
    dj = numpy.zeros((nao,nao))
    dk = numpy.zeros((nao,nao))
    dj[0:4  ]  = djk[0][2,0:4]
    dj[:,0:4] += djk[0][2,0:4].T
    dk[0:4  ]  = djk[1][2,0:4]
    dk[:,0:4] += djk[1][2,0:4].T
    jk1 = mindo3.get_jk(mindo_mol1, dm)
    jk2 = mindo3.get_jk(mindo_mol2, dm)
    print(abs(numpy.einsum('ij,ji->', dm, (jk1[0] - jk2[0])/0.0002 - 2*dj)).max())
    print(abs(numpy.einsum('ij,ji->', dm, (jk1[1] - jk2[1])/0.0002 - 2*dk)).max())
