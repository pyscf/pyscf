import numpy as np
from pyscf.lib import logger
from pyscf.grad import rhf as mol_rhf
from pyscf.grad.rhf import _write


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None, kpt=np.zeros(3)):
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    hcore_deriv = mf_grad.hcore_generator(mol, kpt)
    s1 = mf_grad.get_ovlp(mol, kpt)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')
    vhf = mf_grad.get_veff(mol, dm0, kpt)
    log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = np.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices [ia,2:]
        h1ao = hcore_deriv(ia)
        if np.sum(kpt) < 1e-9:
            de[k] += np.einsum('xij,ij->x', h1ao, dm0)
            de[k] += np.einsum('xij,ij->x', vhf[:,p0:p1], dm0[p0:p1]) * 2
            de[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2
        else:
            de[k] += np.einsum('xij,ji->x', h1ao, dm0)
            de[k] += np.einsum('xij,ij->x', vhf[:,p0:p1], dm0[p0:p1].conj())
            de[k] += np.einsum('xij,ij->x', vhf[:,p0:p1].conj(), dm0[p0:p1].conj())
            de[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1].conj())
            de[k] -= np.einsum('xij,ij->x', s1[:,p0:p1].conj(), dme0[p0:p1].conj())

        de[k] += mf_grad.extra_force(ia, locals())

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of electronic part')
        _write(log, mol, de, atmlst)
    return de


def get_ovlp(mol, kpt=np.zeros(3)):
    return -mol.pbc_intor('int1e_ipovlp', kpt=kpt)


def hcore_generator(mf_grad, mol=None, kpt=np.zeros(3)):
    if mol is None:
        mol = mf_grad.mol
    if len(mol._ecpbas) > 0:
        raise NotImplementedError
    if not mol.pseudo:
        raise NotImplementedError

    h1 = -mol.pbc_intor('int1e_ipkin', kpt=kpt)

    mydf = mf_grad.base.with_df
    aoslices = mol.aoslice_by_atom()
    kpts = kpt.reshape(-1,3)
    def hcore_deriv(atm_id):
        vpp = mydf.get_pp_nuc_grad(kpts=kpts, atm_id=atm_id)
        shl0, shl1, p0, p1 = aoslices[atm_id]
        vpp[:,p0:p1] += h1[:,p0:p1]
        vpp[:,:,p0:p1] += h1[:,p0:p1].transpose(0,2,1).conj()
        return vpp

    return hcore_deriv


def get_veff(mf_grad, mol, dm, kpt=np.zeros(3)):
    mf = mf_grad.base
    mydf = mf.with_df
    xc_code = getattr(mf, 'xc', None)
    kpts = kpt.reshape(-1,3)
    return -mydf.get_veff_ip1(dm, xc_code=xc_code, kpts=kpts)


class GradientsMixin(mol_rhf.GradientsMixin):
    '''Base class for Gamma-point nuclear gradient'''
    def grad_nuc(self, mol=None, atmlst=None):
        from .krhf import grad_nuc
        if mol is None: mol = self.mol
        return grad_nuc(mol, atmlst)

    def get_ovlp(self, mol=None, kpt=np.zeros(3)):
        if mol is None:
            mol = self.mol
        return get_ovlp(mol, kpt)

    hcore_generator = hcore_generator


class Gradients(GradientsMixin):
    '''Non-relativistic Gamma-point restricted Hartree-Fock gradients'''
    def get_veff(self, mol=None, dm=None, kpt=np.zeros(3)):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        return get_veff(self, mol, dm, kpt)

    make_rdm1e = mol_rhf.Gradients.make_rdm1e
    grad_elec = grad_elec
