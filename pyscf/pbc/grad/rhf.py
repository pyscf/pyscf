import ctypes
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rhf as mol_rhf
from pyscf.grad.rhf import _write
from pyscf.pbc.gto.pseudo import pp_int

libpbc = lib.load_library('libpbc')

def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None, kpt=np.zeros(3)):
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    s1 = mf_grad.get_ovlp(mol, kpt)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing Gradients of NR-HF Coulomb repulsion')
    vhf = mf_grad.get_veff(mol, dm0, kpt)
    log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)

    de = np.zeros((len(atmlst),3))
    if np.sum(kpt) < 1e-9:
        de += mf.with_df.vpploc_part1_nuc_grad(dm0, kpts=kpt.reshape(-1,3))
        de += pp_int.vpploc_part2_nuc_grad(mol, dm0)
        de += pp_int.vppnl_nuc_grad(mol, dm0)
        h1  = -mol.pbc_intor('int1e_ipkin', kpt=kpt)
        if mf.with_df.vpplocG_part1 is None or mf.with_df.pp_with_erf:
            h1 += -mf.with_df.get_vpploc_part1_ip1(kpts=kpt.reshape(-1,3))
        de += _contract_vhf_dm(mf_grad, h1+vhf, dm0, atmlst=atmlst) * 2
        de -= _contract_vhf_dm(mf_grad, s1, dme0, atmlst=atmlst) * 2
        #TODO extra_force need rewrite
    else:
        hcore_deriv = mf_grad.hcore_generator(mol, kpt)
        aoslices = mol.aoslice_by_atom()
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices [ia,2:]
            h1ao = hcore_deriv(ia)
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


def _contract_vhf_dm(mf_grad, vhf, dm, comp=3, atmlst=None):
    from pyscf.gto.mole import ao_loc_nr, ATOM_OF
    from pyscf.pbc.gto import build_neighbor_list_for_shlpairs, free_neighbor_list

    t0 = (logger.process_clock(), logger.perf_counter())

    mol = mf_grad.mol
    natm = mol.natm
    nbas = mol.nbas
    shls_slice = np.asarray([0,nbas,0,nbas], order="C", dtype=np.int32)
    ao_loc = np.asarray(ao_loc_nr(mol), order="C", dtype=np.int32)
    shls_atm = np.asarray(mol._bas[:,ATOM_OF].copy(), order="C", dtype=np.int32)

    de = np.zeros((natm,comp), order="C")
    vhf = np.asarray(vhf, order="C")
    dm = np.asarray(dm, order="C")

    neighbor_list = build_neighbor_list_for_shlpairs(mol, mol)
    func = getattr(libpbc, "contract_vhf_dm", None)
    try:
        func(de.ctypes.data_as(ctypes.c_void_p),
             vhf.ctypes.data_as(ctypes.c_void_p),
             dm.ctypes.data_as(ctypes.c_void_p),
             ctypes.byref(neighbor_list),
             shls_slice.ctypes.data_as(ctypes.c_void_p),
             ao_loc.ctypes.data_as(ctypes.c_void_p),
             shls_atm.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_int(comp), ctypes.c_int(natm),
             ctypes.c_int(nbas))
    except RuntimeError:
        raise
    free_neighbor_list(neighbor_list)

    if atmlst is not None:
        de = de[atmlst]

    logger.timer(mf_grad, '_contract_vhf_dm', *t0)
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
