'''
Restricted direct ring-CCD.

See also the relevant dicussions in https://github.com/pyscf/pyscf/issues/1149

Ref: Scuseria et al., J. Chem. Phys. 129, 231101 (2008)
Ref: Masios et al., Phys. Rev. Lett. 131, 186401 (2023)

Original Source:
Modified from pyscf/cc/rccsd_slow.py

Contact: 
caochangsu@gmail.com
'''

from functools import reduce
import numpy as np
from pyscf.cc import ccsd
from pyscf import lib
from pyscf.ao2mo import _ao2mo

einsum = lib.einsum

def update_amps(cc, t1, t2, eris):
    # Ref: Masios et al., Phys. Rev. Lett. 131, 186401 (2023) Eq.(3)
    nocc, nvir = t1.shape
    t1[:] = 0.0

    eris_ovvo = np.asarray(eris.ovvo)
    eris_ovov = np.asarray(eris.ovov)

    t2new = 2*einsum('iack, kjcb->iajb', eris_ovvo, t2)
    t2new += 2*einsum('ikac, jbck->iajb', t2, eris_ovvo)
    tmp = einsum('ikac, kcld->iald', t2, eris_ovov)
    t2new += 4*einsum('iald, ljdb->iajb', tmp, t2)
    t2new = t2new.transpose(0,2,1,3)

    t2new += np.asarray(eris.ovov).conj().transpose(0,2,1,3).copy()

    mo_e = eris.fock.diagonal().real
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t2new /= eijab

    return t1, t2new

def energy(cc, t1, t2, eris):
    # Ref: Scuseria et al., J. Chem. Phys. 129, 231101 (2008) Eq.(11)
    nocc, nvir = t1.shape
    fock = eris.fock
    eris_ovov = np.asarray(eris.ovov)
    e = 2*einsum('ijab,iajb', t2, eris_ovov)
    return e.real

def init_amps(cc, eris):
    nocc = cc.nocc
    mo_e = eris.fock.diagonal().real
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    t1 = np.zeros_like(eris.fock[:nocc,nocc:])

    eris_ovov = np.asarray(eris.ovov)
    t2 = eris_ovov.transpose(0,2,1,3).conj() / eijab
    cc.emp2  = 2*einsum('ijab,iajb', t2, eris_ovov)
    lib.logger.info(cc, 'Init t2, dMP2 energy = %.15g', cc.emp2)
    return cc.emp2, t1, t2

def ao2mo(cc, mo_coeff=None):
    return _ChemistsERIs(cc, mo_coeff)

class _ChemistsERIs:
    def __init__(self, cc, mo_coeff=None, method='incore'):
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, cc.mo_coeff)
        else:
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, mo_coeff)
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
        self.fock = reduce(np.dot, (mo_coeff.T, fockao, mo_coeff))

        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc
        # eri1 = ao2mo.incore.full(cc._scf._eri, mo_coeff)
        # eri1 = ao2mo.restore(1, eri1, nmo)
        mo = cc._scf.mo_coeff
        Lpq = cc._scf._cderi
        ijslice = (0, nocc, nocc, nmo)
        Lov = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', mosym='s1') 
        eris_ovov = Lov.T @ Lov
        self.ovov = eris_ovov.reshape(nocc, nvir, nocc, nvir)
        self.ovvo = self.ovov.transpose(0,1,3,2).copy()
        self.Lov = Lov


class drCCD(ccsd.CCSD):
    init_amps = init_amps
    update_amps = update_amps
    energy = energy
    ao2mo = ao2mo


if __name__ == '__main__':
    from pyscf import gto, dft, scf
    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = 'H 0 0 0; F 0 0 1.1'
    mol.basis = 'ccpvdz'

    mol.build()
    mf = scf.RHF(mol).density_fit()
    mf.kernel()

    mycc = drCCD(mf)
    e_drccd, _, t2 = mycc.kernel()
    from pyscf.gw import rpa
    myrpa = rpa.RPA(mf)
    e_rpa = myrpa.kernel()
    print(f"Difference between drccd and drpa: {e_drccd-e_rpa} Ha." )
