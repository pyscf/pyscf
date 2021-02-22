import logging
import numpy as np

#import pyscf
#import pyscf.mp
import pyscf.lib
from pyscf.mp.mp2 import _mo_without_core
from pyscf.pbc import tools
from pyscf.mp.mp2 import _ChemistsERIs as _ChemistsERIs_mp2
from pyscf.cc.rccsd import _ChemistsERIs as _ChemistsERIs_cc

from .util import *

__all__ = ["ao2mo"]

log = logging.getLogger(__name__)

# ERIs classes which contract j3c just-in-time (jit)

class _ChemistsERIs_mp2_j3c(_ChemistsERIs_mp2):

    def __init__(self, *args, j3c=None, **kwargs):
        super(_ChemistsERIs_mp2_j3c, self).__init__(*args, **kwargs)
        self._j3c = j3c

    @property
    def ovov(self):
        # j3c is (L|ov) for MP2
        l = r = self._j3c
        ovov = (einsum("Lij,Lkl->ijkl", l.real, r.real)
              + einsum("Lij,Lkl->ijkl", l.imag, r.imag))
        return ovov

    # Only here such that parent classes ovov initialization is ignored
    @ovov.setter
    def ovov(self, val):
        pass

class _ChemistsERIs_cc_j3c(_ChemistsERIs_cc):

    def __init__(self, *args, j3c=None, **kwargs):
        super(_ChemistsERIs_cc_j3c, self).__init__(*args, **kwargs)
        self._j3c = j3c

    def _get_ov_masks(self):
        nocc = self.nocc
        return np.s_[:nocc], np.s_[nocc:]

    def _get_eri_part(self, name):
        o, v = self._get_ov_masks()
        masks = {"o" : o, "v" : v}
        l = self._j3c[:,masks[name[0]],masks[name[1]]]
        r = self._j3c[:,masks[name[2]],masks[name[3]]]
        part = (einsum("Lij,Lkl->ijkl", l.real, r.real)
              + einsum("Lij,Lkl->ijkl", l.imag, r.imag))
        # TODO: Check imaginary part?
        return part

    @property
    def oooo(self):
        return self._get_eri_part("oooo")

    @property
    def ovoo(self):
        return self._get_eri_part("ovoo")

    @property
    def ovov(self):
        return self._get_eri_part("ovov")

    @property
    def oovv(self):
        return self._get_eri_part("oovv")

    @property
    def ovvo(self):
        return self._get_eri_part("ovvo")

    @property
    def ovvv(self):
        return self._get_eri_part("ovvv")

    @property
    def vvvv(self):
        return self._get_eri_part("vvvv")

    @oooo.setter
    def oooo(self, val):
        pass

    @ovoo.setter
    def ovoo(self, val):
        pass

    @ovov.setter
    def ovov(self, val):
        pass

    @oovv.setter
    def oovv(self, val):
        pass

    @ovvo.setter
    def ovvo(self, val):
        pass

    @ovvv.setter
    def ovvv(self, val):
        pass

    @vvvv.setter
    def vvvv(self, val):
        pass

#def ao2mo(cc, fock, mp2=False, imag_tol=1e-8, contract_jit="auto", max_mem=200.0):
def ao2mo(cc, fock, mp2=False, imag_tol=1e-8, contract_jit=False, max_mem=200.0):
    """
    Parameters
    ----------

    fock :
        Fock matrix *with* exxdiv correction.

    """
    mf = cc._scf

    mo_coeff = _mo_without_core(cc, cc.mo_coeff)
    norb = mo_coeff.shape[-1]
    nocc = cc.nocc
    nvir = norb - nocc

    # Determine memory needed for final (ij|kl)
    if mp2:
        mem4c = nocc**2 * nvir**2 * 8/1e9
    else:
        mem4c = norb**4 * 8/1e9
    log.debug("Memory needed for full (ij|kl)= %.3f GB", mem4c)
    if contract_jit == "auto":
        contract_jit = (mem4c > max_mem)
    # Never use JIT for MP2:
    if mp2:
        contract_jit = False
    log.debug("Contract j3c just-in-time: %r", contract_jit)

    if mp2:
        if contract_jit:
            eris = _ChemistsERIs_mp2_j3c()
        else:
            eris = _ChemistsERIs_mp2()
    else:
        if contract_jit:
            eris = _ChemistsERIs_cc_j3c()
        else:
            eris = _ChemistsERIs_cc()

    eris.mo_coeff = mo_coeff
    eris.e_hf = mf.e_tot
    eris.nocc = nocc
    eris.mo_energy = fock.diagonal().copy()
    eris.fock = fock.copy()
    # Remove exxdiv correction (important for CCSD)
    if not mp2 and mf.exxdiv is not None:
        madelung = tools.madelung(mf.cell, mf.kpt)
        for i in range(eris.nocc):
            eris.fock[i,i] += madelung

    j3c = mf.with_df._cderi
    compact = (j3c.ndim == 2)
    assert (compact or j3c.ndim == 3)
    naux = j3c.shape[0]
    nao = j3c.shape[1] if not compact else int(np.sqrt(2*j3c.shape[1]))
    o, v = np.s_[:eris.nocc], np.s_[eris.nocc:]
    log.debug("Performing integral transformations using (L|ab) integrals.")
    log.debug("Number of auxiliaries= %d , AOs= %d , occ. MOs= %d , vir MOs= %d", naux, nao, nocc, nvir)
    log.debug("Memory for 3c-integrals (L|ab)= %.3f GB", j3c.nbytes / 1e9)
    if mp2:
        log.debug("Memory for (L|ij)= %.3f GB", naux*nocc*nvir * 16/1e9)

        if not compact:
            j3c_mo = einsum("Lab,ai,bj->Lij", j3c, mo_coeff[:,o], mo_coeff[:,v])
        else:
            j3c_mo = np.zeros((naux, nocc, nvir), dtype=j3c.dtype)
            # Cannot unpack entire j3c due to memory constraints
            for l in range(naux):
                j3c_mo[l] = einsum("ab,ai,bj->ij", pyscf.lib.unpack_tril(j3c[l]), mo_coeff[:,o], mo_coeff[:,v])

        # Test imaginary part (column 0 only)
        if np.iscomplexobj(j3c_mo):
            eri_im = (einsum("Lij,Lk->ijk", j3c_mo.real, j3c_mo[:,:,0].imag)
                    - einsum("Lij,Lk->ijk", j3c_mo.imag, j3c_mo[:,:,0].real))
            #if abs(eri_im).max() > imag_tol:
            #    log.error("Error: Large imaginary part in ERIs: %.3e", abs(eri_im).max())

            #eri_im = (einsum("Lij,Lkl->ijkl", j3c_mo.real, j3c_mo.imag)
            #        - einsum("Lij,Lkl->ijkl", j3c_mo.imag, j3c_mo.real))
            log.info("Imaginary part of (ij|kl=0): norm= %.3e , max= %.3e", np.linalg.norm(eri_im), abs(eri_im).max())
            del eri_im

        if not contract_jit:
            # Avoid storage of complex (ij|kl) [Assumes imaginary part must be zero!]
            eri = einsum("Lij,Lkl->ijkl", j3c_mo.real, j3c_mo.real
            if np.iscomplexobj(j3c_mo):
                 eri += einsum("Lij,Lkl->ijkl", j3c_mo.imag, j3c_mo.imag)
            eris.ovov = eri
        else:
            # Contract (L|ij) just-in-time (oooo, ovoo, ... defined in class)
            eris._j3c = j3c_mo

    # Coupled-cluster
    else:
        log.debug("Memory for (L|ij)= %.3f GB", naux*norb**2 * 16/1e9)

        if not compact:
            j3c_mo = einsum("Lab,ai,bj->Lij", j3c, mo_coeff, mo_coeff)
        else:
            j3c_mo = np.zeros((naux, norb, norb), dtype=j3c.dtype)
            # Cannot unpack entire j3c due to memory constraints
            for l in range(naux):
                j3c_mo[l] = einsum("ab,ai,bj->ij", pyscf.lib.unpack_tril(j3c[l]), mo_coeff, mo_coeff)

        # Test imaginary part (column 0 only)
        if np.iscomplexobj(j3c_mo):
            eri_im = (einsum("Lij,Lk->ijk", j3c_mo.real, j3c_mo[:,:,0].imag)
                    - einsum("Lij,Lk->ijk", j3c_mo.imag, j3c_mo[:,:,0].real))
            #if abs(eri_im).max() > imag_tol:
            #    log.error("Error: Large imaginary part in ERIs: %.3e", abs(eri_im).max())

            #eri_im = (einsum("Lij,Lkl->ijkl", j3c_mo.real, j3c_mo.imag)
            #        - einsum("Lij,Lkl->ijkl", j3c_mo.imag, j3c_mo.real))
            log.info("Imaginary part of (ij|kl=0): norm= %.3e , max= %.3e", np.linalg.norm(eri_im), abs(eri_im).max())
            del eri_im

        if not contract_jit:

            # Avoid storage of complex (ij|kl) [Assumes imaginary part must be zero!]
            eri = einsum("Lij,Lkl->ijkl", j3c_mo.real, j3c_mo.real)
            if np.iscomplexobj(j3c_mo):
                 eri += einsum("Lij,Lkl->ijkl", j3c_mo.imag, j3c_mo.imag)

            eris.oooo = eri[o,o,o,o].copy()
            eris.ovoo = eri[o,v,o,o].copy()
            eris.ovov = eri[o,v,o,v].copy()
            eris.oovv = eri[o,o,v,v].copy()
            eris.ovvo = eri[o,v,v,o].copy()
            eris.ovvv = eri[o,v,v,v].copy()
            eris.vvvv = eri[v,v,v,v].copy()

        # Contract (L|ij) just-in-time (oooo, ovoo, ... defined in class)
        else:
            eris._j3c = j3c_mo

    return eris
