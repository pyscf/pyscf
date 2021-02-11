import logging
import numpy as np

#import pyscf
#import pyscf.mp
import pyscf.lib
from pyscf.mp.mp2 import _mo_without_core
from pyscf.pbc import tools
from .util import *

__all__ = ["ao2mo"]

log = logging.getLogger(__name__)

def ao2mo(cc, fock, mp2=False, imag_tol=1e-8):
    """
    Parameters
    ----------

    fock :
        Fock matrix *with* exxdiv correction.

    """
    mf = cc._scf

    if mp2:
        from pyscf.mp.mp2 import _ChemistsERIs
    else:
        from pyscf.cc.rccsd import _ChemistsERIs
    eris = _ChemistsERIs()

    eris.mo_coeff = mo_coeff = _mo_without_core(cc, cc.mo_coeff)
    eris.e_hf = mf.e_tot
    eris.nocc = cc.nocc
    eris.mo_energy = fock.diagonal().copy()
    eris.fock = fock.copy()
    # Remove exxdiv correction (important for CCSD)
    if not mp2 and mf.exxdiv is not None:
        madelung = tools.madelung(mf.cell, mf.kpt)
        for i in range(eris.nocc):
            eris.fock[i,i] += madelung

    j3c = mf.with_df._cderi
    norb = len(eris.mo_energy)
    compact = (j3c.ndim == 2)
    assert (compact or j3c.ndim == 3)
    nocc = eris.nocc
    nvir = norb-nocc
    naux = j3c.shape[0]
    nao = j3c.shape[1] if not compact else int(np.sqrt(2*j3c.shape[1]))
    o, v = np.s_[:eris.nocc], np.s_[eris.nocc:]
    log.debug("Performing integral transformations using (L|ab) integrals.")
    log.debug("Number of auxiliaries= %d , AOs= %d , occ. MOs= %d , vir MOs= %d", naux, nao, nocc, nvir)
    log.debug("Memory for 3c-integrals (L|ab)= %.3f GB", j3c.nbytes / 1e9)
    if mp2:
        log.debug("Memory for (L|ij)= %.3f GB", naux*nocc*nvir * 16/1e9)
        log.debug("Memory for (ij|kl)= %.3f GB", nocc**2*nvir**2 * 8/1e9)

        if not compact:
            j3c_mo = einsum("Lab,ai,bj->Lij", j3c, mo_coeff[:,o], mo_coeff[:,v])
        else:
            j3c_mo = np.zeros((naux, nocc, nvir), dtype=j3c.dtype)
            # Cannot unpack entire j3c due to memory constraints
            for l in range(naux):
                j3c_mo[l] = einsum("ab,ai,bj->ij", pyscf.lib.unpack_tril(j3c[l]), mo_coeff[:,o], mo_coeff[:,v])

        #eri = einsum("Lij,Lkl->ijkl", j3c_mo.conj(), j3c_mo)
        #if abs(eri.imag).max() > imag_tol:
        #    log.error("Error: Large imaginary part in ERIs: %.3e", abs(eri.imag).max())
        # Avoid storage of complex (ij|kl) [Assumes imaginary part must be zero!]
        eri = (einsum("Lij,Lkl->ijkl", j3c_mo.real, j3c_mo.real)
             - einsum("Lij,Lkl->ijkl", j3c_mo.imag, j3c_mo.imag))
        # Test imaginary part (column 0 only)
        eri_im = (einsum("Lij,Lk->ijk", j3c_mo.real, j3c_mo[:,:,0].imag)
                - einsum("Lij,Lk->ijk", j3c_mo.imag, j3c_mo[:,:,0].real))
        if abs(eri_im).max() > imag_tol:
            log.error("Error: Large imaginary part in ERIs: %.3e", abs(eri_im).max())

        eri = eri.real
        eris.ovov = eri
    else:
        log.debug("Memory for (L|ij)= %.3f GB", naux*norb**2 * 16/1e9)
        log.debug("Memory for (ij|kl)= %.3f GB", norb**4 * 8/1e9)

        if not compact:
            j3c_mo = einsum("Lab,ai,bj->Lij", j3c, mo_coeff, mo_coeff)
        else:
            j3c_mo = np.zeros((naux, norb, norb), dtype=j3c.dtype)
            # Cannot unpack entire j3c due to memory constraints
            for l in range(naux):
                j3c_mo[l] = einsum("ab,ai,bj->ij", pyscf.lib.unpack_tril(j3c[l]), mo_coeff, mo_coeff)

        #eri = einsum("Lij,Lkl->ijkl", j3c_mo.conj(), j3c_mo)
        #if abs(eri.imag).max() > imag_tol:
        #    log.error("Error: Large imaginary part in ERIs: %.3e", abs(eri.imag).max())
        #eri = eri.real
        # Avoid storage of complex (ij|kl) [Assumes imaginary part must be zero!]
        eri = (einsum("Lij,Lkl->ijkl", j3c_mo.real, j3c_mo.real)
             - einsum("Lij,Lkl->ijkl", j3c_mo.imag, j3c_mo.imag))
        # Test imaginary part (column 0 only)
        eri_im = (einsum("Lij,Lk->ijk", j3c_mo.real, j3c_mo[:,:,0].imag)
                - einsum("Lij,Lk->ijk", j3c_mo.imag, j3c_mo[:,:,0].real))
        if abs(eri_im).max() > imag_tol:
            log.error("Error: Large imaginary part in ERIs: %.3e", abs(eri_im).max())

        eris.oooo = eri[o,o,o,o].copy()
        eris.ovoo = eri[o,v,o,o].copy()
        eris.ovov = eri[o,v,o,v].copy()
        eris.oovv = eri[o,o,v,v].copy()
        eris.ovvo = eri[o,v,v,o].copy()
        eris.ovvv = eri[o,v,v,v].copy()
        eris.vvvv = eri[v,v,v,v].copy()

    return eris
