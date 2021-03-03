"""AO to MO transformation routines for density-fitted three-center integrals
"""
import logging
import numpy as np

import pyscf.lib
from pyscf.mp.mp2 import _mo_without_core
from pyscf.mp.mp2 import _ChemistsERIs as _ChemistsERIs_mp2
from pyscf.cc.rccsd import _ChemistsERIs as _ChemistsERIs_cc
from pyscf.pbc import tools

from .util import einsum

MEM_UNITS = {
        # Bits
        "bit" : 1, "kbit" : 1/1e3, "mbit" : 1/1e6, "gbit" : 1/1e9, "tbit" : 1/1e12,
        # Bytes
        "b" : 8, "kb" : 8/1e3, "mb" : 8/1e6, "gb" : 8/1e9, "tb" : 8/1e12,
        }
DEFAULT_MEM_UNIT = "GB"

log = logging.getLogger(__name__)

__all__ = ["ao2mo_mp2", "ao2mo_ccsd"]

def j3c_to_mo_eris(j3c, mo_coeff, nocc, compact=False, ovov_only=False):
    """Convert three-center integrals (L|ab) to (ij|kl) for some set of MO coefficients.

    Parameters
    ----------
    j3c : ndarray
        Three-center integrals in AO basis.
    mo_coeff : ndarray
        MO coefficients.
    nocc : int
        Number of occupied orbitals.
    compact : bool, optional
    ovov_only : bool, optional
        Only tranform the occupied-virtual-occupied-virtual part of (ij|kl),
        as needed by MP2.

    Returns
    -------
    mo_eris : dict
    """
    if compact:
        raise NotImplementedError()
    j3c_compact = (j3c.ndim == 2)
    assert (j3c.ndim in (2, 3))
    norb = mo_coeff.shape[-1]
    nvir = norb - nocc
    naux = j3c.shape[0]
    o, v = np.s_[:nocc], np.s_[nocc:]
    co, cv = mo_coeff[:,o], mo_coeff[:,v]
    assert not np.iscomplexobj(mo_coeff)

    # AO->MO
    if not j3c_compact:
        j3c_ov = einsum("Lab,ai,bj->Lij", j3c, co, cv)
        if not ovov_only:
            j3c_oo = einsum("Lab,ai,bj->Lij", j3c, co, co)
            j3c_vv = einsum("Lab,ai,bj->Lij", j3c, cv, cv)
            # TEST:
            j3c_vo = einsum("Lab,ai,bj->Lij", j3c, cv, co)
    else:
        j3c_ov = np.zeros((naux, nocc, nvir), dtype=j3c.dtype)
        if not ovov_only:
            j3c_oo = np.zeros((naux, nocc, nocc), dtype=j3c.dtype)
            j3c_vv = np.zeros((naux, nvir, nvir), dtype=j3c.dtype)
            # TEST:
            j3c_vo = np.zeros((naux, nvir, nocc), dtype=j3c.dtype)
        # Avoid unpacking entire j3c at once
        #zfac = 2 if np.iscomplexobj(j3c) else 1
        #stepsize = int(500 / (max(nocc, nvir)**2 * zfac * 8/1e6))
        #log.debug("Stepsize= %d", stepsize)
        stepsize = 1
        for lmin, lmax in pyscf.lib.prange(0, naux, stepsize):
            l = np.s_[lmin:lmax]
            j3c_l = pyscf.lib.unpack_tril(j3c[l])
            j3c_ov[l] = einsum("Lab,ai,bj->Lij", j3c_l, co, cv)
            if not ovov_only:
                j3c_oo[l] = einsum("Lab,ai,bj->Lij", j3c_l, co, co)
                j3c_vv[l] = einsum("Lab,ai,bj->Lij", j3c_l, cv, cv)
                # TEST:
                j3c_vo[l] = einsum("Lab,ai,bj->Lij", j3c_l, cv, co)
    # TEST
    assert (ovov_only or np.allclose(j3c_ov, j3c.vo.transpose((0,2,1))))

    # 3c -> 4c
    mo_eris = {"ovov" : einsum("Lij,Lkl->ijkl", j3c_ov, j3c_ov)}
    if not ovov_only:
        mo_eris["oooo"] = einsum("Lij,Lkl->ijkl", j3c_oo, j3c_oo)
        mo_eris["ovoo"] = einsum("Lij,Lkl->ijkl", j3c_vo, j3c_oo)
        mo_eris["oovv"] = einsum("Lij,Lkl->ijkl", j3c_oo, j3c_vv)
        mo_eris["ovvo"] = einsum("Lij,Lkl->ijkl", j3c_ov, j3c_vo)
        mo_eris["ovvv"] = einsum("Lij,Lkl->ijkl", j3c_ov, j3c_vv)
        mo_eris["vvvv"] = einsum("Lij,Lkl->ijkl", j3c_vv, j3c_vv)

    #elif False:
    #    contract = "Lji,Lkl->ijkl"
    #    mo_eris = {"ovov" : einsum(contract, j3c_ov.conj(), j3c_ov)}
    #    if not ovov_only:
    #        mo_eris["oooo"] = einsum(contract, j3c_oo.conj(), j3c_oo)
    #        mo_eris["ovoo"] = einsum(contract, j3c_ov.conj(), j3c_oo)
    #        mo_eris["oovv"] = einsum(contract, j3c_oo.conj(), j3c_vv)
    #        mo_eris["ovvo"] = einsum(contract, j3c_ov.conj(), j3c_vo)         # Note transpose!
    #        mo_eris["ovvv"] = einsum(contract, j3c_ov.conj(), j3c_vv)
    #        mo_eris["vvvv"] = einsum(contract, j3c_vv.conj(), j3c_vv)

    #    mo_eris2 = {}
    #    for key, val in mo_eris.items():
    #        log.info("Max imaginary element in %s = %g", key, (abs(val.imag).max()))
    #        mo_eris2[key] = val.real
    #    mo_eris = mo_eris2

    #else:
    #    mo_eris = {"ovov" : einsum("Lij,Lkl->ijkl", j3c_ov.real, j3c_ov.real)}
    #    if np.iscomplexobj(j3c):
    #        mo_eris["ovov"] += einsum("Lij,Lkl->ijkl", j3c_ov.imag, j3c_ov.imag)
    #    # Remaining parts for CCSD
    #    if not ovov_only:
    #        mo_eris["oooo"] = einsum("Lij,Lkl->ijkl", j3c_oo.real, j3c_oo.real)
    #        mo_eris["ovoo"] = einsum("Lij,Lkl->ijkl", j3c_ov.real, j3c_oo.real)
    #        mo_eris["oovv"] = einsum("Lij,Lkl->ijkl", j3c_oo.real, j3c_vv.real)
    #        mo_eris["ovvo"] = einsum("Lij,Llk->ijkl", j3c_ov.real, j3c_ov.real)         # Note transpose!
    #        mo_eris["ovvv"] = einsum("Lij,Lkl->ijkl", j3c_ov.real, j3c_vv.real)
    #        mo_eris["vvvv"] = einsum("Lij,Lkl->ijkl", j3c_vv.real, j3c_vv.real)
    #        if np.iscomplexobj(j3c):
    #            mo_eris["oooo"] += einsum("Lij,Lkl->ijkl", j3c_oo.imag, j3c_oo.imag)
    #            mo_eris["ovoo"] += einsum("Lij,Lkl->ijkl", j3c_ov.imag, j3c_oo.imag)
    #            mo_eris["oovv"] += einsum("Lij,Lkl->ijkl", j3c_oo.imag, j3c_vv.imag)
    #            mo_eris["ovvo"] -= einsum("Lij,Llk->ijkl", j3c_ov.imag, j3c_ov.imag)    # Note tranpose + complex conjugate!
    #            mo_eris["ovvv"] += einsum("Lij,Lkl->ijkl", j3c_ov.imag, j3c_vv.imag)
    #            mo_eris["vvvv"] += einsum("Lij,Lkl->ijkl", j3c_vv.imag, j3c_vv.imag)

    return mo_eris

def _get_mem_mp2(nocc, nvir, compact=False, unit=DEFAULT_MEM_UNIT):
    if compact:
        raise NotImplementedError()
    mem = nocc**2 * nvir**2 * MEM_UNITS[unit.lower()]
    return mem

def _get_mem_ccsd(nocc, nvir, compact=False, unit=DEFAULT_MEM_UNIT):
    if compact:
        raise NotImplementedError()
    mem = (nocc**4 + nocc**3*nvir + 3*nocc**2*nvir**2 + nocc*nvir**3 + nvir**4)*MEM_UNITS[unit.lower()]
    return mem

def ao2mo_mp2(mp, fock):
    """Creates pyscf.mp compatible _ChemistsERIs object.

    Parameters
    ----------
    mp : pyscf.mp.mp2.MP2
        PySCF MP2 object.
    mo_energy : ndarray
        MO energies.

    Returns
    -------
    eris : pyscf.mp.mp2._ChemistERIs
        ERIs for MP2.
    """
    eris = _ChemistsERIs_mp2()
    mo_coeff = _mo_without_core(mp, mp.mo_coeff)
    eris.mo_coeff = mo_coeff
    eris.nocc = mp.nocc
    eris.e_hf = mp._scf.e_tot
    eris.mo_energy = fock.diagonal().copy()
    eris.fock = fock.copy()

    j3c = mp._scf.with_df._cderi
    g = j3c_to_mo_eris(j3c, mo_coeff, eris.nocc, ovov_only=True)
    for key, val in g.items():
        setattr(eris, key, val)

    return eris

def ao2mo_ccsd(cc, fock):
    """Creates pyscf.cc compatible _ChemistsERIs object.

    Parameters
    ----------
    cc : pyscf.cc.ccsd.CCSD
        PySCF CCSD object.
    fock : ndarray
        Must include exxdiv correction (will be removed).

    Returns
    -------
    eris : pyscf.cc.rccsd._ChemistERIs
        ERIs for CCSD.
    """
    scf = cc._scf
    eris = _ChemistsERIs_cc()
    mo_coeff = _mo_without_core(cc, cc.mo_coeff)
    eris.mo_coeff = mo_coeff
    eris.nocc = cc.nocc
    eris.e_hf = scf.e_tot
    eris.mo_energy = fock.diagonal().copy()
    eris.fock = fock.copy()
    # Remove EXXDIV correction from Fock matrix
    if scf.exxdiv is not None:
        madelung = tools.madelung(scf.cell, scf.kpt)
        for i in range(eris.nocc):
            eris.fock[i,i] += madelung

    j3c = scf.with_df._cderi
    g = j3c_to_mo_eris(j3c, mo_coeff, eris.nocc)
    for key, val in g.items():
        setattr(eris, key, val)

    return eris
