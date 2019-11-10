#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: James D. McClain
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import time
from functools import reduce
import numpy as np
import h5py

from pyscf import lib
import pyscf.ao2mo
from pyscf.lib import logger
from pyscf import __config__

from pyscf.pbc import scf
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import (get_nocc, get_nmo, padding_k_idx, 
                               padded_mo_coeff, get_frozen_mask)
from pyscf.pbc.tdscf.krhf import _get_e_ia

from pyscf.pbc import df
from pyscf.pbc import tools
from pyscf.pbc.cc.ccsd import _adjust_occ

# einsum = np.einsum
einsum = lib.einsum
direct_sum = lib.direct_sum

def kernel(cis, nroots=1, eris=None, kptlist=None, **kargs):
    """CIS excitation energy with k-point sampling.
    
    Arguments:
        cis {KCIS} -- A KCIS instance
    
    Keyword Arguments:
        nroots {int} -- Number of requested excitation energies (default: {1})
        eris {_CIS_ERIS} -- Depending on cis.direct, eris may 
            contain 4-center (cis.direct=False) or 3-center (cis.direct=True) 
            electron repulsion integrals (default: {None})            
        kptlist {list} -- A list of indices for k-shift, i.e. the exciton momentum. 
            Available k-shift indices depend on the k-point mesh. For example, 
            a 2 by 2 by 2 k-point mesh allows at most 8 k-shift values, which can
            be targeted by [0, 1, 2, 3, 4, 5, 6, 7]. When kptlist=None, all k-shift
            will be computed. (default: {None})
    
    Returns:
        tuple -- A tuple of excitation energies and corresponding eigenvectors
    """
    cpu0 = (time.clock(), time.time())
    log = logger.Logger(cis.stdout, cis.verbose)
    cis.dump_flags()

    if eris is None:
        eris = cis.ao2mo()

    nkpts = cis.nkpts
    nocc = cis.nocc
    nvir = cis.nmo - nocc
    dtype = eris.dtype

    if kptlist is None:
        kptlist = range(nkpts)

    r_size = nkpts * nocc * nvir
    nroots = min(nroots, r_size)

    evals = [None] * len(kptlist)
    evecs = [None] * len(kptlist)
    convs = [None] * len(kptlist)

    cpu1 = (time.clock(), time.time())
    for k, kshift in enumerate(kptlist):
        print("\nkshift =", kshift)

        if cis.davidson:
            # Davidson diagonalization
            matvec, diag = cis.gen_matvec(kshift, eris=eris)
            if diag.size != r_size:
                raise ValueError("Number of diagonal elements in H does not match r     vector size")

            guess = cis.get_init_guess(nroots, diag=diag)
            def precond(r, e0, x0):
                return r/(e0-diag+1e-12)

            eig = lib.davidson_nosym1
            conv, eigval, eigvec = eig(matvec, guess, precond, tol=cis.conv_tol, 
                                       max_cycle=cis.max_cycle, max_space=cis.max_space, 
                                       max_memory=cis.max_memory, nroots=nroots, verbose=cis.verbose)

        else:
            # Exact diagonalization            
            if not cis.build_full_H:
                H = np.zeros([r_size, r_size], dtype=dtype)
                for col in range(r_size):
                    vec = np.zeros(r_size, dtype=dtype)
                    vec[col] = 1.0
                    H[:, col] = cis_matvec_singlet(cis, vec, kshift, eris=eris)
            else:
                H = cis_H(cis, kshift, eris=eris)

            eigval, eigvec = np.linalg.eig(H)
            idx = eigval.argsort()[:nroots]
            eigval = eigval[idx]
            eigvec = eigvec[:, idx]

        for n in range(nroots):
            logger.info(cis, "CIS root %d E = %.16g", n, eigval[n])

        evals[k] = eigval
        evecs[k] = eigvec
    log.timer("CIS diagonalization", *cpu1)

    log.timer("CIS", *cpu0)
    return evals, evecs

def cis_matvec_singlet(cis, vector, kshift, eris=None):
    """Compute matrix-vector product of the Hamiltonion matrix and a CIS c
    oefficient vector, in the space of single excitation. 
    
    Arguments:
        cis {KCIS} -- A KCIS instance
        vector {1D array} -- CIS coefficient vector
        kshift {int} -- k-shift index. A k-shift vector is an exciton momentum. 
            Available k-shift indices depend on the k-point mesh. For example, 
            a 2 by 2 by 2 k-point mesh allows at most 8 k-shift values, which can
            be targeted by 0, 1, 2, 3, 4, 5, 6, or 7.
    
    Keyword Arguments:
        eris {_CIS_ERIS} -- Depending on cis.direct, eris may 
            contain 4-center (cis.direct=False) or 3-center (cis.direct=True) 
            electron repulsion integrals (default: {None})            
    
    Returns:
        1D array -- matrix-vector product of the Hamiltonion matrix and the 
            input vector. 
    """
    if eris is None:
        eris = cis.ao2mo()
    nkpts = cis.nkpts
    nocc = cis.nocc
    kconserv_r = cis.get_kconserv_r(kshift)

    r = cis.vector_to_amplitudes(vector)

    # Should use Fock diagonal elements to build (e_a - e_i) matrix
    epsilons = [eris.fock[k].diagonal().real for k in range(nkpts)]
        
    Hr = np.zeros_like(r)
    for ki in range(nkpts):
        ka = kconserv_r[ki]
        Hr[ki] += einsum('ia,a->ia', r[ki], epsilons[ka][nocc:])
        Hr[ki] -= einsum('ia,i->ia', r[ki], epsilons[ki][:nocc])
        
    if not cis.direct:
        for ki in range(nkpts):
            ka = kconserv_r[ki]
            # x: kj
            Hr[ki] += 2.0 * einsum("xjb,xajib->ia", r, eris.voov[ka, :, ki])
            Hr[ki] -= einsum("xjb,xjaib->ia", r, eris.ovov[:, ka, ki])
    else:
        for ki in range(nkpts):
            ka = kconserv_r[ki]
            for kj in range(nkpts):
                kb = kconserv_r[kj]
                # r_ia <- 2 r_jb (ai|jb) = 2 r_jb B^L_jb B^L_ai
                L = 2.0 * einsum("jb,Ljb->L", r[kj], eris.Lpq_mo[kj,kb][:, :nocc, nocc:])
                tmp = einsum("L,Lai->ia", L, eris.Lpq_mo[ka,ki][:, nocc:, :nocc])
        
                # r_ia <- - r_jb (ab|ji) = -r_jb B^L_ab B^L_ji
                Lja = -1.0 * einsum("jb,Lab->Lja", r[kj], eris.Lpq_mo[ka,kb][:, nocc:, nocc:])
                tmp += einsum("Lja,Lji->ia", Lja, eris.Lpq_mo[kj,ki][:, :nocc, :nocc])
                Hr[ki] += (1. / nkpts) * tmp

    vector = cis.amplitudes_to_vector(Hr)
    return vector

def cis_H(cis, kshift, eris=None):
    """Build full Hamiltonian matrix in the space of single excitation, 
    i.e. CIS Hamiltonian.
    
    Arguments:
        cis {KCIS} -- A KCIS instance
        kshift {int} -- k-shift index. A k-shift vector is an exciton momentum. 
            Available k-shift indices depend on the k-point mesh. For example, 
            a 2 by 2 by 2 k-point mesh allows at most 8 k-shift values, which can
            be targeted by 0, 1, 2, 3, 4, 5, 6, or 7.
    
    Keyword Arguments:
        eris {_CIS_ERIS} -- Depending on cis.direct, eris may 
            contain 4-center (cis.direct=False) or 3-center (cis.direct=True) 
            electron repulsion integrals (default: {None})            
    
    Raises:
        MemoryError: MemoryError will be raise if there is not enough space to
            store the full Hamiltonian matrix, which scales as Nk^2 O^2 V^2
    
    Returns:
        2D array -- the Hamiltonian matrix reshaped into (ki,i,a) by (kj,j,b)
    """
    cpu0 = (time.clock(), time.time())
    log = logger.Logger(cis.stdout, cis.verbose)

    if eris is None:
        eris = cis.ao2mo()
    nkpts = cis.nkpts
    nocc = cis.nocc
    nmo = cis.nmo
    nvir = nmo - nocc

    nov = nocc * nvir
    r_size = nkpts * nov

    memory_needed = (r_size ** 2) * 16 / 1e6
    memory_now = lib.current_memory()[0]
    if memory_needed + memory_now >= cis.max_memory:
        raise MemoryError("Not enough memory to store full CIS Hamiltonian")

    kconserv_r = cis.get_kconserv_r(kshift)
    dtype = eris.dtype
    epsilons = [eris.fock[k].diagonal().real for k in range(nkpts)]

    H = np.zeros((nkpts, nkpts, nov, nov), dtype=dtype)
    # <ia|H|jb> <- (esp_a - esp_i) \delta{i,j} \delta{a,b}
    for ki in range(nkpts):
        ka = kconserv_r[ki]
        diag_ia = direct_sum("a-i->ia", epsilons[ka][nocc:], epsilons[ki][:nocc])
        diag_ia = np.ravel(diag_ia)
        np.fill_diagonal(H[ki, ki], diag_ia)

    # <ia|H|jb> <- 2<ja|bi> - <ja|ib>
    if not cis.direct:
        for ki in range(nkpts):
            ka = kconserv_r[ki]
            for kj in range(nkpts):
                kb = kconserv_r[kj]
                # contribution from 2 <ja|bi> = 2 <aj|ib>
                tmp =  2. * eris.voov[ka, kj, ki].transpose(2,0,1,3)
                # contribution from -<ja|ib>
                tmp -= eris.ovov[kj, ka, ki].transpose(2,1,0,3)
                H[ki, kj] += tmp.reshape(nov, nov)
    else:
        for ki in range(nkpts):
            ka = kconserv_r[ki]
            for kj in range(nkpts):
                kb = kconserv_r[kj]
                # contribution from 2 (ai|jb) = 2 B^L_ai B^L_jb
                tmp = 2. * einsum("Lai,Ljb->iajb", eris.Lpq_mo[ka,ki][:, nocc:, :nocc], eris.Lpq_mo[kj,kb][:, :nocc, nocc:])
                # contribution from -(ab|ji) = - B^L_ab B^L_ji
                tmp -= einsum("Lab,Lji->iajb", eris.Lpq_mo[ka,kb][:, nocc:, nocc:], eris.Lpq_mo[kj,ki][:, :nocc, :nocc])
                tmp *= 1. / nkpts
                H[ki, kj] += tmp.reshape(nov, nov)
    
    H = H.reshape(nkpts, nkpts, nocc, nvir, nocc, nvir).transpose(0,2,3,1,4,5).reshape(r_size, r_size)
    log.timer("build full CIS Hamiltonian", *cpu0)
    return H

def cis_diag(cis, kshift, eris=None):
    """Diagonal elements of CIS Hamiltonian.
    
    Arguments:
        cis {KCIS} -- A KCIS instance
        kshift {int} -- k-shift index. A k-shift vector is an exciton momentum. 
            Available k-shift indices depend on the k-point mesh. For example, 
            a 2 by 2 by 2 k-point mesh allows at most 8 k-shift values, which can
            be targeted by 0, 1, 2, 3, 4, 5, 6, or 7.
    
    Keyword Arguments:
        eris {_CIS_ERIS} -- Depending on cis.direct, eris may 
            contain 4-center (cis.direct=False) or 3-center (cis.direct=True) 
            electron repulsion integrals (default: {None})            
     
    Returns:
        1D array -- an array formed by diagonal elements of CIS Hamiltonian
    """
    if eris is None: 
        eris = cis.ao2mo()
    nkpts = cis.nkpts
    nocc = cis.nocc
    nmo = cis.nmo
    nvir = nmo - nocc
    kconserv_r = cis.get_kconserv_r(kshift)
    dtype = eris.dtype
    epsilons = [eris.fock[k].diagonal().real for k in range(nkpts)]

    Hdiag = np.zeros((nkpts, nocc, nvir), dtype=dtype)
    for ki in range(nkpts):
        ka = kconserv_r[ki]
        Hdiag[ki] = direct_sum("a-i->ia", epsilons[ka][nocc:], epsilons[ki][:nocc])

    if not cis.direct:
        for ki in range(nkpts):
            ka = kconserv_r[ki]
            Hdiag[ki] += 2. * einsum("aiia->ia", eris.voov[ka, ki, ki])
            Hdiag[ki] -= einsum("iaia->ia", eris.ovov[ki, ka, ki])
    else:
        for ki in range(nkpts):
            ka = kconserv_r[ki]
            tmp = 2. * einsum("Lai,Lia->ia", eris.Lpq_mo[ka, ki][:, nocc:, :nocc], eris.Lpq_mo[ki,ka][:, :nocc, nocc:])
            tmp -= einsum("Laa,Lii->ia", eris.Lpq_mo[ka, ka][:, nocc:, nocc:], eris.Lpq_mo[ki, ki][:, :nocc, :nocc])
            Hdiag[ki] += (1. / nkpts) * tmp

    return np.ravel(Hdiag)


class KCIS(lib.StreamObject):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        assert isinstance(mf, scf.khf.KSCF)

        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        self._scf = mf
        self.kpts = mf.kpts
        self.verbose = mf.verbose
        self.max_memory = mf.max_memory

        self.max_space = getattr(__config__, 'kcis_rhf_max_space', 20)
        self.max_cycle = getattr(__config__, 'kcis_rhf_max_cycle', 50)
        self.conv_tol = getattr(__config__, 'kcis_rhf_conv_tol', 1e-7)

        ##################################################
        # don't modify the following attributes, unless you know what you are doing
        self.keep_exxdiv = False
        self.direct = False
        self.build_full_H = False
        self.davidson = True

        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.frozen = frozen
        self._nocc = None
        self._nmo = None
        self.voov = None
        self.ovov = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("")
        log.info("******** %s ********", self.__class__)
        log.info("nkpts = %d", self.nkpts)
        log.info("CIS nocc = %d, nmo = %d", self.nocc, self.nmo)
        if self.frozen is not 0:
            log.info("frozen orbitals = %s", self.frozen)
        log.info("max_memory %d MB (current use %d MB)",
                 self.max_memory, lib.current_memory()[0])
        if self.direct:
            log.info("cis.direct = True; voov and ovov will not be computed")
        return self

    @property
    def nkpts(self):
        return len(self.kpts)

    @property
    def nocc(self):
        return self.get_nocc()

    @property
    def nmo(self):
        return self.get_nmo()

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask
    get_diag = cis_diag
    matvec = cis_matvec_singlet
    kernel = kernel

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        return nkpts * nocc * nvir

    def vector_to_amplitudes(self, vector, nkpts=None, nmo=None, nocc=None):
        if nmo is None:
            nmo = self.nmo
        if nocc is None:
            nocc = self.nocc
        if nkpts is None:
            nkpts = self.nkpts

        nvir = nmo - nocc
        return vector[: nkpts * nocc * nvir].copy().reshape(nkpts, nocc, nvir)

    def amplitudes_to_vector(self, r):
        return r.ravel()

    def ao2mo(self, mo_coeff=None):
        return _CIS_ERIS(self, mo_coeff)

    def gen_matvec(self, kshift, eris=None, **kwargs):
        if eris is None: 
            eris = self.ao2mo()
        diag = self.get_diag(kshift, eris)
        matvec = lambda xs: [self.matvec(x, kshift, eris) for x in xs]
        return matvec, diag

    def get_init_guess(self, nroots=1, diag=None):
        idx = diag.argsort()
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', self.mo_coeff[0].dtype)
        nroots = min(nroots, size)
        guess = []
        for i in idx[:nroots]:
            g = np.zeros(size, dtype)
            g[i] = 1.0
            guess.append(g)
        return guess

    def get_kconserv_r(self, kshift):
        """Get the momentum conservation array for a set of k-points.

        Given k-point index m the array kconserv_r1[m] returns the index n that
        satisfies momentum conservation,

            (k(m) - k(n) - kshift) \dot a = 2n\pi

        This is used for symmetry of 1p-1h excitation operator vector
        R_{m k_m}^{n k_n} is zero unless n satisfies the above.

        Note that this method is adapted from `kpts_helper.get_kconserv()`.
        
        Arguments:
            kshift {int} -- index of momentum vector. It can be chosen as any of the available 
            k-point index based on the specified kpt mesh. 
            E.g. int from 0 to 7 can be chosen for a [2,2,2] grid.
        
        Returns:
            list -- a list of k(n) corresponding to k(m) that ranges from 0 to max_k_index
        """
        kconserv = self.khelper.kconserv
        kconserv_r = kconserv[:, kshift, 0].copy()
        return kconserv_r


# TODO Merge this with kccsd_rhf._ERIS, which contains more ints (e.g. oooo,
# ooov, etc.) than we need here
class _CIS_ERIS:
    def __init__(self, cis, mo_coeff=None, method="incore"):
        log = logger.Logger(cis.stdout, cis.verbose)
        cput0 = (time.clock(), time.time())

        moidx = get_frozen_mask(cis)
        cell = cis._scf.cell
        nocc = cis.nocc
        nmo = cis.nmo
        nvir = nmo - nocc
        nkpts = cis.nkpts
        kpts = cis.kpts

        if mo_coeff is None:
            mo_coeff = cis.mo_coeff
        dtype = mo_coeff[0].dtype

        mo_coeff = self.mo_coeff = padded_mo_coeff(cis, mo_coeff)

        # Re-make our fock MO matrix elements from density and fock AO
        dm = cis._scf.make_rdm1(cis.mo_coeff, cis.mo_occ)
        exxdiv = cis._scf.exxdiv if cis.keep_exxdiv else None
        with lib.temporary_env(cis._scf, exxdiv=exxdiv):
            # _scf.exxdiv affects eris.fock. HF exchange correction should be
            # excluded from the Fock matrix.
            fockao = cis._scf.get_hcore() + cis._scf.get_veff(cell, dm)
        self.fock = np.asarray([reduce(np.dot, (mo.T.conj(), fockao[k], mo))
                                for k, mo in enumerate(mo_coeff)])

        self.mo_energy = [self.fock[k].diagonal().real for k in range(nkpts)]

        if not cis.keep_exxdiv:
            # Add HFX correction in the self.mo_energy to improve convergence in
            # CCSD iteration. It is useful for the 2D systems since their occupied and
            # the virtual orbital energies may overlap which may lead to numerical
            # issue in the CCSD iterations.
            # FIXME: Whether to add this correction for other exxdiv treatments?
            # Without the correction, MP2 energy may be largely off the correct value.
            madelung = tools.madelung(cell, kpts)
            self.mo_energy = [
                _adjust_occ(mo_e, nocc, -madelung) for k, mo_e in enumerate(self.mo_energy)
            ]

        # Get location of padded elements in occupied and virtual space.
        nocc_per_kpt = get_nocc(cis, per_kpoint=True)
        nonzero_padding = padding_k_idx(cis, kind="joint")

        # Check direct and indirect gaps for possible issues with CCSD convergence.
        mo_e = [self.mo_energy[kp][nonzero_padding[kp]] for kp in range(nkpts)]
        mo_e = np.sort([y for x in mo_e for y in x])  # Sort de-nested array
        gap = mo_e[np.sum(nocc_per_kpt)] - mo_e[np.sum(nocc_per_kpt) - 1]
        if gap < 1e-5:
            logger.warn(
                cis,
                "HOMO-LUMO gap %s too small for KCCSD. "
                "May cause issues in convergence.",
                gap,
            )

        memory_needed = (nkpts ** 3 * nocc ** 2 * nvir ** 2) * 16 / 1e6
        # CIS only needs two terms: <aj|ib> and <aj|bi>; another factor of two for safety
        memory_needed *= 4

        memory_now = lib.current_memory()[0]
        fao2mo = cis._scf.with_df.ao2mo

        kconserv = cis.khelper.kconserv
        khelper = cis.khelper

        if cis.direct and type(cis._scf.with_df) is not df.GDF:
            raise ValueError("CIS direct method must be used with GDF")

        if (cis.direct and type(cis._scf.with_df) is df.GDF 
            and cell.dimension != 2):
            # cis._scf.with_df needs to be df.GDF only (not MDF)
            _init_cis_df_eris(cis, self)
        else:
            if (
                method == "incore"
                and (memory_needed + memory_now < cis.max_memory)
                or cell.incore_anyway
            ):
                log.info("using incore ERI storage")
                self.ovov = np.empty(
                    (nkpts, nkpts, nkpts, nocc, nvir, nocc, nvir), dtype=dtype
                )
                self.voov = np.empty(
                    (nkpts, nkpts, nkpts, nvir, nocc, nocc, nvir), dtype=dtype
                )

                for (ikp, ikq, ikr) in khelper.symm_map.keys():
                    iks = kconserv[ikp, ikq, ikr]
                    eri_kpt = fao2mo(
                        (mo_coeff[ikp], mo_coeff[ikq], mo_coeff[ikr], mo_coeff[iks]),
                        (kpts[ikp], kpts[ikq], kpts[ikr], kpts[iks]),
                        compact=False,
                    )
                    if dtype == np.float:
                        eri_kpt = eri_kpt.real
                    eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)
                    for (kp, kq, kr) in khelper.symm_map[(ikp, ikq, ikr)]:
                        eri_kpt_symm = khelper.transform_symm(
                            eri_kpt, kp, kq, kr
                        ).transpose(0, 2, 1, 3)
                        self.ovov[kp, kr, kq] = (
                            eri_kpt_symm[:nocc, nocc:, :nocc, nocc:] / nkpts
                        )
                        self.voov[kp, kr, kq] = (
                            eri_kpt_symm[nocc:, :nocc, :nocc, nocc:] / nkpts
                        )

                self.dtype = dtype
            else:
                log.info("using HDF5 ERI storage")
                self.feri1 = lib.H5TmpFile()

                self.ovov = self.feri1.create_dataset(
                    "ovov", (nkpts, nkpts, nkpts, nocc, nvir, nocc, nvir), dtype.char
                )
                self.voov = self.feri1.create_dataset(
                    "voov", (nkpts, nkpts, nkpts, nvir, nocc, nocc, nvir), dtype.char
                )

                # <ia|pq> = (ip|aq)
                cput1 = time.clock(), time.time()
                for kp in range(nkpts):
                    for kq in range(nkpts):
                        for kr in range(nkpts):
                            ks = kconserv[kp, kq, kr]
                            orbo_p = mo_coeff[kp][:, :nocc]
                            orbv_r = mo_coeff[kr][:, nocc:]
                            buf_kpt = fao2mo(
                                (orbo_p, mo_coeff[kq], orbv_r, mo_coeff[ks]),
                                (kpts[kp], kpts[kq], kpts[kr], kpts[ks]),
                                compact=False,
                            )
                            if mo_coeff[0].dtype == np.float:
                                buf_kpt = buf_kpt.real
                            buf_kpt = buf_kpt.reshape(nocc, nmo, nvir, nmo).transpose(
                                0, 2, 1, 3
                            )
                            self.dtype = buf_kpt.dtype
                            self.ovov[kp, kr, kq, :, :, :, :] = (
                                buf_kpt[:, :, :nocc, nocc:] / nkpts
                            )
                            self.voov[kr, kp, ks, :, :, :, :] = (
                                buf_kpt[:, :, nocc:, :nocc].transpose(1, 0, 3, 2) / nkpts
                            )
                cput1 = log.timer_debug1("transforming ovpq", *cput1)

        log.timer("CIS integral transformation", *cput0)


def _init_cis_df_eris(cis, eris):
    """Add 3-center electron repulsion integrals, i.e. (L|pq), in `eris`,
    where `L` denotes DF auxiliary basis functions and `p` and `q` canonical 
    crystalline orbitals. Note that `p` and `q` contain kpt indices `kp` and `kq`, 
    and the third kpt index `kL` is determined by the conservation of momentum.  
    
    Arguments:
        cis {KCIS} -- A KCIS instance
        eris {_CIS_ERIS} -- A _CIS_ERIS instance to which we want to add 3c ints
        
    Returns:
        _CIS_ERIS -- A _CIS_ERIS instance with 3c ints
    """
    from pyscf.pbc.df import df
    from pyscf.ao2mo import _ao2mo
    from pyscf.pbc.lib.kpts_helper import gamma_point

    log = logger.Logger(cis.stdout, cis.verbose)

    if cis._scf.with_df._cderi is None:
        cis._scf.with_df.build()

    cell = cis._scf.cell
    if cell.dimension == 2:
        # 2D ERIs are not positive definite. The 3-index tensors are stored in
        # two part. One corresponds to the positive part and one corresponds
        # to the negative part. The negative part is not considered in the
        # DF-driven CCSD implementation.
        raise NotImplementedError

    nocc = cis.nocc
    nmo = cis.nmo
    nao = cell.nao_nr()

    kpts = cis.kpts
    nkpts = len(kpts)
    if gamma_point(kpts):
        dtype = np.double
    else:
        dtype = np.complex128
    eris.dtype = dtype = np.result_type(dtype, *eris.mo_coeff)
    eris.Lpq_mo = Lpq_mo = np.empty((nkpts, nkpts), dtype=object)

    cput0 = (time.clock(), time.time())

    with h5py.File(cis._scf.with_df._cderi, 'r') as f:
        kptij_lst = f['j3c-kptij'].value 
        tao = []
        ao_loc = None
        for ki, kpti in enumerate(kpts):
            for kj, kptj in enumerate(kpts):
                kpti_kptj = np.array((kpti, kptj))
                Lpq_ao = np.asarray(df._getitem(f, 'j3c', kpti_kptj, kptij_lst))

                mo = np.hstack((eris.mo_coeff[ki], eris.mo_coeff[kj]))
                mo = np.asarray(mo, dtype=dtype, order='F')
                if dtype == np.double:
                    out = _ao2mo.nr_e2(Lpq_ao, mo, (0, nmo, nmo, nmo+nmo), aosym='s2')
                else:
                    #Note: Lpq.shape[0] != naux if linear dependency is found in auxbasis
                    if Lpq_ao[0].size != nao**2:  # aosym = 's2'
                        Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
                    out = _ao2mo.r_e2(Lpq_ao, mo, (0, nmo, nmo, nmo+nmo), tao, ao_loc)
                Lpq_mo[ki, kj] = out.reshape(-1, nmo, nmo)

    log.timer_debug1("transforming DF-CIS integrals", *cput0)

    return eris


if __name__ == "__main__":
    from pyscf.pbc import gto, scf, ci

    cell = gto.Cell()
    cell.atom = """
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.a = """
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000"""
    cell.unit = "B"
    cell.verbose = 7
    cell.build()

    # Running HF and MP2 with 1x1x2 Monkhorst-Pack k-point mesh
    kmf = scf.KRHF(cell, kpts=cell.make_kpts([1, 1, 2]), exxdiv=None)
    ehf = kmf.kernel()

    mycis = ci.KCIS(kmf)
    e_cis, v_cis = mycis.kernel(nroots=1, kptlist=[0])
    print(e_cis[0] - 0.2239201285373249)

