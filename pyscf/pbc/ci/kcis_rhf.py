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

def kernel(cis, nroots=1, eris=None, kptlist=None, **kargs):
    """[summary]
    
    Arguments:
        cis {[type]} -- [description]
    
    Keyword Arguments:
        nroots {int} -- [description] (default: {1})
        eris {[type]} -- [description] (default: {None})
        kptlist {[type]} -- [description] (default: {None})
    
    Returns:
        [type] -- [description]
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
    for k, kshift in enumerate(kptlist):
        print("\nkshift =", kshift)

        H = np.zeros([r_size, r_size], dtype=dtype)
        for col in range(r_size):
            vec = np.zeros(r_size, dtype=dtype)
            vec[col] = 1.0
            H[:, col] = cis_matvec_singlet(cis, vec, kshift, eris=eris)

        eigval, eigvec = np.linalg.eig(H)
        idx = eigval.argsort()[:nroots]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]

        for n in range(nroots):
            logger.info(cis, "CIS root %d E = %.16g", n, eigval[n])

        evals[k] = eigval
        evecs[k] = eigvec

    log.timer("CIS", *cpu0)
    return evals, evecs

def cis_matvec_singlet(cis, vector, kshift, eris=None):
    """[summary]
    
    Arguments:
        cis {[type]} -- [description]
        vector {[type]} -- [description]
        kshift {[type]} -- [description]
    
    Keyword Arguments:
        eris {[type]} -- [description] (default: {None})
    
    Returns:
        [type] -- [description]
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
        # x: kj
        Hr[ki] += 2.0 * einsum("xjb,xajib->ia", r, eris.voov[ka, :, ki])
        Hr[ki] -= einsum("xjb,xjaib->ia", r, eris.ovov[:, ka, ki])

    vector = cis.amplitudes_to_vector(Hr)
    return vector

class KCIS(lib.StreamObject):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, keep_exxdiv=False):
        """[summary]

        Arguments:
            lib {[type]} -- [description]
            mf {[type]} -- [description]

        Keyword Arguments:
            frozen {int} -- [description] (default: {0})
            mo_coeff {[type]} -- [description] (default: {None})
            mo_occ {[type]} -- [description] (default: {None})
        """
        assert isinstance(mf, scf.khf.KSCF)

        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        self._scf = mf
        self.kpts = mf.kpts
        self.verbose = mf.verbose
        self.max_memory = mf.max_memory
        self.keep_exxdiv = keep_exxdiv

        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.frozen = frozen
        self._nocc = None
        self._nmo = None
        self.voov = None
        self.ovov = None

    def dump_flags(self):
        logger.info(self, "")
        logger.info(self, "******** %s ********", self.__class__)
        logger.info(self, "nkpts = %d", self.nkpts)
        logger.info(self, "CIS nocc = %d, nmo = %d", self.nocc, self.nmo)
        if self.frozen is not 0:
            logger.info(self, "frozen orbitals = %s", self.frozen)
        logger.info(
            self,
            "max_memory %d MB (current use %d MB)",
            self.max_memory,
            lib.current_memory()[0],
        )
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

        print(cis._scf.mo_energy)
        if cis.keep_exxdiv:
            self.fock = np.asarray([np.diag(mo_e) for k, mo_e in enumerate(cis._scf.mo_energy)], dtype=dtype)
            self.mo_energy = [self.fock[k].diagonal().real for k in range(nkpts)]
        else:
            # Re-make our fock MO matrix elements from density and fock AO
            dm = cis._scf.make_rdm1(cis.mo_coeff, cis.mo_occ)
            with lib.temporary_env(cis._scf, exxdiv=None):
                # _scf.exxdiv affects eris.fock. HF exchange correction should be
                # excluded from the Fock matrix.
                fockao = cis._scf.get_hcore() + cis._scf.get_veff(cell, dm)
            self.fock = np.asarray(
                [
                    reduce(np.dot, (mo.T.conj(), fockao[k], mo))
                    for k, mo in enumerate(mo_coeff)
                ]
            )

            self.mo_energy = [self.fock[k].diagonal().real for k in range(nkpts)]
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

