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

import numpy
from pyscf import lib
from pyscf.cc.bccd import bccd_kernel_

__all__ = ['spatial2spin', 'spatial2spinorb', 'spin2spatial', 'convert_to_gccsd']

def spatial2spin(tx, orbspin=None):
    '''Convert spatial-orbital T1 or T2 amplitudes to spin-orbital amplitudes.

    Args:
        tx : ndarray or tuple of ndarrays
            RCCSD: T1 of shape (nocc, nvir) or T2 of shape (nocc, nocc, nvir, nvir).
            UCCSD: T1 = (t1a, t1b), or T2 = (t2aa, t2ab, t2bb);
            doubles have shapes (noa, noa, nva, nva), (noa, nob, nva, nvb), and
            (nob, nob, nvb, nvb).

    Kwargs:
        orbspin : (nocc_total + nvir_total,) ndarray or None
            Spin labels (0 for alpha, 1 for beta), ordered as occupied then
            virtual spin orbitals. Their counts must match the input blocks.
            None assumes alternating alpha/beta labels and matching alpha
            and beta occupied/virtual dimensions.

    Returns:
        A T1 array of shape (nocc_total, nvir_total), or a T2 array of shape
        (nocc_total, nocc_total, nvir_total, nvir_total). The spin labels
        are stored in the array's ``orbspin`` attribute.

    Notes:
        Also applies to lambda amplitudes with the same index conventions.
    '''
    if isinstance(tx, numpy.ndarray) and tx.ndim == 2:
        # RCCSD t1 amplitudes
        return spatial2spin((tx,tx), orbspin)
    elif isinstance(tx, numpy.ndarray) and tx.ndim == 4:
        # RCCSD t2 amplitudes
        t2aa = tx - tx.transpose(1,0,2,3)
        return spatial2spin((t2aa,tx,t2aa), orbspin)
    elif len(tx) == 2:  # t1
        t1a, t1b = tx
        nocc_a, nvir_a = t1a.shape
        nocc_b, nvir_b = t1b.shape
    elif len(tx) == 3:  # t2
        t2aa, t2ab, t2bb = tx
        nocc_a, nocc_b, nvir_a, nvir_b = t2ab.shape
    else:
        raise RuntimeError('Unknown T amplitudes')

    if orbspin is None:
        assert nocc_a == nocc_b
        orbspin = numpy.zeros((nocc_a+nvir_a)*2, dtype=int)
        orbspin[1::2] = 1

    nocc = nocc_a + nocc_b
    nvir = nvir_a + nvir_b
    idxoa = numpy.where(orbspin[:nocc] == 0)[0]
    idxob = numpy.where(orbspin[:nocc] == 1)[0]
    idxva = numpy.where(orbspin[nocc:] == 0)[0]
    idxvb = numpy.where(orbspin[nocc:] == 1)[0]

    if len(tx) == 2:  # t1
        t1 = numpy.zeros((nocc,nvir), dtype=t1a.dtype)
        lib.takebak_2d(t1, t1a, idxoa, idxva)
        lib.takebak_2d(t1, t1b, idxob, idxvb)
        t1 = lib.tag_array(t1, orbspin=orbspin)
        return t1

    else:
        t2 = numpy.zeros((nocc**2,nvir**2), dtype=t2aa.dtype)
        idxoaa = idxoa[:,None] * nocc + idxoa
        idxoab = idxoa[:,None] * nocc + idxob
        idxoba = idxob[:,None] * nocc + idxoa
        idxobb = idxob[:,None] * nocc + idxob
        idxvaa = idxva[:,None] * nvir + idxva
        idxvab = idxva[:,None] * nvir + idxvb
        idxvba = idxvb[:,None] * nvir + idxva
        idxvbb = idxvb[:,None] * nvir + idxvb
        t2aa = t2aa.reshape(nocc_a*nocc_a,nvir_a*nvir_a)
        t2ab = t2ab.reshape(nocc_a*nocc_b,nvir_a*nvir_b)
        t2bb = t2bb.reshape(nocc_b*nocc_b,nvir_b*nvir_b)
        lib.takebak_2d(t2, t2aa, idxoaa.ravel()  , idxvaa.ravel()  )
        lib.takebak_2d(t2, t2bb, idxobb.ravel()  , idxvbb.ravel()  )
        lib.takebak_2d(t2, t2ab, idxoab.ravel()  , idxvab.ravel()  )
        lib.takebak_2d(t2, t2ab, idxoba.T.ravel(), idxvba.T.ravel())
        abba = -t2ab
        lib.takebak_2d(t2, abba, idxoab.ravel()  , idxvba.T.ravel())
        lib.takebak_2d(t2, abba, idxoba.T.ravel(), idxvab.ravel()  )
        t2 = lib.tag_array(t2, orbspin=orbspin)
        return t2.reshape(nocc,nocc,nvir,nvir)

spatial2spinorb = spatial2spin

def spin2spatial(tx, orbspin):
    '''Split spin-orbital T1 or T2 amplitudes into alpha and beta blocks.

    Args:
        tx : ndarray
            T1 of shape (nocc, nvir) or T2 of shape (nocc, nocc, nvir, nvir),
            with spin-orbital indices.
        orbspin : (nocc + nvir,) ndarray
            Spin labels, 0 for alpha and 1 for beta, ordered as occupied
            orbitals followed by virtual orbitals.

    Returns:
        For singles, (t1a, t1b) with shapes (noa, nva) and (nob, nvb).
        For doubles, (t2aa, t2ab, t2bb) with shapes (noa, noa, nva, nva),
        (noa, nob, nva, nvb), and (nob, nob, nvb, nvb). The dimensions are
        determined from ``orbspin``.

    Notes:
        Only blocks that conserve alpha and beta spin are extracted. Spin-flip
        blocks are ignored, even if they contain nonzero amplitudes.

        Also applies to lambda amplitudes with the same index conventions.
    '''
    if tx.ndim == 2:  # t1
        nocc, nvir = tx.shape
    elif tx.ndim == 4:
        nocc, nvir = tx.shape[1:3]
    else:
        raise RuntimeError('Unknown T amplitudes')

    idxoa = numpy.where(orbspin[:nocc] == 0)[0]
    idxob = numpy.where(orbspin[:nocc] == 1)[0]
    idxva = numpy.where(orbspin[nocc:] == 0)[0]
    idxvb = numpy.where(orbspin[nocc:] == 1)[0]
    nocc_a = len(idxoa)
    nocc_b = len(idxob)
    nvir_a = len(idxva)
    nvir_b = len(idxvb)

    if tx.ndim == 2:  # t1
        t1a = lib.take_2d(tx, idxoa, idxva)
        t1b = lib.take_2d(tx, idxob, idxvb)
        return t1a, t1b
    else:
        idxoaa = idxoa[:,None] * nocc + idxoa
        idxoab = idxoa[:,None] * nocc + idxob
        idxobb = idxob[:,None] * nocc + idxob
        idxvaa = idxva[:,None] * nvir + idxva
        idxvab = idxva[:,None] * nvir + idxvb
        idxvbb = idxvb[:,None] * nvir + idxvb
        t2 = tx.reshape(nocc**2,nvir**2)
        t2aa = lib.take_2d(t2, idxoaa.ravel(), idxvaa.ravel())
        t2bb = lib.take_2d(t2, idxobb.ravel(), idxvbb.ravel())
        t2ab = lib.take_2d(t2, idxoab.ravel(), idxvab.ravel())
        t2aa = t2aa.reshape(nocc_a,nocc_a,nvir_a,nvir_a)
        t2bb = t2bb.reshape(nocc_b,nocc_b,nvir_b,nvir_b)
        t2ab = t2ab.reshape(nocc_a,nocc_b,nvir_a,nvir_b)
        return t2aa,t2ab,t2bb

def convert_to_uccsd(mycc):
    from pyscf.cc import uccsd, gccsd
    if isinstance(mycc, uccsd.UCCSD):
        return mycc
    elif isinstance(mycc, gccsd.GCCSD):
        raise NotImplementedError

    mf = mycc._scf.to_uhf()
    ucc = uccsd.UCCSD(mf)
    assert (mycc._nocc is None)
    assert (mycc._nmo is None)
    ucc.__dict__.update(mycc.__dict__)
    ucc._scf = mf
    ucc.mo_coeff = mf.mo_coeff
    ucc.mo_occ = mf.mo_occ
    if not (mycc.frozen is None or isinstance(mycc.frozen, (int, numpy.integer))):
        raise NotImplementedError
    ucc.t1, ucc.t2 = uccsd.amplitudes_from_rccsd(mycc.t1, mycc.t2)
    return ucc

def convert_to_gccsd(mycc):
    '''Convert an RCCSD or UCCSD object and its amplitudes to GCCSD.

    Args:
        mycc : CCSD object
            Restricted or unrestricted CCSD object with ``t1`` and ``t2``
            available. Explicit overrides of ``_nocc`` and ``_nmo`` must be
            None.

    Returns:
        A GCCSD object with a generalized HF reference and spin-orbital
        ``t1`` and ``t2`` using the reference's ``orbspin`` ordering.
        If mycc is already a GCCSD object, returns mycc itself.

    Notes:
        No SCF or CCSD optimization is performed. Integer frozen-orbital
        counts are doubled.

        Other attributes are copied by reference and may be shared with the
        original object. Only the reference orbitals, occupations and T
        amplitudes are converted. Recompute any stored lambda amplitudes and
        intermediates before using them with the new spin-orbital basis.
    '''
    from pyscf.cc import gccsd
    if isinstance(mycc, gccsd.GCCSD):
        return mycc

    mf = mycc._scf.to_ghf()
    gcc = gccsd.GCCSD(mf)
    assert (mycc._nocc is None)
    assert (mycc._nmo is None)
    gcc.__dict__.update(mycc.__dict__)
    gcc._scf = mf
    gcc.mo_coeff = mf.mo_coeff
    gcc.mo_occ = mf.mo_occ
    if isinstance(mycc.frozen, (int, numpy.integer)):
        gcc.frozen = mycc.frozen * 2
    elif (mycc.frozen is not None and
          # to distinguish the setting cc.frozen = np.array([0]) and cc.frozen = 0
          mycc.nmo != gcc.mo_occ.size):
        raise NotImplementedError
    gcc.t1 = spatial2spin(mycc.t1, mf.mo_coeff.orbspin)
    gcc.t2 = spatial2spin(mycc.t2, mf.mo_coeff.orbspin)
    return gcc
