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

def spatial2spin(tx, orbspin=None):
    '''Convert T1/T2 of spatial orbital representation to T1/T2 of
    spin-orbital representation
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
    '''Convert T1/T2 in spin-orbital basis to T1/T2 in spatial orbital basis
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
    elif not (mycc.frozen is None or mycc.frozen == 0):
        raise NotImplementedError
    gcc.t1 = spatial2spin(mycc.t1, mf.mo_coeff.orbspin)
    gcc.t2 = spatial2spin(mycc.t2, mf.mo_coeff.orbspin)
    return gcc
