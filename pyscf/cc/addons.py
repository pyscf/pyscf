#!/usr/bin/env python

import numpy
from pyscf import lib

def spatial2spin(tx, orbspin=None):
    '''Convert T1/T2 of spatial orbital representation to T1/T2 of
    spin-orbital representation

    call orbspin_of_sorted_mo_energy to get orbspin
    '''
    if isinstance(tx, numpy.ndarray) and tx.ndim == 2:
        # RCCSD t1 amplitudes
        return spatial2spin((tx,tx), orbspin)
    elif isinstance(tx, numpy.ndarray) and tx.ndim == 4:
        # RCCSD t2 amplitudes
        t2aa = tx - tx.transpose(0,1,3,2)
        return spatial2spin((t2aa,tx,t2aa), orbspin)
    elif len(tx) == 2:  # t1
        t1a, t1b = tx
        nocc_a, nvir_a = t1a.shape
        nocc_b, nvir_b = t1b.shape
    else:
        t2aa, t2ab, t2bb = tx
        nocc_a, nocc_b, nvir_a, nvir_b = t2ab.shape

    if orbspin is None:
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
        return t2.reshape(nocc,nocc,nvir,nvir)

spatial2spinorb = spatial2spin

def spin2spatial(tx, orbspin):
    '''call orbspin_of_sorted_mo_energy to get orbspin'''
    if tx.ndim == 2:  # t1
        nocc, nvir = tx.shape
    else:
        nocc, nvir = tx.shape[1:3]

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
