#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>

from pyscf.lib import _ao2mo
import incore
from incore import gen_int2e_from_full_eri
from incore import get_int2e_from_partial_eri

import direct
from direct import gen_int2e_ao2mo


def restore(symmetry, eri, norb, blockdim=208):
    if symmetry not in (8, 4, 1):
        raise ValueError('symmetry = %s' % symmetry)

    npair = norb*(norb+1)/2
    if eri.size == norb**4:
        if symmetry == 1:
            return eri.reshape(norb,norb,norb,norb)
        elif symmetry == 4:
            return _ao2mo.restore_1to4(eri, norb, blockdim)
        else: # 8-fold
            return _ao2mo.restore_1to8(eri, norb, blockdim)
    elif eri.size == npair**2:
        if symmetry == 1:
            return _ao2mo.restore_4to1(eri, norb, blockdim)
        elif symmetry == 4:
            return eri.reshape(npair,npair)
        else: # 8-fold
            return _ao2mo.restore_4to8(eri, norb, blockdim)
    elif eri.size == npair*(npair+1)/2: # 8-fold
        if symmetry == 1:
            return _ao2mo.restore_8to1(eri, norb, blockdim)
        elif symmetry == 4:
            return _ao2mo.restore_8to4(eri, norb, blockdim)
        else: # 8-fold
            return eri
    else:
        raise ValueError('eri.size = %d, norb = %d' % (eri.size, norb))
