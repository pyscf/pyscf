#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf.pbc.df.fft_ao2mo import _format_kpts
from pyscf.pbc.df import df_ao2mo
from pyscf.pbc.df import aft_ao2mo


def get_eri(mydf, kpts=None, compact=True):
    if mydf._cderi is None:
        mydf.build()

    kptijkl = _format_kpts(kpts)
    eri = aft_ao2mo.get_eri(mydf, kptijkl, compact=compact)
    eri += df_ao2mo.get_eri(mydf, kptijkl, compact=compact)
    return eri


def general(mydf, mo_coeffs, kpts=None, compact=True):
    if mydf._cderi is None:
        mydf.build()

    kptijkl = _format_kpts(kpts)
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        mo_coeffs = (mo_coeffs,) * 4
    eri_mo = aft_ao2mo.general(mydf, mo_coeffs, kptijkl, compact=compact)
    eri_mo += df_ao2mo.general(mydf, mo_coeffs, kptijkl, compact=compact)
    return eri_mo

