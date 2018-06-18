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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy
from pyscf.pbc.df.fft_ao2mo import _format_kpts
from pyscf.pbc.df import df_ao2mo
from pyscf.pbc.df import aft_ao2mo
from pyscf import __config__


def get_eri(mydf, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_get_eri_compact', True)):
    if mydf._cderi is None:
        mydf.build()

    kptijkl = _format_kpts(kpts)
    eri = aft_ao2mo.get_eri(mydf, kptijkl, compact=compact)
    eri += df_ao2mo.get_eri(mydf, kptijkl, compact=compact)
    return eri


def general(mydf, mo_coeffs, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_general_compact', True)):
    if mydf._cderi is None:
        mydf.build()

    kptijkl = _format_kpts(kpts)
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        mo_coeffs = (mo_coeffs,) * 4
    eri_mo = aft_ao2mo.general(mydf, mo_coeffs, kptijkl, compact=compact)
    eri_mo += df_ao2mo.general(mydf, mo_coeffs, kptijkl, compact=compact)
    return eri_mo

