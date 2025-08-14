#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
# Ref:
# Chem Phys Lett, 256, 454
# J. Mol. Struct. THEOCHEM, 914, 3
#

from pyscf import lib
from pyscf.pbc import dft
from pyscf.pbc import df
from pyscf.pbc.tdscf import krhf


class TDA(krhf.TDA):
    def kernel(self, x0=None):
        if hasattr(self._scf, 'U_idx'):
            raise NotImplementedError('TDDFT for DFT+U')
        _rebuild_df(self)
        return krhf.TDA.kernel(self, x0=x0)

KTDA = TDA

class TDDFT(krhf.TDHF):
    def kernel(self, x0=None):
        if hasattr(self._scf, 'U_idx'):
            raise NotImplementedError('TDDFT for DFT+U')
        _rebuild_df(self)
        return krhf.TDHF.kernel(self, x0=x0)

RPA = KTDDFT = TDDFT

def _rebuild_df(td):
    log = lib.logger.new_logger(td)
    mf = td._scf
    if any(k != 0 for k in td.kshift_lst):
        if isinstance(mf.with_df, df.df.DF):
            if mf.with_df._j_only:
                log.warn(f'Non-zero kshift is requested for {td.__class__.__name__}, '
                         f'recomputing DF integrals with _j_only = False')
                mf.with_df._j_only = False
                mf.with_df.build()

dft.krks.KRKS.TDA   = lib.class_as_method(KTDA)
dft.krks.KRKS.TDHF  = None
dft.krks.KRKS.TDDFT = lib.class_as_method(TDDFT)
dft.kroks.KROKS.TDA   = None
dft.kroks.KROKS.TDHF  = None
dft.kroks.KROKS.TDDFT = None
