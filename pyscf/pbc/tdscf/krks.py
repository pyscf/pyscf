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
        _rebuild_df(self)
        return krhf.TDA.kernel(self, x0=x0)

KTDA = TDA

class TDDFT(krhf.TDHF):
    def kernel(self, x0=None):
        _rebuild_df(self)
        return krhf.TDHF.kernel(self, x0=x0)

RPA = KTDDFT = TDDFT

def _rebuild_df(td):
    log = lib.logger.new_logger(td)
    mf = td._scf
    if any([k != 0 for k in td.kshift_lst]):
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


if __name__ == '__main__':
    from pyscf.pbc import gto
    from pyscf.pbc import dft
    cell = gto.Cell()
    cell.unit = 'B'
    cell.atom = '''
    C  0.          0.          0.
    C  1.68506879  1.68506879  1.68506879
    '''
    cell.a = '''
    0.          3.37013758  3.37013758
    3.37013758  0.          3.37013758
    3.37013758  3.37013758  0.
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [25]*3
    cell.build()

    mf = dft.KRKS(cell, cell.make_kpts([2,1,1]))
    #mf.with_df = df.MDF(cell, cell.make_kpts([2,1,1]))
    #mf.with_df.auxbasis = 'weigend'
    #mf.with_df._cderi = 'eri3d-mdf.h5'
    #mf.with_df.build(with_j3c=False)
    mf.xc = 'lda,'
    mf.kernel()
#mesh=12 -10.3077341607895
#mesh=5  -10.3086623157515

    td = TDDFT(mf)
    td.nstates = 5
    td.verbose = 5
    print(td.kernel()[0][0] * 27.2114)
#mesh=12 [ 6.08108297  6.10231481  6.10231478  6.38355803  6.38355804]
#MDF mesh=5 [ 6.07919157  6.10251718  6.10253961  6.37202499  6.37565246]

    td = TDA(mf)
    td.singlet = False
    td.verbose = 5
    print(td.kernel()[0][0] * 27.2114)
#mesh=12 [ 4.01539192  5.1750807   5.17508071]
#MDF mesh=5 [ 4.01148649  5.18043397  5.18043459]
