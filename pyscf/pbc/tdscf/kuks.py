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

from pyscf import lib
from pyscf.pbc import dft
from pyscf.pbc.tdscf import kuhf


KTDA = TDA = kuhf.TDA
RPA = KTDDFT = TDDFT = kuhf.TDHF

dft.kuks.KUKS.TDA   = lib.class_as_method(KTDA)
dft.kuks.KUKS.TDHF  = None
dft.kuks.KUKS.TDDFT = lib.class_as_method(TDDFT)


if __name__ == '__main__':
    from pyscf.pbc import gto
    from pyscf.pbc import dft
    from pyscf.pbc import df
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
    cell.mesh = [37]*3
    cell.build()
    mf = dft.KUKS(cell, cell.make_kpts([2,1,1])).set(exxdiv=None, xc='b88,p86')
    #mf.with_df = df.MDF(cell, cell.make_kpts([2,1,1]))
    #mf.with_df.auxbasis = 'weigend'
    #mf.with_df._cderi = 'eri3d-mdf.h5'
    #mf.with_df.build(with_j3c=False)
    mf.run()

    td = TDDFT(mf)
    td.verbose = 5
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)

    mf.xc = 'lda,vwn'
    mf.run()
    td = TDA(mf)
    td.verbose = 5
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)

    cell.spin = 2
    mf = dft.KUKS(cell, cell.make_kpts([2,1,1])).set(exxdiv=None, xc='b88,p86')
    mf.run()

    td = TDDFT(mf)
    td.verbose = 5
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)

    mf.xc = 'lda,vwn'
    mf.run()
    td = TDA(mf)
    td.verbose = 5
    td.nstates = 5
    print(td.kernel()[0] * 27.2114)
