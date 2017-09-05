#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# Ref:
# Chem Phys Lett, 256, 454
# J. Mol. Struct. THEOCHEM, 914, 3
#

import time
import copy
from functools import reduce
import numpy
from pyscf import lib
from pyscf.dft import numint
from pyscf.ao2mo import _ao2mo
from pyscf.pbc.tddft import rhf


TDA = rhf.TDA

RPA = TDDFT = rhf.TDHF


if __name__ == '__main__':
    from pyscf.pbc import gto
    from pyscf.pbc import scf
    from pyscf.pbc import dft, df
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
    cell.gs = [12]*3
    cell.build()

    mf = dft.KRKS(cell, cell.make_kpts([2,1,1]))
    #mf.with_df = df.MDF(cell, cell.make_kpts([2,1,1]))
    #mf.with_df.auxbasis = 'weigend'
    #mf.with_df._cderi = 'eri3d-mdf.h5'
    #mf.with_df.build(with_j3c=False)
    mf.xc = 'lda'
    mf.kernel()
#gs=12 -10.3077341607895
#gs=5  -10.3086623157515

    td = TDDFT(mf)
    td.nstates = 5
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
#gs=12 [ 6.08108297  6.10231481  6.10231478  6.38355803  6.38355804]
#MDF gs=5 [ 6.07919157  6.10251718  6.10253961  6.37202499  6.37565246]

    td = TDA(mf)
    td.singlet = False
    td.verbose = 5
    print(td.kernel()[0] * 27.2114)
#gs=12 [ 4.01539192  5.1750807   5.17508071]
#MDF gs=5 [ 4.01148649  5.18043397  5.18043459]
