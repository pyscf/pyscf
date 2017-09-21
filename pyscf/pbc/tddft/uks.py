#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from pyscf.pbc.tddft import uhf

TDA = uhf.TDA

RPA = TDDFT = uhf.TDHF


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
    cell.gs = [18]*3
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
