#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A, B matrices of TDDFT method.
'''

import numpy
from pyscf import gto, scf, dft, tddft

mol = gto.Mole()
mol.atom = [
    ['H' , (0. , 0. , .917)],
    ['F' , (0. , 0. , 0.)], ]
mol.basis = '6311g*'
mol.build()

mf = scf.RHF(mol).run()
a, b = tddft.TDHF(mf).get_ab()

def diagonalize(a, b, nroots=5):
    nocc, nvir = a.shape[:2]
    a = a.reshape(nocc*nvir,nocc*nvir)
    b = b.reshape(nocc*nvir,nocc*nvir)
    e = numpy.linalg.eig(numpy.bmat([[a        , b       ],
                                     [-b.conj(),-a.conj()]]))[0]
    lowest_e = numpy.sort(e[e > 0])[:nroots]
    return lowest_e
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDHF(mf).kernel(nstates=5)[0])

mf = dft.RKS(mol).run(xc='lda,vwn')
a, b = tddft.TDDFT(mf).get_ab()
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDHF(mf).kernel(nstates=5)[0])

mf = dft.RKS(mol).run(xc='b3lyp')
a, b = tddft.TDDFT(mf).get_ab()
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDHF(mf).kernel(nstates=5)[0])
