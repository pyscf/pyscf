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

#
# RHF/RKS-TDDFT
#
def diagonalize(a, b, nroots=5):
    nocc, nvir = a.shape[:2]
    a = a.reshape(nocc*nvir,nocc*nvir)
    b = b.reshape(nocc*nvir,nocc*nvir)
    e = numpy.linalg.eig(numpy.bmat([[a        , b       ],
                                     [-b.conj(),-a.conj()]]))[0]
    lowest_e = numpy.sort(e[e > 0])[:nroots]
    return lowest_e
mf = scf.RHF(mol).run()
a, b = tddft.TDHF(mf).get_ab()
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDHF(mf).kernel(nstates=5)[0])

mf = dft.RKS(mol).run(xc='lda,vwn')
a, b = tddft.TDDFT(mf).get_ab()
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDDFT(mf).kernel(nstates=5)[0])

mf = dft.RKS(mol).run(xc='b3lyp')
a, b = tddft.TDDFT(mf).get_ab()
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDDFT(mf).kernel(nstates=5)[0])

#
# with frozen orbitals
#
frozen = 1
mf = scf.RHF(mol).run()
a, b = tddft.TDHF(mf, frozen).get_ab()
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDHF(mf, frozen).kernel(nstates=5)[0])

mf = dft.RKS(mol).run(xc='lda,vwn')
a, b = tddft.TDDFT(mf, frozen).get_ab()
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDDFT(mf, frozen).kernel(nstates=5)[0])

mf = dft.RKS(mol).run(xc='b3lyp')
a, b = tddft.TDDFT(mf, frozen).get_ab()
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDDFT(mf, frozen).kernel(nstates=5)[0])

#
# UHF/UKS-TDDFT
#
def diagonalize(a, b, nroots=4):
    a_aa, a_ab, a_bb = a
    b_aa, b_ab, b_bb = b
    nocc_a, nvir_a, nocc_b, nvir_b = a_ab.shape
    a_aa = a_aa.reshape((nocc_a*nvir_a,nocc_a*nvir_a))
    a_ab = a_ab.reshape((nocc_a*nvir_a,nocc_b*nvir_b))
    a_bb = a_bb.reshape((nocc_b*nvir_b,nocc_b*nvir_b))
    b_aa = b_aa.reshape((nocc_a*nvir_a,nocc_a*nvir_a))
    b_ab = b_ab.reshape((nocc_a*nvir_a,nocc_b*nvir_b))
    b_bb = b_bb.reshape((nocc_b*nvir_b,nocc_b*nvir_b))
    a = numpy.bmat([[ a_aa  , a_ab],
                    [ a_ab.T, a_bb]])
    b = numpy.bmat([[ b_aa  , b_ab],
                    [ b_ab.T, b_bb]])
    e = numpy.linalg.eig(numpy.bmat([[a        , b       ],
                                     [-b.conj(),-a.conj()]]))[0]
    lowest_e = numpy.sort(e[e.real > 0].real)[:nroots]
    return lowest_e

mol.spin = 2
mf = scf.UHF(mol).run()
a, b = tddft.TDHF(mf).get_ab()
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDHF(mf).kernel(nstates=4)[0])

mf = dft.UKS(mol).run(xc='lda,vwn')
a, b = tddft.TDDFT(mf).get_ab()
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDDFT(mf).kernel(nstates=4)[0])

mf = dft.UKS(mol).run(xc='b3lyp')
a, b = tddft.TDDFT(mf).get_ab()
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDDFT(mf).kernel(nstates=4)[0])

#
# with frozen orbitals
#
frozen = 1
mf = scf.UHF(mol).run()
a, b = tddft.TDHF(mf, frozen).get_ab()
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDHF(mf, frozen).kernel(nstates=4)[0])

mf = dft.UKS(mol).run(xc='lda,vwn')
a, b = tddft.TDDFT(mf, frozen).get_ab()
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDDFT(mf, frozen).kernel(nstates=4)[0])

mf = dft.UKS(mol).run(xc='b3lyp')
a, b = tddft.TDDFT(mf, frozen).get_ab()
print('Direct diagoanlization:', diagonalize(a, b))
print('Reference:', tddft.TDDFT(mf, frozen).kernel(nstates=4)[0])
