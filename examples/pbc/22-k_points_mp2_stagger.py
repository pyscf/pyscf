#!/usr/bin/env python

'''
Example code for
k-point spin-restricted periodic MP2 calculation using staggered mesh method

Author: Xin Xing (xxing@berkeley.edu)

Reference: Staggered Mesh Method for Correlation Energy Calculations of Solids:
           Second-Order Møller-Plesset Perturbation Theory, J. Chem. Theory
           Comput. 2021, 17, 8, 4733-4745
'''


from pyscf.pbc.mp.kmp2_stagger import KMP2_stagger
from pyscf.pbc import df, gto, scf, mp


'''
Hydrogen dimer
'''
cell = gto.Cell()
cell.pseudo = 'gth-pade'
cell.basis = 'gth-szv'
cell.ke_cutoff = 100
cell.atom = '''
    H 3.00   3.00   2.10
    H 3.00   3.00   3.90
    '''
cell.a = '''
    6.0   0.0   0.0
    0.0   6.0   0.0
    0.0   0.0   6.0
    '''
cell.unit = 'B'
cell.verbose = 4
cell.build()


#   HF calculation using FFTDF
nks_mf = [2, 2, 2]
kpts = cell.make_kpts(nks_mf, with_gamma_point=True)
kmf = scf.KRHF(cell, kpts, exxdiv='ewald')
ehf = kmf.kernel()

#   staggered mesh KMP2 using two 1*1*1 submeshes in kmf.kpts
kmp = KMP2_stagger(kmf, flag_submesh=True)
emp2 = kmp.kernel()
assert((abs(emp2 - -0.0160902544091997)) < 1e-5)

#   staggered mesh KMP2 using two 2*2*2 submeshes based on non-SCF
kmp = KMP2_stagger(kmf, flag_submesh=False)
emp2 = kmp.kernel()
assert((abs(emp2 - -0.0140289970302513)) < 1e-5)

#   standard KMP2 calculation
kmp = mp.KMP2(kmf)
emp2, _ = kmp.kernel()
assert((abs(emp2 - -0.0143904878990777)) < 1e-5)


#   HF calculation using GDF
nks_mf = [2, 2, 2]
kpts = cell.make_kpts(nks_mf, with_gamma_point=True)
kmf = scf.KRHF(cell, kpts, exxdiv='ewald')
kmf.with_df = df.GDF(cell, kpts).build()
ehf = kmf.kernel()

#   staggered mesh KMP2 using two 1*1*1 submeshes in kmf.kpts
kmp = KMP2_stagger(kmf, flag_submesh=True)
emp2 = kmp.kernel()
assert((abs(emp2 - -0.0158364523431071)) < 1e-5)

#   staggered mesh KMP2 using two 2*2*2 submeshes based on non-SCF
kmp = KMP2_stagger(kmf, flag_submesh=False)
emp2 = kmp.kernel()
assert((abs(emp2 - -0.0140280303691396)) < 1e-5)

#   standard KMP2 calculation
kmp = mp.KMP2(kmf)
emp2, _ = kmp.kernel()
assert((abs(emp2 - -0.0141829343769316)) < 1e-5)


'''
Diamond system
'''

cell = gto.Cell()
cell.pseudo = 'gth-pade'
cell.basis = 'gth-szv'
cell.ke_cutoff = 100
cell.atom = '''
    C     0.      0.      0.
    C     1.26349729, 0.7294805 , 0.51582061
    '''
cell.a = '''
    2.52699457, 0.        , 0.
    1.26349729, 2.18844149, 0.
    1.26349729, 0.7294805 , 2.06328243
    '''
cell.unit = 'angstrom'
cell.verbose = 4
cell.build()


#   HF calculation using FFTDF
nks_mf = [2, 2, 2]
kpts = cell.make_kpts(nks_mf, with_gamma_point=True)
kmf = scf.KRHF(cell, kpts, exxdiv='ewald')
ehf = kmf.kernel()

#   staggered mesh KMP2 using two 1*1*1 submeshes in kmf.kpts
kmp = KMP2_stagger(kmf, flag_submesh=True)
emp2 = kmp.kernel()
assert((abs(emp2 - -0.156289981810986)) < 1e-5)

#   staggered mesh KMP2 using two 2*2*2 submeshes based on non-SCF
kmp = KMP2_stagger(kmf, flag_submesh=False)
emp2 = kmp.kernel()
assert((abs(emp2 - -0.105454107635884)) < 1e-5)

#   standard KMP2 calculation
kmp = mp.KMP2(kmf)
emp2, _ = kmp.kernel()
assert((abs(emp2 - -0.095517731535516)) < 1e-5)


#   HF calculation using GDF
nks_mf = [2, 2, 2]
kpts = cell.make_kpts(nks_mf, with_gamma_point=True)
kmf = scf.KRHF(cell, kpts, exxdiv='ewald')
kmf.with_df = df.GDF(cell, kpts).build()
ehf = kmf.kernel()

#   staggered mesh KMP2 using two 1*1*1 submeshes in kmf.kpts
kmp = KMP2_stagger(kmf, flag_submesh=True)
emp2 = kmp.kernel()
assert((abs(emp2 - -0.154923152683604)) < 1e-5)

#   staggered mesh KMP2 using two 2*2*2 submeshes based on non-SCF
kmp = KMP2_stagger(kmf, flag_submesh=False)
emp2 = kmp.kernel()
assert((abs(emp2 - -0.105421948003715)) < 1e-5)

#   standard KMP2 calculation
kmp = mp.KMP2(kmf)
emp2, _ = kmp.kernel()
assert((abs(emp2 - -0.0952009565805345)) < 1e-5)
