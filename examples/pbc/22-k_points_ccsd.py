#!/usr/bin/env python

'''
CCSD with k-point sampling or at an individual k-point
'''

import numpy
from pyscf.pbc import gto, scf, cc

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.build()

#
# KHF and KCCSD with 2x2x2 k-points
#
kpts = cell.make_kpts([2,2,2])
kmf = scf.KRHF(cell)
kmf.kpts = kpts
ehf = kmf.kernel()

mycc = cc.KCCSD(kmf)
mycc.kernel()
print("KRCCSD energy (per unit cell) =", mycc.e_tot)

#
# The KHF and KCCSD for single k-point calculation.
#
kpts = cell.get_abs_kpts([0.25, 0.25, 0.25])
kmf = scf.KRHF(cell)
kmf.kpts = kpts
ehf = kmf.kernel()

mycc = cc.KRCCSD(kmf)
mycc.kernel()
print("KRCCSD energy (per unit cell) =", mycc.e_tot)


#
# The PBC module provides an separated implementation specified for the single
# k-point calculations.  They are more efficient than the general implementation
# with k-point sampling.  For gamma point, integrals and orbitals are all real
# in this implementation.  They can be mixed with other post-HF methods that
# were provided in the molecular program.
#
kpt = cell.get_abs_kpts([0.25, 0.25, 0.25])
mf = scf.RHF(cell, kpt=kpt)
ehf = mf.kernel()

mycc = cc.RCCSD(mf).run()
print("RCCSD energy (per unit cell) at k-point =", mycc.e_tot)
dm1 = mycc.make_rdm1()
dm2 = mycc.make_rdm2()
nmo = mf.mo_coeff.shape[1]
eri_mo = mf.with_df.ao2mo(mf.mo_coeff, kpts=kpt).reshape([nmo]*4)
h1 = reduce(numpy.dot, (mf.mo_coeff.conj().T, mf.get_hcore(), mf.mo_coeff))
e_tot = numpy.einsum('ij,ji', h1, dm1) + numpy.einsum('ijkl,jilk', eri_mo, dm2)*.5 + mf.energy_nuc()
print("RCCSD energy based on CCSD density matrices =", e_tot.real)


mf = scf.addons.convert_to_uhf(mf)
mycc = cc.UCCSD(mf).run()
print("UCCSD energy (per unit cell) at k-point =", mycc.e_tot)
dm1a, dm1b = mycc.make_rdm1()
dm2aa, dm2ab, dm2bb = mycc.make_rdm2()
nmo = dm1a.shape[0]
eri_aa = mf.with_df.ao2mo(mf.mo_coeff[0], kpts=kpt).reshape([nmo]*4)
eri_bb = mf.with_df.ao2mo(mf.mo_coeff[1], kpts=kpt).reshape([nmo]*4)
eri_ab = mf.with_df.ao2mo((mf.mo_coeff[0],mf.mo_coeff[0],mf.mo_coeff[1],mf.mo_coeff[1]), kpts=kpt).reshape([nmo]*4)
hcore = mf.get_hcore()
h1a = reduce(numpy.dot, (mf.mo_coeff[0].conj().T, hcore, mf.mo_coeff[0]))
h1b = reduce(numpy.dot, (mf.mo_coeff[1].conj().T, hcore, mf.mo_coeff[1]))
e_tot = (numpy.einsum('ij,ji', h1a, dm1a) +
         numpy.einsum('ij,ji', h1b, dm1b) +
         numpy.einsum('ijkl,jilk', eri_aa, dm2aa)*.5 +
         numpy.einsum('ijkl,jilk', eri_ab, dm2ab)    +
         numpy.einsum('ijkl,jilk', eri_bb, dm2bb)*.5 + mf.energy_nuc())
print("UCCSD energy based on CCSD density matrices =", e_tot.real)


mf = scf.addons.convert_to_ghf(mf)
mycc = cc.GCCSD(mf).run()
print("GCCSD energy (per unit cell) at k-point =", mycc.e_tot)
dm1 = mycc.make_rdm1()
dm2 = mycc.make_rdm2()
nao = cell.nao_nr()
nmo = mf.mo_coeff.shape[1]
mo = mf.mo_coeff[:nao] + mf.mo_coeff[nao:]
eri_mo = mf.with_df.ao2mo(mo, kpts=kpt).reshape([nmo]*4)
orbspin = mf.mo_coeff.orbspin
eri_mo[orbspin[:,None]!=orbspin] = 0
eri_mo[:,:,orbspin[:,None]!=orbspin] = 0
h1 = reduce(numpy.dot, (mf.mo_coeff.conj().T, mf.get_hcore(), mf.mo_coeff))
e_tot = numpy.einsum('ij,ji', h1, dm1) + numpy.einsum('ijkl,jilk', eri_mo, dm2)*.5 + mf.energy_nuc()
print("GCCSD energy based on CCSD density matrices =", e_tot.real)
