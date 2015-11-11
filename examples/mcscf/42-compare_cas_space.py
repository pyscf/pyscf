import numpy
from pyscf import gto, scf, mcscf

'''
Compare two CASSCF active space.

It's important to compare multi-reference calculations based on the comparable
reference states.  Here we compute the SVD eig and the determinant value of
the CAS space overlap to measure how close two CASSCF results are.  If two
CASSCF are close, the SVD eig should be close 1
'''

mol1 = gto.M(atom='O 0 0 0; O 0 0 1', basis='ccpvtz', spin=2, symmetry=1)
mf = scf.RHF(mol1)
mf.kernel()
mc = mcscf.CASSCF(mf, 7, 4)
mc.kernel()
mo1cas = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]

mol2 = gto.M(atom='O 0 0 0; O 0 0 1', basis='ccpvdz', spin=2, symmetry=1)
mf = scf.RHF(mol2)
mf.kernel()
mc = mcscf.CASSCF(mf, 7, (2,2))
mc.kernel()
mo2cas = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]

s = gto.intor_cross('cint1e_ovlp_sph', mol1, mol2)
s = reduce(numpy.dot, (mo1cas.T, s, mo2cas))
print('<CAS1|CAS2> SVD eig = %s' % numpy.linalg.svd(s)[1])
print('det(<CAS1|CAS2>)    = %s' % numpy.linalg.det(s))
