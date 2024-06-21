#!/usr/bin/env python

'''
A simple example to run EPH calculation.
'''

from pyscf import gto, dft, eph

mol = gto.M()
mol.atom = [['O', [0.000000000000,  -0.000000000775,   0.923671924285]],
            ['H', [-0.000000000000,  -1.432564848017,   2.125164039823]],
            ['H', [0.000000000000,   1.432564848792,   2.125164035930]]]
mol.unit = 'Bohr'
mol.basis = 'sto3g'
mol.verbose = 4
mol.build() # this is a pre-computed relaxed geometry

mf = mol.RHF()
mf.conv_tol = 1e-16
mf.conv_tol_grad = 1e-10
mf.kernel()

myeph = eph.EPH(mf)
grad = mf.nuc_grad_method().kernel()
print("Force on the atoms/au:")
print(grad)
eph, omega = myeph.kernel(mo_rep=True)
print(np.amax(eph))


mol = gto.M(atom='N 0 0 0; N 0 0 2.100825', basis='def2-svp', verbose=4, unit="bohr")
# this is a pre-computed relaxed molecule
# for geometry relaxation, refer to pyscf/example/geomopt
mf = dft.RKS(mol, xc='pbe,pbe')
mf.run()

grad = mf.nuc_grad_method().kernel()
assert (abs(grad).sum()<1e-5) # making sure the geometry is relaxed

myeph = eph.EPH(mf)
mat, omega = myeph.kernel()
print(mat.shape, omega)
