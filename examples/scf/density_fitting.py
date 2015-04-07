from pyscf import gto
from pyscf import scf

mol = gto.Mole()
mol.build(
    verbose = 0,
    atom = '''8  0  0.     0
              1  0  -0.757 0.587
              1  0  0.757  0.587''',
    basis = 'ccpvdz',
)

mf = scf.density_fit(scf.RHF(mol))
energy = mf.scf()
print('E = %.12f, ref = -76.0259362997' % energy)


mol.spin = 1
mol.charge = 1
mol.build(0, 0)

mf = scf.density_fit(scf.UHF(mol))
# the default auxiliary basis is Weigend Coulomb Fitting basis.
mf.auxbasis = 'cc-pvdz-fit'
energy = mf.scf()
print('E = %.12f, ref = -75.6310072359' % energy)

