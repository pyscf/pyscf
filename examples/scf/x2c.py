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

mf = scf.sfx2c(scf.RHF(mol))
energy = mf.kernel()
print('E = %.12f, ref = -76.081765438082' % energy)


mol.spin = 1
mol.charge = 1
mol.build(0, 0)

mf = scf.sfx2c(scf.UHF(mol))
energy = mf.scf()
print('E = %.12f, ref = -75.687130144740' % energy)

