from pyscf import gto
from pyscf import scf
mol = gto.Mole()
mol.verbose = 4
mol.atom = 'O 0 0 0; O 0 0 1.2'
mol.basis = 'ccpvdz'
mol.spin = 2
mol.build()

m = scf.RHF(mol)
print(m.scf())

from pyscf import mcscf
mc = mcscf.CASSCF(mol, m, 6, 6)
emc = mc.mc1step()[0]
mc.analyze()
print('E = %.9g' % emc)
print('S^2 = %.7f, 2S+1 = %.7f' % mcscf.spin_square(mc))

