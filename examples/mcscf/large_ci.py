from pyscf import gto
from pyscf import scf
from pyscf import mcscf

b = 1.4
mol = gto.Mole()
mol.build(
    verbose = 0,
    #output = 'out-large_ci',
    atom = [['N', (0.,0.,0.)], ['N', (0.,0.,b)]],
    basis = 'cc-pvdz',
    symmetry = True,
)
m = scf.RHF(mol)
m.scf()

mc = mcscf.CASSCF(m, 6, 6)
mc.mc1step()

from pyscf import fci
print(' string alpha, string beta, CI coefficients')
for c,ia,ib in fci.addons.large_ci(mc.ci, 6, 6, tol=.05):
    print('  %9s    %9s    %.12f' % (ia, ib, c))
