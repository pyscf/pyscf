from pyscf import gto
from pyscf import scf
from pyscf import mcscf

mol = gto.Mole()
mol.build(
    verbose = 5,
    output = None,
    atom = [['O' , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
    basis = {'O': 'cc-pvdz', 'H': 'cc-pvdz'},
    symmetry = True
)
mol.symmetry = False
m = scf.RHF(mol)
m.scf()

mol.symmetry = True

mc = mcscf.CASSCF(mol, m, 6, 6)
mc.mc1step()
