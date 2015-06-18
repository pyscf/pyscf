from pyscf import gto
from pyscf import scf
from pyscf import mcscf
from pyscf import dmrgscf

b = 1.2
mol = gto.Mole()
mol.build(
    verbose = 5,
    output = 'out-dmrgscf',
    atom = [['N', (0.,0.,0.)], ['N', (0.,0.,b)]],
    basis = {'N': 'cc-pvdz'},
    symmetry = True,
)
m = scf.RHF(mol)
m.scf()

mc = mcscf.CASSCF(m, 6, 6)
mc.max_cycle_macro = 5
mc.max_cycle_micro = 1
mc.conv_tol = 1e-5
mc.conv_tol_grad = 1e-4
mc.mc1step()
mo = mc.mo_coeff

mol.stdout.write('\n*********** Call DMRGSCF **********\n')
mc = mcscf.CASSCF(m, 8, 8)
mc.max_orb_stepsize = .05
mc.max_cycle_macro = 20
mc.max_cycle_micro = 3
mc.conv_tol = 1e-8
mc.conv_tol_grad = 1e-5
mc.fcisolver = dmrgscf.CheMPS2(mol)
mc.fcisolver.dmrg_e_convergence = 1e-9
emc = mc.mc2step(mo)[0]
print(emc)
