from ase.units import Ry, eV
from ase.calculators.siesta.siesta import Siesta
from ase import Atoms
from pyscf.nao.m_system_vars import system_vars_c
from pyscf.nao.m_prod_basis import prod_basis_c
import numpy as np

label = 'h2o'
H2O = Atoms('H2O', positions=np.array(
              [[0.00000000, -0.00164806, 0.00000000],
              [0.77573521, 0.59332141, 0.00000000],
              [-0.77573521, 0.59332141, 0.00000000]
              ])
)

calc = Siesta(
    label=label,
    mesh_cutoff=150 * Ry,
    basis_set='DZP',
    energy_shift=(10 * 10**-3) * eV,
    fdf_arguments={
    'SCFMustConverge': False,
    'COOP.Write': True,
    'WriteDenchar': True,
    'PAO.BasisType': 'split',
    'DM.Tolerance': 1e-4,
    'DM.MixingWeight': 0.01,
    'MaxSCFIterations': 100,
    'DM.NumberPulay': 4,
    'xml.Write': True}
)

H2O.set_calculator(calc)
#run siesta
print(H2O.get_potential_energy())
pb_params = {'cross_check_global_vertex': True}

sv = system_vars_c(Atoms = H2O, label=label)
pb = prod_basis_c(sv, input_params_pb = pb_params)
print('atom2sp: ', sv.atom2sp)
print(pb.pb_params)
