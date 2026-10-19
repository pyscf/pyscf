import pyscf

cell = pyscf.M(
    atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
''',
    basis='gth-szv',
    pseudo='gth-pade',
    a='''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000''',
    unit='B',
    #verbose=4
)

mf = cell.KRKS(xc='pbe', kpts=cell.make_kpts([2]*3))
opt = mf.Gradients().optimizer()

# By default, both the crystal lattice and atomic positions are optimized.
opt.run()

# Optimize the crystal lattice, while the relative atomic position in unit cell
# are fixed.
opt.target = 'lattice'
opt.run()

# Optimize the atomic position in the unit cell
opt.target = 'atoms'
opt.run()

# The geomeTRIC solver can be used directly when custom stopping thresholds
# are needed. These displacement thresholds are in Angstrom.
from pyscf.pbc.geomopt.geometric_solver import GeometryOptimizer

mf_geo = cell.KRHF(kpts=cell.make_kpts([1, 1, 3]), exxdiv=None)
mf_geo.conv_tol = 1e-14
mf_geo.conv_tol_grad = 1e-8
geo_opt = GeometryOptimizer(mf_geo).set(params={
    'convergence_energy': 1e-4,  # Eh
    'convergence_grms': 1e-5,    # Eh/Bohr
    'convergence_gmax': 1e-5,    # Eh/Bohr
    'convergence_drms': 1.2e-2,  # Angstrom
    'convergence_dmax': 1.8e-2,  # Angstrom
})
optimized_cell = geo_opt.run()
