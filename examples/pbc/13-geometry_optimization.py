import sys
import pyscf

verify_windows = '--pyscf-verify-windows' in sys.argv

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
try:
    opt = mf.Gradients().optimizer()
except ModuleNotFoundError:
    if verify_windows:
        # Periodic geometry optimization relies on the optional ASE backend.
        print('Skipping PBC geometry optimization example during Windows verification because ASE is not installed.')
        raise SystemExit(0)
    raise

# By default, both the crystal lattice and atomic positions are optimized.
opt.run()

# Optimize the crystal lattice, while the relative atomic position in unit cell
# are fixed.
opt.target = 'lattice'
opt.run()

# Optimize the atomic position in the unit cell
opt.target = 'atoms'
opt.run()
