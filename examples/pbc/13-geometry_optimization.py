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
