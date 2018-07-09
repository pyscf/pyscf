import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.dft as pbcdft

import matplotlib.pyplot as plt

from ase.lattice import bulk
from ase.dft.kpoints import ibz_points, get_bandpath
c = bulk('C', 'diamond', a=3.5668)
print c.get_volume()
points = ibz_points['fcc']
G = points['Gamma']
X = points['X']
W = points['W']
K = points['K']
L = points['L']
band_kpts, kpath, sp_points = get_bandpath([L, G, X, W, K, G], c.cell, npoints=30)

#
# band for Gamma point DFT
#
cell = pbcgto.Cell()
cell.atom=pyscf_ase.ase_atoms_to_pyscf(c)
cell.a=c.cell

cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose=5
cell.build(None,None)

mf = pbcdft.RKS(cell)
mf.xc = 'pbe,pbe'
print mf.kernel()

e_kn = mf.get_bands(band_kpts)[0]

emin = -1
emax = 2

plt.figure(figsize=(5, 6))
nbands = cell.nao_nr()
for n in range(nbands):
    plt.plot(kpath, [e[n] for e in e_kn])
for p in sp_points:
    plt.plot([p, p], [emin, emax], 'k-')
plt.plot([0, sp_points[-1]], [0, 0], 'k-')
plt.xticks(sp_points, ['$%s$' % n for n in ['L', 'G', 'X', 'W', 'K', r'\Gamma']])
plt.axis(xmin=0, xmax=sp_points[-1], ymin=emin, ymax=emax)
plt.xlabel('k-vector')

plt.show()



#
# band for k-point DFT
#
c = bulk('C', 'diamond', a=3.5668)
print c.get_volume()
points = ibz_points['fcc']
G = points['Gamma']
X = points['X']
W = points['W']
K = points['K']
L = points['L']
band_kpts, kpath, sp_points = get_bandpath([L, G, X, W, K, G], c.cell, npoints=30)

cell = pbcgto.Cell()
cell.atom=pyscf_ase.ase_atoms_to_pyscf(c)
cell.a=c.cell

cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.verbose=5
cell.build(None,None)

kmf = pbcdft.KRKS(cell, cell.make_kpts([2,2,2]))
kmf.xc = 'lda,vwn'
print kmf.kernel()

e_kn = kmf.get_bands(band_kpts)[0]

emin = -1
emax = 2

plt.figure(figsize=(5, 6))
nbands = cell.nao_nr()
for n in range(nbands):
    plt.plot(kpath, [e[n] for e in e_kn])
for p in sp_points:
    plt.plot([p, p], [emin, emax], 'k-')
plt.plot([0, sp_points[-1]], [0, 0], 'k-')
plt.xticks(sp_points, ['$%s$' % n for n in ['L', 'G', 'X', 'W', 'K', r'\Gamma']])
plt.axis(xmin=0, xmax=sp_points[-1], ymin=emin, ymax=emax)
plt.xlabel('k-vector')

plt.show()

