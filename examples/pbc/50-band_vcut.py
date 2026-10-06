''' This example uses `get_bands` for HF band structure calculations using `vcut_sph/ws` potentials.
'''


import numpy as np

from pyscf.pbc import gto, scf

from pyscf.data.nist import BOHR
from ase.dft.kpoints import sc_special_points as special_points, get_bandpath


atom = '''
Si    0.0000000000    0.0000000000    0.0000000000
Si    1.3577500000    1.3577500000    1.3577500000
'''
a = '''
2.7155000000 2.7155000000 0.0000000000
0.0000000000 2.7155000000 2.7155000000
2.7155000000 0.0000000000 2.7155000000
'''
basis = '''
Si  S
    1.271038   -2.675576e-01
    0.307669    3.996909e-01
    0.141794    5.784306e-01
Si  S
    0.062460    1.000000e+00
Si  P
    1.610683   -2.629981e-02
    0.384570    3.047784e-01
    0.148473    5.453783e-01
Si  P
    0.055964    1.000000e+00
Si  D
    0.285590    1.000000e+00
'''
pseudo = 'gth-hf-rev'

cell = gto.M(atom=atom, a=a, basis=basis, pseudo=pseudo).set(verbose=4)
cell.mesh = [23,23,23]
nocc = cell.nelectron // 2

kpts = cell.make_kpts([2,2,2])

mf = scf.KRHF(cell, kpts, exxdiv='vcut_ws')
mf.kernel()

# ASE preparation
lat_symm = 'fcc'
sp_points_name = 'LGXWKG'
npoints = 50
latvec = cell.lattice_vectors() * BOHR
points = special_points[lat_symm]
sp_points_ase = [points[s] for s in sp_points_name]
kpts_band, kpath, sp_points = get_bandpath(sp_points_ase, latvec, npoints=npoints)
kpts_band = cell.get_abs_kpts(kpts_band)

# vcut_ws bands
en_vs_k_all = {}
for exxdiv in ['ewald', 'vcut_ws', 'vcut_sph']:
    mf1 = scf.KRHF(cell, kpts, exxdiv=exxdiv)
    for k in ['mo_coeff','mo_occ','mo_energy']:
        setattr(mf1, k, getattr(mf, k))
    mf1.converged = True

    band_energy = mf1.get_bands(kpts_band)[0]
    en_vs_k = np.asarray(band_energy, order='C').T  # nband,nkpts
    en_vs_k_all[exxdiv] = (en_vs_k - en_vs_k[nocc-1].max()) * 27.211399


''' Plot band
'''
from matplotlib import pyplot as plt

figsize = (3, 3.5)
fig = plt.figure(figsize=figsize)
ax = fig.gca()

emin = -20
emax = 40
zorder0 = 10

def plot1(en_vs_k, lstyls, label):
    for i,band in enumerate(en_vs_k):
        label1 = label if i == 0 else None
        ax.plot(kpath, band, **lstyls, label=label1)

lstyls = [
    {'ls':'-', 'lw':3, 'color':'#CBCBCD', 'zorder':zorder0+1},
    {'ls':'--', 'lw':1.5, 'color':'#597DAD', 'zorder':zorder0+3},
    {'ls':'--', 'lw':1, 'color':'#CB8680', 'zorder':zorder0+2},
]

for i,exxdiv in enumerate(['ewald', 'vcut_ws', 'vcut_sph']):
    plot1(en_vs_k_all[exxdiv], lstyls[i], exxdiv)

leg = ax.legend(frameon=True, loc='upper right')
leg.set_zorder(zorder0+100)

ax.axhline(0, ls='--', lw=0.7, color='k', zorder=zorder0)
for y in sp_points:
    ax.axvline(y, ls='-', lw=0.7, color='k', zorder=zorder0)

ax.set_xticks(sp_points)
ax.set_xticklabels([r'$\Gamma$' if x == 'G' else r'$%s$'%(x) for x in sp_points_name])

ax.set_xlim((kpath[0], kpath[-1]))

ax.set_ylim([emin, emax])
ax.set_ylabel('Band energy (eV)')

plt.tight_layout()

plt.savefig('band.pdf')
plt.close(fig)
