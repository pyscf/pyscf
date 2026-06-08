#!/usr/bin/env python


'''
PM localization for PBC systems using both Gamma-point and k-point approaches.
'''

import sys
import numpy as np
from pyscf.pbc import gto, scf, lo, tools
from pyscf.lib import logger

from pyscf.data.nist import BOHR
from ase.dft.kpoints import sc_special_points as special_points, get_bandpath

log = logger.Logger(sys.stdout, 6)


''' Perform a k-point SCF calculation
'''
cell = gto.Cell()
cell.atom = '''
Si    0.0000000000    0.0000000000    0.0000000000
Si    1.3575000000    1.3575000000    1.3575000000
'''
cell.a = '''
2.7150000000 2.7150000000 0.0000000000
0.0000000000 2.7150000000 2.7150000000
2.7150000000 0.0000000000 2.7150000000
'''
cell.basis = '''
#BASIS SET: (4s,4p,1d) -> [2s,2p,1d] Si
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
''' # gth-cc-pvdz
cell.pseudo = 'gth-pbe'
cell.mesh = [23]*3
cell.verbose = 4
cell.build()

kmesh = [5,5,5]
kpts = cell.make_kpts(kmesh)
nkpts = len(kpts)

mf = scf.KRKS(cell, kpts=kpts).set(xc='pbe').run()


''' Reference bands from get_bands
'''
lat_symm = 'fcc'
sp_points_name = 'WLGXWK'
npoints = 50

latvec = cell.lattice_vectors() * BOHR
points = special_points[lat_symm]
sp_points_ase = [points[s] for s in sp_points_name]
kpts_band, kpath, sp_points = get_bandpath(sp_points_ase, latvec, npoints=npoints)
kpts_band = cell.get_abs_kpts(kpts_band)

band_energy = mf.get_bands(kpts_band)[0]
en_vs_k_ref = np.asarray(band_energy, order='C').T  # nband,nkpts

# shift VBM to zero
nocc = cell.nelectron // 2
en_vs_k_ref = (en_vs_k_ref - en_vs_k_ref[nocc-1].max()) * 27.211399


''' Wannier interpolation using PMWFs
'''
nocc = cell.nelectron // 2
nvir = 0
nband = nocc + nvir # number of bands to interpolate
mo = np.asarray([x[:,:nband] for x in mf.mo_coeff])
mlo = lo.KPM(cell, mo, kpts)    # k-point PM with complex rotations
mlo.kernel()

# stability check
while True:
    mo, stable = mlo.stability(return_status=True)
    if stable:
        break
    mlo.kernel(mo)

# interpolate
band_energy_wann = lo.interpolation.wannier_interpolation(mf, kpts_band, mlo.mo_coeff)[0]
en_vs_k_wann = np.asarray(band_energy_wann, order='C').T  # nband,nkpts

# shift VBM to zero
nocc = cell.nelectron // 2
en_vs_k_wann = (en_vs_k_wann - en_vs_k_wann[nocc-1].max()) * 27.211399


''' Fourier interpolation using KS orbitals without localization
'''
ks_coeff = [x[:,:nband] for x in mf.mo_coeff]
band_energy_ks = lo.interpolation.wannier_interpolation(mf, kpts_band, ks_coeff)[0]
en_vs_k_ks = np.asarray(band_energy_ks, order='C').T  # nband,nkpts

# shift VBM to zero
nocc = cell.nelectron // 2
en_vs_k_ks = (en_vs_k_ks - en_vs_k_ks[nocc-1].max()) * 27.211399


''' Plot band
'''
from matplotlib import pyplot as plt

figsize = (3, 3.5)
fig = plt.figure(figsize=figsize)
ax = fig.gca()

emin = -15
emax = 10
zorder0 = 10

def plot1(en_vs_k, lstyls, label):
    for i,band in enumerate(en_vs_k):
        label1 = label if i == 0 else None
        ax.plot(kpath, band, **lstyls, label=label1)

lstyls_ref = {'ls':'-', 'lw':3, 'color':'#CBCBCD', 'zorder':zorder0+1}
plot1(en_vs_k_ref, lstyls_ref, 'Ref')

lstyls_wann = {'ls':'--', 'lw':1.5, 'color':'#597DAD', 'zorder':zorder0+3}
plot1(en_vs_k_wann, lstyls_wann, 'PMWF')

lstyls_ks = {'ls':'--', 'lw':1, 'color':'#CB8680', 'zorder':zorder0+2}
plot1(en_vs_k_ks, lstyls_ks, 'KSWF')

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
