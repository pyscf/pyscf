#!/usr/bin/env python

'''
Wannier interpolation using PM localized orbitals.

This example and 34-pm_wannier_interpolation.py demonstrates how to
construct Pipek-Mezey Wannier functions (PMWF) for periodic k-point
calculations. The resulting PMWFs are then used to interpolate the electronic
band structure.
'''


from pyscf.pbc import gto, scf
from pyscf.pbc.lo import KPM
from pyscf.lib import logger

atom = '''
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116
'''
a = numpy.eye(3) * 3
basis = 'cc-pvdz'
nband = 6

kmesh = [5,1,1]

cell = gto.M(atom=atom, basis=basis, a=a)
cell.verbose = 4

log = logger.new_logger(cell, verbose=6)

kpts = cell.make_kpts(kmesh)

''' SCF
'''
mf = scf.KRKS(cell, kpts=kpts).density_fit()
mf.xc = 'pbe'
mf.kernel()

''' Reference bands
'''
blat = cell.reciprocal_vectors()
scaled_kpts_band = numpy.linspace(-0.5,0.5,30)
kpts_band = scaled_kpts_band[:,None] * blat[0]
mo_energy, mo_coeff = mf.get_bands(kpts_band)

band_energy = numpy.asarray([x[:nband] for x in mo_energy]).T * 27.211399

''' PM WF localization
'''
mo = [x[:,:nband] for x in mf.mo_coeff]
mlo = KPM(cell, mo, kpts)
mlo.kernel()

''' WF interpolation
'''
mo_energy, mo_coeff = wannier_interpolation(mf, kpts_band, mlo.mo_coeff)
band_energy_wann = numpy.asarray([x[:nband] for x in mo_energy]).T * 27.211399

err = abs(band_energy-band_energy_wann).max()
log.info('band interpolation error: %.10f eV', err)

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(2.5,2))
ax = fig.gca()

colors = ['gray', 'k']

for iband,band_idx in enumerate([3,4,5]):
    if iband == 0:
        label1 = 'ref'
        label2 = 'interp'
    else:
        label1 = label2 = None
    ax.plot(scaled_kpts_band, band_energy[band_idx], '-', lw=1, color=colors[0],
            label=label1)
    ax.plot(scaled_kpts_band, band_energy_wann[band_idx], '.', markersize=3,
            color=colors[1], label=label2)

ax.legend(frameon=False)

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'Band energy (eV)')

plt.tight_layout()

plt.savefig('band.pdf')
plt.close(fig)
