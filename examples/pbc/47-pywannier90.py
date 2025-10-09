#!/usr/bin/env python
#
# Author: Hung Pham <pqh3.14@gmail.com>
#
"""
This example shows how to construct MLWFs from the pywannier90 tool.
The MLWFs are then used to interpolate the k-space hamiltonian, hence the band structure
"""

import numpy as np
from pyscf.pbc import gto, scf, cc, df
from pyscf.pbc.tools import pywannier90


# Definte unit cell 
cell = gto.Cell()
cell.atom = [['Si',[0.0,0.0,0.0]], ['Si',[1.35775, 1.35775, 1.35775]]]
cell.a = [[0.0, 2.7155, 2.7155], [2.7155, 0.0, 2.7155], [2.7155, 2.7155, 0.0]]
cell.basis = 'gth-dzv'
cell.pseudo = 'gth-pade'
cell.build()


# PBE calculation
kmesh = [3, 3, 3]
kpts = cell.make_kpts(kmesh)
nkpts = kpts.shape[0]
kks = scf.KKS(cell, kpts)
kks.xc = 'PBE'
kks.chkfile = 'chkfile'
kks.init_guess = 'chkfile'
kks.run()


# the kks object can be saved and loaded before running pyWannier90
# Hence, one does not need to perform the kks calculation every time 
# the localization parameters change, for example, the guessing orbitals or the disentanglement windows
pywannier90.save_kmf(kks, 'chk_PBE')
kmf = pywannier90.load_kmf('chk_PBE')


# (1) Construct MLWFs
num_wann = 8
keywords = \
"""
begin projections
Si:sp3
end projections
"""
w90 = pywannier90.W90(kmf, cell, kmesh, num_wann, other_keywords=keywords)
w90.kernel()    


# (2) Export the MWLFs in the xsf format for plotting with VESTA
w90.plot_wf(supercell=kmesh, grid=[20,20,20])


# (3) Export wannier90.mmn, wannier90.amn, wannier90.eig matrix and then run a wannier90 using these
w90.export_AME()
w90.kernel(external_AME='wannier90')


# (4) Interpolate the Fock or band structure using the Slater-Koster scheme
# This can be applied to either a (restricted or unrestricted) DFT or HF wfs to get the band structure
band_kpts = kpts + 0.5 * kpts[1]
frac_kpts = cell.get_scaled_kpts(band_kpts)
interpolated_fock = w90.interpolate_ham_kpts(frac_kpts)     # the interpolated Fock
bands = w90.interpolate_band(frac_kpts)                     # the interpolated band by pyWannier90
bands_ref = kks.get_bands(band_kpts)                        # Band interpolated by PySCF


# This difference should be decreasing when a denser k-mesh is used
print("Difference in the eigenvalues interpolated by scf.get_bands function and by pywannier90: %10.8e" % \
(abs(bands[0] -bands_ref[0]).max()))


# (5) Plotting band structure using mcu: https://github.com/hungpham2017/mcu
import mcu
pyscf_plot = mcu.PYSCF(cell)
kpath = '''    
L 0.5 0.5 0.5       
G 0.0 0.0 0.0
X 0.5 0.0 0.5
W 0.5 0.25 0.75
K 0.375 0.375 0.75
G 0.0 0.0 0.0
'''
frac_kpts, abs_kpts = pyscf_plot.make_kpts(kpath, 11)
bands = w90.interpolate_band(frac_kpts, use_ws_distance=True)
pyscf_plot.set_kpts_bands([frac_kpts, bands])
pyscf_plot.get_bandgap()
pyscf_plot.plot_band(ylim=[-17,17], klabel=kpath)
