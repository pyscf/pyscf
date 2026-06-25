"""
Example for Bethe-Salpeter equation.

########
Reference results for acetone / B3LYP / def2-SVP
acetone geometry from: J. Phys. Chem. Lett. 2016, 7, 3, 586-591

* GW step (fully analytic GW, quasiparticle equation solved iteratively)
                HOMO (eV)   LUMO (eV)
Turbomole        -8.78        2.75
PySCF            -8.79        2.75

* First three singlet excitations (eV) for BSE
                S1      S2      S3
Turbomole      3.41    7.35    8.56
PySCF          3.41    7.36    8.58

* First three triplet excitations (eV) for BSE
                T1      T2      T3
Turbomole      2.68    4.67    7.14
PySCF          2.67    4.67    7.14

"""
import numpy as np
from pyscf import gto, dft
from pyscf.gw.gw_ac import GWAC
from pyscf.gw.ugw_ac import UGWAC
from pyscf.gw.bse import BSE, bse_lanczos, lanczos_estimate_spectrum

# restricted
mol = gto.Mole()
mol.verbose = 5
mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.7571, 0.0, 0.5861)], [1, (-0.7571, 0.0, 0.5861)]]
mol.basis = 'def2-svp'
mol.build()
mf = dft.RKS(mol, xc='pbe')
mf.kernel()

# GW-AC/BSE
gw = GWAC(mf)
gw.kernel()
bse = BSE(gw)
# Davidson algorithm for singlet excitation
bse.TDA = False
bse.kernel('s')
bse.analyze()
# Davidson algorithm for triplet excitation, turn on TDA
bse.TDA = True
bse.kernel('t')
bse.analyze()
# full diagonalization for triplet excitation
bse.full_diagonalization('t')
bse.analyze()

eta = 0.01 # spectrum broadening in eV
omega = np.linspace(0.0, 1.0, 1000)[:, None] + 1j * eta # (nω, 1)

ao_dip = mol.intor('int1e_r', comp=3)
nocc = mol.nelectron // 2
mo_dip = np.einsum('xij,ia,jb->xab', ao_dip, mf.mo_coeff[:, :nocc], mf.mo_coeff[:, nocc:])

bse.TDA = False
lanczos_spectra = []
for j in range(3):
    alphas, betas = bse_lanczos(bse, multi='s', u1=mo_dip[j].flatten(), nsteps=500)
    freqs, density = lanczos_estimate_spectrum(alphas, betas, (0, 1), eta, 1000)
    lanczos_spectra.append(density)
mean_spectrum = np.mean(lanczos_spectra, axis=0) * 4 * np.pi
print("spectrum from Lanczos algorithm:")
for i in range(len(freqs)):
    print(f"{freqs[i]:.6f} {mean_spectrum[i]:.6f}")

# Energy-specific BSE, target excitations above 0.4 AU
gw = GWAC(mf)
gw.kernel()
bse = BSE(gw)
bse.kernel('s', e_min=0.4)
bse.analyze()
bse.kernel('t', e_min=0.4)
bse.analyze()

# unrestricted
mol = gto.Mole()
mol.verbose = 5
mol.atom = [[8, (0.0, 0.0, 0.0)], [1, (0.7571, 0.0, 0.5861)], [1, (-0.7571, 0.0, 0.5861)]]
mol.charge = 1
mol.spin = 1
mol.basis = 'def2-svp'
mol.build()
mf = dft.UKS(mol, xc='pbe')
mf.kernel()

# UGWAC/BSE
gw = UGWAC(mf)
gw.kernel()

bse = BSE(gw)
bse.kernel('u')
bse.analyze()
