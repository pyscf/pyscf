#!/usr/bin/env python

import pyscf
from pyscf import gto, dft

'''
Composite 3c methods: B97-3c, r2SCAN-3c and wB97X-3c.

B97-3c (J. Chem. Phys. 148, 064104 (2018)): B97 + D3(BJ) + SRB in
def2-mTZVP.  r2SCAN-3c (J. Chem. Phys. 154, 064103 (2021)): r2SCAN + D4 +
gCP in def2-mTZVPP.  wB97X-3c (J. Chem. Phys. 158, 014103 (2023)):
wB97X-V + D4 in the ECP-based Grimme vDZP basis (no gCP).

The methods are set up with dft3c() or the mol.RKS3C()/mol.UKS3C()
conveniences.  The molecular basis (and ECPs), the XC functional, and
(via mf.xc) the dispersion and gCP/SRB corrections are configured
automatically; density_fit() selects the RI-J auxiliary basis
(def2-mTZVPP-RIJ) automatically for B97-3c and r2SCAN-3c.

This example requires the pyscf-dispersion and basis-set-exchange
packages.
'''

mol = gto.M(atom='''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''')

#
# The simplest way: mol.RKS3C() works like mol.RKS().
# Density fitting can be applied before or after the 3c setup.
#
mf = mol.RKS3C().density_fit()
mf.kernel()
print('B97-3c  total energy = %.12f' % mf.e_tot)
# The dispersion and gCP/SRB corrections are applied automatically and
# are reported in mf.scf_summary.
print('  E(disp) = %.12f  E(gCP/SRB) = %.12f'
      % (mf.scf_summary['dispersion'], mf.scf_summary['gcp']))

#
# The method can be selected explicitly, for dft3c() as well as for
# mol.RKS3C(method=...).  The default method is b97-3c.
#
mf = dft.RKS(mol).dft3c('r2scan-3c').density_fit()
mf.kernel()
print('r2SCAN-3c total energy = %.12f' % mf.e_tot)

# mol.RKS3C(method='r2scan-3c') is the equivalent of the above
mf = mol.RKS3C(method='r2scan-3c').density_fit()
mf.kernel()
print('r2SCAN-3c total energy = %.12f' % mf.e_tot)

# UKS3C, ROKS3C and GKS3C are available
#mol = gto.M(atom='O 0 0 0', spin=2)
#mf = mol.UKS3C().run()
#mf = mol.ROKS3C().run()
#mf = mol.GKS3C().run()

#
# wB97X-3c: wB97X-V + D4 in the ECP-based Grimme vDZP basis (no gCP).
# density_fit() uses the def2 universal JK-fit basis.
#
#mf = mol.RKS3C(method='wb97x-3c').run()

#
# All spellings of the composite method are equivalent: b97-3c, b97_3c and
# the full libxc canonical name gga_xc_b97_3c all include the dispersion
# and gCP/SRB corrections.
#
mf = dft.RKS(mol, xc='gga_xc_b97_3c').density_fit()
mf.kernel()
print('gga_xc_b97_3c  (DF)   = %.12f' % mf.e_tot)
