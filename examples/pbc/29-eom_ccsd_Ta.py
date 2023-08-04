#!/usr/bin/env python

'''
Showing use of IP/EA-EOM-CCSD(T)*a by Matthews and Stanton
'''

import numpy as np
from pyscf import cc
from pyscf.pbc import cc as pbccc
from pyscf.pbc import scf as pbchf
from pyscf.pbc import gto
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc.cc.eom_kccsd_rhf import EOMIP_Ta, EOMEA_Ta

nmp = [1, 1, 2]
cell = gto.M(
    unit='B',
    a=[[0., 3.37013733, 3.37013733],
       [3.37013733, 0., 3.37013733],
       [3.37013733, 3.37013733, 0.]],
    mesh=[24,]*3,
    atom='''C 0 0 0
              C 1.68506866 1.68506866 1.68506866''',
    basis='gth-szv',
    pseudo='gth-pade',
    verbose=4
)

# First we perform our mean-field calculation with a [1,1,2] grid
kpts = cell.make_kpts(nmp)
kpts -= kpts[0]
kmf = pbchf.KRHF(cell, kpts, exxdiv=None)#, conv_tol=1e-10)
kpoint_energy = kmf.kernel()

# Perform our ground state ccsd calculation
mykcc = pbccc.KRCCSD(kmf)
eris = mykcc.ao2mo()
kccsd_energy = mykcc.ccsd(eris=eris)[0]
ekcc = mykcc.ecc

# To run an EOM-CCSD(T)*a calculation, you need to use the EOMIP_Ta/
# /EOMEA_Ta classes that will add in the contribution of T3[2] to
# T1/T2 as well as any other relevant EOM-CCSD intermediates.
myeom = EOMIP_Ta(mykcc)
# We then call the ip-ccsd* function that will find both the right
# and left eigenvectors of EOM-CCSD (with perturbed intermediates)
# and run EOM-CCSD*
myeom.ipccsd_star(nroots=2, kptlist=[0], eris=eris)

# If we need to run both an IP and EA calculation, the t3[2] intermediates
# would need to be created two times. Because creating the t3[2] intermediates
# is fairly expensive, we can reduce the computational cost by creating
# the intermediates directly and passing them in.
from pyscf.pbc.cc import eom_kccsd_rhf
imds = eom_kccsd_rhf._IMDS(mykcc, eris=eris)
imds = imds.make_t3p2_ip_ea(mykcc)
# Now call EOMIP_Ta/EOMEA_Ta directly using these intermediates
myeom = EOMIP_Ta(mykcc)
myeom.ipccsd_star(nroots=2, kptlist=[0], imds=imds)
myeom = EOMEA_Ta(mykcc)
myeom.eaccsd_star(nroots=2, kptlist=[0], imds=imds)
