#!/usr/bin/env python

from pyscf import gto
from pyscf import scf
from pyscf import mrpt
from pyscf.dmrgscf import DMRGSCF

#
# NEVPT2 calculation requires about 200 GB memory in total
#

b = 1.5
mol = gto.Mole()
mol.verbose = 5
mol.output = 'cr2-%3.2f.out' % b
mol.atom = [
    ['Cr',(  0.000000,  0.000000, -b/2)],
    ['Cr',(  0.000000,  0.000000,  b/2)],
]
mol.basis = {'Cr': 'ccpvdz-dk'}
mol.symmetry = True
mol.build()

m = scf.RHF(mol).x2c().run(conv_tol=1e-9, chkfile='hf_chk-%s'%b, level_shift=0.5)
#
# Note: stream operations are used here.  This one line code is equivalent to
# the following serial statements.
#
#m = scf.sfx2c1e(scf.RHF(mol))
#m.conv_tol = 1e-9
#m.chkfile = 'hf_chk-%s'%b
#m.level_shift = 0.5
#m.kernel()

dm = m.make_rdm1()
m.level_shift = 0
m.scf(dm)

mc = DMRGSCF(m, 20, 28)  # 20o, 28e
mc.fcisolver.maxM = 1000
mc.fcisolver.tol = 1e-6

mc.chkfile = 'mc_chk_18o-%s'%b
cas_occ = {'A1g':4, 'A1u':4,
           'E1ux':2, 'E1uy':2, 'E1gx':2, 'E1gy':2,
           'E2ux':1, 'E2uy':1, 'E2gx':1, 'E2gy':1}
mo = mc.sort_mo_by_irrep(cas_occ)
mc.kernel(mo)

#
# DMRG-NEVPT2 (slow version)
# not available since StackBlock 1.5.3
#
# mrpt.NEVPT(mc).kernel()

#
# The compressed-MPS-perturber DMRG-NEVPT2 is more efficient.
#
mrpt.NEVPT(mc).compress_approx().kernel()
