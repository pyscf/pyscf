#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Initial guess from chkfile.

Caculation can be restarted by providing attribute .chkfile as initial guess.

The intermediate results are saved in the file specified by .chkfile attribute.
If no filename is assigned to .chkfile, chkfile will be initialized as a
temproray file in the temporory directory (controlled by enviroment variable
PYSCF_TMPDIR) and the temporary file will be deleted automatically if
calculation is successfully finished.  If the calculation is failed, we can
search in the output for message  "chkfile to save SCF result = XXXXX"  for the
name of the chkfile of current calculation.  This chkfile can be used as the
restart point.
'''

from pyscf import gto
from pyscf import scf

mol = gto.Mole()
mol.build(
    verbose = 0,
    atom = '''8  0  0.     0
              1  0  -0.757 0.587
              1  0  0.757  0.587''',
    basis = 'ccpvdz',
)
mol.verbose = 4

mf = scf.RHF(mol)
mf.chkfile = 'h2o.chk'
mf.kernel()

# There are many methods to restart calculation from chkfile.
# 1. Giving .chkfile and .init_guess attributes, the program will read
# .chkfile and generate initial guess
mf = scf.RHF(mol)
mf.chkfile = 'h2o.chk'
mf.init_guess = 'chk'
mf.kernel()

# 2. function from_chk can read results from a given chkfile and project the
# density matrix to current system as initial guess
mf = scf.RHF(mol)
dm = mf.from_chk('h2o.chk')
mf.kernel(dm)

# 3. Accessing the chkfile to get orbital coefficients and compute the
# density matrix as initial guess
mf = scf.RHF(mol)
mo_coeff = scf.chkfile.load('h2o.chk', 'scf/mo_coeff')
mo_occ = scf.chkfile.load('h2o.chk', 'scf/mo_occ')
dm = mf.make_rdm1(mo_coeff, mo_occ)
mf.kernel(dm)

# 4. Last method can be simplified.  Calling .__dict__.update function to
# upgrade the SCF object which assign mf.mo_coeff and mf.mo_occ etc. default
# values. Then calling make_rdm1 function to generate a density matrix.
mf = scf.RHF(mol)
mf.__dict__.update(scf.chkfile.load('h2o.chk', 'scf'))
dm = mf.make_rdm1()
mf.kernel(dm)


#
# 5. Restart from an old DIIS file
#
mf = scf.RHF(mol)
mf.diis_file = 'h2o_diis.h5'
mf.kernel()

# In another calculation, DIIS information from previous calculation can be
# restored. Previous calculation can be continued.
mf = scf.RHF(mol)
mf.diis = scf.diis.DIIS().restore('h2o_diis.h5')
mf.max_cycle = 2
mf.kernel()
