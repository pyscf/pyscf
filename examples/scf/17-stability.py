#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
SCF wavefunction stability analysis
'''

from pyscf import gto, scf

mol = gto.M(atom='C 0 0 0; O 0 0 1.2', basis='631g*')
mf = scf.RHF(mol).run()
# Run stability analysis for the SCF wave function
mf.stability()

#
# This instability is associated to a symmetry broken answer.  When point
# group symmetry is used, the same wave function is stable.
#
mol = gto.M(atom='C 0 0 0; O 0 0 1.2', basis='631g*', symmetry=1)
mf = scf.RHF(mol).run()
mf.stability()

#
# In the RHF method, there are three kinds of instability: the internal
# instability for the optimal solution in RHF space and the external
# instability (including the RHF -> UHF instability, and real -> complex
# instability).  By default the stability analysis only detects the internal
# instability.  The external instability can be enabled by passing keyword
# argument external=True to the stability function.
#
mf.stability(external=True)

#
# If the SCF wavefunction is unstable, the stability analysis program can
# transform the SCF wavefunction and generate a set of initial guess (orbitals).
# The initial guess can be used to make density matrix and fed into a new SCF
# iteration (see 15-initial_guess.py).
#
mol = gto.M(atom='O 0 0 0; O 0 0 1.2222', basis='631g*')
mf = scf.UHF(mol).run()
mo1 = mf.stability()[0]
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)
mf.stability()

#
# Also the initial guess orbitals from stability analysis can be used as with
# second order SCF solver
#
mf = mf.newton().run(mo1, mf.mo_occ)
mf.stability()

#
# The UHF method has the internal
# instability for the optimal solution in UHF space and the external
# instability (UHF -> GHF, and real -> complex).  By default the stability
# analysis only detects the internal instability.  The external instability
# can be enabled by passing keyword argument external=True to the stability
# function.
#
mf.stability(external=True)

#
# Loop for optimizing orbitals until stable
#
from pyscf.lib import logger

def stable_opt_internal(mf):
    log = logger.new_logger(mf)
    mo1, _, stable, _ = mf.stability(return_status=True)
    cyc = 0
    while (not stable and cyc < 10):
        log.note('Try to optimize orbitals until stable, attempt %d' % cyc)
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf = mf.run(dm1)
        mo1, _, stable, _ = mf.stability(return_status=True)
        cyc += 1
    if not stable:
        log.note('Stability Opt failed after %d attempts' % cyc)
    return mf
print('loop example')
mol = gto.M(atom='C 0 0 0; C 0 0 2.0', basis='631g*')
mf2 = scf.UHF(mol).run()
mf2 = stable_opt_internal(mf2)

#
# Some cases need tight tolerance to give the stability correctly
#
from pyscf.scf.stability import uhf_internal
mol = gto.M(atom='''
O         -0.23497692        0.90193619       -0.06868800
H          1.10502308        0.90193619       -0.06868800
H         -0.68227811        2.16507579       -0.06868800''', 
basis = 'def2svp', spin=0)
mf = mol.UHF().run()
uhf_internal(mf, verbose=4) # wrong
uhf_internal(mf, verbose=4, tol=1e-6) # correct

# the tolerance and nroots are both adjustable in the stability analysis
mf.stability(verbose=4, tol=1e-6, nroots=5)
