#!/usr/bin/env python
#
# Author: Shirong Wang <srwang20@fudan.edu.cn>
#

'''
SCF wavefunction stability analysis

Comparison between Davidson and exact (slow) approach, for all kinds of stability,
especially real GHF internal, GHF real2complex, and complex GHF internal.
'''

from pyscf import gto, scf
import numpy as np

np.set_printoptions(suppress=False, precision=6, linewidth=400)
from pyscf.scf import stability_slow, stability
from pyscf.scf.stability import stable_opt_internal
from pyscf.soscf import newton_ah


mol = gto.M(atom='''H 0.0 0.0 0.0; H 0.0 0.0 0.7''', basis='cc-pvdz', verbose=4).build()
mf = mol.RHF().run()
stability.rhf_internal(mf)
stability.rhf_external(mf)
stability_slow.rhf_internal(mf)
#stability_slow.rhf_internal_ab(mf)
stability_slow.rhf_external(mf)

mf2 = mf.to_uhf().set(max_cycle=10)
mf2.kernel(dm0 = mf2.make_rdm1())
stability.uhf_internal(mf2)
#stability.uhf_internal(mf2, with_symmetry=False)
stability.uhf_external(mf2)
stability_slow.uhf_internal(mf2)
stability_slow.uhf_external(mf2)

mf2 = mf.to_ghf().set(max_cycle=10)
mf2.kernel(dm0 = mf2.make_rdm1())
stability.ghf_real(mf2, with_symmetry=True, tol=1e-6)
stability.ghf_real(mf2, with_symmetry=False, tol=1e-6)
stability_slow.ghf_real_internal(mf2)

exit()


mol = gto.M(atom=''' 
O1
O2  1  1.2227
O3  1  1.2227  2  114.0451''', 
            basis='6-31gs',
            spin=2, verbose=5).build()
mf = scf.UHF(mol).run()
mf = stable_opt_internal(mf)
mf.stability(external=True, tol=1e-6, verbose=9)

mf2 = mf.to_ghf()
mf2.kernel(dm0 = mf2.make_rdm1())
#mf2.stability(tol=1e-6, verbose=9)
stability.ghf_real(mf2, with_symmetry=True, nroots=6, tol=1e-6, verbose=9)
stability.ghf_real(mf2, with_symmetry=False, nroots=6, tol=1e-6, verbose=9)
stability_slow.ghf_real_internal(mf2, verbose=9)

mf2.verbose=4
mf2.kernel(dm0 = mf2.make_rdm1()+0.01j)
mf2.verbose=9
mf2.stability()

