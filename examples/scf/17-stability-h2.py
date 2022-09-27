#!/usr/bin/env python
#
# Author: Shirong Wang <srwang20@fudan.edu.cn>
#

'''
SCF wavefunction stability analysis
'''

from pyscf import gto, scf
import numpy as np

np.set_printoptions(suppress=True, precision=3, linewidth=400)
from pyscf.scf import stability_slow, stability
from pyscf.soscf import newton_ah

mol = gto.M(atom='''H 0.0 0.0 0.0; H 0.0 0.0 0.7''', basis='cc-pvdz', verbose=4).build()
mf = mol.RHF().run()
stability.rhf_internal(mf,verbose=9)
stability.rhf_external(mf,verbose=9)
stability_slow.rhf_internal(mf, verbose=9)
#stability_slow.rhf_internal_ab(mf, verbose=9)
stability_slow.rhf_external(mf, verbose=9)

#exit()
mf2 = mf.to_uhf().set(max_cycle=10)
mf2.kernel(dm0 = mf2.make_rdm1())
stability.uhf_internal(mf2, verbose=9)
#stability.uhf_internal(mf2, with_symmetry=False, verbose=9)
stability.uhf_external(mf2, verbose=9)
stability_slow.uhf_internal(mf2, verbose=9)
stability_slow.uhf_external(mf2, verbose=9)

mf2 = mf.to_ghf().set(max_cycle=10)
mf2.kernel(dm0 = mf2.make_rdm1())
stability.ghf_real(mf2, with_symmetry=True, tol=1e-6, verbose=9)
stability.ghf_real(mf2, with_symmetry=False, tol=1e-6, verbose=9)
stability_slow.ghf_real_internal(mf2, verbose=9)


exit()


mol = gto.M(atom='O 0 0 0; O 0 0.8 1.0; O 0.0 0.8 -1.0', basis='def2-svp', spin=2)
mf = scf.UHF(mol).run()
mf = stable_opt_internal(mf)
mf.stability(external=True, tol=1e-4, verbose=9)

#exit()
mf2 = mf.to_ghf()
mf2.kernel(dm0 = mf2.make_rdm1())
mf2.stability(tol=1e-4, verbose=9)

stability_slow.ghf_real_internal(mf2, verbose=9)

exit()
mf2.verbose=3
mf2.kernel(dm0 = mf2.make_rdm1()+0.01j)
mf2.verbose=9
mf2.stability()

