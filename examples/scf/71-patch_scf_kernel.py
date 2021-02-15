#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A patch to the SCF scheme to apply various convergence schemes if the default
DIIS SCF does not converge.
'''

from pyscf import lib, scf

scf_kernel = scf.hf.SCF.kernel
def kernel(self, dm0=None, **kwargs):
    scf_kernel(self, dm0, **kwargs)

    if not self.converged and not self.level_shift:
        with lib.temporary_env(self, level_shift=.2):
            lib.logger.note(self, 'DIIS does not converge. Try level shift')
            scf_kernel(self, self.make_rdm1(), **kwargs)

    if not self.converged:
        lib.logger.note(self, 'DIIS does not converge. Try SOSCF')
        mf1 = self.newton().run()

        # Note: delete mf1._scf because it leads to circular reference to self.
        del mf1._scf
        self.__dict__.update(mf1.__dict__)

    return self.e_tot

scf.hf.SCF.kernel = kernel

# Using the patched SCF kernel
import pyscf
pyscf.M(atom='Ne', basis='ccpvdz', verbose=3).RKS().density_fit().run(max_cycle=3)

