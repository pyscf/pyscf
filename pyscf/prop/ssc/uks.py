#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic UKS spin-spin coupling (SSC) constants
(In testing)
'''

from pyscf.lib import logger
from pyscf.prop.ssc import uhf as uhf_ssc


class SpinSpinCoupling(uhf_ssc.SpinSpinCoupling):
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s for %s (In testing) ********',
                 self.__class__, self._scf.__class__)
        log.info('nuc_pair %s', self.nuc_pair)
        log.info('with Fermi-contact  %s', self.with_fc)
        log.info('with Fermi-contact + spin-dipole  %s', self.with_fcsd)
        if self.cphf:
            log.info('Solving MO10 eq with CPHF.')
            log.info('CPHF conv_tol = %g', self.conv_tol)
            log.info('CPHF max_cycle_cphf = %d', self.max_cycle_cphf)
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        return self

    def kernel(self, mo1=None):
        assert(uhf_ssc.ZZ_ONLY)
        return uhf_ssc.SpinSpinCoupling.kernel(self, mo1)

SSC = SpinSpinCoupling

from pyscf import lib
from pyscf import dft
dft.uks.UKS.SSC = dft.uks.UKS.SpinSpinCoupling = \
dft.uks_symm.UKS.SSC = dft.uks_symm.UKS.SpinSpinCoupling = \
        lib.class_as_method(SSC)

if __name__ == '__main__':
    from pyscf import lib, gto, dft

    mol = gto.M(atom='''
                O 0 0      0
                H 0 -0.757 0.587
                H 0  0.757 0.587''',
                basis='6-31g')

    mf1 = dft.UKS(mol).set(xc='b3lyp').run()
    ssc = mf1.SSC()
    ssc.with_fc = True
    ssc.with_fcsd = True
    jj = ssc.kernel()

    from pyscf.prop.ssc import rks
    mol = gto.M(atom='''
                O 0 0      0
                H 0 -0.757 0.587
                H 0  0.757 0.587''',
                basis='6-31g')

    mf = dft.RKS(mol).set(xc='b3lyp').run()
    ssc = rks.SSC(mf)
    ssc.with_fc = True
    ssc.with_fcsd = True
    jj = ssc.kernel()
