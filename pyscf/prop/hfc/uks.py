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
Non-relativistic unrestricted Hartree-Fock hyperfine coupling tensor
(In testing)

Refs:
    JCP, 120, 2127
    JCP, 118, 3939
'''

import numpy
from pyscf import lib
from pyscf.scf import _vhf
from pyscf.prop.hfc import uhf as uhf_hfc
from pyscf.prop.gtensor.uks import get_vxc_soc


# Note the (-) sign of beta-beta block is included in the integral
def make_h1_soc2e(hfcobj, dm0):
    mf = hfcobj._scf
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    if abs(omega) > 1e-10:
        raise NotImplementedError
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    v1 = get_vxc_soc(ni, mol, mf.grids, mf.xc, dm0,
                     max_memory=max_memory, verbose=hfcobj.verbose)
    if abs(hyb) > 1e-10:
        vj, vk = uhf_hfc.get_jk(mol, dm0)
        v1 += vj[0] + vj[1]
        v1 -= vk * hyb
    else:
        vj = _vhf.direct_mapdm(mol._add_suffix('int2e_p1vxp1'),
                               'a4ij', 'lk->s2ij',
                               dm0, 3, mol._atm, mol._bas, mol._env)
        for i in range(3):
            lib.hermi_triu(vj[0,i], hermi=2, inplace=True)
            lib.hermi_triu(vj[1,i], hermi=2, inplace=True)
        v1 += vj[0] + vj[1]
    v1[1] *= -1
    return v1


class HyperfineCoupling(uhf_hfc.HyperfineCoupling):
    make_h1_soc2e = make_h1_soc2e

    def dump_flags(self, verbose=None):
        log = lib.logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s for %s (In testing) ********',
                 self.__class__, self._scf.__class__)
        log.info('HFC for atoms %s', str(self.hfc_nuc))
        if self.cphf:
            log.info('CPHF conv_tol = %g', self.conv_tol)
            log.info('CPHF max_cycle_cphf = %d', self.max_cycle_cphf)
        log.info('para_soc2e = %s', self.para_soc2e)
        log.info('so_eff_charge = %s (1e SO effective charge)',
                 self.so_eff_charge)
        if not self._scf.converged:
            log.warn('Ground state SCF is not converged')
        return self

HFC = HyperfineCoupling


if __name__ == '__main__':
    from pyscf import gto, scf, dft
    mol = gto.M(atom='C 0 0 0; O 0 0 1.12',
                basis='ccpvdz', spin=1, charge=1, verbose=3)
    mf = dft.UKS(mol).run()
    hfc = HFC(mf)
    hfc.verbose = 4
    hfc.so_eff_charge = False
    print(lib.finger(hfc.kernel()) - 255.92807696823797)

    mol = gto.M(atom='H 0 0 0; H 0 0 1.',
                basis='ccpvdz', spin=1, charge=-1, verbose=3)
    mf = scf.UKS(mol).run()
    hfc = HFC(mf)
    hfc.cphf = True
    print(lib.finger(hfc.kernel()) - -25.896662045941071)

    mol = gto.M(atom='''
                Li 0   0   1
                ''',
                basis='ccpvdz', spin=1, charge=0, verbose=3)
    mf = scf.UKS(mol).run()
    hfc = HFC(mf)
    hfc.cphf = True
    print(lib.finger(hfc.kernel()) - 65.396568554095523)

    mol = gto.M(atom='''
                H 0   0   1
                H 1.2 0   1
                H .1  1.1 0.3
                H .8  .7  .6
                ''',
                basis='ccpvdz', spin=1, charge=1, verbose=3)
    mf = scf.UKS(mol).run()
    hfc = HFC(mf)
    print(lib.finger(hfc.kernel()) - 180.05536650105842)

    nao, nmo = mf.mo_coeff[0].shape
    numpy.random.seed(1)
    dm0 = numpy.random.random((2,nao,nao))
    dm0 = dm0 + dm0.transpose(0,2,1)
    hfc.so_eff_charge = False
    h1a, h1b = make_h1_soc2e(hfc, dm0)
    print(lib.finger(h1a) - -10.684681440665429)
    print(lib.finger(h1b) - 10.23699899832944)

