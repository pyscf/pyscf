#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

'''
density fitting MP2,  3-center integrals incore.
'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.mp import dfmp2
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_dfmp2_slow_with_t2', True)


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):

    log = logger.new_logger(mp, verbose)

    if eris is None: eris = mp.ao2mo()

    nocc, nvir = eris.nocc, eris.nvir
    occ_energy, vir_energy = mp.split_mo_energy()[1:3]
    moeoo = np.asarray(occ_energy[:,None] + occ_energy, order='C')
    moevv = np.asarray(vir_energy[:,None] + vir_energy, order='C')

    mem_avail = mp.max_memory - lib.current_memory()[0]

    if with_t2:
        t2 = np.zeros((nocc,nocc,nvir,nvir), dtype=eris.dtype)
        mem_avail -= t2.size * eris.dsize / 1e6
    else:
        t2 = None

    if mem_avail < 0:
        log.error('Insufficient memory for holding t2 incore. Please rerun with `with_t2 = False`.')
        raise MemoryError

    emp2_ss = emp2_os = 0

    cput1 = (logger.process_clock(), logger.perf_counter())

    emp2_ss = emp2_os = 0
    for i in range(nocc):
        ivL = eris.get_occ_blk(i,i+1)[0]
        for j in range(i+1):
            fac = 1 if i == j else 2

            if j == i:
                jvL = ivL
            else:
                jvL = eris.get_occ_blk(j,j+1)[0]

            vab = lib.dot(ivL, jvL.T)
            tab = np.conj(vab) / (moeoo[i,j] - moevv)
            ed =  lib.einsum('ab,ab->', vab, tab) * fac
            ex = -lib.einsum('ab,ba->', vab, tab) * fac
            emp2_ss += ed + ex
            emp2_os += ed

            if with_t2:
                t2[i,j] = tab
                if i != j:
                    t2[j,i] = tab.T.conj()

        cput1 = log.timer_debug1('i-block [%d:%d]/%d' % (i,i+1,nocc), *cput1)

    emp2_ss = emp2_ss.real
    emp2_os = emp2_os.real
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2


class DFMP2(dfmp2.DFMP2):

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

MP2 = DFMP2

del (WITH_T2)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto, df
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.RHF(mol).run()
    pt = DFMP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204004830285)

    pt.with_df = df.DF(mol)
    pt.with_df.auxbasis = 'weigend'
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204254500453)

    mf = scf.density_fit(scf.RHF(mol), 'weigend')
    mf.kernel()
    pt = DFMP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.203986171133)

    pt.with_df = df.DF(mol)
    pt.with_df.auxbasis = df.make_auxbasis(mol, mp2fit=True)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.203738031827)

    pt.frozen = 2
    emp2, t2 = pt.kernel()
    print(emp2 - -0.14433975122418313)
