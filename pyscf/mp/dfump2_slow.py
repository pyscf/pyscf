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
from pyscf.mp import dfump2
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_dfump2_with_t2', True)


def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):

    log = logger.new_logger(mp, verbose)

    if eris is None: eris = mp.ao2mo()

    nocc, nvir = eris.nocc, eris.nvir
    split_mo_energy = mp.split_mo_energy()
    occ_energy = [x[1] for x in split_mo_energy]
    vir_energy = [x[2] for x in split_mo_energy]

    mem_avail = mp.max_memory - lib.current_memory()[0]

    if with_t2:
        t2 = (np.zeros((nocc[0],nocc[0],nvir[0],nvir[0]), dtype=eris.dtype),
              np.zeros((nocc[0],nocc[1],nvir[0],nvir[1]), dtype=eris.dtype),
              np.zeros((nocc[1],nocc[1],nvir[1],nvir[1]), dtype=eris.dtype))
        mem_avail -= sum([x.size for x in t2]) * eris.dsize / 1e6
    else:
        t2 = None

    if mem_avail < 0:
        log.error('Insufficient memory for holding t2 incore. Please rerun with `with_t2 = False`.')
        raise MemoryError

    cput1 = (logger.process_clock(), logger.perf_counter())

    emp2_ss = emp2_os = 0
    # same spin
    for s in [0,1]:
        s_t2 = 0 if s == 0 else 2
        moeoo = occ_energy[s][:,None] + occ_energy[s]
        moevv = lib.asarray(vir_energy[s][:,None] + vir_energy[s], order='C')
        for i in range(nocc[s]):
            ivL = eris.get_occ_blk(s,i,i+1)[0]
            for j in range(i+1):
                fac = 0.5 if i == j else 1

                if j == i:
                    jvL = ivL
                else:
                    jvL = eris.get_occ_blk(s,j,j+1)[0]

                vab = lib.dot(ivL, jvL.T)
                tab = np.conj(vab) / (moeoo[i,j] - moevv)
                ed =  lib.einsum('ab,ab->', vab, tab) * fac
                ex = -lib.einsum('ab,ba->', vab, tab) * fac
                emp2_ss += ed + ex

                if with_t2:
                    tab -= tab.T.conj()
                    t2[s_t2][i,j] = tab
                    if i != j:
                        t2[s_t2][j,i] = tab.T.conj()

            cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (s,s,i,i+1,nocc[s]),
                                     *cput1)

    # opposite spin
    sa, sb = 0, 1
    moeoo = occ_energy[sa][:,None] + occ_energy[sb]
    moevv = lib.asarray(vir_energy[sa][:,None] + vir_energy[sb], order='C')
    for i in range(nocc[sa]):
        ivL = eris.get_occ_blk(sa,i,i+1)[0]
        for j in range(nocc[sb]):
            jvL = eris.get_occ_blk(sb,j,j+1)[0]

            vab = lib.dot(ivL, jvL.T)
            tab = np.conj(vab) / (moeoo[i,j] - moevv)
            ed =  lib.einsum('ab,ab->', vab, tab)
            emp2_os += ed

            if with_t2:
                t2[1][i,j] = tab

        cput1 = log.timer_debug1('(sa,sb) = (%d,%d)  i-block [%d:%d]/%d' % (sa,sb,i,i+1,nocc[sa]),
                                 *cput1)

    emp2_ss = emp2_ss.real
    emp2_os = emp2_os.real
    emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

    return emp2, t2


class DFUMP2(dfump2.DFUMP2):

    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

UMP2 = DFUMP2

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
    mol.charge = 1
    mol.spin = 1
    mol.build()
    mf = scf.UHF(mol).run()
    pt = DFUMP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.15321910988237386)

    pt.with_df = df.DF(mol)
    pt.with_df.auxbasis = 'weigend'
    emp2, t2 = pt.kernel()
    print(emp2 - -0.15345631939967935)

    mf = scf.density_fit(scf.UHF(mol), 'weigend')
    mf.kernel()
    pt = DFUMP2(mf)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.15325911974166384)

    pt.with_df = df.DF(mol)
    pt.with_df.auxbasis = df.make_auxbasis(mol, mp2fit=True)
    emp2, t2 = pt.kernel()
    print(emp2 - -0.15302393704336487)

    pt.frozen = 2
    emp2, t2 = pt.kernel()
    print(emp2 - -0.09563606544882836)
