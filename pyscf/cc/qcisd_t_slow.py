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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc.ccsd_t_slow import r3

'''
QCISD(T)
'''

# t3 as ijkabc

# JCP 94, 442 (1991); DOI:10.1063/1.460359.  Error in Eq (1), should be [ia] >= [jb] >= [kc]
def kernel(mycc, eris, t1=None, t2=None, verbose=logger.NOTE):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mycc.stdout, verbose)

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2

    t1T = t1.T
    t2T = t2.transpose(2,3,0,1)

    nocc, nvir = t1.shape
    mo_e = eris.fock.diagonal().real
    e_occ, e_vir = mo_e[:nocc], mo_e[nocc:]
    eijk = lib.direct_sum('i,j,k->ijk', e_occ, e_occ, e_occ)

    eris_vvov = eris.get_ovvv().conj().transpose(1,3,0,2)
    eris_vooo = numpy.asarray(eris.ovoo).conj().transpose(1,0,2,3)
    eris_vvoo = numpy.asarray(eris.ovov).conj().transpose(1,3,0,2)
    fvo = eris.fock[nocc:,:nocc]
    def get_w(a, b, c):
        w = numpy.einsum('if,fkj->ijk', eris_vvov[a,b], t2T[c,:])
        w-= numpy.einsum('ijm,mk->ijk', eris_vooo[a,:], t2T[b,c])
        return w
    def get_v(a, b, c):
        v = numpy.einsum('ij,k->ijk', eris_vvoo[a,b], t1T[c])
        v+= numpy.einsum('ij,k->ijk', t2T[a,b], fvo[c])
        return v

    et = 0
    for a in range(nvir):
        for b in range(a+1):
            for c in range(b+1):
                d3 = eijk - e_vir[a] - e_vir[b] - e_vir[c]
                if a == c:  # a == b == c
                    d3 *= 6
                elif a == b or b == c:
                    d3 *= 2

                wabc = get_w(a, b, c)
                wacb = get_w(a, c, b)
                wbac = get_w(b, a, c)
                wbca = get_w(b, c, a)
                wcab = get_w(c, a, b)
                wcba = get_w(c, b, a)
                vabc = get_v(a, b, c)
                vacb = get_v(a, c, b)
                vbac = get_v(b, a, c)
                vbca = get_v(b, c, a)
                vcab = get_v(c, a, b)
                vcba = get_v(c, b, a)
                zabc = r3(wabc + vabc) / d3
                zacb = r3(wacb + vacb) / d3
                zbac = r3(wbac + vbac) / d3
                zbca = r3(wbca + vbca) / d3
                zcab = r3(wcab + vcab) / d3
                zcba = r3(wcba + vcba) / d3

                et+= numpy.einsum('ijk,ijk', wabc, zabc.conj())
                et+= numpy.einsum('ikj,ijk', wacb, zabc.conj())
                et+= numpy.einsum('jik,ijk', wbac, zabc.conj())
                et+= numpy.einsum('jki,ijk', wbca, zabc.conj())
                et+= numpy.einsum('kij,ijk', wcab, zabc.conj())
                et+= numpy.einsum('kji,ijk', wcba, zabc.conj())

                et+= numpy.einsum('ijk,ijk', wacb, zacb.conj())
                et+= numpy.einsum('ikj,ijk', wabc, zacb.conj())
                et+= numpy.einsum('jik,ijk', wcab, zacb.conj())
                et+= numpy.einsum('jki,ijk', wcba, zacb.conj())
                et+= numpy.einsum('kij,ijk', wbac, zacb.conj())
                et+= numpy.einsum('kji,ijk', wbca, zacb.conj())

                et+= numpy.einsum('ijk,ijk', wbac, zbac.conj())
                et+= numpy.einsum('ikj,ijk', wbca, zbac.conj())
                et+= numpy.einsum('jik,ijk', wabc, zbac.conj())
                et+= numpy.einsum('jki,ijk', wacb, zbac.conj())
                et+= numpy.einsum('kij,ijk', wcba, zbac.conj())
                et+= numpy.einsum('kji,ijk', wcab, zbac.conj())

                et+= numpy.einsum('ijk,ijk', wbca, zbca.conj())
                et+= numpy.einsum('ikj,ijk', wbac, zbca.conj())
                et+= numpy.einsum('jik,ijk', wcba, zbca.conj())
                et+= numpy.einsum('jki,ijk', wcab, zbca.conj())
                et+= numpy.einsum('kij,ijk', wabc, zbca.conj())
                et+= numpy.einsum('kji,ijk', wacb, zbca.conj())

                et+= numpy.einsum('ijk,ijk', wcab, zcab.conj())
                et+= numpy.einsum('ikj,ijk', wcba, zcab.conj())
                et+= numpy.einsum('jik,ijk', wacb, zcab.conj())
                et+= numpy.einsum('jki,ijk', wabc, zcab.conj())
                et+= numpy.einsum('kij,ijk', wbca, zcab.conj())
                et+= numpy.einsum('kji,ijk', wbac, zcab.conj())

                et+= numpy.einsum('ijk,ijk', wcba, zcba.conj())
                et+= numpy.einsum('ikj,ijk', wcab, zcba.conj())
                et+= numpy.einsum('jik,ijk', wbca, zcba.conj())
                et+= numpy.einsum('jki,ijk', wbac, zcba.conj())
                et+= numpy.einsum('kij,ijk', wacb, zcba.conj())
                et+= numpy.einsum('kji,ijk', wabc, zcba.conj())
    et *= 2
    log.info('QCISD(T) correction = %.15g', et)
    return et
