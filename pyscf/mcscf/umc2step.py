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
#

'''
UCASSCF (CASSCF without spin-degeneracy between alpha and beta orbitals)
2-step optimization algorithm
'''


import numpy
import copy
import pyscf.lib.logger as logger

def kernel(casscf, mo_coeff, tol=1e-7, conv_tol_grad=None,
           ci0=None, callback=None, verbose=None, dump_chk=True):
    if verbose is None:
        verbose = casscf.verbose
    log = logger.Logger(casscf.stdout, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start 2-step CASSCF')

    mo = mo_coeff
    nmo = mo[0].shape[1]
    ncore = casscf.ncore
    eris = casscf.ao2mo(mo)
    e_tot, e_cas, fcivec = casscf.casci(mo, ci0, eris, log, locals())
    if casscf.ncas == nmo and not casscf.internal_rotation:
        return True, e_tot, e_cas, fcivec, mo

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(tol)
        logger.info(casscf, 'Set conv_tol_grad to %g', conv_tol_grad)
    conv_tol_ddm = conv_tol_grad * 3
    conv = False
    de, elast = e_tot, e_tot
    totmicro = totinner = 0
    casdm1 = (0,0)
    r0 = None

    t2m = t1m = log.timer('Initializing 2-step CASSCF', *cput0)
    imacro = 0
    while not conv and imacro < casscf.max_cycle_macro:
        imacro += 1
        njk = 0
        t3m = t2m
        casdm1_old = casdm1
        casdm1, casdm2 = casscf.fcisolver.make_rdm12s(fcivec, casscf.ncas, casscf.nelecas)
        norm_ddm =(numpy.linalg.norm(casdm1[0] - casdm1_old[0]) +
                   numpy.linalg.norm(casdm1[1] - casdm1_old[1]))
        t3m = log.timer('update CAS DM', *t3m)
        max_cycle_micro = 1 # casscf.micro_cycle_scheduler(locals())
        max_stepsize = casscf.max_stepsize_scheduler(locals())
        for imicro in range(max_cycle_micro):
            rota = casscf.rotate_orb_cc(mo, lambda:fcivec, lambda:casdm1, lambda:casdm2,
                                        eris, r0, conv_tol_grad*.3, max_stepsize, log)
            u, g_orb, njk1, r0 = next(rota)
            rota.close()
            njk += njk1
            norm_t = numpy.linalg.norm(u-numpy.eye(nmo))
            norm_gorb = numpy.linalg.norm(g_orb)
            if imicro == 0:
                norm_gorb0 = norm_gorb
            t3m = log.timer('orbital rotation', *t3m)

            eris = None
            u = copy.copy(u)
            g_orb = copy.copy(g_orb)
            mo = casscf.rotate_mo(mo, u, log)
            eris = casscf.ao2mo(mo)
            t3m = log.timer('update eri', *t3m)

            log.debug('micro %d  |u-1|=%5.3g  |g[o]|=%5.3g  |dm1|=%5.3g',
                      imicro, norm_t, norm_gorb, norm_ddm)

            if callable(callback):
                callback(locals())

            t2m = log.timer('micro iter %d'%imicro, *t2m)
            if norm_t < 1e-4 or norm_gorb < conv_tol_grad*.5:
                break

        totinner += njk
        totmicro += imicro+1

        e_tot, e_cas, fcivec = casscf.casci(mo, fcivec, eris, log, locals())
        log.timer('CASCI solver', *t3m)
        t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

        de, elast = e_tot - elast, e_tot
        if (abs(de) < tol and
            norm_gorb < conv_tol_grad and norm_ddm < conv_tol_ddm):
            conv = True

        if dump_chk:
            casscf.dump_chk(locals())

        if callable(callback):
            callback(locals())

    if conv:
        log.info('2-step CASSCF converged in %d macro (%d JK %d micro) steps',
                 imacro+1, totinner, totmicro)
    else:
        log.info('2-step CASSCF not converged, %d macro (%d JK %d micro) steps',
                 imacro+1, totinner, totmicro)
    log.timer('2-step CASSCF', *cput0)
    return conv, e_tot, e_cas, fcivec, mo


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.mcscf import addons
    from pyscf.mcscf import umc1step

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]

    mol.basis = {'H': 'sto-3g',
                 'O': '6-31g',}
    mol.charge = 1
    mol.spin = 1
    mol.build()

    m = scf.UHF(mol)
    ehf = m.scf()
    emc = kernel(umc1step.CASSCF(m, 4, (2,1)), m.mo_coeff, verbose=4)[1]
    print(ehf, emc, emc-ehf)
    print(emc - -2.9782774463926618)


    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.charge = 1
    mol.spin = 1
    mol.build()

    m = scf.UHF(mol)
    ehf = m.scf()
    mc = umc1step.CASSCF(m, 4, (2,1))
    mc.verbose = 4
    mo = addons.sort_mo(mc, m.mo_coeff, (3,4,6,7), 1)
    emc = mc.mc2step(mo)[0]
    print(ehf, emc, emc-ehf)
    print(emc - -75.5644202701263, emc - -75.573930418500652,
          emc - -75.574137883405612, emc - -75.648547447838951)

    mc = umc1step.CASSCF(m, 4, (2,1))
    mc.verbose = 4
    emc = mc.mc2step()[0]
    print(ehf, emc, emc-ehf)
    print(emc - -75.5644202701263, emc - -75.573930418500652,
          emc - -75.574137883405612, emc - -75.648547447838951)


