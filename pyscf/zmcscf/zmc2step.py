#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import copy
import time
from functools import reduce
import numpy
import pyscf.lib.logger as logger
from pyscf.mcscf import mc1step
from pyscf.mcscf import casci
from pyscf import __config__
from pyscf.mcscf.casci import get_fock, cas_natorb, canonicalize
from pyscf.mcscf import mc_ao2mo
from pyscf.mcscf import chkfile

def kernel(casscf, mo_coeff, tol=1e-7, conv_tol_grad=None,
           ci0=None, callback=None, verbose=None, dump_chk=True):
    if verbose is None:
        verbose = casscf.verbose
    if callback is None:
        callback = casscf.callback

    log = logger.Logger(casscf.stdout, verbose)
    cput0 = (time.clock(), time.time())
    log.debug('Start 2-step ZCASSCF')

    mo = mo_coeff
    nmo = mo.shape[1]
    ncore = casscf.ncore
    ncas = casscf.ncas
    nocc = ncore + ncas
    import pdb
    pdb.set_trace()
    eris = casscf.ao2mo(mo)
    e_tot, e_cas, fcivec = casscf.casci(mo, ci0, eris, log, locals())

    if ncas == nmo and not casscf.internal_rotation:
        if casscf.canonicalization:
            log.debug('CASSCF canonicalization')
            mo, fcivec, mo_energy = casscf.canonicalize(mo, fcivec, eris,
                                                        casscf.sorting_mo_energy,
                                                        casscf.natorb, verbose=log)
        else:
            mo_energy = None
        return True, e_tot, e_cas, fcivec, mo, mo_energy

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(tol)
        logger.info(casscf, 'Set conv_tol_grad to %g', conv_tol_grad)
    conv_tol_ddm = conv_tol_grad * 3
    conv = False
    de, elast = e_tot, e_tot
    totmicro = totinner = 0
    casdm1 = 0
    r0 = None

    t2m = t1m = log.timer('Initializing 2-step CASSCF', *cput0)
    imacro = 0
    while not conv and imacro < casscf.max_cycle_macro:
        imacro += 1
        njk = 0
        t3m = t2m
        casdm1_old = casdm1
        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, ncas, casscf.nelecas)
        norm_ddm = numpy.linalg.norm(casdm1 - casdm1_old)
        t3m = log.timer('update CAS DM', *t3m)
        max_cycle_micro = 1 # casscf.micro_cycle_scheduler(locals())

        casscf.optimizeOrbs(mo, lambda:casdm1, lambda:casdm2, eris, r0, conv_tol_grad*0.3, log)
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
            de = numpy.dot(casscf.pack_uniq_var(u), g_orb)
            t3m = log.timer('orbital rotation', *t3m)

            eris = None
            u = u.copy()
            g_orb = g_orb.copy()
            mo = casscf.rotate_mo(mo, u, log)
            eris = casscf.ao2mo(mo)
            t3m = log.timer('update eri', *t3m)

            log.debug('micro %d  ~dE=%5.3g  |u-1|=%5.3g  |g[o]|=%5.3g  |dm1|=%5.3g',
                      imicro, de, norm_t, norm_gorb, norm_ddm)

            if callable(callback):
                callback(locals())

            t2m = log.timer('micro iter %d'%imicro, *t2m)
            if norm_t < 1e-4 or abs(de) < tol*.4 or norm_gorb < conv_tol_grad*.2:
                break

        totinner += njk
        totmicro += imicro + 1

        e_tot, e_cas, fcivec = casscf.casci(mo, fcivec, eris, log, locals())
        log.timer('CASCI solver', *t3m)
        t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

        de, elast = e_tot - elast, e_tot
        if (abs(de) < tol and
            norm_gorb < conv_tol_grad and norm_ddm < conv_tol_ddm):
            conv = True
        else:
            elast = e_tot

        if dump_chk:
            casscf.dump_chk(locals())

        if callable(callback):
            callback(locals())

    if conv:
        log.info('2-step CASSCF converged in %d macro (%d JK %d micro) steps',
                 imacro, totinner, totmicro)
    else:
        log.info('2-step CASSCF not converged, %d macro (%d JK %d micro) steps',
                 imacro, totinner, totmicro)

    if casscf.canonicalization:
        log.info('CASSCF canonicalization')
        mo, fcivec, mo_energy = \
                casscf.canonicalize(mo, fcivec, eris, casscf.sorting_mo_energy,
                                    casscf.natorb, casdm1, log)
        if casscf.natorb and dump_chk: # dump_chk may save casdm1
            occ, ucas = casscf._eig(-casdm1, ncore, nocc)
            casdm1 = numpy.diag(-occ)

    if dump_chk:
        casscf.dump_chk(locals())

    log.timer('2-step CASSCF', *cput0)
    return conv, e_tot, e_cas, fcivec, mo, mo_energy


class ZCASSCF(casci.CASCI):
    __doc__ = casci.CASCI.__doc__ + '''CASSCF

    Extra attributes for CASSCF:

        conv_tol : float
            Converge threshold.  Default is 1e-7
        conv_tol_grad : float
            Converge threshold for CI gradients and orbital rotation gradients.
            Default is 1e-4
        max_cycle_macro : int
            Max number of macro iterations.  Default is 50.

        bb_conv_tol : float, for Barzelai-Browein (BB) solver.
            converge threshold for BB solver.  Default is 1e-6.
        bb_max_cycle : float, for BB solver.
            Max number of iterations allowd in BB solver.  Default is 1000.
        bb_gauss_noise : float, for BB solver. Default is 0.1
            Add some random noise before starting a new BB iteration

        internal_rotation: bool.
            if the CI solver is not FCI then active-active rotations are not redundant.
            Default(True)
        chkfile : str
            Checkpoint file to save the intermediate orbitals during the CASSCF optimization.
            Default is the checkpoint file of mean field object.
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            envrionment.

    Saved results

        e_tot : float
            Total MCSCF energy (electronic energy plus nuclear repulsion)
        e_cas : float
            CAS space FCI energy
        ci : ndarray
            CAS space FCI coefficients
        mo_coeff : ndarray (MxM, but the number of active variables is BB are just first N(=ncore+nact) columns)
            Optimized CASSCF orbitals coefficients. When canonicalization is
            specified, the returned orbitals make the general Fock matrix
            (Fock operator on top of MCSCF 1-particle density matrix)
            diagonalized within each subspace (core, active, external).
            If natorb (natural orbitals in active space) is specified,
            the active segment of the mo_coeff is natural orbitls.
        mo_energy : ndarray
            Diagonal elements of general Fock matrix (in mo_coeff
            representation).

    Examples:
    ********CHANGE THIS EXAMPLE***********
    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = mcscf.CASSCF(mf, 6, 6)
    >>> mc.kernel()[0]
    -109.044401882238134
    '''

# the max orbital rotation and CI increment, prefer small step size
    max_cycle_macro = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_max_cycle_macro', 50)
    conv_tol = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_conv_tol', 1e-7)
    conv_tol_grad = getattr(__config__, 'mcscf_zmc2step_ZCASSCF_conv_tol_grad', None)

    # for BB Solver
    bb_conv_tol = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_bb_conv_tol', 1e-6)
    bb_max_cycle = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_bb_max_cycle', 1000)
    bb_gauss_noise = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_bb_gauss_noise', 0.1)

    internal_rotation = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_internal_rotation', True)

    ao2mo_level = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_ao2mo_level', 2)
    natorb = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_natorb', False)
    canonicalization = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_canonicalization', True)
    sorting_mo_energy = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_sorting_mo_energy', False)

    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, frozen=None):
        casci.CASCI.__init__(self, mf_or_mol, ncas, nelecas, ncore)
        self.frozen = frozen

        self.callback = None
        self.chkfile = self._scf.chkfile

        self.fcisolver.max_cycle = getattr(__config__,
                                           'zmcscf_zmc2step_ZCASSCF_fcisolver_max_cycle', 50)
        self.fcisolver.conv_tol = getattr(__config__,
                                          'zmcscf_zmc2step_ZCASSCF_fcisolver_conv_tol', 1e-8)

##################################################
# don't modify the following attributes, they are not input options
        self.e_tot = None
        self.e_cas = None
        self.ci = None
        self.mo_coeff = self._scf.mo_coeff
        self.mo_energy = self._scf.mo_energy
        self.converged = False
        self._max_stepsize = None

        keys = set(('max_cycle_macro',
                    'conv_tol', 'conv_tol_grad',
                    'bb_conv_tol', 'bb_max_cycle', 
                    'internal_rotation',
                     'fcisolver_max_cycle',
                    'fcisolver_conv_tol', 'natorb', 'canonicalization',
                    'sorting_mo_energy', 'scale_restoration'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        ncore = self.ncore
        ncas = self.ncas
        nvir = self.mo_coeff.shape[1] - ncore - ncas
        log.info('CAS (%de+%de, %do), ncore = %d, nvir = %d', \
                 self.nelecas[0], self.nelecas[1], ncas, ncore, nvir)
        assert(nvir >= 0 and ncore >= 0 and ncas >= 0)
        if self.frozen is not None:
            log.info('frozen orbitals %s', str(self.frozen))
        log.info('max_cycle_macro = %d', self.max_cycle_macro)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_grad = %s', self.conv_tol_grad)
        log.info('Barzilai-Borwein bb_max_cycle = %d', self.bb_max_cycle)
        log.info('Barzilai-Borwein bb_conv_tol = %g', self.bb_conv_tol)
        log.info('Barzilai-Borwein bb_gauss_noise = %g', self.bb_gauss_noise)
        log.info('natorb = %s', self.natorb)
        log.info('canonicalization = %s', self.canonicalization)
        log.info('sorting_mo_energy = %s', self.sorting_mo_energy)
        log.info('ao2mo_level = %d', self.ao2mo_level)
        log.info('chkfile = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        log.info('internal_rotation = %s', self.internal_rotation)
        if getattr(self.fcisolver, 'dump_flags', None):
            self.fcisolver.dump_flags(self.verbose)
        if self.mo_coeff is None:
            log.error('Orbitals for CASCI are not specified. The relevant SCF '
                      'object may not be initialized.')

        if (getattr(self._scf, 'with_solvent', None) and
            not getattr(self, 'with_solvent', None)):
            log.warn('''Solvent model %s was found at SCF level but not applied to the CASSCF object.
The SCF solvent model will not be applied to the current CASSCF calculation.
To enable the solvent model for CASSCF, the following code needs to be called
        from pyscf import solvent
        mc = mcscf.CASSCF(...)
        mc = solvent.ddCOSMO(mc)
''',
                     self._scf.with_solvent.__class__)
        return self

    def kernel(self, mo_coeff=None, ci0=None, callback=None, _kern=kernel):
        '''
        Returns:
            Five elements, they are
            total energy,
            active space CI energy,
            the active space FCI wavefunction coefficients or DMRG wavefunction ID,
            the MCSCF canonical orbital coefficients,
            the MCSCF canonical orbital coefficients.

        They are attributes of mcscf object, which can be accessed by
        .e_tot, .e_cas, .ci, .mo_coeff, .mo_energy
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else: # overwrite self.mo_coeff because it is needed in many methods of this class
            self.mo_coeff = mo_coeff
        if callback is None: callback = self.callback

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.converged, self.e_tot, self.e_cas, self.ci, \
                self.mo_coeff, self.mo_energy = \
                _kern(self, mo_coeff,
                      tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
                      ci0=ci0, callback=callback, verbose=self.verbose)
        logger.note(self, 'CASSCF energy = %.15g', self.e_tot)
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def mc1step(self, mo_coeff=None, ci0=None, callback=None):
        return self.kernel(mo_coeff, ci0, callback)

    def mc2step(self, mo_coeff=None, ci0=None, callback=None):
        from pyscf.mcscf import mc2step
        return self.kernel(mo_coeff, ci0, callback, mc2step.kernel)


    def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        log = logger.new_logger(self, verbose)
        if eris is None:
            fcasci = copy.copy(self)
            fcasci.ao2mo = self.get_h2cas
        else:
            fcasci = _fake_h_for_fast_casci(self, mo_coeff, eris)

        e_tot, e_cas, fcivec = casci.kernel(fcasci, mo_coeff, ci0, log)
        if not isinstance(e_cas, (float, numpy.number)):
            raise RuntimeError('Multiple roots are detected in fcisolver.  '
                               'CASSCF does not know which state to optimize.\n'
                               'See also  mcscf.state_average  or  mcscf.state_specific  for excited states.')
        elif numpy.ndim(e_cas) != 0:
            # This is a workaround for external CI solver compatibility.
            e_cas = e_cas[0]

        if envs is not None and log.verbose >= logger.INFO:
            log.debug('CAS space CI energy = %.15g', e_cas)

            if getattr(self.fcisolver, 'spin_square', None):
                ss = self.fcisolver.spin_square(fcivec, self.ncas, self.nelecas)
            else:
                ss = None

            if 'imicro' in envs:  # Within CASSCF iteration
                if ss is None:
                    log.info('macro iter %d (%d JK  %d micro), '
                             'CASSCF E = %.15g  dE = %.8g',
                             envs['imacro'], envs['njk'], envs['imicro'],
                             e_tot, e_tot-envs['elast'])
                else:
                    log.info('macro iter %d (%d JK  %d micro), '
                             'CASSCF E = %.15g  dE = %.8g  S^2 = %.7f',
                             envs['imacro'], envs['njk'], envs['imicro'],
                             e_tot, e_tot-envs['elast'], ss[0])
                if 'norm_gci' in envs:
                    log.info('               |grad[o]|=%5.3g  '
                             '|grad[c]|= %s  |ddm|=%5.3g',
                             envs['norm_gorb0'],
                             envs['norm_gci'], envs['norm_ddm'])
                else:
                    log.info('               |grad[o]|=%5.3g  |ddm|=%5.3g',
                             envs['norm_gorb0'], envs['norm_ddm'])
            else:  # Initialization step
                if ss is None:
                    log.info('CASCI E = %.15g', e_tot)
                else:
                    log.info('CASCI E = %.15g  S^2 = %.7f', e_tot, ss[0])
        return e_tot, e_cas, fcivec

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
#        nmo = mo.shape[1]
#        ncore = self.ncore
#        ncas = self.ncas
#        nocc = ncore + ncas
#        eri = pyscf.ao2mo.incore.full(self._scf._eri, mo)
#        eri = pyscf.ao2mo.restore(1, eri, nmo)
#        eris = lambda:None
#        eris.j_cp = numpy.einsum('iipp->ip', eri[:ncore,:ncore,:,:])
#        eris.k_cp = numpy.einsum('ippi->ip', eri[:ncore,:,:,:ncore])
#        eris.vhf_c =(numpy.einsum('iipq->pq', eri[:ncore,:ncore,:,:])*2
#                    -numpy.einsum('ipqi->pq', eri[:ncore,:,:,:ncore]))
#        eris.ppaa = numpy.asarray(eri[:,:,ncore:nocc,ncore:nocc], order='C')
#        eris.papa = numpy.asarray(eri[:,ncore:nocc,:,ncore:nocc], order='C')
#        return eris

        return mc_ao2mo._ERIS(self, mo_coeff, method='incore',
                              level=self.ao2mo_level)

    def getGrad(self, mo, casdm1, casdm2, eris):
        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact

        mocopy = 1j*mo
        grad = 0.*mo[:,:nocc]

        import pdb
        pdb.set_trace()
        hcore = self.get_hcore()

        #Ec = numpy.einsum('ij,ji', 
        #contribution from the core energy
        grad[:,:ncore] = numpy.dot(numpy.transpose(hcore), numpy.conjugate(mo[:,:ncore])) 
                         
        
    def optimizeOrbs(self, mo, casdm1, casdm2, eris, r0, conv_tol, log):
        #the part of mo that is relevant 
        
        grad = self.getGrad(mo, casdm1, casdm2, eris)

# to avoid calculating AO integrals
def _fake_h_for_fast_casci(casscf, mo, eris):
    mc = copy.copy(casscf)
    mc.mo_coeff = mo
    ncore = casscf.ncore
    nocc = ncore + casscf.ncas

    mo_core = mo[:,:ncore]
    mo_cas = mo[:,ncore:nocc]
    core_dm = numpy.dot(mo_core, mo_core.T) * 2
    hcore = casscf.get_hcore()
    energy_core = casscf.energy_nuc()
    energy_core += numpy.einsum('ij,ji', core_dm, hcore)
    energy_core += eris.vhf_c[:ncore,:ncore].trace()
    h1eff = reduce(numpy.dot, (mo_cas.T, hcore, mo_cas))
    h1eff += eris.vhf_c[ncore:nocc,ncore:nocc]
    mc.get_h1eff = lambda *args: (h1eff, energy_core)

    eri_cas = eris.ppaa[ncore:nocc,ncore:nocc,:,:].copy()
    mc.get_h2eff = lambda *args: eri_cas
    return mc


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

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
    mol.build()

    m = scf.GHF(mol)
    ehf = m.scf()
    emc = kernel(ZCASSCF(m, 4, 4), m.mo_coeff, verbose=4)[1]
    print(ehf, emc, emc-ehf)
    print(emc - -3.22013929407)
    exit(0)

    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = mc1step.CASSCF(m, 6, 4)
    mc.verbose = 4
    mo = m.mo_coeff.copy()
    mo[:,2:5] = m.mo_coeff[:,[4,2,3]]
    emc = mc.mc2step(mo)[0]
    print(ehf, emc, emc-ehf)
    #-76.0267656731 -76.0873922924 -0.0606266193028
    print(emc - -76.0873923174, emc - -76.0926176464)

