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

import copy, scipy, time, numpy
from functools import reduce
import pyscf.lib.logger as logger
from pyscf.zmcscf import gzcasci, zmc_ao2mo
from pyscf.shciscf import shci
#from pyscf.mcscf import mc1step
#from pyscf.mcscf import casci
from pyscf import __config__, lib, ao2mo
#from pyscf.mcscf.casci import get_fock, cas_natorb, canonicalize
from pyscf.mcscf import mc_ao2mo
from pyscf.mcscf import chkfile
from scipy.linalg import expm as expmat 

from pyscf import mcscf
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

    eris = None
    
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
        casdm1, casdm2 = casscf.fcisolver.make_rdm12Frombin(fcivec, ncas, casscf.nelecas)
        norm_ddm = numpy.linalg.norm(casdm1 - casdm1_old)
        t3m = log.timer('update CAS DM', *t3m)

        mo, gorb, njk, norm_gorb0 = casscf.optimizeOrbs(mo, lambda:casdm1, lambda:casdm2, imacro <= 2, 
                                        eris, r0, conv_tol_grad*0.3, log)
        norm_gorb = numpy.linalg.norm(gorb)
        
        totinner += njk

        e_tot, e_cas, fcivec = casscf.casci(mo, fcivec, eris, log, locals())
        log.timer('CASCI solver', *t3m)
        t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

        de, elast = e_tot - elast, e_tot
        if (abs(de) < tol and
            norm_gorb < conv_tol_grad and norm_ddm < conv_tol_ddm):
            conv = True
        else:
            elast = e_tot

        ###FIX THIS###
        #if dump_chk:
        #casscf.dump_chk(locals())

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

class ZCASSCF(gzcasci.GZCASCI):
    __doc__ = gzcasci.GZCASCI.__doc__ + '''CASSCF

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
        gzcasci.GZCASCI.__init__(self, mf_or_mol, ncas, nelecas, ncore)
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


    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        
        return zmc_ao2mo._ERIS(self, mo_coeff, method='incore',
                              level=self.ao2mo_level)

    def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        log = logger.new_logger(self, verbose)

        fcasci = copy.copy(self)
        fcasci.ao2mo = self.get_h2cas

        e_tot, e_cas, fcivec = gzcasci.kernel(fcasci, mo_coeff, ci0, log)
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

    def dump_chk(self, envs):
        if not self.chkfile:
            return self

        ncore = self.ncore
        nocc = ncore + self.ncas
        if 'mo' in envs:
            mo_coeff = envs['mo']
        else:
            mo_coeff = envs['mo_coeff']
        mo_occ = numpy.zeros(mo_coeff.shape[1])
        mo_occ[:ncore] = 2
        if self.natorb:
            occ = self._eig(-envs['casdm1'], ncore, nocc)[0]
            mo_occ[ncore:nocc] = -occ
        else:
            mo_occ[ncore:nocc] = envs['casdm1'].diagonal().real
# Note: mo_energy in active space =/= F_{ii}  (F is general Fock)
        if 'mo_energy' in envs:
            mo_energy = envs['mo_energy']
        else:
            mo_energy = 'None'
        chkfile.dump_mcscf(self, self.chkfile, 'mcscf', envs['e_tot'],
                           mo_coeff, ncore, self.ncas, mo_occ,
                           mo_energy, envs['e_cas'], None, envs['casdm1'],
                           overwrite_mol=False)
        return self

    def update_from_chk(self, chkfile=None):
        if chkfile is None: chkfile = self.chkfile
        self.__dict__.update(lib.chkfile.load(chkfile, 'mcscf'))
        return self
    update = update_from_chk
                         
    '''
    def calcGrad(self, mo, casdm1, casdm2, ERIS=None):
        #if ERIS is None: ERIS = self.ao2mo(mo)
        hcore = self._scf.get_hcore() 
        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact

        moc = mo[:,:ncore]
        moa = mo[:,ncore:nocc]
        gradC  = 0*mo  #for z
        gradC2 = 0*mo  #for z.conj

        #ecore 
        dmcas = reduce(numpy.dot, (moa, casdm1(), moa.conj().T))
        dmcore = numpy.dot(moc, moc.conj().T)

        jc,kc = self._scf.get_jk(self.mol, dm=dmcore)        
        ja,ka = self._scf.get_jk(self.mol, dm=dmcas)        
        
        #gradC [:,:ncore] = numpy.dot( (hcore + (jc-kc)+ja-ka).T, moc.conj())  
        gradC2[:,:ncore] = numpy.dot( (hcore + (jc-kc)+ja-ka) , moc)  
        #Ecore = numpy.einsum('xy, yx', hcore+0.5*(j-k), dmcore)  ###hcore        


        ###THIS IS THE BIT WE NEED
        eri_ao_sp = mol.intor('int2e_spinor', aosym='s1')
        j1 = lib.einsum('wxyz, wp->pxyz', eri_ao_sp, moa.conj())
        jaapp = lib.einsum('pxyz, xq->pqyz', j1, moa)
        jaapp = lib.einsum('pqyz, prqs->rsyz', jaapp, casdm2())
        ####### FOR GRADIENT


        #gradC [:,ncore:nocc] = reduce(numpy.dot, ( (hcore + jc - kc).T, moa.conj(), casdm1().T))
        gradC2[:,ncore:nocc] = reduce(numpy.dot, ( (hcore + jc - kc)  , moa, casdm1()))


        #gradC [:,ncore:nocc] = gradC [:, ncore:nocc] + numpy.einsum('pqwx, wp->xq', jaapp, moa.conj())
        gradC2[:,ncore:nocc] = gradC2[:, ncore:nocc] + numpy.einsum('pqwx, xq->wp', jaapp, moa)

        return gradC2


        GradReal, GradImag = (gradC + gradC2).real, (1.j*(gradC - gradC2)).real
        Grad = 0.*gradC
        Grad.real, Grad.imag = GradReal, GradImag 
        print (numpy.linalg.norm(gradC - Grad.conj()/2.))
        return Grad
        '''
        #return gradC
        

    def calcGrad(self, mo, casdm1, casdm2, ERIS):
        hcore = self._scf.get_hcore()
        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact

        Grad = numpy.zeros((nmo, nocc), dtype=complex)

        moc, moa = mo[:,:ncore], mo[:,ncore:nocc]
        dmcore = numpy.dot(moc, moc.T.conj())
        dmcas = reduce(numpy.dot, (moa.conj(), casdm1(), moa.T)).conj()
        j,k = self._scf.get_jk(self.mol, dm=dmcore)        
        ja,ka = self._scf.get_jk(self.mol, dm=dmcas)        

        
        hcore = reduce(numpy.dot, (mo.conj().T, hcore , mo))
        #print (hcore.diagonal())
        Fc =  (hcore + reduce(numpy.dot, (mo.conj().T, (j-k), mo)))
        Grad[:,:ncore] = hcore[:,:ncore] + reduce(numpy.dot, (mo.conj().T, j + ja - k - ka, moc))
        
        Grad[:,ncore:nocc] = numpy.einsum('sp,qp->sq', Fc[:,ncore:nocc], casdm1())
        Grad[:,ncore:nocc]+= numpy.einsum('ruvw,tvuw->rt', ERIS.paaa, casdm2())
        return 2*Grad

    def calcH_op(self, Grad, mo, x, casdm1, casdm2, ERIS):
        x1 = casscf.unpack_uniq_var(x)
        eps = 1.e-5
        moxp = mo + eps*numpy.dot(mo, x)
        Gdxp = self.calcGrad(mox, casdm1, casdm2, ERIS)

        moxm = mo - eps*numpy.dot(mo, x)
        Gdxm = self.calcGrad(mox, casdm1, casdm2, ERIS)

        Hx = (Gdxp - Gdxm)/eps/2.
        return Hx

    def calcE(self, mo, casdm1, casdm2, ERIS=None):
        if ERIS is None: ERIS = self.ao2mo(mo)
        hcore = self._scf.get_hcore()
        ncore, nact = self.ncore, self.ncas
        nocc = ncore+nact
        nuc_energy = self.energy_nuc()
        
        moc = mo[:,:ncore]
        dmcore = numpy.dot(moc, moc.conj().T)
        j,k = self._scf.get_jk(self.mol, dm=dmcore)        

        
        hcore = reduce(numpy.dot, (mo.conj().T, hcore , mo))
        Fc = ( hcore + reduce(numpy.dot, (mo.conj().T, (j-k), mo)))
        Ecore = 0.5*numpy.sum((hcore+Fc).diagonal()[:ncore]) 

        Ecas1 = numpy.einsum('tu, tu', Fc[ncore:nocc, ncore:nocc], casdm1())
        Ecas2 = 0.5*numpy.einsum('tuvw, tvuw', ERIS.paaa[ncore:nocc], casdm2())

        #print (Ecore, Ecas1, Ecas2, nuc_energy, "mo")
        return Ecore+Ecas1+Ecas2+nuc_energy


    def optimizeOrbs(self, mo, casdm1, casdm2, addnoise, eris, r0, conv_tol, log):
        #the part of mo that is relevant 
        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact
 
        def GD(casscf, mo, nocc, addnoise):
            
            #add random noise
            if (addnoise):
                noise = numpy.zeros(mo.shape, dtype=complex)
                noise = numpy.random.random(mo.shape) +\
                    numpy.random.random(mo.shape)*1.j
                mo = numpy.dot(mo, expmat(-0.01*(noise - noise.T.conj())))
                S = casscf._scf.get_ovlp()

            
            '''###TEST GRAD####
            Grad, Gradnew = 0.*mo, 0.*mo
            ERIS = casscf.ao2mo(mo)
            E = casscf.calcE(mo, casdm1, casdm2, ERIS).real
            Grad[:,:nocc] = casscf.calcGrad(mo, casdm1, casdm2, ERIS)
            T = numpy.zeros((nmo, nmo),dtype=complex)
            eps = 1.e-5
            for i in range(nmo):
                for j in range(nocc):
                    T[i,j] = eps
                    U = mo + numpy.dot(mo, T) 
                    Ep = casscf.calcE(U, casdm1, casdm2).real

                    T[i,j] = -eps
                    U = mo + numpy.dot(mo, T) 
                    Em = casscf.calcE(U, casdm1, casdm2).real

                    #if (abs((Ep-Em)/2./eps - Grad[i,j].imag) > 1.e-9):
                    print (i,j,Grad[i,j], (Ep-Em)/2./eps, (Ep-2*E+Em)/eps/eps)
                    T[i,j] = 0.0
                exit(0)
            exit(0)
            '''#############
            

            tau = 0.01
            Grad, Gradnew = 0.*mo, 0.*mo
            ERIS = casscf.ao2mo(mo)
            Eold = casscf.calcE(mo, casdm1, casdm2, ERIS).real
            Grad[:,:nocc] = casscf.calcGrad(mo, casdm1, casdm2, ERIS)
            monew = 1.*numpy.dot(mo, expmat(-tau*(Grad-Grad.T.conj())))
            Enew = casscf.calcE(monew, casdm1, casdm2, ERIS).real
            ERIS = casscf.ao2mo(monew)

            for iter in range(1000):
                G = Grad - Grad.conj().T
                if (iter == 0):
                    normG0 = numpy.linalg.norm(G)
                print ("%d  %6.3e  %13.7e   %13.6e   g=%6.2e"\
                    %(iter, tau, Enew, Enew-Eold, numpy.linalg.norm(G)))
                #print (numpy.linalg.norm(reduce(numpy.dot, (mo.conj().T, S, mo))-numpy.eye(mo.shape[0])))

                if (iter == 90):
                    Enew = casscf.calcE(monew, casdm1, casdm2, ERIS).real
                Eold = Enew

                Gradnew[:,:nocc] = casscf.calcGrad(monew, casdm1, casdm2, ERIS)
                
                tau = 1.0
                mo, Grad = 1.*monew, 1.*Gradnew
                monew = numpy.dot(mo, expmat(-tau*(Grad-Grad.T.conj())))
                Gnorm = numpy.linalg.norm(Grad-Grad.conj().T)
                
                while tau > 1e-3:
                    monew = numpy.dot(mo, expmat(-tau*(Grad-Grad.T.conj())))
                    ERIS = casscf.ao2mo(monew)
                    Enew = casscf.calcE(monew, casdm1, casdm2, ERIS).real
                    
                    if (Enew < Eold - tau*1.e-4*Gnorm):
                        break
                    else:
                        tau *= 0.5
                if (tau < 1.e-3 or Gnorm < conv_tol or abs(Enew-Eold)<1.e-9):
                    print ("%d  %6.3e  %13.7e   %13.6e   g=%6.2e"\
                    %(iter, tau, Enew, Enew-Eold, numpy.linalg.norm(G)))
                    #print ("encountered small step size")
                    break
            return mo, Grad-Grad.T.conj(), iter, normG0
        
        return GD(self, mo, nocc, addnoise)
        


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 4
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

    #mol.basis = '6-31g'
    mol.basis = 'sto-3g'
    mol.build()

    '''
    m = scf.RHF(mol)
    ehf = m.kernel()
    print (ehf)
    mc = mcscf.CASSCF(m, 6,6)
    emc = mc.kernel()[0]
    print (emc)
    '''
    m = scf.X2C(mol)
    #m = scf.GHF(mol)
    ehf = m.kernel()
    print (ehf)
    #mc = ZCASSCF(m, 16, 8)
    mc = ZCASSCF(m, 8, 4)
    mc.fcisolver = shci.SHCI(mol)
    mc.fcisolver.sweep_epsilon=[1.e-5]
    mc.fcisolver.sweep_iter=[0]
    mc.fcisolver.davidsonTol = 1.e-6

    mo = 1.*m.mo_coeff
    noise = numpy.zeros(mo.shape, dtype=complex)
    noise = numpy.random.random(mo.shape) +\
                numpy.random.random(mo.shape)*1.j
    mo = numpy.dot(mo, expmat(-0.01*(noise - noise.T.conj())))
    emc = mc.kernel(mo)[0]

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

