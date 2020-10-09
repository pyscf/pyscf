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
from scipy.sparse.linalg import LinearOperator, minres, gmres
from pyscf.soscf import ciah
import pyscf.df
from pyscf import mcscf

def bracket(A,B):
    return numpy.dot(A,B)-numpy.dot(B,A)
def derivOfExp(a,dA, maxT=50):
    fact = 1.0
    deriv = 1.*dA
    bra = 1.*dA
    for i in range(maxT):
        bra = bracket(a, bra)
        fact *= -1./(i+2.)
        deriv += fact*bra

    return deriv

def get_jk_df(cderi, dm, with_j=True, with_k=True):
    if (with_j):
        j1 = lib.einsum('Lij,ji->L', cderi, dm)
        j = lib.einsum('Lij,L->ij', cderi, j1)
    if (with_k):
        kint = lib.einsum('Lij,ja->Lia', cderi, dm)
        k = lib.einsum('Laj,Lia->ij', cderi, kint)
    return j,k

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

        max_cycle_micro = casscf.micro_cycle_scheduler(locals())
        #max_stepsize = casscf.max_stepsize_scheduler(locals())
 

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
    max_cycle_macro = getattr(__config__, 'mcscf_mc1step_CASSCF_max_cycle_macro', 50)
    max_cycle_micro = getattr(__config__, 'mcscf_mc1step_CASSCF_max_cycle_micro', 4)
    conv_tol = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_conv_tol', 1e-7)
    conv_tol_grad = getattr(__config__, 'mcscf_zmc2step_ZCASSCF_conv_tol_grad', None)

    ah_level_shift = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_level_shift', 1e-8)
    ah_conv_tol = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_conv_tol', 1e-12)
    ah_max_cycle = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_max_cycle', 30)
    ah_lindep = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_lindep', 1e-14)
# * ah_start_tol and ah_start_cycle control the start point to use AH step.
#   In function rotate_orb_cc, the orbital rotation is carried out with the
#   approximate aug_hessian step after a few davidson updates of the AH eigen
#   problem.  Reducing ah_start_tol or increasing ah_start_cycle will delay
#   the start point of orbital rotation.
# * We can do early ah_start since it only affect the first few iterations.
#   The start tol will be reduced when approach the convergence point.
# * Be careful with the SYMMETRY BROKEN caused by ah_start_tol/ah_start_cycle.
#   ah_start_tol/ah_start_cycle actually approximates the hessian to reduce
#   the J/K evaluation required by AH.  When the system symmetry is higher
#   than the one given by mol.symmetry/mol.groupname,  symmetry broken might
#   occur due to this approximation,  e.g.  with the default ah_start_tol,
#   C2 (16o, 8e) under D2h symmetry might break the degeneracy between
#   pi_x, pi_y orbitals since pi_x, pi_y belong to different irreps.  It can
#   be fixed by increasing the accuracy of AH solver, e.g.
#               ah_start_tol = 1e-8;  ah_conv_tol = 1e-10
# * Classic AH can be simulated by setting eg
#               ah_start_tol = 1e-7
#               max_stepsize = 1.5
#               ah_grad_trust_region = 1e6
# ah_grad_trust_region allow gradients being increased in AH optimization
    ah_start_tol = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_start_tol', 2.5)
    ah_start_cycle = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_start_cycle', 3)
    ah_grad_trust_region = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_grad_trust_region', 3.0)
    internal_rotation = getattr(__config__, 'zmcscf_zmc2step_ZCASSCF_internal_rotation', True)
    kf_interval = getattr(__config__, 'mcscf_mc1step_CASSCF_kf_interval', 4)
    kf_trust_region = getattr(__config__, 'mcscf_mc1step_CASSCF_kf_trust_region', 3.0)

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

        #calculate the integrals
        self.cderi = pyscf.df.r_incore.cholesky_eri(mol, int3c='int3c2e_spinor')
        self.cderi.shape = (self.cderi.shape[0], self.mo_coeff.shape[0], self.mo_coeff.shape[1])

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

    def micro_cycle_scheduler(self, envs):
        return self.max_cycle_micro

        #log_norm_ddm = numpy.log(envs['norm_ddm'])
        #return max(self.max_cycle_micro, int(self.max_cycle_micro-1-log_norm_ddm))

    def ao2mo(self, mo_coeff=None, level=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if level is None: level=self.ao2mo_level      
        return zmc_ao2mo._ERIS(self, mo_coeff, method='incore',
                              level=level)

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
                         
    
    #Fully uses AO and is currently not efficient because AO 
    #integrals are assumed to be available cheaply
    def calcGradAO(self, mo, casdm1, casdm2):
        hcore = self._scf.get_hcore() 
        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact

        gradC2 = 0*mo
        moc = mo[:,:ncore]
        moa = mo[:,ncore:nocc]

        #ecore 
        dmcas = reduce(numpy.dot, (moa.conj(), casdm1(), moa.T)).conj()
        dmcore = numpy.dot(moc, moc.conj().T)

        jc,kc = self._scf.get_jk(self.mol, dm=dmcore)        
        ja,ka = self._scf.get_jk(self.mol, dm=dmcas)        
        
        gradC2[:,:ncore] = numpy.dot( (hcore + (jc-kc)+ja-ka) , moc)  

        gradC2[:,ncore:nocc] = reduce(numpy.dot, ( (hcore + jc - kc)  , moa, casdm1().T))

        ###THIS IS THE BIT WE NEED
        '''
        eri_ao_sp = mol.intor('int2e_spinor', aosym='s1')
        j1 = lib.einsum('wxyz, wp->pxyz', eri_ao_sp, moa.conj())
        jaapp = lib.einsum('pxyz, xq->pqyz', j1, moa)
        jaaap = lib.einsum('pqyz, zs->pqys', jaapp, moa)
        gradC2[:,ncore:nocc] += lib.einsum('pqys,prqs->yr', jaaap, casdm2())
        '''
        ####### FOR GRADIENT

        ###THIS BIT WILL BE EXPENSIVE       
        eripaaa = numpy.zeros((nmo, nact, nact,nact), dtype = complex)
        for i in range(nact):
            for j in range(i+1):
                dm = lib.einsum('x,y->xy',moa[:,i], moa[:,j].conj())
                j1 = self._scf.get_j(self.mol, dm = dm, hermi=0)
                j1 = numpy.triu(j1)
                j1 = j1 + j1.T - numpy.diag(numpy.diag(j1))
                eripaaa[:,:,i,j] = numpy.dot(j1,moa)
                if (i != j):
                    eripaaa[:,:,j,i] = numpy.dot(j1.conj().T,moa)
        gradC2[:,ncore:nocc] += lib.einsum('ypqr,sqpr->ys', eripaaa, casdm2())

        gradC2 = numpy.dot(mo.conj().T, gradC2)
        return 2*gradC2
       

    def calcGradDF(self, mo, casdm1, casdm2):
        hcore = self._scf.get_hcore()

        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact
        nuc_energy = self.energy_nuc()

        Grad = numpy.zeros((nmo, nmo), dtype=complex)
        moc, moa = mo[:,:ncore], mo[:,ncore:nocc]
        dmcore = numpy.dot(moc, moc.T.conj())
        dmcas = reduce(numpy.dot, (moa.conj(), casdm1(), moa.T)).conj()
        j,k=get_jk_df(self.cderi, dmcore)
        ja,ka=get_jk_df(self.cderi, dmcas)

        hcore = reduce(numpy.dot, (mo.conj().T, hcore , mo))
        Fc =  (hcore + reduce(numpy.dot, (mo.conj().T, (j-k), mo)))
        Grad[:,:ncore] = hcore[:,:ncore] + reduce(numpy.dot, (mo.conj().T, j + ja - k - ka, moc))
        
        Grad[:,ncore:nocc] = numpy.einsum('sp,qp->sq', Fc[:,ncore:nocc], casdm1())

        Lrq = lib.einsum('Lxy,ya->Lxa',self.cderi, moa)
        Lpq = lib.einsum('Lxa,xb->Lba', Lrq, moa.conj())
        Lrq = lib.einsum('Lxa,xy->Lya', Lrq, mo.conj())
        paaa = lib.einsum('Lxa,Lcd->xacd', Lrq, Lpq)
        Grad[:,ncore:nocc]+= lib.einsum('ruvw,tvuw->rt', paaa, casdm2())

        E = nuc_energy + 0.5*numpy.sum((hcore+Fc).diagonal()[:ncore]) 
        E += numpy.einsum('tu, tu', Fc[ncore:nocc, ncore:nocc], casdm1())
        E += 0.5*lib.einsum('ruvw,rvuw', paaa[ncore:nocc], casdm2())
        return 2*Grad, E

    def calcEDF(self, mo, casdm1, casdm2):
        hcore = self._scf.get_hcore()

        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact
        nuc_energy = self.energy_nuc()
 
        Grad = numpy.zeros((nmo, nmo), dtype=complex)
        moc, moa = mo[:,:ncore], mo[:,ncore:nocc]
        dmcore = numpy.dot(moc, moc.T.conj())
        dmcas = reduce(numpy.dot, (moa.conj(), casdm1(), moa.T)).conj()
        j,k=get_jk_df(self.cderi, dmcore)
        ja,ka=get_jk_df(self.cderi, dmcas)

        hcore = reduce(numpy.dot, (mo.conj().T, hcore , mo))
        Fc =  (hcore + reduce(numpy.dot, (mo.conj().T, (j-k), mo)))
        E = nuc_energy + 0.5*numpy.sum((hcore+Fc).diagonal()[:ncore]) 
        E += numpy.einsum('tu, tu', Fc[ncore:nocc, ncore:nocc], casdm1())
 
        Lrq = lib.einsum('Lxy,ya->Lxa',self.cderi, moa)
        Lpq = lib.einsum('Lxa,xb->Lba', Lrq, moa.conj())
        Lrq = lib.einsum('Lxa,xy->Lya', Lrq, moa.conj())
        paaa = lib.einsum('Lxa,Lcd->xacd', Lrq, Lpq)
        E += 0.5*lib.einsum('ruvw,rvuw', paaa, casdm2())

        return E

    def calcGrad(self, mo, casdm1, casdm2, ERIS=None):
        if ERIS is None: ERIS = self.ao2mo(mo,level=2)
        hcore = self._scf.get_hcore()
        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact

        nuc_energy = self.energy_nuc()
        Grad = numpy.zeros((nmo, nmo), dtype=complex)

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

        ###Make the lower traingular zero
        #for i in range(ncore,nocc):
        #    for j in range(i+1,nocc):
        #        Grad[i,j] = 0.0
        #Grad = Grad-Grad.conj().T
        Ecore = 0.5*numpy.sum((hcore+Fc).diagonal()[:ncore]) 
        Ecas1 = numpy.einsum('tu, tu', Fc[ncore:nocc, ncore:nocc], casdm1())
        Ecas2 = 0.5*numpy.einsum('tuvw, tvuw', ERIS.paaa[ncore:nocc], casdm2())

        return 2*Grad, (Ecore+Ecas1+Ecas2+nuc_energy).real
    

    def calcGradOld(self, mo, casdm1, casdm2, ERIS=None):
        if ERIS is None: ERIS = self.ao2mo(mo,level=2)
        hcore = self._scf.get_hcore()
        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact

        Grad = numpy.zeros((nmo, nmo), dtype=complex)

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

    ###IT IS NOT CORRECT, maybe some day i will fix it###
    def calcH(self, mo, x, casdm1, casdm2, ERIS):
        hcore = self._scf.get_hcore()
        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact

        moc, moa = mo[:,:ncore], mo[:,ncore:nocc]
        dmcore = numpy.dot(moc, moc.T.conj())
        dmcas = reduce(numpy.dot, (moa.conj(), casdm1(), moa.T)).conj()
        j,k = self._scf.get_jk(self.mol, dm=dmcore)        
        ja,ka = self._scf.get_jk(self.mol, dm=dmcas)        

        hcore = reduce(numpy.dot, (mo.conj().T, hcore , mo))
        Fc =  (hcore + reduce(numpy.dot, (mo.conj().T, (j-k), mo)))
        Fa = reduce(numpy.dot, (mo.conj().T, (ja-ka), mo))

        Hrr = numpy.zeros((nmo,nocc, nmo, nocc))
        for i in range(ncore):
            Hrr[:,i,:,i] = (Fc+Fa).real

        print (Hrr[2,2,3,3])
        Hrr[:,:ncore,:,:ncore] += \
            (numpy.einsum('xijy->xiyj',ERIS.poop[:,:ncore,:ncore])\
            - numpy.einsum('xyji->xiyj',ERIS.ppoo[:,:,:ncore,:ncore])\
            + numpy.einsum('xiyj->xiyj', ERIS.popo[:,:ncore,:,:ncore])\
            - numpy.einsum('xjyi->xiyj', ERIS.popo[:,:ncore,:,:ncore])).real

        Hrr[:,:ncore, :,ncore:nocc] +=\
            (numpy.einsum('xqyi,pq->yixp',ERIS.popo[:,ncore:nocc,:,:ncore], casdm1())\
            +numpy.einsum('xqiy,pq->yixp',ERIS.poop[:,ncore:nocc,:ncore,:], casdm1())\
            -numpy.einsum('xiyq,pq->yixp',ERIS.popo[:,:ncore,:,ncore:nocc], casdm1())\
            -numpy.einsum('yxpi,pq->yixq',ERIS.ppoo[:,:,ncore:nocc,:ncore], casdm1())).real

        Hrr[:,ncore:nocc,:,ncore:nocc] +=\
            (0*numpy.einsum('xy,pq->xpyq',Fc,casdm1())\
            + 1*numpy.einsum('xyrs,prqs->xpyq',ERIS.ppoo[:,:,ncore:nocc,ncore:nocc].real, casdm2())\
            + 0.5*numpy.einsum('xsry,rpqs->xpyq',ERIS.poop[:,ncore:nocc,ncore:nocc,:], casdm2())\
            + 0.5*numpy.einsum('xsry,prsq->xpyq',ERIS.poop[:,ncore:nocc,ncore:nocc,:], casdm2())\
            + 0.5*numpy.einsum('ysxr,pqrs->xpyq',ERIS.popo[:,ncore:nocc,:,ncore:nocc], casdm2())\
            + 0.5*numpy.einsum('yrxs,srpq->xpyq',ERIS.popo[:,ncore:nocc,:,ncore:nocc].conj(), casdm2())).real

        return 2*Hrr

    def calcE(self, mo, casdm1, casdm2, ERIS=None):
        if ERIS is None: ERIS = self.ao2mo(mo, level=1)
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
        Ecas2 = 0.5*numpy.einsum('tuvw, tvuw', ERIS.aaaa, casdm2())

        return Ecore+Ecas1+Ecas2+nuc_energy


    def uniq_var_indices(self, nmo, ncore, ncas, frozen=None):
        nocc = ncore + ncas
        mask = numpy.zeros((nmo,nmo),dtype=bool)
        mask[ncore:nocc,:ncore] = True
        mask[nocc:,:nocc] = True
        if self.internal_rotation:
            mask[ncore:nocc,ncore:nocc][numpy.tril_indices(ncas,-1)] = True
        if frozen is not None:
            if isinstance(frozen, (int, numpy.integer)):
                mask[:frozen] = mask[:,:frozen] = False
            else:
                frozen = numpy.asarray(frozen)
                mask[frozen] = mask[:,frozen] = False
        return mask

    def pack_vars(self, mat):
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)

        vec1 = mat[idx].real
        vec2 = mat[idx].imag
        vec = numpy.zeros((2*vec1.shape[0],))
        vec[:vec1.shape[0]] = 1*vec1
        vec[vec1.shape[0]:] = 1*vec2
        return vec

    # to anti symmetric matrix
    def unpack_vars(self, v):
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        mat = numpy.zeros((nmo,nmo), dtype=complex)
        nvars = v.shape[0]//2
        mat[idx] += v[:nvars]
        mat[idx] += 1j*v[nvars:]
        return mat - mat.T.conj()


    def optimizeOrbs(self, mo, casdm1, casdm2, addnoise, eris, r0, conv_tol, log):
        #the part of mo that is relevant 
        nmo, ncore, nact = mo.shape[0], self.ncore, self.ncas
        nocc = ncore+nact


        Grad, Gradnew = 0.*mo, 0.*mo
        T = numpy.zeros((nmo, nmo),dtype=complex)
        

        def NewtonStep(casscf, mo, nocc, Grad):
            Gradnewp, Gradnewm, T, monew = 0.*mo, 0.*mo, 0.*mo, 0.*mo

            nvars = casscf.pack_vars(mo).shape[0]
            G = casscf.pack_vars(Grad-Grad.conj().T)

            T, monew = 0*mo, 0.*mo

            Grad0, e = casscf.calcGradDF(mo, casdm1, casdm2)
            G0 = casscf.pack_vars(Grad0-Grad0.conj().T)

            def hop(x):
                Gradnewp, Gradnewm = 0.*mo, 0.*mo

                eps = 1.e-5
                Kappa = eps*casscf.unpack_vars(x)
                
                monew = numpy.dot(mo, expmat(Kappa))
                Gradnewp, e = casscf.calcGradDF(monew, casdm1, casdm2)
                #print (numpy.linalg.norm(Gradnewp-Grad))  

                f = numpy.dot(Kappa.conj().T, Gradnewp)
                h = numpy.dot(Gradnewp, Kappa.conj().T)
                Gradnewp = Gradnewp -0.5*(f-h) #- Gtemp.conj().T - 0.5*(f-f.conj().T+h-h.conj().T)
                Gradnewp = Gradnewp - Gradnewp.conj().T
                Gnewp= casscf.pack_vars(Gradnewp)
              
                
                Hx = (Gnewp - G0)/eps
                '''
                monew = numpy.dot(mo, expmat(-Kappa))
                Gradnewm = casscf.calcGradDF(monew, casdm1, casdm2)
                f = numpy.dot(Kappa.conj().T, Gradnewm)
                h = numpy.dot(Gradnewm, Kappa.conj().T)
                Gradnewm = Gradnewm -0.5*(f-h) #- Gtemp.conj().T - 0.5*(f-f.conj().T+h-h.conj().T)
                Gradnewm = Gradnewm - Gradnewm.conj().T
                Gnewm= casscf.pack_vars(Gradnewm)

                Hx = (Gnewp - Gnewm)/2./eps
                '''
                #print ("hop")
                return Hx

            ##we don't have diagonal elements
            def precond(x, y):
                return x

                
            x = 0*G
            x0 = 1.*G
            index = 0
            imic = 0
            ikf = 0
            g_op = lambda : G
            g_orb = 1.*G
            max_stepsize = 0.02
            norm_gkf = norm_gorb = numpy.linalg.norm(g_orb)

            for ah_end, ihop, w, dxi, hdxi, residual, seig \
                    in ciah.davidson_cc(hop, g_op, precond, x0,
                                        tol=casscf.ah_conv_tol, max_cycle=casscf.ah_max_cycle,
                                        lindep=casscf.ah_lindep, verbose=log):
                # residual = v[0] * (g+(h-e)x) ~ v[0] * grad
                norm_residual = numpy.linalg.norm(residual)

                if (ah_end or ihop == casscf.ah_max_cycle or # make sure to use the last step
                    ((norm_residual < casscf.ah_start_tol) and (ihop >= casscf.ah_start_cycle)) or
                    (seig < casscf.ah_lindep)):
                    imic += 1
                    dxmax = numpy.max(abs(dxi))
                    if dxmax > max_stepsize:
                        scale = max_stepsize / dxmax
                        log.debug1('... scale rotation size %g', scale)
                        dxi *= scale
                        hdxi *= scale
                    else:
                        scale = None

                    g_orb = g_orb + hdxi
                    x = x + dxi
                    norm_gorb = numpy.linalg.norm(g_orb)
                    norm_dxi = numpy.linalg.norm(dxi)
                    norm_dr = numpy.linalg.norm(x)
                    log.debug('    imic %d(%d)  |g[o]|=%5.3g  |dxi|=%5.3g  '
                            'max(|x|)=%5.3g  |dr|=%5.3g  eig=%5.3g  seig=%5.3g',
                            imic, ihop, norm_gorb, norm_dxi,
                            dxmax, norm_dr, w, seig)

                    ikf += 1
                    if ikf > 1 and norm_gorb > norm_gkf*casscf.ah_grad_trust_region:
                        g_orb = g_orb - hdxi
                        x -= dxi
                        #norm_gorb = numpy.linalg.norm(g_orb)
                        log.debug('|g| >> keyframe, Restore previouse step')
                        break

                    elif (norm_gorb < 1e-4*.3):
                        break

                    elif (ikf >= max(casscf.kf_interval, -numpy.log(norm_dr+1e-7)) or
        # Insert keyframe if the keyframe and the esitimated grad are too different
                        norm_gorb < norm_gkf/casscf.kf_trust_region):
                        ikf = 0
                        return x, norm_gorb

            return x, norm_gorb #numpy.linalg.norm(G)


        imicro, nmicro, T, Grad = 0, 2, numpy.zeros_like(mo), 0.*mo
        Enew = 0.
        #Eold = self.calcE(mo, casdm1, casdm2).real
        Grad, Eold = self.calcGrad(mo, casdm1, casdm2)
        Eolddf = self.calcEDF(mo, casdm1, casdm2).real
        while True:
            gnorm = numpy.linalg.norm(Grad-Grad.conj().T)

            #if gradient is converged then exit
            if ( gnorm < conv_tol or imicro >= nmicro ):               
                return mo, Grad, imicro, gnorm
                

            #find the newton step direction
            x, gnorm = NewtonStep(self, mo, nocc, Grad)
            T = self.unpack_vars(x)
 
            ###do line search along the AH direction
            tau = 1.0
            while tau > 1e-3:
                monew = numpy.dot(mo, expmat(tau*(T) ))
                Enewdf = self.calcEDF(monew, casdm1, casdm2).real

                if (Enewdf < Eolddf):# - tau * 1e-4*gnorm):
                    Grad, Enew = self.calcGrad(monew, casdm1, casdm2)
                    print ("%d  %6.3e  %13.7e   %13.6e   g=%6.2e"\
                    %(imicro, tau, Enew, Enew-Eold, gnorm))
                    Eold = Enew
                    Eolddf = Enewdf
                    mo = 1.*monew
                    break
                tau = tau/2.    

            imicro += 1
        exit(0)

        
        
        

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 4
    mol.memory=20000
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

    mol.basis = 'cc-pvtz'
    #mol.basis = 'sto-3g'
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
    
    #numpy.random.seed(5)
    #noise = numpy.zeros(mo.shape, dtype=complex)
    #noise = numpy.random.random(mo.shape) +\
    #            numpy.random.random(mo.shape)*1.j
    #mo = numpy.dot(mo, expmat(-0.01*(noise - noise.T.conj())))
    
    #import cProfile
    #cProfile.run('mc.kernel(mo)')
    #exit(0)
    emc = mc.kernel(mo)[0]
    exit(0)
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
    mc.verbose = 5
    mo = m.mo_coeff.copy()
    mo[:,2:5] = m.mo_coeff[:,[4,2,3]]
    emc = mc.mc2step(mo)[0]
    print(ehf, emc, emc-ehf)
    #-76.0267656731 -76.0873922924 -0.0606266193028
    print(emc - -76.0873923174, emc - -76.0926176464)



            
 