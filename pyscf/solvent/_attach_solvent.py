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
Attach ddCOSMO to SCF, MCSCF, and post-SCF methods.
'''

import copy
import numpy
from pyscf import lib
from pyscf.lib import logger
from functools import reduce
from pyscf import scf

def _for_scf(mf, solvent_obj, dm=None):
    '''Add solvent model to SCF (HF and DFT) method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(mf, _Solvation):
        mf.with_solvent = solvent_obj
        return mf

    oldMF = mf.__class__

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    class SCFWithSolvent(_Solvation, oldMF):
        def __init__(self, mf, solvent):
            self.__dict__.update(mf.__dict__)
            self.with_solvent = solvent
            self._keys.update(['with_solvent'])

        def dump_flags(self, verbose=None):
            oldMF.dump_flags(self, verbose)
            self.with_solvent.check_sanity()
            self.with_solvent.dump_flags(verbose)
            return self

        def reset(self, mol=None):
            self.with_solvent.reset(mol)
            return oldMF.reset(self, mol)

        # Note v_solvent should not be added to get_hcore for scf methods.
        # get_hcore is overloaded by many post-HF methods. Modifying
        # SCF.get_hcore may lead error.

        def get_veff(self, mol=None, dm=None, *args, **kwargs):
            vhf = oldMF.get_veff(self, mol, dm, *args, **kwargs)
            with_solvent = self.with_solvent
            if not with_solvent.frozen:
                with_solvent.e, with_solvent.v = with_solvent.kernel(dm)
            e_solvent, v_solvent = with_solvent.e, with_solvent.v

            # NOTE: v_solvent should not be added to vhf in this place. This is
            # because vhf is used as the reference for direct_scf in the next
            # iteration. If v_solvent is added here, it may break direct SCF.
            return lib.tag_array(vhf, e_solvent=e_solvent, v_solvent=v_solvent)

        def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                     diis=None, diis_start_cycle=None,
                     level_shift_factor=None, damp_factor=None):
            # DIIS was called inside oldMF.get_fock. v_solvent, as a function of
            # dm, should be extrapolated as well. To enable it, v_solvent has to be
            # added to the fock matrix before DIIS was called.
            if getattr(vhf, 'v_solvent', None) is None:
                vhf = self.get_veff(self.mol, dm)
            return oldMF.get_fock(self, h1e, s1e, vhf+vhf.v_solvent, dm, cycle, diis,
                                  diis_start_cycle, level_shift_factor, damp_factor)

        def energy_elec(self, dm=None, h1e=None, vhf=None):
            if dm is None:
                dm = self.make_rdm1()
            if getattr(vhf, 'e_solvent', None) is None:
                vhf = self.get_veff(self.mol, dm)
            e_tot, e_coul = oldMF.energy_elec(self, dm, h1e, vhf)
            e_tot += vhf.e_solvent
            self.scf_summary['e_solvent'] = vhf.e_solvent.real
            logger.debug(self, 'Solvent Energy = %.15g', vhf.e_solvent)
            return e_tot, e_coul

        def nuc_grad_method(self):
            grad_method = oldMF.nuc_grad_method(self)
            return self.with_solvent.nuc_grad_method(grad_method)

        Gradients = nuc_grad_method

        def gen_response(self, *args, **kwargs):
            vind = oldMF.gen_response(self, *args, **kwargs)
            is_uhf = isinstance(self, scf.uhf.UHF)
            # singlet=None is orbital hessian or CPHF type response function
            singlet = kwargs.get('singlet', True)
            singlet = singlet or singlet is None
            def vind_with_solvent(dm1):
                v = vind(dm1)
                if self.with_solvent.equilibrium_solvation:
                    if is_uhf:
                        v_solvent = self.with_solvent._B_dot_x(dm1)
                        v += v_solvent[0] + v_solvent[1]
                    elif singlet:
                        v += self.with_solvent._B_dot_x(dm1)
                return v
            return vind_with_solvent

        def stability(self, *args, **kwargs):
            # When computing orbital hessian, the second order derivatives of
            # solvent energy needs to be computed. It is enabled by
            # the attribute equilibrium_solvation in gen_response method.
            # If solvent was frozen, its contribution is treated as the
            # external potential. The response of solvent does not need to
            # be considered in stability analysis.
            with lib.temporary_env(self.with_solvent,
                                   equilibrium_solvation=not self.with_solvent.frozen):
                return oldMF.stability(self, *args, **kwargs)

    mf1 = SCFWithSolvent(mf, solvent_obj)
    return mf1

def _for_casscf(mc, solvent_obj, dm=None):
    '''Add solvent model to CASSCF method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(mc, _Solvation):
        mc.with_solvent = solvent_obj
        return mc

    oldCAS = mc.__class__

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    class CASSCFWithSolvent(_Solvation, oldCAS):
        def __init__(self, mc, solvent):
            self.__dict__.update(mc.__dict__)
            self.with_solvent = solvent
            self._e_tot_without_solvent = 0
            self._keys.update(['with_solvent'])

        def dump_flags(self, verbose=None):
            oldCAS.dump_flags(self, verbose)
            self.with_solvent.check_sanity()
            self.with_solvent.dump_flags(verbose)
            if self.conv_tol < 1e-7:
                logger.warn(self, 'CASSCF+ddCOSMO may not be able to '
                            'converge to conv_tol=%g', self.conv_tol)

            if (getattr(self._scf, 'with_solvent', None) and
                not getattr(self, 'with_solvent', None)):
                logger.warn(self, '''Solvent model %s was found in SCF object.
COSMO is not applied to the CASCI object. The CASSCF result is not affected by the SCF solvent model.
To enable the solvent model for CASSCF, a decoration to CASSCF object as below needs to be called
        from pyscf import solvent
        mc = mcscf.CASSCF(...)
        mc = solvent.ddCOSMO(mc)
''',
                            self._scf.with_solvent.__class__)
            return self

        def reset(self, mol=None):
            self.with_solvent.reset(mol)
            return oldCAS.reset(self, mol)

        def update_casdm(self, mo, u, fcivec, e_ci, eris, envs={}):
            casdm1, casdm2, gci, fcivec = \
                    oldCAS.update_casdm(self, mo, u, fcivec, e_ci, eris, envs)

# The potential is generated based on the density of current micro iteration.
# It will be added to hcore in casci function. Strictly speaking, this density
# is not the same to the CASSCF density (which was used to measure
# convergence) in the macro iterations.  When CASSCF is converged, it
# should be almost the same to the CASSCF density of the last macro iteration.
            with_solvent = self.with_solvent
            if not with_solvent.frozen:
                # Code to mimic dm = self.make_rdm1(ci=fcivec)
                mocore = mo[:,:self.ncore]
                mocas = mo[:,self.ncore:self.ncore+self.ncas]
                dm = reduce(numpy.dot, (mocas, casdm1, mocas.T))
                dm += numpy.dot(mocore, mocore.T) * 2
                with_solvent.e, with_solvent.v = with_solvent.kernel(dm)

            return casdm1, casdm2, gci, fcivec

# ddCOSMO Potential should be added to the effective potential. However, there
# is no hook to modify the effective potential in CASSCF. The workaround
# here is to modify hcore. It can affect the 1-electron operator in many CASSCF
# functions: gen_h_op, update_casdm, casci.  Note hcore is used to compute the
# energy for core density (Ecore).  The resultant total energy from casci
# function will include the contribution from ddCOSMO potential. The
# duplicated energy contribution from solvent needs to be removed.
        def get_hcore(self, mol=None):
            hcore = self._scf.get_hcore(mol)
            if self.with_solvent.v is not None:
                hcore += self.with_solvent.v
            return hcore

        def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
            log = logger.new_logger(self, verbose)
            log.debug('Running CASCI with solvent. Note the total energy '
                      'has duplicated contributions from solvent.')

            # In oldCAS.casci function, dE was computed based on the total
            # energy without removing the duplicated solvent contributions.
            # However, envs['elast'] is the last total energy with correct
            # solvent effects. Hack envs['elast'] to make oldCAS.casci print
            # the correct energy difference.
            envs['elast'] = self._e_tot_without_solvent
            e_tot, e_cas, fcivec = oldCAS.casci(self, mo_coeff, ci0, eris,
                                                verbose, envs)
            self._e_tot_without_solvent = e_tot

            log.debug('Computing corrections to the total energy.')
            dm = self.make_rdm1(ci=fcivec, ao_repr=True)

            with_solvent = self.with_solvent
            if with_solvent.e is not None:
                edup = numpy.einsum('ij,ji->', with_solvent.v, dm)
                e_tot = e_tot - edup + with_solvent.e
                log.info('Removing duplication %.15g, '
                         'adding E(solvent) = %.15g to total energy:\n'
                         '    E(CASSCF+solvent) = %.15g', edup, with_solvent.e, e_tot)

            # Update solvent effects for next iteration if needed
            if not with_solvent.frozen:
                with_solvent.e, with_solvent.v = with_solvent.kernel(dm)

            return e_tot, e_cas, fcivec

        def nuc_grad_method(self):
            logger.warn(self, '''
The code for CASSCF gradients was based on variational CASSCF wavefunction.
However, the ddCOSMO-CASSCF energy was not computed variationally.
Approximate gradients are evaluated here. A small error may be expected in the
gradients which corresponds to the contribution of
  MCSCF_DM * V_solvent[d/dX MCSCF_DM] + V_solvent[MCSCF_DM] * d/dX MCSCF_DM
''')
            grad_method = oldCAS.nuc_grad_method(self)
            return self.with_solvent.nuc_grad_method(grad_method)

        Gradients = nuc_grad_method

    return CASSCFWithSolvent(mc, solvent_obj)


def _for_casci(mc, solvent_obj, dm=None):
    '''Add solvent model to CASCI method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(mc, _Solvation):
        mc.with_solvent = solvent_obj
        return mc

    oldCAS = mc.__class__

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    class CASCIWithSolvent(_Solvation, oldCAS):
        def __init__(self, mc, solvent):
            self.__dict__.update(mc.__dict__)
            self.with_solvent = solvent
            self._keys.update(['with_solvent'])

        def dump_flags(self, verbose=None):
            oldCAS.dump_flags(self, verbose)
            self.with_solvent.check_sanity()
            self.with_solvent.dump_flags(verbose)
            return self

        def reset(self, mol=None):
            self.with_solvent.reset(mol)
            return oldCAS.reset(self, mol)

        def get_hcore(self, mol=None):
            hcore = self._scf.get_hcore(mol)
            if self.with_solvent.v is not None:
                # NOTE: get_hcore was called by CASCI to generate core
                # potential.  v_solvent is added in this place to take accounts the
                # effects of solvent. Its contribution is duplicated and it
                # should be removed from the total energy.
                hcore += self.with_solvent.v
            return hcore

        def kernel(self, mo_coeff=None, ci0=None, verbose=None):
            with_solvent = self.with_solvent

            log = logger.new_logger(self)
            log.info('\n** Self-consistently update the solvent effects for %s **',
                     oldCAS)
            log1 = copy.copy(log)
            log1.verbose -= 1  # Suppress a few output messages

            def casci_iter_(ci0, log):
                # self.e_tot, self.e_cas, and self.ci are updated in the call
                # to oldCAS.kernel
                e_tot, e_cas, ci0 = oldCAS.kernel(self, mo_coeff, ci0, log)[:3]

                if isinstance(self.e_cas, (float, numpy.number)):
                    dm = self.make_rdm1(ci=ci0)
                else:
                    log.debug('Computing solvent responses to DM of state %d',
                              with_solvent.state_id)
                    dm = self.make_rdm1(ci=ci0[with_solvent.state_id])

                if with_solvent.e is not None:
                    edup = numpy.einsum('ij,ji->', with_solvent.v, dm)
                    self.e_tot += with_solvent.e - edup

                if not with_solvent.frozen:
                    with_solvent.e, with_solvent.v = with_solvent.kernel(dm)
                return self.e_tot, e_cas, ci0

            if with_solvent.frozen:
                with lib.temporary_env(self, _finalize=lambda:None):
                    casci_iter_(ci0, log)
                log.note('Total energy with solvent effects')
                self._finalize()
                return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

            self.converged = False
            with lib.temporary_env(self, canonicalization=False):
                e_tot = e_last = 0
                for cycle in range(self.with_solvent.max_cycle):
                    log.info('\n** Solvent self-consistent cycle %d:', cycle)
                    e_tot, e_cas, ci0 = casci_iter_(ci0, log1)

                    de = e_tot - e_last
                    if isinstance(e_cas, (float, numpy.number)):
                        log.info('Sovlent cycle %d  E(CASCI+solvent) = %.15g  '
                                 'dE = %g', cycle, e_tot, de)
                    else:
                        for i, e in enumerate(e_tot):
                            log.info('Solvent cycle %d  CASCI root %d  '
                                     'E(CASCI+solvent) = %.15g  dE = %g',
                                     cycle, i, e, de[i])

                    if abs(e_tot-e_last).max() < with_solvent.conv_tol:
                        self.converged = True
                        break
                    e_last = e_tot

            # An extra cycle to canonicalize CASCI orbitals
            with lib.temporary_env(self, _finalize=lambda:None):
                casci_iter_(ci0, log)
            if self.converged:
                log.info('self-consistent CASCI+solvent converged')
            else:
                log.info('self-consistent CASCI+solvent not converged')
            log.note('Total energy with solvent effects')
            self._finalize()
            return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

        def nuc_grad_method(self):
            logger.warn(self, '''
The code for CASCI gradients was based on variational CASCI wavefunction.
However, the ddCOSMO-CASCI energy was not computed variationally.
Approximate gradients are evaluated here. A small error may be expected in the
gradients which corresponds to the contribution of
  MCSCF_DM * V_solvent[d/dX MCSCF_DM] + V_solvent[MCSCF_DM] * d/dX MCSCF_DM
''')
            grad_method = oldCAS.nuc_grad_method(self)
            return self.with_solvent.nuc_grad_method(grad_method)

        Gradients = nuc_grad_method

    return CASCIWithSolvent(mc, solvent_obj)


def _for_post_scf(method, solvent_obj, dm=None):
    '''A wrapper of solvent model for post-SCF methods (CC, CI, MP etc.)

    NOTE: this implementation often causes (macro iteration) convergence issue

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(method, _Solvation):
        method.with_solvent = solvent_obj
        method._scf.with_solvent = solvent_obj
        return method

    old_method = method.__class__

    # Ensure that the underlying _scf object has solvent model enabled
    if getattr(method._scf, 'with_solvent', None):
        scf_with_solvent = method._scf
    else:
        scf_with_solvent = _for_scf(method._scf, solvent_obj, dm)
        if dm is None:
            solvent_obj = scf_with_solvent.with_solvent
            solvent_obj.e, solvent_obj.v = \
                    solvent_obj.kernel(scf_with_solvent.make_rdm1())

    # Post-HF objects access the solvent effects indirectly through the
    # underlying ._scf object.
    basic_scanner = method.as_scanner()
    basic_scanner._scf = scf_with_solvent.as_scanner()

    if dm is not None:
        solvent_obj = scf_with_solvent.with_solvent
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    class PostSCFWithSolvent(_Solvation, old_method):
        def __init__(self, method):
            self.__dict__.update(method.__dict__)
            self._scf = scf_with_solvent

        @property
        def with_solvent(self):
            return self._scf.with_solvent

        def dump_flags(self, verbose=None):
            old_method.dump_flags(self, verbose)
            self.with_solvent.check_sanity()
            self.with_solvent.dump_flags(verbose)
            return self

        def reset(self, mol=None):
            self.with_solvent.reset(mol)
            return old_method.reset(self, mol)

        def kernel(self, *args, **kwargs):
            with_solvent = self.with_solvent
            # The underlying ._scf object is decorated with solvent effects.
            # The resultant Fock matrix and orbital energies both include the
            # effects from solvent. It means that solvent effects for post-HF
            # methods are automatically counted if solvent is enabled at scf
            # level.
            if with_solvent.frozen:
                return old_method.kernel(self, *args, **kwargs)

            log = logger.new_logger(self)
            log.info('\n** Self-consistently update the solvent effects for %s **',
                     old_method)
            ##TODO: Suppress a few output messages
            #log1 = copy.copy(log)
            #log1.note, log1.info = log1.info, log1.debug

            e_last = 0
            #diis = lib.diis.DIIS()
            for cycle in range(self.with_solvent.max_cycle):
                log.info('\n** Solvent self-consistent cycle %d:', cycle)
                # Solvent effects are applied when accessing the
                # underlying ._scf objects. The flag frozen=True ensures that
                # the generated potential with_solvent.v is passed to the
                # the post-HF object, without being updated in the implicit
                # call to the _scf iterations.
                with lib.temporary_env(with_solvent, frozen=True):
                    e_tot = basic_scanner(self.mol)
                    dm = basic_scanner.make_rdm1(ao_repr=True)
                    #dm = diis.update(dm)

                # To generate the solvent potential for ._scf object. Since
                # frozen is set when calling basic_scanner, the solvent
                # effects are frozen during the scf iterations.
                with_solvent.e, with_solvent.v = with_solvent.kernel(dm)

                de = e_tot - e_last
                log.info('Sovlent cycle %d  E_tot = %.15g  dE = %g',
                         cycle, e_tot, de)

                if abs(e_tot-e_last).max() < with_solvent.conv_tol:
                    break
                e_last = e_tot

            # An extra cycle to compute the total energy
            log.info('\n** Extra cycle for solvent effects')
            with lib.temporary_env(with_solvent, frozen=True):
                #Update everything except the _scf object and _keys
                basic_scanner(self.mol)
                self.__dict__.update(basic_scanner.__dict__)
                self._scf.__dict__.update(basic_scanner._scf.__dict__)
            self._finalize()
            return self.e_corr, None

        def nuc_grad_method(self):
            logger.warn(self, '''
Approximate gradients are evaluated here. A small error may be expected in the
gradients which corresponds to the contribution of
  DM * V_solvent[d/dX DM] + V_solvent[DM] * d/dX DM
''')
            grad_method = old_method.nuc_grad_method(self)
            return self.with_solvent.nuc_grad_method(grad_method)

        Gradients = nuc_grad_method

    return PostSCFWithSolvent(method)


def _for_tdscf(method, solvent_obj, dm=None):
    '''Add solvent model in TDDFT calculations.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(method, _Solvation):
        method.with_solvent = solvent_obj
        method._scf.with_solvent = solvent_obj
        return method

    old_method = method.__class__

    # Ensure that the underlying _scf object has solvent model enabled
    if getattr(method._scf, 'with_solvent', None):
        scf_with_solvent = method._scf
    else:
        scf_with_solvent = _for_scf(method._scf, solvent_obj, dm).run()

    if dm is not None:
        solvent_obj = scf_with_solvent.with_solvent
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    class TDSCFWithSolvent(_Solvation, old_method):
        def __init__(self, method):
            self.__dict__.update(method.__dict__)
            self._scf = scf_with_solvent
            self.with_solvent = self._scf.with_solvent
            self._keys.update(['with_solvent'])

        @property
        def equilibrium_solvation(self):
            '''Whether to allow the solvent rapidly responds to the changes of
            electronic structure or geometry of solute.
            '''
            return self.with_solvent.equilibrium_solvation
        @equilibrium_solvation.setter
        def equilibrium_solvation(self, val):
            if val and self.with_solvent.frozen:
                logger.warn(self, 'Solvent model was set to be frozen in the '
                            'ground state SCF calculation. It may conflict to '
                            'the assumption of equilibrium solvation.\n'
                            'You may set _scf.with_solvent.frozen = False and '
                            'rerun the ground state calculation _scf.run().')
            self.with_solvent.equilibrium_solvation = val

        def dump_flags(self, verbose=None):
            old_method.dump_flags(self, verbose)
            self.with_solvent.check_sanity()
            self.with_solvent.dump_flags(verbose)
            return self

        def reset(self, mol=None):
            self.with_solvent.reset(mol)
            return old_method.reset(self, mol)

        def get_ab(self, mf=None):
            #if mf is None: mf = self._scf
            #a, b = get_ab(mf)
            if self.equilibrium_solvation:
                raise NotImplementedError

        def nuc_grad_method(self):
            grad_method = old_method.nuc_grad_method(self)
            return self.with_solvent.nuc_grad_method(grad_method)

    mf1 = TDSCFWithSolvent(method)
    return mf1

# 1. A tag to label the derived method class
class _Solvation(object):
    pass
