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
domain decomposition COSMO

See also the code on github

https://github.com/filippolipparini/ddPCM

and the papers

[1] Domain decomposition for implicit solvation models.
E. Cances, Y. Maday, B. Stamm
J. Chem. Phys., 139, 054111 (2013)
http://dx.doi.org/10.1063/1.4816767

[2] Fast Domain Decomposition Algorithm for Continuum Solvation Models: Energy and First Derivatives.
F. Lipparini, B. Stamm, E. Cances, Y. Maday, B. Mennucci
J. Chem. Theory Comput., 9, 3637-3648 (2013)
http://dx.doi.org/10.1021/ct400280b

[3] Quantum, classical, and hybrid QM/MM calculations in solution: General implementation of the ddCOSMO linear scaling strategy.
F. Lipparini, G. Scalmani, L. Lagardere, B. Stamm, E. Cances, Y. Maday, J.-P.Piquemal, M. J. Frisch, B. Mennucci
J. Chem. Phys., 141, 184108 (2014)
http://dx.doi.org/10.1063/1.4901304

-- Dielectric constants (from https://gaussian.com/scrf/) --
More dataset can be found in Minnesota Solvent Descriptor Database
(https://comp.chem.umn.edu/solvation)
Water                                  78.3553
Acetonitrile                           35.688
Methanol                               32.613
Ethanol                                24.852
IsoQuinoline                           11.00
Quinoline                              9.16
Chloroform                             4.7113
DiethylEther                           4.2400
Dichloromethane                        8.93
DiChloroEthane                         10.125
CarbonTetraChloride                    2.2280
Benzene                                2.2706
Toluene                                2.3741
ChloroBenzene                          5.6968
NitroMethane                           36.562
Heptane                                1.9113
CycloHexane                            2.0165
Aniline                                6.8882
Acetone                                20.493
TetraHydroFuran                        7.4257
DiMethylSulfoxide                      46.826
Argon                                  1.430
Krypton                                1.519
Xenon                                  1.706
n-Octanol                              9.8629
1,1,1-TriChloroEthane                  7.0826
1,1,2-TriChloroEthane                  7.1937
1,2,4-TriMethylBenzene                 2.3653
1,2-DiBromoEthane                      4.9313
1,2-EthaneDiol                         40.245
1,4-Dioxane                            2.2099
1-Bromo-2-MethylPropane                7.7792
1-BromoOctane                          5.0244
1-BromoPentane                         6.269
1-BromoPropane                         8.0496
1-Butanol                              17.332
1-ChloroHexane                         5.9491
1-ChloroPentane                        6.5022
1-ChloroPropane                        8.3548
1-Decanol                              7.5305
1-FluoroOctane                         3.89
1-Heptanol                             11.321
1-Hexanol                              12.51
1-Hexene                               2.0717
1-Hexyne                               2.615
1-IodoButane                           6.173
1-IodoHexaDecane                       3.5338
1-IodoPentane                          5.6973
1-IodoPropane                          6.9626
1-NitroPropane                         23.73
1-Nonanol                              8.5991
1-Pentanol                             15.13
1-Pentene                              1.9905
1-Propanol                             20.524
2,2,2-TriFluoroEthanol                 26.726
2,2,4-TriMethylPentane                 1.9358
2,4-DiMethylPentane                    1.8939
2,4-DiMethylPyridine                   9.4176
2,6-DiMethylPyridine                   7.1735
2-BromoPropane                         9.3610
2-Butanol                              15.944
2-ChloroButane                         8.3930
2-Heptanone                            11.658
2-Hexanone                             14.136
2-MethoxyEthanol                       17.2
2-Methyl-1-Propanol                    16.777
2-Methyl-2-Propanol                    12.47
2-MethylPentane                        1.89
2-MethylPyridine                       9.9533
2-NitroPropane                         25.654
2-Octanone                             9.4678
2-Pentanone                            15.200
2-Propanol                             19.264
2-Propen-1-ol                          19.011
3-MethylPyridine                       11.645
3-Pentanone                            16.78
4-Heptanone                            12.257
4-Methyl-2-Pentanone                   12.887
4-MethylPyridine                       11.957
5-Nonanone                             10.6
AceticAcid                             6.2528
AcetoPhenone                           17.44
a-ChloroToluene                        6.7175
Anisole                                4.2247
Benzaldehyde                           18.220
BenzoNitrile                           25.592
BenzylAlcohol                          12.457
BromoBenzene                           5.3954
BromoEthane                            9.01
Bromoform                              4.2488
Butanal                                13.45
ButanoicAcid                           2.9931
Butanone                               18.246
ButanoNitrile                          24.291
ButylAmine                             4.6178
ButylEthanoate                         4.9941
CarbonDiSulfide                        2.6105
Cis-1,2-DiMethylCycloHexane            2.06
Cis-Decalin                            2.2139
CycloHexanone                          15.619
CycloPentane                           1.9608
CycloPentanol                          16.989
CycloPentanone                         13.58
Decalin-mixture                        2.196
DiBromomEthane                         7.2273
DiButylEther                           3.0473
DiEthylAmine                           3.5766
DiEthylSulfide                         5.723
DiIodoMethane                          5.32
DiIsoPropylEther                       3.38
DiMethylDiSulfide                      9.6
DiPhenylEther                          3.73
DiPropylAmine                          2.9112
e-1,2-DiChloroEthene                   2.14
e-2-Pentene                            2.051
EthaneThiol                            6.667
EthylBenzene                           2.4339
EthylEthanoate                         5.9867
EthylMethanoate                        8.3310
EthylPhenylEther                       4.1797
FluoroBenzene                          5.42
Formamide                              108.94
FormicAcid                             51.1
HexanoicAcid                           2.6
IodoBenzene                            4.5470
IodoEthane                             7.6177
IodoMethane                            6.8650
IsoPropylBenzene                       2.3712
m-Cresol                               12.44
Mesitylene                             2.2650
MethylBenzoate                         6.7367
MethylButanoate                        5.5607
MethylCycloHexane                      2.024
MethylEthanoate                        6.8615
MethylMethanoate                       8.8377
MethylPropanoate                       6.0777
m-Xylene                               2.3478
n-ButylBenzene                         2.36
n-Decane                               1.9846
n-Dodecane                             2.0060
n-Hexadecane                           2.0402
n-Hexane                               1.8819
NitroBenzene                           34.809
NitroEthane                            28.29
n-MethylAniline                        5.9600
n-MethylFormamide-mixture              181.56
n,n-DiMethylAcetamide                  37.781
n,n-DiMethylFormamide                  37.219
n-Nonane                               1.9605
n-Octane                               1.9406
n-Pentadecane                          2.0333
n-Pentane                              1.8371
n-Undecane                             1.9910
o-ChloroToluene                        4.6331
o-Cresol                               6.76
o-DiChloroBenzene                      9.9949
o-NitroToluene                         25.669
o-Xylene                               2.5454
Pentanal                               10.0
PentanoicAcid                          2.6924
PentylAmine                            4.2010
PentylEthanoate                        4.7297
PerFluoroBenzene                       2.029
p-IsoPropylToluene                     2.2322
Propanal                               18.5
PropanoicAcid                          3.44
PropanoNitrile                         29.324
PropylAmine                            4.9912
PropylEthanoate                        5.5205
p-Xylene                               2.2705
Pyridine                               12.978
sec-ButylBenzene                       2.3446
tert-ButylBenzene                      2.3447
TetraChloroEthene                      2.268
TetraHydroThiophene-s,s-dioxide        43.962
Tetralin                               2.771
Thiophene                              2.7270
Thiophenol                             4.2728
trans-Decalin                          2.1781
TriButylPhosphate                      8.1781
TriChloroEthene                        3.422
TriEthylAmine                          2.3832
Xylene-mixture                         2.3879
z-1,2-DiChloroEthene                   9.2
'''

import ctypes
import copy
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import df
from pyscf.dft import gen_grid, numint
from pyscf.data import radii
from pyscf.symm import sph
from functools import reduce

def ddcosmo_for_scf(mf, solvent_obj=None, dm=None):
    '''Patch ddCOSMO to SCF (HF and DFT) method.
    
    Kwargs:
        dm : if given, solvent does not response to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if getattr(mf, 'with_solvent', None):
        if solvent_obj is not None:
            mf.with_solvent = solvent_obj
        return mf

    oldMF = mf.__class__
    if solvent_obj is None:
        solvent_obj = DDCOSMO(mf.mol)

    if dm is not None:
        solvent_obj.epcm, solvent_obj.vpcm = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    class SCFWithSolvent(oldMF):
        def __init__(self, mf, solvent):
            self.__dict__.update(mf.__dict__)
            self.with_solvent = solvent
            self._keys.update(['with_solvent'])

        def dump_flags(self, verbose=None):
            oldMF.dump_flags(self, verbose)
            self.with_solvent.check_sanity()
            self.with_solvent.dump_flags(verbose)
            return self

        # Note vpcm should not be added to get_hcore for scf methods.
        # get_hcore is overloaded by many post-HF methods. Modifying
        # SCF.get_hcore may lead error.

        def get_veff(self, mol=None, dm=None, *args, **kwargs):
            vhf = oldMF.get_veff(self, mol, dm, *args, **kwargs)
            with_solvent = self.with_solvent
            if not with_solvent.frozen:
                with_solvent.epcm, with_solvent.vpcm = with_solvent.kernel(dm)
            epcm, vpcm = with_solvent.epcm, with_solvent.vpcm

            # NOTE: vpcm should not be added to vhf in this place. This is
            # because vhf is used as the reference for direct_scf in the next
            # iteration. If vpcm is added here, it may break direct SCF.
            return lib.tag_array(vhf, epcm=epcm, vpcm=vpcm)

        def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                     diis=None, diis_start_cycle=None,
                     level_shift_factor=None, damp_factor=None):
            # DIIS was called inside oldMF.get_fock. vpcm, as a function of
            # dm, should be extrapolated as well. To enable it, vpcm has to be
            # added to the fock matrix before DIIS was called.
            if getattr(vhf, 'vpcm', None) is None:
                vhf = self.get_veff(self.mol, dm)
            return oldMF.get_fock(self, h1e, s1e, vhf+vhf.vpcm, dm, cycle, diis,
                                  diis_start_cycle, level_shift_factor, damp_factor)

        def energy_elec(self, dm=None, h1e=None, vhf=None):
            if dm is None:
                dm = self.make_rdm1()
            if getattr(vhf, 'epcm', None) is None:
                vhf = self.get_veff(self.mol, dm)
            e_tot, e_coul = oldMF.energy_elec(self, dm, h1e, vhf)
            e_tot += vhf.epcm
            self.scf_summary['epcm'] = vhf.epcm.real
            logger.debug(self, '  E_diel = %.15g', vhf.epcm)
            return e_tot, e_coul

        def nuc_grad_method(self):
            from pyscf.solvent import ddcosmo_grad
            grad_method = oldMF.nuc_grad_method(self)
            return ddcosmo_grad.ddcosmo_grad(grad_method, self.with_solvent)

        Gradients = nuc_grad_method

    mf1 = SCFWithSolvent(mf, solvent_obj)
    return mf1

def ddcosmo_for_casscf(mc, solvent_obj=None, dm=None):
    '''Patch ddCOSMO to CASSCF method.
    
    Kwargs:
        dm : if given, solvent does not response to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if getattr(mc, 'with_solvent', None):
        if solvent_obj is not None:
            mc.with_solvent = solvent_obj
        return mc

    oldCAS = mc.__class__
    if solvent_obj is None:
        if getattr(mc._scf, 'with_solvent', None):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = DDCOSMO(mc.mol)

    if dm is not None:
        solvent_obj.epcm, solvent_obj.vpcm = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    class CASSCFWithSolvent(oldCAS):
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
            return self

        def update_casdm(self, mo, u, fcivec, e_ci, eris, envs={}):
            casdm1, casdm2, gci, fcivec = \
                    oldCAS.update_casdm(self, mo, u, fcivec, e_ci, eris, envs)

# The potential is generated based on the density of current micro iteration.
# It will be added to hcore in casci function. Strictly speaking, this density
# is not the same to the CASSCF density (which was used to measure
# convergence) in the macro iterations.  When CASSCF is converged, it
# should be almost the same to the CASSCF density of the macro iterations.
            with_solvent = self.with_solvent
            if not with_solvent.frozen:
                # Code to mimic dm = self.make_rdm1(ci=fcivec)
                mocore = mo[:,:self.ncore]
                mocas = mo[:,self.ncore:self.ncore+self.ncas]
                dm = reduce(numpy.dot, (mocas, casdm1, mocas.T))
                dm += numpy.dot(mocore, mocore.T) * 2
                with_solvent.epcm, with_solvent.vpcm = with_solvent.kernel(dm)

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
            if self.with_solvent.vpcm is not None:
                hcore += self.with_solvent.vpcm
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
            if with_solvent.epcm is not None:
                edup = numpy.einsum('ij,ji->', with_solvent.vpcm, dm)
                ediel = with_solvent.epcm
                e_tot = e_tot - edup + ediel
                log.info('Removing duplication %.15g, '
                         'adding E_diel = %.15g to total energy:\n'
                         '    E(CASSCF+solvent) = %.15g', edup, ediel, e_tot)

            # Update solvent effects for next iteration if needed
            if not with_solvent.frozen:
                with_solvent.epcm, with_solvent.vpcm = with_solvent.kernel(dm)

            return e_tot, e_cas, fcivec

        def nuc_grad_method(self):
            from pyscf.solvent import ddcosmo_grad
            grad_method = oldCAS.nuc_grad_method(self)
            return ddcosmo_grad.ddcosmo_grad(grad_method, self.with_solvent)

        Gradients = nuc_grad_method

    return CASSCFWithSolvent(mc, solvent_obj)


def ddcosmo_for_casci(mc, solvent_obj=None, dm=None):
    '''Patch ddCOSMO to CASCI method.
    
    Kwargs:
        dm : if given, solvent does not response to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if getattr(mc, 'with_solvent', None):
        if solvent_obj is not None:
            mc.with_solvent = solvent_obj
        return mc

    oldCAS = mc.__class__
    if solvent_obj is None:
        if getattr(mc._scf, 'with_solvent', None):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = DDCOSMO(mc.mol)

    if dm is not None:
        solvent_obj.epcm, solvent_obj.vpcm = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    class CASCIWithSolvent(oldCAS):
        def __init__(self, mc, solvent):
            self.__dict__.update(mc.__dict__)
            self.with_solvent = solvent
            self._keys.update(['with_solvent'])

        def dump_flags(self, verbose=None):
            oldCAS.dump_flags(self, verbose)
            self.with_solvent.check_sanity()
            self.with_solvent.dump_flags(verbose)
            return self

        def get_hcore(self, mol=None):
            hcore = self._scf.get_hcore(mol)
            if self.with_solvent.vpcm is not None:
                # NOTE: get_hcore was called by CASCI to generate core
                # potential.  vpcm is added in this place to take accounts the
                # effects of solvent. Its contribution is duplicated and it
                # should be removed from the total energy.
                hcore += self.with_solvent.vpcm
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

                if with_solvent.epcm is not None:
                    edup = numpy.einsum('ij,ji->', with_solvent.vpcm, dm)
                    self.e_tot += with_solvent.epcm - edup

                if not with_solvent.frozen:
                    with_solvent.epcm, with_solvent.vpcm = with_solvent.kernel(dm)
                log.debug('  E_diel = %.15g', with_solvent.epcm)
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
            from pyscf.solvent import ddcosmo_grad
            grad_method = oldCAS.nuc_grad_method(self)
            return ddcosmo_grad.ddcosmo_grad(grad_method, self.with_solvent)

        Gradients = nuc_grad_method

    return CASCIWithSolvent(mc, solvent_obj)


def ddcosmo_for_post_scf(method, solvent_obj=None, dm=None):
    '''Default wrapper to patch ddCOSMO to post-SCF methods (CC, CI, MP,
    TDDFT etc.)

    NOTE: this implementation often causes (macro iteration) convergence issue
    
    Kwargs:
        dm : if given, solvent does not response to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if getattr(method, 'with_solvent', None):
        if solvent_obj is not None:
            method.with_solvent = solvent_obj
            method._scf.with_solvent = solvent_obj
        return method

    old_method = method.__class__

    if getattr(method._scf, 'with_solvent', None):
        scf_with_solvent = method._scf
        if solvent_obj is not None:
            scf_with_solvent.with_solvent = solvent_obj
    else:
        scf_with_solvent = ddcosmo_for_scf(method._scf, solvent_obj, dm)
        if dm is None:
            solvent_obj = scf_with_solvent.with_solvent
            solvent_obj.epcm, solvent_obj.vpcm = \
                    solvent_obj.kernel(scf_with_solvent.make_rdm1())

    # Post-HF objects access the solvent effects indirectly through the
    # underlying ._scf object.
    basic_scanner = method.as_scanner()
    basic_scanner._scf = scf_with_solvent.as_scanner()

    if dm is not None:
        solvent_obj = scf_with_solvent.with_solvent
        solvent_obj.epcm, solvent_obj.vpcm = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    class PostSCFWithSolvent(old_method):
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
                # the generated potential with_solvent.vpcm is passed to the
                # the post-HF object, without being updated in the implicit
                # call to the _scf iterations.
                with lib.temporary_env(with_solvent, frozen=True):
                    e_tot = basic_scanner(self.mol)
                    dm = basic_scanner.make_rdm1(ao_repr=True)
                    #dm = diis.update(dm)

                # To generate the solvent potential for ._scf object. Since
                # frozen is set when calling basic_scanner, the solvent
                # effects are frozen during the scf iterations.
                with_solvent.epcm, with_solvent.vpcm = with_solvent.kernel(dm)

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
            from pyscf.solvent import ddcosmo_grad
            grad_method = old_method.nuc_grad_method(self)
            return ddcosmo_grad.ddcosmo_grad(grad_method, self.with_solvent)

        Gradients = nuc_grad_method

    return PostSCFWithSolvent(method)


# Inject DDCOSMO into other methods
from pyscf import scf
from pyscf import mcscf
from pyscf import mp, ci, cc
scf.hf.SCF.DDCOSMO = ddcosmo_for_scf
mcscf.casci.DDCOSMO = ddcosmo_for_casci
mcscf.mc1step.DDCOSMO = ddcosmo_for_casscf
mp.mp2.MP2.DDCOSMO = ddcosmo_for_post_scf
ci.cisd.CISD.DDCOSMO = ddcosmo_for_post_scf
cc.ccsd.CCSD.DDCOSMO = ddcosmo_for_post_scf


# TODO: Testing the value of psi (make_psi_vmat).  All intermediates except
# psi are tested against ddPCM implementation on github. Psi needs to be
# computed by the host program. It requires the numerical integration code. 
def gen_ddcosmo_solver(pcmobj, verbose=None):
    '''Generate ddcosmo function to compute energy and potential matrix
    '''
    mol = pcmobj.mol
    if pcmobj.grids.coords is None:
        pcmobj.grids.build(with_non0tab=True)

    natm = mol.natm
    lmax = pcmobj.lmax

    r_vdw = pcmobj.get_atomic_radii()
    coords_1sph, weights_1sph = make_grids_one_sphere(pcmobj.lebedev_order)
    ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, lmax, True))

    fi = make_fi(pcmobj, r_vdw)
    ui = 1 - fi
    ui[ui<0] = 0
    nexposed = numpy.count_nonzero(ui==1)
    nbury = numpy.count_nonzero(ui==0)
    on_shell = numpy.count_nonzero(ui>0) - nexposed
    logger.debug(pcmobj, 'Num points exposed %d', nexposed)
    logger.debug(pcmobj, 'Num points buried %d', nbury)
    logger.debug(pcmobj, 'Num points on shell %d', on_shell)

    nlm = (lmax+1)**2
    Lmat = make_L(pcmobj, r_vdw, ylm_1sph, fi)
    Lmat = Lmat.reshape(natm*nlm,-1)

    cached_pol = cache_fake_multipoles(pcmobj.grids, r_vdw, lmax)

    def gen_vind(dm):
        pcmobj._dm = dm
        if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
            # spin-traced DM for UHF or ROHF
            dm = dm[0] + dm[1]

        phi = make_phi(pcmobj, dm, r_vdw, ui)
        L_X = numpy.linalg.solve(Lmat, phi.ravel()).reshape(natm,-1)
        psi, vmat = make_psi_vmat(pcmobj, dm, r_vdw, ui, pcmobj.grids, ylm_1sph,
                                  cached_pol, L_X, Lmat)[:2]
        dielectric = pcmobj.eps
        if dielectric > 0:
            f_epsilon = (dielectric-1.)/dielectric
        else:
            f_epsilon = 1
        pcmobj.epcm = .5 * f_epsilon * numpy.einsum('jx,jx', psi, L_X)
        pcmobj.vpcm = .5 * f_epsilon * vmat
        return pcmobj.epcm, pcmobj.vpcm
    return gen_vind

def energy(pcmobj, dm):
    '''
    ddCOSMO energy
    Es = 1/2 f(eps) \int rho(r) W(r) dr
    '''
    epcm = gen_ddcosmo_solver(pcmobj, pcmobj.verbose)(dm)[0]
    return epcm

def get_atomic_radii(pcmobj):
    mol = pcmobj.mol
    vdw_radii = pcmobj.radii_table
    atom_radii = pcmobj.atom_radii

    atom_symb = [mol.atom_symbol(i) for i in range(mol.natm)]
    r_vdw = [vdw_radii[gto.charge(x)] for x in atom_symb]
    if atom_radii is not None:
        for i in range(mol.natm):
            if atom_symb[i] in atom_radii:
                r_vdw[i] = atom_radii[atom_symb[i]]
    return numpy.asarray(r_vdw)


def regularize_xt(t, eta):
    xt = numpy.zeros_like(t)
    inner = t <= 1-eta
    on_shell = (1-eta < t) & (t < 1)
    xt[inner] = 1
    ti = t[on_shell]
# JCTC, 9, 3637
    xt[on_shell] = 1./eta**5 * (1-ti)**3 * (6*ti**2 + (15*eta-12)*ti
                                            + 10*eta**2 - 15*eta + 6)
# JCP, 139, 054111
#    xt[on_shell] = 1./eta**4 * (1-ti)**2 * (ti-1+2*eta)**2
    return xt

def make_grids_one_sphere(lebedev_order):
    ngrid_1sph = gen_grid.LEBEDEV_ORDER[lebedev_order]
    leb_grid = numpy.empty((ngrid_1sph,4))
    gen_grid.libdft.MakeAngularGrid(leb_grid.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(ngrid_1sph))
    coords_1sph = leb_grid[:,:3]
    # Note the Lebedev angular grids are normalized to 1 in pyscf
    weights_1sph = 4*numpy.pi * leb_grid[:,3]
    return coords_1sph, weights_1sph

def make_L(pcmobj, r_vdw, ylm_1sph, fi):
    # See JCTC, 9, 3637, Eq (18)
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    eta = pcmobj.eta
    nlm = (lmax+1)**2

    coords_1sph, weights_1sph = make_grids_one_sphere(pcmobj.lebedev_order)
    ngrid_1sph = weights_1sph.size
    atom_coords = mol.atom_coords()
    ylm_1sph = ylm_1sph.reshape(nlm,ngrid_1sph)

# JCP, 141, 184108 Eq (9), (12) is incorrect
# L_diag = <lm|(1/|s-s'|)|l'm'>
# Using Laplace expansion for electrostatic potential 1/r
# L_diag = 4pi/(2l+1)/|s| <lm|l'm'>
    L_diag = numpy.zeros((natm,nlm))
    p1 = 0
    for l in range(lmax+1):
        p0, p1 = p1, p1 + (l*2+1)
        L_diag[:,p0:p1] = 4*numpy.pi/(l*2+1)
    L_diag *= 1./r_vdw.reshape(-1,1)
    Lmat = numpy.diag(L_diag.ravel()).reshape(natm,nlm,natm,nlm)

    for ja in range(natm):
        # scale the weight, precontract d_nj and w_n
        # see JCTC 9, 3637, Eq (16) - (18)
        # Note all values are scaled by 1/r_vdw to make the formulas
        # consistent to Psi in JCP, 141, 184108
        part_weights = weights_1sph.copy()
        part_weights[fi[ja]>1] /= fi[ja,fi[ja]>1]
        for ka in atoms_with_vdw_overlap(ja, atom_coords, r_vdw):
            vjk = r_vdw[ja] * coords_1sph + atom_coords[ja] - atom_coords[ka]
            tjk = lib.norm(vjk, axis=1) / r_vdw[ka]
            wjk = pcmobj.regularize_xt(tjk, eta, r_vdw[ka])
            wjk *= part_weights
            pol = sph.multipoles(vjk, lmax)
            p1 = 0
            for l in range(lmax+1):
                fac = 4*numpy.pi/(l*2+1) / r_vdw[ka]**(l+1)
                p0, p1 = p1, p1 + (l*2+1)
                a = numpy.einsum('xn,n,mn->xm', ylm_1sph, wjk, pol[l])
                Lmat[ja,:,ka,p0:p1] += -fac * a
    return Lmat

def make_fi(pcmobj, r_vdw):
    coords_1sph, weights_1sph = make_grids_one_sphere(pcmobj.lebedev_order)
    mol = pcmobj.mol
    eta = pcmobj.eta
    natm = mol.natm
    atom_coords = mol.atom_coords()
    ngrid_1sph = coords_1sph.shape[0]
    fi = numpy.zeros((natm,ngrid_1sph))
    for ia in range(natm):
        for ja in atoms_with_vdw_overlap(ia, atom_coords, r_vdw):
            v = r_vdw[ia]*coords_1sph + atom_coords[ia] - atom_coords[ja]
            rv = lib.norm(v, axis=1)
            t = rv / r_vdw[ja]
            xt = pcmobj.regularize_xt(t, eta, r_vdw[ja])
            fi[ia] += xt
    fi[fi < 1e-20] = 0
    return fi

def make_phi(pcmobj, dm, r_vdw, ui):
    mol = pcmobj.mol
    natm = mol.natm
    coords_1sph, weights_1sph = make_grids_one_sphere(pcmobj.lebedev_order)
    ngrid_1sph = coords_1sph.shape[0]

    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]
    tril_dm = lib.pack_tril(dm+dm.T)
    nao = dm.shape[0]
    diagidx = numpy.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    tril_dm[diagidx] *= .5

    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()

    extern_point_idx = ui > 0
    cav_coords = (atom_coords.reshape(natm,1,3)
                  + numpy.einsum('r,gx->rgx', r_vdw, coords_1sph))

    v_phi = numpy.empty((natm,ngrid_1sph))
    for ia in range(natm):
# Note (-) sign is not applied to atom_charges, because (-) is explicitly
# included in rhs and L matrix
        d_rs = atom_coords.reshape(-1,1,3) - cav_coords[ia]
        v_phi[ia] = numpy.einsum('z,zp->p', atom_charges, 1./lib.norm(d_rs,axis=2))

    max_memory = pcmobj.max_memory - lib.current_memory()[0]
    blksize = int(max(max_memory*1e6/8/nao**2, 400))

    cav_coords = cav_coords[extern_point_idx]
    v_phi_e = numpy.empty(cav_coords.shape[0])
    int3c2e = mol._add_suffix('int3c2e')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas,
                                         mol._env, int3c2e)
    for i0, i1 in lib.prange(0, cav_coords.shape[0], blksize):
        fakemol = gto.fakemol_for_charges(cav_coords[i0:i1])
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s2ij',
                                cintopt=cintopt)
        v_phi_e[i0:i1] = numpy.einsum('x,xk->k', tril_dm, v_nj)
    v_phi[extern_point_idx] -= v_phi_e

    ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, pcmobj.lmax, True))
    phi = -numpy.einsum('n,xn,jn,jn->jx', weights_1sph, ylm_1sph, ui, v_phi)
    return phi

def make_psi_vmat(pcmobj, dm, r_vdw, ui, grids, ylm_1sph, cached_pol, L_X, L):
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    nlm = (lmax+1)**2

    i1 = 0
    scaled_weights = numpy.empty(grids.weights.size)
    for ia in range(natm):
        fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
        i0, i1 = i1, i1 + fak_pol[0].shape[1]
        eta_nj = 0
        p1 = 0
        for l in range(lmax+1):
            fac = 4*numpy.pi/(l*2+1)
            p0, p1 = p1, p1 + (l*2+1)
            eta_nj += fac * numpy.einsum('mn,m->n', fak_pol[l], L_X[ia,p0:p1])
        scaled_weights[i0:i1] = eta_nj * grids.weights[i0:i1]

    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]
    ni = numint.NumInt()
    max_memory = pcmobj.max_memory - lib.current_memory()[0]
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    den = numpy.empty(grids.weights.size)
    vmat = numpy.zeros((nao,nao))
    p1 = 0
    aow = None
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, 0, max_memory):
        p0, p1 = p1, p1 + weight.size
        den[p0:p1] = weight * make_rho(0, ao, mask, 'LDA')
        aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
        aow = numpy.einsum('pi,p->pi', ao, scaled_weights[p0:p1], out=aow)
        vmat -= numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    ao = aow = scaled_weights = None

    nelec_leak = 0
    psi = numpy.empty((natm,nlm))
    i1 = 0
    for ia in range(natm):
        fak_pol, leak_idx = cached_pol[mol.atom_symbol(ia)]
        i0, i1 = i1, i1 + fak_pol[0].shape[1]
        nelec_leak += den[i0:i1][leak_idx].sum()
        p1 = 0
        for l in range(lmax+1):
            fac = 4*numpy.pi/(l*2+1)
            p0, p1 = p1, p1 + (l*2+1)
            psi[ia,p0:p1] = -fac * numpy.einsum('n,mn->m', den[i0:i1], fak_pol[l])
# Contribution of nuclear charge to the total density
# The factor numpy.sqrt(4*numpy.pi) is due to the product of 4*pi * Y_0^0
        psi[ia,0] += numpy.sqrt(4*numpy.pi)/r_vdw[ia] * mol.atom_charge(ia)
    logger.debug(pcmobj, 'electron leak %f', nelec_leak)

    # <Psi, L^{-1}g> -> Psi = SL the adjoint equation to LX = g
    L_S = numpy.linalg.solve(L.T.reshape(natm*nlm,-1), psi.ravel()).reshape(natm,-1)
    coords_1sph, weights_1sph = make_grids_one_sphere(pcmobj.lebedev_order)
    # JCP, 141, 184108, Eq (39)
    xi_jn = numpy.einsum('n,jn,xn,jx->jn', weights_1sph, ui, ylm_1sph, L_S)
    extern_point_idx = ui > 0
    cav_coords = (mol.atom_coords().reshape(natm,1,3)
                  + numpy.einsum('r,gx->rgx', r_vdw, coords_1sph))
    cav_coords = cav_coords[extern_point_idx]
    xi_jn = xi_jn[extern_point_idx]

    max_memory = pcmobj.max_memory - lib.current_memory()[0]
    blksize = int(max(max_memory*1e6/8/nao**2, 400))

    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas,
                                         mol._env, 'int3c2e')
    vmat_tril = 0
    for i0, i1 in lib.prange(0, xi_jn.size, blksize):
        fakemol = gto.fakemol_for_charges(cav_coords[i0:i1])
        v_nj = df.incore.aux_e2(mol, fakemol, intor='int3c2e', aosym='s2ij',
                                cintopt=cintopt)
        vmat_tril += numpy.einsum('xn,n->x', v_nj, xi_jn[i0:i1])
    vmat += lib.unpack_tril(vmat_tril)
    return psi, vmat, L_S

def cache_fake_multipoles(grids, r_vdw, lmax):
# For each type of atoms, cache the product of last two terms in
# JCP, 141, 184108, Eq (31):
# x_{<}^{l} / x_{>}^{l+1} Y_l^m
    mol = grids.mol
    atom_grids_tab = grids.gen_atomic_grids(mol)
    r_vdw_type = {}
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb not in r_vdw_type:
            r_vdw_type[symb] = r_vdw[ia]

    cached_pol = {}
    for symb in atom_grids_tab:
        x_nj, w = atom_grids_tab[symb]
        r = lib.norm(x_nj, axis=1)
        # Different equations are used in JCTC, 9, 3637. r*Ys (the fake_pole)
        # is computed as r^l/r_vdw. "leak_idx" is not needed.
        # Here, the implementation is based on JCP, 141, 184108
        leak_idx = r > r_vdw_type[symb]

        pol = sph.multipoles(x_nj, lmax)
        fak_pol = []
        for l in range(lmax+1):
            # x_{<}^{l} / x_{>}^{l+1} Y_l^m  in JCP, 141, 184108, Eq (31)
            #:Ys = sph.real_sph_vec(x_nj/r.reshape(-1,1), lmax, True)
            #:rr = numpy.zeros_like(r)
            #:rr[r<=r_vdw[ia]] = r[r<=r_vdw[ia]]**l / r_vdw[ia]**(l+1)
            #:rr[r> r_vdw[ia]] = r_vdw[ia]**l / r[r>r_vdw[ia]]**(l+1)
            #:xx_ylm = numpy.einsum('n,mn->mn', rr, Ys[l])
            xx_ylm = pol[l] * (1./r_vdw_type[symb]**(l+1))
            # The line below is not needed for JCTC, 9, 3637
            xx_ylm[:,leak_idx] *= (r_vdw_type[symb]/r[leak_idx])**(2*l+1)
            fak_pol.append(xx_ylm)
        cached_pol[symb] = (fak_pol, leak_idx)
    return cached_pol

def atoms_with_vdw_overlap(atm_id, atom_coords, r_vdw):
    atm_dist = atom_coords - atom_coords[atm_id]
    atm_dist = numpy.einsum('pi,pi->p', atm_dist, atm_dist)
    atm_dist[atm_id] = 1e200
    vdw_sum = r_vdw + r_vdw[atm_id]
    atoms_nearby = numpy.where(atm_dist < vdw_sum**2)[0]
    return atoms_nearby

class DDCOSMO(lib.StreamObject):
    def __init__(self, mol):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory

        #self.radii_table = radii.VDW
        self.radii_table = radii.UFF*1.1
        #self.radii_table = radii.MM3
        self.atom_radii = None
        self.lebedev_order = 17
        self.lmax = 6  # max angular momentum of spherical harmonics basis
        self.eta = .1  # regularization parameter
        self.eps = 78.3553
        self.grids = gen_grid.Grids(mol)

        # The maximum iterations and convergence tolerance to update solvent
        # effects in CASCI, CC, MP, CI, ... methods 
        self.max_cycle = 20
        self.conv_tol = 1e-7
        self.state_id = 0

        # Set frozen to enable/disable the frozen ddCOSMO solvent potential.
        # If frozen is set, _dm (density matrix) needs to be specified to
        # generate the potential.
        self.frozen = False

##################################################
# don't modify the following attributes, they are not input options
        # epcm (the dielectric correction) and vpcm (the additional
        # potential) are updated during the SCF iterations
        self.epcm = None
        self.vpcm = None
        self._dm = None

        # _solver_ is a cached function returned by self.as_solver() to reduce
        # the overhead of initialization. It should be cleared whenever the
        # solvent parameters or the integration grids were changed.
        self._solver_ = None

        self._keys = set(self.__dict__.keys())

    @property
    def dm(self):
        '''Density matrix to generate the frozen ddCOSMO solvent potential.'''
        return self._dm
    @dm.setter
    def dm(self, dm):
        '''Set dm to enable/disable the frozen ddCOSMO solvent potential.
        Setting dm to None will disable the frozen potental, i.e. the
        potential will be response to the change of the density during SCF
        iterations.
        '''
        if isinstance(dm, numpy.ndarray):
            self._dm = dm
            self.epcm, self.vpcm = self.kernel(dm)
        else:
            self.epcm = self.vpcm = self._dm = None

    def __setattr__(self, key, val):
        if key in ('radii_table', 'atom_radii', 'lebedev_order', 'lmax',
                   'eta', 'eps', 'grids'):
            self._solver_ = None
        super(DDCOSMO, self).__setattr__(key, val)

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'lebedev_order = %s (%d grids per sphere)',
                    self.lebedev_order, gen_grid.LEBEDEV_ORDER[self.lebedev_order])
        logger.info(self, 'lmax = %s'         , self.lmax)
        logger.info(self, 'eta = %s'          , self.eta)
        logger.info(self, 'eps = %s'          , self.eps)
        logger.debug2(self, 'radii_table %s', self.radii_table)
        if self.atom_radii:
            logger.info(self, 'User specified atomic radii %s', str(self.atom_radii))
        self.grids.dump_flags(verbose)
        return self

    def kernel(self, dm):
        '''A single shot solvent effects for given density matrix.
        '''
        if (self._solver_ is None or
# If self.grids.coords is None, it is very likely caused by the updates of the
# "grids" parameters. The COSMO solver should be updated to adapt the new
# integral grids.
            self.grids.coords is None):
            self._solver_ = self.as_solver()

        epcm, vpcm = self._solver_(dm)
        return epcm, vpcm

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
        self._solver_ = None
        self.grids.reset(mol)
        return self

    energy = energy
    gen_solver = as_solver = gen_ddcosmo_solver
    get_atomic_radii = get_atomic_radii

    def regularize_xt(self, t, eta, scale=1):
        # scale = eta*scale, is it correct?
        return regularize_xt(t, eta*scale)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import mcscf
    from pyscf import cc
    mol = gto.M(atom='H 0 0 0; H 0 1 1.2; H 1. .1 0; H .5 .5 1')
    natm = mol.natm
    r_vdw = [radii.VDW[gto.charge(mol.atom_symbol(i))]
             for i in range(natm)]
    r_vdw = numpy.asarray(r_vdw)
    pcmobj = DDCOSMO(mol)
    pcmobj.regularize_xt = lambda t, eta, scale: regularize_xt(t, eta)
    pcmobj.lebedev_order = 7
    pcmobj.lmax = 6
    pcmobj.eta = 0.1
    nlm = (pcmobj.lmax+1)**2
    coords_1sph, weights_1sph = make_grids_one_sphere(pcmobj.lebedev_order)
    fi = make_fi(pcmobj, r_vdw)
    ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, pcmobj.lmax, True))
    L = make_L(pcmobj, r_vdw, ylm_1sph, fi)
    print(lib.finger(L) - 6.2823493771037473)

    mol = gto.Mole()
    mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                   H                 -0.00000000   -0.84695236    0.59109389
                   H                 -0.00000000    0.89830571    0.52404783 '''
    mol.basis = '3-21g' #cc-pvdz'
    mol.build()
    cm = DDCOSMO(mol)
    cm.verbose = 4
    mf = ddcosmo_for_scf(scf.RHF(mol), cm)#.newton()
    mf.verbose = 4
    print(mf.kernel() - -75.570364368059)
    cm.verbose = 3
    e = ddcosmo_for_casci(mcscf.CASCI(mf, 4, 4)).kernel()[0]
    print(e - -75.5743583693215)
    cc_cosmo = ddcosmo_for_post_scf(cc.CCSD(mf)).run()
    print(cc_cosmo.e_tot - -75.70961637250134)

    mol = gto.Mole()
    mol.atom = ''' Fe                 0.00000000    0.00000000   -0.11081188
                   H                 -0.00000000   -0.84695236    0.59109389
                   H                 -0.00000000    0.89830571    0.52404783 '''
    mol.basis = '3-21g' #cc-pvdz'
    mol.build()
    cm = DDCOSMO(mol)
    cm.eps = -1
    cm.verbose = 4
    mf = ddcosmo_for_scf(scf.ROHF(mol), cm).newton()
    mf.verbose=4
    mf.kernel()
