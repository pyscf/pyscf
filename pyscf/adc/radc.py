# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
# Author: Abdelrahman Ahmed <>
#         Samragni Banerjee <samragnibanerjee4@gmail.com>
#         James Serna <jamcar456@gmail.com>
#         Terrence Stahl <terrencestahl1@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

'''
Restricted algebraic diagrammatic construction
'''
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import radc_ao2mo
from pyscf.adc import radc_amplitudes
from pyscf import __config__
from pyscf import df
from pyscf.mp import mp2


# Excited-state kernel
def kernel(adc, nroots=1, guess=None, eris=None, verbose=None):

    adc.method = adc.method.lower()
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)
    if adc.verbose >= logger.WARN:
        adc.check_sanity()
    adc.dump_flags()

    if eris is None:
        eris = adc.transform_integrals()

    if adc.approx_trans_moments:
        if adc.method in ("adc(2)", "adc(2)-x"):
            logger.warn(
                adc,
                "Approximations for transition moments are requested...\n"
                + adc.method
                + " transition properties will neglect second-order amplitudes...")
        else:
            logger.warn(
                adc,
                "Approximations for transition moments are requested...\n"
                + adc.method
                + " transition properties will neglect third-order amplitudes...")

    imds = adc.get_imds(eris)
    matvec, diag = adc.gen_matvec(imds, eris)

    guess = adc.get_init_guess(nroots, diag, ascending = True)

    conv, adc.E, U = lib.linalg_helper.davidson_nosym1(
        lambda xs : [matvec(x) for x in xs],
        guess, diag, nroots=nroots, verbose=log, tol=adc.conv_tol, max_memory=adc.max_memory,
        max_cycle=adc.max_cycle, max_space=adc.max_space, tol_residual=adc.tol_residual)

    adc.U = np.array(U).T.copy()

    if adc.compute_properties and adc.method_type != "ee":
        adc.P,adc.X = adc.get_properties(nroots)
    else:
        adc.P = None
        adc.X = None

    nfalse = np.shape(conv)[0] - np.sum(conv)

    spin_mult = None
    if adc.method_type in ("ip", "ea"):
        spin_mult = "doublet"
    else:
        spin_mult = "singlet"

    header = ("\n*************************************************************"
              "\n        ADC calculation summary (" + spin_mult + " states only)"
              "\n*************************************************************")
    logger.info(adc, header)

    for n in range(nroots):
        print_string = ('%s root %d  |  Energy (Eh) = %14.10f  |  Energy (eV) = %12.8f  ' %
                        (adc.method, n, adc.E[n], adc.E[n]*27.2114))
        if adc.compute_properties and adc.method_type != "ee":
            print_string += ("|  Spec. factor = %10.8f  " % adc.P[n])
        print_string += ("|  conv = %s" % conv[n])
        logger.info(adc, print_string)

    if nfalse >= 1:
        logger.warn(adc, "Davidson iterations for " + str(nfalse) + " root(s) did not converge!!!")

    log.timer('ADC', *cput0)

    return adc.E, adc.U, adc.P, adc.X


def make_ref_rdm1(adc):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    t1 = adc.t1
    t2 = adc.t2
    t2_ce = t1[0][:]
    t1_ccee = t2[0][:]

    ######################
    einsum_type = True
    nocc = adc._nocc
    nvir = adc._nvir

    nmo = nocc + nvir

    OPDM = np.zeros((nmo,nmo))

    ####### ADC(2) SPIN ADAPTED REF OPDM with SQA ################
    ### OCC-OCC ###
    OPDM[:nocc, :nocc] += lib.einsum('IJ->IJ', np.identity(nocc), optimize = einsum_type).copy()
    OPDM[:nocc, :nocc] -= 2 * lib.einsum('Iiab,Jiab->IJ', t1_ccee, t1_ccee, optimize = einsum_type)
    OPDM[:nocc, :nocc] += lib.einsum('Iiab,Jiba->IJ', t1_ccee, t1_ccee, optimize = einsum_type)

    ### OCC-VIR ###
    OPDM[:nocc, nocc:] += lib.einsum('IA->IA', t2_ce, optimize = einsum_type).copy()

    ### VIR-OCC ###
    OPDM[nocc:, :nocc] += lib.einsum('IA->AI', t2_ce, optimize = einsum_type).copy()

    ### VIR-VIR ###
    OPDM[nocc:, nocc:] += 2 * lib.einsum('ijAa,ijBa->AB', t1_ccee, t1_ccee, optimize = einsum_type)
    OPDM[nocc:, nocc:] -= lib.einsum('ijAa,jiBa->AB', t1_ccee, t1_ccee, optimize = einsum_type)

    ####### ADC(3) SPIN ADAPTED REF OPDM WITH SQA ################
    if adc.method == "adc(3)":
        t3_ce = adc.t1[1][:]
        t2_ccee = t2[1][:]

        #### OCC-OCC ###
        OPDM[:nocc, :nocc] -= 2 * lib.einsum('Iiab,Jiab->IJ',
                                             t1_ccee, t2_ccee, optimize = einsum_type)
        OPDM[:nocc, :nocc] += lib.einsum('Iiab,Jiba->IJ', t1_ccee, t2_ccee, optimize = einsum_type)
        OPDM[:nocc, :nocc] -= 2 * lib.einsum('Jiab,Iiab->IJ',
                                             t1_ccee, t2_ccee, optimize = einsum_type)
        OPDM[:nocc, :nocc] += lib.einsum('Jiab,Iiba->IJ', t1_ccee, t2_ccee, optimize = einsum_type)

        ##### OCC-VIR ### ####
        OPDM[:nocc, nocc:]  += lib.einsum('IA->IA', t3_ce, optimize = einsum_type).copy()
        OPDM[:nocc, nocc:] +=  lib.einsum('IiAa,ia->IA', t1_ccee, t2_ce, optimize = einsum_type)
        OPDM[:nocc, nocc:] -= 1/2 * \
            lib.einsum('iIAa,ia->IA', t1_ccee, t2_ce, optimize = einsum_type)
        ###### VIR-OCC ###
        OPDM[nocc:, :nocc]  += lib.einsum('IA->AI', t3_ce, optimize = einsum_type).copy()
        OPDM[nocc:, :nocc]  += lib.einsum('IiAa,ia->AI', t1_ccee, t2_ce, optimize = einsum_type)
        OPDM[nocc:, :nocc]  -= 1/2 * \
            lib.einsum('iIAa,ia->AI', t1_ccee, t2_ce, optimize = einsum_type)

        ##### VIR=VIR ###
        OPDM[nocc:, nocc:] += 2 * lib.einsum('ijAa,ijBa->AB',
                                             t1_ccee, t2_ccee, optimize = einsum_type)
        OPDM[nocc:, nocc:] -= lib.einsum('ijAa,jiBa->AB', t1_ccee, t2_ccee, optimize = einsum_type)
        OPDM[nocc:, nocc:] += 2 * lib.einsum('ijBa,ijAa->AB',
                                             t1_ccee, t2_ccee, optimize = einsum_type)
        OPDM[nocc:, nocc:] -= lib.einsum('ijBa,jiAa->AB', t1_ccee, t2_ccee, optimize = einsum_type)

    return 2 * OPDM

def get_fno_ref(myadc,nroots,ref_state):
    adc2_ref = RADC(myadc._scf,myadc.frozen).set(verbose = 0,method_type = myadc.method_type,with_df = myadc.with_df,if_naf = myadc.if_naf,thresh_naf = myadc.thresh_naf)
    myadc.e2_ref,myadc.v2_ref,myadc.p2_ref,myadc.x2_ref = adc2_ref.kernel(nroots)
    rdm1_gs = adc2_ref.make_ref_rdm1()
    if ref_state is not None:
        rdm1_ref = adc2_ref.make_rdm1()[ref_state - 1]
        myadc.rdm1_ss = rdm1_ref + rdm1_gs
    else:
        myadc.rdm1_ss = rdm1_gs

def make_fno(myadc, rdm1_ss, mf, thresh):
    import numpy
    from pyscf.mp import mp2
    nocc = myadc._nocc
    masks = mp2._mo_splitter(myadc)
    
    n,V = numpy.linalg.eigh(rdm1_ss[nocc:,nocc:])
    idx = numpy.argsort(n)[::-1]
    n,V = n[idx], V[:,idx]
    T = n > thresh
    n_fro_vir = numpy.sum(T == 0)
    T = numpy.diag(T)
    V_trunc = V.dot(T)
    n_keep = V_trunc.shape[0]-n_fro_vir

    moeoccfrz0, moeocc, moevir, moevirfrz0 = [mf.mo_energy[m] for m in masks]
    orboccfrz0, orbocc, orbvir, orbvirfrz0 = [mf.mo_coeff[:,m] for m in masks]
    F_can =  numpy.diag(moevir)
    F_na_trunc = V_trunc.T.dot(F_can).dot(V_trunc)
    _,Z_na_trunc = numpy.linalg.eigh(F_na_trunc[:n_keep,:n_keep])
    U_vir_act = orbvir.dot(V_trunc[:,:n_keep]).dot(Z_na_trunc)
    U_vir_fro = orbvir.dot(V_trunc[:,n_keep:]) 
    no_comp = (orboccfrz0,orbocc,U_vir_act,U_vir_fro,orbvirfrz0)
    no_coeff = numpy.hstack(no_comp)
    nocc_loc = numpy.cumsum([0]+[x.shape[1] for x in no_comp]).astype(int)
    no_frozen = numpy.hstack((numpy.arange(nocc_loc[0], nocc_loc[1]),
                              numpy.arange(nocc_loc[3], nocc_loc[5]))).astype(int)
    return no_coeff,no_frozen


class RADC(lib.StreamObject):
    '''Ground state calculations

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
        incore_complete : bool
            Avoid all I/O. Default is False.
        method : string
            nth-order ADC method. Options are : ADC(2), ADC(2)-X, ADC(3). Default is ADC(2).

            >>> mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis = 'ccpvdz')
            >>> mf = scf.RHF(mol).run()
            >>> myadc = adc.RADC(mf).run()

    Saved results

        e_corr : float
            MPn correlation correction
        e_tot : float
            Total energy (HF + correlation)
        t1, t2 :
            T amplitudes t1[i,a], t2[i,j,a,b]  (i,j in occ, a,b in virt)
    '''
    incore_complete = getattr(__config__, 'adc_radc_RADC_incore_complete', False)
    async_io = getattr(__config__, 'adc_radc_RADC_async_io', True)
    blkmin = getattr(__config__, 'adc_radc_RADC_blkmin', 4)
    memorymin = getattr(__config__, 'adc_radc_RADC_memorymin', 2000)

    _keys = {
        'tol_residual','conv_tol', 'e_corr', 'method', 'method_type', 'mo_coeff',
        'mol', 'mo_energy', 'incore_complete',
        'scf_energy', 'e_tot', 't1', 't2', 'frozen', 'chkfile',
        'max_space', 'mo_occ', 'max_cycle', 'imds', 'with_df', 'compute_properties',
        'approx_trans_moments', 'evec_print_tol', 'spec_factor_print_tol',
        'E', 'U', 'P', 'X', 'ncvs', 'dip_mom', 'dip_mom_nuc', 'if_naf', 'thresh_naf', 
        'naux', 'if_heri_eris'
    }

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        import numpy

        if 'dft' in str(mf.__module__):
            raise NotImplementedError('DFT reference for UADC')

        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.max_space = getattr(__config__, 'adc_radc_RADC_max_space', 12)
        self.max_cycle = getattr(__config__, 'adc_radc_RADC_max_cycle', 50)
        self.conv_tol = getattr(__config__, 'adc_radc_RADC_conv_tol', 1e-8)
        self.tol_residual = getattr(__config__, 'adc_radc_RADC_tol_residual', 1e-5)

        self.frozen = frozen
        self.incore_complete = self.incore_complete or self.mol.incore_anyway

        self.mo_occ = mo_occ
        self.e_corr = None
        self.t1 = None
        self.t2 = None
        self.imds = lambda:None
        self._nocc = mf.mol.nelectron//2
        self.mo_coeff = mo_coeff
        self.mo_energy = mf.mo_energy
#NOTE:update_start
        self.naux = None
        self.if_naf = False
        self.thresh_naf = 1e-2
        self.if_heri_eris = False
        self._nmo = None
        #fno modify
        if frozen is None:
            self._nmo = mo_coeff.shape[1]
        elif isinstance(frozen, (int, numpy.integer)):
            self._nmo = mo_coeff.shape[1]-frozen
        elif hasattr(frozen, '__len__'):
            self._nmo = mo_coeff.shape[1]-len(frozen)
        else:
            raise NotImplementedError
        self._nvir = self._nmo - self._nocc
        mask = self.get_frozen_mask()
        self.mo_coeff = mo_coeff[:,mask]
        if self.mo_coeff is self._scf.mo_coeff and self._scf.converged:
            self.mo_energy = self.mo_energy[mask]
            self.scf_energy = self._scf.e_tot
        else:
            dm = self._scf.make_rdm1(mo_coeff, self.mo_occ)
            vhf = self._scf.get_veff(self.mol, dm)
            fockao = self._scf.get_fock(vhf=vhf, dm=dm)
            fock = self.mo_coeff.conj().T.dot(fockao).dot(self.mo_coeff)
            self.mo_energy = fock.diagonal().real
            self.scf_energy = self._scf.energy_tot(dm=dm, vhf=vhf)
#NOTE:update_end
        self.chkfile = mf.chkfile
        self.method = "adc(2)"
        self.method_type = "ip"
        self.with_df = None
        self.compute_properties = True
        self.approx_trans_moments = False
        self.evec_print_tol = 0.1
        self.spec_factor_print_tol = 0.1
        self.ncvs = None

        self.E = None
        self.U = None
        self.P = None
        self.X = None

#TODO:check if needs complete coefficient
        dip_ints = -self.mol.intor('int1e_r',comp=3)
        dip_mom = np.zeros((dip_ints.shape[0], self._nmo, self._nmo))

        for i in range(dip_ints.shape[0]):
            dip = dip_ints[i,:,:]
            dip_mom[i,:,:] = np.dot(self.mo_coeff.T, np.dot(dip, self.mo_coeff))

        self.dip_mom = dip_mom

        charges = self.mol.atom_charges()
        coords  = self.mol.atom_coords()
        self.dip_mom_nuc = lib.einsum('i,ix->x', charges, coords)

    compute_amplitudes = radc_amplitudes.compute_amplitudes
    compute_energy = radc_amplitudes.compute_energy
    transform_integrals = radc_ao2mo.transform_integrals_incore
    make_ref_rdm1 = make_ref_rdm1
    get_frozen_mask = mp2.get_frozen_mask

    def dump_flags(self, verbose=None):
        logger.info(self, '')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'max_space = %d', self.max_space)
        logger.info(self, 'max_cycle = %d', self.max_cycle)
        logger.info(self, 'conv_tol = %s', self.conv_tol)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        return self

    def dump_flags_gs(self, verbose=None):
        logger.info(self, '')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        return self

    def kernel_gs(self):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)

        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags_gs()

        nmo = self._nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]

        if getattr(self, 'with_df', None) or getattr(self._scf, 'with_df', None):
            if getattr(self, 'with_df', None):
                self.with_df = self.with_df
            else:
                self.with_df = self._scf.with_df

            def df_transform():
                return radc_ao2mo.transform_integrals_df(self)
            self.transform_integrals = df_transform
        elif (self._scf._eri is None or
              (mem_incore+mem_now >= self.max_memory and not self.incore_complete)):
            def outcore_transform():
                return radc_ao2mo.transform_integrals_outcore(self)
            self.transform_integrals = outcore_transform

        eris = self.transform_integrals()

        self.e_corr, self.t1, self.t2 = radc_amplitudes.compute_amplitudes_energy(
            self, eris=eris, verbose=self.verbose)
        self._finalize()

        return self.e_corr, self.t1, self.t2

    def kernel(self, nroots=1, guess=None, eris=None):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)

        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags_gs()

        nmo = self._nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]

        if eris is None:
            if getattr(self, 'with_df', None) or getattr(self._scf, 'with_df', None):
                if getattr(self, 'with_df', None):
                    self.with_df = self.with_df
                else:
                    self.with_df = self._scf.with_df

                def df_transform():
                    return radc_ao2mo.transform_integrals_df(self)
                self.transform_integrals = df_transform
            elif (self._scf._eri is None or
                (mem_incore+mem_now >= self.max_memory and not self.incore_complete)):
                def outcore_transform():
                    return radc_ao2mo.transform_integrals_outcore(self)
                self.transform_integrals = outcore_transform

            eris = self.transform_integrals()

        self.e_corr, self.t1, self.t2 = radc_amplitudes.compute_amplitudes_energy(
            self, eris=eris, verbose=self.verbose)
        self._finalize()

        self.method_type = self.method_type.lower()
        if (self.method_type == "ea"):
            e_exc, v_exc, spec_fac, x, adc_es = self.ea_adc(nroots=nroots, guess=guess, eris=eris)

        elif (self.method_type == "ee"):
            e_exc, v_exc, spec_fac, x, adc_es = self.ee_adc(nroots=nroots, guess=guess, eris=eris)

        elif(self.method_type == "ip"):
            if not isinstance(self.ncvs, type(None)) and self.ncvs > 0:
                e_exc, v_exc, spec_fac, x, adc_es = self.ip_cvs_adc(
                    nroots=nroots, guess=guess, eris=eris)
            else:
                e_exc, v_exc, spec_fac, x, adc_es = self.ip_adc(
                    nroots=nroots, guess=guess, eris=eris)
        else:
            raise NotImplementedError(self.method_type)
        self._adc_es = adc_es
        if self.if_heri_eris:
            if self.if_naf:
                return e_exc, v_exc, spec_fac, x, eris, self.naux
            else:
                eris.vvvv = None
                return e_exc, v_exc, spec_fac, x, eris
        else:
            return e_exc, v_exc, spec_fac, x

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        logger.note(self, 'MP%s correlation energy of reference state (a.u.) = %.8f',
                    self.method[4], self.e_corr)
        return self

    def ea_adc(self, nroots=1, guess=None, eris=None):
        from pyscf.adc import radc_ea
        adc_es = radc_ea.RADCEA(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        return e_exc, v_exc, spec_fac, x, adc_es

    def ee_adc(self, nroots=1, guess=None, eris=None):
        from pyscf.adc import radc_ee
        adc_es = radc_ee.RADCEE(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        return e_exc, v_exc, spec_fac, x, adc_es

    def ip_adc(self, nroots=1, guess=None, eris=None):
        from pyscf.adc import radc_ip
        adc_es = radc_ip.RADCIP(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        return e_exc, v_exc, spec_fac, x, adc_es

    def ip_cvs_adc(self, nroots=1, guess=None, eris=None):
        from pyscf.adc import radc_ip_cvs
        adc_es = radc_ip_cvs.RADCIPCVS(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        return e_exc, v_exc, spec_fac, x, adc_es

    def density_fit(self, auxbasis=None, with_df=None):
        if with_df is None:
            self.with_df = df.DF(self._scf.mol)
            self.with_df.max_memory = self.max_memory
            self.with_df.stdout = self.stdout
            self.with_df.verbose = self.verbose
            if auxbasis is None:
                self.with_df.auxbasis = self._scf.with_df.auxbasis
            else:
                self.with_df.auxbasis = auxbasis
        else:
            self.with_df = with_df
        return self

    def analyze(self):
        self._adc_es.analyze()

    def compute_dyson_mo(self):
        return self._adc_es.compute_dyson_mo()

    def make_rdm1(self):
        return self._adc_es.make_rdm1()

class RFNOADC3(RADC):
    #J. Chem. Phys. 159, 084113 (2023)
    _keys = RADC._keys | {'delta_e','delta_v','delta_p','delta_x',
                          'e2_ref','v2_ref','p2_ref','x2_ref',
                          'rdm1_ss','correction','frozen_core','ref_state'
                          }

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, correction=True):
        import copy
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        self.delta_e = None
        self.delta_v = None
        self.delta_p = None
        self.delta_x = None
        self.method = "adc(3)"
        self.correction = correction
        self.e2_ref = None
        self.v2_ref = None
        self.p2_ref = None
        self.x2_ref = None
        self.rdm1_ss = None
        self.ref_state = None
        self.if_naf = True
        self.frozen_core = copy.deepcopy(self.frozen)

    def compute_correction(self, mf, frozen, nroots, eris):
        adc2_ssfno = RADC(mf, frozen, self.mo_coeff).set(verbose = 0,method_type = self.method_type,\
                                                         with_df = self.with_df,if_naf = self.if_naf,thresh_naf = self.thresh_naf,naux = self.naux)
        e2_ssfno,v2_ssfno,p2_ssfno,x2_ssfno = adc2_ssfno.kernel(nroots, eris = eris)
        self.delta_e = self.e2_ref - e2_ssfno
        #self.delta_v = self.v2_ref - v2_ssfno
        #self.delta_p = self.p2_ref - p2_ssfno
        #self.delta_x = self.x2_ref - x2_ssfno

    def kernel(self, nroots=1, guess=None, eris=None, thresh = 1e-4, ref_state = None):
        import copy
        self.frozen = copy.deepcopy(self.frozen_core)
        self.ref_state = ref_state
        self.naux = None
        if ref_state is None:
            print("Do fnoadc3 calculation")
            self.if_naf = False
        elif isinstance(ref_state, int) and 0<ref_state<=nroots:
            print(f"Do ss-fno adc3 calculation, the specic state is {ref_state}")
            if self.with_df == None:
                self.if_naf = False
        else:
            raise ValueError("ref_state should be an int type and in (0,nroots]")
        
        get_fno_ref(self, nroots, self.ref_state)
        self.mo_coeff,self.frozen = make_fno(self, self.rdm1_ss, self._scf, thresh)
        adc3_ssfno = RADC(self._scf, self.frozen, self.mo_coeff).set(verbose = self.verbose,method_type = self.method_type,method = "adc(3)",\
                                                         with_df = self.with_df,if_naf = self.if_naf,thresh_naf = self.thresh_naf,\
                                                            if_heri_eris = self.if_heri_eris)
        if not self.correction or self.if_heri_eris is False:
            e_exc, v_exc, spec_fac, x = adc3_ssfno.kernel(nroots, guess, eris)
        elif self.if_naf:
            e_exc, v_exc, spec_fac, x, eris, self.naux = adc3_ssfno.kernel(nroots, guess, eris)
            self.compute_correction(self._scf, self.frozen, nroots, eris)
            e_exc = e_exc + self.delta_e
        else:
            e_exc, v_exc, spec_fac, x, eris = adc3_ssfno.kernel(nroots, guess, eris)
            self.compute_correction(self._scf, self.frozen, nroots, eris)
            e_exc = e_exc + self.delta_e
        v_exc = v_exc# + self.delta_v
        spec_fac = spec_fac# + self.delta_p
        x = x# + self.delta_x
        return e_exc, v_exc, spec_fac, x


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf import adc

    r = 1.098
    mol = gto.Mole()
    mol.atom = [
        ['N', (0., 0.    , -r/2   )],
        ['N', (0., 0.    ,  r/2)],]
    mol.basis = {'N':'aug-cc-pvdz'}
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    myadc = adc.ADC(mf)
    ecorr, t_amp1, t_amp2 = myadc.kernel_gs()
    print(ecorr - -0.32201692360512324)

    myadcip = adc.radc_ip.RADCIP(myadc)
    e,v,p = kernel(myadcip,nroots=3)
    print("ADC(2) IP energies")
    print (e[0] - 0.5434389910483670)
    print (e[1] - 0.6240296243595950)
    print (e[2] - 0.6240296243595956)

    print("ADC(2) IP spectroscopic factors")
    print (p[0] - 1.7688097076459075)
    print (p[1] - 1.8192921131700284)
    print (p[2] - 1.8192921131700293)

    myadcea = adc.radc_ea.RADCEA(myadc)
    e,v,p = kernel(myadcea,nroots=3)
    print("ADC(2) EA energies")
    print (e[0] - 0.0961781923822576)
    print (e[1] - 0.1258326916409743)
    print (e[2] - 0.1380779405750178)

    print("ADC(2) EA spectroscopic factors")
    print (p[0] - 1.9832854445007961)
    print (p[1] - 1.9634368668786559)
    print (p[2] - 1.9783719593912672)

    myadc = adc.ADC(mf)
    myadc.method = "adc(3)"
    ecorr, t_amp1, t_amp2 = myadc.kernel_gs()
    print(ecorr - -0.31694173142858517)

    myadcip = adc.radc_ip.RADCIP(myadc)
    e,v,p = kernel(myadcip,nroots=3)
    print("ADC(3) IP energies")
    print (e[0] - 0.5667526829981027)
    print (e[1] - 0.6099995170092525)
    print (e[2] - 0.6099995170092529)

    print("ADC(3) IP spectroscopic factors")
    print (p[0] - 1.8173191958988848)
    print (p[1] - 1.8429224413853840)
    print (p[2] - 1.8429224413853851)

    myadcea = adc.radc_ea.RADCEA(myadc)
    e,v,p = kernel(myadcea,nroots=3)

    print("ADC(3) EA energies")
    print (e[0] - 0.0936790850738445)
    print (e[1] - 0.0983654552141278)
    print (e[2] - 0.1295709313652367)

    print("ADC(3) EA spectroscopic factors")
    print (p[0] - 1.8324175318668088)
    print (p[1] - 1.9840991060607487)
    print (p[2] - 1.9638550014980212)

    myadc.method = "adc(2)-x"
    e,v,p = myadc.kernel(nroots=4)
    print("ADC(2)-x IP energies")
    print (e[0] - 0.5405255360673724)
    print (e[1] - 0.6208026698756577)
    print (e[2] - 0.6208026698756582)
    print (e[3] - 0.6465332771967947)

    myadc.method_type = "ea"
    e,v,p = myadc.kernel(nroots=4)
    print("ADC(2)-x EA energies")
    print (e[0] - 0.0953065329985665)
    print (e[1] - 0.1238833070823509)
    print (e[2] - 0.1365693811939308)
    print (e[3] - 0.1365693811939316)
