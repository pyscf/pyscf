#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0

'''
UHF-orbital UCASCI analytical nuclear gradients
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.grad import uhf as uhf_grad
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import casci as casci_grad
from pyscf.grad import uccsd as uccsd_grad
from pyscf import ao2mo
from pyscf.mcscf.addons import StateAverageMCSCFSolver

RANGE_TYPE = range


def _check_supported_orbital_source(mc):
    if getattr(mc, '_scf_df_source', False):
        raise NotImplementedError(
            'UCASCI gradients with density-fitted UHF orbitals require '
            'DF-UHF orbital-response terms')
    if getattr(mc._scf, 'with_df', None):
        raise NotImplementedError(
            'UCASCI gradients with density-fitted UHF orbitals require '
            'DF-UHF orbital-response terms')
    if isinstance(mc._scf, hf.KohnShamDFT):
        raise NotImplementedError(
            'UCASCI gradients with KS orbitals require the UKS/CPKS '
            'orbital response; use pyscf.grad.ukscasci')


# Present the UCASCI object with the attributes expected by grad.uccsd: the
# CAS active space is the CC active space, and inactive/external orbitals are
# hidden through the frozen mask while the temporary occupations define the
# artificial UHF reference used by the repacked RDM intermediates.
def _uccsd_env(mc, mo_coeff):
    ncorea, ncoreb = mc.ncore
    neleca, nelecb = mc.nelecas
    nmoa = mo_coeff[0].shape[1]
    nmob = mo_coeff[1].shape[1]
    mo_occ = (numpy.zeros(nmoa), numpy.zeros(nmob))
    mo_occ[0][:ncorea] = 1
    mo_occ[0][ncorea:ncorea+neleca] = 1
    mo_occ[1][:ncoreb] = 1
    mo_occ[1][ncoreb:ncoreb+nelecb] = 1
    masks = (numpy.zeros(nmoa, dtype=bool), numpy.zeros(nmob, dtype=bool))
    masks[0][ncorea:ncorea+mc.ncas] = True
    masks[1][ncoreb:ncoreb+mc.ncas] = True
    frozen = [numpy.where(~mask)[0] for mask in masks]
    frozen = None if all(x.size == 0 for x in frozen) else frozen
    return lib.temporary_env(mc, mo_coeff=mo_coeff, mo_occ=mo_occ,
                             frozen=frozen,
                             get_frozen_mask=lambda *args: masks)


# The UHF-CASCI gradient can be written in the same AO contraction and UCPHF
# response form used by UCCSD after the CASCI RDMs are expressed relative to a
# single UHF determinant.  The determinant is artificial: core orbitals are
# occupied and the first nelecas active orbitals are used as the alpha/beta
# reference.  This routine removes those reference 1- and 2-particle density
# contributions from the exact active-space CASCI RDMs, then repacks the
# reference-subtracted CASCI RDM blocks into the d1/d2 intermediate layout
# expected by grad.uccsd.  grad_elec passes these intermediates to the UCCSD
# gradient driver, which then supplies the shared MO-to-AO 2-RDM
# transformation, generalized Fock contractions, and UCPHF orbital-response
# terms.
def _casci_active_rdm_to_uccsd_intermediates(casdm1s, casdm2s, nelecas):
    nocca, noccb = nelecas
    casdm1a, casdm1b = casdm1s
    casdm2aa, casdm2ab, casdm2bb = casdm2s
    ncas = casdm1a.shape[0]
    nvira = ncas - nocca
    nvirb = ncas - noccb
    oa = slice(0, nocca)
    va = slice(nocca, ncas)
    ob = slice(0, noccb)
    vb = slice(noccb, ncas)

    dm1a = casdm1a.copy()
    dm1b = casdm1b.copy()
    dm1a[numpy.diag_indices(nocca)] -= 1
    dm1b[numpy.diag_indices(noccb)] -= 1
    d1 = ((dm1a[oa,oa], dm1b[ob,ob]),
          (dm1a[oa,va], dm1b[ob,vb]),
          (dm1a[va,oa], dm1b[vb,ob]),
          (dm1a[va,va], dm1b[vb,vb]))

    # uccsd_rdm._make_rdm2 builds an internal tensor and returns
    # internal.transpose(1,0,3,2).  Undo that final transpose, then remove the
    # UHF-reference and 1-pdm terms that _make_rdm2 adds when with_dm1=True.
    dm2aa = casdm2aa.transpose(1,0,3,2).copy()
    dm2ab = casdm2ab.transpose(1,0,3,2).copy()
    dm2bb = casdm2bb.transpose(1,0,3,2).copy()

    for i in range(nocca):
        dm2aa[i,i,:,:] -= dm1a
        dm2aa[:,:,i,i] -= dm1a
        dm2aa[:,i,i,:] += dm1a
        dm2aa[i,:,:,i] += dm1a.T
        dm2ab[i,i,:,:] -= dm1b
    for i in range(noccb):
        dm2bb[i,i,:,:] -= dm1b
        dm2bb[:,:,i,i] -= dm1b
        dm2bb[:,i,i,:] += dm1b
        dm2bb[i,:,:,i] += dm1b.T
        dm2ab[:,:,i,i] -= dm1a

    for i in range(nocca):
        for j in range(nocca):
            dm2aa[i,i,j,j] -= 1
            dm2aa[i,j,j,i] += 1
    for i in range(noccb):
        for j in range(noccb):
            dm2bb[i,i,j,j] -= 1
            dm2bb[i,j,j,i] += 1
    for i in range(nocca):
        for j in range(noccb):
            dm2ab[i,i,j,j] -= 1

    dovov = dm2aa[oa,va,oa,va]
    dovvo = dm2aa[oa,va,va,oa]
    doovv = dm2aa[oa,oa,va,va]
    dvvvv = dm2aa[va,va,va,va]
    dvvvv = ao2mo.restore(4, dvvvv + dvvvv.transpose(1,0,2,3),
                           nvira) * .5
    doooo = dm2aa[oa,oa,oa,oa]
    dovvv = dm2aa[oa,va,va,va]
    dooov = dm2aa[oa,oa,oa,va]

    dOVOV = dm2bb[ob,vb,ob,vb]
    dOVVO = dm2bb[ob,vb,vb,ob]
    dOOVV = dm2bb[ob,ob,vb,vb]
    dVVVV = dm2bb[vb,vb,vb,vb]
    dVVVV = ao2mo.restore(4, dVVVV + dVVVV.transpose(1,0,2,3),
                           nvirb) * .5
    dOOOO = dm2bb[ob,ob,ob,ob]
    dOVVV = dm2bb[ob,vb,vb,vb]
    dOOOV = dm2bb[ob,ob,ob,vb]

    dovOV = dm2ab[oa,va,ob,vb]
    dooOO = dm2ab[oa,oa,ob,ob]
    dvvVV = dm2ab[va,va,vb,vb]
    dvvVV = dvvVV + dvvVV.transpose(1,0,2,3)
    dvvVV = lib.pack_tril(dvvVV[numpy.tril_indices(nvira)]) * .5
    dovVO = dm2ab[oa,va,vb,ob]
    dooOV = dm2ab[oa,oa,ob,vb]
    dovVV = dm2ab[oa,va,vb,vb]
    dooVV = dm2ab[oa,oa,vb,vb]

    dOVvo = dm2ab[va,oa,ob,vb].transpose(3,2,1,0).conj()
    dOOov = dm2ab[oa,va,ob,ob].transpose(2,3,0,1)
    dOVvv = dm2ab[va,va,ob,vb].transpose(2,3,0,1)
    dOOvv = dm2ab[va,va,ob,ob].transpose(2,3,0,1)

    d2 = ((dovov, dovOV, None, dOVOV),
          (dvvvv, dvvVV, None, dVVVV),
          (doooo, dooOO, None, dOOOO),
          (doovv, dooVV, dOOvv, dOOVV),
          (dovvo, dovVO, dOVvo, dOVVO),
          (None, None, None, None),
          (dovvv, dovVV, dOVvv, dOVVV),
          (dooov, dooOV, dOOov, dOOOV))
    return d1, d2


def grad_elec(mc_grad, mo_coeff=None, ci=None, atmlst=None, verbose=None):
    mc = mc_grad.base
    _check_supported_orbital_source(mc)
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci

    ncas = mc.ncas

    casdm1s, casdm2s = mc.fcisolver.make_rdm12s(ci, ncas, mc.nelecas)
    d1, d2 = _casci_active_rdm_to_uccsd_intermediates(
        casdm1s, casdm2s, mc.nelecas)
    t1 = t2 = l1 = l2 = ()
    with _uccsd_env(mc, mo_coeff):
        return uccsd_grad.grad_elec(mc_grad, t1, t2, l1, l2, None, atmlst,
                                    d1, d2, verbose)


class Gradients(uhf_grad.Gradients):
    '''Non-relativistic UHF-CASCI gradients'''

    _keys = {'state'}

    grad_elec = grad_elec

    def __init__(self, mc):
        self.state = 0
        uhf_grad.Gradients.__init__(self, mc)

    def kernel(self, mo_coeff=None, ci=None, atmlst=None,
               state=None, verbose=None):
        log = logger.new_logger(self, verbose)
        if ci is None:
            if self.base.ci is None:
                self.base.ci.run()
            ci = self.base.ci
        if (isinstance(ci, (list, tuple, RANGE_TYPE)) and
            not isinstance(self.base, StateAverageMCSCFSolver)):
            if state is None:
                state = self.state
            else:
                self.state = state
            ci = ci[state]
            log.info('Multiple roots are found in UCASCI solver. '
                     'Nuclear gradients of root %d are computed.', state)
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()
        de = self.grad_elec(mo_coeff, ci, atmlst, log)
        self.de = de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de

    def hcore_generator(self, mol=None):
        mf_grad = self.base._scf.nuc_grad_method()
        return mf_grad.hcore_generator(mol)

    def grad_nuc(self, mol=None, atmlst=None):
        mf_grad = self.base._scf.nuc_grad_method()
        return mf_grad.grad_nuc(mol, atmlst)

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------- %s gradients ----------',
                        self.base.__class__.__name__)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    as_scanner = casci_grad.as_scanner

    to_gpu = lib.to_gpu


Grad = Gradients

from pyscf import mcscf
def _ucasci_grad_method(mc, *args, **kwargs):
    # Keep mc.Gradients() source-aware, matching UCASCI.nuc_grad_method().
    if isinstance(mc._scf, hf.KohnShamDFT):
        from pyscf.grad import ukscasci
        return ukscasci.Gradients(mc, *args, **kwargs)
    return Gradients(mc, *args, **kwargs)

mcscf.ucasci.UCASCI.Gradients = _ucasci_grad_method
