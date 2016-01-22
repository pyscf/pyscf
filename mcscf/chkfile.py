#!/usr/bin/env python
#
# Contributor(s):
# - Wirawan Purwanto <wirawan0@gmail.com>
# - Qiming Sun <osirpt.sun@gmail.com>
#
#

import numpy
import h5py
from pyscf.lib.chkfile import load
from pyscf.lib.chkfile import dump, save
from pyscf.lib.chkfile import load_mol, save_mol


def load_mcscf(chkfile):
    return load_mol(chkfile), load(chkfile, 'mcscf')

def dump_mcscf(mc, chkfile=None, key='mcscf',
               e_tot=None, mo_coeff=None, ncore=None, ncas=None,
               mo_occ=None, mo_energy=None, e_cas=None, ci_vector=None):
    '''Save CASCI/CASSCF calculation results or intermediates in chkfile.
    '''
    if chkfile is None: chkfile = mc.chkfile
    if ncore is None: ncore = mc.ncore
    if ncas is None: ncas = mc.ncas
    if e_tot is None: e_tot = mc.e_tot
    if e_cas is None: e_cas = mc.e_cas
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci_vector is None: ci_vector = mc.ci

#TODO: Add UCAS interface
    if mo_occ is None or mo_energy is None:
        mo_coeff = mo_coeff.copy()
        nocc = ncore + ncas
        casdm1 = mc.fcisolver.make_rdm1(mc.ci, ncas, mc.nelecas)
        fock_ao = mc.get_fock(casdm1=casdm1)

        occ, ucas = mc._eig(-casdm1, ncore, nocc)
        mo_coeff[:,ncore:nocc] = numpy.dot(mo_coeff[:,ncore:nocc], ucas)
        # diagonal term of Fock
        mo_energy = numpy.einsum('ji,ji->i', mo_coeff, fock_ao.dot(mo_coeff))
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[:ncore] = 2
        mo_occ[ncore:nocc] = -occ

    if h5py.is_hdf5(chkfile):
        fh5 = h5py.File(chkfile)
        if 'mcscf' in fh5:
            del(fh5['mcscf'])
    else:
        fh5 = h5py.File(chkfile, 'w')
    if 'mol' not in fh5:
        fh5['mol'] = format(mc.mol.pack())

    fh5['mcscf/mo_coeff'] = mo_coeff

    def store(key, val):
        if val is not None:
            fh5[key] = val
    store('mcscf/e_tot', e_tot)
    store('mcscf/e_cas', e_cas)
    store('mcscf/ci', ci_vector)
    store('mcscf/ncore', ncore)
    store('mcscf/ncas', ncas)
    store('mcscf/mo_occ', mo_occ)
    store('mcscf/mo_energy', mo_energy)
    fh5.close()
