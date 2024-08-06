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
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
Analytical electron-phonon matrix for restricted hartree fock
'''

import numpy as np
import scipy.linalg
from pyscf.hessian import rhf
from pyscf.lib import logger
from pyscf.scf._response_functions import _gen_rhf_response
from pyscf import __config__
from pyscf.data.nist import HARTREE2WAVENUMBER, MP_ME

CUTOFF_FREQUENCY = getattr(__config__, 'eph_cutoff_frequency', 80)  # 80 cm-1
KEEP_IMAG_FREQUENCY = getattr(__config__, 'eph_keep_imaginary_frequency', False)
IMAG_CUTOFF_FREQUENCY = getattr(__config__, 'eph_imag_cutoff_frequency', 1e-4)

def kernel(ephobj, mo_energy=None, mo_coeff=None, mo_occ=None, mo_rep=False):
    if mo_energy is None: mo_energy = ephobj.base.mo_energy
    if mo_coeff is None: mo_coeff = ephobj.base.mo_coeff
    if mo_occ is None: mo_occ = ephobj.base.mo_occ

    h1ao = ephobj.make_h1(mo_coeff, mo_occ)
    mo1, mo_e1 = ephobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao)

    de = ephobj.hess_elec(mo_energy, mo_coeff, mo_occ,
                          mo1=mo1, mo_e1=mo_e1, h1ao=h1ao)
    ephobj.de = de + ephobj.hess_nuc(ephobj.mol)

    omega, vec = ephobj.get_mode(ephobj.mol, ephobj.de)
    ephobj.omega, ephobj.vec = omega, vec
    ephobj.eph = ephobj.get_eph(mo1, omega, vec, mo_rep)
    return ephobj.eph, ephobj.omega

def solve_hmat(mol, hmat, cutoff_frequency=CUTOFF_FREQUENCY,
               keep_imag_frequency=KEEP_IMAG_FREQUENCY):
    log = logger.new_logger(mol, mol.verbose)
    mass = mol.atom_mass_list() * MP_ME
    natom = len(mass)
    h = np.empty_like(hmat) #[atom, axis, atom, axis]
    for i in range(natom):
        for j in range(natom):
            h[i,j] = hmat[i,j] / np.sqrt(mass[i]*mass[j])
    forcemat = h.transpose(0,2,1,3).reshape(natom*3, natom*3)
    forcemat[abs(forcemat)<1e-12]=0 #improve stability
    w, c = scipy.linalg.eig(forcemat)
    idx = np.argsort(w.real)[::-1] # sort the mode of the frequency
    w = w[idx]
    c = c[:,idx]
    w_au = w**0.5
    w_cm = w_au * HARTREE2WAVENUMBER
    log.info('****Eigenmodes(cm-1)****')
    for i, omega in enumerate(w_cm):
        if abs(omega.imag) < IMAG_CUTOFF_FREQUENCY:
            w_au[i] = w_au[i].real
            w_cm[i] = w_cm[i].real
            if omega.real > cutoff_frequency:
                log.info("Mode %i Omega=%.4f", i, omega.real)
            else:
                log.info("Mode %i Omega=%.4f, mode filtered", i, omega.real)
        else:
            log.info("Mode %i Omega=%.4fj, imaginary mode", i, omega.imag)
    if KEEP_IMAG_FREQUENCY:
        idx_real = np.where(w_cm.real>cutoff_frequency)[0]
        idx_imag = np.where(abs(w_cm.imag)>IMAG_CUTOFF_FREQUENCY)[0]
        idx = np.concatenate([idx_real, idx_imag])
    else:
        w_au = w_au.real
        idx = np.where(w_cm.real>cutoff_frequency)[0]
    w_new = w_au[idx]
    c_new = c[:,idx]
    log.info('****Remaining Eigenmodes(cm-1)****')
    for i, omega in enumerate(w_cm[idx]):
        if omega.imag == 0:
            log.info("Mode %i Omega=%.4f", i, omega.real)
        else:
            log.info("Mode %i Omega=%.4fj", i, omega.imag)
    return w_new, c_new


def get_mode(ephobj, mol=None, de=None):
    if mol is None: mol = ephobj.mol
    if de is None:
        if ephobj.de is None:
            de = ephobj.hess_elec() + ephobj.hess_nuc()
        else:
            de = ephobj.de
    return solve_hmat(mol, de, ephobj.cutoff_frequency, ephobj.keep_imag_frequency)


def rhf_deriv_generator(mf, mo_coeff, mo_occ):
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    vresp = _gen_rhf_response(mf, mo_coeff, mo_occ, hermi=1)
    def fx(mo1):
        mo1 = mo1.reshape(-1,nmo,nocc)
        nset = len(mo1)
        dm1 = np.empty((nset,nao,nao))
        for i, x in enumerate(mo1):
            dm = np.dot(x*2, mocc.T) # *2 for double occupancy
            dm1[i] = dm + dm.T
        v1 = vresp(dm1)
        return v1
    return fx

def vnuc_generator(ephobj, mol):
    if mol is None: mol = ephobj.mol
    aoslices = mol.aoslice_by_atom()
    def vnuc_deriv(atm_id):
        shl0, shl1, p0, p1 = aoslices[atm_id]
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
            vrinv *= -mol.atom_charge(atm_id)
        return vrinv + vrinv.transpose(0,2,1)
    return vnuc_deriv

def _freq_mass_weighted_vec(vec, omega, mass):
    nmodes, natoms = len(omega), len(mass)
    dtype = np.result_type(omega, vec)
    vec = vec.reshape(natoms,3,nmodes).astype(dtype)
    for i in range(natoms):
        for j in range(nmodes):
            vec[i,:,j] /= np.sqrt(2*mass[i]*omega[j])
    vec = vec.reshape(3*natoms,nmodes)
    return vec

def get_eph(ephobj, mo1, omega, vec, mo_rep):
    mol = ephobj.mol
    mf = ephobj.base
    vnuc_deriv = ephobj.vnuc_generator(mol)
    aoslices = mol.aoslice_by_atom()
    vind = rhf_deriv_generator(mf, mf.mo_coeff, mf.mo_occ)
    mocc = mf.mo_coeff[:,mf.mo_occ>0]
    dm0 = np.dot(mocc, mocc.T) * 2

    natoms = mol.natm
    nao = mol.nao_nr()

    vcore = []
    for ia in range(natoms):
        h1 = vnuc_deriv(ia)
        v1 = vind(mo1[ia])
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        vj1, vk1= rhf._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                     ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                      'li->s1kj', -dm0[:,p0:p1]], # vk1
                                     shls_slice=shls_slice)
        vhf = vj1 - vk1*.5
        vtot = h1 + v1 + vhf + vhf.transpose(0,2,1)
        vcore.append(vtot)
    vcore = np.asarray(vcore).reshape(-1,nao,nao)
    mass = mol.atom_mass_list() * MP_ME
    vec = _freq_mass_weighted_vec(vec, omega, mass)
    mat = np.einsum('xJ,xuv->Juv', vec, vcore)
    if mo_rep:
        mat = np.einsum('Juv,up,vq->Jpq', mat, mf.mo_coeff.conj(), mf.mo_coeff, optimize=True)
    return mat


class EPH(rhf.Hessian):
    '''EPH for restricted Hartree Fock

    Attributes:
        cutoff_frequency : float or int
            cutoff frequency in cm-1. Default is 80
        keep_imag_frequency : bool
            Whether to keep imaginary frequencies in the output.  Default is False

    Saved results

        omega : numpy.ndarray
            Vibrational frequencies in au.
        vec : numpy.ndarray
            Polarization vectors of the vibration modes
        eph : numpy.ndarray
            Electron phonon matrix eph[j,a,b] (j in nmodes, a,b in norbs)
    '''

    def __init__(self, scf_method, cutoff_frequency=CUTOFF_FREQUENCY,
                 keep_imag_frequency=KEEP_IMAG_FREQUENCY):
        rhf.Hessian.__init__(self, scf_method)
        self.cutoff_frequency = cutoff_frequency
        self.keep_imag_frequency = keep_imag_frequency

    get_mode = get_mode
    get_eph = get_eph
    vnuc_generator = vnuc_generator
    kernel = kernel
