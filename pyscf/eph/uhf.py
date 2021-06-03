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
Analytical electron-phonon matrix for unrestricted hartree fock
'''
import numpy as np
from pyscf import lib
from pyscf.eph import rhf as rhf_eph
from pyscf.hessian import uhf as uhf_hess
from pyscf.hessian import rhf as rhf_hess
from pyscf.scf._response_functions import _gen_uhf_response
from pyscf.data.nist import MP_ME

CUTOFF_FREQUENCY = rhf_eph.CUTOFF_FREQUENCY
KEEP_IMAG_FREQUENCY = rhf_eph.KEEP_IMAG_FREQUENCY

def uhf_deriv_generator(mf, mo_coeff, mo_occ):
    nao, nmoa = mo_coeff[0].shape
    nmob = mo_coeff[1].shape[1]
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]
    vresp = _gen_uhf_response(mf, mo_coeff, mo_occ, hermi=1)
    def fx(mo1):
        mo1 = mo1.reshape(-1,nmoa*nocca+nmob*noccb)
        nset = len(mo1)
        dm1 = np.empty((2,nset,nao,nao))
        for i, x in enumerate(mo1):
            xa = x[:nmoa*nocca].reshape(nmoa,nocca)
            xb = x[nmoa*nocca:].reshape(nmob,noccb)
            dma = np.dot(xa, mocca.T)
            dmb = np.dot(xb, moccb.T)
            dm1[0,i] = dma + dma.T
            dm1[1,i] = dmb + dmb.T
        v1 = vresp(dm1)
        return v1
    return fx

def get_eph(ephobj, mo1, omega, vec, mo_rep):
    if isinstance(mo1, str):
        mo1 = lib.chkfile.load(mo1, 'scf_mo1')
        mo1a = mo1['0']
        mo1b = mo1['1']
        mo1a = dict([(int(k), mo1a[k]) for k in mo1a])
        mo1b = dict([(int(k), mo1b[k]) for k in mo1b])

    mol = ephobj.mol
    mf = ephobj.base
    vnuc_deriv = ephobj.vnuc_generator(mol)
    aoslices = mol.aoslice_by_atom()

    mo_coeff, mo_occ = mf.mo_coeff, mf.mo_occ
    vind = uhf_deriv_generator(mf, mf.mo_coeff, mf.mo_occ)
    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = np.dot(mocca, mocca.T)
    dm0b = np.dot(moccb, moccb.T)

    natoms = mol.natm
    vcorea = []
    vcoreb = []
    for ia in range(natoms):
        h1 = vnuc_deriv(ia)
        moia = np.hstack((mo1a[ia], mo1b[ia]))
        v1 = vind(moia)
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        vja, vjb, vka, vkb= rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                             ['ji->s2kl', -dm0a[:,p0:p1],  # vja
                                              'ji->s2kl', -dm0b[:,p0:p1],  # vjb
                                              'li->s1kj', -dm0a[:,p0:p1],  # vka
                                              'li->s1kj', -dm0b[:,p0:p1]], # vkb
                                             shls_slice=shls_slice)
        vhfa = vja + vjb - vka
        vhfb = vjb + vjb - vkb
        vtota = h1 + v1[0] + vhfa + vhfa.transpose(0,2,1)
        vtotb = h1 + v1[1] + vhfb + vhfb.transpose(0,2,1)
        vcorea.append(vtota)
        vcoreb.append(vtotb)

    vcorea = np.asarray(vcorea).reshape(-1,nao,nao)
    vcoreb = np.asarray(vcoreb).reshape(-1,nao,nao)

    mass = mol.atom_mass_list() * MP_ME
    vec = rhf_eph._freq_mass_weighted_vec(vec, omega, mass)
    mata = np.einsum('xJ,xuv->Juv', vec, vcorea)
    matb = np.einsum('xJ,xuv->Juv', vec, vcoreb)
    if mo_rep:
        mata = np.einsum('Juv,up,vq->Jpq', mata, mf.mo_coeff[0].conj(), mf.mo_coeff[0], optimize=True)
        matb = np.einsum('Juv,up,vq->Jpq', matb, mf.mo_coeff[1].conj(), mf.mo_coeff[1], optimize=True)
    return np.asarray([mata,matb])


class EPH(uhf_hess.Hessian):
    '''EPH for unrestricted Hartree Fock

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
            Electron phonon matrix eph[spin,j,a,b] (j in nmodes, a,b in norbs)
    '''

    def __init__(self, scf_method, cutoff_frequency=CUTOFF_FREQUENCY,
                 keep_imag_frequency=KEEP_IMAG_FREQUENCY):
        uhf_hess.Hessian.__init__(self, scf_method)
        self.cutoff_frequency = cutoff_frequency
        self.keep_imag_frequency = keep_imag_frequency

    get_mode = rhf_eph.get_mode
    get_eph = get_eph
    vnuc_generator = rhf_eph.vnuc_generator
    kernel = rhf_eph.kernel

if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.M()
    mol.atom = [['O', [0.000000000000,  -0.000000000775,   0.923671924285]],
                ['H', [-0.000000000000,  -1.432564848017,   2.125164039823]],
                ['H', [0.000000000000,   1.432564848792,   2.125164035930]]]
    mol.unit = 'Bohr'
    mol.basis = 'sto3g'
    mol.verbose=4
    mol.build() # this is a pre-computed relaxed geometry

    mf = scf.UHF(mol)
    mf.conv_tol = 1e-16
    mf.conv_tol_grad = 1e-10
    mf.max_cycle=100
    mf.kernel()

    myeph = EPH(mf)

    grad = mf.nuc_grad_method().kernel()
    print("Force on the atoms/au:")
    print(grad)

    ephmat, omega = myeph.kernel()
    print(np.linalg.norm(ephmat[0]-ephmat[1]))

    from pyscf.eph.rhf import EPH as REPH

    mf1 = scf.RHF(mol)
    mf1.verbose=0
    mf1.conv_tol = 1e-16
    mf1.conv_tol_grad = 1e-10
    mf1.max_cycle=100
    mf1.kernel()

    myeph1 = REPH(mf1)
    rmat, omega = myeph1.kernel()
    for i in range(len(rmat)):
        print(min(np.linalg.norm(ephmat[0,i]-rmat[i]), np.linalg.norm(ephmat[0,i]+rmat[i])))
