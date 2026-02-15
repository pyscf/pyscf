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
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#         Gengzhi Yang <genzyang17@gmail.com>
#


r''' Helper functions for Fourier interpolation:
        F(R) = \sum_{k} F(k) exp(ikR) / sqrt(Nk)
        F(q) = \sum_{R \in WS} F(R) exp(-ikR) / sqrt(Nk) * ws_weight(R)
    where the backwards transform is performed over the Wigner-Seitz cell.

    See e.g.,
        Comput. Phys. Commun. 178, 685–699 (2008).
        https://doi.org/10.1016/j.cpc.2007.11.016

    This particular implementation is described in:
        [ref to be added]
'''


import numpy
from functools import reduce

from pyscf import lib
from pyscf.pbc.lo.base import get_kmesh


def wannier_interpolation(mf, kpts_band, lo_coeff):
    lo_coeff = numpy.asarray(lo_coeff)
    mo_coeff = numpy.asarray(mf.mo_coeff)
    mo_energy = numpy.asarray(mf.mo_energy)
    fk_band = _wannier_interpolation(mf.cell, mf.kpts, kpts_band, lo_coeff,
                                     mo_coeff, mo_energy)
    norb = lo_coeff[0].shape[1]
    sk = [numpy.eye(norb)]*len(fk_band)
    mo_energy, mo_coeff = mf.eig(fk_band, sk)

    return mo_energy, mo_coeff

def _wannier_interpolation(cell, kpts, kpts_band, lo_coeff, mo_coeff, mo_energy):
    f_ks = get_fock_ks(cell, kpts, mo_coeff, mo_energy, lo_coeff)

    return fourier_interpolate(cell, kpts, kpts_band, f_ks)

def get_fock_ks(cell, kpts, mo_coeff, mo_energy, lo_coeff):
    nkpts = len(kpts)
    norb = lo_coeff[0].shape[1]

    s_ks = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)

    f_ks = numpy.empty((nkpts,norb,norb), dtype=numpy.result_type(mo_coeff, lo_coeff))
    for k in range(nkpts):
        csc = reduce(numpy.dot, (lo_coeff[k].conj().T, s_ks[k], mo_coeff[k]))
        f_ks[k] = numpy.dot(csc*mo_energy[k], csc.conj().T)

    return f_ks

def fourier_interpolate(cell, kpts, kpts_band, f_ks):
    kmesh = get_kmesh(cell, kpts)
    Rs, ws = get_WigerSeitz_Rs(cell.lattice_vectors(), kmesh)

    f_Rs = xform_k2R(f_ks, kpts, Rs, ws)
    f_ks_band = xform_R2k(f_Rs, kpts, kpts_band, Rs, ws)

    return f_ks_band

def xform_k2R(f_ks, kpts, Rs, ws):
    nkpts = len(kpts)
    phase = numpy.exp(1j*numpy.dot(Rs, kpts.T)) / nkpts**0.5 * ws[:,None]
    f_Rs = lib.einsum('Rk,kij->Rij', phase, f_ks)
    return f_Rs

def xform_R2k(f_Rs, kpts, kpts_band, Rs, ws):
    nkpts = len(kpts)
    phase = numpy.exp(1j*numpy.dot(Rs, kpts_band.T)) / nkpts**0.5 * ws[:,None]
    f_ks = lib.einsum('Rk,Rij->kij', phase.conj(), f_Rs)
    return f_ks

def get_WigerSeitz_Rs(a, kmesh, search_mesh=None, tol=1e-6, forPlot = False):
    ''' Find lattice vectors that lie inside the Wigner-Seitz cell of the BvK supercell.

        Args:
            a: numpy.ndarray
                3x3 matrix. Each row gives a lattice vector
            kmesh: 1D array-like
                Size of the k-point mesh or equivalently the BvK supercell
            search_mesh: 1D array-like
                Searching mesh applied to the BvK supercell. If unspecified, [2,2,2] is used.
            tol: float
                Tolerance for determining degenerate points. Default: 1e-6

        Returns:
            Rs: numpy.ndarray
                (Nws, 3) matrix. Each row gives one WS lattice vector
            ws: 1D numpy.ndarray
                Weights corresponding to the WS lattice vectors
    '''
    kmesh = numpy.asarray(kmesh)
    Nk = numpy.prod(kmesh)
    a_sc = numpy.einsum('x,xy->xy', kmesh, a)

    if search_mesh is None:
        search_mesh = [2,2,2]
    search_mesh = numpy.asarray(search_mesh)

    Ts = lib.cartesian_prod([numpy.arange(-x,x+1) for x in search_mesh])
    Xs = numpy.dot(Ts, a_sc)
    idx0 = numpy.where(numpy.linalg.norm(Xs-numpy.zeros(3), axis=-1) < tol)[0][0]

    Ts = lib.cartesian_prod([numpy.arange(-x,x+1) for x in search_mesh*kmesh])
    Rs = numpy.dot(Ts, a)
    if forPlot:
        return Rs, numpy.zeros(Rs.shape[0])

    D = numpy.linalg.norm(Rs[:,None,:] - Xs, axis=-1)
    D0 = D[:,idx0]
    Dmin = D.min(axis=1)

    # Find R where D0 is (one of) the minimum
    idxs = numpy.where(abs(D0 - Dmin) < tol)[0]

    # Update Rs and D
    Rs = Rs[idxs]
    D = D[idxs]
    Dmin = Dmin[idxs]

    if idxs.size == Nk:
        ws = numpy.ones(Nk)
    elif idxs.size > Nk:
        ws = 1./numpy.count_nonzero( abs(D - Dmin.reshape(-1,1)) < tol, axis=1)
    else:
        raise RuntimeError

    if abs(numpy.sum(ws) - Nk) > tol:
        raise RuntimeError('Weights do not sum to kmesh size. Please use a larger '
                           'search_mesh than the default value, [2,2,2].')

    return Rs, numpy.sqrt(ws)



if __name__ == '__main__':
    from pyscf.pbc import gto, scf
    from pyscf.pbc.lo import KPM
    from pyscf.lib import logger

    atom = '''
    O          0.00000        0.00000        0.11779
    H          0.00000        0.75545       -0.47116
    H          0.00000       -0.75545       -0.47116
    '''
    a = numpy.eye(3) * 3
    basis = 'cc-pvdz'
    nband = 6

    kmesh = [5,1,1]

    cell = gto.M(atom=atom, basis=basis, a=a)
    cell.verbose = 4

    log = logger.new_logger(cell, verbose=6)

    kpts = cell.make_kpts(kmesh)

    ''' SCF
    '''
    mf = scf.KRKS(cell, kpts=kpts).density_fit()
    mf.xc = 'pbe'
    mf.kernel()

    ''' Reference bands
    '''
    blat = cell.reciprocal_vectors()
    scaled_kpts_band = numpy.linspace(-0.5,0.5,30)
    kpts_band = scaled_kpts_band[:,None] * blat[0]
    mo_energy, mo_coeff = mf.get_bands(kpts_band)

    band_energy = numpy.asarray([x[:nband] for x in mo_energy]).T * 27.211399

    ''' PM WF localization
    '''
    mo = [x[:,:nband] for x in mf.mo_coeff]
    mlo = KPM(cell, mo, kpts)
    mlo.kernel()

    ''' WF interpolation
    '''
    mo_energy, mo_coeff = wannier_interpolation(mf, kpts_band, mlo.mo_coeff)
    band_energy_wann = numpy.asarray([x[:nband] for x in mo_energy]).T * 27.211399

    err = abs(band_energy-band_energy_wann).max()
    log.info('band interpolation error: %.10f eV', err)

    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(2.5,2))
    ax = fig.gca()

    colors = ['gray', 'k']

    for iband,band_idx in enumerate([3,4,5]):
        if iband == 0:
            label1 = 'ref'
            label2 = 'interp'
        else:
            label1 = label2 = None
        ax.plot(scaled_kpts_band, band_energy[band_idx], '-', lw=1, color=colors[0],
                label=label1)
        ax.plot(scaled_kpts_band, band_energy_wann[band_idx], '.', markersize=3,
                color=colors[1], label=label2)

    ax.legend(frameon=False)

    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'Band energy (eV)')

    plt.tight_layout()

    plt.savefig('band.pdf')
    plt.close(fig)
