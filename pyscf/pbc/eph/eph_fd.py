#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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

import numpy as np
from pyscf.pbc import scf, dft, gto
from pyscf.eph.rhf import solve_hmat, _freq_mass_weighted_vec
from pyscf.lib import logger, param
from pyscf.data.nist import MP_ME

'''Electron-Phonon matrix from finite difference for Gamma Point'''
# Note, the code now only return eph matrix at Gamma Point
# cell relaxation needs to be performed before computing eph matrix

def copy_mf(mf, cell):
    mf1 = mf.__class__(cell)
    if isinstance(mf, scf.hf.KohnShamDFT):
        mf1.xc = mf.xc
    mf1.kpts = mf.kpts
    mf1.exxdiv = getattr(mf, 'exxdiv', None)
    mf1.conv_tol = mf.conv_tol
    mf1.conv_tol_grad = mf.conv_tol_grad
    return mf1

def run_mfs(mf, cells_a, cells_b):
    '''perform a set of calculations on given two sets of cell'''
    nconfigs = len(cells_a)
    dm0 = mf.make_rdm1()
    mflist = []
    for i in range(nconfigs):
        mf1 = copy_mf(mf, cells_a[i])
        mf2 = copy_mf(mf, cells_b[i])
        mf1.kernel(dm0=dm0)
        mf2.kernel(dm0=dm0)
        if not (mf1.converged):
            logger.warn(mf, "%ith config mf1 not converged", i)
        if not (mf2.converged):
            logger.warn(mf, "%ith config mf2 not converged", i)
        mflist.append((mf1, mf2))
    return mflist

def gen_cells(cell, disp):
    """From the given cell, generate 3N cells with a shift on
    + displacement(cell_a) and - displacement(cell_s) on each Cartesian
    coordinates
    """
    coords = cell.atom_coords()
    if cell.unit[0].lower() == 'a':
        coords = np.asarray(coords) * param.BOHR
        disp_ = disp * param.BOHR
    else:
        disp_ = disp
    natoms = len(coords)
    cell_a, cell_s = [],[]
    for i in range(natoms):
        for x in range(3):
            new_coords_a, new_coords_s = coords.copy(), coords.copy()
            new_coords_a[i][x] += disp_
            new_coords_s[i][x] -= disp_
            atoma = [[cell.atom_symbol(j), coord] for (j, coord) in zip(range(natoms), new_coords_a)]
            atoms = [[cell.atom_symbol(j), coord] for (j, coord) in zip(range(natoms), new_coords_s)]
            cell_a.append(cell.set_geom_(atoma, inplace=False))
            cell_s.append(cell.set_geom_(atoms, inplace=False))
    return cell_a, cell_s

def get_vmat(mf, mfset, disp):
    nconfigs = len(mfset)
    vmat=[]
    mygrad = mf.nuc_grad_method()
    veff  = mygrad.get_veff()
    RESTRICTED = (veff.ndim==4)
    v1e = mygrad.get_hcore() - np.asarray(mf.cell.pbc_intor("int1e_ipkin", kpts=mf.kpts))
    if RESTRICTED:
        vtmp = veff - v1e.transpose(1,0,2,3)
    else:
        vtmp = veff - v1e.transpose(1,0,2,3)[:,None]

    aoslice = mf.cell.aoslice_by_atom()
    for i in range(nconfigs):
        atmid, axis = np.divmod(i, 3)
        p0, p1 = aoslice[atmid][2:]
        mf1, mf2 = mfset[i]
        if RESTRICTED:
            vfull1 = mf1.get_veff() + mf1.get_hcore() \
                   - np.asarray(mf1.cell.pbc_intor('int1e_kin', kpts=mf1.kpts))  # <u+|V+|v+>
            vfull2 = mf2.get_veff() + mf2.get_hcore() \
                   - np.asarray(mf2.cell.pbc_intor('int1e_kin', kpts=mf2.kpts))  # <u-|V-|v->
        else:
            vfull1 = mf1.get_veff() + mf1.get_hcore()[None] \
                   - np.asarray(mf1.cell.pbc_intor('int1e_kin', kpts=mf1.kpts))[None]  # <u+|V+|v+>
            vfull2 = mf2.get_veff() + mf2.get_hcore()[None] \
                   - np.asarray(mf2.cell.pbc_intor('int1e_kin', kpts=mf2.kpts))[None]  # <u-|V-|v->
        vfull = (vfull1 - vfull2)/disp  # (<p+|V+|q+>-<p-|V-|q->)/dR
        if RESTRICTED:
            vfull[:,p0:p1] -= vtmp[axis,:,p0:p1]
            vfull[:,:,p0:p1] -= vtmp[axis,:,p0:p1].transpose(0,2,1).conj()
        else:
            vfull[:,:,p0:p1] -= vtmp[axis,:,:,p0:p1]
            vfull[:,:,:,p0:p1] -= vtmp[axis,:,:,p0:p1].transpose(0,1,3,2).conj()

        vmat.append(vfull)

    vmat= np.asarray(vmat)
    if RESTRICTED:
        return vmat[:,0]
    else:
        return vmat[:,:,0]

def run_hess(mfset, disp):
    natoms = len(mfset[0][0].cell.atom_mass_list())
    hess=[]
    for (mf1, mf2) in mfset:
        grad1 = mf1.nuc_grad_method()
        grad2 = mf2.nuc_grad_method()
        g1 = grad1.kernel()
        g2 = grad2.kernel()
        gdelta = (g1-g2) / disp
        hess.append(gdelta)
    hess = np.asarray(hess).reshape(natoms, 3, natoms, 3).transpose(0,2,1,3)
    return hess


def kernel(mf, disp=1e-4, mo_rep=False):
    if not mf.converged: mf.kernel()
    mo_coeff = np.asarray(mf.mo_coeff)
    RESTRICTED= (mo_coeff.ndim==3)
    cell = mf.cell
    cells_a, cells_b = gen_cells(cell, disp/2.0) # generate a bunch of cells with disp/2 on each cartesian coord
    mfset = run_mfs(mf, cells_a, cells_b) # run mean field calculations on all these cells
    vmat = get_vmat(mf, mfset, disp) # extracting <u|dV|v>/dR
    hmat = run_hess(mfset, disp)
    omega, vec = solve_hmat(cell, hmat)
    mass = cell.atom_mass_list() * MP_ME
    vec = _freq_mass_weighted_vec(vec, omega, mass)
    if mo_rep:
        if RESTRICTED:
            vmat = np.einsum('xuv,up,vq->xpq', vmat, mo_coeff[0].conj(), mo_coeff[0])
        else:
            vmat = np.einsum('xsuv,sup,svq->xspq', vmat, mo_coeff[:,0].conj(), mo_coeff[:,0])

    if RESTRICTED:
        mat = np.einsum('xJ,xpq->Jpq', vec, vmat)
    else:
        mat = np.einsum('xJ,xspq->sJpq', vec, vmat)
    return mat, omega

if __name__ == '__main__':
    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()

    kpts = cell.make_kpts([1,1,1])
    mf = dft.KRKS(cell, kpts)
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-8
    mf.kernel()

    vmat, omega = kernel(mf, mo_rep=True)
