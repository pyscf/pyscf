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

'''
There are two options to call wannier90 in PySCF.  One is the pyWannier90.py
interface as implemented in this file.

(1)
pyWannier90: Wannier90 for PySCF (https://github.com/hungpham2017/pyWannier90)
Hung Q. Pham
email: pqh3.14@gmail.com

(2)
Another wannier90 python interface is available on the repo:
    https://github.com/zhcui/wannier90
Contact its author "Zhihao Cui" <zcui@caltech.edu> for more details of
installation and implementations.
'''

import os, time
import numpy as np
from scipy.io import FortranFile
import pyscf.data.nist as param
from pyscf import lib
from pyscf.pbc import df
from pyscf.pbc.dft import gen_grid, numint

try:
    import libwannier90
except ImportError:
    print('WARNING: Check the installation of libwannier90 and its path in pyscf/pbc/tools/pywannier90.py')
    print('libwannier90 can be found at: https://github.com/hungpham2017/pyWannier90')
    raise


def save_kmf(kmf, chkfile):
    ''' Save a wavefunction'''
    from pyscf.lib.chkfile import save
    kpts = kmf.kpts
    mo_energy_kpts = kmf.mo_energy_kpts
    mo_coeff_kpts = kmf.mo_coeff_kpts

    scf_dic = {'kpts'          : kpts,
               'mo_energy_kpts': mo_energy_kpts,
               'mo_coeff_kpts' : mo_coeff_kpts}
    save(chkfile, 'scf', scf_dic)

def load_kmf(chkfile):
    ''' Load a wavefunction'''
    from pyscf.lib.chkfile import load
    kmf = load(chkfile, 'scf')
    class fake_kmf:
        def __init__(self, kmf):
            self.kpts = kmf['kpts']
            self.mo_energy_kpts = kmf['mo_energy_kpts']
            self.mo_coeff_kpts = kmf['mo_coeff_kpts']
    kmf = fake_kmf(kmf)
    return kmf

def angle(v1, v2):
    '''
    Return the angle (in radiant between v1 and v2)
    '''

    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    cosa = v1.dot(v2)/ np.linalg.norm(v1) / np.linalg.norm(v2)
    return np.arccos(cosa)

def transform(x_vec, z_vec):
    '''
    Construct a transformation matrix to transform r_vec to the new coordinate system defined by x_vec and z_vec
    '''

    x_vec = x_vec/np.linalg.norm(np.asarray(x_vec))
    z_vec = z_vec/np.linalg.norm(np.asarray(z_vec))
    assert x_vec.dot(z_vec) == 0    # x and z have to be orthogonal to one another
    y_vec = -np.cross(x_vec,z_vec)
    new = np.asarray([x_vec, y_vec, z_vec])
    original = np.asarray([[1,0,0],[0,1,0],[0,0,1]])

    tran_matrix = np.empty([3,3])
    for row in range(3):
        for col in range(3):
            tran_matrix[row,col] = np.cos(angle(original[row],new[col]))

    return tran_matrix.T

def cartesian_prod(arrays, out=None, order='C'):
    '''
    This function is similar to lib.cartesian_prod of PySCF, except the output can be in Fortran or in C order
    '''
    arrays = [np.asarray(x) for x in arrays]
    dtype = np.result_type(*arrays)
    nd = len(arrays)
    dims = [nd] + [len(x) for x in arrays]

    if out is None:
        out = np.empty(dims, dtype)
    else:
        out = np.ndarray(dims, dtype, buffer=out)
    tout = out.reshape(dims)

    shape = [-1] + [1] * nd
    for i, arr in enumerate(arrays):
        tout[i] = arr.reshape(shape[:nd-i])

    return tout.reshape((nd,-1),order=order).T

def periodic_grid(cell, grid=[50,50,50], supercell=[1,1,1], order='C'):
    '''
    Generate a periodic grid for the unit/computational cell in F/C order
    '''
    ngrid = np.asarray(grid)
    qv = cartesian_prod([np.arange(-ngrid[i]*(supercell[i]//2),ngrid[i]*((supercell[i]+1)//2)) for i in range(3)], order=order)
    a_frac = lib.einsum('i,ij->ij', 1./ngrid, cell.lattice_vectors())
    coords = np.dot(qv, a_frac)

    # Compute weight
    ngrids = np.prod(grid)
    ncells = np.prod(supercell)
    weights = np.empty(ngrids*ncells)
    weights[:] = cell.vol / ngrids / ncells
    return coords, weights

def R_r(r_norm, r=1, zona=1):
    r'''
    Radial functions used to compute \Theta_{l,m_r}(\theta,\phi)
    '''

    if r == 1:
        R_r = 2 * zona**(3/2) * np.exp(-zona*r_norm)
    elif r == 2:
        R_r = 1 / 2 / np.sqrt(2) * zona**(3/2) * (2 - zona*r_norm) * np.exp(-zona*r_norm/2)
    else:
        R_r = np.sqrt(4/27) * zona**(3/2) * (1 - 2*zona*r_norm/3 + 2*(zona**2)*(r_norm**2)/27) * np.exp(-zona*r_norm/3)

    return R_r

def theta(func, cost, phi):
    r'''
    Basic angular functions (s,p,d,f) used to compute \Theta_{l,m_r}(\theta,\phi)
    ref: Table 3.1 of the Wannier90 User guide
        Link: https://github.com/wannier-developers/wannier90/raw/v3.1.0/doc/compiled_docs/user_guide.pdf
    '''
    sint = np.sqrt(1 - cost**2)
    if func == 's':
        theta = 1 / np.sqrt(4 * np.pi) * np.ones([cost.shape[0]])
    elif func == 'pz':
        theta = np.sqrt(3 / 4 / np.pi) * cost
    elif func == 'px':
        theta = np.sqrt(3 / 4 / np.pi) * sint * np.cos(phi)
    elif func == 'py':
        theta = np.sqrt(3 / 4 / np.pi) * sint * np.sin(phi)
    elif func == 'dz2':
        theta = np.sqrt(5 / 16 / np.pi) * (3*cost**2 - 1)
    elif func == 'dxz':
        theta = np.sqrt(15 / 4 / np.pi) * sint * cost * np.cos(phi)
    elif func == 'dyz':
        theta = np.sqrt(15 / 4 / np.pi) * sint * cost * np.sin(phi)
    elif func == 'dx2-y2':
        theta = np.sqrt(15 / 16 / np.pi) * (sint**2) * np.cos(2*phi)
    elif func == 'dxy':
        theta = np.sqrt(15 / 16 / np.pi) * (sint**2) * np.sin(2*phi)
    elif func == 'fz3':
        theta = np.sqrt(7) / 4 / np.sqrt(np.pi) * (5*cost**3 - 3*cost)
    elif func == 'fxz2':
        theta = np.sqrt(21) / 4 / np.sqrt(2*np.pi) * (5*cost**2 - 1) * sint * np.cos(phi)
    elif func == 'fyz2':
        theta = np.sqrt(21) / 4 / np.sqrt(2*np.pi) * (5*cost**2 - 1) * sint * np.sin(phi)
    elif func == 'fz(x2-y2)':
        theta = np.sqrt(105) / 4 / np.sqrt(np.pi) * sint**2 * cost * np.cos(2*phi)
    elif func == 'fxyz':
        theta = np.sqrt(105) / 4 / np.sqrt(np.pi) * sint**2 * cost * np.sin(2*phi)
    elif func == 'fx(x2-3y2)':
        theta = np.sqrt(35) / 4 / np.sqrt(2*np.pi) * sint**3 * (np.cos(phi)**2 - 3*np.sin(phi)**2) * np.cos(phi)
    elif func == 'fy(3x2-y2)':
        theta = np.sqrt(35) / 4 / np.sqrt(2*np.pi) * sint**3 * (3*np.cos(phi)**2 - np.sin(phi)**2) * np.sin(phi)

    return theta

def theta_lmr(l, mr, cost, phi):
    r'''
    Compute the value of \Theta_{l,m_r}(\theta,\phi)
    ref: Table 3.1 and 3.2 of the Wannier90 User guide
        Link: https://github.com/wannier-developers/wannier90/raw/v3.1.0/doc/compiled_docs/user_guide.pdf
    '''
    assert l in [0,1,2,3,-1,-2,-3,-4,-5]
    assert mr in [1,2,3,4,5,6,7]

    if l == 0:                           # s
        theta_lmr = theta('s', cost, phi)
    elif (l == 1) and (mr == 1):         # pz
        theta_lmr = theta('pz', cost, phi)
    elif (l == 1) and (mr == 2):         # px
        theta_lmr = theta('px', cost, phi)
    elif (l == 1) and (mr == 3):         # py
        theta_lmr = theta('py', cost, phi)
    elif (l == 2) and (mr == 1):         # dz2
        theta_lmr = theta('dz2', cost, phi)
    elif (l == 2) and (mr == 2):         # dxz
        theta_lmr = theta('dxz', cost, phi)
    elif (l == 2) and (mr == 3):         # dyz
        theta_lmr = theta('dyz', cost, phi)
    elif (l == 2) and (mr == 4):         # dx2-y2
        theta_lmr = theta('dx2-y2', cost, phi)
    elif (l == 2) and (mr == 5):         # dxy
        theta_lmr = theta('dxy', cost, phi)
    elif (l == 3) and (mr == 1):         # fz3
        theta_lmr = theta('fz3', cost, phi)
    elif (l == 3) and (mr == 2):         # fxz2
        theta_lmr = theta('fxz2', cost, phi)
    elif (l == 3) and (mr == 3):         # fyz2
        theta_lmr = theta('fyz2', cost, phi)
    elif (l == 3) and (mr == 4):         # fz(x2-y2)
        theta_lmr = theta('fz(x2-y2)', cost, phi)
    elif (l == 3) and (mr == 5):         # fxyz
        theta_lmr = theta('fxyz', cost, phi)
    elif (l == 3) and (mr == 6):         # fx(x2-3y2)
        theta_lmr = theta('fx(x2-3y2)', cost, phi)
    elif (l == 3) and (mr == 7):         # fy(3x2-y2)
        theta_lmr = theta('fy(3x2-y2)', cost, phi)
    elif (l == -1) and (mr == 1):         # sp-1
        theta_lmr = 1/np.sqrt(2) * (theta('s', cost, phi) + theta('px', cost, phi))
    elif (l == -1) and (mr == 2):         # sp-2
        theta_lmr = 1/np.sqrt(2) * (theta('s', cost, phi) - theta('px', cost, phi))
    elif (l == -2) and (mr == 1):         # sp2-1
        theta_lmr = 1/np.sqrt(3) * theta('s', cost, phi) - 1/np.sqrt(6) *theta('px', cost, phi) + 1/np.sqrt(2) * theta('py', cost, phi)
    elif (l == -2) and (mr == 2):         # sp2-2
        theta_lmr = 1/np.sqrt(3) * theta('s', cost, phi) - 1/np.sqrt(6) *theta('px', cost, phi) - 1/np.sqrt(2) * theta('py', cost, phi)
    elif (l == -2) and (mr == 3):         # sp2-3
        theta_lmr = 1/np.sqrt(3) * theta('s', cost, phi) + 2/np.sqrt(6) *theta('px', cost, phi)
    elif (l == -3) and (mr == 1):         # sp3-1
        theta_lmr = 1/2 * (theta('s', cost, phi) + theta('px', cost, phi) + theta('py', cost, phi) + theta('pz', cost, phi))
    elif (l == -3) and (mr == 2):         # sp3-2
        theta_lmr = 1/2 * (theta('s', cost, phi) + theta('px', cost, phi) - theta('py', cost, phi) - theta('pz', cost, phi))
    elif (l == -3) and (mr == 3):         # sp3-3
        theta_lmr = 1/2 * (theta('s', cost, phi) - theta('px', cost, phi) + theta('py', cost, phi) - theta('pz', cost, phi))
    elif (l == -3) and (mr == 4):         # sp3-4
        theta_lmr = 1/2 * (theta('s', cost, phi) - theta('px', cost, phi) - theta('py', cost, phi) + theta('pz', cost, phi))
    elif (l == -4) and (mr == 1):         # sp3d-1
        theta_lmr = 1/np.sqrt(3) * theta('s', cost, phi) - 1/np.sqrt(6) *theta('px', cost, phi) + 1/np.sqrt(2) * theta('py', cost, phi)
    elif (l == -4) and (mr == 2):         # sp3d-2
        theta_lmr = 1/np.sqrt(3) * theta('s', cost, phi) - 1/np.sqrt(6) *theta('px', cost, phi) - 1/np.sqrt(2) * theta('py', cost, phi)
    elif (l == -4) and (mr == 3):         # sp3d-3
        theta_lmr = 1/np.sqrt(3) * theta('s', cost, phi) + 2/np.sqrt(6) * theta('px', cost, phi)
    elif (l == -4) and (mr == 4):         # sp3d-4
        theta_lmr = 1/np.sqrt(2) * (theta('pz', cost, phi) + theta('dz2', cost, phi))
    elif (l == -4) and (mr == 5):         # sp3d-5
        theta_lmr = 1/np.sqrt(2) * (-theta('pz', cost, phi) + theta('dz2', cost, phi))
    elif (l == -5) and (mr == 1):         # sp3d2-1
        theta_lmr = 1/np.sqrt(6) * theta('s', cost, phi) - 1/np.sqrt(2) *theta('px', cost, phi) - 1/np.sqrt(12) *theta('dz2', cost, phi) \
                    + 1/2 *theta('dx2-y2', cost, phi)
    elif (l == -5) and (mr == 2):         # sp3d2-2
        theta_lmr = 1/np.sqrt(6) * theta('s', cost, phi) + 1/np.sqrt(2) *theta('px', cost, phi) - 1/np.sqrt(12) *theta('dz2', cost, phi) \
                    + 1/2 *theta('dx2-y2', cost, phi)
    elif (l == -5) and (mr == 3):         # sp3d2-3
        theta_lmr = 1/np.sqrt(6) * theta('s', cost, phi) - 1/np.sqrt(2) *theta('py', cost, phi) - 1/np.sqrt(12) *theta('dz2', cost, phi) \
                    - 1/2 *theta('dx2-y2', cost, phi)
    elif (l == -5) and (mr == 4):         # sp3d2-4
        theta_lmr = 1/np.sqrt(6) * theta('s', cost, phi) + 1/np.sqrt(2) *theta('py', cost, phi) - 1/np.sqrt(12) *theta('dz2', cost, phi) \
                    - 1/2 *theta('dx2-y2', cost, phi)
    elif (l == -5) and (mr == 5):         # sp3d2-5
        theta_lmr = 1/np.sqrt(6) * theta('s', cost, phi) - 1/np.sqrt(2) *theta('pz', cost, phi) + 1/np.sqrt(3) *theta('dz2', cost, phi)
    elif (l == -5) and (mr == 6):         # sp3d2-6
        theta_lmr = 1/np.sqrt(6) * theta('s', cost, phi) + 1/np.sqrt(2) *theta('pz', cost, phi) + 1/np.sqrt(3) *theta('dz2', cost, phi)

    return theta_lmr

def g_r(grids_coor, site, l, mr, r, zona, x_axis=[1,0,0], z_axis=[0,0,1], unit='B'):
    r'''
    Evaluate the projection function g(r) or \Theta_{l,m_r}(\theta,\phi) on a grid
    ref: Chapter 3, wannier90 User Guide
    Attributes:
        grids_coor : a grids for the cell of interest
        site       : absolute coordinate (in Borh/Angstrom) of the g(r) in the cell
        l, mr      : l and mr value in the Table 3.1 and 3.2 of the ref
    Return:
        theta_lmr  : an array (ngrid, value) of g(r)

    '''

    unit_conv = 1
    if unit == 'A': unit_conv = param.BOHR

    r_vec = (grids_coor - site)
    r_vec = lib.einsum('iv,uv ->iu', r_vec, transform(x_axis, z_axis))
    r_norm = np.linalg.norm(r_vec,axis=1)
    if (r_norm < 1e-8).any():
        r_vec = (grids_coor - site - 1e-5)
        r_vec = lib.einsum('iv,uv ->iu', r_vec, transform(x_axis, z_axis))
        r_norm = np.linalg.norm(r_vec,axis=1)
    cost = r_vec[:,2]/r_norm

    phi = np.empty_like(r_norm)
    larger_idx = r_vec[:, 0] > 1e-8
    smaller_idx = r_vec[:, 0] < -1e-8
    neither_idx = not larger_idx and not smaller_idx
    phi[larger_idx] = np.arctan(r_vec[larger_idx,1]/r_vec[larger_idx,0])
    phi[smaller_idx] = np.arctan(r_vec[smaller_idx,1]/r_vec[smaller_idx,0])  + np.pi
    phi[neither_idx] = np.sign(r_vec[neither_idx,1]) * 0.5 * np.pi

    return theta_lmr(l, mr, cost, phi) * R_r(r_norm * unit_conv, r = r, zona = zona)

def get_wigner_seitz_supercell(w90, ws_search_size=[2,2,2], ws_distance_tol=1e-6):
    '''
    Return a grid that contains all the lattice within the Wigner-Seitz supercell
    Ref: the hamiltonian_wigner_seitz(count_pts) in wannier90/src/hamittonian.F90
    '''

    real_metric = w90.real_lattice_loc.T.dot(w90.real_lattice_loc)
    dist_dim = np.prod(2 * (np.asarray(ws_search_size) + 1) + 1)
    ndegen = []
    irvec = []
    mp_grid = np.asarray(w90.mp_grid_loc)
    n1_range =  np.arange(-ws_search_size[0] * mp_grid[0], ws_search_size[0]*mp_grid[0] + 1)
    n2_range =  np.arange(-ws_search_size[1] * mp_grid[1], ws_search_size[1]*mp_grid[1] + 1)
    n3_range =  np.arange(-ws_search_size[2] * mp_grid[2], ws_search_size[2]*mp_grid[2] + 1)
    x, y, z = np.meshgrid(n1_range, n2_range, n3_range)
    n_list = np.vstack([z.flatten('F'), x.flatten('F'), y.flatten('F')]).T
    i1 = np.arange(- ws_search_size[0] - 1, ws_search_size[0] + 2)
    i2 = np.arange(- ws_search_size[1] - 1, ws_search_size[1] + 2)
    i3 = np.arange(- ws_search_size[2] - 1, ws_search_size[2] + 2)
    x, y, z = np.meshgrid(i1, i2, i3)
    i_list = np.vstack([z.flatten('F'), x.flatten('F'), y.flatten('F')]).T

    nrpts = 0
    for n in n_list:
        # Calculate |r-R|^2
        ndiff = n - i_list * mp_grid
        dist = (ndiff.dot(real_metric).dot(ndiff.T)).diagonal()

        dist_min = dist.min()
        if abs(dist[(dist_dim + 1)//2 -1] - dist_min) < ws_distance_tol**2:
            temp = 0
            for i in range(0, dist_dim):
                if (abs(dist[i] - dist_min) < ws_distance_tol**2):
                    temp = temp + 1
            ndegen.append(temp)
            irvec.append(n.tolist())
            if (n**2).sum() < 1.e-10: rpt_origin = nrpts
            nrpts = nrpts + 1

    irvec = np.asarray(irvec)
    ndegen = np.asarray(ndegen)

    # Check the "sum rule"
    tot = np.sum(1/np.asarray(ndegen))
    assert tot - np.prod(mp_grid) < 1e-8, "Error in finding Wigner-Seitz points!!!"

    return ndegen, irvec, rpt_origin

def R_wz_sc(w90, R_in, R0, ws_search_size=[2,2,2], ws_distance_tol=1e-6):
    '''
    TODO: document it
    Ref: This is the replication of the R_wz_sc function of ws_distance.F90
    '''
    ndegenx = 8 #max number of unit cells that can touch in a single point (i.e.  vertex of cube)
    R_bz = np.asarray(R_in).reshape(-1, 3)
    nR = R_bz.shape[0]
    R0 = np.asarray(R0)
    ndeg = np.zeros([nR], dtype=np.int32)
    ndeg_ = np.zeros([nR, ndegenx])
    shifts = np.zeros([nR, ndegenx, 3])
    R_out = np.zeros([nR, ndegenx, 3])

    mod2_R_bz = np.sum((R_bz - R0)**2, axis=1)
    R_in_f = R_bz.dot(w90.recip_lattice_loc.T / 2 / np.pi)
    n1_range =  np.arange(-ws_search_size[0] - 1, ws_search_size[0] + 2)
    n2_range =  np.arange(-ws_search_size[1] - 1, ws_search_size[1] + 2)
    n3_range =  np.arange(-ws_search_size[2] - 1, ws_search_size[2] + 2)
    x, y, z = np.meshgrid(n1_range, n2_range, n3_range)
    n_list = np.vstack([z.flatten('F'), x.flatten('F'), y.flatten('F')]).T
    trans_vecs = n_list * w90.mp_grid_loc

    # First loop:
    R_f = np.repeat(R_in_f[:,np.newaxis,:], trans_vecs.shape[0], axis=1) + trans_vecs
    R = R_f.dot(w90.real_lattice_loc)
    mod2_R = np.sum((R - R0)**2, axis=2)
    mod2_R_min = mod2_R.min(axis=1)
    mod2_R_min_idx = np.argmin(mod2_R, axis=1)
    idx = mod2_R_min < mod2_R_bz
    R_bz[idx] = R[idx, mod2_R_min_idx[idx]]
    mod2_R_bz[idx] = mod2_R_min[idx]
    shifts_data = np.repeat(trans_vecs[np.newaxis,:,:], nR, axis=0)[idx, mod2_R_min_idx[idx]]
    shifts[idx] = np.repeat(shifts_data[:,np.newaxis,:], ndegenx, axis=1)

    idx = mod2_R_bz < ws_distance_tol**2
    ndeg[idx] = 1
    R_out[idx, 0] = R0

    # Second loop:
    R_in_f = R_bz.dot(w90.recip_lattice_loc.T / 2 / np.pi)
    R_f = np.repeat(R_in_f[:,np.newaxis,:], trans_vecs.shape[0], axis=1) + trans_vecs
    R = R_f.dot(w90.real_lattice_loc)
    mod2_R = np.sum((R - R0)**2, axis=2)
    mod2_R_bz = np.repeat(mod2_R_bz[:,np.newaxis], trans_vecs.shape[0], axis=1)
    abs_diff = abs(np.sqrt(mod2_R) - np.sqrt(mod2_R_bz))
    idx = abs_diff < ws_distance_tol
    ndeg = idx.sum(axis=1)
    assert (ndeg <= 8).all(), "The degeneracy cannot be larger than 8"
    for i in range(nR):
        R_out[i, :ndeg[i]] = R[i, idx[i]]
        shifts[i, :ndeg[i]] = shifts[i, :ndeg[i]] + trans_vecs[idx[i]]
        ndeg_[i, :ndeg[i]] = 1.0

    return ndeg_, ndeg, R_out, shifts

def ws_translate_dist(w90, irvec, ws_search_size=[2,2,2], ws_distance_tol=1e-6):
    '''
    TODO: document it
    Ref: This is the replication of the ws_translate_dist function of ws_distance.F90
    '''
    nrpts = irvec.shape[0]
    ndegenx = 8 #max number of unit cells that can touch in a single point (i.e.  vertex of cube)
    num_wann = w90.num_wann
    assert ndegenx*num_wann*nrpts > 0, "Unexpected dimensions in ws_translate_dist"

    irvec_ = []
    wann_centres_i = []
    wann_centres_j = []
    for i in range(3):
        x, y, z = np.meshgrid(irvec[:,i], np.zeros(num_wann), np.zeros(num_wann), indexing='ij')
        irvec_.append(x.flatten())
        x, y, z = np.meshgrid(np.zeros(nrpts), np.zeros(num_wann), w90.wann_centres[:,i], indexing='ij')
        wann_centres_i.append(z.flatten())
        x, y, z = np.meshgrid(np.zeros(nrpts), w90.wann_centres[:,i], np.zeros(num_wann), indexing='ij')
        wann_centres_j.append(y.flatten())


    irvec_list = np.vstack(irvec_).T
    irvec_cart_list = irvec_list.dot(w90.real_lattice_loc)
    wann_centres_i_list = np.vstack(wann_centres_i).T
    wann_centres_j_list = np.vstack(wann_centres_j).T
    R_in = irvec_cart_list - wann_centres_i_list + wann_centres_j_list
    wdist_ndeg_, wdist_ndeg, R_out, shifts = w90.R_wz_sc(R_in, [0,0,0], ws_search_size, ws_distance_tol)
    ndegenx = wdist_ndeg_.shape[1]
    irdist_ws = np.repeat(irvec_list[:,np.newaxis,:], ndegenx, axis=1) + shifts
    crdist_ws = irdist_ws.dot(w90.real_lattice_loc)

    # Reformat the matrices for the computational convenience in lib.einsum
    wdist_ndeg = wdist_ndeg.reshape(nrpts, num_wann, num_wann)
    wdist_ndeg_ = wdist_ndeg_.reshape(nrpts, num_wann, num_wann, ndegenx).transpose(3,0,1,2)
    irdist_ws = irdist_ws.reshape(nrpts, num_wann, num_wann, ndegenx, 3).transpose(3,0,1,2,4)
    crdist_ws = crdist_ws.reshape(nrpts, num_wann, num_wann, ndegenx, 3).transpose(3,0,1,2,4)

    return wdist_ndeg, wdist_ndeg_, irdist_ws, crdist_ws


'''Main class of pyWannier90'''
class W90:
    def __init__(self, kmf, cell, mp_grid, num_wann, gamma=False, spinors=False, spin_up=None, other_keywords=None):

        if isinstance(kmf, str):
            self.kmf = load_kmf(kmf)
        else:
            self.kmf = kmf
        self.cell = cell
        self.num_wann = num_wann
        self.keywords = other_keywords

        # Collect the pyscf calculation info
        nao_kpts = []
        for mo_energy in kmf.mo_energy_kpts:
            nao_kpts.append(mo_energy.shape[0])

        self.num_bands_tot = np.min(nao_kpts)
        if self.num_bands_tot < cell.nao_nr():
            print(('The number of bands at different k-point are not the same. '
                   'The first %d bands are used.') % (self.num_bands_tot) )

        self.num_kpts_loc = self.kmf.kpts.shape[0]
        self.mp_grid_loc = mp_grid
        assert self.num_kpts_loc == np.asarray(self.mp_grid_loc).prod()
        self.real_lattice_loc = self.cell.lattice_vectors() * param.BOHR
        self.recip_lattice_loc = self.cell.reciprocal_vectors() / param.BOHR
        self.kpt_latt_loc = self.cell.get_scaled_kpts(self.kmf.kpts)
        self.num_atoms_loc = self.cell.natm
        self.atom_symbols_loc = [atom[0] for atom in self.cell._atom]
        self.atom_atomic_loc = [int(self.cell._atm[atom][0] + self.cell.atom_nelec_core(atom))
                                for atom in range(self.num_atoms_loc)]
        self.atoms_cart_loc = np.asarray([(np.asarray(atom[1])* param.BOHR).tolist()
                                          for atom in self.cell._atom])
        self.gamma_only, self.spinors = (0 , 0)
        if gamma: self.gamma_only = 1
        if spinors: self.spinors = 1

        # Wannier90_setup outputs
        self.num_bands_loc = None
        self.num_wann_loc = None
        self.nntot_loc = None
        self.nn_list = None
        self.proj_site = None
        self.proj_l = None
        self.proj_m = None
        self.proj_radial = None
        self.proj_z = None
        self.proj_x = None
        self.proj_zona = None
        self.exclude_bands = None
        self.proj_s = None
        self.proj_s_qaxis = None

        # Input for Wannier90_run
        self.band_included_list = None
        self.A_matrix_loc = None
        self.M_matrix_loc = None
        self.eigenvalues_loc = None

        # Wannier90_run outputs
        self.U_matrix = None
        self.U_matrix_opt = None
        self.lwindow = None
        self.wann_centres = None
        self.wann_spreads = None
        self.spread = None

        # Others
        self.use_bloch_phases = False
        self.spin_up = spin_up
        self.mo_energy_kpts = []
        self.mo_coeff_kpts = []
        if np.mod(self.cell.nelectron,2) !=0:
            if spin_up:
                for kpt in range(self.num_kpts_loc):
                    self.mo_energy_kpts.append(self.kmf.mo_energy_kpts[0][kpt][:self.num_bands_tot])
                    self.mo_coeff_kpts.append(self.kmf.mo_coeff_kpts[0][kpt][:,:self.num_bands_tot])
            else:
                for kpt in range(self.num_kpts_loc):
                    self.mo_energy_kpts.append(self.kmf.mo_energy_kpts[1][kpt][:self.num_bands_tot])
                    self.mo_coeff_kpts.append(self.kmf.mo_coeff_kpts[1][kpt][:,:self.num_bands_tot])
        else:

            for kpt in range(self.num_kpts_loc):
                self.mo_energy_kpts.append(self.kmf.mo_energy_kpts[kpt][:self.num_bands_tot])
                self.mo_coeff_kpts.append(self.kmf.mo_coeff_kpts[kpt][:,:self.num_bands_tot])

    def kernel(self, external_AME=None):
        '''
        Main kernel for pyWannier90
        '''
        self.make_win()
        self.setup()
        if external_AME is not None:
            self.M_matrix_loc = self.read_M_mat(external_AME + '.mmn')
            self.A_matrix_loc = self.read_A_mat(external_AME + '.amn')
            self.eigenvalues_loc = self.read_epsilon_mat(external_AME + '.eig')
        else:
            self.M_matrix_loc = self.get_M_mat()
            self.A_matrix_loc = self.get_A_mat()
            self.eigenvalues_loc = self.get_epsilon_mat()
        self.run()

    def make_win(self):
        '''
        Make a basic *.win file for wannier90
        '''

        win_file = open('wannier90.win', "w")
        win_file.write('! Basic input generated by the pyWannier90. Date: %s\n' % (time.ctime()))
        win_file.write('\n')
        win_file.write('num_bands       = %d\n' % (self.num_bands_tot))
        win_file.write('num_wann       = %d\n' % (self.num_wann))
        win_file.write('\n')
        win_file.write('Begin Unit_Cell_Cart\n')
        for row in range(3):
            win_file.write('%10.7f  %10.7f  %10.7f\n' %
                           (self.real_lattice_loc[0, row], self.real_lattice_loc[1, row],
                            self.real_lattice_loc[2, row]))
        win_file.write('End Unit_Cell_Cart\n')
        win_file.write('\n')
        win_file.write('Begin atoms_cart\n')
        for atom in range(len(self.atom_symbols_loc)):
            win_file.write('%s  %7.7f  %7.7f  %7.7f\n' %
                           (self.atom_symbols_loc[atom], self.atoms_cart_loc[atom,0],
                            self.atoms_cart_loc[atom,1], self.atoms_cart_loc[atom,2]))
        win_file.write('End atoms_cart\n')
        win_file.write('\n')
        if self.use_bloch_phases: win_file.write('use_bloch_phases = T\n\n')
        if self.keywords is not None:
            win_file.write('!Additional keywords\n')
            win_file.write(self.keywords)
        win_file.write('\n\n\n')
        win_file.write('mp_grid        = %d %d %d\n' %
                       (self.mp_grid_loc[0], self.mp_grid_loc[1], self.mp_grid_loc[2]))
        if self.gamma_only == 1: win_file.write('gamma_only : true\n')
        win_file.write('begin kpoints\n')
        for kpt in range(self.num_kpts_loc):
            win_file.write('%7.7f  %7.7f  %7.7f\n' %
                           (self.kpt_latt_loc[kpt][0], self.kpt_latt_loc[kpt][1], self.kpt_latt_loc[kpt][2]))
        win_file.write('End Kpoints\n')
        win_file.close()

    def get_M_mat(self):
        r'''
        Construct the ovelap matrix: M_{m,n}^{(\mathbf{k,b})}
        Equation (25) in MV, Phys. Rev. B 56, 12847
        '''

        M_matrix_loc = np.empty([self.num_kpts_loc, self.nntot_loc,
                                 self.num_bands_loc, self.num_bands_loc],
                                dtype = np.complex128)

        for k_id in range(self.num_kpts_loc):
            for nn in range(self.nntot_loc):
                k1 = self.cell.get_abs_kpts(self.kpt_latt_loc[k_id])
                k_id2 = self.nn_list[nn, k_id, 0] - 1
                k2_ = self.kpt_latt_loc[k_id2]
                k2_scaled = k2_ + self.nn_list[nn, k_id, 1:4]
                k2 = self.cell.get_abs_kpts(k2_scaled)
                s_AO = df.ft_ao.ft_aopair(self.cell, -k2+k1, kpti_kptj=[k2,k1], q = np.zeros(3))[0]
                Cm = self.mo_coeff_kpts[k_id][:,self.band_included_list]
                Cn = self.mo_coeff_kpts[k_id2][:,self.band_included_list]
                M_matrix_loc[k_id, nn,:,:] = lib.einsum('nu,vm,uv->nm', Cn.T.conj(), Cm, s_AO).conj()

        return M_matrix_loc

    def read_M_mat(self, filename=None):
        r'''
        Read the ovelap matrix: M_{m,n}^{(\mathbf{k,b})} from seedname.mnn
        '''
        if filename is None: filename = 'wannier90.mmn'
        assert os.path.exists(filename), "Cannot find " + filename

        with open(filename, 'r') as f:
            data = f.readlines()
            num_bands_loc, num_kpts_loc, nntot_loc = np.int64(data[1].split())
            data = data[2:]
            nn_list = []
            nline = num_bands_loc**2 + 1
            M_matrix_loc = np.empty([num_kpts_loc, nntot_loc, num_bands_loc, num_bands_loc], dtype = np.complex128)
            jump = 0
            for kpt in range(num_kpts_loc):
                for nn in range(nntot_loc):
                    temp = data[jump : jump + nline]
                    nn_list.append(np.int64(temp[0].split()))
                    val_in_float = np.float64(" ".join(temp[1:]).split()).reshape(-1,2)
                    val_in_complex = val_in_float[:,0] + 1j * val_in_float[:,1]
                    M_matrix_loc[kpt, nn] = val_in_complex.reshape(num_bands_loc, num_bands_loc)
                    jump += nline

        return M_matrix_loc

    def get_A_mat(self):
        r'''
        Construct the projection matrix: A_{m,n}^{\mathbf{k}}
        Equation (62) in MV, Phys. Rev. B 56, 12847 or equation (22) in SMV, Phys. Rev. B 65, 035109
        '''

        A_matrix_loc = np.empty([self.num_kpts_loc, self.num_wann_loc, self.num_bands_loc], dtype = np.complex128)

        if self.use_bloch_phases:
            Amn = np.zeros([self.num_wann_loc, self.num_bands_loc])
            np.fill_diagonal(Amn, 1)
            A_matrix_loc[:,:,:] = Amn
        else:
            from pyscf.dft import numint as mol_numint
            from pyscf.dft import gen_grid as mol_gen_grid
            grids = mol_gen_grid.Grids(self.cell).build()
            coords = grids.coords
            weights = grids.weights
            for ith_wann in range(self.num_wann_loc):
                frac_site = self.proj_site[ith_wann]
                abs_site = frac_site.dot(self.real_lattice_loc) / param.BOHR
                l = self.proj_l[ith_wann]
                mr = self.proj_m[ith_wann]
                r = self.proj_radial[ith_wann]
                zona = self.proj_zona[ith_wann]
                x_axis = self.proj_x[ith_wann]
                z_axis = self.proj_z[ith_wann]
                gr = g_r(coords, abs_site, l, mr, r, zona, x_axis, z_axis, unit = 'B')
                ao_L0 = mol_numint.eval_ao(self.cell, coords)
                s_aoL0_g = lib.einsum('i,i,iv->v', weights, gr, ao_L0)
                for k_id in range(self.num_kpts_loc):
                    kpt = self.cell.get_abs_kpts(self.kpt_latt_loc[k_id])
                    mo_included = self.mo_coeff_kpts[k_id][:,self.band_included_list]
                    s_kpt = self.cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpt, pbcopt=lib.c_null_ptr())
                    A_matrix_loc[k_id,ith_wann,:] = lib.einsum('v,vu,um->m', s_aoL0_g, s_kpt, mo_included,
                                                               optimize=True).conj()

        return A_matrix_loc

    def read_A_mat(self, filename=None):
        r'''
        Read the ovelap matrix: M_{m,n}^{(\mathbf{k,b})} from seedname.mnn
        '''
        if filename is None: filename = 'wannier90.amn'
        assert os.path.exists(filename), "Cannot find " + filename

        with open(filename, 'r') as f:
            data = f.readlines()
            num_bands_loc, num_kpts_loc, num_wann_loc = np.int64(data[1].split())
            data = data[2:]
            A_matrix_loc = np.empty([num_kpts_loc, num_wann_loc, num_bands_loc], dtype = np.complex128)
            val_in_float = np.float64(" ".join(data).split()).reshape(-1,5)
            A_matrix_loc = (val_in_float[:,3] + 1j * val_in_float[:,4]).reshape(num_kpts_loc, num_wann_loc, num_bands_loc)

        return A_matrix_loc

    def get_epsilon_mat(self):
        r'''
        Construct the eigenvalues matrix: \epsilon_{n}^(\mathbf{k})
        '''

        return np.asarray(self.mo_energy_kpts, dtype=np.float64)[:,self.band_included_list] * param.HARTREE2EV

    def read_epsilon_mat(self, filename=None):
        r'''
        Read the eigenvalues matrix: \epsilon_{n}^(\mathbf{k})
        '''
        if filename is None: filename = 'wannier90.eig'
        assert os.path.exists(filename), "Cannot find " + filename
        with open(filename, 'r') as f:
            data = f.read()
            temp = np.float64(data.split()).reshape(-1, 3)
            nbands = int(temp[:,0].max())
            nkpts = int(temp[:,1].max())
            eigenvals = temp[:,2].reshape(nkpts, nbands)

        return eigenvals

    def setup(self):
        '''
        Execute the Wannier90_setup
        '''

        real_lattice_loc = self.real_lattice_loc.T.flatten()
        recip_lattice_loc = self.recip_lattice_loc.T.flatten()
        kpt_latt_loc = self.kpt_latt_loc.flatten()
        atoms_cart_loc = self.atoms_cart_loc.flatten()

        (bands_wann_nntot, nn_list, proj_site, proj_l, proj_m, proj_radial,
         proj_z, proj_x, proj_zona, exclude_bands, proj_s, proj_s_qaxis) = \
                libwannier90.setup(self.mp_grid_loc, self.num_kpts_loc, real_lattice_loc,
                                   recip_lattice_loc, kpt_latt_loc,
                                   self.num_bands_tot, self.num_atoms_loc,
                                   self.atom_atomic_loc, atoms_cart_loc, self.gamma_only, self.spinors)

        # Convert outputs to the correct data type
        self.num_bands_loc, self.num_wann_loc, self.nntot_loc = np.int32(bands_wann_nntot)
        self.nn_list = np.int32(nn_list)
        self.proj_site = proj_site
        self.proj_l = np.int32(proj_l)
        self.proj_m = np.int32(proj_m)
        self.proj_radial = np.int32(proj_radial)
        self.proj_z = proj_z
        self.proj_x = proj_x
        self.proj_zona = proj_zona
        self.exclude_bands = np.int32(exclude_bands)
        self.band_included_list = [i for i in range(self.num_bands_tot) if (i + 1) not in self.exclude_bands]
        self.proj_s = np.int32(proj_s)
        self.proj_s_qaxis = proj_s_qaxis

    def run(self):
        '''
        Execute the Wannier90_run
        '''

        assert self.num_wann_loc is not None
        assert isinstance(self.M_matrix_loc, np.ndarray)
        assert isinstance(self.A_matrix_loc, np.ndarray)
        assert isinstance(self.eigenvalues_loc, np.ndarray)

        real_lattice_loc = self.real_lattice_loc.T.flatten()
        recip_lattice_loc = self.recip_lattice_loc.T.flatten()
        kpt_latt_loc = self.kpt_latt_loc.flatten()
        atoms_cart_loc = self.atoms_cart_loc.flatten()
        M_matrix_loc = self.M_matrix_loc.flatten()
        A_matrix_loc = self.A_matrix_loc.flatten()
        eigenvalues_loc = self.eigenvalues_loc.flatten()

        U_matrix, U_matrix_opt, lwindow, wann_centres, wann_spreads, spread = \
                libwannier90.run(self.mp_grid_loc, self.num_kpts_loc, real_lattice_loc,
                                 recip_lattice_loc, kpt_latt_loc, self.num_bands_loc,
                                 self.num_wann_loc, self.nntot_loc, self.num_atoms_loc,
                                 self.atom_atomic_loc, atoms_cart_loc, self.gamma_only,
                                 M_matrix_loc, A_matrix_loc, eigenvalues_loc)

        # Convert outputs to the correct data typ
        self.U_matrix = U_matrix
        self.U_matrix_opt = U_matrix_opt
        lwindow = np.int32(np.abs(lwindow.real))
        self.lwindow = (lwindow == 1)
        self.wann_centres = wann_centres.real
        self.wann_spreads = wann_spreads.real
        self.spread = spread.real

    get_wigner_seitz_supercell = get_wigner_seitz_supercell
    R_wz_sc = R_wz_sc
    ws_translate_dist = ws_translate_dist

    def get_hamiltonian_kpts(self):
        '''Get the Hamiltonian in k-space, this should be identical to Fock matrix from PySCF'''

        assert self.U_matrix is not None, "You must wannierize first, then you can run this function"
        eigenvals_in_window = []
        for k_id in range(self.num_kpts_loc):
            mo_included = self.mo_energy_kpts[k_id][self.band_included_list]
            orbs_in_win = self.lwindow[k_id]
            mo_in_window = mo_included[orbs_in_win]
            U_matrix_opt = self.U_matrix_opt[k_id][ :, orbs_in_win].T
            eigenvals = lib.einsum('m,mo,mo->o', mo_in_window, U_matrix_opt.conj(), U_matrix_opt)
            eigenvals_in_window.append(eigenvals)

        hamiltonian_kpts = lib.einsum('kso,ko,kto->kst', self.U_matrix.conj(), eigenvals_in_window, self.U_matrix)
        return hamiltonian_kpts

    def get_hamiltonian_Rs(self, Rs, ham_kpts=None):
        '''Get the R-space Hamiltonian H(R0, R) centered at R0 or the first R in Rs list
        '''

        assert self.U_matrix is not None, "You must wannierize first, then you can run this function"
        nkpts = self.kpt_latt_loc.shape[0]
        if ham_kpts is not None:
            hamiltonian_kpts = ham_kpts
        else:
            hamiltonian_kpts = self.get_hamiltonian_kpts()

        # Find the center either R(0,0,0) or the first R in the Rs list
        ngrid = len(Rs)
        center = np.arange(ngrid)[(np.asarray(Rs)**2).sum(axis=1) < 1e-10]
        if center.shape[0] == 1:
            center = center[0]
        else:
            center = 0

        # The phase factor is computed using the exp(1j*R.dot(k)) rather than exp(-1j*R.dot(k)) in wannier90
        phase = 1/np.sqrt(nkpts) * np.exp(1j* 2*np.pi * np.dot(Rs, self.kpt_latt_loc.T))
        hamiltonian_R0 = lib.einsum('k,kst,Rk->Rst', phase[center], hamiltonian_kpts, phase.conj())

        return hamiltonian_R0

    def interpolate_ham_kpts(self, frac_kpts, ham_kpts=None,
                             use_ws_distance=True, ws_search_size=[2,2,2],
                             ws_distance_tol=1e-6):
        ''' Interpolate the band structure using the Slater-Koster scheme
            Return:
                eigenvalues and eigenvectors at the desired kpts
        '''

        assert self.U_matrix is not None, "You must wannierize first, then you can run this function"
        ndegen, Rs, center = self.get_wigner_seitz_supercell(ws_search_size, ws_distance_tol)
        hamiltonian_R0 = self.get_hamiltonian_Rs(Rs, ham_kpts)

        # Interpolate H(kpts) at the desired k-pts
        if use_ws_distance:
            wdist_ndeg, wdist_ndeg_, irdist_ws, crdist_ws = self.ws_translate_dist(Rs)
            temp = lib.einsum('iRstx,kx->iRstk', irdist_ws, frac_kpts)
            phase = lib.einsum('iRstk,iRst->Rstk', np.exp(1j* 2*np.pi * temp), wdist_ndeg_)
            inter_hamiltonian_kpts = \
                    lib.einsum('R,Rst,Rts,Rstk->kst', 1/ndegen, 1/wdist_ndeg, hamiltonian_R0, phase)
        else:
            phase = np.exp(1j* 2*np.pi * np.dot(Rs, frac_kpts.T))
            inter_hamiltonian_kpts = \
                    lib.einsum('R,Rst,Rk->kst', 1/ndegen, hamiltonian_R0, phase)

        return inter_hamiltonian_kpts

    def interpolate_band(self, frac_kpts, ham_kpts=None, use_ws_distance=True,
                         ws_search_size=[2,2,2], ws_distance_tol=1e-6):
        ''' Interpolate the band structure using the Slater-Koster scheme
            Return:
                eigenvalues and eigenvectors at the desired kpts
        '''

        assert self.U_matrix is not None, (
            "You must wannierize first, then you can run this function")
        inter_hamiltonian_kpts = self.interpolate_ham_kpts(
            frac_kpts, ham_kpts, use_ws_distance, ws_search_size, ws_distance_tol)
        # Diagonalize H(kpts) to get eigenvalues and eigenvector
        nkpts = frac_kpts.shape[0]
        eigvals, eigvecs = np.linalg.eigh(inter_hamiltonian_kpts)
        idx_kpts = eigvals.argsort()
        eigvals = np.asarray([eigvals[kpt][idx_kpts[kpt]] for kpt in range(nkpts)])
        eigvecs = np.asarray([eigvecs[kpt][:,idx_kpts[kpt]] for kpt in range(nkpts)])

        return eigvals, eigvecs

    def is_real(self, threshold=1.e-6):
        '''
        Fourier transform the mo coefficients to real space and check if it is real
        '''

        assert self.U_matrix is not None, "You must wannierize first, then you can run this function"
        eigenvecs_in_window = []
        for k_id in range(self.num_kpts_loc):
            mo_included = self.mo_coeff_kpts[k_id][:,self.band_included_list]
            orbs_in_win = self.lwindow[k_id]
            mo_in_window = mo_included[:, orbs_in_win].dot(self.U_matrix_opt[k_id][ :, orbs_in_win].T)
            eigenvecs_in_window.append(mo_in_window)

        # Rotate the mo(kpts) into localized basis
        rotated_mo_coeff_kpts = lib.einsum('kum,ksm->kus', eigenvecs_in_window, self.U_matrix)

        # Fourier transform the localized mo
        nkx, nky, nkz = self.mp_grid_loc
        Ts = lib.cartesian_prod((np.arange(nkx), np.arange(nky), np.arange(nkz)))
        nkpts = self.kpt_latt_loc.shape[0]
        phase = 1/np.sqrt(nkpts) * np.exp(1j* 2*np.pi * np.dot(Ts, self.kpt_latt_loc.T))
        mo_coeff_Rs = lib.einsum('k,kus,Rk->Rus', phase[0], rotated_mo_coeff_kpts, phase.conj())

        return mo_coeff_Rs.imag.max() < threshold

    def export_unk(self, grid=[50,50,50]):
        '''
        Export the periodic part of BF in a real space grid for plotting with wannier90
        '''

        grids_coor, weights = periodic_grid(self.cell, grid, order = 'F')

        for k_id in range(self.num_kpts_loc):
            if self.spin_up:
                spin = '.1'
            else:
                spin = '.2'
            kpt = self.cell.get_abs_kpts(self.kpt_latt_loc[k_id])
            ao = numint.eval_ao(self.cell, grids_coor, kpt = kpt)
            u_ao = lib.einsum('x,xi->xi', np.exp(-1j*np.dot(grids_coor, kpt)), ao)
            unk_file = FortranFile('UNK' + "%05d" % (k_id + 1) + spin, 'w')
            unk_file.write_record(np.asarray([grid[0], grid[1], grid[2], k_id + 1, self.num_bands_loc], dtype = np.int32))
            mo_included = self.mo_coeff_kpts[k_id][:,self.band_included_list]
            u_mo = lib.einsum('xi,in->xn', u_ao, mo_included)
            for band in range(len(self.band_included_list)):
                unk_file.write_record(np.asarray(u_mo[:,band], dtype = np.complex128))
            unk_file.close()

    def export_AME(self, grid=[50,50,50]):
        r'''
        Export A_{m,n}^{\mathbf{k}} and M_{m,n}^{(\mathbf{k,b})} and \epsilon_{n}^(\mathbf{k})
        '''

        if self.A_matrix_loc is None:
            self.make_win()
            self.setup()
            self.M_matrix_loc = self.get_M_mat()
            self.A_matrix_loc = self.get_A_mat()
            self.eigenvalues_loc = self.get_epsilon_mat()
            self.export_unk(self, grid = grid)

        with open('wannier90.mmn', 'w') as f:
            f.write('Generated by the pyWannier90. Date: %s\n' % (time.ctime()))
            f.write('    %d    %d    %d\n' % (self.num_bands_loc, self.num_kpts_loc, self.nntot_loc))

            for k_id in range(self.num_kpts_loc):
                for nn in range(self.nntot_loc):
                    k_id1 = k_id + 1
                    k_id2 = self.nn_list[nn, k_id, 0]
                    nnn, nnm, nnl = self.nn_list[nn, k_id, 1:4]
                    f.write('    %d  %d    %d  %d  %d\n' % (k_id1, k_id2, nnn, nnm, nnl))
                    for m in range(self.num_bands_loc):
                        for n in range(self.num_bands_loc):
                            f.write('    %22.18e  %22.18e\n' % (self.M_matrix_loc[k_id, nn,m,n].real,
                                                                self.M_matrix_loc[k_id, nn,m,n].imag))

        with open('wannier90.amn', 'w') as f:
            f.write('Generated by the pyWannier90. Date: %s\n' % (time.ctime()))
            f.write('    %d    %d    %d\n' % (self.num_bands_loc, self.num_kpts_loc, self.num_wann_loc))

            for k_id in range(self.num_kpts_loc):
                for ith_wann in range(self.num_wann_loc):
                    for band in range(self.num_bands_loc):
                        f.write('    %d    %d    %d    %22.18e    %22.18e\n' %
                                (band+1, ith_wann+1, k_id+1,
                                 self.A_matrix_loc[k_id,ith_wann,band].real,
                                 self.A_matrix_loc[k_id,ith_wann,band].imag))

        with open('wannier90.eig', 'w') as f:
            for k_id in range(self.num_kpts_loc):
                for band in range(self.num_bands_loc):
                    f.write('    %d    %d    %22.18e\n' % (band+1, k_id+1, self.eigenvalues_loc[k_id,band]))

    def get_wannier(self, supercell=[1,1,1], grid=[50,50,50]):
        '''
        Evaluate the MLWF using a periodic grid
        '''

        grids_coor, weights = periodic_grid(self.cell, grid, supercell = [1,1,1], order = 'C')
        kpts = self.cell.get_abs_kpts(self.kpt_latt_loc)

        u_mo  = []
        for k_id in range(self.num_kpts_loc):
            mo_included = self.mo_coeff_kpts[k_id][:,self.band_included_list]
            mo_in_window = self.lwindow[k_id]
            C_opt = mo_included[:,mo_in_window].dot(self.U_matrix_opt[k_id][ :, mo_in_window].T)
            C_tildle = C_opt.dot(self.U_matrix[k_id].T)
            kpt = kpts[k_id]
            ao = numint.eval_ao(self.cell, grids_coor, kpt = kpt)
            u_ao = lib.einsum('x,xi->xi', np.exp(-1j*np.dot(grids_coor, kpt)), ao)
            u_mo.append(lib.einsum('xi,in->xn', u_ao, C_tildle))

        u_mo = np.asarray(u_mo)
        WF0 = libwannier90.get_WF0s(self.kpt_latt_loc.shape[0],self.kpt_latt_loc, supercell, grid, u_mo)

        # Fix the global phase following the pw2wannier90 procedure
        max_index = (WF0*WF0.conj()).real.argmax(axis=0)
        norm_wfs = np.diag(WF0[max_index,:])
        norm_wfs = norm_wfs/np.absolute(norm_wfs)
        WF0 = WF0/norm_wfs/self.num_kpts_loc

        # Check the 'reality' following the pw2wannier90 procedure
        for WF_id in range(self.num_wann_loc):
            ratio_max = np.abs(WF0[np.abs(WF0[:,WF_id].real) >= 0.01,WF_id].imag /
                               WF0[np.abs(WF0[:,WF_id].real) >= 0.01,WF_id].real).max(axis=0)
            print('The maximum imag/real for wannier function ', WF_id,' : ', ratio_max)
        return WF0

    def get_guess_orb(self, frac_site=[0,0,0], l=0, mr=1, r=1,
                      zona=1.0, x_axis=[1,0,0], z_axis=[0,0,1],
                      supercell=[1,1,1], grid=[50,50,50]):
        '''
        Evaluate a guess orbital using a periodic uniform grid
        '''
        grids_coor, weights = periodic_grid(self.cell, grid, supercell = supercell, order = 'C')
        frac_site = np.asarray(frac_site)
        abs_site = frac_site.dot(self.real_lattice_loc) / param.BOHR
        gr = g_r(grids_coor, abs_site, l, mr, r, zona, x_axis, z_axis, unit = 'B')
        return gr

    def plot_wf(self, outfile='MLWF', wf_list=None, supercell=[1,1,1], grid=[50,50,50]):
        '''
        Export Wannier function at cell R
        xsf format: http://web.mit.edu/xcrysden_v1.5.60/www/XCRYSDEN/doc/XSF.html
        Attributes:
            wf_list        : a list of MLWFs to plot
            supercell    : a supercell used for plotting
        '''

        if wf_list is None:
            wf_list = list(range(self.num_wann_loc))

        grid = np.asarray(grid)
        origin = np.asarray([-(grid[i]*(supercell[i]//2) + 1)/grid[i] for i in range(3)]).dot(
            self.cell.lattice_vectors().T)* param.BOHR
        real_lattice_loc = (grid*supercell-1)/grid * self.cell.lattice_vectors() * param.BOHR
        nx, ny, nz = grid*supercell
        WF0 = self.get_wannier(supercell = supercell, grid = grid)


        for wf_id in wf_list:
            assert wf_id in list(range(self.num_wann_loc))
            WF = WF0[:,wf_id].reshape(nx,ny,nz).real

            with open(outfile + '-' + str(wf_id) + '.xsf', 'w') as f:
                f.write('Generated by the pyWannier90. Date: %s\n\n' % (time.ctime()))
                f.write('CRYSTAL\n')
                f.write('PRIMVEC\n')
                for row in range(3):
                    f.write('%10.7f  %10.7f  %10.7f\n' %
                            (self.real_lattice_loc[row,0], self.real_lattice_loc[row,1],
                             self.real_lattice_loc[row,2]))
                f.write('CONVVEC\n')
                for row in range(3):
                    f.write('%10.7f  %10.7f  %10.7f\n' %
                            (self.real_lattice_loc[row,0], self.real_lattice_loc[row,1],
                             self.real_lattice_loc[row,2]))
                f.write('PRIMCOORD\n')
                f.write('%3d %3d\n' % (self.num_atoms_loc, 1))
                for atom in range(len(self.atom_symbols_loc)):
                    f.write('%s  %7.7f  %7.7f  %7.7f\n' %
                            (self.atom_symbols_loc[atom], self.atoms_cart_loc[atom][0],
                             self.atoms_cart_loc[atom][1], self.atoms_cart_loc[atom][2]))
                f.write('\n\n')
                f.write('BEGIN_BLOCK_DATAGRID_3D\n3D_field\nBEGIN_DATAGRID_3D_UNKNOWN\n')
                f.write('   %5d     %5d  %5d\n' % (nx, ny, nz))
                f.write('   %10.7f  %10.7f  %10.7f\n' % (origin[0],origin[1],origin[2]))
                for row in range(3):
                    f.write('   %10.7f  %10.7f  %10.7f\n' %
                            (real_lattice_loc[row,0], real_lattice_loc[row,1], real_lattice_loc[row,2]))

                fmt = ' %13.5e' * nx + '\n'
                for iz in range(nz):
                    for iy in range(ny):
                        f.write(fmt % tuple(WF[:,iy,iz].tolist()))
                f.write('END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D')

    def plot_guess_orbs(self, outfile='guess_orb', frac_site=[0,0,0], l=0, mr=1, r=1,
                        zona=1.0, x_axis=[1,0,0], z_axis=[0,0,1],
                        supercell=[1,1,1], grid=[50,50,50]):
        '''
        Export Wannier function at cell R
        xsf format: http://web.mit.edu/xcrysden_v1.5.60/www/XCRYSDEN/doc/XSF.html
        Attributes:
            wf_list        : a list of MLWFs to plot
            supercell    : a supercell used for plotting
        '''

        grid = np.asarray(grid)
        origin = np.asarray([-grid[i]*(supercell[i]//2)/grid[i] for i in range(3)]).dot(
            self.cell.lattice_vectors().T) * param.BOHR
        real_lattice_loc = (grid*supercell-1)/grid * self.cell.lattice_vectors() * param.BOHR
        nx, ny, nz = grid*supercell
        guess_orb = self.get_guess_orb(frac_site=frac_site, l=l, mr=mr, r=r,
                                       zona=zona, x_axis=x_axis, z_axis=z_axis,
                                       supercell=supercell, grid=grid)
        guess_orb = guess_orb.reshape(nx,ny,nz).real

        with open(outfile + '.xsf', 'w') as f:
            f.write('Generated by the pyWannier90\n\n')
            f.write('CRYSTAL\n')
            f.write('PRIMVEC\n')
            for row in range(3):
                f.write('%10.7f  %10.7f  %10.7f\n' % (self.real_lattice_loc[row,0], self.real_lattice_loc[row,1],
                                                      self.real_lattice_loc[row,2]))
            f.write('CONVVEC\n')
            for row in range(3):
                f.write('%10.7f  %10.7f  %10.7f\n' % (self.real_lattice_loc[row,0], self.real_lattice_loc[row,1],
                                                      self.real_lattice_loc[row,2]))
            f.write('PRIMCOORD\n')
            f.write('%3d %3d\n' % (self.num_atoms_loc, 1))
            for atom in range(len(self.atom_symbols_loc)):
                f.write('%s  %7.7f  %7.7f  %7.7f\n' % (self.atom_symbols_loc[atom], self.atoms_cart_loc[atom][0],
                                                       self.atoms_cart_loc[atom][1], self.atoms_cart_loc[atom][2]))
            f.write('\n\n')
            f.write('BEGIN_BLOCK_DATAGRID_3D\n3D_field\nBEGIN_DATAGRID_3D_UNKNOWN\n')
            f.write('   %5d     %5d  %5d\n' % (nx, ny, nz))
            f.write('   %10.7f  %10.7f  %10.7f\n' % (origin[0],origin[1],origin[2]))
            for row in range(3):
                f.write('   %10.7f  %10.7f  %10.7f\n' % (real_lattice_loc[row,0], real_lattice_loc[row,1],
                                                         real_lattice_loc[row,2]))

            fmt = ' %13.5e' * nx + '\n'
            for iz in range(nz):
                for iy in range(ny):
                    f.write(fmt % tuple(guess_orb[:,iy,iz].tolist()))
            f.write('END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D')

if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf
    import pywannier90

    # build cell object
    cell = pgto.Cell()
    cell.a = [[0.0, 2.7155, 2.7155], [2.7155, 0.0, 2.7155], [2.7155, 2.7155, 0.0]]
    cell.atom = [['Si',[0.0,0.0,0.0]], ['Si',[1.35775, 1.35775, 1.35775]]]
    cell.basis = 'gth-dzv'
    cell.pseudo = 'gth-pade'
    cell.exp_to_discard = 0.1
    cell.build()

    # build and run scf object
    kmesh = [3, 1, 1]
    kpts = cell.make_kpts(kmesh)
    kmf = pscf.KKS(cell, kpts)
    kmf.xc = 'pbe'
    kmf.run()

    # build and run w90 object
    num_wann = 8
    keywords = '''
    begin projections
    Si:sp3
    end projections
    '''
    w90 = pywannier90.W90(kmf, cell, kmesh, num_wann, other_keywords=keywords)
    w90.kernel()
