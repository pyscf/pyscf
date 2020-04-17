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
A hacky implementation of electron-phonon matrix from finite difference
'''

from pyscf import scf, dft, gto, hessian
from pyscf.eph.rhf import solve_hmat
import numpy as np
import scipy
from pyscf.lib import logger
import copy

def run_mfs(mf, mols_a, mols_b):
    '''perform a set of calculations on given two sets of molecules'''
    nconfigs = len(mols_a)
    dm0 = mf.make_rdm1()
    mflist = []
    for i in range(nconfigs):
        mf1 = copy.copy(mf)
        mf1.reset(mols_a[i])
        mf2 = copy.copy(mf)
        mf2.reset(mols_b[i])
        mf1.kernel(dm0=dm0)
        mf2.kernel(dm0=dm0)
        if not (mf1.converged):
            logger.warn(mf, "%ith config mf1 not converged", i)
        if not (mf2.converged):
            logger.warn(mf, "%ith config mf2 not converged", i)
        mflist.append((mf1, mf2))
    return mflist

def get_mode(mf):
    hmat = mf.Hessian().kernel()
    w_new, c_new = solve_hmat(mf.mol, hmat)
    return w_new, c_new

def gen_moles(mol, disp):
    """From the given equilibrium molecule, generate 3N molecules with a shift on + displacement(mol_a) and - displacement(mol_s) on each Cartesian coordinates"""
    coords = mol.atom_coords()
    natoms = len(coords)
    mol_a, mol_s = [],[]
    for i in range(natoms):
        for x in range(3):
            new_coords_a, new_coords_s = coords.copy(), coords.copy()
            new_coords_a[i][x] += disp
            new_coords_s[i][x] -= disp
            atoma = [[mol.atom_symbol(j), coord] for (j, coord) in zip(range(natoms), new_coords_a)]
            atoms = [[mol.atom_symbol(j), coord] for (j, coord) in zip(range(natoms), new_coords_s)]
            mol_a.append(mol.set_geom_(atoma, inplace=False, unit='B'))
            mol_s.append(mol.set_geom_(atoms, inplace=False, unit='B'))
    return mol_a, mol_s

def get_vmat(mf, mfset, disp):
    '''
    computing <u|dVxc/dR|v>
    '''
    vmat=[]
    mygrad = mf.nuc_grad_method()
    ve = mygrad.get_veff() + mygrad.get_hcore() + mf.mol.intor("int1e_ipkin")
    RESTRICTED = (ve.ndim==3)
    aoslice = mf.mol.aoslice_by_atom()
    for ki, (mf1, mf2) in enumerate(mfset):
        atmid, axis = np.divmod(ki, 3)
        p0, p1 = aoslice[atmid][2:]
        vfull1 = mf1.get_veff() + mf1.get_hcore() - mf1.mol.intor_symmetric('int1e_kin')  # <u+|V+|v+>
        vfull2 = mf2.get_veff() + mf2.get_hcore() - mf2.mol.intor_symmetric('int1e_kin')  # <u-|V-|v->
        vfull = (vfull1 - vfull2)/disp  # (<p+|V+|q+>-<p-|V-|q->)/dR
        if RESTRICTED:
            vfull[p0:p1] -= ve[axis,p0:p1]
            vfull[:,p0:p1] -= ve[axis,p0:p1].T
        else:
            vfull[:,p0:p1] -= ve[:,axis,p0:p1]
            vfull[:,:,p0:p1] -= ve[:,axis,p0:p1].transpose(0,2,1)
        vmat.append(vfull)

    return np.asarray(vmat)

def kernel(mf, disp=1e-5, mo_rep=False):
    if hasattr(mf, 'xc'): mf.grids.build()
    if not mf.converged: mf.kernel()
    RESTRICTED = (mf.mo_coeff.ndim==2)
    mol = mf.mol
    omega, vec = get_mode(mf)
    mass = mol.atom_mass_list() * 1836.15
    nmodes, natoms = len(omega), len(mass)
    vec = vec.reshape(natoms, 3, nmodes)
    for i in range(natoms):
        for j in range(nmodes):
            vec[i,:,j] /= np.sqrt(2*mass[i]*omega[j])
    vec = vec.reshape(3*natoms,nmodes)
    mols_a, mols_b = gen_moles(mol, disp/2.0) # generate a bunch of molecules with disp/2 on each cartesion coord
    mfset = run_mfs(mf, mols_a, mols_b) # run mean field calculations on all these molecules
    vmat = get_vmat(mf, mfset, disp) # extracting <p|dV|q>/dR
    if mo_rep:
        if RESTRICTED:
            vmat = np.einsum('xuv,up,vq->xpq', vmat, mf.mo_coeff.conj(), mf.mo_coeff)
        else:
            vmat = np.einsum('xsuv,sup,svq->xspq', vmat, mf.mo_coeff.conj(), mf.mo_coeff)

    if RESTRICTED:
        mat = np.einsum('xJ,xpq->Jpq', vec, vmat)
    else:
        mat = np.einsum('xJ,xspq->sJpq', vec, vmat)
    return mat, omega

if __name__ == '__main__':
    mol = gto.M()
    mol.atom = '''O 0.000000000000  0.00000000136 0.459620634131
                  H 0.000000000000 -0.77050867841 1.139170094494
                  H 0.000000000000  0.77050867841 1.139170094494'''

    mol.unit = 'angstrom'
    mol.basis = 'sto3g'
    mol.verbose=4
    mol.build() # this is a pre-computed relaxed geometry

    mf = dft.RKS(mol)
    mf.grids.level=4
    mf.grids.build()
    mf.xc = 'b3lyp'
    mf.conv_tol = 1e-14
    mf.conv_tol_grad = 1e-8
    mf.kernel()

    grad = mf.nuc_grad_method().kernel()
    print("Force on the atoms/au:")
    print(grad)
    assert(abs(grad).max()<1e-5)

    mat, omega = kernel(mf)
    matmo, _ = kernel(mf, mo_rep=True)

    from pyscf.eph.rks import EPH
    myeph = EPH(mf)
    eph, _ = myeph.kernel()
    ephmo, _ = myeph.kernel(mo_rep=True)
    print("***Testing on RKS***")
    for i in range(len(mat)):
        print("AO",min(np.linalg.norm(eph[i]-mat[i]), np.linalg.norm(eph[i]+mat[i])))
        print("AO", min(abs(eph[i]-mat[i]).max(), abs(eph[i]+mat[i]).max()))
        print("MO",min(np.linalg.norm(ephmo[i]-matmo[i]), np.linalg.norm(ephmo[i]+matmo[i])))
        print("MO", min(abs(ephmo[i]-matmo[i]).max(), abs(ephmo[i]+matmo[i]).max()))
