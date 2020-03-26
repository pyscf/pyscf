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

def copy_mf(mf, mol):
    RESTRICTED=(mf.mo_coeff.ndim==2)
    DFT = hasattr(mf, 'xc')
    if DFT:
        if RESTRICTED:
            mf1 = dft.RKS(mol, xc=mf.xc)
        else:
            mf1 = dft.UKS(mol, xc=mf.xc)
        mf1.grids.level=mf.grids.level
        mf1.conv_tol = mf.conv_tol
        mf1.conv_tol_grad = mf.conv_tol_grad
    else:
        mf1  = copy.copy(mf)
        mf1.reset(mol)
    return mf1

def run_mfs(mf, mols_a, mols_b):
    '''perform a set of calculations on given two sets of molecules'''
    nconfigs = len(mols_a)
    dm0 = mf.make_rdm1()
    mflist = []
    for i in range(nconfigs):
        mf1 = copy_mf(mf, mols_a[i])
        mf2 = copy_mf(mf, mols_b[i])
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
            mol_a.append(mol.set_geom_(atoma, inplace=False))
            mol_s.append(mol.set_geom_(atoms, inplace=False))
    return mol_a, mol_s

def get_v_bra(mf, mf1):
    '''
    computing # <u+|Vxc(0)|v0> + <u0|Vxc(0)|v+>
    '''
    mol, mol1 = mf.mol, mf1.mol # construct a mole that contains both u0 and u+
    atoms = []
    for symbol, pos in mol1._atom:
        atoms.append(('ghost'+symbol, pos))

    fused_mol  = mol.set_geom_(mol._atom+atoms, inplace=False)

    nao = mol.nao_nr()
    dm0 = mf.make_rdm1()
    RESTRICTED = (dm0.ndim==2)

    mf0 = copy_mf(mf, fused_mol)

    if hasattr(mf, 'xc'):
        mf0.grids = mf.grids
    if RESTRICTED:
        dm = np.zeros([2*nao,2*nao]) # construct a fake DM to get Vxc matrix
        dm[:nao,:nao] = dm0
    else:
        dm = np.zeros([2,2*nao,2*nao])
        dm[:,:nao,:nao] = dm0

    veff = mf0.get_veff(fused_mol, dm) #<u*|Vxc(0)|v*> here p* includes both u0 and v+, Vxc is at equilibrium geometry because only the u0 block of the dm is filled
    vnu = mf0.get_hcore(fused_mol) - fused_mol.intor_symmetric('int1e_kin') # <u*|h1|v*>, the kinetic part subtracted
    vtot = veff + vnu

    if RESTRICTED:
        vtot = vtot[nao:,:nao]
        vtot += vtot.T.conj()
    else:
        vtot = vtot[:,nao:,:nao]
        vtot += vtot.transpose(0,2,1).conj()
    return vtot


def get_vmat(mf, mfset, disp):
    '''
    computing <u|dVxc/dR|v>
    '''
    vmat=[]
    for (mf1, mf2) in mfset:
        vfull1 = mf1.get_veff() + mf1.get_hcore() - mf1.mol.intor_symmetric('int1e_kin')  # <u+|V+|v+>
        vfull2 = mf2.get_veff() + mf2.get_hcore() - mf2.mol.intor_symmetric('int1e_kin')  # <u-|V-|v->
        vfull = (vfull1 - vfull2)/disp  # (<p+|V+|q+>-<p-|V-|q->)/dR
        vbra1 = get_v_bra(mf, mf1)   #<p+|V0|q0> + <p0|V0|q+>
        vbra2 = get_v_bra(mf, mf2)   #<p-|V0|q0> + <p0|V0|q->
        vbra = (vbra1-vbra2)/disp

        vtot = vfull - vbra   #<p0|dV0|q0> = d<p|V|q> - <dp|V0|q> - <p|V0|dq>
        vmat.append(vtot)
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
    mol.atom = '''O 0.000000000000 0.000000002577 0.868557119905
                  H 0.000000000000 -1.456050381698 2.152719488376
                  H 0.000000000000 1.456050379121 2.152719486067'''

    mol.unit = 'Bohr'
    mol.basis = 'sto3g'
    mol.verbose=0
    mol.build() # this is a pre-computed relaxed geometry
    from pyscf import hessian
    mf = dft.RKS(mol)
    mf.grids.level=6
    mf.grids.build()
    mf.xc = 'b3lyp'
    mf.conv_tol = 1e-16
    mf.conv_tol_grad = 1e-10
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
