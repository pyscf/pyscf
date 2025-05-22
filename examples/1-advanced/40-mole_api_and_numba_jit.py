#!/usr/bin/env python

'''
This example demonstrates some APIs of the Mole class and utilizes Numba JIT
compilation to compute integrals. The evaluated integrals are the overlap of
absolute value of Cartesian GTOs.

The overlap integral between the absolute values of Cartesian GTOs can be
computed as the square root of the overlap of four orbitals: sqrt(<ii|jj>) .

This trick can only be applied to Cartesian orbitals. These functions cannot be
transformed to spherical basis.

See relevant discussions in https://github.com/pyscf/pyscf/issues/2805
'''

import numpy as np
import numba

@numba.njit(cache=True)
def primitive_overlap(li, lj, ai, aj, ci, cj, Ra, Rb) -> np.ndarray:
    norm_fac = (np.pi/(ai+aj))**1.5 * ci * cj
    # Unconventional normalization for Cartesian functions in PySCF
    if li <= 1: norm_fac *= ((li*2+1)/(4*np.pi))**.5
    if lj <= 1: norm_fac *= ((lj*2+1)/(4*np.pi))**.5

    # Mapping chi^2 to one orbital r^(l*2)*exp(-2*alpha r^2)
    ai = ai * 2
    aj = aj * 2
    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    Rpa = Rp - Ra
    theta_ij = ai * aj / aij
    lij = li + lj
    # Three Cartesian components of the intermediates: Ix, Iy, Iz
    I = np.empty((lij*2+1, lj*2+1, 3))
    I[0,0] = np.exp(-theta_ij * Rab**2)
    # VRR
    for i in range(lij*2):
        if i == 0: I[1,0] = Rpa * I[0,0]
        else: I[i+1,0] = Rpa * I[i,0] + i/(2*aij) * I[i-1,0]
    # HRR
    for j in range(1, lj*2+1):
        for i in range(lij*2+1-j):
            I[i,j] = Rab * I[i,j-1] + I[i+1,j-1]
    # sqrt(<ii|jj>)
    I = I[::2,::2]**.5

    nfi = (li+1)*(li+2)//2
    nfj = (lj+1)*(lj+2)//2
    s = np.empty((nfi, nfj))
    i = 0
    for ix in range(li, -1, -1):
        for iy in range(li-ix, -1, -1):
            iz = li - ix - iy
            j = 0
            for jx in range(lj, -1, -1):
                for jy in range(lj-jx, -1, -1):
                    jz = lj - jx - jy
                    s[i,j] = I[ix,jx,0] * I[iy,jy,1] * I[iz,jz,2] * norm_fac
                    j += 1
            i += 1
    return s

@numba.njit(cache=True)
def primitive_overlap_matrix(ls, exps, norm_coef, bas_coords):
    nbas = len(ls)
    dims = [(l + 1) * (l + 2) // 2 for l in ls]
    nao = sum(dims)
    smat = np.empty((nao, nao))
    i0 = 0
    for i in range(nbas):
        j0 = 0
        for j in range(i+1):
            s = primitive_overlap(ls[i], ls[j], exps[i], exps[j],
                                  norm_coef[i], norm_coef[j],
                                  bas_coords[i], bas_coords[j])
            smat[i0:i0+dims[i], j0:j0+dims[j]] = s
            # smat is a symmetric matrix
            if i != j: smat[j0:j0+dims[j], i0:i0+dims[i]] = s.T
            j0 += dims[j]
        i0 += dims[i]
    return smat

def absolute_overlap_matrix(mol):
    assert mol.cart
    # Integrals are computed using primitive GTOs. ctr_mat transforms the
    # primitive GTOs to the contracted GTOs.
    pmol, ctr_mat = mol.decontract_basis(aggregate=True)
    # Angular momentum for each shell
    ls = np.array([pmol.bas_angular(i) for i in range(pmol.nbas)])
    # need to access only one exponent for primitive gaussians
    exps = np.array([pmol.bas_exp(i)[0] for i in range(pmol.nbas)])
    # Normalization coefficients
    norm_coef = gto.gto_norm(ls, exps)
    # Position for each shell
    bas_coords = np.array([pmol.bas_coord(i) for i in range(pmol.nbas)])
    s = primitive_overlap_matrix(ls, exps, norm_coef, bas_coords)
    return ctr_mat.T.dot(s).dot(ctr_mat)

if __name__ == '__main__':
    import pyscf
    from pyscf import gto
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1', basis='def2-svp')
    pmol, ctr_mat = mol.decontract_basis(to_cart=True, aggregate=True)
    print(absolute_overlap_matrix(mol))
    print(mol.intor('int1e_ovlp'))
