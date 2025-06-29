#!/usr/bin/env python

'''
Calculates the overlap integral between the absolute values of Cartesian GTOs.

This example demonstrates some APIs of the Mole class and utilizes Numba JIT
compilation to compute integrals.

This overlap integral is computed as the product of the Gaussian integrals of
the three components along the three Cartesian directions. Note, these
integrals cannot be transformed to spherical GTOs.

In each component, the integrals are approximately evaluated using
Gauss-Hermite quadrature. The absolute value of the polynomial part is
continuous but not analytical. It cannot be expanded using a limited number of
polynomials. A large number of quadrature roots are required to accurately
evaluate the integrals. In the current implementation, the error is estimated
around ~1e-3.

This example is created to provide an implementation for issue 
https://github.com/pyscf/pyscf/issues/2805

For more technical discussions, please refer to:
"Enhancing PySCF-based Quantum Chemistry Simulations with Modern Hardware,
Algorithms, and Python Tools", arXiv:2506.06661.
This example provides a complete implementation of the code discussed in that paper.
'''

import numpy as np
import numba
from scipy.special import roots_hermite
from pyscf import gto

@numba.njit(cache=True, fastmath=True)
def primitive_overlap(li, lj, ai, aj, ci, cj, Ra, Rb, roots, weights) -> np.ndarray:
    norm_fac = ci * cj
    # Unconventional normalization for Cartesian functions in PySCF
    if li <= 1: norm_fac *= ((li*2+1)/(4*np.pi))**.5
    if lj <= 1: norm_fac *= ((lj*2+1)/(4*np.pi))**.5

    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    theta_ij = ai * aj / aij
    scale = 1./np.sqrt(aij)
    norm_fac *= scale**3 * np.exp(-theta_ij * Rab.dot(Rab))
    x = roots * scale + Rp[:,None]
    xa = x - Ra[:,None]
    xb = x - Rb[:,None]

    # Build mu = xa ** np.arange(li+1)[:,None,None]
    # Build nu = xb ** np.arange(lj+1)[:,None,None]
    nroots = len(weights)
    mu = np.empty((li+1,3,nroots))
    nu = np.empty((lj+1,3,nroots))
    for n in range(nroots):
        powx = 1.
        powy = 1.
        powz = 1.
        mu[0,0,n] = 1.
        mu[0,1,n] = 1.
        mu[0,2,n] = 1.
        for i in range(1, li+1):
            powx = powx * xa[0,n]
            powy = powy * xa[1,n]
            powz = powz * xa[2,n]
            mu[i,0,n] = powx
            mu[i,1,n] = powy
            mu[i,2,n] = powz

        powx = 1.
        powy = 1.
        powz = 1.
        nu[0,0,n] = 1.
        nu[0,1,n] = 1.
        nu[0,2,n] = 1.
        for i in range(1, lj+1):
            powx = powx * xb[0,n]
            powy = powy * xb[1,n]
            powz = powz * xb[2,n]
            nu[i,0,n] = powx
            nu[i,1,n] = powy
            nu[i,2,n] = powz

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
                    Ix = 0
                    Iy = 0
                    Iz = 0
                    for n in range(nroots):
                        Ix += mu[ix,0,n] * nu[jx,0,n] * weights[n]
                        Iy += mu[iy,1,n] * nu[jy,1,n] * weights[n]
                        Iz += mu[iz,2,n] * nu[jz,2,n] * weights[n]
                    s[i,j] = Ix * Iy * Iz * norm_fac
                    j += 1
            i += 1
    return s

@numba.njit(cache=True)
def primitive_overlap_matrix(ls, exps, norm_coef, bas_coords, roots, weights):
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
                                  bas_coords[i], bas_coords[j], roots, weights)
            smat[i0:i0+dims[i], j0:j0+dims[j]] = s
            # smat is a symmetric matrix
            if i != j: smat[j0:j0+dims[j], i0:i0+dims[i]] = s.T
            j0 += dims[j]
        i0 += dims[i]
    return smat

def absolute_overlap_matrix(mol, nroots=500):
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
    r, w = roots_hermite(nroots)
    s = primitive_overlap_matrix(ls, exps, norm_coef, bas_coords, r, w)
    return ctr_mat.T.dot(s).dot(ctr_mat)

def find_instr(func, keyword, sig=0, limit=5):
    count = 0
    for l in func.inspect_asm(func.signatures[sig]).split('\n'):
        if keyword in l:
            count += 1
            print(l)
            if count >= limit:
                break
    if count == 0:
        print('No instructions found')

if __name__ == '__main__':
    import pyscf
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1', basis='def2-svp', cart=True)
    pmol, ctr_mat = mol.decontract_basis(to_cart=True, aggregate=True)
    absolute_overlap_matrix(mol)
