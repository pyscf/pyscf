#!/usr/bin/env python

"""
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
"""

import numpy as np
from numba import njit, prange
from scipy.special import roots_hermite

from pyscf import gto, M


@njit(cache=True, nogil=True)
def gauss_sum(n: int) -> int:
    return n * (n + 1) // 2


@njit(cache=True, nogil=True)
def unravel_symmetric(i: int) -> tuple[int, int]:
    a = int((np.sqrt(8 * i + 1) - 1) // 2)
    offset = gauss_sum(a)
    b = i - offset
    if b > a:
        a, b = b, a
    return a, b


@njit(cache=True, fastmath=True, nogil=True)
def primitive_overlap(li, lj, ai, aj, ci, cj, Ra, Rb, roots, weights) -> np.ndarray:
    norm_fac = ci * cj
    # Unconventional normalization for Cartesian functions in PySCF
    if li <= 1:
        norm_fac *= ((2 * li + 1) / (4 * np.pi)) ** 0.5
    if lj <= 1:
        norm_fac *= ((2 * lj + 1) / (4 * np.pi)) ** 0.5

    aij = ai + aj
    Rab = Ra - Rb
    Rp = (ai * Ra + aj * Rb) / aij
    theta_ij = ai * aj / aij
    scale = 1.0 / np.sqrt(aij)
    norm_fac *= scale**3 * np.exp(-theta_ij * (Rab @ Rab))

    nroots = len(weights)
    x = roots * scale + Rp[:, None]
    xa = x - Ra[:, None]
    xb = x - Rb[:, None]

    mu = np.empty((li + 1, 3, nroots))
    nu = np.empty((lj + 1, 3, nroots))
    mu[0, :, :] = 1.0
    nu[0, :, :] = 1.0

    for d in range(3):
        for p in range(1, li + 1):
            mu[p, d, :] = mu[p - 1, d, :] * xa[d, :]
        for p in range(1, lj + 1):
            nu[p, d, :] = nu[p - 1, d, :] * xb[d, :]

    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    s = np.empty((nfi, nfj))

    i = 0
    for ix in range(li, -1, -1):
        for iy in range(li - ix, -1, -1):
            iz = li - ix - iy
            j = 0
            for jx in range(lj, -1, -1):
                for jy in range(lj - jx, -1, -1):
                    jz = lj - jx - jy

                    Ix = 0.0
                    Iy = 0.0
                    Iz = 0.0
                    for n in range(nroots):
                        w = weights[n]
                        Ix += abs(mu[ix, 0, n] * nu[jx, 0, n]) * w
                        Iy += abs(mu[iy, 1, n] * nu[jy, 1, n]) * w
                        Iz += abs(mu[iz, 2, n] * nu[jz, 2, n]) * w

                    s[i, j] = Ix * Iy * Iz * norm_fac
                    j += 1
            i += 1
    return s


@njit(cache=True, parallel=True)
def primitive_overlap_matrix(ls, exps, norm_coef, bas_coords, roots, weights):
    nbas = len(ls)
    dims = [(l + 1) * (l + 2) // 2 for l in ls]
    nao = sum(dims)
    smat = np.zeros((nao, nao))

    npairs = gauss_sum(nbas)

    for idx in prange(npairs):
        i, j = unravel_symmetric(idx)

        i0 = sum(dims[:i])
        j0 = sum(dims[:j])
        ni = dims[i]
        nj = dims[j]

        s = primitive_overlap(
            ls[i], ls[j], exps[i], exps[j], norm_coef[i], norm_coef[j], bas_coords[i], bas_coords[j], roots, weights
        )
        smat[i0 : i0 + ni, j0 : j0 + nj] = s
        if i != j:
            smat[j0 : j0 + nj, i0 : i0 + ni] = s.T

    return smat


def get_cart_mol(mol):
    return M(atom=mol.atom, basis=mol.basis, charge=mol.charge, spin=mol.spin, cart=True)


def _cart_mol_abs_ovlp_matrix(mol, nroots=500):
    if not mol.cart:
        raise ValueError('Molecule has to use cartesian basis functions.')
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
    assert (s >= 0).all()
    return s, ctr_mat


def approx_S_abs(mol, nroots=500):
    """Compute the approximated absolute overlap matrix.

    The calculation is only exact for uncontracted, cartesian basis functions.
    Since the absolute value is not a linear function, the
    value after contraction and/or transformation to spherical-harmonics is approximated
    via the RHS of the triangle inequality:

    .. math::

        \int |\phi_i(\mathbf{r})| \, |\phi_j(\mathbf{r})| \, d\mathbf{r}
        \leq
        \sum_{\alpha,\beta} |c_{\alpha i}| \, |c_{\beta j}| \int |\chi_\alpha(\mathbf{r})| \, |\chi_\beta(\mathbf{r})| \, d\mathbf{r}
    """
    if mol.cart:
        s, ctr_mat = _cart_mol_abs_ovlp_matrix(mol, nroots)
        return abs(ctr_mat.T) @ s @ abs(ctr_mat)
    else:
        cart_mol = get_cart_mol(mol)
        s, ctr_mat = _cart_mol_abs_ovlp_matrix(cart_mol, nroots)
        cart2spher = cart_mol.cart2sph_coeff(normalized='sp')
        return abs(cart2spher.T @ ctr_mat.T) @ s @ abs(ctr_mat @ cart2spher)


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

    spher_mol = pyscf.M(atom='H 0 0 0; H 0 0 1', basis='def2-svp', cart=False)
    approx_S_abs(spher_mol)

    cart_mol = pyscf.M(atom='H 0 0 0; H 0 0 1', basis='def2-svp', cart=True)
    approx_S_abs(cart_mol)
