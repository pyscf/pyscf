import numpy as np
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.lib.kpts_helper import is_zero

'''
Stress tensor is defined as the energy derivatives for the strain tensor e_ij

                1  d E
    sigma_ij = --- ------
                V  d e_ij

The strain tesnor e_ij describes the transformation for real space coordinates
in the crystal

    \sum_j [\deta_ij + e_ij] R_j  [for j = x, y, z]

The strain tensor is generally not a symmetric tensor. Symmetrization

    [e1   e6/2 e5/2]
    [e6/2 e2   e4/2]
    [e5/2 e4/2 e3  ]

is applied to form 6 independent component.

    e1 = e_11
    e2 = e_22
    e3 = e_33
    e6 = e_12 + e_21
    e5 = e_13 + e_31
    e4 = e_32 + e_23

The 6 component strain is then used to define the symmetric stress tensor.

               1  d E
    sigma_i = --- ------  for i = 1 .. 6
               V  d e_i

The symmetric stress tensor represented in the 6 Voigt notation can be
transformed from the asymmetric stress tensor sigma_ij

    sigma1 = sigma_11
    sigma2 = sigma_22
    sigma3 = sigma_33
    sigma6 = (sigma_12 + sigma_21)/2
    sigma5 = (sigma_13 + sigma_31)/2
    sigma4 = (sigma_23 + sigma_32)/2

See K. Doll, Mol Phys (2010), 108, 223
'''

def strain_tensor_dispalcement(x, y, disp):
    E_strain = np.eye(3)
    E_strain[x,y] += disp
    return E_strain

def _finite_diff_cells(cell, x, y, disp=1e-4, precision=None):
    if precision is not None:
        cell = cell.copy()
        cell.precision = precision
    a = cell.lattice_vectors()
    r = cell.atom_coords()
    e_strain = strain_tensor_dispalcement(x, y, disp)
    cell1 = cell.set_geom_(r.dot(e_strain.T), a=a.dot(e_strain.T), unit='AU', inplace=False)

    e_strain = strain_tensor_dispalcement(x, y, -disp)
    cell2 = cell.set_geom_(r.dot(e_strain.T), a=a.dot(e_strain.T), unit='AU', inplace=False)
    return cell1, cell2

def get_ovlp(cell, kpts=None):
    disp = 1e-5
    s = []
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            s1 = np.asarray(cell1.pbc_intor('int1e_ovlp', kpts=kpts))
            s2 = np.asarray(cell2.pbc_intor('int1e_ovlp', kpts=kpts))
            s.append((s1 - s2) / (2*disp))
    return s

def get_kin(cell, kpts=None):
    disp = 1e-5
    t = []
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            t1 = np.asarray(cell1.pbc_intor('int1e_kin', kpts=kpts))
            t2 = np.asarray(cell2.pbc_intor('int1e_kin', kpts=kpts))
            t.append((t1 - t2) / (2*disp))
    return t

def _get_coulG_strain_derivatives(cell, Gv):
    '''derivatives of 4pi/G^2'''
    G2 = np.einsum('gx,gx->g', Gv, Gv)
    G2[0] = np.inf
    G4 = G2 * G2
    coulG_0 = 4 * np.pi / G2
    coulG_1 = np.einsum('g,gx,gy->xyg', 2/G2, Gv, Gv)
    coulG_1 *= coulG_0
    return coulG_0, coulG_1

def _get_weight_strain_derivatives(cell, ngrids):
    weight_0 = cell.vol / ngrids
    weight_1 = np.eye(3) * weight_0
    return weight_0, weight_1

def _eval_ao_strain_derivatives(cell, coords):
    pass

def get_veff():
    '''Strain derivatives for Coulomb and XC at gamma point
    '''
    mesh = mydf.mesh
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension != 1
    assert is_zero(kpts)
    #assert Uniform grids

    ngrids = np.prod(mesh)
    Gv = cell.get_Gv(mesh)
    coulG_0, coulG_1 = _get_weight_strain_derivatives(cell, Gv)
    weight_0, weight_1 = _get_weight_strain_derivatives(cell, ngrids)

    block_size = 2000
    for p0, p1 in lib.prange(0, ngrids, block_size):
        ao = None
        ao_strain = None
        ao_ks, mask = ao_ks_etc[0], ao_ks_etc[2]
        rhoR[i,p0:p1] += make_rho(i, ao_ks, mask, 'LDA').real
        rho0 = None
        rho1 = None

        rhoG = pbctools.fft(rhoR[i], mesh)
        vG = coulG * rhoG
        vR = pbctools.ifft(vG, mesh).real

        vxc1 += vR1 * weight_0 + vR0 * weight_1

        aow = np.einsum('xi,x->xi', ao[0], vR[i,p0:p1])
        vj_kpts[:,i,k] -= lib.einsum('axi,xj->aij', ao[1:].conj(), aow)
        ao = ao_strain = None
    return de

def get_pp():
    # increase cell.precision call finite difference
    pass

def ewald():
    # increase cell.precision call finite difference
    pass
