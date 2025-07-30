import numpy as np
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf.pbc.dft.gen_grid import UniformGrids
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.dft.numint import _scale_ao, _contract_rho

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

def _get_weight_strain_derivatives(cell, grids):
    ngrids = grids.size
    weight_0 = cell.vol / ngrids
    weight_1 = np.eye(3) * weight_0
    return weight_0, weight_1

def _eval_ao_strain_derivatives(cell, coords, kpts=None, deriv=0,
                                shls_slice=None, non0tab=None, cutoff=None, out=None):
    '''
    Returns:
        ao_kpts: (nkpts, 3x3xcomp, ngrids, nao) ndarray
            AO values at each k-point
    '''
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    comp_3x3 = comp * 9
    if cell.cart:
        feval = 'GTOval_cart_deriv%d_strain_tensor' % deriv
    else:
        feval = 'GTOval_sph_deriv%d_strain_tensor' % deriv
    out = cell.pbc_eval_gto(feval, coords, comp_3x3, kpts, shls_slice=shls_slice,
                            non0tab=non0tab, cutoff=cutoff, out=out)
    ngrids = len(coords)
    if isinstance(out, np.ndarray):
        out = out.reshape(3,3,comp,ngrids,-1)
    else:
        nkpts = len(out)
        out = [x.reshape(3,3,comp,ngrids,-1) for x in out]
    return out

def get_veff(ks_grad, cell, dm, kpts=None):
    '''Strain derivatives for Coulomb and XC at gamma point
    '''
    mf = ks_grad.base
    cell = mf.cell
    if dm is None: dm = mf.make_rdm1()
    if kpts is None: kpts = mf.kpts
    assert kpts is None or is_zero(kpts)
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension != 1
    assert is_zero(kpts)

    ni = mf._numint
    if ks_grad.grids is not None:
        grids = ks_grad.grids
    else:
        grids = mf.grids
    assert isinstance(grids, UniformGrids)

    xc_code = mf.xc
    xctype = ni._xc_type(xc_code)
    if xctype == 'LDA':
        deriv = 0
        nvar = 1
    elif xctype == 'GGA':
        deriv = 1
        nvar = 4
    elif xctype == 'MGGA':
        deriv = 1
        nvar = 5
    else:
        raise NotImplementedError

    assert dm.ndim == 2
    nao = dm.shape[-1]

    coords = grids.coords
    ngrids = len(coords)
    mesh = grids.mesh
    Gv = cell.get_Gv(mesh)
    coulG_0, coulG_1 = _get_coulG_strain_derivatives(cell, Gv)
    weight_0, weight_1 = _get_weight_strain_derivatives(cell, grids)
    out = np.zeros((3,3))
    rho0 = np.empty((nvar, ngrids))
    rho1 = np.empty((3,3, nvar, ngrids))

    XY, YY, ZY, XZ, YZ, ZZ = 5, 7, 8, 6, 8, 9
    p1 = 0
    for ao, _, mask, weight, coords \
            in ni.block_loop(cell, grids, nao, deriv+1, kpts, None):
        p0, p1 = p1, p1 + weight.size
        ao_strain = _eval_ao_strain_derivatives(cell, coords, kpts, deriv)
        ao_strain = ao_strain[0] # gamma point only
        if xctype == 'LDA':
            ao1 = ao_strain[:,:,0]
            # Adding the response of the grids
            ao1 += np.einsum('xgi,gy->xygi', ao[1:4], coords)
            c0 = ao[0].dot(dm)
            rho0[0,p0:p1] = _contract_rho(ao[0], c0).real
            rho1[:,:,0,p0:p1] = np.einsum('xygi,gi->xyg', ao1, c0.conj()).real
            rho1[:,:,0,p0:p1] *= 2

            exc, vxc = ni.eval_xc_eff(xc_code, rho0[0,p0:p1], 1, xctype=xctype, spin=0)[:2]
            out += np.einsum('xyg,g->xy', rho1[:,:,0,p0:p1], vxc[0]).real * weight_0
            out += np.einsum('g,g->', rho0[0,p0:p1], exc).real * weight_1
        elif xctype == 'GGA':
            ao_strain[:,:,0] += np.einsum('xgi,gy->xygi', ao[1:4], coords)
            ao_strain[:,:,1] += np.einsum('xgi,gy->xygi', ao[4:7], coords)
            ao_strain[0,:,2] += np.einsum('gi,gy->ygi', ao[XY], coords)
            ao_strain[1,:,2] += np.einsum('gi,gy->ygi', ao[YY], coords)
            ao_strain[2,:,2] += np.einsum('gi,gy->ygi', ao[ZY], coords)
            ao_strain[0,:,3] += np.einsum('gi,gy->ygi', ao[XZ], coords)
            ao_strain[1,:,3] += np.einsum('gi,gy->ygi', ao[YZ], coords)
            ao_strain[2,:,3] += np.einsum('gi,gy->ygi', ao[ZZ], coords)
            c0 = lib.einsum('xgi,ij->xgj', ao[:4], dm)
            for i in range(4):
                rho0[i,p0:p1] = _contract_rho(ao[0], c0[i]).real
            rho0[1:4,p0:p1] *= 2 # dm should be hermitian
            rho1[:,:, : ,p0:p1]  = np.einsum('xyngi,gi->xyng', ao_strain, c0[0].conj()).real
            rho1[:,:,1:4,p0:p1] += np.einsum('xygi,ngi->xyng', ao_strain[:,:,0], c0[1:4].conj()).real
            rho1[:,:,:,p0:p1] *= 2

            exc, vxc = ni.eval_xc_eff(xc_code, rho0[:,p0:p1], 1, xctype=xctype, spin=0)[:2]
            out += np.einsum('xyng,ng->xy', rho1[:,:,:,p0:p1], vxc).real * weight_0
            out += np.einsum('g,g->', rho0[0,p0:p1], exc).real * weight_1
        else: # MGGA
            ao_strain[:,:,0] += np.einsum('xgi,gy->xygi', ao[1:4], coords)
            ao_strain[:,:,1] += np.einsum('xgi,gy->xygi', ao[4:7], coords)
            ao_strain[0,:,2] += np.einsum('gi,gy->ygi', ao[XY], coords)
            ao_strain[1,:,2] += np.einsum('gi,gy->ygi', ao[YY], coords)
            ao_strain[2,:,2] += np.einsum('gi,gy->ygi', ao[ZY], coords)
            ao_strain[0,:,3] += np.einsum('gi,gy->ygi', ao[XZ], coords)
            ao_strain[1,:,3] += np.einsum('gi,gy->ygi', ao[YZ], coords)
            ao_strain[2,:,3] += np.einsum('gi,gy->ygi', ao[ZZ], coords)
            c0 = lib.einsum('xgi,ij->xgj', ao[:4], dm)
            for i in range(4):
                rho0[i,p0:p1] = _contract_rho(ao[0], c0[i]).real
            rho0[4,p0:p1]  = _contract_rho(ao[1], c0[1]).real
            rho0[4,p0:p1] += _contract_rho(ao[2], c0[2]).real
            rho0[4,p0:p1] += _contract_rho(ao[3], c0[3]).real
            rho0[4,p0:p1] *= .5
            rho0[1:4,p0:p1] *= 2 # dm should be hermitian
            rho1[:,:, :4,p0:p1]  = np.einsum('xyngi,gi->xyng', ao_strain, c0[0].conj()).real
            rho1[:,:,1:4,p0:p1] += np.einsum('xygi,ngi->xyng', ao_strain[:,:,0], c0[1:4].conj()).real
            rho1[:,:,4,p0:p1] = np.einsum('xyngi,ngi->xyg', ao_strain[:,:,1:4], c0[1:4].conj()).real
            rho1[:,:,4,p0:p1] *= .5
            rho1[:,:,:,p0:p1] *= 2

            exc, vxc = ni.eval_xc_eff(xc_code, rho0[:,p0:p1], 1, xctype=xctype, spin=0)[:2]
            out += np.einsum('xyng,ng->xy', rho1[:,:,:,p0:p1], vxc).real * weight_0
            out += np.einsum('g,g->', rho0[0,p0:p1], exc).real * weight_1

    rhoG = pbctools.fft(rho0[0], mesh)
    vG = rhoG * coulG_0
    vR = pbctools.ifft(vG, mesh)
    EJ = np.einsum('xyg,g->xy', rho1[:,:,0], vR).real * weight_0 * 2
    EJ += np.einsum('g,g->', rho0[0], vR).real * weight_1
    EJ += np.einsum('g,xyg,g->xy', rhoG, coulG_1, rhoG.conj()).real * (weight_0/ngrids)
    out += .5 * EJ
    return out

def get_pp(cell, kpts=None):
    disp = max(1e-5, cell.precision**.5)
    out = []
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            v1 = np.array(get_pp(cell1, kpts))
            v2 = np.array(get_pp(cell2, kpts))
            out.append((v1 - v2) / (2*disp))
    return out

def ewald(cell, ew_eta=None, ew_cut=None):
    disp = max(1e-5, (cell.precision*.1)**.5)
    out = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp)
            e1 = cell1.ewald()
            e2 = cell2.ewald()
            out[i,j] = (e1 - e2) / (2*disp)
    return out
