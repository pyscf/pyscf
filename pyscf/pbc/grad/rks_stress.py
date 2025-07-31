import numpy as np
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.pbc.tools import pbc as pbctools
from pyscf.pbc.lib.kpts_helper import is_zero
from pyscf.pbc.dft.gen_grid import UniformGrids
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import FFTDF, ft_ao
from pyscf.dft.numint import _scale_ao, _contract_rho

r'''
The energy derivatives for the strain tensor e_ij is

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
    coulG_0 = 4 * np.pi / G2
    coulG_1 = np.einsum('g,gx,gy->xyg', 2/G2, Gv, Gv)
    coulG_1 *= coulG_0
    return coulG_0, coulG_1

def _get_weight_strain_derivatives(cell, grids):
    ngrids = grids.size
    weight_0 = cell.vol / ngrids
    weight_1 = np.eye(3) * weight_0
    return weight_0, weight_1

def _eval_ao_strain_derivatives(cell, coords, kpts=None, deriv=0, out=None):
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
    out = cell.pbc_eval_gto(feval, coords, comp_3x3, kpts, out=out)
    ngrids = len(coords)
    if isinstance(out, np.ndarray):
        out = out.reshape(3,3,comp,ngrids,-1)
    else:
        out = [x.reshape(3,3,comp,ngrids,-1) for x in out]
    return out

def get_vxc(ks_grad, cell, dm, kpt=None, with_j=False, with_nuc=False):
    '''Strain derivatives for Coulomb and XC at gamma point

    Kwargs:
        with_j : Whether to include the electron-electron Coulomb interactions
        with_nuc : Whether to include the electron-nuclear Coulomb interactions
    '''
    from pyscf.pbc.dft.numint import NumInt
    mf = ks_grad.base
    if dm is None: dm = mf.make_rdm1()
    if kpt is None: kpt = mf.kpt
    assert kpt is None or is_zero(kpt)
    assert cell.low_dim_ft_type != 'inf_vacuum'
    assert cell.dimension != 1

    ni = mf._numint
    assert isinstance(ni, NumInt)
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
    weight_0, weight_1 = _get_weight_strain_derivatives(cell, grids)
    out = np.zeros((3,3))
    rho0 = np.empty((nvar, ngrids))
    rho1 = np.empty((3,3, nvar, ngrids))

    XY, YY, ZY, XZ, YZ, ZZ = 5, 7, 8, 6, 8, 9
    p1 = 0
    for ao, _, mask, weight, coords \
            in ni.block_loop(cell, grids, nao, deriv+1):
        p0, p1 = p1, p1 + weight.size
        ao_strain = _eval_ao_strain_derivatives(cell, coords, deriv=deriv)
        ao_strain = ao_strain
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

    Gv = cell.get_Gv(mesh)
    coulG_0, coulG_1 = _get_coulG_strain_derivatives(cell, Gv)
    rhoG = pbctools.fft(rho0[0], mesh)
    if with_j:
        vR = pbctools.ifft(rhoG * coulG_0, mesh)
        EJ = np.einsum('xyg,g->xy', rho1[:,:,0], vR).real * weight_0 * 2
        EJ += np.einsum('g,g->', rho0[0], vR).real * weight_1
        EJ += np.einsum('g,xyg,g->xy', rhoG.conj(), coulG_1, rhoG).real * (weight_0/ngrids)
        out += .5 * EJ

    if with_nuc:
        if cell._pseudo:
            vpplocG_0, vpplocG_1 = _get_vpplocG_strain_derivatives(cell, mesh)
            vpplocR = pbctools.ifft(vpplocG_0, mesh).real
            Ene = np.einsum('xyg,g->xy', rho1[:,:,0], vpplocR).real
            Ene += np.einsum('g,xyg->xy', rhoG.conj(), vpplocG_1).real * (1./ngrids)
            Ene += _get_pp_nonloc_strain_derivatives(cell, mesh, dm)
        else:
            charge = -cell.atom_charges()
            # SI corresponds to Fourier components of the fractional atomic
            # positions within the cell. It does not respond to the strain
            # transformation
            SI = cell.get_SI(mesh=mesh)
            ZG = np.dot(charge, SI)
            vR = pbctools.ifft(ZG * coulG_0, mesh)
            Ene = np.einsum('xyg,g->xy', rho1[:,:,0], vR).real
            Ene += np.einsum('g,xyg,g->xy', rhoG.conj(), coulG_1, ZG).real * (1./ngrids)
        out += Ene
    return out

def _get_vpplocG_strain_derivatives(cell, mesh):
    disp = 1e-5
    ngrids = np.prod(mesh)
    v1 = np.empty((3,3, ngrids), dtype=np.complex128)
    SI = cell.get_SI(mesh=mesh)
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            vpplocG1 = pseudo.get_vlocG(cell1, cell1.get_Gv(mesh))
            vpplocG2 = pseudo.get_vlocG(cell2, cell2.get_Gv(mesh))
            vpplocG1 = -np.einsum('ij,ij->j', SI, vpplocG1)
            vpplocG2 = -np.einsum('ij,ij->j', SI, vpplocG2)
            v1[x,y] = (vpplocG1 - vpplocG2) / (2*disp)
    vpplocG = pseudo.get_vlocG(cell, cell.get_Gv(mesh))
    v0 = -np.einsum('ij,ij->j', SI, vpplocG)
    return v0, v1

def _get_pp_nonloc_strain_derivatives(cell, mesh, dm, kpts=None):
    assert dm.ndim == 2
    dm = dm[None,:,:]
    fakemol = gto.Mole()
    fakemol._atm = np.zeros((1,gto.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1,gto.BAS_SLOTS), dtype=np.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = np.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    ngrids = np.prod(mesh)
    buf = np.empty((48,ngrids), dtype=np.complex128)
    if kpts is None:
        kpts = np.zeros((1, 3))
    else:
        kpts = kpts.reshape((-1, 3))
    scaled_kpts = kpts.dot(cell.lattice_vectors().T)
    nkpts = len(kpts)

    def eval_pp_nonloc(cell):
        vol = cell.vol
        b = cell.reciprocal_vectors(norm_to=1)
        Gv = cell.get_Gv(mesh)
        SI = cell.get_SI(mesh=mesh)
        # buf for SPG_lmi upto l=0..3 and nl=3
        vppnl = 0
        for k, kpt in enumerate(scaled_kpts):
            kpt = kpt.dot(b)
            Gk = Gv + kpt
            G_rad = lib.norm(Gk, axis=1)
            aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt) * (1/vol)**.5
            for ia in range(cell.natm):
                symb = cell.atom_symbol(ia)
                if symb not in cell._pseudo:
                    continue
                pp = cell._pseudo[symb]
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        fakemol._bas[0,gto.ANG_OF] = l
                        fakemol._env[ptr+3] = .5*rl**2
                        fakemol._env[ptr+4] = rl**(l+1.5)*np.pi**1.25
                        pYlm_part = fakemol.eval_gto('GTOval', Gk)

                        p0, p1 = p1, p1+nl*(l*2+1)
                        # pYlm is real, SI[ia] is complex
                        pYlm = np.ndarray((nl,l*2+1,ngrids), dtype=np.complex128, buffer=buf[p0:p1])
                        for k in range(nl):
                            qkl = pseudo.pp._qli(G_rad*rl, l, k)
                            pYlm[k] = pYlm_part.T * qkl
                if p1 > 0:
                    SPG_lmi = buf[:p1]
                    SPG_lmi *= SI[ia].conj()
                    SPG_lm_aoGs = SPG_lmi.dot(aokG)
                    rho = SPG_lm_aoGs.dot(dm[k]).dot(SPG_lm_aoGs.conj().T).real
                    p1 = 0
                    for l, proj in enumerate(pp[5:]):
                        rl, nl, hl = proj
                        if nl > 0:
                            p0, p1 = p1, p1+nl*(l*2+1)
                            hl = np.asarray(hl)
                            vppnl += np.einsum('ij,ji->', hl, rho[p0:p1,p0:p1])
        return vppnl / (nkpts*vol)

    disp = max(1e-5, (cell.precision*.1)**.5)
    out = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp)
            e1 = eval_pp_nonloc(cell1)
            e2 = eval_pp_nonloc(cell2)
            out[i,j] = (e1 - e2) / (2*disp)
    return out

def ewald(cell):
    disp = max(1e-5, (cell.precision*.1)**.5)
    out = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            cell1, cell2 = _finite_diff_cells(cell, i, j, disp)
            e1 = cell1.ewald()
            e2 = cell2.ewald()
            out[i,j] = (e1 - e2) / (2*disp)
    return out

def kernel(mf_grad):
    '''Compute the energy derivatives for strain tensor (e_ij)

                1  d E
    sigma_ij = --- ------
                V  d e_ij

    sigma is a asymmetric 3x3 matrix. The symmetric stress tensor in the 6 Voigt
    notation can be transformed from the asymmetric stress tensor

    sigma1 = sigma_11
    sigma2 = sigma_22
    sigma3 = sigma_33
    sigma6 = (sigma_12 + sigma_21)/2
    sigma5 = (sigma_13 + sigma_31)/2
    sigma4 = (sigma_23 + sigma_32)/2

    See K. Doll, Mol Phys (2010), 108, 223
    '''
    mf = mf_grad.base
    kpt = mf.kpt
    assert is_zero(kpt)
    with_df = mf.with_df
    assert isinstance(with_df, FFTDF)
    if hasattr(mf, 'U_idx'):
        raise NotImplementedError('Stress tensor for DFT+U')

    log = logger.new_logger(mf_grad)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Computing stress tensor')

    cell = mf.cell
    dm0 = mf.make_rdm1()
    dme0 = mf_grad.make_rdm1e()
    sigma = ewald(cell)

    disp = 1e-5
    for x in range(3):
        for y in range(3):
            cell1, cell2 = _finite_diff_cells(cell, x, y, disp)
            t1 = cell1.pbc_intor('int1e_kin')
            t2 = cell2.pbc_intor('int1e_kin')
            t1 = np.einsum('ij,ji->', t1, dm0)
            t2 = np.einsum('ij,ji->', t2, dm0)
            sigma[x,y] += (t1 - t2) / (2*disp)
            s1 = cell1.pbc_intor('int1e_ovlp')
            s2 = cell2.pbc_intor('int1e_ovlp')
            s1 = np.einsum('ij,ji->', s1, dme0)
            s2 = np.einsum('ij,ji->', s2, dme0)
            sigma[x,y] -= (s1 - s2) / (2*disp)

    t0 = log.timer_debug1('hcore derivatives', *t0)

    sigma += get_vxc(mf_grad, cell, dm0, with_j=True, with_nuc=True)
    t0 = log.timer_debug1('Vxc and Coulomb derivatives', *t0)

    sigma /= cell.vol
    if log.verbose >= logger.DEBUG:
        log.debug('Asymmetric strain tensor')
        log.debug('%s', sigma)
    return sigma

# TODO: DFT+U
