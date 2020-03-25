from pyscf.pbc import scf, dft, gto, grad, df
from pyscf.eph.rhf import solve_hmat
import numpy as np
import scipy
from pyscf.lib import logger
import copy

'''Hacky implementation of Electron-Phonon matrix from finite difference'''
# Note, the code can break down when orbital degeneracy is present

AU_TO_CM = 2.19475 * 1e5
CUTOFF_FREQUENCY = 80

def copy_mf(mf, cell):
    if mf.__class__.__name__ == 'KRHF':
        mf1 = scf.KRHF(cell)
    elif mf.__class__.__name__ == 'KUHF':
        mf1 = scf.KUHF(cell)
    elif mf.__class__.__name__ == 'KRKS':
        mf1 = dft.KRKS(cell)
        mf1.xc = mf.xc
    elif mf.__class__.__name__ == 'KUKS':
        mf1 = dft.KUKS(cell)
        mf1.xc = mf.xc
    mf1.kpts = mf.kpts
    mf1.exxdiv = getattr(mf, 'exxdiv', None)
    mf1.conv_tol = mf.conv_tol
    mf1.conv_tol_grad = mf.conv_tol_grad
    return mf1

def run_mfs(mf, cells_a, cells_b):
    '''perform a set of calculations on given two sets of cellcules'''
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
    """From the given equilibrium cellcule, generate 3N cellcules with a shift on + displacement(cell_a) and - displacement(cell_s) on each Cartesian coordinates"""
    coords = cell.atom_coords()
    natoms = len(coords)
    cell_a, cell_s, coords_a, coords_s = [],[],[],[]
    for i in range(natoms):
        for x in range(3):
            new_coords_a, new_coords_s = coords.copy(), coords.copy()
            new_coords_a[i][x] += disp
            new_coords_s[i][x] -= disp
            coords_a.append(new_coords_a)
            coords_s.append(new_coords_s)
    nconfigs = 3*natoms
    for i in range(nconfigs):
        atoma, atoms = [], []
        for j in range(natoms):
            atoma.append([cell.atom_symbol(j), coords_a[i][j]])
            atoms.append([cell.atom_symbol(j), coords_s[i][j]])
        cella = cell.set_geom_(atoma, inplace=False)
        cells = cell.set_geom_(atoms, inplace=False)
        cell_a.append(cella)
        cell_s.append(cells)
    return cell_a, cell_s


def get_v_bra(mf, mf1):
    '''
    computing # <u+|Vxc(0)|v0> + <u0|Vxc(0)|v+>
    '''
    cell, cell1 = mf.cell, mf1.cell # construct a cell that contains both u0 and u+
    atoms = []
    symlst = []
    for symbol, pos in cell1._atom:
        sym = 'ghost'+symbol
        atoms.append((sym, pos))

    # warning: can not use set_geom_
    fused_cell = gto.Cell()
    fused_cell.atom = cell._atom + atoms
    fused_cell.a = cell.a
    fused_cell.mesh = cell.mesh
    fused_cell.unit = cell.unit
    fused_cell.pseudo = cell.pseudo
    fused_cell.precision = cell.precision
    fused_cell.basis = cell.basis
    fused_cell.verbose = 0
    fused_cell.build()

    nao = cell.nao_nr()
    dm0 = mf.make_rdm1()
    RESTRICTED = (dm0.ndim==3)

    nkpts = len(mf.kpts)
    if RESTRICTED:
        dm = np.zeros([nkpts,2*nao,2*nao]) # construct a fake DM to get Vxc matrix
        dm[:,:nao,:nao] = dm0
    else:
        dm = np.zeros([2,nkpts,2*nao,2*nao])
        dm[:,:,:nao,:nao] = dm0

    mf0 = copy_mf(mf, fused_cell)
    mf0.with_df.mesh = mf.with_df.mesh

    veff = mf0.get_veff(fused_cell, dm) #<u*|Vxc(0)|v*> here p* includes both u0 and v+, Vxc is at equilibrium geometry because only the u0 block of the dm is filled
    vnu = mf0.get_hcore(fused_cell) - fused_cell.pbc_intor('int1e_kin', kpts=mf.kpts)

    vtot = veff + vnu
    if RESTRICTED:
        vtot = vtot[:,nao:,:nao]
        vtot += vtot.transpose(0,2,1).conj()
    else:
        vtot = vtot[:,:,nao:,:nao]
        vtot += vtot.transpose(0,1,3,2).conj()
    return vtot


def get_vmat(mf, mfset, disp):
    RESTRICTED = (mf.__class__.__name__[1] == 'R')
    nconfigs = len(mfset)
    vmat=[]
    for i in range(nconfigs):
        mf1, mf2 = mfset[i]
        vfull1 = mf1.get_veff() + mf1.get_hcore() - mf1.cell.pbc_intor('int1e_kin', kpts=mf1.kpts)  # <u+|V+|v+>
        vfull2 = mf2.get_veff() + mf2.get_hcore() - mf2.cell.pbc_intor('int1e_kin', kpts=mf2.kpts)  # <u-|V-|v->
        vfull = (vfull1 - vfull2)/disp  # (<p+|V+|q+>-<p-|V-|q->)/dR
        vbra1 = get_v_bra(mf, mf1)   #<p+|V0|q0> + <p0|V0|q+>
        vbra2 = get_v_bra(mf, mf2)   #<p-|V0|q0> + <p0|V0|q->
        vbra = (vbra1-vbra2)/disp
        vtot = vfull - vbra   #<p0|dV0|q0> = d<p|V|q> - <dp|V0|q> - <p|V0|dq>
        vmat.append(vtot)
    vmat= np.asarray(vmat)
    if vmat.ndim == 4:
        return vmat[:,0]
    elif vmat.ndim==5:
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


def kernel(mf, disp=1e-5, mo_rep=False):
    if hasattr(mf, 'xc'): mf.grids.build(with_non0tab=True)
    if not mf.converged: mf.kernel()
    mo_coeff = np.asarray(mf.mo_coeff)
    RESTRICTED= (mo_coeff.ndim==3)
    cell = mf.cell
    cells_a, cells_b = gen_cells(cell, disp/2.0) # generate a bunch of cellcules with disp/2 on each cartesion coord
    mfset = run_mfs(mf, cells_a, cells_b) # run mean field calculations on all these cellcules
    vmat = get_vmat(mf, mfset, disp) # extracting <u|dV|v>/dR
    hmat = run_hess(mfset, disp)
    omega, vec = solve_hmat(cell, hmat)

    mass = cell.atom_mass_list() * 1836.15
    nmodes, natoms = len(omega), len(mass)
    vec = vec.reshape(natoms, 3, nmodes)
    for i in range(natoms):
        for j in range(nmodes):
            vec[i,:,j] /= np.sqrt(2*mass[i]*omega[j])
    vec = vec.reshape(3*natoms,nmodes)
    if mo_rep:
        if RESTRICTED:
            vmat = np.einsum('xuv,up,vq->xpq', vmat, mo_coeff[0].conj(), mo_coeff[0])
        else:
            vmat = np.einsum('xsuv,sup,svq->xspq', vmat, mo_coeff[:,0].conj(), mo_coeff[:,0])

    if vmat.ndim == 3:
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
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    #cell.precision=1e-9
    cell.verbose = 4
    cell.build()

    kpts = cell.make_kpts([1,1,1])
    mf = dft.KRKS(cell, kpts)
    mf.kernel()

    mat, omega = kernel(mf, disp=1e-4, mo_rep=True)
    print("|Mat|_{max}",abs(mat).max())
