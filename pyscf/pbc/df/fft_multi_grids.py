import time
import copy
import numpy
import numpy as np
from pyscf import lib
from pyscf.gto import ATOM_OF, NPRIM_OF, PTR_EXP, PTR_COEFF
from pyscf.pbc import tools
from pyscf.pbc import gto
from pyscf.pbc.dft import numint
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.df.df_jk import _ewald_exxdiv_3d
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf.pbc.gto import eval_gto
from pyscf import __config__


BLKSIZE = numint.BLKSIZE
EXTRA_PREC = getattr(__config__, 'pbc_gto_eval_gto_extra_precision', 1e-2)
TO_EVEN_GRIDS = getattr(__config__, 'pbc_gto_df_fft_multi_grids_to_even', False)


def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None):
    '''Get the Coulomb (J) AO matrix at sampled k-points.

    Args:
        dm_kpts : (nkpts, nao, nao) ndarray or a list of (nkpts,nao,nao) ndarray
            Density matrix at each k-point.  If a list of k-point DMs, eg,
            UHF alpha and beta DM, the alpha and beta DMs are contracted
            separately.
        kpts : (nkpts, 3) ndarray

    Kwargs:
        kpts_band : (3,) ndarray or (*,3) ndarray
            A list of arbitrary "band" k-points at which to evalute the matrix.

    Returns:
        vj : (nkpts, nao, nao) ndarray
        or list of vj if the input dm_kpts is a list of DMs
    '''
    log = lib.logger.Logger(mydf.stdout, mydf.verbose)
    cell = mydf.cell
    low_dim_ft_type = mydf.low_dim_ft_type

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    tasks = multi_grids_tasks(cell, log)
    log.debug('Grids are divided into ntasks %s', len(tasks))

    ni = mydf._numint
    nx, ny, nz = cell.mesh
    rhoG = numpy.zeros((nset,nx,ny,nz), dtype=numpy.complex)
    if abs(dms - dms.transpose(0,1,3,2).conj()).max() < 1e-9:
        def make_rho(ao_l, ao_h, dm_lh, dm_hl):
            c0 = lib.dot(ao_l, dm_lh.astype(ao_l.dtype))
            rho = numpy.einsum('gi,gi->g', c0.real, ao_h.real)
            if c0.dtype == numpy.complex:
                rho+= numpy.einsum('gi,gi->g', c0.imag, ao_h.imag)
            return rho * 2
    else:
        assert(dms.dtype == numpy.double)
        def make_rho(ao_l, ao_h, dm_lh, dm_hl):
            c0 = lib.dot(ao_l, dm_lh)
            rho = numpy.einsum('gi,gi->g', c0, ao_h)
            c0 = lib.dot(ao_l, dm_hl.T)
            rho+= numpy.einsum('gi,gi->g', c0, ao_h)
            return rho

    for cell_low, idx_l, cell_high, idx_h, coords_idx in tasks:
        mesh = cell_high.mesh
        print mesh
        log.debug('mesh %s', mesh)
        dms_hh = numpy.asarray(dms[:,:,idx_h[:,None],idx_h], order='C')
        if cell_low is not None:
            dms_hl = numpy.asarray(dms[:,:,idx_h[:,None],idx_l], order='C')
            dms_lh = numpy.asarray(dms[:,:,idx_l[:,None],idx_h], order='C')

        coords = cell.gen_uniform_grids(mesh)[coords_idx]
        ngrids = numpy.prod(mesh)
        rhoR = np.zeros((nset,ngrids))
        aoR_h = ni.eval_ao(cell_high, coords, kpts)
        for k in range(nkpts):
            for i in range(nset):
                rhoR[i,coords_idx] += numint.eval_rho(cell_high, aoR_h[k], dms_hh[i,k])

        if cell_low is not None:
            #non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE, cell_low.nbas),
            #                     dtype=numpy.uint8)
            #shls_slice = (0, cell_low.nbas)
            #ao_loc = cell_low.ao_loc_nr()
            aoR_l = ni.eval_ao(cell_low, coords, kpts)
            for k in range(nkpts):
                for i in range(nset):
                    #c0 = numint._dot_ao_dm(cell_low, aoR_l[k], dms_lh[i,k],
                    #                       non0tab, shls_slice, ao_loc)
                    #rho = numpy.einsum('gi,gi->g', c0, aoR_h[k].conj())
                    #c0 = numint._dot_ao_dm(cell_low, aoR_l[k], dms_hl[i,k].T.conj(),
                    #                       non0tab, shls_slice, ao_loc)
                    #rho += numpy.einsum('gi,gi->g', c0.conj(), aoR_h[k])
                    rho = make_rho(aoR_l[k], aoR_h[k], dms_lh[i,k], dms_hl[i,k])
                    rhoR[i,coords_idx] += rho.real
        aoR_h = aoR_l = c0 = None

        gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(int)
        gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(int)
        gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(int)
        for i in range(nset):
            rhoR[i] *= 1./nkpts
            rho_freq = tools.fft(rhoR[i], mesh) * cell.vol/ngrids
            rhoG[i,gx[:,None,None],gy[:,None],gz] += rho_freq.reshape(mesh)

    coulG = tools.get_coulG(cell, mesh=cell.mesh, low_dim_ft_type=low_dim_ft_type)
    vG = numpy.einsum('nxyz,xyz->nxyz', rhoG, coulG.reshape(nx,ny,nz))

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    if gamma_point(kpts_band):
        vj_kpts = np.zeros((nset,nband,nao,nao))
    else:
        vj_kpts = np.zeros((nset,nband,nao,nao), dtype=np.complex128)

    for cell_low, idx_l, cell_high, idx_h, coords_idx in tasks:
        mesh = cell_high.mesh
        log.debug('mesh %s', mesh)
        ngrids = numpy.prod(mesh)
        gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(int)
        gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(int)
        gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(int)
        sub_vG = vG[:,gx[:,None,None],gy[:,None],gz].reshape(nset,ngrids)

        vR = tools.ifft(sub_vG, mesh).real.reshape(nset,ngrids)
        vR = vR[:,coords_idx]

        coords = cell.gen_uniform_grids(mesh)[coords_idx]
        aoR_h = ni.eval_ao(cell_high, coords, kpts)
        for k in range(nkpts):
            for i in range(nset):
                vj_sub = lib.dot(aoR_h[k].T.conj()*vR[i], aoR_h[k])
                vj_kpts[i,k,idx_h[:,None],idx_h] += vj_sub

        if cell_low is not None:
            non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE, cell_low.nbas),
                                 dtype=numpy.uint8)
            shls_slice = (0, cell_low.nbas)
            ao_loc = cell_low.ao_loc_nr()
            aoR_l = ni.eval_ao(cell_low, coords, kpts)
            for k in range(nkpts):
                for i in range(nset):
                    vj_sub = lib.dot(aoR_h[k].T.conj()*vR[i], aoR_l[k])
                    vj_kpts[i,k,idx_h[:,None],idx_l] += vj_sub
                    vj_kpts[i,k,idx_l[:,None],idx_h] += vj_sub.conj().T

    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)

@profile
def multi_grids_tasks(cell, verbose=None):
    log = lib.logger.new_logger(cell, verbose)
    tasks = []

    a = cell.lattice_vectors()
    neighbour_images = lib.cartesian_prod(([0, -1, 1],
                                           [0, -1, 1],
                                           [0, -1, 1]))
    # Remove the first one which is the unit cell itself
    neighbour_images = neighbour_images[1:]
    neighbour_images = neighbour_images.dot(a)
    b = numpy.linalg.inv(a.T)
    heights = 1. / numpy.linalg.norm(b, axis=1)
    normal_vector = b * heights.reshape(-1,1)
    distance_to_edge = cell.atom_coords().dot(normal_vector.T)
    # multi-grids do not support atoms out of unit cell
    assert(numpy.all(distance_to_edge >= 0))
    distance_to_edge = numpy.hstack([distance_to_edge, heights-distance_to_edge])
    min_distance_to_edge = distance_to_edge.min(axis=1)

    # Split shells based on rcut
    ke_factor = abs(numpy.linalg.det(b))
    rcuts_pgto = _primitive_gto_rcut(cell)
    ao_loc = cell.ao_loc_nr()
    def make_cell_high_exp(shls_high, r0, r1):
        rcut_atom = [0] * cell.natm
        cell_high = cell.copy()
        for ib in shls_high:
            rc = rcuts_pgto[ib]
            idx = numpy.where((r1 <= rc) & (rc < r0))[0]

            es = cell.bas_exp(ib)
            cs = cell._libcint_ctr_coeff(ib)
            np, nc = cs.shape
            np1 = len(idx)
            if np1 != np:  # no pGTO splitting within the shell
                pexp = cell._bas[ib,PTR_EXP]
                pcoeff = cell._bas[ib,PTR_COEFF]
                cs1 = cs[idx]
                cell_high._env[pcoeff:pcoeff+cs1.size] = cs1.T.ravel()
                cell_high._env[pexp:pexp+np1] = es[idx]
                cell_high._bas[ib,NPRIM_OF] = np1

            ia = cell.bas_atom(ib)
            rcut_atom[ia] = max(rcut_atom[ia], rc[idx].max())
        cell_high._bas = cell_high._bas[shls_high]

        ke_cutoff = 0
        for ib in range(cell_high.nbas):
            l = cell_high.bas_angular(ib)
            es = cell_high.bas_exp(ib)
            cs = abs(cell_high.bas_ctr_coeff(ib)).max(axis=1)
            ke_guess = gto.cell._estimate_ke_cutoff(es, l, cs, cell.precision, ke_factor)
            ke_cutoff = max(ke_cutoff, ke_guess)

        ao_idx = numpy.hstack([numpy.arange(ao_loc[i], ao_loc[i+1])
                               for i in shls_high])
        return cell_high, ao_idx, ke_cutoff, rcut_atom

    def make_cell_low_exp(shls_low, r0, r1):
        cell_low = cell.copy()
        for ib in shls_low:
            idx = numpy.where(r0 <= rcuts_pgto[ib])[0]
            cs = cell._libcint_ctr_coeff(ib)
            np, nc = cs.shape
            if len(idx) == np:  # no pGTO splitting within the shell
                continue
            pexp = cell._bas[ib,PTR_EXP]
            pcoeff = cell._bas[ib,PTR_COEFF]
            cs1 = cs[idx]
            np1 = cs1.shape[0]
            cell_low._env[pcoeff:pcoeff+cs1.size] = cs1.T.ravel()
            cell_low._env[pexp:pexp+np1] = cell.bas_exp(ib)[idx]
            cell_low._bas[ib,NPRIM_OF] = np1
        cell_low._bas = cell_low._bas[shls_low]
        ao_idx = numpy.hstack([numpy.arange(ao_loc[i], ao_loc[i+1])
                               for i in shls_low])
        return cell_low, ao_idx

    rcut_delimeter = a.max() * (.5 ** numpy.arange(10))
    for r0, r1 in zip(numpy.append(1e9, rcut_delimeter),
                      numpy.append(rcut_delimeter, 0)):
        shls_low = []  # low exp
        shls_high = [] # high exp
        for ib, rc in enumerate(rcuts_pgto):
            if numpy.any(r0 <= rc):
                shls_low.append(ib)
            if numpy.any((r1 <= rc) & (rc < r0)):
                shls_high.append(ib)
        if len(shls_high) == 0:
            continue

        if len(shls_low) == 0:
            cell_low = ao_idx_low = None
        else:
            cell_low, ao_idx_low = make_cell_low_exp(shls_low, r0, r1)
        cell_high, ao_idx_high, ke_cutoff, rcut_atom = \
                make_cell_high_exp(shls_high, r0, r1)

        mesh = tools.cutoff_to_mesh(a, ke_cutoff)
        if TO_EVEN_GRIDS:
            mesh = (mesh+1)//2 * 2  # to the nearest even number
        if numpy.all(mesh >= cell.mesh):
            # Including all rest shells
            shls_high = [ib for ib, rc in enumerate(rcuts_pgto)
                         if numpy.any(rc < r0)]
            cell_high, ao_idx_high = make_cell_high_exp(shls_high, r0, 0)[:2]
        cell_high.mesh = mesh = numpy.min([mesh, cell.mesh], axis=0)

        coords_f4 = cell.gen_uniform_grids(mesh).astype(numpy.float32)
        ngrids = coords_f4.shape[0]
        coords_idx = numpy.zeros(ngrids, dtype=bool)
        for ia in set(cell_high._bas[:,ATOM_OF]):
            rcut = rcut_atom[ia]
            log.debug('mesh %s atom %d rcut %g', mesh, ia, rcut)

            atom_coord = cell.atom_coord(ia)
            dr = coords_f4 - atom_coord.astype(numpy.float32)
            coords_idx |= numpy.einsum('px,px->p', dr, dr) <= rcut**2

            if min_distance_to_edge[ia] > rcut:
                # atom + rcut is completely inside the unit cell
                continue

            atoms_in_neighbour = neighbour_images + atom_coord
            distance_to_unit_cell = atoms_in_neighbour.dot(normal_vector.T)
            distance_to_unit_cell = numpy.hstack([abs(distance_to_unit_cell),
                                                  abs(heights-distance_to_unit_cell)])
            idx = distance_to_unit_cell.min(axis=1) <= rcut
            for r_atom in atoms_in_neighbour[idx]:
                dr = coords_f4 - r_atom.astype(numpy.float32)
                coords_idx |= numpy.einsum('px,px->p', dr, dr) <= rcut**2

        tasks.append([cell_low, ao_idx_low, cell_high, ao_idx_high, coords_idx])
        if numpy.all(mesh >= cell.mesh):
            break

    return tasks

def _primitive_gto_rcut(cell):
    '''Cutoff raidus, above which each shell decays to a value less than the
    required precsion'''
    log_prec = numpy.log(cell.precision * EXTRA_PREC)
    rcut = []
    for ib in range(cell.nbas):
        l = cell.bas_angular(ib)
        es = cell.bas_exp(ib)
        cs = abs(cell.bas_ctr_coeff(ib)).max(axis=1)
        r = 5.
        r = (((l+2)*numpy.log(r)+numpy.log(cs) - log_prec) / es)**.5
        r = (((l+2)*numpy.log(r)+numpy.log(cs) - log_prec) / es)**.5
        rcut.append(r)
    return rcut


if __name__ == '__main__':
    from pyscf.pbc import gto, scf
    from pyscf.pbc import df
    from pyscf.pbc.df import fft_jk
    cell = gto.M(
        a = numpy.eye(3)*3.5668,
        atom = '''C     0.      0.      0.    
                  C     0.8917  0.8917  0.8917
                  C     1.7834  1.7834  0.    
                  C     2.6751  2.6751  0.8917
                  C     1.7834  0.      1.7834
                  C     2.6751  0.8917  2.6751
                  C     0.      1.7834  1.7834
                  C     0.8917  2.6751  2.6751''',
        #basis = 'sto3g',
        basis = 'gth-dzvp',
        #basis = 'unc-gth-szv',
        #basis = 'gth-szv',
        #basis = [[0, (1,1)], [0, (0.2, 1)]],
        #verbose = 5,
        #mesh = [15]*3,
        #precision=1e-6
    )
    print cell.mesh

    mydf = df.FFTDF(cell)
    nao = cell.nao_nr()
    numpy.random.seed(1)
    kpts = cell.make_kpts([2,2,2])
    dm = numpy.random.random((len(kpts),nao,nao))
    dm = dm + dm.transpose(0,2,1)
    print(time.clock())
    ref = fft_jk.get_j_kpts(mydf, dm, kpts=kpts)
    print(time.clock())
    v = get_j_kpts(mydf, dm, kpts=kpts)
    print(time.clock())
    print('diff', abs(ref-v).max(), lib.finger(v)-lib.finger(ref))
