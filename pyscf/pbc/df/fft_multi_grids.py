import time
import copy
import numpy
import numpy as np
from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc import gto
from pyscf.pbc.dft import numint
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.df.df_jk import _ewald_exxdiv_3d
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf.pbc.gto import eval_gto
from pyscf import __config__


EXP_DELIMITER = getattr(__config__, 'pbc_gto_df_fft_multi_grids_exp_delimiter',
                        [512., 128., 32., 8., 2.0, 0.5, 0])
#EXP_DELIMITER = [400., 40., 4.0, 0.4, 0]
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

    cell, ctr_coeff = gto.cell._split_basis(cell, EXP_DELIMITER)
    nao = cell.nao_nr()
    log.debug('splitted cell ctr_coeff shape %s', ctr_coeff.shape)
    dms = lib.einsum('nkpq,ip,jq->nkij', dms, ctr_coeff, ctr_coeff)

    tasks = multi_grids_tasks(cell, log)
    log.debug('Grids are divided into ntasks %s', len(tasks))

    ao_loc = cell.ao_loc_nr()
    ni = mydf._numint
    pcell = copy.copy(cell)
    nx, ny, nz = cell.mesh
    rhoG = numpy.zeros((nset,nx,ny,nz), dtype=numpy.complex)
    ao_idx_last = numpy.zeros(0, dtype=int)
    shls_idx_last = numpy.zeros(0, dtype=int)
    for shls_idx, mesh, coords_idx in tasks:
        log.debug('%s %s', mesh, shls_idx)

        ao_idx = numpy.hstack([numpy.arange(ao_loc[i], ao_loc[i+1])
                               for i in shls_idx])
        nao_last = ao_idx_last.size
        ao_idx = numpy.append(ao_idx_last, ao_idx)
        sub_dms = numpy.asarray(dms[:,:,ao_idx[:,None],ao_idx], order='C')
        # Exclude the contributions of current task from the rest tasks
        dms[:,:,ao_idx[:,None],ao_idx] = 0

        shls_idx = numpy.append(shls_idx_last, shls_idx)
        pcell._bas = cell._bas[shls_idx]

        ao_idx_last = ao_idx
        shls_idx_last = shls_idx

        coords = cell.gen_uniform_grids(mesh)
        ngrids = coords.shape[0]
        rhoR = np.zeros((nset,ngrids))
        for k, aoR in enumerate(ni.eval_ao(pcell, coords[coords_idx], kpts)):
            for i in range(nset):
                rhoR[i,coords_idx] += numint.eval_rho(pcell, aoR, sub_dms[i,k])
        aoR = None

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

    ao_to_include = numpy.ones(cell.nao_nr(), dtype=bool)
    shls_to_include = numpy.ones(cell.nbas, dtype=bool)
    # from steep/local functions (grids within small region of real space) to
    # smooth/diffused functions (grids over the entire unit cell).  Use the vj
    # sub-matrix of the smooth functions (which is accurate) to overwrite the
    # existing sub-matrix (which is inaccurate when computing with the grids in
    # small region of real space)
    for shls_idx, mesh, coords_idx in reversed(tasks):
        log.debug('%s %s', mesh, shls_idx)
        ao_idx = numpy.where(ao_to_include)[0]
        pcell._bas = cell._bas[shls_to_include]

        coords = cell.gen_uniform_grids(mesh)
        ngrids = coords.shape[0]
        gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(int)
        gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(int)
        gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(int)
        sub_vG = vG[:,gx[:,None,None],gy[:,None],gz].reshape(nset,ngrids)

        vR = tools.ifft(sub_vG, mesh).real.reshape(nset,ngrids)
        vR = vR[:,coords_idx]
        for k, aoR in enumerate(ni.eval_ao(pcell, coords[coords_idx], kpts_band)):
            for i in range(nset):
                vj_kpts[i,k,ao_idx[:,None],ao_idx] = lib.dot(aoR.T.conj()*vR[i], aoR)
        aoR = None

        ao_idx = numpy.hstack([numpy.arange(ao_loc[i], ao_loc[i+1])
                               for i in shls_idx])
        ao_to_include[ao_idx] = False
        shls_to_include[shls_idx] = False

    vj_kpts = lib.einsum('nkpq,pi,qj->nkij', vj_kpts, ctr_coeff, ctr_coeff)

    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)

def estimate_ke_cutoff(cell, precision):
    '''Energy cutoff estimation'''
    b = cell.reciprocal_vectors()
    if cell.dimension == 0:
        w = 1
    elif cell.dimension == 1:
        w = np.linalg.norm(b[0]) / (2*np.pi)
    elif cell.dimension == 2:
        w = np.linalg.norm(np.cross(b[0], b[1])) / (2*np.pi)**2
    else:
        w = abs(np.linalg.det(b)) / (2*np.pi)**3

    Ecut = []
    for i in range(cell.nbas):
        l = cell.bas_angular(i)
        es = cell.bas_exp(i)
        cs = abs(cell.bas_ctr_coeff(i)).max(axis=1)
        Ecut.append(gto.cell._estimate_ke_cutoff(es, l, cs, precision, w))
    if len(Ecut) == 0:
        Ecut = [0]
    return numpy.array(Ecut)

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

    ATOM_OF = 0
    rcuts = eval_gto._estimate_rcut(cell)
    ke_cutoffs = estimate_ke_cutoff(cell, cell.precision)
    rcut_delimeter = a.max() * (.5 ** numpy.arange(10))
    for r0, r1 in zip(numpy.append(1e9, rcut_delimeter),
                      numpy.append(rcut_delimeter, 0)):
        shls_idx = numpy.where((r1 <= rcuts) & (rcuts < r0))[0]
        if shls_idx.size == 0:
            continue

        ke_cutoff = ke_cutoffs[shls_idx].max()
        mesh = tools.cutoff_to_mesh(a, ke_cutoffs[shls_idx].max())
        #mesh = (mesh+1)//2 * 2  # to the nearest even number
        if numpy.all(mesh >= cell.mesh):
            shls_idx = numpy.where(rcuts < r0)[0]
            mesh = cell.mesh
        else:
            mesh = numpy.min([mesh, cell.mesh], axis=0)

        sub_bas = cell._bas[shls_idx]
        coords_f4 = cell.gen_uniform_grids(mesh).astype(numpy.float32)
        coords_idx = numpy.zeros(coords_f4.shape[0], dtype=bool)
        for ia in set(sub_bas[:,ATOM_OF]):
            shls_for_atm = shls_idx[sub_bas[:,ATOM_OF] == ia]
            rcut = rcuts[shls_for_atm].max()
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

        tasks.append([shls_idx, mesh, coords_idx])

        if numpy.all(mesh >= cell.mesh):
            break

    return tasks


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
        #basis = 'gth-dzvp',
        #basis = 'unc-gth-szv',
        basis = 'gth-dzv',
        #verbose = 5,
        #mesh = [15]*3,
    )

    mydf = df.FFTDF(cell)
    nao = cell.nao_nr()
    numpy.random.seed(1)
    kpts = cell.make_kpts([1,2,2])
    dm = numpy.random.random((len(kpts),nao,nao))
    dm = dm + dm.transpose(0,2,1)
    ref = 0
    print(time.clock())
    ref = fft_jk.get_j_kpts(mydf, dm, kpts=kpts)
    print(time.clock())
    v = get_j_kpts(mydf, dm, kpts=kpts)
    print(time.clock())
    print('diff', abs(ref-v).max(), lib.finger(v)-lib.finger(ref))
