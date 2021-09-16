import ctypes
import numpy as np
from pyscf import __config__
from pyscf import lib
from pyscf.gto import moleintor
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts_helper import gamma_point
from pyscf.pbc.df.df_jk import _format_dms
from pyscf.pbc.dft import multigrid
from pyscf.pbc.dft.multigrid import (EXTRA_PREC, PTR_EXPDROP, EXPDROP, RHOG_HIGH_ORDER, IMAG_TOL,
                                     _take_4d, _takebak_4d)
from pyscf.pbc.gto.cell import build_neighbor_list_for_shlpairs

NGRIDS = getattr(__config__, 'pbc_dft_multigrid_ngrids', 4)
KE_RATIO = getattr(__config__, 'pbc_dft_multigrid_ke_ratio', 3.0)
REL_CUTOFF = getattr(__config__, 'pbc_dft_multigrid_rel_cutoff', 15.0)

libdft = lib.load_library('libdft')


class GridLevel_Info(ctypes.Structure):
    '''
    Info about grid levels
    '''
    _fields_ = [("nlevels", ctypes.c_int), # number of grid levels
                ("rel_cutoff", ctypes.c_double),
                ("cutoff", ctypes.POINTER(ctypes.c_double)),
                ("mesh", ctypes.POINTER(ctypes.c_int))]

class RS_Grid(ctypes.Structure):
    '''
    Values on real space multigrid
    '''
    _fields_ = [("nlevels", ctypes.c_int),
                ("gridlevel_info", ctypes.POINTER(GridLevel_Info)),
                ("comp", ctypes.c_int),
                # data is list of 1d arrays
                ("data", ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))]

class PGFPair(ctypes.Structure):
    '''
    Primitive Gaussian function pair
    '''
    _fields_ = [("ish", ctypes.c_int),
                ("ipgf", ctypes.c_int),
                ("jsh", ctypes.c_int),
                ("jpgf", ctypes.c_int),
                ("iL", ctypes.c_int),
                ("radius", ctypes.c_double)]


class Task(ctypes.Structure):
    _fields_ = [("buf_size", ctypes.c_size_t),
                ("ntasks", ctypes.c_size_t),
                ("pgfpairs", ctypes.POINTER(ctypes.POINTER(PGFPair))),
                ("radius", ctypes.c_double)]


class TaskList(ctypes.Structure):
    _fields_ = [("nlevels", ctypes.c_int),
                ("gridlevel_info", ctypes.POINTER(GridLevel_Info)),
                ("tasks", ctypes.POINTER(ctypes.POINTER(Task)))]


def multi_grids_tasks(cell, ke_cutoff=None, hermi=0, rel_cutoff=REL_CUTOFF, ngrids=NGRIDS):
    if ke_cutoff is None:
        ke_cutoff = cell.ke_cutoff
    if ke_cutoff is None:
        raise ValueError("cell.ke_cutoff is not set.")
    ke1 = ke_cutoff
    cutoff = [ke1,]
    for i in range(ngrids-1):
        ke1 /= KE_RATIO
        cutoff.append(ke1)
    cutoff.reverse()
    a = cell.lattice_vectors()
    mesh = []
    for ke in cutoff:
        mesh.append(tools.cutoff_to_mesh(a, ke))
    gridlevel_info = init_gridlevel_info(cutoff, rel_cutoff, mesh)
    task_list = build_task_list(cell, gridlevel_info, hermi=hermi)
    return task_list


def init_gridlevel_info(cutoff, rel_cutoff, mesh):
    if cutoff[0] < 1e-15:
        cutoff = cutoff[1:]
    cutoff = np.asarray(cutoff, order='C', dtype=np.double)
    mesh = np.asarray(np.asarray(mesh).reshape(-1,3), order='C', dtype=np.int32)
    nlevels = len(cutoff)
    gridlevel_info = ctypes.POINTER(GridLevel_Info)()
    fn = getattr(libdft, "init_gridlevel_info", None)
    try:
        fn(ctypes.byref(gridlevel_info),
           cutoff.ctypes.data_as(ctypes.c_void_p),
           mesh.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int(nlevels), ctypes.c_double(rel_cutoff))
    except:
        raise RuntimeError("Failed to init grid level info")
    return gridlevel_info


def init_rs_grid(gridlevel_info, comp):
    '''
    Initialize values on real space multigrid
    '''
    rs_grid = ctypes.POINTER(RS_Grid)()
    fn = getattr(libdft, "init_rs_grid", None)
    try:
        fn(ctypes.byref(rs_grid), 
           ctypes.byref(gridlevel_info),
           ctypes.c_int(comp))
    except:
        raise RuntimeError("Failed to initialize real space multigrid data.")
    return rs_grid


def build_task_list(cell0, gridlevel_info, cell1=None, Ls=None, hermi=0, precision=None):
    if cell1 is None:
        cell1 = cell0
    if Ls is None:
        Ls = cell0.get_lattice_Ls()
    if precision is None:
        precision = cell0.precision

    if hermi == 1:
        assert cell1 is cell0

    ish_atm = np.asarray(cell0._atm, order='C', dtype=np.int32)
    ish_bas = np.asarray(cell0._bas, order='C', dtype=np.int32)
    ish_env = np.asarray(cell0._env, order='C', dtype=np.double)
    nish = len(ish_bas)

    ish_rcut, ipgf_rcut = cell0.rcut_by_shells(return_pgf_radius=True)
    ptr_ipgf_rcut = lib.ndarray_pointer_2d(ipgf_rcut)

    if cell1 is cell0:
        jsh_atm = ish_atm
        jsh_bas = ish_bas
        jsh_env = ish_env
        jsh_rcut = ish_rcut
        jpgf_rcut = ipgf_rcut
        ptr_jpgf_rcut = ptr_ipgf_rcut
    else:
        jsh_atm = np.asarray(cell1._atm, order='C', dtype=np.int32)
        jsh_bas = np.asarray(cell1._bas, order='C', dtype=np.int32)
        jsh_env = np.asarray(cell1._env, order='C', dtype=np.double)

        jsh_rcut, jpgf_rcut = cell1.rcut_by_shells(return_pgf_radius=True)
        ptr_jpgf_rcut = lib.ndarray_pointer_2d(jpgf_rcut)
    njsh = len(jsh_bas)

    nl = build_neighbor_list_for_shlpairs(cell0, cell1,
                                          ish_rcut=ish_rcut, jsh_rcut=jsh_rcut, 
                                          hermi=hermi)

    task_list = ctypes.POINTER(TaskList)()
    func = getattr(libdft, "build_task_list", None)
    try:
        func(ctypes.byref(task_list),
             ctypes.byref(nl), ctypes.byref(gridlevel_info),
             ish_atm.ctypes.data_as(ctypes.c_void_p),
             ish_bas.ctypes.data_as(ctypes.c_void_p),
             ish_env.ctypes.data_as(ctypes.c_void_p),
             ish_rcut.ctypes.data_as(ctypes.c_void_p),
             ptr_ipgf_rcut,
             jsh_atm.ctypes.data_as(ctypes.c_void_p),
             jsh_bas.ctypes.data_as(ctypes.c_void_p),
             jsh_env.ctypes.data_as(ctypes.c_void_p),
             jsh_rcut.ctypes.data_as(ctypes.c_void_p),
             ptr_jpgf_rcut,
             ctypes.c_int(nish), ctypes.c_int(njsh),
             Ls.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_double(precision), ctypes.c_int(hermi))
    except:
        raise RuntimeError("Failed to get task list.")
    return task_list


def eval_rho(cell, dm, task_list, shls_slice=None, hermi=0, xctype='LDA', kpts=None,
             dimension=None, cell1=None, shls_slice1=None, Ls=None, 
             a=None, ignore_imag=False):
    '''
    Collocate density (opt. gradients) on the real-space grid.
    The two sets of Gaussian functions can be different.

    Returns:
        rho: RS_Grid object
            Densities on real space multigrids.
    '''
    cell0 = cell
    shls_slice0 = shls_slice
    if cell1 is None:
        cell1 = cell0

    #TODO mixture of cartesian and spherical bases
    assert cell0.cart == cell1.cart

    ish_atm = np.asarray(cell0._atm, order='C', dtype=np.int32)
    ish_bas = np.asarray(cell0._bas, order='C', dtype=np.int32)
    ish_env = np.asarray(cell0._env, order='C', dtype=np.double)
    ish_env[PTR_EXPDROP] = min(cell0.precision*EXTRA_PREC, EXPDROP)

    if cell1 is cell0:
        jsh_atm = ish_atm
        jsh_bas = ish_bas
        jsh_env = ish_env
    else:
        jsh_atm = np.asarray(cell1._atm, order='C', dtype=np.int32)
        jsh_bas = np.asarray(cell1._bas, order='C', dtype=np.int32)
        jsh_env = np.asarray(cell1._env, order='C', dtype=np.double)
        jsh_env[PTR_EXPDROP] = min(cell1.precision*EXTRA_PREC, EXPDROP)

    if shls_slice0 is None:
        shls_slice0 = (0, cell0.nbas)
    i0, i1 = shls_slice0
    if shls_slice1 is None:
        shls_slice1 = shls_slice0
    j0, j1 = shls_slice1

    if hermi == 1:
        assert cell1 is cell0
        assert i0 == j0 and i1 == j1

    key0 = 'cart' if cell0.cart else 'sph'
    ao_loc0 = moleintor.make_loc(ish_bas, key0)
    naoi = ao_loc0[i1] - ao_loc0[i0]
    if hermi == 1:
        ao_loc1 = ao_loc0
    else:
        key1 = 'cart' if cell1.cart else 'sph'
        ao_loc1 = moleintor.make_loc(jsh_bas, key1)
    naoj = ao_loc1[j1] - ao_loc1[j0]

    dm = np.asarray(dm, order='C')
    assert dm.shape[-2:] == (naoi, naoj)

    if dimension is None:
        dimension = cell0.dimension
    assert dimension == getattr(cell1, "dimension", None)
 
    if Ls is None and dimension > 0:
        Ls = np.asarray(cell0.get_lattice_Ls(), order='C')
    elif Ls is None and dimension == 0:
        Ls = np.zeros((1,3))

    if dimension == 0 or kpts is None or gamma_point(kpts):
        nkpts, nimgs = 1, Ls.shape[0]
        dm = dm.reshape(-1,1,naoi,naoj)
    else:
        expkL = np.exp(1j*kpts.reshape(-1,3).dot(Ls.T))
        nkpts, nimgs = expkL.shape
        dm = dm.reshape(-1,nkpts,naoi,naoj)
    n_dm = dm.shape[0]

    #TODO check if cell1 has the same lattice vectors
    if a is None:
        a = cell0.lattice_vectors()
    b = np.linalg.inv(a.T)

    if abs(a-np.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'
    xctype = xctype.upper()
    if xctype == 'LDA':
        comp = 1
    elif xctype == 'GGA':
        if hermi == 1:
            raise RuntimeError('hermi=1 is not supported for GGA functional')
        comp = 4
    else:
        raise NotImplementedError('meta-GGA')

    eval_fn = 'collocate_rho_' + xctype.lower() + lattice_type
    drv = getattr(libdft, "grid_collocate_drv", None)

    def make_rho_(rs_rho, dm):
        try:
            drv(getattr(libdft, eval_fn, None),
                ctypes.byref(rs_rho),
                dm.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(task_list),
                ctypes.c_int(comp), ctypes.c_int(hermi),
                (ctypes.c_int*4)(i0, i1, j0, j1),
                ao_loc0.ctypes.data_as(ctypes.c_void_p),
                ao_loc1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(dimension),
                Ls.ctypes.data_as(ctypes.c_void_p),
                a.ctypes.data_as(ctypes.c_void_p),
                b.ctypes.data_as(ctypes.c_void_p),
                ish_atm.ctypes.data_as(ctypes.c_void_p),
                ish_bas.ctypes.data_as(ctypes.c_void_p),
                ish_env.ctypes.data_as(ctypes.c_void_p),
                jsh_atm.ctypes.data_as(ctypes.c_void_p),
                jsh_bas.ctypes.data_as(ctypes.c_void_p),
                jsh_env.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(cell0.cart))
        except:
            raise RuntimeError("Failed to compute rho.")
        return rs_rho

    gridlevel_info = task_list.contents.gridlevel_info
    rs_rho = init_rs_grid(gridlevel_info, comp)

    rho = []
    for i, dm_i in enumerate(dm):
        if dimension == 0 or kpts is None or gamma_point(kpts):
            make_rho_(rs_rho, dm_i)
        else:
            raise NotImplementedError
        rho.append(rs_rho)

    if n_dm == 1:
        rho = rho[0]
    return rho


def _eval_rhoG(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), deriv=0,
               rhog_high_order=RHOG_HIGH_ORDER):
    cell = mydf.cell

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    task_list = getattr(mydf, 'task_list', None)
    if task_list is None:
        mydf.task_list = task_list = multi_grids_tasks(cell, hermi=hermi)

    assert(deriv < 2)

    gga_high_order = False
    if deriv == 0:
        xctype = 'LDA'
        rhodim = 1
    elif deriv == 1:
        if rhog_high_order:
            xctype = 'GGA'
            rhodim = 4
        else:  # approximate high order derivatives in reciprocal space
            gga_high_order = True
            xctype = 'LDA'
            rhodim = 1
            deriv = 0
        assert(hermi == 1 or gamma_point(kpts))
    elif deriv == 2:  # meta-GGA
        raise NotImplementedError
        assert(hermi == 1 or gamma_point(kpts))

    ignore_imag = (hermi == 1)

    rs_rho = eval_rho(cell, dms, task_list, hermi=hermi, xctype=xctype, kpts=kpts,
                      ignore_imag=ignore_imag)

    nx, ny, nz = mydf.mesh
    rhoG = np.zeros((nset*rhodim,nx,ny,nz), dtype=np.complex128)

    nlevels = task_list.contents.nlevels
    meshes = task_list.contents.gridlevel_info.contents.mesh
    meshes = np.ctypeslib.as_array(meshes, shape=(nlevels,3))
    for ilevel in range(nlevels):
        mesh = meshes[ilevel]
        ngrids = np.prod(mesh)
        if nset > 1:
            rho = []
            for i in range(nset):
                rho.append(np.ctypeslib.as_array(rs_rho[i].contents.data[ilevel], shape=(ngrids,)))
            rho = np.asarray(rho)
        else:
            rho = np.ctypeslib.as_array(rs_rho.contents.data[ilevel], shape=(ngrids,))

        weight = 1./nkpts * cell.vol/ngrids
        rho_freq = tools.fft(rho.reshape(nset*rhodim, -1), mesh)
        rho_freq *= weight
        gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(np.int32)
        gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(np.int32)
        gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(np.int32)
        _takebak_4d(rhoG, rho_freq.reshape((-1,) + tuple(mesh)), (None, gx, gy, gz))

    rhoG = rhoG.reshape(nset,rhodim,-1)

    if gga_high_order:
        Gv = cell.get_Gv(mydf.mesh)
        rhoG1 = np.einsum('np,px->nxp', 1j*rhoG[:,0], Gv)
        rhoG = np.concatenate([rhoG, rhoG1], axis=1)
    return rhoG


def eval_mat(cell, weights, task_list, shls_slice=None, comp=1, hermi=0,
             xctype='LDA', kpts=None, grid_level=None, dimension=None, mesh=None, 
             cell1=None, shls_slice1=None, Ls=None, a=None):

    cell0 = cell
    shls_slice0 = shls_slice
    if cell1 is None:
        cell1 = cell0

    if mesh is None:
        mesh = cell0.mesh

    #TODO mixture of cartesian and spherical bases
    assert cell0.cart == cell1.cart

    ish_atm = np.asarray(cell0._atm, order='C', dtype=np.int32)
    ish_bas = np.asarray(cell0._bas, order='C', dtype=np.int32)
    ish_env = np.asarray(cell0._env, order='C', dtype=np.double)
    ish_env[PTR_EXPDROP] = min(cell0.precision*EXTRA_PREC, EXPDROP)

    if cell1 is cell0:
        jsh_atm = ish_atm
        jsh_bas = ish_bas
        jsh_env = ish_env
    else:
        jsh_atm = np.asarray(cell1._atm, order='C', dtype=np.int32)
        jsh_bas = np.asarray(cell1._bas, order='C', dtype=np.int32)
        jsh_env = np.asarray(cell1._env, order='C', dtype=np.double)
        jsh_env[PTR_EXPDROP] = min(cell1.precision*EXTRA_PREC, EXPDROP)

    if shls_slice0 is None:
        shls_slice0 = (0, cell0.nbas)
    i0, i1 = shls_slice0
    if shls_slice1 is None:
        shls_slice1 = (0, cell1.nbas)
    j0, j1 = shls_slice1

    if hermi == 1:
        assert cell1 is cell0
        assert i0 == j0 and i1 == j1

    key0 = 'cart' if cell0.cart else 'sph'
    ao_loc0 = moleintor.make_loc(ish_bas, key0)
    naoi = ao_loc0[i1] - ao_loc0[i0]
    if hermi == 1:
        ao_loc1 = ao_loc0
    else:
        key1 = 'cart' if cell1.cart else 'sph'
        ao_loc1 = moleintor.make_loc(jsh_bas, key1)
    naoj = ao_loc1[j1] - ao_loc1[j0]

    if dimension is None:
        dimension = cell0.dimension
    assert dimension == getattr(cell1, "dimension", None)

    if Ls is None and dimension > 0:
        Ls = np.asarray(cell0.get_lattice_Ls(), order='C')
    elif Ls is None and dimension == 0:
        Ls = np.zeros((1,3))

    if dimension == 0 or kpts is None or gamma_point(kpts):
        nkpts, nimgs = 1, Ls.shape[0]
    else:
        expkL = np.exp(1j*kpts.reshape(-1,3).dot(Ls.T))
        nkpts, nimgs = expkL.shape

    #TODO check if cell1 has the same lattice vectors
    if a is None:
        a = cell0.lattice_vectors()
    b = np.linalg.inv(a.T)

    if abs(a-np.diag(a.diagonal())).max() < 1e-12:
        lattice_type = '_orth'
    else:
        lattice_type = '_nonorth'

    weights = np.asarray(weights, order='C')
    assert(weights.dtype == np.double)
    xctype = xctype.upper()
    n_mat = None
    if xctype == 'LDA':
        if weights.ndim == 1:
            weights = weights.reshape(-1, np.prod(mesh))
        else:
            n_mat = weights.shape[0]
    elif xctype == 'GGA':
        if hermi == 1:
            raise RuntimeError('hermi=1 is not supported for GGA functional')
        if weights.ndim == 2:
            weights = weights.reshape(-1, 4, np.prod(mesh))
        else:
            n_mat = weights.shape[0]
    else:
        raise NotImplementedError

    eval_fn = 'eval_mat_' + xctype.lower() + lattice_type
    drv = getattr(libdft, "grid_eval_drv", None)

    def make_mat(wv):
        if comp == 1:
            mat = np.zeros((naoi, naoj))
        else:
            mat = np.zeros((comp, naoi, naoj))

        try:
            drv(getattr(libdft, eval_fn, None),
                mat.ctypes.data_as(ctypes.c_void_p),
                wv.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(task_list),
                ctypes.c_int(comp), ctypes.c_int(hermi),
                ctypes.c_int(grid_level),
                (ctypes.c_int*4)(i0, i1, j0, j1),
                ao_loc0.ctypes.data_as(ctypes.c_void_p),
                ao_loc1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(dimension),
                Ls.ctypes.data_as(ctypes.c_void_p),
                a.ctypes.data_as(ctypes.c_void_p),
                b.ctypes.data_as(ctypes.c_void_p),
                ish_atm.ctypes.data_as(ctypes.c_void_p),
                ish_bas.ctypes.data_as(ctypes.c_void_p),
                ish_env.ctypes.data_as(ctypes.c_void_p),
                jsh_atm.ctypes.data_as(ctypes.c_void_p),
                jsh_bas.ctypes.data_as(ctypes.c_void_p),
                jsh_env.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(cell0.cart))
        except:
            raise RuntimeError("Failed to compute rho.")
        return mat

    out = []
    for wv in weights:
        if dimension == 0 or kpts is None or gamma_point(kpts):
            mat = make_mat(wv)
        else:
            raise NotImplementedError
        out.append(mat)

    if n_mat is None:
        out = out[0]
    return out


def _get_j_pass2(mydf, vG, kpts=np.zeros((1,3)), hermi=1, verbose=None):
    cell = mydf.cell
    nkpts = len(kpts)
    nao = cell.nao_nr()
    nx, ny, nz = mydf.mesh
    vG = vG.reshape(-1,nx,ny,nz)
    nset = vG.shape[0]

    task_list = getattr(mydf, 'task_list', None)
    if task_list is None:
        mydf.task_list = task_list = multi_grids_tasks(cell, hermi=hermi)

    at_gamma_point = gamma_point(kpts)
    if at_gamma_point:
        vj_kpts = np.zeros((nset,nkpts,nao,nao))
    else:
        vj_kpts = np.zeros((nset,nkpts,nao,nao), dtype=np.complex128)

    nlevels = task_list.contents.nlevels
    meshes = task_list.contents.gridlevel_info.contents.mesh
    meshes = np.ctypeslib.as_array(meshes, shape=(nlevels,3))
    for ilevel in range(nlevels):
        mesh = meshes[ilevel]
        ngrids = np.prod(mesh)

        gx = np.fft.fftfreq(mesh[0], 1./mesh[0]).astype(np.int32)
        gy = np.fft.fftfreq(mesh[1], 1./mesh[1]).astype(np.int32)
        gz = np.fft.fftfreq(mesh[2], 1./mesh[2]).astype(np.int32)
        sub_vG = _take_4d(vG, (None, gx, gy, gz)).reshape(nset,ngrids)

        v_rs = tools.ifft(sub_vG, mesh).reshape(nset,ngrids)
        vR = np.asarray(v_rs.real, order='C')
        vI = np.asarray(v_rs.imag, order='C')
        if at_gamma_point:
            v_rs = vR

        for i in range(nset):
            mat = eval_mat(cell, vR[i], task_list, comp=1, hermi=hermi,
                           xctype='LDA', kpts=kpts, grid_level=ilevel, mesh=mesh)
            vj_kpts[i] += mat
        if not at_gamma_point and abs(vI).max() > IMAG_TOL:
            raise NotImplementedError

    if nset == 1:
        vj_kpts = vj_kpts[0]
    return vj_kpts


class MultiGridFFTDF2(multigrid.MultiGridFFTDF):

    _eval_rhoG = _eval_rhoG
    _get_j_pass2 = _get_j_pass2
