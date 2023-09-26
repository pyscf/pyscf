import ctypes
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band
from pyscf.gto import moleintor
from .multigrid_pair import _eval_rhoG, _update_task_list, EXTRA_PREC, EXPDROP, PTR_EXPDROP
from pyscf.pbc.dft.multigrid.utils import _take_4d
from pyscf.pbc.lib.kpts_helper import gamma_point

libpbc = lib.load_library('libpbc')
libdft = lib.load_library('libdft')


def eval_pgfpairs(cell, task_list, shls_slice=None, hermi=0, xctype='LDA', kpts=None,
             dimension=None, cell1=None, shls_slice1=None, Ls=None,
             a=None, ignore_imag=False):
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
    xctype = xctype.upper()
    if xctype == 'LDA':
        comp = 1
    else:
        raise RuntimeError

    drv = getattr(libdft, "eval_pgfpairs", None)
    if dimension == 0 or kpts is None or gamma_point(kpts):
        drv(ctypes.byref(task_list),
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

    else:
        raise NotImplementedError
    return


def eval_mat(cell, weights, mo_coeff, task_list, shls_slice=None, comp=1, hermi=0, deriv=0,
             xctype='LDA', kpts=None, grid_level=None, dimension=None, mesh=None,
             cell1=None, shls_slice1=None, Ls=None, a=None):

    cell0 = cell
    shls_slice0 = shls_slice
    if cell1 is None:
        cell1 = cell0

    assert cell1 is cell0
    assert hermi == 0
    assert deriv == 0

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
    else:
        raise NotImplementedError

    eval_fn = 'eval_mat_' + xctype.lower() + lattice_type
    drv = getattr(libdft, "grid_hfx_integrate", None)

    def make_mat(wv):
        if comp == 1:
            mat = np.zeros((naoj,))
        else:
            raise RuntimeError

        try:
            drv(getattr(libdft, eval_fn, None),
                mat.ctypes.data_as(ctypes.c_void_p),
                mo_coeff.ctypes.data_as(ctypes.c_void_p),
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
        except Exception as e:
            raise RuntimeError("Failed to compute rho. %s" % e)
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


def contract(mydf, vG, mo_coeff, kpts=np.zeros((1,3)), hermi=0, verbose=None):
    cell = mydf.cell
    nkpts = len(kpts)
    nao = cell.nao_nr()
    nx, ny, nz = mydf.mesh
    vG = vG.reshape(-1,nx,ny,nz)
    nset = vG.shape[0]
    assert nset == 1
    assert nkpts == 1

    task_list = _update_task_list(mydf, hermi=hermi, ngrids=mydf.ngrids,
                                  ke_ratio=mydf.ke_ratio, rel_cutoff=mydf.rel_cutoff)

    at_gamma_point = gamma_point(kpts)
    if at_gamma_point:
        vj_kpts = np.zeros((nset,nao))
    else:
        raise NotImplementedError

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
        #vI = np.asarray(v_rs.imag, order='C')
        if at_gamma_point:
            v_rs = vR

        mat = eval_mat(cell, vR, mo_coeff, task_list, comp=1, hermi=hermi,
                       xctype='LDA', kpts=kpts, grid_level=ilevel, mesh=mesh)
        vj_kpts += np.asarray(mat).reshape(nset,nao)

    if nset == 1:
        vj_kpts = vj_kpts[0]
    return vj_kpts


def get_k(mydf, dm, hermi=1, kpt=np.zeros(3), kpts_band=None, exxdiv=None, verbose=None):
    kpts = np.asarray(kpt).reshape(-1,3)
    if kpts_band is None:
        kpts_band = np.zeros((1,3))
    if not (gamma_point(kpts) and gamma_point(kpts_band)):
        raise NotImplementedError

    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    coulG = tools.get_coulG(cell, mesh=mesh)

    if getattr(dm, 'mo_coeff', None) is not None:
        mo_coeff = dm.mo_coeff
        mo_occ   = dm.mo_occ
    else:
        raise NotImplementedError

    nocc = np.sum(mo_occ > 0)
    mo_coeff = mo_coeff[:,mo_occ>0]
    nao = mo_coeff.shape[0]

    mo_coeff = np.asarray(mo_coeff.T, order='C')

    vk = np.zeros((nocc,nao), order='C')
    buf = np.empty((nao,nao), order='C')
    t0 = (logger.process_clock(), logger.perf_counter())
    for i in range(1):
        for j in range(1):
            fake_dm = np.outer(mo_coeff[i], mo_coeff[j], out=buf)
            rhoG = _eval_rhoG(mydf, fake_dm, hermi=0, kpts=kpts, deriv=0)
            rhoG = rhoG.flatten()
            vG = lib.multiply(rhoG, coulG)
            vk[i] += contract(mydf, vG, mo_coeff[j], kpts=kpts, hermi=0)
        t0 = log.timer(f'get_k_occRI iter {i}', *t0)

    coulG = None
    buf = None

    eval_pgfpairs(cell, mydf.task_list, hermi=0)
    t0 = log.timer(f'eval_pgfpairs', *t0)
    return vk

"""
def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=np.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    cell = mydf.cell
    mesh = mydf.mesh
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]

    if getattr(dm_kpts, 'mo_coeff', None) is not None:
        mo_coeff = dm_kpts.mo_coeff
        mo_occ   = dm_kpts.mo_occ
    else:
        mo_coeff = None

    kpts = np.asarray(kpts)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    weight = 1./nkpts * (cell.vol/ngrids)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=dms.dtype)
    else:
        vk_kpts = np.zeros((nset,nband,nao,nao), dtype=np.complex128)

    coords = mydf.grids.coords
    ao2_kpts = [np.asarray(ao.T, order='C')
                for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts)]
    if input_band is None:
        ao1_kpts = ao2_kpts
    else:
        ao1_kpts = [np.asarray(ao.T, order='C')
                    for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts_band)]
    if mo_coeff is not None and nset == 1:
        mo_coeff = [mo_coeff[k][:,occ>0] * np.sqrt(occ[occ>0])
                    for k, occ in enumerate(mo_occ)]
        ao2_kpts = [np.dot(mo_coeff[k].T, ao) for k, ao in enumerate(ao2_kpts)]

    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/ngrids/nao)))
    logger.debug1(mydf, 'fft_jk: get_k_kpts max_memory %s  blksize %d',
                  max_memory, blksize)
    #ao1_dtype = np.result_type(*ao1_kpts)
    #ao2_dtype = np.result_type(*ao2_kpts)
    vR_dm = np.empty((nset,nao,ngrids), dtype=vk_kpts.dtype)

    t1 = (logger.process_clock(), logger.perf_counter())
    for k2, ao2T in enumerate(ao2_kpts):
        if ao2T.size == 0:
            continue

        kpt2 = kpts[k2]
        naoj = ao2T.shape[0]
        if mo_coeff is None or nset > 1:
            ao_dms = [lib.dot(dms[i,k2], ao2T.conj()) for i in range(nset)]
        else:
            ao_dms = [ao2T.conj()]

        for k1, ao1T in enumerate(ao1_kpts):
            kpt1 = kpts_band[k1]

            # If we have an ewald exxdiv, we add the G=0 correction near the
            # end of the function to bypass any discretization errors
            # that arise from the FFT.
            if exxdiv == 'ewald' or exxdiv is None:
                coulG = tools.get_coulG(cell, kpt2-kpt1, False, mydf, mesh)
            else:
                coulG = tools.get_coulG(cell, kpt2-kpt1, exxdiv, mydf, mesh)
            if is_zero(kpt1-kpt2):
                expmikr = np.array(1.)
            else:
                expmikr = np.exp(-1j * np.dot(coords, kpt2-kpt1))

            for p0, p1 in lib.prange(0, nao, blksize):
                rho1 = np.einsum('ig,jg->ijg', ao1T[p0:p1].conj()*expmikr, ao2T)
                vG = tools.fft(rho1.reshape(-1,ngrids), mesh)
                rho1 = None
                vG *= coulG
                vR = tools.ifft(vG, mesh).reshape(p1-p0,naoj,ngrids)
                vG = None
                if vR_dm.dtype == np.double:
                    vR = vR.real
                for i in range(nset):
                    np.einsum('ijg,jg->ig', vR, ao_dms[i], out=vR_dm[i,p0:p1])
                vR = None
            vR_dm *= expmikr.conj()

            for i in range(nset):
                vk_kpts[i,k1] += weight * lib.dot(vR_dm[i], ao1T.T)
        t1 = logger.timer_debug1(mydf, 'get_k_kpts: make_kpt (%d,*)'%k2, *t1)

    if exxdiv == 'ewald':
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)

    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)
"""


def isdf(mydf, dm_kpts, hermi=1, naux=None, max_cycle=100, kpts=None, kpts_band=None, verbose=None):
    if kpts is None:
        kpts = mydf.kpts
    log = logger.new_logger(mydf, verbose)
    cell = mydf.cell
    mesh = mydf.mesh
    ngrids = np.prod(mesh)
    if naux is None:
        naux = cell.nao * 4

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    assert nset == 1
    assert nkpts == 1
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band

    rhoG = _eval_rhoG(mydf, dm_kpts, hermi, kpts, deriv=0)

    weight = cell.vol / ngrids
    rhoR = tools.ifft(rhoG.reshape(-1,ngrids), mesh).real * (1./weight)
    rhoR = rhoR.reshape(nset,-1,ngrids)
    rhoR = np.asarray(rhoR.flatten(), order='C')

    coords = np.asarray(mydf.grids.coords, order='C')
    assert coords.shape[0] == ngrids

    centroids = np.empty((naux,), dtype=np.int32)
    a = np.asarray(cell.lattice_vectors(), order='C')
    Ls = np.asarray(cell.get_lattice_Ls(), order='C')
    mesh = np.asarray(mesh, order='C', dtype=np.int32)

    def temp_fn(centroids):
        drv = getattr(libpbc, 'kmeans_orth')
        drv(centroids.ctypes.data_as(ctypes.c_void_p),
            rhoR.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(naux),
            coords.ctypes.data_as(ctypes.c_void_p),
            a.ctypes.data_as(ctypes.c_void_p),
            mesh.ctypes.data_as(ctypes.c_void_p),
            Ls.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(Ls)),
            ctypes.c_int(max_cycle))
        return centroids
    centroids = temp_fn(centroids)
    return centroids



##################
# Short-range HF #
##################
'''
def sr_hfx(cell, dms, omega, hyb, intor="int2e", shls_slice=None, Ls=None, precision=None, direct_scf_tol=None):
    if precision is None:
        precision = cell.precision
    if direct_scf_tol is None:
        direct_scf_tol = cell.precision**1.5
    if Ls is None:
        Ls = cell.get_lattice_Ls()
    nbas = cell.nbas
    if shls_slice is None:
        shls_slice = (0, nbas, 0, nbas, 0, nbas, 0, nbas)

    a = cell.lattice_vectors()
    b = numpy.linalg.inv(a.T)

    nao = cell.nao
    dms = dms.reshape(-1, nao, nao)
    ndm = len(dms)
    vk = numpy.zeros((ndm, nao, nao))

    nl = _set_q_cond(cell, omega=omega)

    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    pcell = copy.copy(cell)
    pcell._atm, pcell._bas, pcell._env = \
            atm, bas, env = gto.conc_env(atm, bas, env,
                                         atm, bas, env)
    intor = cell._add_suffix(intor)
    pcell.omega = -omega
    pcell._env[PTR_EXPCUTOFF] = env[PTR_EXPCUTOFF] = abs(numpy.log(direct_scf_tol**2))
    #cintopt = _vhf.make_cintopt(atm, bas, env, intor)
    #libcgto.CINTdel_pairdata_optimizer(cintopt)
    cintopt = lib.c_null_ptr()
    ao_loc = gto.moleintor.make_loc(bas, intor)

    vhfopt = _vhf.VHFOpt(pcell, intor)
    vhfopt._cintopt = libcgto.CINTdel_pairdata_optimizer(vhfopt._cintopt)
    vhfopt.direct_scf_tol = direct_scf_tol
    dm_cond = [lib.condense('NP_absmax', dm, ao_loc[:nbas+1], ao_loc[:nbas+1])
               for dm in dms]
    dm_cond = numpy.asarray(numpy.max(dm_cond, axis=0), order='C')
    libcvhf.CVHFset_dm_cond(vhfopt._this,
                            dm_cond.ctypes.data_as(ctypes.c_void_p), dm_cond.size)
    dm_cond = None

    shls_slice = (shls_slice[0], shls_slice[1],
                  shls_slice[2]+nbas, shls_slice[3]+nbas,
                  shls_slice[4]+nbas*2, shls_slice[5]+nbas*2,
                  shls_slice[6]+nbas*3, shls_slice[7]+nbas*3)

    fdot = getattr(libdft, 'PBCDFT_contract_k_s1')
    drv = getattr(libdft, 'PBCDFT_direct_drv')
    drv(fdot, getattr(libcgto, intor),
        vk.ctypes.data_as(ctypes.c_void_p),
        dms.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ndm),
        ctypes.c_int(nao), ctypes.byref(nl),
        (ctypes.c_int*8)(*shls_slice),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        cintopt, vhfopt._this,
        Ls.ctypes.data_as(ctypes.c_void_p),
        a.ctypes.data_as(ctypes.c_void_p),
        b.ctypes.data_as(ctypes.c_void_p),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
        ctypes.c_int(nbas),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))

    if ndm == 1:
        vk = vk[0]
    return -.5 * hyb * vk

def _set_q_cond(cell, intor="int2e", omega=None, shls_slice=None,
                precision=None, direct_scf_tol=None, hermi=0, Ls=None):
    if precision is None:
        precision = cell.precision
    if direct_scf_tol is None:
        direct_scf_tol = cell.precision**1.5
    if Ls is None:
        Ls = cell.get_lattice_Ls()
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas)

    ish_rcut = cell.rcut_by_shells(precision=precision,
                                   return_pgf_radius=False)

    nl = build_neighbor_list_for_shlpairs(cell, Ls=Ls, ish_rcut=ish_rcut,
                                          hermi=hermi, precision=precision)

    intor = cell._add_suffix(intor)
    pcell = copy.copy(cell)
    pcell._atm, pcell._bas, pcell._env = \
            atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                         cell._atm, cell._bas, cell._env)
    pcell.omega = -omega
    pcell._env[PTR_EXPCUTOFF] = env[PTR_EXPCUTOFF] = abs(numpy.log(precision**2))
    #cintopt = _vhf.make_cintopt(atm, bas, env, intor)
    #libcgto.CINTdel_pairdata_optimizer(cintopt)
    cintopt = lib.c_null_ptr()
    ao_loc = gto.moleintor.make_loc(bas, intor)

    shls_slice = (shls_slice[0], shls_slice[1],
                  shls_slice[2] + cell.nbas, shls_slice[3] + cell.nbas)

    set_q_cond = getattr(libdft, "PBCDFT_set_int2e_q_cond")
    set_q_cond(getattr(libcgto, intor), cintopt,
               ctypes.byref(nl), Ls.ctypes.data_as(ctypes.c_void_p),
               (ctypes.c_int*4)(*shls_slice),
               ao_loc.ctypes.data_as(ctypes.c_void_p),
               atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
               bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
               ctypes.c_int(cell.nbas),
               env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))
    return nl

'''
if __name__ == "__main__":
    from pyscf import scf
    from pyscf.pbc import gto as pbcgto
    from pyscf.pbc import dft as pbcdft
    boxlen = 10.0
    cell = pbcgto.Cell()
    cell.a=numpy.array([[boxlen,0.0,0.0],
                        [0.0,boxlen,0.0],
                        [0.0,0.0,boxlen]])
    cell.atom="""
        O          1.84560        1.21649        1.10372
        H          2.30941        1.30070        1.92953
        H          0.91429        1.26674        1.28886
    """
    cell.basis='gth-szv'
    cell.precision=1e-8
    cell.pseudo='gth-pade'
    cell.build()
    print(cell.rcut)

    mf = scf.RHF(cell)
    mf.kernel()
    dm0 = mf.make_rdm1()

    mf = pbcdft.RKS(cell)
    mf.xc = "hse06"
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=cell.spin)
    print(omega, alpha, hyb)
    vk = sr_hfx(cell, dm0, omega, hyb, direct_scf_tol=cell.precision)
    ek = numpy.einsum("ij,ji->", vk, dm0)
    from pyscf import scf
    vk0 = scf.hf.get_jk(cell, dm0, hermi=1, vhfopt=None, with_j=False, with_k=True, omega=-omega)[1]
    vk0 = -.5 * hyb * vk0
    ek0 = numpy.einsum("ij,ji->", vk0, dm0)
    print(abs(ek - ek0))
    print(abs(vk - vk0).max())

    '''
    Ls = cell.get_lattice_Ls()
    print("nimgs = ", len(Ls))
    nl = _set_q_cond(cell, omega=0.11)
    ni = nl.contents.nish
    nj = nl.contents.njsh
    print(cell.nbas, ni, nj)
    for i in range(ni):
        for j in range(nj):
            pair = nl.contents.pairs[i*nj+j]
            nimgs = pair.contents.nimgs
            if nimgs > 0:
                iL = pair.contents.Ls_list
                q_cond = pair.contents.q_cond
                center = pair.contents.center
                print("shell pair ", i, j)
                for k in range(nimgs):
                    print(iL[k])
                    print(q_cond[k])
                    print(center[k*3+0], center[k*3+1], center[k*3+2])
    '''
