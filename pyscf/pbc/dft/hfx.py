import copy
import ctypes
import numpy
from pyscf import lib, gto
from pyscf.gto import PTR_EXPCUTOFF
from pyscf.pbc.gto.cell import build_neighbor_list_for_shlpairs
from pyscf.scf import _vhf

libcgto = lib.load_library('libcgto')
libdft = lib.load_library('libdft')
libcvhf = lib.load_library('libcvhf')

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
