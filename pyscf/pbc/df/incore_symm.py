import ctypes
import copy
import numpy
from pyscf import lib
from pyscf import gto
from pyscf.scf import _vhf
from pyscf.pbc.gto import _pbcintor
from pyscf.pbc.symm import symmetry
from pyscf.pbc.symm import petite_list
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique, KPT_DIFF_TOL

libpbc = lib.load_library('libpbc')

def wrap_int3c_symm(cell, auxcell, intor='int3c2e', aosym='s1', comp=1,
               kptij_lst=numpy.zeros((1,2,3)), cintopt=None, pbcopt=None):
    intor = cell._add_suffix(intor)
    pcell = copy.copy(cell)
    pcell._atm, pcell._bas, pcell._env = \
    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    ao_loc = gto.moleintor.make_loc(bas, intor)
    if cell.cart or auxcell.cart or 'ssc' in intor:
        raise NotImplementedError
    aux_loc = auxcell.ao_loc_nr(auxcell.cart or 'ssc' in intor)
    ao_loc = numpy.asarray(numpy.hstack([ao_loc, ao_loc[-1]+aux_loc[1:]]),
                           dtype=numpy.int32)
    atm, bas, env = gto.conc_env(atm, bas, env,
                                 auxcell._atm, auxcell._bas, auxcell._env)
    Ls = cell.get_lattice_Ls()
    nimgs = len(Ls)
    nbas = cell.nbas

    shlcen = numpy.asarray([env[atm[bas[ia,0],1]:atm[bas[ia,0],1]+3] for ia in range(len(bas))])
    atm_coord = numpy.asarray([cell._env[cell._atm[ia,1]:cell._atm[ia,1]+3] for ia in range(cell.natm)])
    shlcen_atm_idx = numpy.where(abs(shlcen.reshape(-1,1,3) - atm_coord).sum(axis=2) < 1e-6)[1]
    assert(len(shlcen_atm_idx) == len(bas))
    shlcen_atm_idx =  shlcen_atm_idx.astype(numpy.int32)

    pet = petite_list.Petite_List(cell,Ls,auxcell=auxcell)
    pet.kernel()
    shltrip_cen_idx = pet.shltrip_cen_idx
    L2iL = numpy.asarray(pet.buf[:,:,0]).copy()
    ops = numpy.asarray(pet.buf[:,:,1]).copy()
    nop = len(pet.Dmats)
    rot_mat_size = 0
    for mat in pet.Dmats[0]:
        rot_mat_size += mat.size
    Dmats = numpy.empty((nop, rot_mat_size), dtype=numpy.double)
    for i in range(nop):
        Dmats[i] = numpy.concatenate(pet.Dmats[i], axis=None)
    rot_loc = symmetry.make_rot_loc(pet.l_max,intor)

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    if gamma_point(kptij_lst):
        kk_type = 'g'
        dtype = numpy.double
        nkpts = nkptij = 1
        kptij_idx = numpy.array([0], dtype=numpy.int32)
        expkL = numpy.ones(1)
    elif is_zero(kpti-kptj):  # j_only
        kk_type = 'k'
        dtype = numpy.complex128
        kpts = kptij_idx = numpy.asarray(kpti, order='C')
        expkL = numpy.exp(1j * numpy.dot(kpts, Ls.T))
        nkpts = nkptij = len(kpts)
    else:
        kk_type = 'kk'
        dtype = numpy.complex128
        kpts = unique(numpy.vstack([kpti,kptj]))[0]
        expkL = numpy.exp(1j * numpy.dot(kpts, Ls.T))
        wherei = numpy.where(abs(kpti.reshape(-1,1,3)-kpts).sum(axis=2) < KPT_DIFF_TOL)[1]
        wherej = numpy.where(abs(kptj.reshape(-1,1,3)-kpts).sum(axis=2) < KPT_DIFF_TOL)[1]
        nkpts = len(kpts)
        kptij_idx = numpy.asarray(wherei*nkpts+wherej, dtype=numpy.int32)
        nkptij = len(kptij_lst)


    fill = 'PBCnr3c_fill_symm_%s%s' % (kk_type, aosym[:2])
    drv = libpbc.PBCnr3c_symm_drv
    if cintopt is None:
        if nbas > 0:
            cintopt = _vhf.make_cintopt(atm, bas, env, intor)
        else:
            cintopt = lib.c_null_ptr()
# Remove the precomputed pair data because the pair data corresponds to the
# integral of cell #0 while the lattice sum moves shls to all repeated images.
        if intor[:3] != 'ECP':
            libpbc.CINTdel_pairdata_optimizer(cintopt)
    if pbcopt is None:
        pbcopt = _pbcintor.PBCOpt(pcell).init_rcut_cond(pcell)
    if isinstance(pbcopt, _pbcintor.PBCOpt):
        cpbcopt = pbcopt._this
    else:
        cpbcopt = lib.c_null_ptr()

    def int3c(shls_slice, out):
        shls_slice = (shls_slice[0], shls_slice[1],
                      nbas+shls_slice[2], nbas+shls_slice[3],
                      nbas*2+shls_slice[4], nbas*2+shls_slice[5])
        drv(getattr(libpbc, intor), getattr(libpbc, fill),
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkptij), ctypes.c_int(nkpts),
            ctypes.c_int(comp), ctypes.c_int(nimgs),
            Ls.ctypes.data_as(ctypes.c_void_p),
            expkL.ctypes.data_as(ctypes.c_void_p),
            kptij_idx.ctypes.data_as(ctypes.c_void_p),
            (ctypes.c_int*6)(*shls_slice),
            ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cpbcopt,
            atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm),
            bas.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nbas),  # need to pass cell.nbas to libpbc.PBCnr3c_drv
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size),
            shlcen_atm_idx.ctypes.data_as(ctypes.c_void_p), shltrip_cen_idx.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(len(shltrip_cen_idx)),
            L2iL.ctypes.data_as(ctypes.c_void_p), ops.ctypes.data_as(ctypes.c_void_p),
            Dmats.ctypes.data_as(ctypes.c_void_p), rot_loc.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nop), ctypes.c_int(rot_mat_size))
        return out
    return int3c
