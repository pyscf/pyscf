#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import numpy
import pyscf.lib
import pyscf.gto
import pyscf.df
from pyscf.lib import logger
from pyscf.scf import _vhf

libpbc = pyscf.lib.load_library('libpbc')

try:
## Moderate speedup by caching eval_ao
    from joblib import Memory
    memory = Memory(cachedir='./tmp/', mmap_mode='r', verbose=0)
    def memory_cache(f):
        g = memory.cache(f)
        def maybe_cache(*args, **kwargs):
            if pyscf.pbc.DEBUG:
                return g(*args, **kwargs)
            else:
                return f(*args, **kwargs)
        return maybe_cache
except:
    memory_cache = lambda f: f

def format_aux_basis(cell, auxbasis='weigend+etb'):
    '''
    See df.incore.format_aux_basis
    '''
    auxcell = pyscf.df.incore.format_aux_basis(cell, auxbasis)
    auxcell.nimgs = auxcell.get_nimgs(auxcell.precision)
    return auxcell

@memory_cache
def aux_e2(cell, auxcell, intor='cint3c2e_sph', aosym='s1', comp=1,
           kpti_kptj=numpy.zeros((2,3))):
    '''3-center AO integrals (ij|L) with double lattice sum:
    \sum_{lm} (i[l]j[m]|L[0]), where L is the auxiliary basis.

    Returns:
        (nao_pair, naux) array
    '''
    assert(comp == 1)
    # sum over largest number of images in either cell or auxcell
    nimgs = numpy.max((cell.nimgs, auxcell.nimgs), axis=0)
    Ls = cell.get_lattice_Ls(nimgs)
    logger.debug1(cell, "Images %s", nimgs)
    logger.debug3(cell, "Ls = %s", Ls)

    kpti, kptj = kpti_kptj
    expkL = numpy.exp(1j*numpy.asarray(numpy.dot(Ls, numpy.reshape(kpti, (1,3)).T), order='C'))
    expkR = numpy.exp(1j*numpy.asarray(numpy.dot(Ls, numpy.reshape(kptj, (1,3)).T), order='C'))
    gamma_point = abs(kpti).sum() < 1e-9 and abs(kptj).sum() < 1e-9

    nao = cell.nao_nr()
    #naux = auxcell.nao_nr('ssc' in intor)
    naux = auxcell.nao_nr()
    buf = [numpy.zeros((nao,nao,naux,comp), order='F', dtype=numpy.complex128)]
    ints = _wrap_int3c(cell, auxcell, intor, comp, Ls, buf)
    atm, bas, env = ints._envs[:3]
    shls_slice = (0, cell.nbas, cell.nbas, cell.nbas*2,
                  cell.nbas*2, cell.nbas*2+auxcell.nbas)
    c_shls_slice = (ctypes.c_int*6)(*(shls_slice[:6]))

    xyz = cell.atom_coords().copy('C')
    ptr_coordL = atm[:cell.natm,pyscf.gto.PTR_COORD]
    ptr_coordL = numpy.vstack((ptr_coordL,ptr_coordL+1,ptr_coordL+2)).T.copy('C')

    if aosym == 's1' or abs(kpti-kptj).sum() > 1e-9:
        for l, L1 in enumerate(Ls):
            env[ptr_coordL] = xyz + L1
            exp_Lk = expkL[l].conj() * expkR
            ints(exp_Lk, c_shls_slice)
        mat, buf = buf[0], None
    else:
        for l, L1 in enumerate(Ls):
            env[ptr_coordL] = xyz + L1
            exp_Lk = expkL[l].conj() * expkR[:l+1]
            exp_Lk[l] = .5
            ints(exp_Lk, c_shls_slice)
        mat, buf = buf[0], None
        if gamma_point:
            mat = mat.real + mat.real.swapaxes(0,1)
        else:
            mat = mat + mat.swapaxes(0,1).conj()
        mat = mat[numpy.tril_indices(nao)]
    if comp == 1:
        mat = mat.reshape(-1,naux)
    else:
        mat = numpy.rollaxis(mat, -1, 0).reshape(comp,-1,naux)
    return mat


def fill_2c2e(cell, auxcell, intor='cint2c2e_sph', hermi=0, kpt=numpy.zeros(3)):
    '''2-center 2-electron AO integrals (L|ij), where L is the auxiliary basis.
    '''
    if hermi != 0:
        hermi = pyscf.lib.SYMMETRIC
    return auxcell.pbc_intor(intor, 1, hermi, kpt)


def _wrap_int3c(cell, auxcell, intor, comp, Ls, out_lst):
    atm, bas, env = pyscf.gto.conc_env(cell._atm, cell._bas, cell._env,
                                       cell._atm, cell._bas, cell._env)
    atm, bas, env = pyscf.gto.conc_env(atm, bas, env,
                                       auxcell._atm, auxcell._bas, auxcell._env)
    atm = numpy.asarray(atm, dtype=numpy.int32)
    bas = numpy.asarray(bas, dtype=numpy.int32)
    env = numpy.asarray(env, dtype=numpy.double)
    natm = len(atm)
    nbas = len(bas)
    if 'ssc' in intor:
        ao_loc = cell.ao_loc_nr()
        ao_loc = numpy.hstack((ao_loc[:-1], ao_loc[-1]+ao_loc))
        ao_loc = numpy.hstack((ao_loc[:-1], ao_loc[-1]+auxcell.ao_loc_nr(cart=True)))
        ao_loc = numpy.asarray(ao_loc, dtype=numpy.int32)
    else:
        ao_loc = pyscf.gto.moleintor.make_loc(bas, intor)

    cintopt = _vhf.make_cintopt(atm, bas, env, intor)
    fintor = pyscf.gto.moleintor._fpointer(intor)
    fill = getattr(libpbc, 'PBCnr3c_fill_s1')
    drv = libpbc.PBCnr3c_drv
    outs = (ctypes.c_void_p*len(out_lst))(
            *[out.ctypes.data_as(ctypes.c_void_p) for out in out_lst])
    xyz = numpy.asarray(cell.atom_coords(), order='C')
    ptr_coords = numpy.asarray(atm[cell.natm:cell.natm*2,pyscf.gto.PTR_COORD],
                               dtype=numpy.int32, order='C')
    c_ptr_coords = ptr_coords.ctypes.data_as(ctypes.c_void_p)
    c_xyz = xyz.ctypes.data_as(ctypes.c_void_p)
    c_nxyz = ctypes.c_int(len(xyz))
    Ls = numpy.asarray(Ls, order='C')
    c_Ls = Ls.ctypes.data_as(ctypes.c_void_p)
    c_comp = ctypes.c_int(comp)
    c_ao_loc = ao_loc.ctypes.data_as(ctypes.c_void_p)
    c_atm = atm.ctypes.data_as(ctypes.c_void_p)
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    c_env = env.ctypes.data_as(ctypes.c_void_p)
    c_natm = ctypes.c_int(natm)
    c_nbas = ctypes.c_int(nbas)
    def ints(facs, c_shls_slice):
        nimgs, nkpts = facs.shape
        drv(fintor, fill, outs, c_xyz, c_ptr_coords, c_nxyz,
            c_Ls, ctypes.c_int(nimgs),
            facs.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nkpts), c_comp,
            c_shls_slice, c_ao_loc, cintopt, c_atm, c_natm, c_bas, c_nbas, c_env)
    # Save the numpy arrays in envs because ctypes does not increase their
    # reference counting.
    ints._envs = (atm, bas, env, ao_loc, xyz, ptr_coords, Ls)
    return ints
