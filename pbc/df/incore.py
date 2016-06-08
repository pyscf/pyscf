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
           kpti_kptj=numpy.zeros((2,3)), out=None):
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
    expkL = numpy.exp(1j*numpy.dot(Ls, numpy.reshape(kpti, 3)))
    expkR = numpy.exp(1j*numpy.dot(Ls, numpy.reshape(kptj, 3)))
    gamma_point = abs(kpti).sum() < 1e-9 and abs(kptj).sum() < 1e-9

    nao = cell.nao_nr()
    #naoaux = auxcell.nao_nr('ssc' in intor)
    naoaux = auxcell.nao_nr()
    buf = numpy.empty((nao,nao,naoaux,comp), order='F')
    ints = _wrap_int3c(cell, auxcell, intor, comp, buf)
    atm, bas, env, ao_loc = ints._envs[:4]
    shls_slice = (0, cell.nbas, cell.nbas, cell.nbas*2,
                  cell.nbas*2, cell.nbas*2+auxcell.nbas)
    c_shls_slice = (ctypes.c_int*6)(*(shls_slice[:6]))

    xyz = cell.atom_coords().copy('C')
    ptr_coordL = atm[         :cell.natm  ,pyscf.gto.PTR_COORD]
    ptr_coordR = atm[cell.natm:cell.natm*2,pyscf.gto.PTR_COORD]
    ptr_coordL = numpy.vstack((ptr_coordL,ptr_coordL+1,ptr_coordL+2)).T.copy('C')
    ptr_coordR = numpy.vstack((ptr_coordR,ptr_coordR+1,ptr_coordR+2)).T.copy('C')

    mat = 0
    if aosym == 's1' or abs(kpti-kptj).sum() > 1e-9:
        for l, L1 in enumerate(Ls):
            env[ptr_coordL] = xyz + L1
            for m, L2 in enumerate(Ls):
                env[ptr_coordR] = xyz + L2
                if gamma_point:
                    mat += ints(c_shls_slice)
                else:
                    mat += ints(c_shls_slice) * (expkL[l].conj() * expkR[m])
        mat = mat.reshape((nao*nao,naoaux,comp), order='A')
    else:
        for l, L1 in enumerate(Ls):
            env[ptr_coordL] = xyz + L1
            for m in range(l):
                env[ptr_coordR] = xyz + Ls[m]
                if gamma_point:
                    mat += ints(c_shls_slice)
                else:
                    mat += ints(c_shls_slice) * (expkL[l].conj() * expkR[m])

            env[ptr_coordR] = xyz + L1
            if gamma_point:
                mat += ints(c_shls_slice) * .5
            else:
                mat += ints(c_shls_slice) * (.5+0j)
        if gamma_point:
            mat = mat + mat.swapaxes(0,1)
        else:
            mat = mat + mat.swapaxes(0,1).conj()
        mat = mat[numpy.tril_indices(nao)]
    if comp == 1:
        mat = mat.reshape(mat.shape[:-1], order='A')
    else:
        mat = numpy.rollaxis(mat, -1, 0)
    return mat


def fill_2c2e(cell, auxcell, intor='cint2c2e_sph', hermi=0, kpt=numpy.zeros(3)):
    '''2-center 2-electron AO integrals (L|ij), where L is the auxiliary basis.
    '''
    if hermi != 0:
        hermi = pyscf.lib.SYMMETRIC
    return auxcell.pbc_intor(intor, 1, hermi, kpt)


def _wrap_int3c(cell, auxcell, intor, comp, buf):
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

    if buf is None:
        nao = ao_loc[cell.nbas]
        naoaux = ao_loc[-1] - ao_loc[cell.nbas*2]
        buf = numpy.empty((nao,nao,naoaux,comp), order='F')

    cintopt = _vhf.make_cintopt(atm, bas, env, intor)
    fintor = pyscf.gto.moleintor._fpointer(intor)
    fill = pyscf.gto.moleintor._fpointer('GTOnr3c_fill_s1')
    drv = pyscf.gto.moleintor.libcgto.GTOnr3c_drv
    c_buf = buf.ctypes.data_as(ctypes.c_void_p)
    c_comp = ctypes.c_int(comp)
    c_ao_loc = ao_loc.ctypes.data_as(ctypes.c_void_p)
    c_atm = atm.ctypes.data_as(ctypes.c_void_p)
    c_bas = bas.ctypes.data_as(ctypes.c_void_p)
    c_env = env.ctypes.data_as(ctypes.c_void_p)
    c_natm = ctypes.c_int(natm)
    c_nbas = ctypes.c_int(nbas)
    def ints(c_shls_slice):
        drv(fintor, fill, c_buf, c_comp, c_shls_slice, c_ao_loc, cintopt,
            c_atm, c_natm, c_bas, c_nbas, c_env)
        return buf
    # Save the numpy arrays in envs because ctypes does not increase their
    # reference counting.
    # MUST be keeping track them out of this function scope.
    ints._envs = (atm, bas, env, ao_loc, buf)
    return ints
