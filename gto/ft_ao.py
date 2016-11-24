#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Analytic Fourier transformation for AO and AO-pair value
'''

import ctypes
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.gto.moleintor import libcgto

# TODO: in C code, store complex data in two vectors for real and imag part

#
# \int mu*nu*exp(-ik*r) dr
#
# Note for nonuniform_orth grids, invh is reloaded as the base grids for x,y,z axes
#
def ft_aopair(mol, Gv, shls_slice=None, aosym='s1',
              invh=None, gxyz=None, gs=None, buf=None, verbose=None):
    ''' FT transform AO pair
    \int i(r) j(r) exp(-ikr) dr^3
    '''
    if shls_slice is None:
        shls_slice = (0, mol.nbas, 0, mol.nbas)
    nGv = Gv.shape[0]
    if gxyz is None or invh is None or gs is None:
        GvT = numpy.asarray(Gv.T, order='C')
        p_gxyzT = lib.c_null_ptr()
        p_gs = (ctypes.c_int*3)(0,0,0)
        p_invh = (ctypes.c_double*1)(0)
        eval_gz = 'GTO_Gv_general'
    else:
        GvT = numpy.asarray(Gv.T, order='C')
        gxyzT = numpy.asarray(gxyz.T, order='C', dtype=numpy.int32)
        p_gxyzT = gxyzT.ctypes.data_as(ctypes.c_void_p)
        p_gs = (ctypes.c_int*3)(*gs)
# Guess what type of eval_gz to use
        if isinstance(invh, numpy.ndarray) and invh.shape == (3,3):
            p_invh = invh.ctypes.data_as(ctypes.c_void_p)
            if numpy.allclose(invh-numpy.diag(invh.diagonal()), 0):
                eval_gz = 'GTO_Gv_uniform_orth'
            else:
                eval_gz = 'GTO_Gv_uniform_nonorth'
        else:
            invh = numpy.hstack(invh)
            p_invh = invh.ctypes.data_as(ctypes.c_void_p)
            eval_gz = 'GTO_Gv_nonuniform_orth'

    fn = libcgto.GTO_ft_ovlp_mat
    intor = getattr(libcgto, 'GTO_ft_ovlp_sph')
    eval_gz = getattr(libcgto, eval_gz)

    ao_loc = numpy.asarray(mol.ao_loc_nr(), dtype=numpy.int32)
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    if aosym == 's1':
        if shls_slice[:2] == shls_slice[2:4]:
            fill = getattr(libcgto, 'GTO_ft_fill_s1hermi')
        else:
            fill = getattr(libcgto, 'GTO_ft_fill_s1')
        shape = (nGv,ni,nj)
    else:
        fill = getattr(libcgto, 'GTO_ft_fill_s2')
        i0 = ao_loc[shls_slice[0]]
        i1 = ao_loc[shls_slice[1]]
        nij = i1*(i1+1)//2 - i0*(i0+1)//2
        shape = (nGv,nij)
    mat = numpy.ndarray(shape, order='F', dtype=numpy.complex128, buffer=buf)

    fn(intor, eval_gz, fill, mat.ctypes.data_as(ctypes.c_void_p),
       (ctypes.c_int*4)(*shls_slice),
       ao_loc.ctypes.data_as(ctypes.c_void_p), ctypes.c_double(0),
       GvT.ctypes.data_as(ctypes.c_void_p),
       p_invh, p_gxyzT, p_gs, ctypes.c_int(nGv),
       mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
       mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
       mol._env.ctypes.data_as(ctypes.c_void_p))

    return mat


def ft_ao(mol, Gv, shls_slice=None,
          invh=None, gxyz=None, gs=None, verbose=None):
    ''' FT transform AO
    '''
    if shls_slice is None:
        shls_slice = (0, mol.nbas)
    nGv = Gv.shape[0]
    if gxyz is None or invh is None or gs is None:
        GvT = numpy.asarray(Gv.T, order='C')
        p_gxyzT = lib.c_null_ptr()
        p_gs = (ctypes.c_int*3)(0,0,0)
        p_invh = (ctypes.c_double*1)(0)
        eval_gz = 'GTO_Gv_general'
    else:
        GvT = numpy.asarray(Gv.T, order='C')
        gxyzT = numpy.asarray(gxyz.T, order='C', dtype=numpy.int32)
        p_gxyzT = gxyzT.ctypes.data_as(ctypes.c_void_p)
        p_gs = (ctypes.c_int*3)(*gs)
# Guess what type of eval_gz to use
        if isinstance(invh, numpy.ndarray) and invh.shape == (3,3):
            p_invh = invh.ctypes.data_as(ctypes.c_void_p)
            if numpy.allclose(invh-numpy.diag(invh.diagonal()), 0):
                eval_gz = 'GTO_Gv_uniform_orth'
            else:
                eval_gz = 'GTO_Gv_uniform_nonorth'
        else:
            invh = numpy.hstack(invh)
            p_invh = invh.ctypes.data_as(ctypes.c_void_p)
            eval_gz = 'GTO_Gv_nonuniform_orth'

    fn = libcgto.GTO_ft_ovlp_mat
    intor = getattr(libcgto, 'GTO_ft_ovlp_sph')
    eval_gz = getattr(libcgto, eval_gz)
    fill = getattr(libcgto, 'GTO_ft_fill_s1')

    ghost_atm = numpy.array([[0,0,0,0,0,0]], dtype=numpy.int32)
    ghost_bas = numpy.array([[0,0,1,1,0,0,3,0]], dtype=numpy.int32)
    ghost_env = numpy.zeros(4)
    ghost_env[3] = numpy.sqrt(4*numpy.pi)  # s function spherical norm
    atm, bas, env = gto.conc_env(mol._atm, mol._bas, mol._env,
                                 ghost_atm, ghost_bas, ghost_env)
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[mol.nbas]
    ao_loc = numpy.asarray(numpy.hstack((ao_loc, [nao+1])), dtype=numpy.int32)
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    mat = numpy.zeros((nGv,ni), order='F', dtype=numpy.complex)

    shls_slice = shls_slice + (mol.nbas, mol.nbas+1)
    fn(intor, eval_gz, fill, mat.ctypes.data_as(ctypes.c_void_p),
       (ctypes.c_int*4)(*shls_slice),
       ao_loc.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_double(0),
       GvT.ctypes.data_as(ctypes.c_void_p),
       p_invh, p_gxyzT, p_gs, ctypes.c_int(nGv),
       atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(atm)),
       bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(bas)),
       env.ctypes.data_as(ctypes.c_void_p))
    return mat


if __name__ == '__main__':
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = '''C    1.3    .2       .3
                  C     .1    .1      1.1
                  '''
    mol.basis = 'ccpvdz'
    #mol.basis = {'C': [[0, (2.4, .1, .6), (1.0,.8, .4)], [1, (1.1, 1)]]}
    #mol.basis = {'C': [[0, (2.4, 1)]]}
    mol.unit = 'B'
    mol.build(0,0)

    L = 5.
    n = 20
    h = numpy.diag([L,L,L])
    invh = scipy.linalg.inv(h)
    gs = [n,n,n]
    gxrange = range(gs[0]+1)+range(-gs[0],0)
    gyrange = range(gs[1]+1)+range(-gs[1],0)
    gzrange = range(gs[2]+1)+range(-gs[2],0)
    gxyz = lib.cartesian_prod((gxrange, gyrange, gzrange))
    Gv = 2*numpy.pi * numpy.dot(gxyz, invh)

    import time
    print time.clock()
    print(numpy.linalg.norm(ft_aopair(mol, Gv, None, 's1', invh, gxyz, gs)) - 63.0239113778)
    print time.clock()
    print(numpy.linalg.norm(ft_ao(mol, Gv, None, invh, gxyz, gs))-56.8273147065)
    print time.clock()
