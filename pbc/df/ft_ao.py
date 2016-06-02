#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Analytic Fourier transformation AO-pair value for PBC
'''

import ctypes
import _ctypes
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
import pyscf.df.ft_ao
from pyscf.df.ft_ao import libpbc

#
# \int mu*nu*exp(-ik*r) dr
#
def ft_aopair(cell, Gv, shls_slice=None, hermi=False,
              invh=None, gxyz=None, gs=None,
              kpti_kptj=numpy.zeros((2,3)), verbose=None):
    ''' FT transform AO pair
    \int i(r) j(r) exp(-ikr) dr^3
    '''
    nGv = Gv.shape[0]
    kpti, kptj = kpti_kptj
    Gv = Gv + (kptj - kpti)

    if (gxyz is None or invh is None or gs is None or
        (abs(kpti-kptj).sum() > 1e-9)):
        GvT = numpy.asarray(Gv.T, order='C')
        p_gxyzT = lib.c_null_ptr()
        p_gs = (ctypes.c_int*3)(0,0,0)
        p_invh = (ctypes.c_double*1)(0)
        eval_gz = 'PBC_Gv_general'
    else:
        GvT = numpy.asarray(Gv.T, order='C')
        gxyzT = numpy.asarray(gxyz.T, order='C', dtype=numpy.int32)
        p_gxyzT = gxyzT.ctypes.data_as(ctypes.c_void_p)
        p_gs = (ctypes.c_int*3)(*gs)
# Guess what type of eval_gz to use
        if isinstance(invh, numpy.ndarray) and invh.shape == (3,3):
            p_invh = invh.ctypes.data_as(ctypes.c_void_p)
            if abs(invh-numpy.diag(invh.diagonal())).sum() < 1e-8:
                eval_gz = 'PBC_Gv_uniform_orth'
            else:
                eval_gz = 'PBC_Gv_uniform_nonorth'
        else:
            invh = numpy.hstack(invh)
            p_invh = invh.ctypes.data_as(ctypes.c_void_p)
            eval_gz = 'PBC_Gv_nonuniform_orth'

    fn = libpbc.PBC_ft_ovlp_mat
    intor = ctypes.c_void_p(_ctypes.dlsym(libpbc._handle, 'PBC_ft_ovlp_sph'))
    eval_gz = ctypes.c_void_p(_ctypes.dlsym(libpbc._handle, eval_gz))

    atm, bas, env = gto.conc_env(cell._atm, cell._bas, cell._env,
                                 cell._atm, cell._bas, cell._env)
    ao_loc = cell.ao_loc_nr()
    nao = ao_loc[cell.nbas]
    ao_loc = numpy.asarray(numpy.hstack((ao_loc[:-1], ao_loc+nao)),
                           dtype=numpy.int32)
    if shls_slice is None:
        shls_slice = (0, cell.nbas, cell.nbas, cell.nbas*2)
    else:
        shls_slice = (shls_slice[0], shls_slice[1],
                      cell.nbas+shls_slice[2], cell.nbas+shls_slice[3])
    ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
    nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
    mat = numpy.zeros((nGv,ni,nj), order='F', dtype=numpy.complex)
    if hermi:
        assert(ni == nj)

    xyz = cell.atom_coords()
    cell1_ptr_coord = atm[cell.natm:,gto.PTR_COORD]
    Ls = cell.get_lattice_Ls(cell.nimgs)
    for l, L1 in enumerate(Ls):
        env[cell1_ptr_coord+0] = xyz[:,0] + L1[0]
        env[cell1_ptr_coord+1] = xyz[:,1] + L1[1]
        env[cell1_ptr_coord+2] = xyz[:,2] + L1[2]
        fn(intor, eval_gz, mat.ctypes.data_as(ctypes.c_void_p),
           (ctypes.c_int*4)(*shls_slice),
           ao_loc.ctypes.data_as(ctypes.c_void_p),
           ctypes.c_int(hermi), ctypes.c_double(numpy.dot(kptj, L1)),
           GvT.ctypes.data_as(ctypes.c_void_p),
           p_invh, p_gxyzT, p_gs, ctypes.c_int(nGv),
           atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm*2),
           bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas*2),
           env.ctypes.data_as(ctypes.c_void_p))

    if hermi and abs(kpti).sum() < 1e-9 and abs(kptj).sum() < 1e-9:
# Theoretically, hermitian symmetry can be also found for ki == kj:
#       f_ji(G) = \int f_ji exp(-iGr) = \int f_ij^* exp(-iGr) = [f_ij(-G)]^*
# The hermi operation needs reordering the axis-0.  It is not obvious for an
# arbitrary Gv vector.  Thus do not consider the hermi symmetry here.
        for i in range(1,ni):
            mat[:,:i,i] = mat[:,i,:i]
    return mat


def ft_ao(mol, Gv, shls_slice=None,
          invh=None, gxyz=None, gs=None, kpt=numpy.zeros(3), verbose=None):
    if abs(kpt).sum() < 1e-9:
        return pyscf.df.ft_ao.ft_ao(mol, Gv, shls_slice, invh, gxyz, gs, verbose)
    else:
        kG = Gv + kpt
        return pyscf.df.ft_ao.ft_ao(mol, kG, shls_slice, None, None, None, verbose)

if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    import pyscf.dft.numint
    from pyscf.pbc import tools

    L = 5.
    n = 10
    cell = pgto.Cell()
    cell.h = numpy.diag([L,L,L])
    cell.gs = numpy.array([n,n,n])

    cell.atom = '''C    1.3    .2       .3
                   C     .1    .1      1.1
                   '''
    cell.basis = 'ccpvdz'
    #cell.basis = {'C': [[0, (2.4, .1, .6), (1.0,.8, .4)], [1, (1.1, 1)]]}
    #cell.basis = {'C': [[0, (2.4, 1)]]}
    cell.unit = 'B'
    #cell.verbose = 4
    cell.build(0,0)
    #cell.nimgs = (2,2,2)

    ao2 = ft_aopair(cell, cell.Gv)
    nao = cell.nao_nr()
    coords = pyscf.pbc.dft.gen_grid.gen_uniform_grids(cell)
    aoR = pyscf.pbc.dft.numint.eval_ao(cell, coords)
    aoR2 = numpy.einsum('ki,kj->kij', aoR.conj(), aoR)
    ngs = aoR.shape[0]

    for i in range(nao):
        for j in range(nao):
            ao2ref = tools.fft(aoR2[:,i,j], cell.gs) * cell.vol/ngs
            print i, j, numpy.linalg.norm(ao2ref - ao2[:,i,j])

    aoG = ft_ao(cell, cell.Gv)
    for i in range(nao):
        aoref = tools.fft(aoR[:,i], cell.gs) * cell.vol/ngs
        print i, numpy.linalg.norm(aoref - aoG[:,i])

