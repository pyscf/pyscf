#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Analytic Fourier transformation AO-pair value for PBC
'''

import ctypes
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf.gto.ft_ao import ft_ao as mol_ft_ao

libpbc = lib.load_library('libpbc')

#
# \int mu*nu*exp(-ik*r) dr
#
def ft_aopair(cell, Gv, shls_slice=None, aosym='s1',
              b=None, gxyz=None, Gvbase=None,
              kpti_kptj=numpy.zeros((2,1,3)), q=None, verbose=None):
    ''' FT transform AO pair
    \int exp(-i(G+q)r) i(r) j(r) exp(-ikr) dr^3
    '''
    kpti, kptj = kpti_kptj
    if q is None:
        q = kptj - kpti
    val = _ft_aopair_kpts(cell, Gv, shls_slice, aosym, b, gxyz, Gvbase,
                          q, kptj.reshape(1,3))
    return val[0]

# NOTE buffer out must be initialized to 0
# gxyz is the index for Gvbase
def _ft_aopair_kpts(cell, Gv, shls_slice=None, aosym='s1',
                    b=None, gxyz=None, Gvbase=None,
                    q=numpy.zeros(3), kptjs=numpy.zeros((1,3)),
                    out=None):
    ''' FT transform AO pair
    \int exp(-i(G+q)r) i(r) j(r) exp(-ikr) dr^3
    The return list holds the AO pair array
    corresponding to the kpoints given by kptjs
    '''
    q = numpy.reshape(q, 3)
    kptjs = numpy.asarray(kptjs, order='C').reshape(-1,3)
    nGv = Gv.shape[0]
    GvT = numpy.asarray(Gv.T, order='C')
    GvT += q.reshape(-1,1)

    if (gxyz is None or b is None or Gvbase is None or (abs(q).sum() > 1e-9)
# backward compatibility for pyscf-1.2, in which the argument Gvbase is gs
        or (Gvbase is not None and isinstance(Gvbase[0], (int, numpy.integer)))):
        p_gxyzT = lib.c_null_ptr()
        p_gs = (ctypes.c_int*3)(0,0,0)
        p_b = (ctypes.c_double*1)(0)
        eval_gz = 'GTO_Gv_general'
    else:
        gxyzT = numpy.asarray(gxyz.T, order='C', dtype=numpy.int32)
        p_gxyzT = gxyzT.ctypes.data_as(ctypes.c_void_p)
        b = numpy.hstack((b.ravel(), q) + Gvbase)
        p_b = b.ctypes.data_as(ctypes.c_void_p)
        p_gs = (ctypes.c_int*3)(*[len(x) for x in Gvbase])
        eval_gz = 'GTO_Gv_cubic'

    drv = libpbc.PBC_ft_latsum_kpts
    intor = getattr(libpbc, 'GTO_ft_ovlp_sph')
    eval_gz = getattr(libpbc, eval_gz)

    # make copy of atm,bas,env because they are modified in the lattice sum
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
    shape = (nGv, ni, nj)
    fill = getattr(libpbc, 'PBC_ft_fill_'+aosym)
# Theoretically, hermitian symmetry can be also found for kpti == kptj:
#       f_ji(G) = \int f_ji exp(-iGr) = \int f_ij^* exp(-iGr) = [f_ij(-G)]^*
# The hermi operation needs reordering the axis-0.  It is inefficient.
    if aosym == 's1hermi': # Symmetry for Gamma point
        assert(abs(q).sum() < 1e-9 and abs(kptjs).sum() < 1e-9)
    elif aosym == 's2':
        i0 = ao_loc[shls_slice[0]]
        i1 = ao_loc[shls_slice[1]]
        nij = i1*(i1+1)//2 - i0*(i0+1)//2
        shape = (nGv, nij)

    if out is None:
        out = [numpy.zeros(shape, order='F', dtype=numpy.complex128)
               for k in range(len(kptjs))]
    else:
        out = [numpy.ndarray(shape, order='F', dtype=numpy.complex128,
                             buffer=out[k]) for k in range(len(kptjs))]
    out_ptrs = (ctypes.c_void_p*len(out))(
            *[x.ctypes.data_as(ctypes.c_void_p) for x in out])

    xyz = numpy.asarray(cell.atom_coords(), order='C')
    ptr_coord = numpy.asarray(atm[cell.natm:,gto.PTR_COORD],
                              dtype=numpy.int32, order='C')
    Ls = cell.get_lattice_Ls()
    exp_Lk = numpy.einsum('ik,jk->ij', Ls, kptjs)
    exp_Lk = numpy.exp(1j * numpy.asarray(exp_Lk, order='C'))
    drv(intor, eval_gz, fill, out_ptrs, xyz.ctypes.data_as(ctypes.c_void_p),
        ptr_coord.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(xyz)),
        Ls.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(Ls)),
        exp_Lk.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(kptjs)),
        (ctypes.c_int*4)(*shls_slice),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        GvT.ctypes.data_as(ctypes.c_void_p),
        p_b, p_gxyzT, p_gs, ctypes.c_int(nGv),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm*2),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas*2),
        env.ctypes.data_as(ctypes.c_void_p))

    if aosym == 's1hermi':
        for mat in out:
            for i in range(1,ni):
                mat[:,:i,i] = mat[:,i,:i]
    return out


def ft_ao(mol, Gv, shls_slice=None, b=None,
          gxyz=None, Gvbase=None, kpt=numpy.zeros(3), verbose=None):
    if abs(kpt).sum() < 1e-9:
        return mol_ft_ao(mol, Gv, shls_slice, b, gxyz, Gvbase, verbose)
    else:
        kG = Gv + kpt
        return mol_ft_ao(mol, kG, shls_slice, None, None, None, verbose)

if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    import pyscf.dft.numint
    from pyscf.pbc import tools

    L = 5.
    n = 10
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
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
            print(i, j, numpy.linalg.norm(ao2ref - ao2[:,i,j]))

    aoG = ft_ao(cell, cell.Gv)
    for i in range(nao):
        aoref = tools.fft(aoR[:,i], cell.gs) * cell.vol/ngs
        print(i, numpy.linalg.norm(aoref - aoG[:,i]))

