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
              invh=None, gxyz=None, gs=None,
              kpti_kptj=numpy.zeros((2,3)), verbose=None):
    kpti, kptj = kpti_kptj
    val = _ft_aopair_kpts(cell, Gv, shls_slice, aosym, invh, gxyz, gs,
                          kptj-kpti, kptj.reshape(1,3))
    return val[0]

# NOTE buffer out must be initialized to 0
def _ft_aopair_kpts(cell, Gv, shls_slice=None, aosym='s1',
                    invh=None, gxyz=None, gs=None,
                    kpt=numpy.zeros(3), kptjs=numpy.zeros((2,3)), out=None):
    ''' FT transform AO pair
    \int i(r) j(r) exp(-ikr) dr^3
    for all  kpt = kptj - kpti.  The return list holds the AO pair array
    corresponding to the kpoints given by kptjs
    '''
    kpt = numpy.reshape(kpt, 3)
    kptjs = numpy.asarray(kptjs, order='C').reshape(-1,3)
    nGv = Gv.shape[0]
    Gv = Gv + kpt # kptis = kptjs - kpt

    if (gxyz is None or invh is None or gs is None or (abs(kpt).sum() > 1e-9)):
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
            if abs(invh-numpy.diag(invh.diagonal())).sum() < 1e-8:
                eval_gz = 'GTO_Gv_uniform_orth'
            else:
                eval_gz = 'GTO_Gv_uniform_nonorth'
        else:
            invh = numpy.hstack(invh)
            p_invh = invh.ctypes.data_as(ctypes.c_void_p)
            eval_gz = 'GTO_Gv_nonuniform_orth'

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
    if aosym == 's1hermi': # Symmetry for Gamma point
        assert(abs(kpt).sum() < 1e-9 and abs(kptjs).sum() < 1e-9)
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
        p_invh, p_gxyzT, p_gs, ctypes.c_int(nGv),
        atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.natm*2),
        bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(cell.nbas*2),
        env.ctypes.data_as(ctypes.c_void_p))

    if aosym == 's1hermi':
        for mat in out:
            for i in range(1,ni):
                mat[:,:i,i] = mat[:,i,:i]
    return out


def ft_ao(mol, Gv, shls_slice=None,
          invh=None, gxyz=None, gs=None, kpt=numpy.zeros(3), verbose=None):
    if abs(kpt).sum() < 1e-9:
        return mol_ft_ao(mol, Gv, shls_slice, invh, gxyz, gs, verbose)
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
            print i, j, numpy.linalg.norm(ao2ref - ao2[:,i,j])

    aoG = ft_ao(cell, cell.Gv)
    for i in range(nao):
        aoref = tools.fft(aoR[:,i], cell.gs) * cell.vol/ngs
        print i, numpy.linalg.norm(aoref - aoG[:,i])

