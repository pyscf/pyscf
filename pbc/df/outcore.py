#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.df import incore
from pyscf.gto import PTR_COORD
from pyscf.ao2mo.outcore import balance_segs

def aux_e2(cell, auxcell, erifile, intor='cint3c2e_sph', aosym='s1', comp=1,
           kptij_lst=None, dataname='eri_mo', max_memory=2000, verbose=0):
    '''3-center AO integrals (ij|L) with double lattice sum:
    \sum_{lm} (i[l]j[m]|L[0]), where L is the auxiliary basis.
    On diks, the integrals are stored as (kptij_idx, naux, nao_pair)

    Args:
        kptij_lst : (*,2,3) array
            A list of (kpti, kptj)
    '''
    if comp > 1:
        raise NotImplementedError('comp = %d' % comp)
    if h5py.is_hdf5(erifile):
        feri = h5py.File(erifile)
        if dataname in feri:
            del(feri[dataname])
        if dataname+'-kptij' in feri:
            del(feri[dataname+'-kptij'])
    else:
        feri = h5py.File(erifile, 'w')

    if kptij_lst is None:
        kptij_lst = numpy.zeros((1,2,3))
    feri[dataname+'-kptij'] = kptij_lst
    nkptij = len(kptij_lst)

    # sum over largest number of images in either cell or auxcell
    rcut = max(cell.rcut, auxcell.rcut)
    Ls = cell.get_lattice_Ls(rcut=rcut)
    logger.debug1(cell, "pbc.df.outcore.rcut %s", rcut)
    logger.debug3(cell, "Ls = %s", Ls)

    nao = cell.nao_nr()
    #naux = auxcell.nao_nr('ssc' in intor)
    naux = auxcell.nao_nr()
    aosym_s2 = numpy.zeros(nkptij, dtype=bool)
    for k, kptij in enumerate(kptij_lst):
        key = '%s/%d' % (dataname, k)
        if abs(kptij).sum() < 1e-9:  # gamma_point:
            dtype = 'f8'
        else:
            dtype = 'c16'
        aosym_s2[k] = abs(kptij[0]-kptij[1]).sum() < 1e-9
        if aosym_s2[k]:
            nao_pair = nao * (nao+1) // 2
        else:
            nao_pair = nao * nao
        if comp == 1:
            shape = (naux,nao_pair)
        else:
            shape = (comp,naux,nao_pair)
        chunks = (min(256,naux), min(256,nao_pair))  # 512 KB
        feri.create_dataset(key, shape, dtype, chunks=chunks)
    if naux == 0:
        feri.close()
        return erifile

    aux_loc = auxcell.ao_loc_nr('ssc' in intor)
    buflen = max(8, int(max_memory*1e6/16/(nkptij*nao**2*comp)))
    auxranges = balance_segs(aux_loc[1:]-aux_loc[:-1], buflen)
    buflen = max([x[2] for x in auxranges])
    buf = [numpy.zeros(nao*nao*buflen*comp, dtype=numpy.complex128)
           for k in range(nkptij)]
    ints = incore._wrap_int3c(cell, auxcell, intor, comp, Ls, buf)
    atm, bas, env = ints._envs[:3]

    xyz = numpy.asarray(cell.atom_coords(), order='C')
    ptr_coordL = atm[:cell.natm,PTR_COORD]
    ptr_coordL = numpy.vstack((ptr_coordL,ptr_coordL+1,ptr_coordL+2)).T.copy('C')
    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]

    if numpy.all(aosym_s2):
        def ccsum_or_reorder(Lpq):
            tmp = numpy.asarray(Lpq.transpose(0,2,1).conj(), order='C')
            tmp += Lpq
            return tmp
    else:
        def ccsum_or_reorder(Lpq):
            return numpy.asarray(Lpq, order='C')

    naux0 = 0
    for istep, auxrange in enumerate(auxranges):
        sh0, sh1, nrow = auxrange
        c_shls_slice = (ctypes.c_int*6)(0, cell.nbas, cell.nbas, cell.nbas*2,
                                        cell.nbas*2+sh0, cell.nbas*2+sh1)
        if numpy.all(aosym_s2):
            for l, L1 in enumerate(Ls):
                env[ptr_coordL] = xyz + L1
                e = numpy.dot(Ls[:l+1]-L1, kptj.T)  # Lattice sum over half of the images {1..l}
                exp_Lk = numpy.exp(1j * numpy.asarray(e, order='C'))
                exp_Lk[l] = .5
                ints(exp_Lk, c_shls_slice)
        else:
            for l, L1 in enumerate(Ls):
                env[ptr_coordL] = xyz + L1
                e = numpy.dot(Ls, kptj.T) - numpy.dot(L1, kpti.T)
                exp_Lk = numpy.exp(1j * numpy.asarray(e, order='C'))
                ints(exp_Lk, c_shls_slice)

        for k, kptij in enumerate(kptij_lst):
            h5dat = feri['%s/%d'%(dataname,k)]
            # transpose 3201 as (comp,L,i,j)
            mat = numpy.ndarray((nao,nao,nrow,comp), order='F',
                                dtype=numpy.complex128, buffer=buf[k])
            for icomp, vi in enumerate(mat.transpose(3,2,0,1)):
                v = ccsum_or_reorder(vi)
                if abs(kptij).sum() < 1e-9:  # gamma_point:
                    v = v.real
                if aosym_s2[k]:
                    v = lib.pack_tril(v)
                else:
                    v = v.reshape(nrow,-1)
                if comp == 1:
                    h5dat[naux0:naux0+nrow] = v
                else:
                    h5dat[icomp,naux0:naux0+nrow] = v
            mat[:] = 0
        naux0 += nrow

    feri.close()
    return erifile


