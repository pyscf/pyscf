#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import numpy
import h5py
import pyscf.gto
import pyscf.ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import _vhf
from pyscf.pbc.df import incore

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

    if nkptij == 1:
        if numpy.linalg.norm(kptij_lst[0,0]-kptij_lst[0,1]) < 1e-9:
            aosym = 's2ij'
        else:
            aosym = 's1'
        mat = incore.aux_e2(cell, auxcell, intor, aosym, comp, kptij_lst[0])
        if comp == 1:
            feri[dataname+'/0'] = mat.T
        else:
            # (kpt,comp,i,j,L) -> (kpt,comp,L,i,j)
            feri[dataname+'/0'] = mat.transpose(0,2,1)
        feri.close()
        return erifile

    # sum over largest number of images in either cell or auxcell
    nimgs = numpy.max((cell.nimgs, auxcell.nimgs), axis=0)
    Ls = cell.get_lattice_Ls(nimgs)
    logger.debug1(cell, "Images %s", nimgs)
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
            feri.create_dataset(key, (naux,nao_pair), dtype)
        else:
            feri.create_dataset(key, (comp,naux,nao_pair), dtype)

    aux_loc = auxcell.ao_loc_nr('ssc' in intor)
    buflen = max(8, int(max_memory*1e6/16/(nkptij*nao**2*comp)))
    auxranges = pyscf.ao2mo.outcore.group_segs_filling_block(aux_loc[1:]-aux_loc[:-1], buflen)
    buflen = max([x[2] for x in auxranges])
    buf = numpy.empty((nao,nao,buflen,comp), order='F')
    ints = incore._wrap_int3c(cell, auxcell, intor, comp, buf)
    atm, bas, env, ao_loc = ints._envs[:4]

    xyz = cell.atom_coords().copy('C')
    ptr_coordL = atm[         :cell.natm  ,pyscf.gto.PTR_COORD]
    ptr_coordR = atm[cell.natm:cell.natm*2,pyscf.gto.PTR_COORD]
    ptr_coordL = numpy.vstack((ptr_coordL,ptr_coordL+1,ptr_coordL+2)).T.copy('C')
    ptr_coordR = numpy.vstack((ptr_coordR,ptr_coordR+1,ptr_coordR+2)).T.copy('C')

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
        mat = [0] * nkptij
        if numpy.all(aosym_s2):
            for l, L1 in enumerate(Ls):
                env[ptr_coordL] = xyz + L1
                for m in range(l):
                    L2 = Ls[m]
                    env[ptr_coordR] = xyz + L2
                    j3c = numpy.ndarray((nao,nao,nrow,comp), order='F',
                                        buffer=ints(c_shls_slice))
                    for k, (kpti, kptj) in enumerate(kptij_lst):
                        e = numpy.dot(L2-L1, kptj)
                        mat[k] += j3c * numpy.exp(1j*e)

                env[ptr_coordR] = xyz + L1
                j3c = numpy.ndarray((nao,nao,nrow,comp), order='F',
                                    buffer=ints(c_shls_slice))
                for k, (kpti, kptj) in enumerate(kptij_lst):
                    mat[k] += j3c * (.5+0j)
        else:
            for l, L1 in enumerate(Ls):
                env[ptr_coordL] = xyz + L1
                for m, L2 in enumerate(Ls):
                    env[ptr_coordR] = xyz + L2
                    j3c = numpy.ndarray((nao,nao,nrow,comp), order='F',
                                        buffer=ints(c_shls_slice))
                    for k, (kpti, kptj) in enumerate(kptij_lst):
                        e = numpy.dot(L2, kptj) - numpy.dot(L1, kpti)
                        mat[k] += j3c * numpy.exp(1j*e)

        for k, kptij in enumerate(kptij_lst):
            h5dat = feri['%s/%d'%(dataname,k)]
            # transpose 3201 as (comp,L,i,j)
            for icomp, vi in enumerate(mat[k].transpose(3,2,0,1)):
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
        naux0 += nrow

    feri.close()
    return erifile


