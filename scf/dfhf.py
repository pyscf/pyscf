#!/usr/bin/env python

import time
import ctypes
from functools import reduce
import numpy
import scipy.linalg
import pyscf.lib
import pyscf.lib.logger as log


def density_fit(mf):
    class HF(mf.__class__):
        def __init__(self):
            self.__dict__.update(mf.__dict__)
            self.auxbasis = 'weigend'
            self._cderi = None
            self._ovlp = None
            self.direct_scf = False
            self._keys = self._keys.union(['auxbasis', '_ovlp', '_cderi'])

        def get_jk(self, mol, dm, hermi=1):
            return get_jk_(self, mol, dm, hermi)
    return HF()

def get_jk_(mf, mol, dm, hermi=1):
    from pyscf import df
    t0 = (time.clock(), time.time())
    if not hasattr(mf, '_cderi') or mf._cderi is None:
        mf._ovlp = mf.get_ovlp(mol)
        mf._cderi = df.incore.cholesky_eri(mol, auxbasis=mf.auxbasis,
                                           verbose=mf.verbose)
    vj = _make_j(mf, dm, hermi)
    vk = _make_k(mf, dm, hermi)
    log.timer(mf, 'vj and vk', *t0)
    return vj, vk

def _make_j(mf, dms, hermi):
    cderi = mf._cderi
    def fvj(dm):
        nao = dm.shape[0]
        #:vj = reduce(numpy.dot, (cderi.reshape(-1,nao*nao), dm.reshape(-1),
        #:                        cderi.reshape(-1,nao*nao))).reshape(nao,nao)
        dmtril = pyscf.lib.pack_tril(dm+dm.T)
        for i in range(nao):
            dmtril[i*(i+1)//2+i] *= .5
        vj = reduce(numpy.dot, (cderi, dmtril, cderi))
        vj = pyscf.lib.unpack_tril(vj, 1)
        return vj
    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        return fvj(dms)
    else:
        return numpy.array([fvj(dm) for dm in dms])

OCCDROP = 1e-12
BLOCKDIM = 160
def _make_k(mf, dms, hermi):
    from pyscf import df
    from pyscf.ao2mo import _ao2mo
    s = mf._ovlp
    cderi = mf._cderi
    def fvk(dm):
        nao = dm.shape[0]
        naoaux = cderi.shape[0]
        fmmm = df.incore._fpointer('RIhalfmmm_nr_s2_bra')
        fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
        ftrans = _ao2mo._fpointer('AO2MOtranse2_nr_s2kl')

        if hermi == 1:
# I cannot assume dm is positive definite because it is the density matrix
# difference when the mf.direct_scf flag is set.
            e, c = scipy.linalg.eigh(dm, s, type=2)
            pos = e > OCCDROP
            neg = e < -OCCDROP
            vk = numpy.zeros_like(dm)
            if sum(pos)+sum(neg) > 0:
                #:vk = numpy.einsum('pij,jk->kpi', cderi, c[:,abs(e)>OCCDROP])
                #:vk = numpy.einsum('kpi,kpj->ij', vk, vk)
                cpos = numpy.einsum('ij,j->ij', c[:,pos], numpy.sqrt(e[pos]))
                cpos = numpy.asfortranarray(cpos)
                cneg = numpy.einsum('ij,j->ij', c[:,neg], numpy.sqrt(-e[neg]))
                cneg = numpy.asfortranarray(cneg)
                cposargs = (ctypes.c_int(nao),
                            ctypes.c_int(0), ctypes.c_int(cpos.shape[1]),
                            ctypes.c_int(0), ctypes.c_int(0))
                cnegargs = (ctypes.c_int(nao),
                            ctypes.c_int(0), ctypes.c_int(cneg.shape[1]),
                            ctypes.c_int(0), ctypes.c_int(0))
                for b0, b1 in prange(0, naoaux, BLOCKDIM):
                    eri1 = numpy.ascontiguousarray(cderi[b0:b1])
                    if cpos.shape[1] > 0:
                        buf = numpy.empty(((b1-b0)*cpos.shape[1],nao))
                        fdrv(ftrans, fmmm,
                             buf.ctypes.data_as(ctypes.c_void_p),
                             eri1.ctypes.data_as(ctypes.c_void_p),
                             cpos.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(b1-b0), *cposargs)
                        vk += numpy.dot(buf.T, buf)
                    if cneg.shape[1] > 0:
                        buf = numpy.empty(((b1-b0)*cneg.shape[1],nao))
                        fdrv(ftrans, fmmm,
                             buf.ctypes.data_as(ctypes.c_void_p),
                             eri1.ctypes.data_as(ctypes.c_void_p),
                             cneg.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(b1-b0), *cnegargs)
                        vk -= numpy.dot(buf.T, buf)
        else:
            #:vk = numpy.einsum('pij,jk->pki', cderi, dm)
            #:vk = numpy.einsum('pki,pkj->ij', cderi, vk)
            fcopy = df.incore._fpointer('RImmm_nr_s2_copy')
            rargs = (ctypes.c_int(nao),
                     ctypes.c_int(0), ctypes.c_int(nao),
                     ctypes.c_int(0), ctypes.c_int(0))
            vk = numpy.zeros_like(dm)
            dm = numpy.asfortranarray(dm)
            for b0, b1 in prange(0, naoaux, BLOCKDIM):
                buf = numpy.empty((b1-b0,nao,nao))
                eri1 = numpy.ascontiguousarray(cderi[b0:b1])
                fdrv(ftrans, fmmm,
                     buf.ctypes.data_as(ctypes.c_void_p),
                     eri1.ctypes.data_as(ctypes.c_void_p),
                     dm.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(b1-b0), *rargs)
                buf1 = numpy.empty((b1-b0,nao,nao))
                fdrv(ftrans, fcopy,
                     buf1.ctypes.data_as(ctypes.c_void_p),
                     eri1.ctypes.data_as(ctypes.c_void_p),
                     dm.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(b1-b0), *rargs)
                vk += numpy.dot(buf.reshape(-1,nao).T, buf1.reshape(-1,nao))
        return vk
    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        return fvk(dms)
    else:
        return numpy.array([fvk(dm) for dm in dms])

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)


if __name__ == '__main__':
    import pyscf.gto
    import pyscf.scf
    mol = pyscf.gto.Mole()
    mol.build(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)] ],
        basis = 'ccpvdz',
    )

    method = density_fit(pyscf.scf.RHF(mol))
    energy = method.scf()
    print(energy) # -76.0259362997

    mol.build(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)] ],
        basis = 'ccpvdz',
        spin = 1,
        charge = 1,
    )

    method = density_fit(pyscf.scf.UHF(mol))
    energy = method.scf()
    print(energy) # -75.6310072359
