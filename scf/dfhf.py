#!/usr/bin/env python

import time
import ctypes
from functools import reduce
import numpy
import scipy.linalg
import pyscf.lib
import pyscf.lib.logger as log
from pyscf.scf import hf
from pyscf.scf import uhf


class RHF(hf.RHF):
    def __init__(self, *args, **kwargs):
        hf.RHF.__init__(self, *args, **kwargs)
        self.auxbasis = 'weigend'
        self._cderi = None
        self._keys = self._keys.union(['auxbasis', '_cderi'])

    def build_(self, mol):
        if mol is None:
            mol = self.mol
        mol.check_sanity(self)

    def get_veff(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        import pyscf.df
        t0 = (time.clock(), time.time())
        if not hasattr(self, '_cderi') or self._cderi is None:
            self._ovlp = self.get_ovlp(mol)
            self._cderi = pyscf.df.incore.cholesky_eri(mol, auxbasis=self.auxbasis,
                                                       verbose=self.verbose)
        vj = _make_j(self, dm, hermi)
        vk = _make_k(self, dm, hermi)
        vhf = vj - vk * .5
        log.timer(self, 'vj and vk', *t0)
        return vhf

class UHF(uhf.UHF):
    def __init__(self, *args, **kwargs):
        uhf.UHF.__init__(self, *args, **kwargs)
        self.auxbasis = 'weigend'
        self._cderi = None
        self._keys = self._keys.union(['auxbasis', '_cderi'])

    def get_veff(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        return _veff_uhf_(self, mol, dm, dm_last, vhf_last, hermi)


class ROHF(hf.ROHF):
    def __init__(self, *args, **kwargs):
        hf.ROHF.__init__(self, *args, **kwargs)
        self.auxbasis = 'weigend'
        self._cderi = None
        self._keys = self._keys.union(['auxbasis', '_cderi'])

    def get_veff(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        return _veff_uhf_(self, mol, dm, dm_last, vhf_last, hermi)


def _veff_uhf_(mf, mol, dm, dm_last=0, vhf_last=0, hermi=1):
    t0 = (time.clock(), time.time())
    if not hasattr(mf, '_cderi') or mf._cderi is None:
        mf._ovlp = mf.get_ovlp(mol)
        mf._cderi = pyscf.df.incore.cholesky_eri(mol, auxbasis=mf.auxbasis,
                                                 verbose=mf.verbose)
    naoaux,nao = mf._cderi.shape[:2]
    if len(dm) == 2:
        vj = _make_j(mf, dm[0]+dm[1], hermi)
        vk = _make_k(mf, dm, hermi)
        vhf = numpy.array((vj-vk[0], vj-vk[1]))
    else:
        dm = numpy.array(dm, copy=False)
        nd = len(dm) // 2
        vj = _make_j(mf, dm[:nd]+dm[nd:], hermi)
        vk = _make_k(mf, dm, hermi)
        vhf = numpy.array((vj-vk[:nd], vj-vk[nd:]))
    log.timer(mf, 'vj and vk', *t0)
    return vhf

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
        vj = fvj(dms)
    else:
        vj = numpy.array([fvj(dm) for dm in dms])
    return vj

OCCDROP = 1e-8
BLOCKDIM = 160
def _make_k(mf, dms, hermi):
    import pyscf.df
    from pyscf.ao2mo import _ao2mo
    s = mf._ovlp
    cderi = mf._cderi
    def fvk(dm):
        nao = dm.shape[0]
        naoaux = cderi.shape[0]
        fmmm = pyscf.df.incore._fpointer('RIhalfmmm_nr_s2_bra')
        fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
        ftrans = _ao2mo._fpointer('AO2MOtranse2_nr_s2kl')

        if hermi == 1:
            e, c = scipy.linalg.eigh(dm, s, type=2)
            c = numpy.einsum('ij,j->ij', c[:,e>OCCDROP],
                             numpy.sqrt(e[e>OCCDROP]))
            c = numpy.asfortranarray(c)
            #:vk = numpy.einsum('pij,jk->kpi', cderi, c)
            #:vk = numpy.dot(vk.reshape(-1,nao).T, vk.reshape(-1,nao))
            rargs = (ctypes.c_int(nao),
                     ctypes.c_int(0), ctypes.c_int(c.shape[1]),
                     ctypes.c_int(0), ctypes.c_int(0))
            vk = numpy.zeros_like(dm)
            for b0, b1 in prange(0, naoaux, BLOCKDIM):
                buf = numpy.empty((b1-b0,c.shape[1],nao))
                eri1 = numpy.ascontiguousarray(cderi[b0:b1])
                fdrv(ftrans, fmmm,
                     buf.ctypes.data_as(ctypes.c_void_p),
                     eri1.ctypes.data_as(ctypes.c_void_p),
                     c.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(b1-b0), *rargs)
                vk += numpy.dot(buf.reshape(-1,nao).T, buf.reshape(-1,nao))
        else:
            #:vk = numpy.einsum('pij,jk->pki', cderi, dm)
            #:vk = numpy.einsum('pki,pkj->ij', cderi, vk)
            fcopy = pyscf.df.incore._fpointer('RImmm_nr_s2_copy')
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
        nao = dms.shape[0]
        vk = fvk(dms)
    else:
        nao = dms[0].shape[0]
        vk = numpy.array([fvk(dm) for dm in dms])
    return vk

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)


if __name__ == '__main__':
    import pyscf.gto
    mol = pyscf.gto.Mole()
    mol.build(
        verbose = 0,
        atom = [["O" , (0. , 0.     , 0.)],
                [1   , (0. , -0.757 , 0.587)],
                [1   , (0. , 0.757  , 0.587)] ],
        basis = 'ccpvdz',
    )

    method = RHF(mol)
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

    method = UHF(mol)
    energy = method.scf()
    print(energy) # -75.6310072359
