#!/usr/bin/env python

import time
import ctypes
import tempfile
from functools import reduce
import numpy
import scipy.linalg
import pyscf.lib
from pyscf.lib import logger


def density_fit(mf, auxbasis='weigend'):
    '''For the given SCF object, update the J, K matrix constructor with
    corresponding density fitting integrals.

    Args:
        mf : an SCF object

    Kwargs:
        auxbasis : str

    Returns:
        An SCF object with a modified J, K matrix constructor which uses density
        fitting integrals to compute J and K

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.density_fit(scf.RHF(mol))
    >>> mf.scf()
    -100.005306000435510

    >>> mol.symmetry = 1
    >>> mol.build(0, 0)
    >>> mf = scf.density_fit(scf.UHF(mol))
    >>> mf.scf()
    -100.005306000435510
    '''

    import pyscf.scf
    class HF(mf.__class__):
        def __init__(self):
            self.__dict__.update(mf.__dict__)
            self.auxbasis = auxbasis
            self._cderi = None
            self.direct_scf = False
            self._keys = self._keys.union(['auxbasis'])

        def get_jk(self, mol=None, dm=None, hermi=1):
            if mol is None: mol = self.mol
            if dm is None: dm = self.make_rdm1()
            if isinstance(self, pyscf.scf.dhf.UHF):
                return r_get_jk_(self, mol, dm, hermi)
            else:
                return get_jk_(self, mol, dm, hermi)
    return HF()

def density_fit_(mf, auxbasis='weigend'):
    '''Replace J K constructor of HF object.  See the usage of :func:`density_fit`
    '''
    import pyscf.scf
    def get_jk(mol, dm, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = mf.make_rdm1()
        if isinstance(mf, pyscf.scf.dhf.UHF):
            return r_get_jk_(mf, mol, dm, hermi)
        else:
            return get_jk_(mf, mol, dm, hermi)
    mf.get_jk = get_jk
    mf.auxbasis = auxbasis
    mf._cderi = None
    mf.direct_scf = False
    mf._keys = mf._keys.union(['auxbasis'])
    return mf


OCCDROP = 1e-12
BLOCKDIM = 120
_call_count = 0
def get_jk_(mf, mol, dms, hermi=1):
    from pyscf import df
    from pyscf.ao2mo import _ao2mo
    t0 = (time.clock(), time.time())
    log = logger.Logger(mf.stdout, mf.verbose)
    if not hasattr(mf, '_cderi') or mf._cderi is None:
        nao = mol.nao_nr()
        auxmol = df.incore.format_aux_basis(mol, mf.auxbasis)
        mf._naoaux = auxmol.nao_nr()
        if nao*(nao+1)/2*mf._naoaux*8 < mf.max_memory*1e6:
            mf._cderi = df.incore.cholesky_eri(mol, auxbasis=mf.auxbasis,
                                               verbose=log)
        else:
            mf._cderi_file = tempfile.NamedTemporaryFile()
            mf._cderi = mf._cderi_file.name
            mf._cderi = df.outcore.cholesky_eri(mol, mf._cderi,
                                                auxbasis=mf.auxbasis,
                                                verbose=log)

    if len(dms) == 0:
        return [], []

    global _call_count
    _call_count += 1

    cderi = mf._cderi
    nao = mol.nao_nr()
    fmmm = df.incore._fpointer('RIhalfmmm_nr_s2_bra')
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo._fpointer('AO2MOtranse2_nr_s2kl')

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        dms = [dms]
        nset = 1
    else:
        nset = len(dms)
    vj = numpy.zeros((nset,nao,nao))
    vk = numpy.zeros((nset,nao,nao))

    #:vj = reduce(numpy.dot, (cderi.reshape(-1,nao*nao), dm.reshape(-1),
    #:                        cderi.reshape(-1,nao*nao))).reshape(nao,nao)
    if hermi == 1:
# I cannot assume dm is positive definite because it might be the density
# matrix difference when the mf.direct_scf flag is set.
        dmtril = []
        cpos = []
        cneg = []
        for k, dm in enumerate(dms):
            dmtril.append(pyscf.lib.pack_tril(dm+dm.T))
            for i in range(nao):
                dmtril[k][i*(i+1)//2+i] *= .5

            e, c = scipy.linalg.eigh(dm)
            pos = e > OCCDROP
            neg = e < -OCCDROP

            #:vk = numpy.einsum('pij,jk->kpi', cderi, c[:,abs(e)>OCCDROP])
            #:vk = numpy.einsum('kpi,kpj->ij', vk, vk)
            cpos.append(numpy.asarray(numpy.einsum('ij,j->ij', c[:,pos],
                                                   numpy.sqrt(e[pos])), order='F'))
            cneg.append(numpy.asarray(numpy.einsum('ij,j->ij', c[:,neg],
                                                   numpy.sqrt(-e[neg])),order='F'))
        if mf.verbose >= logger.DEBUG1:
            t1 = log.timer('Initialization', *t0)
        with df.load(cderi) as feri:
            for b0, b1 in prange(0, mf._naoaux, BLOCKDIM):
                eri1 = numpy.array(feri[b0:b1], copy=False)
                if mf.verbose >= logger.DEBUG1:
                    t1 = log.timer('load buf %d:%d'%(b0,b1), *t1)
                for k in range(nset):
                    buf = reduce(numpy.dot, (eri1, dmtril[k], eri1))
                    vj[k] += pyscf.lib.unpack_tril(buf, hermi)
                    if cpos[k].shape[1] > 0:
                        buf = numpy.empty(((b1-b0)*cpos[k].shape[1],nao))
                        fdrv(ftrans, fmmm,
                             buf.ctypes.data_as(ctypes.c_void_p),
                             eri1.ctypes.data_as(ctypes.c_void_p),
                             cpos[k].ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(b1-b0), ctypes.c_int(nao),
                             ctypes.c_int(0), ctypes.c_int(cpos[k].shape[1]),
                             ctypes.c_int(0), ctypes.c_int(0))
                        vk[k] += pyscf.lib.dot(buf.T, buf)
                    if cneg[k].shape[1] > 0:
                        buf = numpy.empty(((b1-b0)*cneg[k].shape[1],nao))
                        fdrv(ftrans, fmmm,
                             buf.ctypes.data_as(ctypes.c_void_p),
                             eri1.ctypes.data_as(ctypes.c_void_p),
                             cneg[k].ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(b1-b0), ctypes.c_int(nao),
                             ctypes.c_int(0), ctypes.c_int(cneg[k].shape[1]),
                             ctypes.c_int(0), ctypes.c_int(0))
                        vk[k] -= pyscf.lib.dot(buf.T, buf)
                if mf.verbose >= logger.DEBUG1:
                    t1 = log.timer('jk', *t1)
    else:
        #:vk = numpy.einsum('pij,jk->pki', cderi, dm)
        #:vk = numpy.einsum('pki,pkj->ij', cderi, vk)
        fcopy = df.incore._fpointer('RImmm_nr_s2_copy')
        rargs = (ctypes.c_int(nao),
                 ctypes.c_int(0), ctypes.c_int(nao),
                 ctypes.c_int(0), ctypes.c_int(0))
        dms = [numpy.asarray(dm, order='F') for dm in dms]
        if mf.verbose >= logger.DEBUG1:
            t1 = log.timer('Initialization', *t0)
        with df.load(cderi) as feri:
            for b0, b1 in prange(0, mf._naoaux, BLOCKDIM):
                eri1 = numpy.array(feri[b0:b1], copy=False)
                if mf.verbose >= logger.DEBUG1:
                    t1 = log.timer('load buf %d:%d'%(b0,b1), *t1)
                for k in range(nset):
                    buf = numpy.empty((b1-b0,nao,nao))
                    fdrv(ftrans, fmmm,
                         buf.ctypes.data_as(ctypes.c_void_p),
                         eri1.ctypes.data_as(ctypes.c_void_p),
                         dms[k].ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(b1-b0), *rargs)
                    rho = numpy.einsum('kii->k', buf)
                    vj[k] += pyscf.lib.unpack_tril(numpy.dot(rho, eri1), 1)

                    buf1 = numpy.empty((b1-b0,nao,nao))
                    fdrv(ftrans, fcopy,
                         buf1.ctypes.data_as(ctypes.c_void_p),
                         eri1.ctypes.data_as(ctypes.c_void_p),
                         dms[k].ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(b1-b0), *rargs)
                    vk[k] += pyscf.lib.dot(buf.reshape(-1,nao).T,
                                           buf1.reshape(-1,nao))
                if mf.verbose >= logger.DEBUG1:
                    t1 = log.timer('jk', *t1)

    if len(dms) == 1:
        vj = vj[0]
        vk = vk[0]
    logger.timer(mf, 'vj and vk', *t0)
    return vj, vk


def r_get_jk_(mf, mol, dms, hermi=1):
    '''Relativistic density fitting JK'''
    from pyscf import df
    from pyscf.ao2mo import _ao2mo
    t0 = (time.clock(), time.time())
    if not hasattr(mf, '_cderi') or mf._cderi is None:
        log = logger.Logger(mf.stdout, mf.verbose)
        n2c = mol.nao_2c()
        auxmol = df.incore.format_aux_basis(mol, mf.auxbasis)
        mf._naoaux = auxmol.nao_nr()
        if n2c*(n2c+1)/2*mf._naoaux*16 < mf.max_memory*1e6:
            mf._cderi = df.r_incore.cholesky_eri(mol, auxbasis=mf.auxbasis,
                                                 aosym='s2', verbose=log)
        else:
            assert(0)
            mf._cderi_file = tempfile.NamedTemporaryFile()
            mf._cderi = mf._cderi_file.name
            mf._cderi = df.r_outcore.cholesky_eri(mol, mf._cderi,
                                                  auxbasis=mf.auxbasis,
                                                  verbose=log)
    n2c = mol.nao_2c()
    c1 = .5 / mol.light_speed

    def fjk(dm):
        fmmm = df.r_incore._fpointer('RIhalfmmm_r_s2_bra_noconj')
        fdrv = _ao2mo.libao2mo.AO2MOr_e2_drv
        ftrans = df.r_incore._fpointer('RItranse2_r_s2')
        vj = numpy.zeros_like(dm)
        vk = numpy.zeros_like(dm)
        fcopy = df.incore._fpointer('RImmm_r_s2_transpose')
        rargs = (ctypes.c_int(n2c),
                 ctypes.c_int(0), ctypes.c_int(n2c),
                 ctypes.c_int(0), ctypes.c_int(0))
        dmll = numpy.asarray(dm[:n2c,:n2c], order='C')
        dmls = numpy.asarray(dm[:n2c,n2c:], order='C') * c1
        dmsl = numpy.asarray(dm[n2c:,:n2c], order='C') * c1
        dmss = numpy.asarray(dm[n2c:,n2c:], order='C') * c1**2
        with df.load(mf._cderi[0]) as ferill:
            with df.load(mf._cderi[1]) as feriss: # python2.6 not support multiple with
                for b0, b1 in prange(0, mf._naoaux, BLOCKDIM):
                    erill = numpy.array(ferill[b0:b1], copy=False)
                    eriss = numpy.array(feriss[b0:b1], copy=False)
                    buf = numpy.empty((b1-b0,n2c,n2c), dtype=numpy.complex)
                    buf1 = numpy.empty((b1-b0,n2c,n2c), dtype=numpy.complex)

                    fdrv(ftrans, fmmm,
                         buf.ctypes.data_as(ctypes.c_void_p),
                         erill.ctypes.data_as(ctypes.c_void_p),
                         dmll.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(b1-b0), *rargs) # buf == (P|LL)
                    rho = numpy.einsum('kii->k', buf)

                    fdrv(ftrans, fcopy,
                         buf1.ctypes.data_as(ctypes.c_void_p),
                         erill.ctypes.data_as(ctypes.c_void_p),
                         dmll.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(b1-b0), *rargs) # buf1 == (P|LL)
                    vk[:n2c,:n2c] += numpy.dot(buf1.reshape(-1,n2c).T,
                                               buf.reshape(-1,n2c))

                    fdrv(ftrans, fmmm,
                         buf.ctypes.data_as(ctypes.c_void_p),
                         eriss.ctypes.data_as(ctypes.c_void_p),
                         dmls.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(b1-b0), *rargs) # buf == (P|LS)
                    vk[:n2c,n2c:] += numpy.dot(buf1.reshape(-1,n2c).T,
                                               buf.reshape(-1,n2c)) * c1

                    fdrv(ftrans, fmmm,
                         buf.ctypes.data_as(ctypes.c_void_p),
                         eriss.ctypes.data_as(ctypes.c_void_p),
                         dmss.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(b1-b0), *rargs) # buf == (P|SS)
                    rho += numpy.einsum('kii->k', buf)
                    vj[:n2c,:n2c] += pyscf.lib.unpack_tril(numpy.dot(rho, erill), 1)
                    vj[n2c:,n2c:] += pyscf.lib.unpack_tril(numpy.dot(rho, eriss), 1) * c1**2

                    fdrv(ftrans, fcopy,
                         buf1.ctypes.data_as(ctypes.c_void_p),
                         eriss.ctypes.data_as(ctypes.c_void_p),
                         dmss.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(b1-b0), *rargs) # buf == (P|SS)
                    vk[n2c:,n2c:] += numpy.dot(buf1.reshape(-1,n2c).T,
                                               buf.reshape(-1,n2c)) * c1**2

                    if not hermi == 1:
                        fdrv(ftrans, fmmm,
                             buf.ctypes.data_as(ctypes.c_void_p),
                             erill.ctypes.data_as(ctypes.c_void_p),
                             dmsl.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(b1-b0), *rargs) # buf == (P|SL)
                        vk[n2c:,:n2c] += numpy.dot(buf1.reshape(-1,n2c).T,
                                                   buf.reshape(-1,n2c)) * c1
        if hermi == 1:
            vk[n2c:,:n2c] = vk[:n2c,n2c:].T.conj()
        return vj, vk

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        vj, vk = fjk(dms)
    else:
        vjk = [fjk(dm) for dm in dms]
        vj = numpy.array([x[0] for x in vjk])
        vk = numpy.array([x[1] for x in vjk])
    logger.timer(mf, 'vj and vk', *t0)
    return vj, vk


def prange(start, end, step):
    if _call_count % 2 == 1:
        for i in reversed(range(start, end, step)):
            yield i, min(i+step, end)
    else:
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
    method.max_memory = 0
    energy = method.scf()
    print(energy, -76.0259362997)

    method = density_fit(pyscf.scf.DHF(mol))
    energy = method.scf()
    print(energy, -76.0807386852) # normal DHF energy is -76.0815679438127

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
    print(energy, -75.6310072359)
