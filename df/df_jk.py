#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import time
import ctypes
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.df import _ri
from pyscf import df


OCCDROP = 1e-12

def density_fit(mf, auxbasis='weigend+etb', with_df=None):
    '''For the given SCF object, update the J, K matrix constructor with
    corresponding density fitting integrals.

    Args:
        mf : an SCF object

    Kwargs:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.
            The default basis 'weigend+etb' means weigend-coulomb-fit basis
            for light elements and even-tempered basis for heavy elements.

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

    from pyscf.scf import dhf
    mf_class = mf.__class__
    if mf_class.__doc__ is None:
        doc = ''
    else:
        doc = mf_class.__doc__

    if with_df is None:
        if isinstance(mf, dhf.UHF):
            with_df = df.DF4C(mf.mol)
        else:
            with_df = df.DF(mf.mol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    class DFHF(mf_class):
        __doc__ = doc + \
        '''
        Attributes for density-fitting SCF:
            auxbasis : str or basis dict
                Same format to the input attribute mol.basis.
                The default basis 'weigend+etb' means weigend-coulomb-fit basis
                for light elements and even-tempered basis for heavy elements.
        '''
        def __init__(self):
            self.__dict__.update(mf.__dict__)
            self.auxbasis = auxbasis
            self.direct_scf = False
            self.with_df = with_df
            self._keys = self._keys.union(['auxbasis', 'with_df'])

        def get_jk(self, mol=None, dm=None, hermi=1):
            if self.with_df:
                if mol is None: mol = self.mol
                if dm is None: dm = self.make_rdm1()
                return self.with_df.get_jk(dm, hermi)
            else:
                return mf_class.get_jk(self, mol, dm, hermi)

        def get_j(self, mol=None, dm=None, hermi=1):
            if self.with_df:
                if mol is None: mol = self.mol
                if dm is None: dm = self.make_rdm1()
                return self.with_df.get_jk(dm, hermi, with_k=False)[0]
            else:
                return mf_class.get_j(self, mol, dm, hermi)

        def get_k(self, mol=None, dm=None, hermi=1):
            if self.with_df:
                if mol is None: mol = self.mol
                if dm is None: dm = self.make_rdm1()
                return self.with_df.get_jk(dm, hermi, with_j=False)[1]
            else:
                return mf_class.get_k(self, mol, dm, hermi)

# _cderi accesser for pyscf 1.0, 1.1 compatibility
        @property
        def _cderi(self):
            return self.with_df._cderi
        @_cderi.setter
        def _cderi(self, x):
            self.with_df._cderi = x

        @property
        def _tag_df(self):
            sys.stderr.write('WARN: Deprecated attribute ._tag_df will be removed in future release. '
                             'It is replaced by attribute .with_df\n')
            if self.with_df:
                return True
            else:
                return False

    return DFHF()


def get_jk(dfobj, dms, hermi=1, vhfopt=None, with_j=True, with_k=True):
    t0 = t1 = (time.clock(), time.time())
    log = logger.Logger(dfobj.stdout, dfobj.verbose)

    if len(dms) == 0:
        return [], []
    elif isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        nset = 1
        dms = [dms]
    else:
        nset = len(dms)
    nao = dms[0].shape[0]

    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2

    vj = numpy.zeros((nset,nao,nao))
    vk = numpy.zeros((nset,nao,nao))
    null = lib.c_null_ptr()

    #:vj = reduce(numpy.dot, (cderi.reshape(-1,nao*nao), dm.reshape(-1),
    #:                        cderi.reshape(-1,nao*nao))).reshape(nao,nao)
    if hermi == 1: # and numpy.einsum('ij,ij->', dm, ovlp) > 0.1
# I cannot assume dm is positive definite because it might be the density
# matrix difference when the mf.direct_scf flag is set.
        dmtril = []
        cpos = []
        cneg = []
        for k, dm in enumerate(dms):
            if with_j:
                dmtril.append(lib.pack_tril(dm+dm.T))
                i = numpy.arange(nao)
                dmtril[k][i*(i+1)//2+i] *= .5

            if with_k:
                e, c = scipy.linalg.eigh(dm)
                pos = e > OCCDROP
                neg = e < -OCCDROP

                #:vk = numpy.einsum('pij,jk->kpi', cderi, c[:,abs(e)>OCCDROP])
                #:vk = numpy.einsum('kpi,kpj->ij', vk, vk)
                tmp = numpy.einsum('ij,j->ij', c[:,pos], numpy.sqrt(e[pos]))
                cpos.append(numpy.asarray(tmp, order='F'))
                tmp = numpy.einsum('ij,j->ij', c[:,neg], numpy.sqrt(-e[neg]))
                cneg.append(numpy.asarray(tmp, order='F'))
        buf = numpy.empty((dfobj.blockdim*nao,nao))
        for eri1 in dfobj.loop():
            naux, nao_pair = eri1.shape
            assert(nao_pair == nao*(nao+1)//2)
            for k in range(nset):
                if with_j:
                    buf1 = reduce(numpy.dot, (eri1, dmtril[k], eri1))
                    vj[k] += lib.unpack_tril(buf1, hermi)
                if with_k and cpos[k].shape[1] > 0:
                    buf1 = buf[:naux*cpos[k].shape[1]]
                    fdrv(ftrans, fmmm,
                         buf1.ctypes.data_as(ctypes.c_void_p),
                         eri1.ctypes.data_as(ctypes.c_void_p),
                         cpos[k].ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(naux), ctypes.c_int(nao),
                         (ctypes.c_int*4)(0, cpos[k].shape[1], 0, 0),
                         null, ctypes.c_int(0))
                    vk[k] += lib.dot(buf1.T, buf1)
                if with_k and cneg[k].shape[1] > 0:
                    buf1 = buf[:naux*cneg[k].shape[1]]
                    fdrv(ftrans, fmmm,
                         buf1.ctypes.data_as(ctypes.c_void_p),
                         eri1.ctypes.data_as(ctypes.c_void_p),
                         cneg[k].ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(naux), ctypes.c_int(nao),
                         (ctypes.c_int*4)(0, cneg[k].shape[1], 0, 0),
                         null, ctypes.c_int(0))
                    vk[k] -= lib.dot(buf1.T, buf1)
            t1 = log.timer_debug1('jk', *t1)
    else:
        #:vk = numpy.einsum('pij,jk->pki', cderi, dm)
        #:vk = numpy.einsum('pki,pkj->ij', cderi, vk)
        rargs = (ctypes.c_int(nao), (ctypes.c_int*4)(0, nao, 0, 0),
                 null, ctypes.c_int(0))
        dms = [numpy.asarray(dm, order='F') for dm in dms]
        buf = numpy.empty((2,dfobj.blockdim,nao,nao))
        for eri1 in dfobj.loop():
            naux, nao_pair = eri1.shape
            for k in range(nset):
                buf1 = buf[0,:naux]
                fdrv(ftrans, fmmm,
                     buf1.ctypes.data_as(ctypes.c_void_p),
                     eri1.ctypes.data_as(ctypes.c_void_p),
                     dms[k].ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(naux), *rargs)
                rho = numpy.einsum('kii->k', buf1)
                vj[k] += lib.unpack_tril(numpy.dot(rho, eri1), 1)

                if with_k:
                    buf2 = lib.unpack_tril(eri1, out=buf[1])
                    vk[k] += lib.dot(buf1.reshape(-1,nao).T,
                                     buf2.reshape(-1,nao))
            t1 = log.timer_debug1('jk', *t1)

    if len(dms) == 1:
        vj = vj[0]
        vk = vk[0]
    logger.timer(dfobj, 'vj and vk', *t0)
    return vj, vk


def r_get_jk(dfobj, dms, hermi=1):
    '''Relativistic density fitting JK'''
    t0 = t1 = (time.clock(), time.time())
    mol = dfobj.mol
    c1 = .5 / lib.param.LIGHT_SPEED
    tao = mol.tmap()
    ao_loc = mol.ao_loc_2c()
    n2c = ao_loc[-1]

    def fjk(dm):
        fmmm = _ri.libri.RIhalfmmm_r_s2_bra_noconj
        fdrv = _ao2mo.libao2mo.AO2MOr_e2_drv
        ftrans = _ri.libri.RItranse2_r_s2
        vj = numpy.zeros_like(dm)
        vk = numpy.zeros_like(dm)
        fcopy = _ri.libri.RImmm_r_s2_transpose
        rargs = (ctypes.c_int(n2c), (ctypes.c_int*4)(0, n2c, 0, 0),
                 tao.ctypes.data_as(ctypes.c_void_p),
                 ao_loc.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(mol.nbas))
        dmll = numpy.asarray(dm[:n2c,:n2c], order='C')
        dmls = numpy.asarray(dm[:n2c,n2c:], order='C') * c1
        dmsl = numpy.asarray(dm[n2c:,:n2c], order='C') * c1
        dmss = numpy.asarray(dm[n2c:,n2c:], order='C') * c1**2
        for erill, eriss in dfobj.loop():
            naux, nao_pair = erill.shape
            buf = numpy.empty((naux,n2c,n2c), dtype=numpy.complex)
            buf1 = numpy.empty((naux,n2c,n2c), dtype=numpy.complex)

            fdrv(ftrans, fmmm,
                 buf.ctypes.data_as(ctypes.c_void_p),
                 erill.ctypes.data_as(ctypes.c_void_p),
                 dmll.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), *rargs) # buf == (P|LL)
            rho = numpy.einsum('kii->k', buf)

            fdrv(ftrans, fcopy,
                 buf1.ctypes.data_as(ctypes.c_void_p),
                 erill.ctypes.data_as(ctypes.c_void_p),
                 dmll.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), *rargs) # buf1 == (P|LL)
            vk[:n2c,:n2c] += numpy.dot(buf1.reshape(-1,n2c).T,
                                       buf.reshape(-1,n2c))

            fdrv(ftrans, fmmm,
                 buf.ctypes.data_as(ctypes.c_void_p),
                 eriss.ctypes.data_as(ctypes.c_void_p),
                 dmls.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), *rargs) # buf == (P|LS)
            vk[:n2c,n2c:] += numpy.dot(buf1.reshape(-1,n2c).T,
                                       buf.reshape(-1,n2c)) * c1

            fdrv(ftrans, fmmm,
                 buf.ctypes.data_as(ctypes.c_void_p),
                 eriss.ctypes.data_as(ctypes.c_void_p),
                 dmss.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), *rargs) # buf == (P|SS)
            rho += numpy.einsum('kii->k', buf)
            vj[:n2c,:n2c] += lib.unpack_tril(numpy.dot(rho, erill), 1)
            vj[n2c:,n2c:] += lib.unpack_tril(numpy.dot(rho, eriss), 1) * c1**2

            fdrv(ftrans, fcopy,
                 buf1.ctypes.data_as(ctypes.c_void_p),
                 eriss.ctypes.data_as(ctypes.c_void_p),
                 dmss.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), *rargs) # buf == (P|SS)
            vk[n2c:,n2c:] += numpy.dot(buf1.reshape(-1,n2c).T,
                                       buf.reshape(-1,n2c)) * c1**2

            if not hermi == 1:
                fdrv(ftrans, fmmm,
                     buf.ctypes.data_as(ctypes.c_void_p),
                     erill.ctypes.data_as(ctypes.c_void_p),
                     dmsl.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(naux), *rargs) # buf == (P|SL)
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
    logger.timer(dfobj, 'vj and vk', *t0)
    return vj, vk


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
