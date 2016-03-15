#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import time
import tempfile
import ctypes
from functools import reduce
import numpy
import h5py
import pyscf.lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.mcscf import mc_ao2mo
from pyscf.scf import dfhf
from pyscf import df


def density_fit(casscf, auxbasis=None):
    '''Generate DF-CASSCF for given CASSCF object.  It is done by overwriting
    three CASSCF member functions:
        * casscf.ao2mo which generates MO integrals
        * casscf.get_veff which generate JK from core density matrix
        * casscf.get_jk which

    Args:
        casscf : an CASSCF object

    Kwargs:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.
            The default basis 'weigend+etb' means weigend-coulomb-fit basis
            for light elements and even-tempered basis for heavy elements.

    Returns:
        An CASSCF object with a modified J, K matrix constructor which uses density
        fitting integrals to compute J and K

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = DFCASSCF(mf, 4, 4)
    -100.05994191570504
    '''
    casscf_class = casscf.__class__

    class CASSCF(casscf_class):
        def __init__(self):
            self.__dict__.update(casscf.__dict__)
            #self.grad_update_dep = 0
            if auxbasis is None:
                self.auxbasis = 'weigend+etb'
            else:
                self.auxbasis = auxbasis

# _tag_df to indicate whether the CASSCF instance is a density-fitting object.
            self._tag_df = True
            if (hasattr(self._scf, '_cderi') and
                (auxbasis is None or auxbasis == self._scf.auxbasis)):
# Use the underlying density-fitting integrals by default
                self._cderi = self._scf._cderi
                self.auxbasis = self._scf.auxbasis
            else:
                if auxbasis is None:
                    self.auxbasis = 'weigend+etb'
                else:
                    self.auxbasis = auxbasis
# By overwriting _cderi, one can provide the Cholesky integrals for "DF/RI" calculation
                self._cderi = None

            self._naoaux = None
            self._keys = self._keys.union(['auxbasis'])

        def dump_flags(self):
            casscf_class.dump_flags(self)
            logger.info(self, 'DFCASCI/DFCASSCF: density fitting for JK matrix and 2e integral transformation')

        def ao2mo(self, mo_coeff):
# using dm=[], a hacky call to dfhf.get_jk, to generate self._cderi
            dfhf.get_jk_(self, self.mol, [])
            return _ERIS(self, mo_coeff)

        def get_h2eff(self, mo_coeff=None):  # For CASCI
            dfhf.get_jk_(self, self.mol, [])
            mo = numpy.asarray(mo_coeff, order='F')
            nao, nmo = mo.shape
            buf = numpy.empty((self._naoaux,nmo*(nmo+1)//2))
            fmmm = _ao2mo._fpointer('AO2MOmmm_nr_s2_s2')
            fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
            ftrans = _ao2mo._fpointer('AO2MOtranse2_nr_s2')
            with df.load(self._cderi) as feri:
                for b0, b1 in dfhf.prange(0, self._naoaux, dfhf.BLOCKDIM):
                    eri1 = numpy.asarray(feri[b0:b1], order='C')
                    fdrv(ftrans, fmmm,
                         buf[b0:b1].ctypes.data_as(ctypes.c_void_p),
                         eri1.ctypes.data_as(ctypes.c_void_p),
                         mo.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(b1-b0), ctypes.c_int(nao),
                         ctypes.c_int(0), ctypes.c_int(nmo),
                         ctypes.c_int(0), ctypes.c_int(nmo),
                         ctypes.c_void_p(0), ctypes.c_int(0))
                    eri1 = None
            eri = pyscf.lib.dot(buf.T, buf)
            return eri

# Modify get_veff for JK matrix of core densityBecause get_h1eff calls self.get_veff to generate core JK
        def get_veff(self, mol=None, dm=None, hermi=1):
            return dfhf.get_jk_(self, mol, dm, hermi=hermi)

# We don't modify self._scf because it changes self.h1eff function.
# We only need approximate jk for self.update_jk_in_ah
        def get_jk(self, mol, dm, hermi=1):
            return dfhf.get_jk_(self, mol, dm, hermi=hermi)

    return CASSCF()


def approx_hessian(casscf, auxbasis=None):
    '''Approximate the orbital hessian with density fitting integrals
    
    Note this function has no effects if the input casscf object is DF-CASSCF.
    It only modifies the orbital hessian of normal CASSCF object.

    Args:
        casscf : an CASSCF object

    Kwargs:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.
            The default basis 'weigend+etb' means weigend-coulomb-fit basis
            for light elements and even-tempered basis for heavy elements.

    Returns:
        A CASSCF object with approximated JK contraction for orbital hessian

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = mcscf.approx_hessian(mcscf.CASSCF(mf, 4, 4))
    -100.06458716530391
    '''
    casscf_class = casscf.__class__

    if 'CASCI' in str(casscf_class):
        return casscf  # because CASCI does not need orbital optimization

    if hasattr(casscf, '_tag_df') and casscf._tag_df:
        return casscf

    class CASSCF(casscf_class):
        def __init__(self):
            self.__dict__.update(casscf.__dict__)
            #self.grad_update_dep = 0
# _tag_df to indicate whether the CASSCF instance is a density-fitting object.
            self._tag_df = False
            if (hasattr(self._scf, '_cderi') and
                (auxbasis is None or auxbasis == self._scf.auxbasis)):
                self._cderi = self._scf._cderi
                self.auxbasis = self._scf.auxbasis
            else: # what if self._cderi is not None?
                if auxbasis is None:
                    self.auxbasis = 'weigend+etb' #?
                else:
                    self.auxbasis = auxbasis
                self._cderi = None

            self._naoaux = None
            self._keys = self._keys.union(['auxbasis'])

        def dump_flags(self):
            casscf_class.dump_flags(self)
            logger.info(self, 'CASSCF: density fitting for orbital hessian')

        def ao2mo(self, mo_coeff):
            log = logger.Logger(self.stdout, self.verbose)
# the exact integral transformation
            eris = casscf_class.ao2mo(self, mo_coeff)

# using dm=[], a hacky call to dfhf.get_jk, to generate self._cderi
            dfhf.get_jk_(self, self.mol, [])

# Add the approximate diagonal term for orbital hessian
            t0 = (time.clock(), time.time())
            mo = numpy.asarray(mo_coeff, order='F')
            nao, nmo = mo.shape
            ncore = self.ncore
            eris.j_pc = numpy.zeros((nmo,ncore))
            k_cp = numpy.zeros((ncore,nmo))
            fmmm = _ao2mo._fpointer('AO2MOmmm_nr_s2_iltj')
            fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
            ftrans = _ao2mo._fpointer('AO2MOtranse2_nr_s2')
            bufs1 = numpy.empty((dfhf.BLOCKDIM,nmo,nmo))
            t1 = t0
            with df.load(self._cderi) as feri:
                for b0, b1 in dfhf.prange(0, self._naoaux, dfhf.BLOCKDIM):
                    eri1 = numpy.asarray(feri[b0:b1], order='C')
                    buf = bufs1[:b1-b0]
                    if log.verbose >= logger.DEBUG1:
                        t1 = log.timer('load buf %d:%d'%(b0,b1), *t1)
                    fdrv(ftrans, fmmm,
                         buf.ctypes.data_as(ctypes.c_void_p),
                         eri1.ctypes.data_as(ctypes.c_void_p),
                         mo.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(b1-b0), ctypes.c_int(nao),
                         ctypes.c_int(0), ctypes.c_int(nmo),
                         ctypes.c_int(0), ctypes.c_int(nmo),
                         ctypes.c_void_p(0), ctypes.c_int(0))
                    if log.verbose >= logger.DEBUG1:
                        t1 = log.timer('transform [%d:%d]'%(b0,b1), *t1)
                    bufd = numpy.einsum('kii->ki', buf)
                    eris.j_pc += numpy.einsum('ki,kj->ij', bufd, bufd[:,:ncore])
                    k_cp += numpy.einsum('kij,kij->ij', buf[:,:ncore], buf[:,:ncore])
                    if log.verbose >= logger.DEBUG1:
                        t1 = log.timer('j_pc and k_pc', *t1)
                    eri1 = None
            eris.k_pc = k_cp.T.copy()
            log.timer('ao2mo density fit part', *t0)
            return eris

        def get_jk(self, mol, dm, hermi=1):
            return dfhf.get_jk_(self, mol, dm, hermi=hermi)

    return CASSCF()


class _ERIS(object):
    def __init__(self, casscf, mo):
        import gc
        gc.collect()
        log = logger.Logger(casscf.stdout, casscf.verbose)

        mol = casscf.mol
        nao, nmo = mo.shape
        ncore = casscf.ncore
        ncas = casscf.ncas
        nocc = ncore + ncas
        naoaux = casscf._naoaux

        mem_incore, mem_outcore, mem_basic = _mem_usage(ncore, ncas, nmo)
        mem_now = pyscf.lib.current_memory()[0]
        max_memory = max(3000, casscf.max_memory*.9-mem_now)
        if max_memory < mem_basic:
            log.warn('Calculation needs %d MB memory, over CASSCF.max_memory (%d MB) limit',
                     (mem_basic+mem_now)/.9, casscf.max_memory)

        t0 = (time.clock(), time.time())
        self._tmpfile = tempfile.NamedTemporaryFile()
        self.feri = h5py.File(self._tmpfile.name, 'w')
        self.ppaa = self.feri.create_dataset('ppaa', (nmo,nmo,ncas,ncas), 'f8')
        self.papa = self.feri.create_dataset('papa', (nmo,ncas,nmo,ncas), 'f8')
        self.j_pc = numpy.zeros((nmo,ncore))
        k_cp = numpy.zeros((ncore,nmo))

        mo = numpy.asarray(mo, order='F')
        _tmpfile1 = tempfile.NamedTemporaryFile()
        fxpp = h5py.File(_tmpfile1.name)
        bufpa = numpy.empty((naoaux,nmo,ncas))
        bufs1 = numpy.empty((dfhf.BLOCKDIM,nmo,nmo))
        fmmm = _ao2mo._fpointer('AO2MOmmm_nr_s2_iltj')
        fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
        ftrans = _ao2mo._fpointer('AO2MOtranse2_nr_s2')
        t2 = t1 = t0
        fxpp_keys = []
        with df.load(casscf._cderi) as feri:
            for b0, b1 in dfhf.prange(0, naoaux, dfhf.BLOCKDIM):
                eri1 = numpy.asarray(feri[b0:b1], order='C')
                if log.verbose >= logger.DEBUG1:
                    t2 = log.timer('load buf %d:%d'%(b0,b1), *t2)
                bufpp = bufs1[:b1-b0]
                fdrv(ftrans, fmmm,
                     bufpp.ctypes.data_as(ctypes.c_void_p),
                     eri1.ctypes.data_as(ctypes.c_void_p),
                     mo.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(b1-b0), ctypes.c_int(nao),
                     ctypes.c_int(0), ctypes.c_int(nmo),
                     ctypes.c_int(0), ctypes.c_int(nmo),
                     ctypes.c_void_p(0), ctypes.c_int(0))
                fxpp_keys.append([str(b0), b0, b1])
                fxpp[str(b0)] = bufpp.transpose(1,2,0)
                bufpa[b0:b1] = bufpp[:,:,ncore:nocc]
                bufd = numpy.einsum('kii->ki', bufpp)
                self.j_pc += numpy.einsum('ki,kj->ij', bufd, bufd[:,:ncore])
                k_cp += numpy.einsum('kij,kij->ij', bufpp[:,:ncore], bufpp[:,:ncore])
                if log.verbose >= logger.DEBUG1:
                    t1 = log.timer('j_pc and k_pc', *t1)
                eri1 = None
        self.k_pc = k_cp.T.copy()
        bufs1 = bufpp = None
        t1 = log.timer('density fitting ao2mo pass1', *t0)

        mem_now = pyscf.lib.current_memory()[0]
        nblk = int(max(8, min(nmo, ((max_memory-mem_now)*1e6/8-bufpa.size)/(ncas**2*nmo))))
        bufs1 = numpy.empty((nblk,ncas,nmo,ncas))
        dgemm = pyscf.lib.numpy_helper._dgemm
        for p0, p1 in prange(0, nmo, nblk):
            #tmp = numpy.dot(bufpa[:,p0:p1].reshape(naoaux,-1).T,
            #                bufpa.reshape(naoaux,-1))
            tmp = bufs1[:p1-p0]
            dgemm('T', 'N', (p1-p0)*ncas, nmo*ncas, naoaux,
                  bufpa.reshape(naoaux,-1), bufpa.reshape(naoaux,-1),
                  tmp.reshape(-1,nmo*ncas), 1, 0, p0*ncas, 0, 0)
            self.papa[p0:p1] = tmp.reshape(p1-p0,ncas,nmo,ncas)
        bufaa = bufpa[:,ncore:nocc,:].copy().reshape(-1,ncas**2)
        bufs1 = bufpa = None
        t1 = log.timer('density fitting papa pass2', *t1)

        mem_now = pyscf.lib.current_memory()[0]
        nblk = int(max(8, min(nmo, (max_memory-mem_now)*1e6/8/(nmo*naoaux+ncas**2*nmo))))
        bufs1 = numpy.empty((nblk,nmo,naoaux))
        bufs2 = numpy.empty((nblk,nmo,ncas,ncas))
        for p0, p1 in prange(0, nmo, nblk):
            nrow = p1 - p0
            buf = bufs1[:nrow]
            tmp = bufs2[:nrow].reshape(-1,ncas**2)
            col0 = 0
            for key, col0, col1 in fxpp_keys:
                buf[:nrow,:,col0:col1] = fxpp[key][p0:p1]
            pyscf.lib.dot(buf.reshape(-1,naoaux), bufaa, 1, tmp)
            self.ppaa[p0:p1] = tmp.reshape(p1-p0,nmo,ncas,ncas)
        bufs1 = bufs2 = buf = None
        t1 = log.timer('density fitting ppaa pass2', *t1)

        fxpp.close()
        self.feri.flush()
        dm_core = numpy.dot(mo[:,:ncore], mo[:,:ncore].T)
        vj, vk = casscf.get_jk(mol, dm_core)
        self.vhf_c = reduce(numpy.dot, (mo.T, vj*2-vk, mo))
        t0 = log.timer('density fitting ao2mo', *t0)

    def __del__(self):
        if hasattr(self, 'feri'):
            self.feri.close()
            self.feri = None
            self._tmpfile = None

def _mem_usage(ncore, ncas, nmo):
    nvir = nmo - ncore
    outcore = basic = ncas**2*nmo**2*2 * 8/1e6
    incore = outcore + (ncore+ncas)*nmo**3*4/1e6
    if outcore > 10000:
        sys.stderr.write('Be careful with the virtual memorty address space `ulimit -v`\n')
    return incore, outcore, basic

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf
    from pyscf.mcscf import addons

    mol = gto.Mole()
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = density_fit(mcscf.CASSCF(m, 6, 4))
    mc.verbose = 4
    mo = addons.sort_mo(mc, m.mo_coeff, (3,4,6,7,8,9), 1)
    emc = mc.kernel(mo)[0]
    print(ehf, emc, emc-ehf)
    #-76.0267656731 -76.0873922924 -0.0606266193028
    print(emc - -76.0873923174, emc - -76.0926176464)

    mc = density_fit(mcscf.CASSCF(m, 6, (3,1)))
    mc.verbose = 4
    emc = mc.mc2step(mo)[0]
    print(emc - -75.7155632535814)

    mf = scf.density_fit(m)
    mf.kernel()
    #mc = density_fit(mcscf.CASSCF(mf, 6, 4))
    #mc = mcscf.CASSCF(mf, 6, 4)
    mc = mcscf.DFCASSCF(mf, 6, 4)
    mc.verbose = 4
    mo = addons.sort_mo(mc, mc.mo_coeff, (3,4,6,7,8,9), 1)
    emc = mc.kernel(mo)[0]
    print(emc, 'ref = -76.0917567904955', emc - -76.0917567904955)

    #mc = density_fit(mcscf.CASCI(mf, 6, 4))
    #mc = mcscf.CASCI(mf, 6, 4)
    mc = mcscf.DFCASCI(mf, 6, 4)
    mo = addons.sort_mo(mc, mc.mo_coeff, (3,4,6,7,8,9), 1)
    emc = mc.kernel(mo)[0]
    print(emc, 'ref = -76.0476686258461', emc - -76.0476686258461)
