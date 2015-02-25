#!/usr/bin/env python

import os, sys
import ctypes
import time
import tempfile
import numpy
import h5py
import pyscf.lib
import pyscf.lib.numpy_helper
import pyscf.ao2mo
from pyscf.ao2mo import _ao2mo

# least memory requirements:
# nmo  ncore  ncas  outcore  incore
# 200  40     16    0.8GB    3.7 GB (_eri 1.6GB intermediates 1.3G)
# 250  50     16    1.7GB    8.2 GB (_eri 3.9GB intermediates 2.6G)
# 300  60     16    3.1GB    16.8GB (_eri 8.1GB intermediates 5.6G)
# 400  80     16    8.5GB    53  GB (_eri 25.6GB intermediates 19G)
# 500  100    16    19 GB
# 600  120    16    37 GB
# 750  150    16    85 GB



def trans_e1_incore(eri_ao, mo, ncore, ncas):
    nmo = mo.shape[1]
    nocc = ncore + ncas
    eri1 = pyscf.ao2mo.incore.half_e1(eri_ao, (mo[:,:nocc],mo), compact=False)

    load_buf = lambda bufid: eri1[bufid*nmo:bufid*nmo+nmo]
    aapp, appa, Iapcv = _trans_aapp_(mo, ncore, ncas, load_buf)
    jc_pp, kc_pp, Icvcv = _trans_cvcv_(mo, ncore, ncas, load_buf)
    return jc_pp, kc_pp, aapp, appa, Iapcv, Icvcv


def trans_e1_outcore(casscf, mo, max_memory=None, ioblk_size=512, tmpdir=None,
                     verbose=0):
    time0 = (time.clock(), time.time())
    mol = casscf.mol
    log = pyscf.lib.logger.Logger(casscf.stdout, verbose)
    ncore = casscf.ncore
    ncas = casscf.ncas
    nao, nmo = mo.shape
    nao_pair = nao*(nao+1)//2
    nocc = ncore + ncas

    swapfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    pyscf.ao2mo.outcore.half_e1(mol, (mo[:,:nocc],mo), swapfile.name,
                                verbose=log, compact=False)

    fswap = h5py.File(swapfile.name, 'r')
    klaoblks = len(fswap['0'])
    def load_buf(bfn_id):
        if mol.verbose >= pyscf.lib.logger.DEBUG1:
            time1[:] = pyscf.lib.logger.timer(mol, 'between load_buf',
                                              *tuple(time1))
        buf = numpy.empty((nmo,nao_pair))
        col0 = 0
        for ic in range(klaoblks):
            dat = fswap['0/%d'%ic]
            col1 = col0 + dat.shape[1]
            buf[:nmo,col0:col1] = dat[bfn_id*nmo:(bfn_id+1)*nmo]
            col0 = col1
        if mol.verbose >= pyscf.lib.logger.DEBUG1:
            time1[:] = pyscf.lib.logger.timer(mol, 'load_buf', *tuple(time1))
        return buf
    time0 = pyscf.lib.logger.timer(mol, 'halfe1', *time0)
    time1 = [time.clock(), time.time()]
    aapp, appa, Iapcv = _trans_aapp_(mo, ncore, ncas, load_buf)
    time0 = pyscf.lib.logger.timer(mol, 'trans_aapp', *time0)
    jc_pp, kc_pp, Icvcv = _trans_cvcv_(mo, ncore, ncas, load_buf)
    time0 = pyscf.lib.logger.timer(mol, 'trans_cvcv', *time0)
    fswap.close()
    return jc_pp, kc_pp, aapp, appa, Iapcv, Icvcv


def _trans_aapp_(mo, ncore, ncas, fload):
    nmo = mo.shape[1]
    nocc = ncore + ncas
    c_nmo = ctypes.c_int(nmo)
    funpack = pyscf.lib.numpy_helper._np_helper.NPdunpack_tril

    klshape = (0, nmo, 0, nmo)

    japcv = numpy.empty((ncas,nmo,ncore,nmo-ncore))
    aapp = numpy.empty((ncas,ncas,nmo,nmo))
    appa = numpy.empty((ncas,nmo,nmo,ncas))
    ppp = numpy.empty((nmo,nmo,nmo))
    for i in range(ncas):
        buf = _ao2mo.nr_e2_(fload(ncore+i), mo, klshape, aosym='s4',mosym='s2')
        for j in range(nmo):
            funpack(c_nmo, buf[j].ctypes.data_as(ctypes.c_void_p),
                    ppp[j].ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1))
        aapp[i] = ppp[ncore:nocc]
        appa[i] = ppp[:,:,ncore:nocc]
        #japcv = apcv * 4 - acpv.transpose(0,2,1,3) - avcp.transpose(0,3,2,1)
        japcv[i] = ppp[:,:ncore,ncore:] * 4 \
                 - ppp[:ncore,:,ncore:].transpose(1,0,2) \
                 - ppp[ncore:,:ncore,:].transpose(2,1,0)
    return aapp, appa, japcv

def _trans_cvcv_(mo, ncore, ncas, fload):
    nmo = mo.shape[1]
    nocc = ncore + ncas
    c_nmo = ctypes.c_int(nmo)
    funpack = pyscf.lib.numpy_helper._np_helper.NPdunpack_tril

    jc_pp = numpy.empty((ncore,nmo,nmo))
    kc_pp = numpy.empty((ncore,nmo,nmo))
    jcvcv = numpy.zeros((ncore,nmo-ncore,ncore,nmo-ncore))
    vcp = numpy.empty((nmo-ncore,ncore,nmo))
    cpp = numpy.empty((ncore,nmo,nmo))
    for i in range(ncore):
        buf = fload(i)
        klshape = (0, ncore, 0, nmo)
        _ao2mo.nr_e2_(buf[ncore:nmo], mo, klshape,
                      aosym='s4', mosym='s1', vout=vcp)
        kc_pp[i,ncore:] = vcp[:,i]

        klshape = (0, nmo, 0, nmo)
        _ao2mo.nr_e2_(buf[:ncore], mo, klshape,
                      aosym='s4', mosym='s2', vout=buf[:ncore])
        for j in range(ncore):
            funpack(c_nmo, buf[j].ctypes.data_as(ctypes.c_void_p),
                    cpp[j].ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1))
        jc_pp[i] = cpp[i]
        kc_pp[i,:ncore] = cpp[:,i]

        #jcvcv = cvcv * 4 - cvcv.transpose(0,3,2,1) - ccvv.transpose(0,2,1,3)
        jcvcv[i] = vcp[:,:,ncore:] * 4 \
                 - vcp[:,:,ncore:].transpose(2,1,0) \
                 - cpp[:,ncore:,ncore:].transpose(1,0,2)
    return jc_pp, kc_pp, jcvcv



class _ERIS(object):
    def __init__(self, casscf, mo, method='incore'):
        mol = casscf.mol
        self.ncore = casscf.ncore
        self.ncas = casscf.ncas
        nmo = mo.shape[1]
        ncore = self.ncore
        ncas = self.ncas

        if method == 'outcore' \
           or _mem_usage(ncore, ncas, nmo)[0] + nmo**4*2/1e6 > casscf.max_memory*.9 \
           or casscf._scf._eri is None:
            self.jc_pp, self.kc_pp, \
            self.aapp, self.appa, \
            self.Iapcv, self.Icvcv = \
                    trans_e1_outcore(casscf, mo, max_memory=casscf.max_memory,
                                     verbose=casscf.verbose)
        elif method == 'incore' and casscf._scf._eri is not None:
            self.jc_pp, self.kc_pp, \
            self.aapp, self.appa, \
            self.Iapcv, self.Icvcv = \
                    trans_e1_incore(casscf._scf._eri, mo,
                                    casscf.ncore, casscf.ncas)
        else:
            raise KeyError('update ao2mo')

def _mem_usage(ncore, ncas, nmo):
    nvir = nmo - ncore
    outcore = (ncore**2*nvir**2 + ncas*nmo*ncore*nvir + ncore*nmo**2*3 +
               ncas**2*nmo**2*2 + nmo**3*2) * 8/1e6
    incore = outcore + nmo**4/1e6 + ncore*nmo**3*4/1e6
    if outcore > 10000:
        sys.stderr.write('Be careful with the virtual memorty address space `ulimit -v`\n')
    return incore, outcore

if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf import ao2mo
    from pyscf.mcscf import mc1step

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvtz',
                 'O': 'cc-pvtz',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()

    mc = mc1step.CASSCF(m, 6, 4)
    mc.verbose = 4
    mo = m.mo_coeff.copy()

    eris0 = _ERIS(mc, mo, 'incore')
    eris1 = _ERIS(mc, mo, 'outcore')
    print('jc_pp', numpy.allclose(eris0.jc_pp, eris1.jc_pp))
    print('kc_pp', numpy.allclose(eris0.kc_pp, eris1.kc_pp))
    print('aapp ', numpy.allclose(eris0.aapp , eris1.aapp ))
    print('appa ', numpy.allclose(eris0.appa , eris1.appa ))
    print('Iapcv ', numpy.allclose(eris0.Iapcv , eris1.Iapcv ))
    print('Icvcv ', numpy.allclose(eris0.Icvcv , eris1.Icvcv ))

    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nmo = mo.shape[1]
    eri = ao2mo.incore.full(m._eri, mo, compact=False).reshape((nmo,)*4)
    aaap = numpy.array(eri[ncore:nocc,ncore:nocc,ncore:nocc,:])
    jc_pp = numpy.einsum('iipq->ipq', eri[:ncore,:ncore,:,:])
    kc_pp = numpy.einsum('ipqi->ipq', eri[:ncore,:,:,:ncore])
    aapp = numpy.array(eri[ncore:nocc,ncore:nocc,:,:])
    appa = numpy.array(eri[ncore:nocc,:,:,ncore:nocc])
    capv = eri[:ncore,ncore:nocc,:,ncore:]
    cvap = eri[:ncore,ncore:,ncore:nocc,:]
    cpav = eri[:ncore,:,ncore:nocc,ncore:]
    ccvv = eri[:ncore,:ncore,ncore:,ncore:]
    cvcv = eri[:ncore,ncore:,:ncore,ncore:]

    cVAp = cvap * 4 \
         - capv.transpose(0,3,1,2) \
         - cpav.transpose(0,3,2,1)
    cVCv = cvcv * 4 \
         - ccvv.transpose(0,3,1,2) \
         - cvcv.transpose(0,3,2,1)

    print('jc_pp', numpy.allclose(jc_pp, eris0.jc_pp))
    print('kc_pp', numpy.allclose(kc_pp, eris0.kc_pp))
    print('aapp ', numpy.allclose(aapp , eris0.aapp ))
    print('appa ', numpy.allclose(appa , eris0.appa ))
    print('Iapcv ', numpy.allclose(cVAp.transpose(2,3,0,1), eris1.Iapcv ))
    print('Icvcv ', numpy.allclose(cVCv.transpose(2,3,0,1), eris1.Icvcv ))
