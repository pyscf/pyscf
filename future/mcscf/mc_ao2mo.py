#!/usr/bin/env python

import os
import ctypes
import tempfile
import numpy
import h5py
from pyscf import lib
import pyscf.lib.numpy_helper
from pyscf import ao2mo
import pyscf.ao2mo._ao2mo

_alib = os.path.join(os.path.dirname(lib.__file__), 'libmcscf.so')
libmcscf = ctypes.CDLL(_alib)

# least memory requirements:
#       ncore**2*(nmo-ncore)*nmo + ncas**2*nmo**2*2 + nmo**3   words
# nmo  ncore  ncas  outcore  incore
# 200  40     16    0.6GB    3.5 GB (_eri 1.6GB intermediates 1.3G)
# 250  50     16    1.4GB    8.4 GB (_eri 3.9GB intermediates 3.1G)
# 300  60     16    2.7GB    17.2GB (_eri 8.1GB intermediates 6.5G)
# 400  80     16    7.8GB    54  GB (_eri 25.6GB intermediates 20.5G)
# 500  100    16    18 GB
# 600  120    16    37 GB
# 750  150    16    87 GB


def trans_e1_incore(casscf, mo):
    ncore = casscf.ncore
    ncas = casscf.ncas
    nmo = mo.shape[1]
    nocc = ncore + ncas
    nvir = nmo - nocc
    moji = numpy.array(numpy.hstack((mo,mo[:,:nocc])), order='F')
    ijshape = (nmo, nocc, 0, nmo)

    eri1 = ao2mo._ao2mo.nr_e1_incore(casscf._scf._eri, moji, ijshape)

    c_nmo = ctypes.c_int(nmo)
    funpack = lib.numpy_helper._np_helper.NPdunpack_tril
    jc_pp = numpy.empty((ncore,nmo,nmo))
    kc_pp = numpy.empty((ncore,nmo,nmo))

    klshape = (0, nmo, 0, nmo)
    buf = ao2mo._ao2mo.nr_e2(eri1[ncore*nmo:nocc*nmo], moji, klshape)
    japcp = numpy.empty((ncas,nmo,ncore,nmo))
    appp = numpy.empty((ncas,nmo,nmo,nmo))
    ij = 0
    for i in range(ncas):
        for j in range(nmo):
            funpack(c_nmo, buf[ij].ctypes.data_as(ctypes.c_void_p),
                    appp[i,j].ctypes.data_as(ctypes.c_void_p))
            ij += 1
        libmcscf.MCSCFinplace_apcp(japcp[i].ctypes.data_as(ctypes.c_void_p),
                                   appp[i].ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_int(ncore), ctypes.c_int(ncas),
                                   ctypes.c_int(nmo))
    aapp = appp[:,ncore:nocc,:,:].copy()
    appa = appp[:,:,:,ncore:nocc].copy()
    aaaa = aapp[:,:,ncore:nocc,ncore:nocc].copy()
    #apcp = appp[:,:,:ncore,:]
    #acpp = appp[:,:ncore,:,:]
    #japcp = apcp * 4 \
    #        - acpp.transpose(0,2,1,3) \
    #        - apcp.transpose(0,3,2,1)
    buf = appp = None

#    cvcp = numpy.empty((ncore,nmo-ncore,ncore,nmo))
#    klshape = (nmo, ncore, 0, nmo)
#    for i in range(ncore):
#        ao2mo._ao2mo.nr_e2(eri1[i*nmo+ncore:i*nmo+nmo], moji, klshape, cvcp[i])
#        kc_pp[i,ncore:] = cvcp[i,:,i]
#    jcvcp = cvcp * 4 - cvcp.transpose(2,1,0,3)

#    cpp = numpy.empty((ncore,nmo,nmo))
#    klshape = (0, nmo, 0, nmo)
#    for i in range(ncore):
#        buf = ao2mo._ao2mo.nr_e2(eri1[i*nmo:i*nmo+ncore], moji, klshape)
#        for j in range(ncore):
#            funpack(c_nmo, buf[j].ctypes.data_as(ctypes.c_void_p),
#                    cpp[j].ctypes.data_as(ctypes.c_void_p))
#        jc_pp[i] = cpp[i]
#        kc_pp[i,:ncore] = cpp[:,i]
#        jcvcp[i] -= cpp[:,ncore:].transpose(1,0,2)

    jcvcp = numpy.zeros((ncore,nmo-ncore,ncore,nmo))
    vcp = numpy.empty((nmo-ncore,ncore,nmo))
    cpp = numpy.empty((ncore,nmo,nmo))
    for i in range(ncore):
        klshape = (nmo, ncore, 0, nmo)
        ao2mo._ao2mo.nr_e2(eri1[i*nmo+ncore:i*nmo+nmo], moji, klshape, vcp)
        kc_pp[i,ncore:] = vcp[:,i]

        klshape = (0, nmo, 0, nmo)
        buf = ao2mo._ao2mo.nr_e2(eri1[i*nmo:i*nmo+ncore], moji, klshape)
        for j in range(ncore):
            funpack(c_nmo, buf[j].ctypes.data_as(ctypes.c_void_p),
                    cpp[j].ctypes.data_as(ctypes.c_void_p))
        jc_pp[i] = cpp[i]
        kc_pp[i,:ncore] = cpp[:,i]

        #jcvcp = cvcp * 4 - cvcp.transpose(2,1,0,3) - ccvp.transpose(0,2,1,3)
        #jcvcp[i] += vcp * 4 - cpp[:,ncore:].transpose(1,0,2)
        #jcvcp[:,:,i] -= vcp.transpose(1,0,2)
        libmcscf.MCSCFinplace_cvcp(jcvcp.ctypes.data_as(ctypes.c_void_p),
                                   vcp.ctypes.data_as(ctypes.c_void_p),
                                   cpp.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_int(i), ctypes.c_int(ncore),
                                   ctypes.c_int(ncas), ctypes.c_int(nmo))
    vcp = cpp = None

    return jc_pp, kc_pp, aapp, appa, aaaa, japcp, jcvcp


def trans_e1_outcore(casscf, mo, max_memory=None, ioblk_size=512, tmpdir=None,
                     verbose=0):
    log = lib.logger.Logger(casscf.stdout, verbose)
    mol = casscf.mol
    ncore = casscf.ncore
    ncas = casscf.ncas
    nao, nmo = mo.shape
    nocc = ncore + ncas
    nvir = nmo - nocc
    moji = numpy.array(numpy.hstack((mo,mo[:,:nocc])), order='F')
    ijshape = (nmo, nocc, 0, nmo)

    nij_pair = nocc * nmo
    nao_pair = nao*(nao+1)/2
    mem_words, ioblk_words = \
            ao2mo.direct._memory_and_ioblk_size(max_memory, ioblk_size,
                                                nij_pair, nao_pair)
    e1_buflen = min(int(mem_words/nij_pair), nao_pair)
    ish_ranges = ao2mo.outcore._info_sh_ranges(mol, e1_buflen)

    log.debug1('require disk %.8g MB, swap-block-shape (%d,%d), mem cache size %.8g MB',
               nij_pair*nao_pair*8/1e6, e1_buflen, nmo,
               max(e1_buflen*nij_pair,nmo*nao_pair)*8/1e6)

    swapfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    fswap = h5py.File(swapfile.name)

    for istep, sh_range in enumerate(ish_ranges):
        log.debug1('step 1, AO %d:%d, [%d/%d], len(buf) = %d', \
                  sh_range[0], sh_range[1], istep+1, len(ish_ranges), \
                  sh_range[2])
        try:
            buf = ao2mo._ao2mo.nr_e1range(moji, sh_range, ijshape, \
                                          mol._atm, mol._bas, mol._env)
        except MemoryError:
            log.warn('not enough memory or limited virtual address space `ulimit -v`')
            raise MemoryError
        for ic in range(nocc):
            col0 = ic * nmo
            col1 = ic * nmo + nmo
            fswap['%d/%d'%(istep,ic)] = lib.transpose(buf[:,col0:col1])
        buf = None

    ###########################
    def load_buf(bfn_id):
        buf = numpy.empty((nmo,nao_pair))
        col0 = 0
        for ic, _ in enumerate(ish_ranges):
            dat = fswap['%d/%d'%(ic,bfn_id)]
            col1 = col0 + dat.shape[1]
            buf[:nmo,col0:col1] = dat
            col0 = col1
        return buf
    c_nmo = ctypes.c_int(nmo)
    funpack = lib.numpy_helper._np_helper.NPdunpack_tril
    jc_pp = numpy.empty((ncore,nmo,nmo))
    kc_pp = numpy.empty((ncore,nmo,nmo))

    klshape = (0, nmo, 0, nmo)
    japcp = numpy.empty((ncas,nmo,ncore,nmo))
    aapp = numpy.empty((ncas,ncas,nmo,nmo))
    appa = numpy.empty((ncas,nmo,nmo,ncas))
    ppp = numpy.empty((nmo,nmo,nmo))
    for i in range(ncas):
        buf = ao2mo._ao2mo.nr_e2(load_buf(ncore+i), moji, klshape)
        for j in range(nmo):
            funpack(c_nmo, buf[j].ctypes.data_as(ctypes.c_void_p),
                    ppp[j].ctypes.data_as(ctypes.c_void_p))
        buf = None
        aapp[i] = ppp[ncore:nocc]
        appa[i] = ppp[:,:,ncore:nocc]
        libmcscf.MCSCFinplace_apcp(japcp[i].ctypes.data_as(ctypes.c_void_p),
                                   ppp.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_int(ncore), ctypes.c_int(ncas),
                                   ctypes.c_int(nmo))
    aaaa = aapp[:,:,ncore:nocc,ncore:nocc].copy()
    ppp = None

    jcvcp = numpy.zeros((ncore,nmo-ncore,ncore,nmo))
    vcp = numpy.empty((nmo-ncore,ncore,nmo))
    cpp = numpy.empty((ncore,nmo,nmo))
    for i in range(ncore):
        buf = load_buf(i)
        klshape = (nmo, ncore, 0, nmo)
        ao2mo._ao2mo.nr_e2(buf[ncore:nmo], moji, klshape, vcp)
        kc_pp[i,ncore:] = vcp[:,i]

        klshape = (0, nmo, 0, nmo)
        ao2mo._ao2mo.nr_e2(buf[:ncore], moji, klshape, buf[:ncore])
        for j in range(ncore):
            funpack(c_nmo, buf[j].ctypes.data_as(ctypes.c_void_p),
                    cpp[j].ctypes.data_as(ctypes.c_void_p))
        jc_pp[i] = cpp[i]
        kc_pp[i,:ncore] = cpp[:,i]

        #jcvcp = cvcp * 4 - cvcp.transpose(2,1,0,3) - ccvp.transpose(0,2,1,3)
        #jcvcp[i] += vcp * 4 - cpp[:,ncore:].transpose(1,0,2)
        #jcvcp[:,:,i] -= vcp.transpose(1,0,2)
        libmcscf.MCSCFinplace_cvcp(jcvcp.ctypes.data_as(ctypes.c_void_p),
                                   vcp.ctypes.data_as(ctypes.c_void_p),
                                   cpp.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_int(i), ctypes.c_int(ncore),
                                   ctypes.c_int(ncas), ctypes.c_int(nmo))
    vcp = cpp = None
    fswap.close()

    return jc_pp, kc_pp, aapp, appa, aaaa, japcp, jcvcp



class _ERIS(object):
    def __init__(self, casscf, mo, method='incore'):
        mol = casscf.mol
        self.ncore = casscf.ncore
        self.ncas = casscf.ncas
        self.nmo = mo.shape[1]
        ncore = self.ncore
        ncas = self.ncas
        nmo = self.nmo
        nocc = ncore + ncas
        mocc = mo[:,:nocc]

        if method == 'outcore' \
           or _mem_usage(ncore, ncas, nmo)[0] + nmo**4*2/1e6 > casscf.max_memory*.9:
            self.jc_pp, self.kc_pp, \
            self.aapp, self.appa, self.aaaa, \
            self.ApcP, self.CvcP = \
                    trans_e1_outcore(casscf, mo, max_memory=casscf.max_memory,
                                     verbose=casscf.verbose)
        elif method == 'incore' or casscf._scf._eri is not None:
            self.jc_pp, self.kc_pp, \
            self.aapp, self.appa, self.aaaa, \
            self.ApcP, self.CvcP = \
                    trans_e1_incore(casscf, mo)
        else:
            raise KeyError('update ao2mo')

def _mem_usage(ncore, ncas, nmo):
    outcore = (ncore**2*(nmo-ncore)*nmo + ncas**2*nmo**2*2 + nmo**3) * 8/1e6
    incore = outcore + nmo**4/1e6 + ncore*nmo**3*4/1e6
    if outcore > 10000:
        print 'Be careful with the virtual memorty address space `ulimit -v`'
    return incore, outcore

if __name__ == '__main__':
    import scf
    import gto
    import mc1step

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()

    mc = mc1step.CASSCF(mol, m, 6, 4)
    mc.verbose = 4
    mo = m.mo_coeff.copy()
    eris0 = _ERIS(mc, mo, 'incore')
    eris1 = _ERIS(mc, mo, 'outcore')
    print(numpy.allclose(eris0.jc_pp, eris1.jc_pp))
    print(numpy.allclose(eris0.kc_pp, eris1.kc_pp))
    print(numpy.allclose(eris0.aapp , eris1.aapp ))
    print(numpy.allclose(eris0.appa , eris1.appa ))
    print(numpy.allclose(eris0.aaaa , eris1.aaaa ))
    print(numpy.allclose(eris0.ApcP , eris1.ApcP ))
    print(numpy.allclose(eris0.CvcP , eris1.CvcP ))

