import time
import tempfile
import numpy
import numpy as np
import h5py

import pyscf.ao2mo
from pyscf.lib import logger
import pyscf.cc
import pyscf.cc.ccsd
import pyscf.cc.ccsd_eom


class UCCSD(pyscf.cc.ccsd_eom.CCSD):

    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
        pyscf.cc.ccsd.CCSD.__init__(self, mf, frozen, mo_energy, mo_coeff, mo_occ)
        # Spin-orbital CCSD needs a stricter tolerance
        self.conv_tol = 1e-8
        self.conv_tol_normt = 1e-6

    def nocc(self):
        # Spin orbitals
        self._nocc = self._scf.mol.nelectron
        return self._nocc

    def nmo(self):
        # Spin orbitals
        self._nmo = 2*self.mo_energy[0].size
        return self._nmo

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)


class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore', 
                 ao2mofn=pyscf.ao2mo.outcore.general_iofree):
        cput0 = (time.clock(), time.time())
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = cc.mo_coeff
            self.fock = numpy.diag(np.append(cc.mo_energy[np.array(cc.mo_occ,dtype=bool)], 
                                             cc.mo_energy[np.logical_not(np.array(cc.mo_occ,dtype=bool))])).astype(mo_coeff.dtype)

        nocc = cc.nocc()
        nmo = cc.nmo()
        nvir = nmo - nocc
        #mem_incore, mem_outcore, mem_basic = pyscf.cc.ccsd._mem_usage(nocc, nvir)
        mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
        mem_incore *= 4
        mem_now = pyscf.lib.current_memory()[0]

        # Convert to spin-orbitals and anti-symmetrize 
        so_coeff = np.zeros((nmo/2,nmo), dtype=mo_coeff.dtype)
        nocc_a = int(sum(cc.mo_occ[0]))
        nocc_b = int(sum(cc.mo_occ[1]))
        nvir_a = nmo/2 - nocc_a
        nvir_b = nmo/2 - nocc_b
        spin = np.zeros(nmo, dtype=int)
        spin[:nocc_a] = 0
        spin[nocc_a:nocc] = 1
        spin[nocc:nocc+nvir_a] = 0
        spin[nocc+nvir_a:nmo] = 1
        so_coeff[:,:nocc_a] = mo_coeff[0][:,:nocc_a]
        so_coeff[:,nocc_a:nocc] = mo_coeff[1][:,:nocc_b]
        so_coeff[:,nocc:nocc+nvir_a] = mo_coeff[0][:,nocc_a:nmo/2]
        so_coeff[:,nocc+nvir_a:nmo] = mo_coeff[1][:,nocc_b:nmo/2]

        log = logger.Logger(cc.stdout, cc.verbose)
        if (method == 'incore' and (mem_incore+mem_now < cc.max_memory) 
            or cc.mol.incore_anyway):
            eri = ao2mofn(cc._scf.mol, (so_coeff,so_coeff,so_coeff,so_coeff), compact=0)
            if mo_coeff.dtype == np.float: eri = eri.real
            eri = eri.reshape((nmo,)*4)
            for i in range(nmo):
                for j in range(i):
                    if spin[i] != spin[j]:
                        eri[i,j,:,:] = eri[j,i,:,:] = 0.
                        eri[:,:,i,j] = eri[:,:,j,i] = 0.
            eri1 = eri - eri.transpose(0,3,2,1) 
            eri1 = eri1.transpose(0,2,1,3) 

            self.dtype = eri1.dtype
            self.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
            self.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
            self.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
            self.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
            self.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
            self.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
            self.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy() 
            #ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
            #self.ovvv = numpy.empty((nocc,nvir,nvir*(nvir+1)//2))
            #for i in range(nocc):
            #    for j in range(nvir):
            #        self.ovvv[i,j] = lib.pack_tril(ovvv[i,j])
            #self.vvvv = pyscf.ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)

        else:
            print "*** Using HDF5 ERI storage ***"
            _tmpfile1 = tempfile.NamedTemporaryFile()
            self.feri1 = h5py.File(_tmpfile1.name)
            orbo = so_coeff[:,:nocc]
            orbv = so_coeff[:,nocc:]
            if mo_coeff.dtype == np.complex: ds_type = 'c16'
            else: ds_type = 'f8'
            self.oooo = self.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), ds_type)
            self.ooov = self.feri1.create_dataset('ooov', (nocc,nocc,nocc,nvir), ds_type)
            self.ovoo = self.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), ds_type)
            self.oovv = self.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), ds_type)
            self.ovov = self.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), ds_type)
            self.ovvv = self.feri1.create_dataset('ovvv', (nocc,nvir,nvir,nvir), ds_type)
            self.vvvv = self.feri1.create_dataset('vvvv', (nvir,nvir,nvir,nvir), ds_type)

            cput1 = time.clock(), time.time()
            # <ij||pq> = <ij|pq> - <ij|qp> = (ip|jq) - (iq|jp)
            buf = ao2mofn(cc._scf.mol, (orbo,so_coeff,orbo,so_coeff), compact=0)
            if mo_coeff.dtype == np.float: buf = buf.real
            buf = buf.reshape((nocc,nmo,nocc,nmo))
            #buf[::2,1::2] = buf[1::2,::2] = buf[:,:,::2,1::2] = buf[:,:,1::2,::2] = 0.
            for i in range(nocc):
                for p in range(nmo):
                    if spin[i] != spin[p]:
                        buf[i,p,:,:] = 0.
                        buf[:,:,i,p] = 0.
            buf1 = buf - buf.transpose(0,3,2,1)
            buf1 = buf1.transpose(0,2,1,3) 
            cput1 = log.timer_debug1('transforming oopq', *cput1)
            self.dtype = buf1.dtype
            self.oooo[:,:,:,:] = buf1[:,:,:nocc,:nocc]
            self.ooov[:,:,:,:] = buf1[:,:,:nocc,nocc:]
            self.oovv[:,:,:,:] = buf1[:,:,nocc:,nocc:]

            cput1 = time.clock(), time.time()
            # <ia||pq> = <ia|pq> - <ia|qp> = (ip|aq) - (iq|ap)
            buf = ao2mofn(cc._scf.mol, (orbo,so_coeff,orbv,so_coeff), compact=0)
            if mo_coeff.dtype == np.float: buf = buf.real
            buf = buf.reshape((nocc,nmo,nvir,nmo))
            #buf[::2,1::2] = buf[1::2,::2] = buf[:,:,::2,1::2] = buf[:,:,1::2,::2] = 0.
            for p in range(nmo):
                for i in range(nocc):
                    if spin[i] != spin[p]:
                        buf[i,p,:,:] = 0.
                for a in range(nvir):
                    if spin[nocc+a] != spin[p]:
                        buf[:,:,a,p] = 0.
            buf1 = buf - buf.transpose(0,3,2,1)
            buf1 = buf1.transpose(0,2,1,3) 
            cput1 = log.timer_debug1('transforming ovpq', *cput1)
            self.ovoo[:,:,:,:] = buf1[:,:,:nocc,:nocc]
            self.ovov[:,:,:,:] = buf1[:,:,:nocc,nocc:]
            self.ovvv[:,:,:,:] = buf1[:,:,nocc:,nocc:]

            for a in range(nvir):
                orbva = orbv[:,a].reshape(-1,1)
                buf = ao2mofn(cc._scf.mol, (orbva,orbv,orbv,orbv), compact=0)
                if mo_coeff.dtype == np.float: buf = buf.real
                buf = buf.reshape((1,nvir,nvir,nvir))
                #if a%2 == 0:
                #    buf[0,1::2,:,:] = 0.
                #else:
                #    buf[0,0::2,:,:] = 0.
                #buf[:,:,::2,1::2] = buf[:,:,1::2,::2] = 0.
                for b in range(nvir):
                    if spin[nocc+a] != spin[nocc+b]:
                        buf[0,b,:,:] = 0.
                    for c in range(nvir):
                        if spin[nocc+b] != spin[nocc+c]:
                            buf[:,:,b,c] = buf[:,:,c,b] = 0.
                buf1 = buf - buf.transpose(0,3,2,1) 
                buf1 = buf1.transpose(0,2,1,3) 
                self.vvvv[a] = buf1[:]

            cput1 = log.timer_debug1('transforming vvvv', *cput1)

        log.timer('CCSD integral transformation', *cput0)

def _mem_usage(nocc, nvir):
    basic = nocc*(nocc+1)//2*nvir**2 + (nocc*nvir)**2*2
    basic = basic * 8/1e6
    nmo = nocc + nvir
    incore = (max((nmo*(nmo+1)//2)**2*2*8/1e6, basic) +
              (nocc**4 + nocc*nvir**3 + nvir**4 + nocc**2*nvir**2*2 +
               nocc**3*nvir*2)*8/1e6)
    outcore = basic
    return incore, outcore, basic
