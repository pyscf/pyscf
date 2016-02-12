import time
import tempfile
import numpy
import numpy as np
import h5py

from pyscf.lib import logger
from pyscf.pbc import lib as pbclib
import pyscf.cc
import pyscf.cc.ccsd
import pyscf.pbc.ao2mo

from pyscf.cc.ccsd_eom import CCSD as molCCSD
from pyscf.cc.ccsd_eom import _ERIS

#einsum = np.einsum
einsum = pbclib.einsum

class CCSD(molCCSD):

    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
        molCCSD.__init__(self, mf, frozen, mo_energy, mo_coeff, mo_occ)

    def dump_flags(self):
        molCCSD.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** PBC CC flags ********')

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff, ao2mofn=pyscf.pbc.ao2mo.general)


class _XERIS:
    """_ERIS handler for PBCs."""
    def __init__(self, cc, mo_coeff=None, method='incore'):
        cput0 = (time.clock(), time.time())
        moidx = numpy.ones(cc.mo_energy.size, dtype=numpy.bool)
        if isinstance(cc.frozen, (int, numpy.integer)):
            moidx[:cc.frozen] = False
        elif len(cc.frozen) > 0:
            moidx[numpy.asarray(cc.frozen)] = False
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = cc.mo_coeff[:,moidx]
            self.fock = numpy.diag(cc.mo_energy[moidx]).astype(mo_coeff.dtype)
        else:  # If mo_coeff is not canonical orbital
            self.mo_coeff = mo_coeff = mo_coeff[:,moidx]
            dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
            fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
            self.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))

        nocc = cc.nocc()
        nmo = cc.nmo()
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = pyscf.cc.ccsd._mem_usage(nocc, nvir)
        mem_now = pyscf.lib.current_memory()[0]

        # Convert to spin-orbitals and anti-symmetrize 
        so_coeff = np.zeros((nmo/2,nmo), dtype=mo_coeff.dtype)
        so_coeff[:,::2] = so_coeff[:,1::2] = mo_coeff[:nmo/2,::2]

        log = logger.Logger(cc.stdout, cc.verbose)
        if (method == 'incore' and cc._scf._eri is not None and
            (mem_incore+mem_now < cc.max_memory) or cc.mol.incore_anyway):

            eri = pyscf.pbc.ao2mo.general(cc._scf.cell, (so_coeff,so_coeff,
                                                         so_coeff,so_coeff))
                                                        #so_coeff,so_coeff)).real
            eri = eri.reshape((nmo,)*4)
            eri[::2,1::2] = eri[1::2,::2] = eri[:,:,::2,1::2] = eri[:,:,1::2,::2] = 0.
            #print "ERI ="
            #print eri
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
            self.oooo = self.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'c16')
            self.ooov = self.feri1.create_dataset('ooov', (nocc,nocc,nocc,nvir), 'c16')
            self.ovoo = self.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'c16')
            self.oovv = self.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'c16')
            self.ovov = self.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'c16')
            self.ovvv = self.feri1.create_dataset('ovvv', (nocc,nvir,nvir,nvir), 'c16')

            cput1 = time.clock(), time.time()
            buf = pyscf.pbc.ao2mo.general(cc._scf.cell, (orbo,so_coeff,so_coeff,so_coeff))
            buf = buf.reshape((nocc,nmo,nmo,nmo))
            buf[::2,1::2] = buf[1::2,::2] = buf[:,:,::2,1::2] = buf[:,:,1::2,::2] = 0.
            buf1 = buf - buf.transpose(0,3,2,1) 
            buf1 = buf1.transpose(0,2,1,3) 
            cput1 = log.timer_debug1('transforming oppp', *cput1)

            self.dtype = buf1.dtype
            self.oooo[:,:,:,:] = buf1[:,:nocc,:nocc,:nocc]
            self.ooov[:,:,:,:] = buf1[:,:nocc,:nocc,nocc:]
            self.ovoo[:,:,:,:] = buf1[:,nocc:,:nocc,:nocc]
            self.oovv[:,:,:,:] = buf1[:,:nocc,nocc:,nocc:]
            self.ovov[:,:,:,:] = buf1[:,nocc:,:nocc,nocc:]
            self.ovvv[:,:,:,:] = buf1[:,nocc:,nocc:,nocc:]

            self.vvvv = self.feri1.create_dataset('vvvv', (nvir,nvir,nvir,nvir), 'c16')
            for a in range(nvir):
                orbva = orbv[:,a].reshape(-1,1)
                buf = pyscf.pbc.ao2mo.general(cc._scf.cell, (orbva,orbv,orbv,orbv))
                buf = buf.reshape((1,nvir,nvir,nvir))
                if a%2 == 0:
                    buf[0,1::2,:,:] = 0.
                else:
                    buf[0,0::2,:,:] = 0.
                buf[:,:,::2,1::2] = buf[:,:,1::2,::2] = 0.
                buf1 = buf - buf.transpose(0,3,2,1) 
                buf1 = buf1.transpose(0,2,1,3) 
                self.vvvv[a] = buf1[:]

            cput1 = log.timer_debug1('transforming vvvv', *cput1)

        log.timer('CCSD integral transformation', *cput0)
