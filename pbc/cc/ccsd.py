import time
import numpy
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import lib as pbclib
import pyscf.cc
import pyscf.cc.ccsd
import pyscf.pbc.ao2mo
from pyscf.pbc.cc import intermediates

#einsum = np.einsum
einsum = pbclib.einsum

def kernel(cc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6,
           max_memory=2000, verbose=logger.INFO):
    """Exactly the same as pyscf.cc.ccsd.kernel, which calls a
    *local* energy() function."""
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cc.stdout, verbose)

    if t1 is None and t2 is None:
        t1, t2 = cc.init_amps(eris)[1:]
    elif t1 is None:
        nocc = cc.nocc()
        nvir = cc.nmo() - nocc
        t1 = numpy.zeros((nocc,nvir))
    elif t2 is None:
        t2 = cc.init_amps(eris)[2]

    cput1 = cput0 = (time.clock(), time.time())
    nocc, nvir = t1.shape
    eold = 0
    eccsd = 0
    if cc.diis:
        adiis = lib.diis.DIIS(cc, cc.diis_file)
        adiis.space = cc.diis_space
    else:
        adiis = lambda t1,t2,*args: (t1,t2)

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = cc.update_amps(t1, t2, eris, max_memory)
        normt = numpy.linalg.norm(t1new-t1) + numpy.linalg.norm(t2new-t2)
        t1, t2 = t1new, t2new
        t1new = t2new = None
        if cc.diis:
            t1, t2 = cc.diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, energy(cc, t1, t2, eris)
        log.info('istep = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2

@profile
def update_amps(cc, t1, t2, eris, max_memory=2000):
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nocc, nvir = t1.shape
    #nov = nocc*nvir
    fock = eris.fock
    #t1new = numpy.zeros_like(t1)
    #t2new = numpy.zeros_like(t2)

    fov = fock[:nocc,nocc:].copy()
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()

    mo_e = eris.fock.diagonal()
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)

    tau = intermediates.make_tau(t2,t1,t1)

    ### From eom-cc hackathon code ###
    Fvv = intermediates.cc_Fvv(t1,t2,eris)
    Foo = intermediates.cc_Foo(t1,t2,eris)
    Fov = intermediates.cc_Fov(t1,t2,eris)
    Woooo = intermediates.cc_Woooo(t1,t2,eris)
    Wvvvv = intermediates.cc_Wvvvv(t1,t2,eris)
    Wovvo = intermediates.cc_Wovvo(t1,t2,eris)

    # Move energy terms to the other side
    Fvv -= fvv
    Foo -= foo

    eris_oovo = -eris.ooov.transpose(0,1,3,2)
    eris_ovvo = -eris.ovov.transpose(0,1,3,2)
    eris_vvvo = eris.ovvv.transpose(2,3,1,0) # conj() for complex

    # T1 equation
    t1new = fov.copy() 
    t1new +=   einsum('ie,ae->ia',t1,Fvv)
    t1new += - einsum('ma,mi->ia',t1,Foo) 
    t1new +=   einsum('imae,me->ia',t2,Fov)
    t1new += - einsum('nf,naif->ia',t1,eris.ovov)
    t1new += - 0.5*einsum('imef,maef->ia',t2,eris.ovvv)
    t1new += - 0.5*einsum('mnae,nmei->ia',t2,eris_oovo)
    # T2 equation
    t2new = eris.oovv.copy()
    Ftmp = Fvv - 0.5*einsum('mb,me->be',t1,Fov)
    tmp = einsum('ijae,be->ijab',t2,Ftmp)
    t2new += (tmp - tmp.transpose(0,1,3,2))
    Ftmp = Foo + 0.5*einsum('je,me->mj',t1,Fov)
    tmp = einsum('imab,mj->ijab',t2,Ftmp)
    t2new -= (tmp - tmp.transpose(1,0,2,3))
    t2new += ( 0.5*einsum('mnab,mnij->ijab',tau,Woooo) 
             + 0.5*einsum('ijef,abef->ijab',tau,Wvvvv) )
    tmp = einsum('imae,mbej->ijab',t2,Wovvo) 
    tmp -= einsum('ie,ma,mbej->ijab',t1,t1,eris_ovvo)
    t2new += ( tmp - tmp.transpose(0,1,3,2) 
             - tmp.transpose(1,0,2,3) + tmp.transpose(1,0,3,2) )
    tmp = einsum('ie,abej->ijab',t1,eris_vvvo)
    t2new += (tmp - tmp.transpose(1,0,2,3))
    tmp = einsum('ma,mbij->ijab',t1,eris.ovoo)
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    t1new /= eia
    t2new /= eijab

    time0 = log.timer_debug1('update t1 t2', *time0)

    return t1new, t2new


def energy(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    eris_oovv = eris.oovv.copy()
    e = einsum('ia,ia', fock[:nocc,nocc:], t1)
    t1t1 = einsum('ia,jb->ijab',t1,t1)
    tau = t2 + 2*t1t1
    e += 0.25 * np.dot(tau.flatten(), eris_oovv.flatten())
    #e += (0.25*einsum('ijab,ijab',t2,eris_oovv)
    #      + 0.5*einsum('ia,jb,ijab',t1,t1,eris_oovv))
    return e


class CCSD(pyscf.cc.ccsd.CCSD):

    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
        nso = 2*mf.mo_energy.size
        if mo_energy is None: 
            mo_energy = np.zeros(nso)
            mo_energy[0::2] = mo_energy[1::2] = mf.mo_energy
        if mo_coeff is None:
            # TODO: Careful for real/complex here, in the future
            mo_coeffT = mf.mo_coeff.T.real
            so_coeffT = np.zeros((nso,nso))
            for i in range(nso):
                if i%2 == 0:
                    so_coeffT[i][:nso/2] = mo_coeffT[i//2]
                else:
                    so_coeffT[i][nso/2:] = mo_coeffT[i//2]
            # Each col is an eigenvector, first n/2 rows are alpha, then n/2 beta
            mo_coeff = so_coeffT.T
        if mo_occ is None: 
            mo_occ = np.zeros(nso)
            mo_occ[0:mf.cell.nelectron] = 1

        pyscf.cc.ccsd.CCSD.__init__(self, mf, frozen, mo_energy, mo_coeff, mo_occ)

    def nocc(self):
        # Spin orbitals
        self._nocc = 2*pyscf.cc.ccsd.CCSD.nocc(self)
        return self._nocc

    #def nmo(self):
    #    # Spin orbitals
    #    self._nmo = 2*pyscf.cc.ccsd.CCSD.nmo(self)
    #    return self._nmo

    def dump_flags(self):
        pyscf.cc.ccsd.CCSD.dump_flags(self)
        logger.info(self, '\n')
        logger.info(self, '******** PBC CC flags ********')

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        mo_e = eris.fock.diagonal()
        nocc = self.nocc()
        nvir = mo_e.size - nocc
        t1 = np.zeros((nocc,nvir))
        #eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        #t1 = eris.fock[:nocc,nocc:] / eia
        t2 = np.zeros((nocc,nocc,nvir,nvir))
        self.emp2 = 0
        foo = eris.fock[:nocc,:nocc].copy()
        fvv = eris.fock[nocc:,nocc:].copy()
        eris_oovv = eris.oovv.copy()
        eia = np.zeros((nocc,nvir))
        eijab = np.zeros((nocc,nocc,nvir,nvir))
        for i in range(nocc):
            for a in range(nvir):
                eia[i,a] = foo[i,i] - fvv[a,a]
                for j in range(nocc):
                    for b in range(nvir):
                        eijab[i,j,a,b] = ( foo[i,i] + foo[j,j]
                                      - fvv[a,a] - fvv[b,b] )
                        t2[i,j,a,b] = eris_oovv[i,j,a,b]/eijab[i,j,a,b]
        self.emp2 = 0.25*einsum('ijab,ijab',t2,eris_oovv)
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    def ccsd(self, t1=None, t2=None, mo_coeff=None, eris=None):
        if eris is None: eris = self.ao2mo(mo_coeff)
        self._conv, self.ecc, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol,
                       tolnormt=self.conv_tol_normt,
                       max_memory=self.max_memory, verbose=self.verbose)
        if self._conv:
            logger.info(self, 'CCSD converged')
        else:
            logger.info(self, 'CCSD not converge')
        if self._scf.e_tot == 0:
            logger.info(self, 'E_corr = %.16g', self.ecc)
        else:
            logger.info(self, 'E(CCSD) = %.16g  E_corr = %.16g',
                        self.ecc+self._scf.e_tot, self.ecc)
        return self.ecc, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)

    def update_amps(self, t1, t2, eris, max_memory=2000):
        return update_amps(self, t1, t2, eris, max_memory)


class _ERIS:
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
            self.fock = numpy.diag(cc.mo_energy[moidx])
        else:  # If mo_coeff is not canonical orbital
            self.mo_coeff = mo_coeff = mo_coeff[:,moidx]
            dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
            fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
            self.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))

        nocc = cc.nocc()
        nmo = cc.nmo()
        #nvir = nmo - nocc
        #mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
        #mem_now = pyscf.lib.current_memory()[0]
        #
        log = logger.Logger(cc.stdout, cc.verbose)
        #if (method == 'incore' and cc._scf._eri is not None and
        #    (mem_incore+mem_now < cc.max_memory) or cc.mol.incore_anyway):

        # Convert to spin-orbitals and anti-symmetrize 
        print "mo_coeff ="
        print mo_coeff
        so_coeff = np.zeros((nmo/2,nmo))
        so_coeff[:,::2] = so_coeff[:,1::2] = mo_coeff[:nmo/2,::2]
        print "so_coeff ="
        print so_coeff
        eri = pyscf.pbc.ao2mo.get_mo_eri(cc._scf.cell, (so_coeff,so_coeff),
                                                       (so_coeff,so_coeff)).real
        eri = eri.reshape((nmo,)*4)
        eri[::2,1::2] = eri[1::2,::2] = eri[:,:,::2,1::2] = eri[:,:,1::2,::2] = 0.
        print "ERI ="
        print eri.real
        eri1 = eri - eri.transpose(0,3,2,1) 
        eri1 = eri1.transpose(0,2,1,3) 

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

        log.timer('CCSD integral transformation', *cput0)

