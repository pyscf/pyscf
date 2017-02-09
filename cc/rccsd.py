import time
import tempfile
import numpy
import numpy as np
import h5py

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import rintermediates as imd
from pyscf.lib import linalg_helper

#einsum = np.einsum
einsum = lib.einsum

# This is restricted (R)CCSD
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

def kernel(cc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6,
           verbose=logger.INFO):
    """Exactly the same as pyscf.cc.ccsd.kernel, which calls a
    *local* energy() function."""
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cc.stdout, verbose)

    if t1 is None and t2 is None:
        t1, t2 = cc.init_amps(eris)[1:]
    elif t1 is None:
        nocc = cc.nocc
        nvir = cc.nmo - nocc
        t1 = numpy.zeros((nocc,nvir), eris.dtype)
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
        t1new, t2new = cc.update_amps(t1, t2, eris)
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


def update_amps(cc, t1, t2, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:]
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    mo_e = eris.fock.diagonal()
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)

    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)
    Loo = imd.Loo(t1,t2,eris)
    Lvv = imd.Lvv(t1,t2,eris)
    Woooo = imd.cc_Woooo(t1,t2,eris)
    Wvvvv = imd.cc_Wvvvv(t1,t2,eris)
    Wvoov = imd.cc_Wvoov(t1,t2,eris)
    Wvovo = imd.cc_Wvovo(t1,t2,eris)

    # Move energy terms to the other side
    Foo -= np.diag(np.diag(foo))
    Fvv -= np.diag(np.diag(fvv))
    Loo -= np.diag(np.diag(foo))
    Lvv -= np.diag(np.diag(fvv))

    # T1 equation
    t1new = np.array(fov).conj()
    t1new += -2*einsum('kc,ka,ic->ia',fov,t1,t1)
    t1new +=   einsum('ac,ic->ia',Fvv,t1)
    t1new +=  -einsum('ki,ka->ia',Foo,t1)
    t1new += 2*einsum('kc,kica->ia',Fov,t2)
    t1new +=  -einsum('kc,ikca->ia',Fov,t2)
    t1new +=   einsum('kc,ic,ka->ia',Fov,t1,t1)
    t1new += 2*einsum('akic,kc->ia',eris.voov,t1)
    t1new +=  -einsum('kaic,kc->ia',eris.ovov,t1)
    t1new += 2*einsum('akcd,ikcd->ia',eris.vovv,t2)
    t1new +=  -einsum('akdc,ikcd->ia',eris.vovv,t2)
    t1new += 2*einsum('akcd,ic,kd->ia',eris.vovv,t1,t1)
    t1new +=  -einsum('akdc,ic,kd->ia',eris.vovv,t1,t1)
    t1new += -2*einsum('klic,klac->ia',eris.ooov,t2)
    t1new +=  einsum('lkic,klac->ia',eris.ooov,t2)
    t1new += -2*einsum('klic,ka,lc->ia',eris.ooov,t1,t1)
    t1new +=  einsum('lkic,ka,lc->ia',eris.ooov,t1,t1)

    # T2 equation
    t2new = np.array(eris.oovv).conj()
    t2new += einsum('klij,klab->ijab',Woooo,t2)
    t2new += einsum('klij,ka,lb->ijab',Woooo,t1,t1)
    for a in range(nvir):
        Wvvvv_a = np.array(Wvvvv[a])
        t2new[:,:,a,:] += einsum('bcd,ijcd->ijb',Wvvvv_a,t2)
        t2new[:,:,a,:] += einsum('bcd,ic,jd->ijb',Wvvvv_a,t1,t1)
    tmp = einsum('ac,ijcb->ijab',Lvv,t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp = einsum('ki,kjab->ijab',Loo,t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp2 = np.array(eris.vovv).transpose(3,2,1,0).conj() \
            - einsum('kbic,ka->abic',eris.ovov,t1)
    tmp = einsum('abic,jc->ijab',tmp2,t1)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp2 = np.array(eris.ooov).transpose(3,2,1,0).conj() \
            + einsum('akic,jc->akij',eris.voov,t1)
    tmp = einsum('akij,kb->ijab',tmp2,t1)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp = 2*einsum('akic,kjcb->ijab',Wvoov,t2) - einsum('akci,kjcb->ijab',Wvovo,t2)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp = einsum('akic,kjbc->ijab',Wvoov,t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    tmp = einsum('bkci,kjac->ijab',Wvovo,t2)
    t2new -= (tmp + tmp.transpose(1,0,3,2))

    t1new /= eia
    t2new /= eijab

    time0 = log.timer_debug1('update t1 t2', *time0)

    return t1new, t2new


def energy(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    e = 2*einsum('ia,ia', fock[:nocc,nocc:], t1)
    t1t1 = einsum('ia,jb->ijab',t1,t1)
    tau = t2 + t1t1
    e += einsum('ijab,ijab', 2*tau, eris.oovv)
    e += einsum('ijab,ijba',  -tau, eris.oovv)
    return e.real


class RCCSD(ccsd.CCSD):

    def __init__(self, mf, frozen=[], mo_coeff=None, mo_occ=None):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.made_ee_imds = False
        self.max_space = 20

    def dump_flags(self):
        ccsd.CCSD.dump_flags(self)

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        mo_e = eris.fock.diagonal()
        nocc = self.nocc
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
        eris_oovv = np.array(eris.oovv)
        t1 = eris.fock[:nocc,nocc:] / eia
        t2 = eris_oovv/eijab
        wvvoo = (2*eris_oovv 
                  -eris_oovv.transpose(0,1,3,2)).transpose(2,3,0,1).conj()
        self.emp2 = einsum('ijab,abij',t2,wvvoo).real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2)
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.eris = eris
        self.dump_flags()
        if mbpt2:
            cctyp = 'MBPT2'
            self.e_corr, self.t1, self.t2 = self.init_amps(eris)
        else:
            cctyp = 'CCSD'
            self.converged, self.e_corr, self.t1, self.t2 = \
                    kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                           tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                           verbose=self.verbose)
            if self.converged:
                logger.info(self, 'CCSD converged')
            else:
                logger.info(self, 'CCSD not converged')
        if self._scf.e_tot == 0:
            logger.note(self, 'E_corr = %.16g', self.e_corr)
        else:
            logger.note(self, 'E(%s) = %.16g  E_corr = %.16g',
                        cctyp, self.e_tot, self.e_corr)
        return self.e_corr, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)

    def update_amps(self, t1, t2, eris):
        return update_amps(self, t1, t2, eris)

    def nip(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        self._nip = nocc + nocc*nocc*nvir
        return self._nip

    def nea(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        self._nea = nvir + nocc*nvir*nvir
        return self._nea

    def nee(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        self._nee = nocc*nvir + nocc*nocc*nvir*nvir
        return self._nee

    def ipccsd(self, nroots=1, koopmans=False, guess=None, partition=None):
        '''Calculate (N-1)-electron charged excitations via IP-EOM-CCSD.

        Kwargs:
            nroots : int
                Number of roots (eigenvalues) requested
            partition : bool or str
                Use a matrix-partitioning for the doubles-doubles block.
                Can be None, 'mp' (Moller-Plesset, i.e. orbital energies on the diagonal),
                or 'full' (full diagonal elements).
            koopmans : bool
                Calculate Koopmans'-like (quasiparticle) excitations only, targeting via
                overlap.
            guess : list of ndarray
                List of guess vectors to use for targeting via overlap.
        '''
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        size = self.nip()
        nroots = min(nroots,size)
        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ip_partition = partition
        if partition == 'full':
            self._ipccsd_diag_matrix2 = self.vector_to_amplitudes_ip(self.ipccsd_diag())[1]

        adiag = self.ipccsd_diag()
        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            guess = []
            if koopmans:
                for n in range(nroots):
                    g = np.zeros(size)
                    g[self.nocc-n-1] = 1.0
                    guess.append(g)
            else:
                idx = adiag.argsort()[:nroots]
                for i in idx:
                    g = np.zeros(size)
                    g[i] = 1.0
                    guess.append(g)

        def precond(r, e0, x0):
            return r/(e0-adiag+1e-12)

        eig = linalg_helper.eig
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return w[idx].real, v[:,idx].real, idx
            eip, evecs = eig(self.ipccsd_matvec, guess, precond, pick=pickeig,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=7)
        else:
            eip, evecs = eig(self.ipccsd_matvec, guess, precond,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=7)

        self.eip = eip.real

        if nroots == 1:
            eip, evecs = [self.eip], [evecs]
        for n, en, vn in zip(range(nroots), eip, evecs):
            logger.info(self, 'IP root %d E = %.16g  qpwt = %0.6g', 
                        n, en, np.linalg.norm(vn[:self.nocc])**2)
        log.timer('IP-CCSD', *cput0)
        if nroots == 1:
            return eip[0], evecs[0]
        else:
            return eip, evecs

    def ipccsd_matvec(self, vector):
        # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
        if not self.made_ip_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ip()
            self.made_ip_imds = True
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ip(vector)

        # 1h-1h block
        Hr1 = -einsum('ki,k->i',imds.Loo,r1)
        #1h-2h1p block
        Hr1 += 2*einsum('ld,ild->i',imds.Fov,r2)
        Hr1 +=  -einsum('kd,kid->i',imds.Fov,r2)
        Hr1 += -2*einsum('klid,kld->i',imds.Wooov,r2)
        Hr1 +=    einsum('lkid,kld->i',imds.Wooov,r2)

        # 2h1p-1h block
        Hr2 = -einsum('kbij,k->ijb',imds.Wovoo,r1)
        # 2h1p-2h1p block
        if self.ip_partition == 'mp':
            nocc, nvir = self.t1.shape
            fock = self.eris.fock
            foo = fock[:nocc,:nocc]
            fvv = fock[nocc:,nocc:]
            Hr2 += einsum('bd,ijd->ijb',fvv,r2)
            Hr2 += -einsum('ki,kjb->ijb',foo,r2)
            Hr2 += -einsum('lj,ilb->ijb',foo,r2)
        elif self.ip_partition == 'full':
            Hr2 += self._ipccsd_diag_matrix2*r2
        else:
            Hr2 += einsum('bd,ijd->ijb',imds.Lvv,r2)
            Hr2 += -einsum('ki,kjb->ijb',imds.Loo,r2)
            Hr2 += -einsum('lj,ilb->ijb',imds.Loo,r2)
            Hr2 +=  einsum('klij,klb->ijb',imds.Woooo,r2)
            Hr2 += 2*einsum('lbdj,ild->ijb',imds.Wovvo,r2)
            Hr2 +=  -einsum('kbdj,kid->ijb',imds.Wovvo,r2)
            Hr2 +=  -einsum('lbjd,ild->ijb',imds.Wovov,r2) #typo in Ref 
            Hr2 +=  -einsum('kbid,kjd->ijb',imds.Wovov,r2)
            tmp = 2*einsum('lkdc,kld->c',imds.Woovv,r2)
            tmp += -einsum('kldc,kld->c',imds.Woovv,r2)
            Hr2 += -einsum('c,ijcb->ijb',tmp,self.t2)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def ipccsd_diag(self):
        if not self.made_ip_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ip()
            self.made_ip_imds = True
        imds = self.imds

        t1, t2 = self.t1, self.t2
        nocc, nvir = t1.shape
        fock = self.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]

        Hr1 = -np.diag(imds.Loo)
        Hr2 = np.zeros((nocc,nocc,nvir), t1.dtype)
        for i in range(nocc):
            for j in range(nocc):
                for b in range(nvir):
                    if self.ip_partition == 'mp':
                        Hr2[i,j,b] += fvv[b,b]
                        Hr2[i,j,b] += -foo[i,i]
                        Hr2[i,j,b] += -foo[j,j]
                    else:
                        Hr2[i,j,b] += imds.Lvv[b,b]
                        Hr2[i,j,b] += -imds.Loo[i,i]
                        Hr2[i,j,b] += -imds.Loo[j,j]
                        Hr2[i,j,b] += imds.Woooo[i,j,i,j]
                        Hr2[i,j,b] += 2*imds.Wovvo[j,b,b,j]
                        Hr2[i,j,b] += -imds.Wovvo[i,b,b,i]*(i==j)
                        Hr2[i,j,b] += -imds.Wovov[j,b,j,b]
                        Hr2[i,j,b] += -imds.Wovov[i,b,i,b]
                        Hr2[i,j,b] += -2*np.dot(imds.Woovv[j,i,b,:],t2[i,j,:,b])
                        Hr2[i,j,b] += np.dot(imds.Woovv[i,j,b,:],t2[i,j,:,b])

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ip(self,vector):
        nocc = self.nocc
        nvir = self.nmo - nocc
        r1 = vector[:nocc].copy()
        r2 = vector[nocc:].copy().reshape(nocc,nocc,nvir)
        return [r1,r2]

    def amplitudes_to_vector_ip(self,r1,r2):
        nocc = self.nocc
        nvir = self.nmo - nocc
        size = self.nip()
        vector = np.zeros(size, r1.dtype)
        vector[:nocc] = r1.copy()
        vector[nocc:] = r2.copy().reshape(nocc*nocc*nvir)
        return vector

    def eaccsd(self, nroots=1, koopmans=False, guess=None, partition=None):
        '''Calculate (N+1)-electron charged excitations via EA-EOM-CCSD.

        Kwargs:
            See ipccd()
        '''
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        size = self.nea()
        nroots = min(nroots,size)
        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ea_partition = partition
        if partition == 'full':
            self._eaccsd_diag_matrix2 = self.vector_to_amplitudes_ea(self.eaccsd_diag())[1]

        adiag = self.eaccsd_diag()
        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            guess = []
            if koopmans:
                for n in range(nroots):
                    g = np.zeros(size)
                    g[n] = 1.0
                    guess.append(g)
            else:
                idx = adiag.argsort()[:nroots]
                for i in idx:
                    g = np.zeros(size)
                    g[i] = 1.0
                    guess.append(g)

        def precond(r, e0, x0):
            return r/(e0-adiag+1e-12)

        eig = linalg_helper.eig
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return w[idx].real, v[:,idx].real, idx
            eea, evecs = eig(self.eaccsd_matvec, guess, precond, pick=pickeig,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=7)
        else:
            eea, evecs = eig(self.eaccsd_matvec, guess, precond,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=7)

        self.eea = eea.real

        if nroots == 1:
            eea, evecs = [self.eea], [evecs]
        nvir = self.nmo - self.nocc
        for n, en, vn in zip(range(nroots), eea, evecs):
            logger.info(self, 'EA root %d E = %.16g  qpwt = %0.6g', 
                        n, en, np.linalg.norm(vn[:nvir])**2)
        log.timer('EA-CCSD', *cput0)
        if nroots == 1:
            return eea[0], evecs[0]
        else:
            return eea, evecs

    def eaccsd_matvec(self,vector):
        # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
        if not self.made_ea_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ea()
            self.made_ea_imds = True
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ea(vector)

        # Eq. (30)
        # 1p-1p block
        Hr1 =  einsum('ac,c->a',imds.Lvv,r1)
        # 1p-2p1h block
        Hr1 += einsum('ld,lad->a',2.*imds.Fov,r2)
        Hr1 += einsum('ld,lda->a',  -imds.Fov,r2)
        Hr1 += 2*einsum('alcd,lcd->a',imds.Wvovv,r2)
        Hr1 +=  -einsum('aldc,lcd->a',imds.Wvovv,r2)
        # Eq. (31)
        # 2p1h-1p block
        Hr2 = einsum('abcj,c->jab',imds.Wvvvo,r1)
        # 2p1h-2p1h block
        if self.ea_partition == 'mp':
            nocc, nvir = self.t1.shape
            fock = self.eris.fock
            foo = fock[:nocc,:nocc]
            fvv = fock[nocc:,nocc:]
            Hr2 +=  einsum('ac,jcb->jab',fvv,r2)
            Hr2 +=  einsum('bd,jad->jab',fvv,r2)
            Hr2 += -einsum('lj,lab->jab',foo,r2)
        elif self.ea_partition == 'full':
            Hr2 += self._eaccsd_diag_matrix2*r2
        else:
            Hr2 +=  einsum('ac,jcb->jab',imds.Lvv,r2)
            Hr2 +=  einsum('bd,jad->jab',imds.Lvv,r2)
            Hr2 += -einsum('lj,lab->jab',imds.Loo,r2)
            Hr2 += 2*einsum('lbdj,lad->jab',imds.Wovvo,r2)
            Hr2 +=  -einsum('lbjd,lad->jab',imds.Wovov,r2)
            Hr2 +=  -einsum('lajc,lcb->jab',imds.Wovov,r2)
            Hr2 +=  -einsum('lbcj,lca->jab',imds.Wovvo,r2)
            nvir = self.nmo-self.nocc
            for a in range(nvir):
                Hr2[:,a,:] += einsum('bcd,jcd->jb',imds.Wvvvv[a],r2)
            tmp = (2*einsum('klcd,lcd->k',imds.Woovv,r2)
                    -einsum('kldc,lcd->k',imds.Woovv,r2))
            Hr2 += -einsum('k,kjab->jab',tmp,self.t2)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def eaccsd_diag(self):
        if not self.made_ea_imds:
            if not hasattr(self,'imds'):
                self.imds = _IMDS(self)
            self.imds.make_ea()
            self.made_ea_imds = True
        imds = self.imds

        t1, t2 = self.t1, self.t2
        nocc, nvir = t1.shape

        fock = self.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]

        Hr1 = np.diag(imds.Lvv)
        Hr2 = np.zeros((nocc,nvir,nvir), t1.dtype)
        for a in range(nvir):
            if self.ea_partition != 'mp':
                _Wvvvva = np.array(imds.Wvvvv[a])
            for b in range(nvir):
                for j in range(nocc):
                    if self.ea_partition == 'mp':
                        Hr2[j,a,b] += fvv[a,a]
                        Hr2[j,a,b] += fvv[b,b]
                        Hr2[j,a,b] += -foo[j,j]
                    else:
                        Hr2[j,a,b] += imds.Lvv[a,a]
                        Hr2[j,a,b] += imds.Lvv[b,b]
                        Hr2[j,a,b] += -imds.Loo[j,j]
                        Hr2[j,a,b] += 2*imds.Wovvo[j,b,b,j]
                        Hr2[j,a,b] += -imds.Wovov[j,b,j,b]
                        Hr2[j,a,b] += -imds.Wovov[j,a,j,a]
                        Hr2[j,a,b] += -imds.Wovvo[j,b,b,j]*(a==b)
                        Hr2[j,a,b] += _Wvvvva[b,a,b]
                        Hr2[j,a,b] += -2*np.dot(imds.Woovv[:,j,a,b],t2[:,j,a,b])
                        Hr2[j,a,b] += np.dot(imds.Woovv[:,j,b,a],t2[:,j,a,b])

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ea(self,vector):
        nocc = self.nocc
        nvir = self.nmo - nocc
        r1 = vector[:nvir].copy()
        r2 = vector[nvir:].copy().reshape(nocc,nvir,nvir)
        return [r1,r2]

    def amplitudes_to_vector_ea(self,r1,r2):
        nocc = self.nocc
        nvir = self.nmo - nocc
        size = self.nea()
        vector = np.zeros(size, r1.dtype)
        vector[:nvir] = r1.copy()
        vector[nvir:] = r2.copy().reshape(nocc*nvir*nvir)
        return vector

    def eeccsd(self, nroots=1, koopmans=False, guess=None, partition=None):
        '''Calculate N-electron neutral excitations via EE-EOM-CCSD.

        Kwargs:
            nroots : int
                Number of roots (eigenvalues) requested
            partition : bool or str
                Use a matrix-partitioning for the doubles-doubles block.
                Can be None, 'mp' (Moller-Plesset, i.e. orbital energies on the diagonal),
                or 'full' (full diagonal elements).
            koopmans : bool
                Calculate Koopmans'-like (1p1h) excitations only, targeting via
                overlap.
            guess : list of ndarray
                List of guess vectors to use for targeting via overlap.
        '''
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        size = self.nee()
        nroots = min(nroots,size)
        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ee_partition = partition
        if partition == 'full':
            self._eeccsd_diag_matrix2 = self.vector_to_amplitudes_ee(self.eeccsd_diag())[1]

        nvir = self.nmo - self.nocc
        adiag = self.eeccsd_diag()
        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            guess = []
            idx = adiag.argsort()
            n = 0
            for i in idx:
                g = np.zeros(size)
                g[i] = 1.0
                if koopmans:
                    if np.linalg.norm(g[:self.nocc*nvir])**2 > 0.8:
                        guess.append(g)
                        n += 1
                else:
                    guess.append(g)
                    n += 1
                if n == nroots:
                    break

        def precond(r, e0, x0):
            return r/(e0-adiag+1e-12)

        eig = linalg_helper.eig
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return w[idx].real, v[:,idx].real, idx
            eee, evecs = eig(self.eeccsd_matvec, guess, precond, pick=pickeig,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=7)
        else:
            eee, evecs = eig(self.eeccsd_matvec, guess, precond,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=7)

        self.eee = eee.real

        if nroots == 1:
            eee, evecs = [self.eee], [evecs]
        for n, en, vn in zip(range(nroots), eee, evecs):
            logger.info(self, 'EE root %d E = %.16g  qpwt = %0.6g', 
                        n, en, np.linalg.norm(vn[:self.nocc*nvir])**2)
        log.timer('EE-CCSD', *cput0)
        if nroots == 1:
            return eee[0], evecs[0]
        else:
            return eee, evecs

    def eeccsd_matvec(self,vector):
        raise NotImplementedError

    def vector_to_amplitudes_ee(self,vector):
        nocc = self.nocc
        nvir = self.nmo - nocc
        r1 = vector[:nocc*nvir].copy().reshape((nocc,nvir))
        r2 = vector[nocc*nvir:].copy().reshape((nocc,nocc,nvir,nvir))
        return [r1,r2]

    def amplitudes_to_vector_ee(self,r1,r2):
        nocc = self.nocc
        nvir = self.nmo - nocc
        size = self.nee()
        vector = np.zeros(size, r1.dtype)
        vector[:nocc*nvir] = r1.copy().reshape(nocc*nvir)
        vector[nocc*nvir:] = r2.copy().reshape(nocc*nocc*nvir*nvir)
        return vector


class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore',
                 ao2mofn=ao2mo.outcore.general_iofree):
        cput0 = (time.clock(), time.time())
        moidx = numpy.ones(cc.mo_occ.size, dtype=numpy.bool)
        if isinstance(cc.frozen, (int, numpy.integer)):
            moidx[:cc.frozen] = False
        elif len(cc.frozen) > 0:
            moidx[numpy.asarray(cc.frozen)] = False
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = cc.mo_coeff[:,moidx]
        else:  # If mo_coeff is not canonical orbital
            self.mo_coeff = mo_coeff = mo_coeff[:,moidx]
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
        self.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))

        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
        mem_now = lib.current_memory()[0]

        log = logger.Logger(cc.stdout, cc.verbose)
        if (method == 'incore' and (mem_incore+mem_now < cc.max_memory)
            or cc.mol.incore_anyway):
            eri = ao2mofn(cc._scf.mol, (mo_coeff,mo_coeff,mo_coeff,mo_coeff), compact=0)
            if mo_coeff.dtype == np.float: eri = eri.real
            eri = eri.reshape((nmo,)*4)
            # <ij|kl> = (ik|jl)
            eri = eri.transpose(0,2,1,3)

            self.dtype = eri.dtype
            self.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
            self.ooov = eri[:nocc,:nocc,:nocc,nocc:].copy()
            self.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
            self.ovov = eri[:nocc,nocc:,:nocc,nocc:].copy()
            self.voov = eri[nocc:,:nocc,:nocc,nocc:].copy()
            self.vovv = eri[nocc:,:nocc,nocc:,nocc:].copy()
            self.vvvv = eri[nocc:,nocc:,nocc:,nocc:].copy()
        else:
            _tmpfile1 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self.feri1 = h5py.File(_tmpfile1.name)
            orbo = mo_coeff[:,:nocc]
            orbv = mo_coeff[:,nocc:]
            if mo_coeff.dtype == np.complex: ds_type = 'c16'
            else: ds_type = 'f8'
            self.oooo = self.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), ds_type)
            self.ooov = self.feri1.create_dataset('ooov', (nocc,nocc,nocc,nvir), ds_type)
            self.oovv = self.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), ds_type)
            self.ovov = self.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), ds_type)
            self.voov = self.feri1.create_dataset('voov', (nvir,nocc,nocc,nvir), ds_type)
            self.vovv = self.feri1.create_dataset('vovv', (nvir,nocc,nvir,nvir), ds_type)
            self.vvvv = self.feri1.create_dataset('vvvv', (nvir,nvir,nvir,nvir), ds_type)

            cput1 = time.clock(), time.time()
            # <ij|pq>  = (ip|jq)
            buf = ao2mofn(cc._scf.mol, (orbo,mo_coeff,orbo,mo_coeff), compact=0)
            if mo_coeff.dtype == np.float: buf = buf.real
            buf = buf.reshape((nocc,nmo,nocc,nmo)).transpose(0,2,1,3)
            cput1 = log.timer_debug1('transforming oopq', *cput1)
            self.dtype = buf.dtype
            self.oooo[:,:,:,:] = buf[:,:,:nocc,:nocc]
            self.ooov[:,:,:,:] = buf[:,:,:nocc,nocc:]
            self.oovv[:,:,:,:] = buf[:,:,nocc:,nocc:]

            cput1 = time.clock(), time.time()
            # <ia|pq> = (ip|aq)
            buf = ao2mofn(cc._scf.mol, (orbo,mo_coeff,orbv,mo_coeff), compact=0)
            if mo_coeff.dtype == np.float: buf = buf.real
            buf = buf.reshape((nocc,nmo,nvir,nmo)).transpose(0,2,1,3)
            cput1 = log.timer_debug1('transforming ovpq', *cput1)
            self.ovov[:,:,:,:] = buf[:,:,:nocc,nocc:]
            self.vovv[:,:,:,:] = buf[:,:,nocc:,nocc:].transpose(1,0,3,2)
            self.voov[:,:,:,:] = buf[:,:,nocc:,:nocc].transpose(1,0,3,2)

            _tmpfile2 = tempfile.NamedTemporaryFile()
            self.feri2 = h5py.File(_tmpfile2.name, 'w')
            ao2mo.full(cc.mol, orbv, self.feri2, max_memory=cc.max_memory, 
                             verbose=log, compact=False)
            vvvv_buf = self.feri2['eri_mo']
            for a in range(nvir):
                abrange = a*nvir + np.arange(nvir)
                self.vvvv[a,:,:,:] = np.array(vvvv_buf[abrange,:]).reshape((nvir,nvir,nvir)).transpose(1,0,2)

            cput1 = log.timer_debug1('transforming vvvv', *cput1)

        log.timer('CCSD integral transformation', *cput0)

class _IMDS:
    def __init__(self, cc):
        self.cc = cc
        self._made_shared_2e = False

    def _make_shared_1e(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.cc.stdout, self.cc.verbose)

        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris
        self.Loo = imd.Loo(t1,t2,eris)
        self.Lvv = imd.Lvv(t1,t2,eris)
        self.Fov = imd.cc_Fov(t1,t2,eris)

        log.timer('EOM-CCSD shared one-electron intermediates', *cput0)

    def _make_shared_2e(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.cc.stdout, self.cc.verbose)

        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris
        # 2 virtuals
        self.Wovov = imd.Wovov(t1,t2,eris)
        self.Wovvo = imd.Wovvo(t1,t2,eris)
        self.Woovv = eris.oovv

        log.timer('EOM-CCSD shared two-electron intermediates', *cput0)

    def make_ip(self):
        self._make_shared_1e()
        if self._made_shared_2e is False and self.cc.ip_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.cc.stdout, self.cc.verbose)

        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris

        # 0 or 1 virtuals
        if self.cc.ip_partition != 'mp':
            self.Woooo = imd.Woooo(t1,t2,eris)
        self.Wooov = imd.Wooov(t1,t2,eris)
        self.Wovoo = imd.Wovoo(t1,t2,eris)

        log.timer('EOM-CCSD IP intermediates', *cput0)

    def make_ea(self):
        self._make_shared_1e()
        if self._made_shared_2e is False and self.cc.ea_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.cc.stdout, self.cc.verbose)

        t1,t2,eris = self.cc.t1, self.cc.t2, self.cc.eris
        
        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1,t2,eris)
        if self.cc.ea_partition == 'mp' and not np.any(t1):
            self.Wvvvo = imd.Wvvvo(t1,t2,eris)
        else:
            self.Wvvvv = imd.Wvvvv(t1,t2,eris)
            self.Wvvvo = imd.Wvvvo(t1,t2,eris,self.Wvvvv)

        log.timer('EOM-CCSD EA intermediates', *cput0)

    def make_ee(self):
        raise NotImplementedError


def _mem_usage(nocc, nvir):
    incore = (nocc+nvir)**4
    # Roughly, factor of two for intermediates and factor of two 
    # for safety (temp arrays, copying, etc)
    incore *= 4
    # TODO: Improve incore estimate and add outcore estimate
    outcore = basic = incore
    return incore, outcore, basic


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol)
    print(mf.scf())

    mycc = RCCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.2133432712431435)
    e,v = mycc.ipccsd(nroots=3)
    print(e[0] - 0.4335604332073799)
    print(e[1] - 0.5187659896045407)
    print(e[2] - 0.6782876002229172)
