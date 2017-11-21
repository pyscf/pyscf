import time
import tempfile
from functools import reduce
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


#def update_amps(cc, t1, t2, eris):
#    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
#    time0 = time.clock(), time.time()
#    log = logger.Logger(cc.stdout, cc.verbose)
#    nocc, nvir = t1.shape
#    fock = eris.fock
#
#    fov = fock[:nocc,nocc:]
#    foo = fock[:nocc,:nocc]
#    fvv = fock[nocc:,nocc:]
#
#    mo_e = eris.fock.diagonal()
#    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
#    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
#
#    Foo = imd.cc_Foo(t1,t2,eris)
#    Fvv = imd.cc_Fvv(t1,t2,eris)
#    Fov = imd.cc_Fov(t1,t2,eris)
#    Loo = imd.Loo(t1,t2,eris)
#    Lvv = imd.Lvv(t1,t2,eris)
#    Woooo = imd.cc_Woooo(t1,t2,eris)
#    Wvvvv = imd.cc_Wvvvv(t1,t2,eris)
#    Wvoov = imd.cc_Wvoov(t1,t2,eris)
#    Wvovo = imd.cc_Wvovo(t1,t2,eris)
#
#    # Move energy terms to the other side
#    Foo -= np.diag(np.diag(foo))
#    Fvv -= np.diag(np.diag(fvv))
#    Loo -= np.diag(np.diag(foo))
#    Lvv -= np.diag(np.diag(fvv))
#
#    # T1 equation
#    t1new = np.array(fov).conj()
#    t1new += -2*einsum('kc,ka,ic->ia',fov,t1,t1)
#    eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
#    t1new +=   einsum('ac,ic->ia',Fvv,t1)
#    t1new +=  -einsum('ki,ka->ia',Foo,t1)
#    t1new += 2*einsum('kc,kica->ia',Fov,t2)
#    t1new +=  -einsum('kc,ikca->ia',Fov,t2)
#    t1new +=   einsum('kc,ic,ka->ia',Fov,t1,t1)
#    t1new += 2*einsum('iack,kc->ia',eris.ovvo,t1)
#    t1new +=  -einsum('kiac,kc->ia',eris.oovv,t1)
#    t1new += 2*einsum('kdac,ikcd->ia',eris_ovvv,t2)
#    t1new +=  -einsum('kcad,ikcd->ia',eris_ovvv,t2)
#    t1new += 2*einsum('kdac,ic,kd->ia',eris_ovvv,t1,t1)
#    t1new +=  -einsum('kcad,ic,kd->ia',eris_ovvv,t1,t1)
#    t1new += -2*einsum('kilc,klac->ia',eris.ooov,t2)
#    t1new +=  einsum('likc,klac->ia',eris.ooov,t2)
#    t1new += -2*einsum('kilc,ka,lc->ia',eris.ooov,t1,t1)
#    t1new +=  einsum('likc,ka,lc->ia',eris.ooov,t1,t1)
#
#    # T2 equation
#    t2new = np.array(eris.ovov).transpose(0,2,1,3).conj().copy()
#    t2new += einsum('klij,klab->ijab',Woooo,t2)
#    t2new += einsum('klij,ka,lb->ijab',Woooo,t1,t1)
#    for a in range(nvir):
#        Wvvvv_a = np.array(Wvvvv[a]).copy()
#        t2new[:,:,a,:] += einsum('bcd,ijcd->ijb',Wvvvv_a,t2)
#        t2new[:,:,a,:] += einsum('bcd,ic,jd->ijb',Wvvvv_a,t1,t1)
#    tmp = einsum('ac,ijcb->ijab',Lvv,t2)
#    t2new += (tmp + tmp.transpose(1,0,3,2))
#    tmp = einsum('ki,kjab->ijab',Loo,t2)
#    t2new -= (tmp + tmp.transpose(1,0,3,2))
#    tmp2 = np.array(eris_ovvv).transpose(1,3,0,2).conj() \
#            - einsum('kibc,ka->abic',eris.oovv,t1)
#    tmp = einsum('abic,jc->ijab',tmp2,t1)
#    t2new += (tmp + tmp.transpose(1,0,3,2))
#    tmp2 = np.array(eris.ooov).transpose(3,1,2,0).conj() \
#            + einsum('iack,jc->akij',eris.ovvo,t1)
#    tmp = einsum('akij,kb->ijab',tmp2,t1)
#    t2new -= (tmp + tmp.transpose(1,0,3,2))
#    tmp = 2*einsum('akic,kjcb->ijab',Wvoov,t2) - einsum('akci,kjcb->ijab',Wvovo,t2)
#    t2new += (tmp + tmp.transpose(1,0,3,2))
#    tmp = einsum('akic,kjbc->ijab',Wvoov,t2)
#    t2new -= (tmp + tmp.transpose(1,0,3,2))
#    tmp = einsum('bkci,kjac->ijab',Wvovo,t2)
#    t2new -= (tmp + tmp.transpose(1,0,3,2))
#
#    t1new /= eia
#    t2new /= eijab
#
#    time0 = log.timer_debug1('update t1 t2', *time0)
#
#    return t1new, t2new


def energy(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    e = 2*einsum('ia,ia', fock[:nocc,nocc:], t1)
    t1t1 = einsum('ia,jb->ijab',t1,t1)
    tau = t2 + t1t1
    e += einsum('ijab,iajb', 2*tau, eris.ovov)
    e += einsum('ijab,ibja',  -tau, eris.ovov)
    return e.real


class RCCSD(ccsd.CCSD):
    '''restricted CCSD with IP-EOM, EA-EOM, EE-EOM, and SF-EOM capabilities

    Ground-state CCSD is performed in optimized ccsd.CCSD and EOM is performed here.
    '''
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.max_space = 20
        self._keys = self._keys.union(['max_space'])

    def dump_flags(self):
        ccsd.CCSD.dump_flags(self)
        return self

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        mo_e = eris.fock.diagonal()
        nocc = self.nocc
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
        eris_oovv = np.array(eris.ovov).transpose(0,2,1,3)
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
                logger.note(self, 'CCSD not converged')
        if self._scf.e_tot == 0:
            logger.note(self, 'E_corr = %.16g', self.e_corr)
        else:
            logger.note(self, 'E(%s) = %.16g  E_corr = %.16g',
                        cctyp, self.e_tot, self.e_corr)
        return self.e_corr, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)

#    def update_amps(self, t1, t2, eris):
#        return update_amps(self, t1, t2, eris)

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

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None, partition=None):
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

        if left:
            matvec = lambda xs: [self.lipccsd_matvec(x) for x in xs]
        else:
            matvec = lambda xs: [self.ipccsd_matvec(x) for x in xs]
        eig = linalg_helper.davidson_nosym1
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return w[idx].real, v[:,idx].real, idx
            conv, eip, evecs = eig(matvec, guess, precond, pick=pickeig,
                                   tol=self.conv_tol, max_cycle=self.max_cycle,
                                   max_space=self.max_space, nroots=nroots, verbose=self.verbose)
        else:
            conv, eip, evecs = eig(matvec, guess, precond,
                                   tol=self.conv_tol, max_cycle=self.max_cycle,
                                   max_space=self.max_space, nroots=nroots, verbose=self.verbose)

        self.eip = np.array(eip).real

        for n, en, vn, convn in zip(range(nroots), eip, evecs, conv):
            logger.info(self, 'IP root %d E = %.16g  qpwt = %.6g  conv = %s',
                        n, en, np.linalg.norm(vn[:self.nocc])**2, convn)
        log.timer('IP-CCSD', *cput0)
        if nroots == 1:
            return eip[0], evecs[0]
        else:
            return eip, evecs

    def ipccsd_matvec(self, vector):
        # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition)
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

    def lipccsd_matvec(self, vector):
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition)
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ip(vector)

        # 1h-1h block
        Hr1 = -einsum('ki,i->k',imds.Loo,r1)
        #1h-2h1p block
        Hr1 += -einsum('kbij,ijb->k',imds.Wovoo,r2)

        # 2h1p-1h block
        Hr2 = -einsum('kd,l->kld',imds.Fov,r1)
        Hr2 += 2.*einsum('ld,k->kld',imds.Fov,r1)
        Hr2 += -einsum('klid,i->kld',2.*imds.Wooov-imds.Wooov.transpose(1,0,2,3),r1)
        # 2h1p-2h1p block
        if self.ip_partition == 'mp':
            nocc, nvir = self.t1.shape
            fock = self.eris.fock
            foo = fock[:nocc,:nocc]
            fvv = fock[nocc:,nocc:]
            Hr2 += einsum('bd,klb->kld',fvv,r2)
            Hr2 += -einsum('ki,ild->kld',foo,r2)
            Hr2 += -einsum('lj,kjd->kld',foo,r2)
        elif self.ip_partition == 'full':
            Hr2 += self._ipccsd_diag_matrix2*r2
        else:
            Hr2 += einsum('bd,klb->kld',imds.Lvv,r2)
            Hr2 += -einsum('ki,ild->kld',imds.Loo,r2)
            Hr2 += -einsum('lj,kjd->kld',imds.Loo,r2)
            Hr2 += einsum('lbdj,kjb->kld',2.*imds.Wovvo-imds.Wovov.transpose(0,1,3,2),r2)
            Hr2 += -einsum('kbdj,ljb->kld',imds.Wovvo,r2)
            Hr2 += einsum('klij,ijd->kld',imds.Woooo,r2)
            Hr2 += -einsum('kbid,ilb->kld',imds.Wovov,r2)
            tmp = einsum('ijcb,ijb->c',self.t2,r2)
            Hr2 += -einsum('lkdc,c->kld',2.*imds.Woovv-imds.Woovv.transpose(1,0,2,3),tmp)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def ipccsd_diag(self):
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition)
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

    def ipccsd_star(self, ipccsd_evals, ipccsd_evecs, lipccsd_evecs):
        assert(self.ip_partition == None)
        t1,t2,eris = self.t1, self.t2, self.eris
        fock = eris.fock
        nocc = self.nocc
        nvir = self.nmo - nocc

        fov = fock[:nocc,nocc:]
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]

        oovv = _cp(eris.ovov).transpose(0,2,1,3)
        eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,-1)).reshape(nocc,nvir,nvir,nvir)
        ovvv = eris_ovvv.transpose(0,2,1,3)
        ovov = _cp(eris.oovv).transpose(0,2,1,3)
        ovvo = _cp(eris.ovvo).transpose(0,2,1,3)
        ooov = _cp(eris.ooov).transpose(0,2,1,3)
        vooo = ooov.conj().transpose(3,2,1,0)
        vvvo = _cp(ovvv).conj().transpose(3,2,1,0)
        oooo = _cp(eris.oooo).transpose(0,2,1,3)

        eijkab = np.zeros((nocc,nocc,nocc,nvir,nvir))
        for i,j,k in lib.cartesian_prod([range(nocc),range(nocc),range(nocc)]):
            for a,b in lib.cartesian_prod([range(nvir),range(nvir)]):
                eijkab[i,j,k,a,b] = foo[i,i] + foo[j,j] + foo[k,k] - fvv[a,a] - fvv[b,b]

        ipccsd_evecs  = np.array(ipccsd_evecs)
        lipccsd_evecs = np.array(lipccsd_evecs)
        e = []
        for _eval, _evec, _levec in zip(ipccsd_evals, ipccsd_evecs, lipccsd_evecs):
            l1,l2 = self.vector_to_amplitudes_ip(_levec)
            r1,r2 = self.vector_to_amplitudes_ip(_evec)
            ldotr = np.dot(l1,r1) + np.dot(l2.ravel(),r2.ravel())
            l1 /= ldotr
            l2 /= ldotr
            l2 = 1./3*(l2 + 2.*l2.transpose(1,0,2))

            _eijkab = eijkab + _eval
            _eijkab = 1./_eijkab

            lijkab = 0.5*einsum('ijab,k->ijkab',oovv,l1)
            lijkab += einsum('ieab,jke->ijkab',ovvv,l2)
            lijkab += -einsum('kjmb,ima->ijkab',ooov,l2)
            lijkab += -einsum('ijmb,mka->ijkab',ooov,l2)
            lijkab = lijkab + lijkab.transpose(1,0,2,4,3)

            tmp = einsum('mbke,m->bke',ovov,r1)
            rijkab = -einsum('bke,ijae->ijkab',tmp,t2)
            tmp = einsum('mbej,m->bej',ovvo,r1)
            rijkab += -einsum('bej,ikae->ijkab',tmp,t2)
            tmp = einsum('mnjk,n->mjk',oooo,r1)
            rijkab += einsum('mjk,imab->ijkab',tmp,t2)
            rijkab += einsum('baei,kje->ijkab',vvvo,r2)
            rijkab += -einsum('bmjk,mia->ijkab',vooo,r2)
            rijkab += -einsum('bmji,kma->ijkab',vooo,r2)
            rijkab = rijkab + rijkab.transpose(1,0,2,4,3)

            lijkab = 4.*lijkab \
                   - 2.*lijkab.transpose(1,0,2,3,4) \
                   - 2.*lijkab.transpose(2,1,0,3,4) \
                   - 2.*lijkab.transpose(0,2,1,3,4) \
                   + 1.*lijkab.transpose(1,2,0,3,4) \
                   + 1.*lijkab.transpose(2,0,1,3,4)

            deltaE = 0.5*einsum('ijkab,ijkab,ijkab',lijkab,rijkab,_eijkab)
            deltaE = deltaE.real
            logger.info(self, "Exc. energy, delta energy = %16.12f, %16.12f",
                        _eval+deltaE, deltaE)
            e.append(_eval+deltaE)
        return e

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None, partition=None):
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

        eig = linalg_helper.davidson_nosym1
        if left:
            matvec = lambda xs: [self.leaccsd_matvec(x) for x in xs]
        else:
            matvec = lambda xs: [self.eaccsd_matvec(x) for x in xs]
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return w[idx].real, v[:,idx].real, idx
            conv, eea, evecs = eig(matvec, guess, precond, pick=pickeig,
                                   tol=self.conv_tol, max_cycle=self.max_cycle,
                                   max_space=self.max_space, nroots=nroots,
                                   verbose=self.verbose)
        else:
            conv, eea, evecs = eig(matvec, guess, precond,
                                   tol=self.conv_tol, max_cycle=self.max_cycle,
                                   max_space=self.max_space, nroots=nroots,
                                   verbose=self.verbose)

        self.eea = np.array(eea).real

        nvir = self.nmo - self.nocc
        for n, en, vn, convn in zip(range(nroots), eea, evecs, conv):
            logger.info(self, 'EA root %d E = %.16g  qpwt = %.6g  conv = %s',
                        n, en, np.linalg.norm(vn[:nvir])**2, convn)
        log.timer('EA-CCSD', *cput0)
        if nroots == 1:
            return eea[0], evecs[0]
        else:
            return eea, evecs

    def eaccsd_matvec(self,vector):
        # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition)
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ea(vector)

        # Eq. (30)
        # 1p-1p block
        Hr1 =  einsum('ac,c->a',imds.Lvv,r1)
        # 1p-2p1h block
        Hr1 += einsum('ld,lad->a',2.*imds.Fov,r2)
        Hr1 += einsum('ld,lda->a',  -imds.Fov,r2)
        Hr1 += einsum('alcd,lcd->a',2.*imds.Wvovv-imds.Wvovv.transpose(0,1,3,2),r2)
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
            Hr2 += einsum('lbdj,lad->jab',2.*imds.Wovvo-imds.Wovov.transpose(0,1,3,2),r2)
            Hr2 += -einsum('lajc,lcb->jab',imds.Wovov,r2)
            Hr2 += -einsum('lbcj,lca->jab',imds.Wovvo,r2)
            nvir = self.nmo-self.nocc
            for a in range(nvir):
                Hr2[:,a,:] += einsum('bcd,jcd->jb',imds.Wvvvv[a],r2)
            tmp = einsum('klcd,lcd->k',2.*imds.Woovv-imds.Woovv.transpose(0,1,3,2),r2)
            Hr2 += -einsum('k,kjab->jab',tmp,self.t2)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def leaccsd_matvec(self,vector):
        # Note this is not the same left EA equations used by Nooijen and Bartlett.
        # Small changes were made so that the same type L2 basis was used for both the
        # left EA and left IP equations.  You will note more similarity for these
        # equations to the left IP equations than for the left EA equations by Nooijen.
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition)
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ea(vector)

        # Eq. (30)
        # 1p-1p block
        Hr1 = einsum('ac,a->c',imds.Lvv,r1)
        # 1p-2p1h block
        Hr1 += einsum('abcj,jab->c',imds.Wvvvo,r2)
        # Eq. (31)
        # 2p1h-1p block
        Hr2 = 2.*einsum('c,ld->lcd',r1,imds.Fov)
        Hr2 +=  -einsum('d,lc->lcd',r1,imds.Fov)
        Hr2 += einsum('a,alcd->lcd',r1,2.*imds.Wvovv-imds.Wvovv.transpose(0,1,3,2))
        # 2p1h-2p1h block
        if self.ea_partition == 'mp':
            nocc, nvir = self.t1.shape
            fock = self.eris.fock
            foo = fock[:nocc,:nocc]
            fvv = fock[nocc:,nocc:]
            Hr2 += einsum('lad,ac->lcd',r2,fvv)
            Hr2 += einsum('lcb,bd->lcd',r2,fvv)
            Hr2 += -einsum('jcd,lj->lcd',r2,foo)
        elif self.ea_partition == 'full':
            Hr2 += self._eaccsd_diag_matrix2*r2
        else:
            Hr2 += einsum('lad,ac->lcd',r2,imds.Lvv)
            Hr2 += einsum('lcb,bd->lcd',r2,imds.Lvv)
            Hr2 += -einsum('jcd,lj->lcd',r2,imds.Loo)
            Hr2 += einsum('jcb,lbdj->lcd',r2,2.*imds.Wovvo-imds.Wovov.transpose(0,1,3,2))
            Hr2 += -einsum('lajc,jab->lcb',imds.Wovov,r2)
            Hr2 += -einsum('lbcj,jab->lca',imds.Wovvo,r2)
            nvir = self.nmo-self.nocc
            for a in range(nvir):
                Hr2 += einsum('lb,bcd->lcd',r2[:,a,:],imds.Wvvvv[a])
            tmp = einsum('ijcb,ibc->j',self.t2,r2)
            Hr2 += -einsum('kjfe,j->kef',2.*imds.Woovv-imds.Woovv.transpose(0,1,3,2),tmp)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def eaccsd_diag(self):
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition)
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

    def eaccsd_star(self, eaccsd_evals, eaccsd_evecs, leaccsd_evecs):
        assert(self.ea_partition == None)
        t1,t2,eris = self.t1, self.t2, self.eris
        fock = eris.fock
        nocc = self.nocc
        nvir = self.nmo - nocc

        fov = fock[:nocc,nocc:]
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]

        oovv = _cp(eris.ovov).transpose(0,2,1,3)
        eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,-1)).reshape(nocc,nvir,nvir,nvir)
        ovvv = _cp(eris_ovvv).transpose(0,2,1,3)
        vvov = ovvv.conj().transpose(2,3,0,1)
        ooov = _cp(eris.ooov).transpose(0,2,1,3)
        vooo = ooov.conj().transpose(3,2,1,0)
        ovov = _cp(eris.oovv).transpose(0,2,1,3)
        vvvv = ao2mo.restore(1,np.asarray(eris.vvvv),nvir).transpose(0,2,1,3)
        ovvo = _cp(eris.ovvo).transpose(0,2,1,3)

        eijabc = np.zeros((nocc,nocc,nvir,nvir,nvir))
        for i,j in lib.cartesian_prod([range(nocc),range(nocc)]):
            for a,b,c in lib.cartesian_prod([range(nvir),range(nvir),range(nvir)]):
                eijabc[i,j,a,b,c] = foo[i,i] + foo[j,j] - fvv[a,a] - fvv[b,b] - fvv[c,c]

        eaccsd_evecs  = np.array(eaccsd_evecs)
        leaccsd_evecs = np.array(leaccsd_evecs)
        e = []
        for _eval, _evec, _levec in zip(eaccsd_evals, eaccsd_evecs, leaccsd_evecs):
            l1,l2 = self.vector_to_amplitudes_ea(_levec)
            r1,r2 = self.vector_to_amplitudes_ea(_evec)
            ldotr = np.dot(l1,r1) + np.dot(l2.ravel(),r2.ravel())
            l1 /= ldotr
            l2 /= ldotr
            l2 = 1./3*(1.*l2 + 2.*l2.transpose(0,2,1))
            r2 = r2.transpose(0,2,1)

            _eijabc = eijabc + _eval
            _eijabc = 1./_eijabc

            lijabc = -0.5*einsum('c,ijab->ijabc',l1,oovv)
            lijabc += einsum('jima,mbc->ijabc',ooov,l2)
            lijabc -= einsum('ieab,jec->ijabc',ovvv,l2)
            lijabc -= einsum('jebc,iae->ijabc',ovvv,l2)
            lijabc = lijabc + lijabc.transpose(1,0,3,2,4)

            tmp = einsum('bcef,f->bce',vvvv,r1)
            rijabc = -einsum('bce,ijae->ijabc',tmp,t2)
            tmp = einsum('mcje,e->mcj',ovov,r1)
            rijabc += einsum('mcj,imab->ijabc',tmp,t2)
            tmp = einsum('mbej,e->mbj',ovvo,r1)
            rijabc += einsum('mbj,imac->ijabc',tmp,t2)
            rijabc += einsum('amij,mbc->ijabc',vooo,r2)
            rijabc += -einsum('bcje,iae->ijabc',vvov,r2)
            rijabc += -einsum('abie,jec->ijabc',vvov,r2)
            rijabc = rijabc + rijabc.transpose(1,0,3,2,4)

            lijabc =  4.*lijabc \
                    - 2.*lijabc.transpose(0,1,3,2,4) \
                    - 2.*lijabc.transpose(0,1,4,3,2) \
                    - 2.*lijabc.transpose(0,1,2,4,3) \
                    + 1.*lijabc.transpose(0,1,3,4,2) \
                    + 1.*lijabc.transpose(0,1,4,2,3)
            deltaE = 0.5*einsum('ijabc,ijabc,ijabc',lijabc,rijabc,_eijabc)
            deltaE = deltaE.real
            logger.info(self, "Exc. energy, delta energy = %16.12f, %16.12f",
                        _eval+deltaE, deltaE)
            e.append(_eval+deltaE)
        return e

    #TODO: double spin-flip EOM-EE
    def eeccsd(self, nroots=1, koopmans=False, guess=None):
        '''Calculate N-electron neutral excitations via EE-EOM-CCSD.

        Kwargs:
            nroots : int
                Number of roots (eigenvalues) requested
            koopmans : bool
                Calculate Koopmans'-like (1p1h) excitations only, targeting via
                overlap.
            guess : list of ndarray
                List of guess vectors to use for targeting via overlap.
        '''

        spinvec_size = self.nee()
        nroots = min(nroots, spinvec_size)

        diag_eeS, diag_eeT, diag_sf = self.eeccsd_diag()
        guess_eeS = []
        guess_eeT = []
        guess_sf = []
        if guess and guess[0].size == spinvec_size:
            raise NotImplementedError
        elif guess:
            for g in guess:
                if g is None:
                    pass
                elif g.size == diag_eeS.size:
                    guess_eeS.append(g)
                elif g.size == diag_eeT.size:
                    guess_eeT.append(g)
                else:
                    guess_sf.append(g)
            nroots_eeS = len(guess_eeS)
            nroots_eeT = len(guess_eeT)
            nroots_sf = len(guess_sf)
        else:
            deeS = np.sort(diag_eeS)[:nroots]
            deeT = np.sort(diag_eeT)[:nroots]
            dsf = np.sort(diag_sf)[:nroots]
            dmax = np.sort(np.hstack([deeS,deeT,dsf,dsf]))[nroots-1]
            nroots_eeS = np.count_nonzero(deeS <= dmax)
            nroots_eeT = np.count_nonzero(deeT <= dmax)
            nroots_sf = np.count_nonzero(dsf <= dmax)
            guess_eeS = guess_eeT = guess_sf = None

        e0 = e1 = e2 = []
        v0 = v1 = v2 = []
        if nroots_eeS > 0:
            e0, v0 = self.eomee_ccsd_singlet(nroots_eeS, koopmans, guess_eeS, diag_eeS)
            if nroots_eeS == 1:
                e0, v0 = [e0], [v0]
        if nroots_eeT > 0:
            e2, v2 = self.eomee_ccsd_triplet(nroots_eeT, koopmans, guess_eeT, diag_eeT)
            if nroots_eeT == 1:
                e2, v2 = [e2], [v2]
        if nroots_sf > 0:
            e1, v1 = self.eomsf_ccsd(nroots_sf, koopmans, guess_sf, diag_sf)
            if nroots_sf == 1:
                e1, v1 = [e1], [v1]
            # The associated solution
            nocc = self.nocc
            nvir = self.nmo - nocc
            e1 = list(e1) + list(e1)
            v1 = list(v1) + [None] * len(v1)
        e = np.hstack([e0,e2,e1])
        v = v0 + v2 + v1
        if nroots == 1:
            return e[0], v[0]
        else:
            idx = e.argsort()
            return e[idx], [v[x] for x in idx]


    def eomee_ccsd_singlet(self, nroots=1, koopmans=False, guess=None, diag=None):
        cput0 = (time.clock(), time.time())
        if diag is None:
            diag = self.eeccsd_diag()[0]
        nocc = self.nocc
        nmo = self.nmo

        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == diag.size
        else:
            idx = diag.argsort()
            guess = []
            if koopmans:
                n = 0
                for i in idx:
                    g = np.zeros_like(diag)
                    g[i] = 1.0
                    t1, t2 = self.vector_to_amplitudes(g, nmo, nocc)
                    if np.linalg.norm(t1) > .9:
                        guess.append(g)
                        n += 1
                        if n == nroots:
                            break
            else:
                for i in idx[:nroots]:
                    g = np.zeros_like(diag)
                    g[i] = 1.0
                    guess.append(g)

        def precond(r, e0, x0):
            return r/(e0-diag+1e-12)

        eig = linalg_helper.davidson_nosym1
        matvec = lambda xs: [self.eomee_ccsd_matvec_singlet(x) for x in xs]
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return w[idx].real, v[:,idx].real, idx
            conv, eee, evecs = eig(matvec, guess, precond, pick=pickeig,
                                   tol=self.conv_tol, max_cycle=self.max_cycle,
                                   max_space=self.max_space, nroots=nroots,
                                   verbose=self.verbose)
        else:
            conv, eee, evecs = eig(matvec, guess, precond,
                                   tol=self.conv_tol, max_cycle=self.max_cycle,
                                   max_space=self.max_space, nroots=nroots,
                                   verbose=self.verbose)

        self.eee = np.array(eee).real

        for n, en, vn, convn in zip(range(nroots), eee, evecs, conv):
            t1, t2 = self.vector_to_amplitudes(vn, nmo, nocc)
            logger.info(self, 'EOM-EE singlet root %d E = %.16g  qpwt = %.6g  conv = %s',
                        n, en, np.linalg.norm(t1)**2, convn)
        logger.timer(self, 'EOM-EE-CCSD singlet', *cput0)
        if nroots == 1:
            return eee[0], evecs[0]
        else:
            return eee, evecs

    def eomee_ccsd_triplet(self, nroots=1, koopmans=False, guess=None, diag=None):
        cput0 = (time.clock(), time.time())
        if diag is None:
            diag = self.eeccsd_diag()[1]
        nocc = self.nocc
        nvir = self.nmo - nocc

        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == diag.size
        else:
            idx = diag.argsort()
            guess = []
            if koopmans:
                n = 0
                for i in idx:
                    g = np.zeros_like(diag)
                    g[i] = 1.0
                    t1, t2 = self.vector_to_amplitudes_triplet(g, nocc, nvir)
                    if np.linalg.norm(t1) > .9:
                        guess.append(g)
                        n += 1
                        if n == nroots:
                            break
            else:
                for i in idx[:nroots]:
                    g = np.zeros_like(diag)
                    g[i] = 1.0
                    guess.append(g)

        def precond(r, e0, x0):
            return r/(e0-diag+1e-12)

        eig = linalg_helper.davidson_nosym1
        matvec = lambda xs: [self.eomee_ccsd_matvec_triplet(x) for x in xs]
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return w[idx].real, v[:,idx].real, idx
            conv, eee, evecs = eig(matvec, guess, precond, pick=pickeig,
                                   tol=self.conv_tol, max_cycle=self.max_cycle,
                                   max_space=self.max_space, nroots=nroots,
                                   verbose=self.verbose)
        else:
            conv, eee, evecs = eig(matvec, guess, precond,
                                   tol=self.conv_tol, max_cycle=self.max_cycle,
                                   max_space=self.max_space, nroots=nroots,
                                   verbose=self.verbose)

        self.eee = np.array(eee).real

        for n, en, vn, convn in zip(range(nroots), eee, evecs, conv):
            t1, t2 = self.vector_to_amplitudes_triplet(vn, nocc, nvir)
            logger.info(self, 'EOM-EE triplet root %d E = %.16g  qpwt = %.6g  conv = %s',
                        n, en, np.linalg.norm(t1)**2, convn)
        logger.timer(self, 'EOM-EE-CCSD triplet', *cput0)
        if nroots == 1:
            return eee[0], evecs[0]
        else:
            return eee, evecs

    def eomsf_ccsd(self, nroots=1, koopmans=False, guess=None, diag=None):
        cput0 = (time.clock(), time.time())
        if diag is None:
            diag = self.eeccsd_diag()[2]
        nocc = self.nocc
        nvir = self.nmo - nocc

        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == diag.size
        else:
            idx = diag.argsort()
            guess = []
            if koopmans:
                n = 0
                for i in idx:
                    g = np.zeros_like(diag)
                    g[i] = 1.0
                    t1, t2 = self.vector_to_amplitudes_eomsf(g, nocc, nvir)
                    if np.linalg.norm(t1) > .9:
                        guess.append(g)
                        n += 1
                        if n == nroots:
                            break
            else:
                for i in idx[:nroots]:
                    g = np.zeros_like(diag)
                    g[i] = 1.0
                    guess.append(g)

        def precond(r, e0, x0):
            return r/(e0-diag+1e-12)

        eig = linalg_helper.davidson_nosym1
        matvec = lambda xs: [self.eomsf_ccsd_matvec(x) for x in xs]
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return w[idx].real, v[:,idx].real, idx
            conv, eee, evecs = eig(matvec, guess, precond, pick=pickeig,
                                   tol=self.conv_tol, max_cycle=self.max_cycle,
                                   max_space=self.max_space, nroots=nroots,
                                   verbose=self.verbose)
        else:
            conv, eee, evecs = eig(matvec, guess, precond,
                                   tol=self.conv_tol, max_cycle=self.max_cycle,
                                   max_space=self.max_space, nroots=nroots,
                                   verbose=self.verbose)

        self.eee = np.array(eee).real

# EOM-SF solutions are degenerated.  Here the solution (r1ab, r2baaa, r2aaba)
# is computed.  The associated solution is
# (r1ba, r2abbb, r2bbab) = (-r1ab,-r2baaa, r2aaba)
        for n, en, vn, convn in zip(range(nroots), eee, evecs, conv):
            t1, t2 = self.vector_to_amplitudes_eomsf(vn, nocc, nvir)
            qpwt = np.linalg.norm(t1)**2
            logger.info(self, 'EOM-SF root %d E = %.16g  qpwt = %.6g  conv = %s',
                        n, en, qpwt, convn)
        logger.timer(self, 'EOM-SF-CCSD', *cput0)
        if nroots == 1:
            return eee[0], evecs[0]
        else:
            return eee, evecs

    def eomee_ccsd_matvec_singlet(self, vector):
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ee_imds:
            self.imds.make_ee()
        imds = self.imds

        r1, r2 = self.vector_to_amplitudes(vector)
        t1, t2, eris = self.t1, self.t2, self.eris
        nocc, nvir = t1.shape

        rho = r2*2 - r2.transpose(0,1,3,2)
        Hr1  = lib.einsum('ae,ie->ia', imds.Fvv, r1)
        Hr1 -= lib.einsum('mi,ma->ia', imds.Foo, r1)
        Hr1 += np.einsum('me,imae->ia',imds.Fov, rho)

        Hr2 = lib.einsum('mnij,mnab->ijab', imds.woOoO, r2) * .5
        Hr2+= lib.einsum('be,ijae->ijab', imds.Fvv   , r2)
        Hr2-= lib.einsum('mj,imab->ijab', imds.Foo   , r2)

        tau2 = make_tau(r2, r1, t1, fac=2)
        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
        #:Hr1 += lib.einsum('mfae,imef->ia', eris_ovvv, rho)
        #:tmp = lib.einsum('meaf,ijef->maij', eris_ovvv, tau2)
        #:Hr2 -= lib.einsum('ma,mbij->ijab', t1, tmp)
        #:tmp  = lib.einsum('meaf,me->af', eris_ovvv, r1) * 2
        #:tmp -= lib.einsum('mfae,me->af', eris_ovvv, r1)
        mem_now = lib.current_memory()[0]
        max_memory = lib.param.MAX_MEMORY - mem_now
        blksize = max(int(max_memory*1e6/8/(nvir**3*3)), 2)
        for p0,p1 in lib.prange(0, nocc, blksize):
            ovvv = np.asarray(eris.ovvv[p0:p1]).reshape((p1-p0)*nvir,-1)
            ovvv = lib.unpack_tril(ovvv).reshape(-1,nvir,nvir,nvir)
            Hr1 += lib.einsum('mfae,imef->ia', ovvv, rho[:,p0:p1])
            tmp = lib.einsum('meaf,ijef->maij', ovvv, tau2)
            Hr2 -= lib.einsum('ma,mbij->ijab', t1[p0:p1], tmp)
            tmp  = lib.einsum('meaf,me->af', ovvv, r1[p0:p1]) * 2
            tmp -= lib.einsum('mfae,me->af', ovvv, r1[p0:p1])
            Hr2 += lib.einsum('af,ijfb->ijab', tmp, t2)
            ovvv = tmp = None
        Hr2 -= lib.einsum('mbij,ma->ijab', imds.woVoO, r1)

        Hr1-= lib.einsum('mnie,mnae->ia', imds.woOoV, rho)
        tmp = lib.einsum('nmie,me->ni', imds.woOoV, r1) * 2
        tmp-= lib.einsum('mnie,me->ni', imds.woOoV, r1)
        Hr2 -= lib.einsum('ni,njab->ijab', tmp, t2)
        tmp = None
        for p0, p1 in lib.prange(0, nvir, nocc):
            Hr2 += lib.einsum('ejab,ie->ijab', np.asarray(imds.wvOvV[p0:p1]), r1[:,p0:p1])

        oVVo = np.asarray(imds.woVVo)
        tmp = lib.einsum('mbej,imea->jiab', oVVo, r2)
        Hr2 += tmp
        Hr2 += tmp.transpose(0,1,3,2) * .5
        oVvO = np.asarray(imds.woVvO) + oVVo * .5
        oVVo = tmp = None
        Hr1 += np.einsum('maei,me->ia', oVvO, r1) * 2
        Hr2 += lib.einsum('mbej,imae->ijab', oVvO, rho)
        oVvO = None

        eris_ovov = np.asarray(eris.ovov)
        tau2 = make_tau(r2, r1, t1, fac=2)
        tmp = lib.einsum('menf,ijef->mnij', eris_ovov, tau2)
        tau2 = None
        tau = make_tau(t2, t1, t1)
        Hr2 += lib.einsum('mnij,mnab->ijab', tmp, tau) * .5
        tau = tmp = None

        tmp = lib.einsum('nemf,imef->ni', eris_ovov, rho)
        Hr1 -= lib.einsum('na,ni->ia', t1, tmp)
        Hr2 -= lib.einsum('mj,miab->ijba', tmp, t2)
        tmp = None

        tmp  = np.einsum('mfne,mf->en', eris_ovov, r1) * 2
        tmp -= np.einsum('menf,mf->en', eris_ovov, r1)
        tmp  = np.einsum('en,nb->eb', tmp, t1)
        tmp += lib.einsum('menf,mnbf->eb', eris_ovov, rho)
        Hr2 -= lib.einsum('eb,ijea->jiab', tmp, t2)
        tmp = eris_ovov = rho = None

        #:eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv),t1.shape[1])
        #:Hr2 += lib.einsum('ijef,aebf->ijab', tau2, eris_vvvv) * .5
        tau2 = make_tau(r2, r1, t1, fac=2)
        _add_vvvv_(self, tau2, eris, Hr2)
        tau2 = None

        Hr2 = Hr2 + Hr2.transpose(1,0,3,2)
        vector = self.amplitudes_to_vector(Hr1, Hr2)
        return vector

    def eomee_ccsd_matvec_triplet(self, vector):
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ee_imds:
            self.imds.make_ee()
        imds = self.imds

        r1, r2 = self.vector_to_amplitudes_triplet(vector)
        r2aa, r2ab = r2
        t1, t2, eris = self.t1, self.t2, self.eris
        nocc, nvir = t1.shape

        theta = r2aa + r2ab

        Hr1  = lib.einsum('ae,ie->ia', imds.Fvv, r1)
        Hr1 -= lib.einsum('mi,ma->ia', imds.Foo, r1)
        Hr1 += np.einsum('me,imae->ia',imds.Fov, r2aa)
        Hr1 += np.einsum('ME,iMaE->ia',imds.Fov, r2ab)

        Hr2aa = lib.einsum('mnij,mnab->ijab', imds.woOoO, r2aa) * .25
        Hr2ab = lib.einsum('mNiJ,mNaB->iJaB', imds.woOoO, r2ab) * .5
        Hr2aa+= lib.einsum('be,ijae->ijab', imds.Fvv*.5, r2aa)
        Hr2aa-= lib.einsum('mj,imab->ijab', imds.Foo*.5, r2aa)
        Hr2ab+= lib.einsum('BE,iJaE->iJaB', imds.Fvv, r2ab)
        Hr2ab-= lib.einsum('MJ,iMaB->iJaB', imds.Foo, r2ab)

        tau2ab = np.einsum('ia,jb->ijab', r1, t1)
        tau2ab-= np.einsum('ia,jb->ijab', t1, r1)
        tau2ab+= r2ab
        tau2aa = np.einsum('ia,jb->ijab', r1, t1)
        tau2aa-= np.einsum('ia,jb->jiab', r1, t1)
        tau2aa = tau2aa - tau2aa.transpose(0,1,3,2)
        tau2aa+= r2aa
        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
        #:Hr1 += lib.einsum('mfae,imef->ia', eris_ovvv, theta)
        #:tmpaa = lib.einsum('meaf,ijef->maij', eris_ovvv, tau2aa)
        #:tmpab = lib.einsum('meAF,iJeF->mAiJ', eris_ovvv, tau2ab)
        #:tmp1 = lib.einsum('mfae,me->af', eris_ovvv, r1)
        #:Hr2aa+= lib.einsum('mb,maij->ijab', t1*.5, tmpaa)
        #:Hr2ab-= lib.einsum('mb,mAiJ->iJbA', t1, tmpab)
        mem_now = lib.current_memory()[0]
        max_memory = lib.param.MAX_MEMORY - mem_now
        blksize = max(int(max_memory*1e6/8/(nvir**3*3)), 2)
        tmp1 = np.zeros((nvir,nvir), dtype=r1.dtype)
        for p0,p1 in lib.prange(0, nocc, blksize):
            ovvv = np.asarray(eris.ovvv[p0:p1]).reshape((p1-p0)*nvir,-1)
            ovvv = lib.unpack_tril(ovvv).reshape(-1,nvir,nvir,nvir)
            Hr1 += lib.einsum('mfae,imef->ia', ovvv, theta[:,p0:p1])
            tmpaa = lib.einsum('meaf,ijef->maij', ovvv, tau2aa)
            tmpab = lib.einsum('meAF,iJeF->mAiJ', ovvv, tau2ab)
            tmp1 += lib.einsum('mfae,me->af', ovvv, r1[p0:p1])
            Hr2aa+= lib.einsum('mb,maij->ijab', t1[p0:p1]*.5, tmpaa)
            Hr2ab-= lib.einsum('mb,mAiJ->iJbA', t1[p0:p1], tmpab)
            ovvv = tmpaa = tmpab = None
        tmpa = lib.einsum('mnie,me->ni', imds.woOoV, r1)
        tmp  = lib.einsum('ni,njab->ijab', tmpa, t2)
        tmp -= lib.einsum('af,ijfb->ijab', tmp1, t2)
        tmp -= lib.einsum('mbij,ma->ijab', imds.woVoO, r1)
        for p0,p1 in lib.prange(0, nvir, nocc):
            tmp += lib.einsum('ejab,ie->ijab', np.asarray(imds.wvOvV[p0:p1]), r1[:,p0:p1])

        oVVo = np.asarray(imds.woVVo)
        Hr1 += np.einsum('maei,me->ia',imds.woVVo,r1)
        Hr2aa+= lib.einsum('mbej,imae->ijba', oVVo, r2ab)
        Hr2ab+= lib.einsum('MBEJ,iMEa->iJaB', oVVo, r2aa)
        Hr2ab+= lib.einsum('MbeJ,iMeA->iJbA', oVVo, r2ab)
        oVVo += np.asarray(imds.woVvO)
        tmp += lib.einsum('mbej,imae->ijab', oVVo, theta)
        oVVo = None
        Hr1-= lib.einsum('mnie,mnae->ia', imds.woOoV, theta)
        Hr2aa+= tmp
        Hr2ab+= tmp
        tmp = None

        eris_ovov = np.asarray(eris.ovov)
        tau = make_tau(t2, t1, t1)
        tmpaa = lib.einsum('menf,ijef->mnij', eris_ovov, tau2aa)
        tmpab = lib.einsum('meNF,iJeF->mNiJ', eris_ovov, tau2ab)
        Hr2aa += lib.einsum('mnij,mnab->ijab', tmpaa, tau) * 0.25
        Hr2ab += lib.einsum('mNiJ,mNaB->iJaB', tmpab, tau) * .5
        tmpaa = tmpab = tau = None

        tmpa = -lib.einsum('menf,imfe->ni', eris_ovov, theta)
        Hr1 += lib.einsum('na,ni->ia', t1, tmpa)
        tmp  = lib.einsum('mj,imab->ijab', tmpa, t2)
        tmp1 = np.einsum('menf,mf->en', eris_ovov, r1)
        tmpa = np.einsum('en,nb->eb', tmp1, t1)
        tmpa-= lib.einsum('menf,mnbf->eb', eris_ovov, theta)
        tmp += lib.einsum('eb,ijae->ijab', tmpa, t2)
        Hr2aa+= tmp
        Hr2ab-= tmp
        tmp = theta = eris_ovov = None

        #:eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv),t1.shape[1])
        #:Hr2aa += lib.einsum('ijef,aebf->ijab', tau2aa, eris_vvvv) * .25
        #:Hr2ab += lib.einsum('ijef,aebf->ijab', tau2ab, eris_vvvv) * .5
        tau2aa *= .5
        _add_vvvv_(self, tau2aa, eris, Hr2aa)
        _add_vvvv_(self, tau2ab, eris, Hr2ab)
        tau2aa = tau2ab = None

        Hr2aa = Hr2aa - Hr2aa.transpose(0,1,3,2)
        Hr2aa = Hr2aa - Hr2aa.transpose(1,0,2,3)
        Hr2ab = Hr2ab - Hr2ab.transpose(1,0,3,2)
        vector = self.amplitudes_to_vector_triplet(Hr1, (Hr2aa,Hr2ab))
        return vector

    def eomsf_ccsd_matvec(self, vector):
        '''Spin flip EOM-CCSD'''
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ee_imds:
            self.imds.make_ee()
        imds = self.imds

        t1, t2, eris = self.t1, self.t2, self.eris
        r1, r2 = self.vector_to_amplitudes_eomsf(vector)
        r2baaa, r2aaba = r2
        nocc, nvir = t1.shape

        Hr1  = np.einsum('ae,ie->ia', imds.Fvv, r1)
        Hr1 -= np.einsum('mi,ma->ia', imds.Foo, r1)
        Hr1 += np.einsum('me,imae->ia', imds.Fov, r2baaa)
        Hr1 += np.einsum('me,imae->ia', imds.Fov, r2aaba)
        Hr2baaa = .5*lib.einsum('mnij,mnab->ijab', imds.woOoO, r2baaa)
        Hr2aaba = .5*lib.einsum('mnij,mnab->ijab', imds.woOoO, r2aaba)
        Hr2baaa -= lib.einsum('mj,imab->ijab', imds.Foo*.5, r2baaa)
        Hr2aaba -= lib.einsum('mj,imab->ijab', imds.Foo*.5, r2aaba)
        Hr2baaa -= lib.einsum('mj,miab->jiab', imds.Foo*.5, r2baaa)
        Hr2aaba -= lib.einsum('mj,miab->jiab', imds.Foo*.5, r2aaba)
        Hr2baaa += lib.einsum('be,ijae->ijab', imds.Fvv*.5, r2baaa)
        Hr2aaba += lib.einsum('be,ijae->ijab', imds.Fvv*.5, r2aaba)
        Hr2baaa += lib.einsum('be,ijea->ijba', imds.Fvv*.5, r2baaa)
        Hr2aaba += lib.einsum('be,ijea->ijba', imds.Fvv*.5, r2aaba)

        tau2baaa = np.einsum('ia,jb->ijab', r1, t1)
        tau2baaa += r2baaa * .5
        tau2baaa = tau2baaa - tau2baaa.transpose(0,1,3,2)
        tau2aaba = np.einsum('ia,jb->ijab', r1, t1)
        tau2aaba += r2aaba * .5
        tau2aaba = tau2aaba - tau2aaba.transpose(1,0,2,3)

        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
        #:Hr1 += einsum('mfae,imef->ia', eris_ovvv, r2baaa)
        #:Hr1 += einsum('mfae,imef->ia', eris_ovvv, r2aaba)
        #:tmp1aaba = lib.einsum('meaf,ijef->maij', eris_ovvv, tau2baaa)
        #:tmp1baaa = lib.einsum('meaf,ijef->maij', eris_ovvv, tau2aaba)
        #:tmp1abaa = lib.einsum('meaf,ijfe->maij', eris_ovvv, tau2aaba)
        #:tmp2aaba = lib.einsum('meaf,ijfe->maij', eris_ovvv, tau2baaa)
        #:Hr2baaa -= lib.einsum('mb,maij->ijab', t1*.5, tmp2aaba)
        #:Hr2aaba -= lib.einsum('mb,maij->ijab', t1*.5, tmp1abaa)
        #:Hr2baaa -= lib.einsum('mb,maij->ijba', t1*.5, tmp1aaba)
        #:Hr2aaba -= lib.einsum('mb,maij->ijba', t1*.5, tmp1baaa)
        #:tmp = lib.einsum('mfae,me->af', eris_ovvv, r1)
        #:tmp = lib.einsum('af,jibf->ijab', tmp, t2)
        #:Hr2baaa -= tmp
        #:Hr2aaba -= tmp
        tmp1 = np.zeros((nvir,nvir), dtype=r1.dtype)
        mem_now = lib.current_memory()[0]
        max_memory = lib.param.MAX_MEMORY - mem_now
        blksize = max(int(max_memory*1e6/8/(nvir**3*3)), 2)
        for p0,p1 in lib.prange(0, nocc, blksize):
            ovvv = np.asarray(eris.ovvv[p0:p1]).reshape((p1-p0)*nvir,-1)
            ovvv = lib.unpack_tril(ovvv).reshape(-1,nvir,nvir,nvir)
            Hr1 += einsum('mfae,imef->ia', ovvv, r2baaa[:,p0:p1])
            Hr1 += einsum('mfae,imef->ia', ovvv, r2aaba[:,p0:p1])
            tmp1aaba = lib.einsum('meaf,ijef->maij', ovvv, tau2baaa)
            tmp1baaa = lib.einsum('meaf,ijef->maij', ovvv, tau2aaba)
            tmp1abaa = lib.einsum('meaf,ijfe->maij', ovvv, tau2aaba)
            tmp2aaba = lib.einsum('meaf,ijfe->maij', ovvv, tau2baaa)
            Hr2baaa -= lib.einsum('mb,maij->ijab', t1[p0:p1]*.5, tmp2aaba)
            Hr2aaba -= lib.einsum('mb,maij->ijab', t1[p0:p1]*.5, tmp1abaa)
            Hr2baaa -= lib.einsum('mb,maij->ijba', t1[p0:p1]*.5, tmp1aaba)
            Hr2aaba -= lib.einsum('mb,maij->ijba', t1[p0:p1]*.5, tmp1baaa)
            tmp = lib.einsum('mfae,me->af', ovvv, r1[p0:p1])
            tmp = lib.einsum('af,jibf->ijab', tmp, t2)
            Hr2baaa -= tmp
            Hr2aaba -= tmp
            tmp1aaba = tmp1baaa = tmp1abaa = tmp2aaba = ovvv = None
        tmp = lib.einsum('mbij,ma->ijab', imds.woVoO, r1)
        Hr2baaa -= tmp
        Hr2aaba -= tmp
        tmp = None

        Hr1 -= lib.einsum('mnie,mnae->ia', imds.woOoV, r2aaba)
        Hr1 -= lib.einsum('mnie,mnae->ia', imds.woOoV, r2baaa)
        tmp = lib.einsum('mnie,me->ni', imds.woOoV, r1)
        tmp = lib.einsum('ni,njab->ijab', tmp, t2)
        Hr2baaa += tmp
        Hr2aaba += tmp
        for p0,p1 in lib.prange(0, nvir, nocc):
            tmp = lib.einsum('ejab,ie->ijab', np.asarray(imds.wvOvV[p0:p1]), r1[:,p0:p1])
            Hr2baaa += tmp
            Hr2aaba += tmp
        tmp = None

        oVVo = np.asarray(imds.woVVo)
        Hr1 += np.einsum('maei,me->ia', oVVo, r1)
        Hr2baaa += lib.einsum('mbej,miea->jiba', oVVo, r2baaa)
        Hr2aaba += lib.einsum('mbej,miea->jiba', oVVo, r2aaba)
        oVvO = np.asarray(imds.woVvO)
        Hr2baaa += lib.einsum('mbej,imae->ijab', oVvO, r2aaba)
        Hr2aaba += lib.einsum('mbej,imae->ijab', oVvO, r2baaa)
        oVvO += oVVo
        Hr2baaa += lib.einsum('mbej,imae->ijab', oVvO, r2baaa)
        Hr2aaba += lib.einsum('mbej,imae->ijab', oVvO, r2aaba)
        oVvO = oVVo = None

        eris_ovov = np.asarray(eris.ovov)
        tau = make_tau(t2, t1, t1)
        tmp1baaa = lib.einsum('menf,ijef->mnij', eris_ovov, tau2aaba)
        tmp1aaba = lib.einsum('menf,ijef->mnij', eris_ovov, tau2baaa)
        Hr2baaa += .5*lib.einsum('mnij,mnab->ijab', tmp1aaba, tau)
        Hr2aaba += .5*lib.einsum('mnij,mnab->ijab', tmp1baaa, tau)
        tau2 = tmp1baaa = tmp1aaba = None

        rhoaaba = r2aaba + r2baaa
        tmp = lib.einsum('nfme,imfe->ni', eris_ovov, rhoaaba)
        Hr1 -= np.einsum('na,ni->ia', t1, tmp)
        Hr2baaa -= lib.einsum('mj,imba->jiab', tmp, t2)
        Hr2aaba -= lib.einsum('mj,imba->jiab', tmp, t2)

        tmp = np.einsum('menf,mf->en', eris_ovov, r1)
        tmp = np.einsum('en,nb->eb', tmp, t1)
        tmp-= lib.einsum('menf,mnbf->eb', eris_ovov, rhoaaba)
        Hr2baaa += lib.einsum('ea,ijbe->jiab', tmp, t2)
        Hr2aaba += lib.einsum('ea,ijbe->jiab', tmp, t2)
        eris_ovov = rhoaaba = tmp = None

        #:eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv),t1.shape[1])
        #:Hr2baaa += .5*lib.einsum('ijef,aebf->ijab', tau2baaa, eris_vvvv)
        #:Hr2aaba += .5*lib.einsum('ijef,aebf->ijab', tau2aaba, eris_vvvv)
        _add_vvvv_(self, tau2aaba, eris, Hr2aaba)
        tau2baaa *= .5
        _add_vvvv1_(self, tau2baaa, eris, Hr2baaa)
        tau2aaba = tau2baaa = None

        Hr2baaa = Hr2baaa - Hr2baaa.transpose(0,1,3,2)
        Hr2aaba = Hr2aaba - Hr2aaba.transpose(1,0,2,3)
        vector = self.amplitudes_to_vector_eomsf(Hr1, (Hr2baaa,Hr2aaba))
        return vector

    def eeccsd_diag(self):
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ee_imds:
            self.imds.make_ee()
        imds = self.imds

        eris = self.eris
        t1, t2 = self.t1, self.t2
        tau = make_tau(t2, t1, t1)
        nocc, nvir = t1.shape

        Fo = imds.Foo.diagonal()
        Fv = imds.Fvv.diagonal()
        Wovab = np.einsum('iaai->ia', imds.woVVo)
        Wovaa = Wovab + np.einsum('iaai->ia', imds.woVvO)

        eia = lib.direct_sum('-i+a->ia', Fo, Fv)
        Hr1aa = eia + Wovaa
        Hr1ab = eia + Wovab

        eris_ovov = np.asarray(eris.ovov)
        Wvvab = np.einsum('mnab,manb->ab', tau, eris_ovov)
        Wvvaa = .5*Wvvab - .5*np.einsum('mnba,manb->ab', tau, eris_ovov)
        ijb = np.einsum('iejb,ijeb->ijb', eris_ovov, t2)
        Hr2ab = lib.direct_sum('iJB+a->iJaB',-ijb, Fv)
        jab = np.einsum('kajb,kjab->jab', eris_ovov, t2)
        Hr2ab+= lib.direct_sum('-i-jab->ijab', Fo, jab)

        jib = np.einsum('iejb,ijbe->jib', eris_ovov, t2)
        jib = jib + jib.transpose(1,0,2)
        jib-= ijb + ijb.transpose(1,0,2)
        jba = np.einsum('kajb,jkab->jba', eris_ovov, t2)
        jba = jba + jba.transpose(0,2,1)
        jba-= jab + jab.transpose(0,2,1)
        Hr2aa = lib.direct_sum('jib+a->jiba', jib, Fv)
        Hr2aa+= lib.direct_sum('-i+jba->ijba', Fo, jba)
        eris_ovov = None

        Hr2baaa = lib.direct_sum('ijb+a->ijba',-ijb, Fv)
        Hr2baaa += Wovaa.reshape(1,nocc,1,nvir)
        Hr2baaa += Wovab.reshape(nocc,1,1,nvir)
        Hr2baaa = Hr2baaa + Hr2baaa.transpose(0,1,3,2)
        Hr2baaa+= lib.direct_sum('-i+jab->ijab', Fo, jba)
        Hr2baaa-= Fo.reshape(1,-1,1,1)
        Hr2aaba = lib.direct_sum('-i-jab->ijab', Fo, jab)
        Hr2aaba += Wovaa.reshape(1,nocc,1,nvir)
        Hr2aaba += Wovab.reshape(1,nocc,nvir,1)
        Hr2aaba = Hr2aaba + Hr2aaba.transpose(1,0,2,3)
        Hr2aaba+= lib.direct_sum('ijb+a->ijab', jib, Fv)
        Hr2aaba+= Fv.reshape(1,1,1,-1)
        Hr2ab += Wovaa.reshape(1,nocc,1,nvir)
        Hr2ab += Wovab.reshape(nocc,1,1,nvir)
        Hr2ab = Hr2ab + Hr2ab.transpose(1,0,3,2)
        Hr2aa += Wovaa.reshape(1,nocc,1,nvir) * 2
        Hr2aa = Hr2aa + Hr2aa.transpose(0,1,3,2)
        Hr2aa = Hr2aa + Hr2aa.transpose(1,0,2,3)
        Hr2aa *= .5

        Wooab = np.einsum('ijij->ij', imds.woOoO)
        Wooaa = Wooab - np.einsum('ijji->ij', imds.woOoO)
        Hr2aa += Wooaa.reshape(nocc,nocc,1,1)
        Hr2ab += Wooab.reshape(nocc,nocc,1,1)
        Hr2baaa += Wooab.reshape(nocc,nocc,1,1)
        Hr2aaba += Wooaa.reshape(nocc,nocc,1,1)

        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
        #:tmp = np.einsum('mb,mbaa->ab', t1, eris_ovvv)
        #:Wvvaa += np.einsum('mb,maab->ab', t1, eris_ovvv)
        mem_now = lib.current_memory()[0]
        max_memory = lib.param.MAX_MEMORY - mem_now
        blksize = max(int(max_memory*1e6/8/(nvir**3*3)), 2)
        tmp = np.zeros((nvir,nvir), dtype=t1.dtype)
        for p0,p1 in lib.prange(0, nocc, blksize):
            ovvv = np.asarray(eris.ovvv[p0:p1]).reshape((p1-p0)*nvir,-1)
            ovvv = lib.unpack_tril(ovvv).reshape(-1,nvir,nvir,nvir)
            tmp += np.einsum('mb,mbaa->ab', t1[p0:p1], ovvv)
            Wvvaa += np.einsum('mb,maab->ab', t1[p0:p1], ovvv)
            ovvv = None
        Wvvaa -= tmp
        Wvvab -= tmp
        Wvvab -= tmp.T
        Wvvaa = Wvvaa + Wvvaa.T
        #:eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv),t1.shape[1])
        #:tmp = np.einsum('aabb->ab', eris_vvvv)
        #:Wvvaa += tmp
        #:Wvvaa -= np.einsum('abba->ab', eris_vvvv)
        #:Wvvab += tmp
        for i in range(nvir):
            i0 = i*(i+1)//2
            vvv = lib.unpack_tril(np.asarray(eris.vvvv[i0:i0+i+1]))
            tmp = np.einsum('bb->b', vvv[i])
            Wvvaa[i] += tmp
            Wvvab[i] += tmp
            tmp = np.einsum('bb->b', vvv[:,:i+1,i])
            Wvvaa[i,:i+1] -= tmp
            Wvvaa[:i  ,i] -= tmp[:i]
            vvv = None
        Hr2aa += Wvvaa.reshape(1,1,nvir,nvir)
        Hr2ab += Wvvab.reshape(1,1,nvir,nvir)
        Hr2baaa += Wvvaa.reshape(1,1,nvir,nvir)
        Hr2aaba += Wvvab.reshape(1,1,nvir,nvir)

        vec_eeS = self.amplitudes_to_vector(Hr1aa, Hr2ab)
        vec_eeT = self.amplitudes_to_vector_triplet(Hr1aa, (Hr2aa,Hr2ab))
        vec_sf = self.amplitudes_to_vector_eomsf(Hr1ab, (Hr2baaa,Hr2aaba))
        return vec_eeS, vec_eeT, vec_sf

    def amplitudes_to_vector(self, t1, t2, out=None):
        nocc, nvir = t1.shape
        nov = nocc * nvir
        size = nov + nocc**2*nvir*(nvir+1)//2
        vector = np.ndarray(size, t1.dtype, buffer=out)
        vector[:nov] = t1.ravel()
        lib.pack_tril(t2.reshape(-1,nvir,nvir), out=vector[nov:])
        return vector

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        nvir = nmo - nocc
        nov = nocc * nvir
        size = nov + nocc**2*nvir*(nvir+1)//2
        t1 = vector[:nov].copy().reshape((nocc,nvir))
        t2 = np.zeros((nocc,nocc,nvir,nvir), dtype=vector.dtype)
        t2tril = vector[nov:size].reshape(nocc**2,nvir*(nvir+1)//2)
        oidx = np.arange(nocc**2).reshape(nocc,nocc).T.ravel()
        vtril = np.tril_indices(nvir)
        lib.takebak_2d(t2.reshape(nocc**2,nvir**2), t2tril, oidx, vtril[1]*nvir+vtril[0])
        lib.unpack_tril(t2tril, filltriu=0, out=t2)
        return t1, t2

    def vector_to_amplitudes_ee(self, vector):
        return self.vector_to_amplitudes(vector)

    def amplitudes_to_vector_ee(self,r1,r2):
        return self.amplitudes_to_vector(r1, r2)

    def amplitudes_to_vector_eomsf(self, t1, t2, out=None):
        nocc, nvir = t1.shape
        t2baaa, t2aaba = t2

        nbaaa = nocc*nocc*nvir*(nvir-1)//2
        naaba = nocc*(nocc-1)//2*nvir*nvir
        size = t1.size + nbaaa + naaba
        vector = np.ndarray(size, t2baaa.dtype, buffer=out)
        vector[:t1.size] = t1.ravel()
        pvec = vector[t1.size:]

        t2baaa = t2baaa.reshape(nocc*nocc,nvir*nvir)
        t2aaba = t2aaba.reshape(nocc*nocc,nvir*nvir)
        otril = np.tril_indices(nocc, k=-1)
        vtril = np.tril_indices(nvir, k=-1)
        oidxab = np.arange(nocc**2, dtype=np.int32)
        vidxab = np.arange(nvir**2, dtype=np.int32)
        lib.take_2d(t2baaa, oidxab, vtril[0]*nvir+vtril[1], out=pvec)
        lib.take_2d(t2aaba, otril[0]*nocc+otril[1], vidxab, out=pvec[nbaaa:])
        return vector

    def vector_to_amplitudes_eomsf(self, vector, nocc=None, nvir=None):
        if nocc is None:
            nocc = self.nocc
        if nvir is None:
            nmo = self.nmo
            nvir = nmo - nocc

        t1 = vector[:nocc*nvir].reshape(nocc,nvir).copy()
        pvec = vector[t1.size:]

        nbaaa = nocc*nocc*nvir*(nvir-1)//2
        naaba = nocc*(nocc-1)//2*nvir*nvir
        t2baaa = np.zeros((nocc*nocc,nvir*nvir), dtype=vector.dtype)
        t2aaba = np.zeros((nocc*nocc,nvir*nvir), dtype=vector.dtype)
        otril = np.tril_indices(nocc, k=-1)
        vtril = np.tril_indices(nvir, k=-1)
        oidxab = np.arange(nocc**2, dtype=np.int32)
        vidxab = np.arange(nvir**2, dtype=np.int32)

        v = pvec[:nbaaa].reshape(nocc*nocc,nvir*(nvir-1)//2)
        lib.takebak_2d(t2baaa, v, oidxab, vtril[0]*nvir+vtril[1])
        lib.takebak_2d(t2baaa,-v, oidxab, vtril[1]*nvir+vtril[0])
        v = pvec[nbaaa:nbaaa+naaba].reshape(-1,nvir*nvir)
        lib.takebak_2d(t2aaba, v, otril[0]*nocc+otril[1], vidxab)
        lib.takebak_2d(t2aaba,-v, otril[1]*nocc+otril[0], vidxab)
        t2baaa = t2baaa.reshape(nocc,nocc,nvir,nvir)
        t2aaba = t2aaba.reshape(nocc,nocc,nvir,nvir)
        return t1, (t2baaa, t2aaba)

    def amplitudes_to_vector_s4(self, t1, t2, out=None):
        nocc, nvir = t1.shape
        nov = nocc * nvir
        size = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
        vector = np.ndarray(size, t1.dtype, buffer=out)
        vector[:nov] = t1.ravel()
        otril = np.tril_indices(nocc, k=-1)
        vtril = np.tril_indices(nvir, k=-1)
        otril = otril[0]*nocc + otril[1]
        vtril = vtril[0]*nvir + vtril[1]
        lib.take_2d(t2.reshape(nocc**2,nvir**2), otril, vtril, out=vector[nov:])
        return vector

    def vector_to_amplitudes_s4(self, vector, nocc=None, nvir=None):
        if nocc is None:
            nocc = self.nocc
        if nvir is None:
            nvir = self.nmo - nocc
        nov = nocc * nvir
        size = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
        t1 = vector[:nov].copy().reshape((nocc,nvir))
        t2 = np.zeros((nocc,nocc,nvir,nvir), dtype=vector.dtype)
        t2 = _unpack_4fold(vector[nov:size], nocc, nvir)
        return t1, t2

    def amplitudes_to_vector_triplet(self, t1, t2, out=None):
        nocc, nvir = t1.shape
        nov = nocc * nvir
        size1 = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
        size = size1 + nov*(nov+1)//2
        vector = np.ndarray(size, t1.dtype, buffer=out)
        self.amplitudes_to_vector_s4(t1, t2[0], out=vector)
        t2ab = t2[1].transpose(0,2,1,3).reshape(nov,nov)
        lib.pack_tril(t2ab, out=vector[size1:])
        return vector

    def vector_to_amplitudes_triplet(self, vector, nocc=None, nvir=None):
        if nocc is None:
            nocc = self.nocc
        if nvir is None:
            nvir = self.nmo - nocc
        nov = nocc * nvir
        size1 = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
        size = size1 + nov*(nov+1)//2
        t1, t2aa = self.vector_to_amplitudes_s4(vector[:size1], nocc, nvir)
        t2ab = lib.unpack_tril(vector[size1:size], filltriu=2)
        t2ab = t2ab.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3).copy()
        return t1, (t2aa, t2ab)


class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore',
                 ao2mofn=ao2mo.full):
        cput0 = (time.clock(), time.time())
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, cc.mo_coeff)
        else:  # If mo_coeff is not canonical orbital
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, mo_coeff)
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
            if ao2mofn == ao2mo.full:
                if cc._scf._eri is not None:
                    eri = ao2mo.restore(1, ao2mofn(cc._scf._eri, mo_coeff), nmo)
                else:
                    eri = ao2mo.restore(1, ao2mofn(cc._scf.mol, mo_coeff, compact=0), nmo)
            else:
                eri = ao2mofn(cc._scf.mol, (mo_coeff,mo_coeff,mo_coeff,mo_coeff), compact=0)
                if mo_coeff.dtype == np.float: eri = eri.real
                eri = eri.reshape((nmo,)*4)
            self.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
            self.ooov = eri[:nocc,:nocc,:nocc,nocc:].copy()
            self.ovoo = eri[:nocc,nocc:,:nocc,:nocc].copy()
            self.oovo = eri[:nocc,:nocc,nocc:,:nocc].copy()
            self.ovov = eri[:nocc,nocc:,:nocc,nocc:].copy()
            self.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
            self.ovvo = eri[:nocc,nocc:,nocc:,:nocc].copy()
            ovvv = eri[:nocc,nocc:,nocc:,nocc:].reshape(-1,nvir,nvir)
            self.ovvv = lib.pack_tril(ovvv).reshape(nocc,nvir,nvir*(nvir+1)//2)
            self.vvvv = ao2mo.restore(4, eri[nocc:,nocc:,nocc:,nocc:].copy(), nvir)

        elif hasattr(cc._scf, 'with_df') and cc._scf.with_df:
            raise NotImplementedError

        else:
            orbo = mo_coeff[:,:nocc]
            orbv = mo_coeff[:,nocc:]
            self.dtype = mo_coeff.dtype
            ds_type = mo_coeff.dtype.char
            self.feri = lib.H5TmpFile()
            nvv = nvir*(nvir+1)//2
            self.oooo = self.feri.create_dataset('oooo', (nocc,nocc,nocc,nocc), ds_type)
            self.ooov = self.feri.create_dataset('ooov', (nocc,nocc,nocc,nvir), ds_type)
            self.ovoo = self.feri.create_dataset('ovoo', (nocc,nvir,nocc,nocc), ds_type)
            self.oovo = self.feri.create_dataset('oovo', (nocc,nocc,nvir,nocc), ds_type)
            self.ovov = self.feri.create_dataset('ovov', (nocc,nvir,nocc,nvir), ds_type)
            self.oovv = self.feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), ds_type)
            self.ovvo = self.feri.create_dataset('ovvo', (nocc,nvir,nvir,nocc), ds_type)
            self.ovvv = self.feri.create_dataset('ovvv', (nocc,nvir,nvv), ds_type)
            #self.vvvv = self.feri.create_dataset('vvvv', (nvir,nvir,nvir,nvir), ds_type)

            cput1 = time.clock(), time.time()
            # <ij||pq> = <ij|pq> - <ij|qp> = (ip|jq) - (iq|jp)
            tmpfile2 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            ao2mo.general(cc.mol, (orbo,mo_coeff,mo_coeff,mo_coeff), tmpfile2.name, 'aa')
            with h5py.File(tmpfile2.name) as f:
                buf = numpy.empty((nmo,nmo,nmo))
                for i in range(nocc):
                    lib.unpack_tril(f['aa'][i*nmo:(i+1)*nmo], out=buf)
                    self.oooo[i] = buf[:nocc,:nocc,:nocc]
                    self.ooov[i] = buf[:nocc,:nocc,nocc:]
                    self.ovoo[i] = buf[nocc:,:nocc,:nocc]
                    self.ovov[i] = buf[nocc:,:nocc,nocc:]
                    self.oovo[i] = buf[:nocc,nocc:,:nocc]
                    self.oovv[i] = buf[:nocc,nocc:,nocc:]
                    self.ovvo[i] = buf[nocc:,nocc:,:nocc]
                    self.ovvv[i] = lib.pack_tril(buf[nocc:,nocc:,nocc:])
                del(f['aa'])
                buf = None

            cput1 = log.timer_debug1('transforming oopq, ovpq', *cput1)

#            ao2mo.full(cc.mol, orbv, tmpfile2.name, dataname='vvvv')
#            with h5py.File(tmpfile2.name) as f:
#                self.feri['vvvv'] = ao2mo.restore(1, f['vvvv'], nvir)
            ao2mo.full(cc.mol, orbv, self.feri, dataname='vvvv')
            self.vvvv = self.feri['vvvv']

            cput1 = log.timer_debug1('transforming vvvv', *cput1)
        log.timer('CCSD integral transformation', *cput0)

class _IMDS:
    def __init__(self, cc):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.t1 = cc.t1
        self.t2 = cc.t2
        self.eris = cc.eris
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.made_ee_imds = False
        self._made_shared_2e = False

    def _make_shared_1e(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris
        self.Loo = imd.Loo(t1,t2,eris)
        self.Lvv = imd.Lvv(t1,t2,eris)
        self.Fov = imd.cc_Fov(t1,t2,eris)

        log.timer('EOM-CCSD shared one-electron intermediates', *cput0)

    def _make_shared_2e(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris
        # 2 virtuals
        self.Wovov = imd.Wovov(t1,t2,eris)
        self.Wovvo = imd.Wovvo(t1,t2,eris)
        self.Woovv = np.asarray(eris.ovov).transpose(0,2,1,3)

        log.timer('EOM-CCSD shared two-electron intermediates', *cput0)

    def make_ip(self, ip_partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and ip_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        if ip_partition != 'mp':
            self.Woooo = imd.Woooo(t1,t2,eris)
        self.Wooov = imd.Wooov(t1,t2,eris)
        self.Wovoo = imd.Wovoo(t1,t2,eris)
        self.made_ip_imds = True
        log.timer('EOM-CCSD IP intermediates', *cput0)

    def make_ea(self, ea_partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and ea_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1,t2,eris)
        if ea_partition == 'mp' and not np.any(t1):
            self.Wvvvo = imd.Wvvvo(t1,t2,eris)
        else:
            self.Wvvvv = imd.Wvvvv(t1,t2,eris)
            self.Wvvvo = imd.Wvvvo(t1,t2,eris,self.Wvvvv)
        self.made_ea_imds = True
        log.timer('EOM-CCSD EA intermediates', *cput0)


    def make_ee(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        nocc, nvir = t1.shape

        self.saved = lib.H5TmpFile()
        self.wvOvV = self.saved.create_dataset('vOvV', (nvir,nocc,nvir,nvir), t1.dtype.char)

        foo = eris.fock[:nocc,:nocc]
        fov = eris.fock[:nocc,nocc:]
        fvv = eris.fock[nocc:,nocc:]

        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
        #:self.Fvv  = np.einsum('mf,mfae->ae', t1, eris_ovvv) * 2
        #:self.Fvv -= np.einsum('mf,meaf->ae', t1, eris_ovvv)
        #:self.woVvO = lib.einsum('jf,mebf->mbej', t1, eris_ovvv)
        #:self.woVVo = lib.einsum('jf,mfbe->mbej',-t1, eris_ovvv)
        #:tau = make_tau(t2, t1, t1)
        #:self.woVoO  = 0.5 * lib.einsum('mebf,ijef->mbij', eris_ovvv, tau)
        #:self.woVoO += 0.5 * lib.einsum('mfbe,ijfe->mbij', eris_ovvv, tau)
        self.Fvv = np.zeros((nvir,nvir), dtype=t1.dtype)
        self.woVoO = np.empty((nocc,nvir,nocc,nocc), dtype=t1.dtype)
        woVvO = np.empty((nocc,nvir,nvir,nocc), dtype=t1.dtype)
        woVVo = np.empty((nocc,nvir,nvir,nocc), dtype=t1.dtype)
        tau = make_tau(t2, t1, t1)
        mem_now = lib.current_memory()[0]
        max_memory = lib.param.MAX_MEMORY - mem_now
        blksize = max(int(max_memory*1e6/8/(nvir**3*3)), 2)
        for p0,p1 in lib.prange(0, nocc, blksize):
            ovvv = np.asarray(eris.ovvv[p0:p1]).reshape((p1-p0)*nvir,-1)
            ovvv = lib.unpack_tril(ovvv).reshape(-1,nvir,nvir,nvir)
            self.Fvv += np.einsum('mf,mfae->ae', t1[p0:p1], ovvv) * 2
            self.Fvv -= np.einsum('mf,meaf->ae', t1[p0:p1], ovvv)
            self.woVoO[p0:p1] = 0.5 * lib.einsum('mebf,ijef->mbij', ovvv, tau)
            self.woVoO[p0:p1]+= 0.5 * lib.einsum('mfbe,ijfe->mbij', ovvv, tau)
            woVvO[p0:p1] = lib.einsum('jf,mebf->mbej', t1, ovvv)
            woVVo[p0:p1] = lib.einsum('jf,mfbe->mbej',-t1, ovvv)
            ovvv = None

        eris_ovov = np.asarray(eris.ovov)
        tmp = lib.einsum('njbf,mfne->mbej', t2, eris_ovov)
        woVvO -= tmp * .5
        woVVo += tmp

        eris_ooov = np.asarray(eris.ooov)
        ooov = eris_ooov + np.einsum('jf,nfme->njme', t1, eris_ovov)
        woVvO -= lib.einsum('nb,njme->mbej', t1, ooov)
        woVVo += lib.einsum('nb,mjne->mbej', t1, ooov)

        ovov = eris_ovov*2 - eris_ovov.transpose(0,3,2,1)
        eris_ovov = tmp = None
        self.Fov = np.einsum('nf,menf->me', t1, ovov)
        tilab = make_tau(t2, t1, t1, fac=.5)
        self.Foo  = lib.einsum('inef,menf->mi', tilab, ovov)
        self.Fvv -= lib.einsum('mnaf,menf->ae', tilab, ovov)
        theta = t2*2 - t2.transpose(1,0,2,3)
        woVvO += einsum('njfb,menf->mbej', theta, ovov) * .5
        ovov = tilab = None

        self.Foo += foo + 0.5*einsum('me,ie->mi', self.Fov+fov, t1)
        self.Fvv += fvv - 0.5*einsum('me,ma->ae', self.Fov+fov, t1)

        # 0 or 1 virtuals
        self.woOoO = lib.einsum('je,mine->mnij', t1, eris_ooov)
        self.woOoO = self.woOoO + self.woOoO.transpose(1,0,3,2)
        self.woOoO += np.asarray(eris.oooo).transpose(0,2,1,3)
        tmp = lib.einsum('nime,jneb->mbji', eris_ooov, t2)
        self.woVoO -= tmp.transpose(0,1,3,2) * .5
        self.woVoO -= tmp
        tmp = None
        ooov = eris_ooov*2 - eris_ooov.transpose(2,0,1,3)
        self.woVoO += lib.einsum('mine,njeb->mbij', ooov, theta) * .5
        self.woOoV = eris_ooov.transpose(0,2,1,3).copy()
        self.Foo += np.einsum('ne,mine->mi', t1, ooov)
        ooov = None

        eris_ovov = np.asarray(eris.ovov)
        tau = make_tau(t2, t1, t1)
        self.woOoO += lib.einsum('ijef,menf->mnij', tau, eris_ovov)
        self.woOoV += lib.einsum('if,mfne->mnie', t1, eris_ovov)
        tau = None

        ovov = eris_ovov*2 - eris_ovov.transpose(0,3,2,1)
        tmp1abba = lib.einsum('njbf,nemf->mbej', t2, eris_ovov)
        eris_ovov = None
        tmp1ab = lib.einsum('nifb,menf->mbei', theta, ovov) * -.5
        tmp1ab+= tmp1abba * .5
        tmpab = einsum('ie,mbej->mbij', t1, tmp1ab)
        tmpab+= einsum('ie,mbej->mbji', t1, tmp1abba)
        self.woVoO -= tmpab
        eris_oovo = numpy.asarray(eris.oovo)
        self.woVoO += eris_oovo.transpose(0,2,1,3)
        eris_oovo = tmpab = None

        eris_ovvo = np.asarray(eris.ovvo)
        eris_oovv = np.asarray(eris.oovv)
        woVvO += eris_ovvo.transpose(0,2,1,3)
        woVVo -= eris_oovv.transpose(0,2,3,1)
        self.saved['woVvO'] = woVvO
        self.saved['woVVo'] = woVVo
        self.woVvO = self.saved['woVvO']
        self.woVVo = self.saved['woVVo']

        self.woVoO += lib.einsum('ie,mebj->mbij', t1, eris_ovvo)
        self.woVoO += lib.einsum('ie,mjbe->mbji', t1, eris_oovv)
        self.woVoO += lib.einsum('me,ijeb->mbij', self.Fov, t2)
        self.woVoO -= lib.einsum('nb,mnij->mbij', t1, self.woOoO)

        # 3 or 4 virtuals
        #:eris_oovv = np.asarray(eris.oovv)
        #:eris_ovvo = np.asarray(eris.ovvo)
        #:self.wvOvV = einsum('nime,mnab->eiab', eris_ooov, tau)
        #:self.wvOvV -= lib.einsum('me,miab->eiab', self.Fov, t2)
        #:tmpab = lib.einsum('ma,mbei->eiab', t1, tmp1ab)
        #:tmpab+= lib.einsum('ma,mbei->eiba', t1, tmp1abba)
        #:tmpab-= einsum('ma,mibe->eiba', t1, eris_oovv)
        #:tmpab-= einsum('ma,mebi->eiab', t1, eris_ovvo)
        #:self.wvOvV += tmpab

        #:theta = t2*2 - t2.transpose(0,1,3,2)
        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
        #:ovvv = eris_ovvv*2 - eris_ovvv.transpose(0,3,2,1)
        #:tmpab = lib.einsum('mebf,miaf->eiab', eris_ovvv, t2)
        #:tmpab = tmpab + tmpab.transpose(0,1,3,2) * .5
        #:tmpab-= lib.einsum('mfbe,mifa->eiba', ovvv, theta) * .5
        #:self.wvOvV += eris_ovvv.transpose(2,0,3,1).conj()
        #:self.wvOvV -= tmpab
        tmp1ab -= eris_ovvo.transpose(0,2,1,3)
        tmp1abba -= eris_oovv.transpose(0,2,3,1)
        eris_ovvo = eris_oovv = None
        tau = make_tau(t2, t1, t1)
        mem_now = lib.current_memory()[0]
        max_memory = lib.param.MAX_MEMORY - mem_now
        blksize = max(int(max_memory*1e6/8/(nvir**3*6)), 2)
        for i0,i1 in lib.prange(0, nocc, blksize):
            wvOvV = einsum('nime,mnab->eiab', eris_ooov[:,i0:i1], tau)

            wvOvV -= lib.einsum('me,miab->eiab', self.Fov, t2[:,i0:i1])
            tmpab = lib.einsum('ma,mbei->eiab', t1, tmp1ab[:,:,:,i0:i1])
            tmpab+= lib.einsum('ma,mbei->eiba', t1, tmp1abba[:,:,:,i0:i1])
            wvOvV += tmpab

            for p0,p1 in lib.prange(0, nocc, blksize):
                ovvv = np.asarray(eris.ovvv[p0:p1]).reshape((p1-p0)*nvir,-1)
                ovvv = lib.unpack_tril(ovvv).reshape(-1,nvir,nvir,nvir)
                if p0 == i0:
                    wvOvV += ovvv.transpose(2,0,3,1).conj()
                tmpab = lib.einsum('mebf,miaf->eiab', ovvv, t2[p0:p1,i0:i1])
                tmpab = tmpab + tmpab.transpose(0,1,3,2) * .5
                ovvv = ovvv*2 - ovvv.transpose(0,3,2,1)
                tmpab-= lib.einsum('mfbe,mifa->eiba', ovvv, theta[p0:p1,i0:i1]) * .5
                wvOvV -= tmpab
                self.wvOvV[:,i0:i1] = wvOvV
                ovvv = tmpab = None

        self.made_ee_imds = True
        log.timer('EOM-CCSD EE intermediates', *cput0)

def make_tau(t2, t1, r1, fac=1, out=None):
    tau = np.einsum('ia,jb->ijab', t1, r1)
    tau = tau + tau.transpose(1,0,3,2)
    tau *= fac * .5
    tau += t2
    return tau

def _unpack_4fold(c2vec, nocc, nvir):
    t2 = np.zeros((nocc**2,nvir**2), dtype=c2vec.dtype)
    if nocc > 1 and nvir > 1:
        t2tril = c2vec.reshape(nocc*(nocc-1)//2,nvir*(nvir-1)//2)
        otril = np.tril_indices(nocc, k=-1)
        vtril = np.tril_indices(nvir, k=-1)
        lib.takebak_2d(t2, t2tril, otril[0]*nocc+otril[1], vtril[0]*nvir+vtril[1])
        lib.takebak_2d(t2, t2tril, otril[1]*nocc+otril[0], vtril[1]*nvir+vtril[0])
        t2tril = -t2tril
        lib.takebak_2d(t2, t2tril, otril[0]*nocc+otril[1], vtril[1]*nvir+vtril[0])
        lib.takebak_2d(t2, t2tril, otril[1]*nocc+otril[0], vtril[0]*nvir+vtril[1])
    return t2.reshape(nocc,nocc,nvir,nvir)

def _add_vvvv_(cc, t2, eris, Ht2):
    nocc = t2.shape[0]
    nvir = t2.shape[2]
    t2tril = numpy.zeros((nocc*(nocc+1)//2,nvir,nvir))
    t2tril = ccsd.add_wvvVV_(cc, np.zeros((nocc,nvir)), t2, eris, t2tril,
                             with_ovvv=False)
    idxo = np.arange(nocc)
    t2tril[idxo*(idxo+1)//2+idxo] *= .5
    idxo = np.tril_indices(nocc)
    lib.takebak_2d(Ht2.reshape(nocc**2,nvir**2),
                   t2tril.reshape(nocc*(nocc+1)//2,nvir**2),
                   idxo[0]*nocc+idxo[1], np.arange(nvir**2))
    return Ht2

def _add_vvvv1_(cc, t2, eris, Ht2):
    nvir = t2.shape[2]
    for i in range(nvir):
        i0 = i*(i+1)//2
        vvv = lib.unpack_tril(np.asarray(eris.vvvv[i0:i0+i+1]))
        Ht2[:,:, i] += lib.einsum('ijef,ebf->ijb', t2[:,:,:i+1], vvv)
        Ht2[:,:,:i] += lib.einsum('ijf,abf->ijab', t2[:,:,i], vvv[:i])
        vvv = None
    return Ht2

def _mem_usage(nocc, nvir):
    incore = (nocc+nvir)**4
    # Roughly, factor of two for intermediates and factor of two
    # for safety (temp arrays, copying, etc)
    incore *= 4
    # TODO: Improve incore estimate and add outcore estimate
    outcore = basic = incore
    return incore*8/1e6, outcore*8/1e6, basic*8/1e6

def _cp(a):
    return numpy.array(a, copy=False, order='C')


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

    print("IP energies... (right eigenvector)")
    part = None
    e,v = mycc.ipccsd(nroots=3,partition=part)
    print(e)
    print(e[0] - 0.4335604332073799)
    print(e[1] - 0.5187659896045407)
    print(e[2] - 0.6782876002229172)

    print("IP energies... (left eigenvector)")
    e,lv = mycc.ipccsd(nroots=3,left=True,partition=part)
    print(e)
    print(e[0] - 0.4335604332073799)
    print(e[1] - 0.5187659896045407)
    print(e[2] - 0.6782876002229172)

    mycc.ipccsd_star(e,v,lv)

    print("EA energies... (right eigenvector)")
    e,v = mycc.eaccsd(nroots=3,partition=part)
    print(e)
    print(e[0] - 0.16737886338859731)
    print(e[1] - 0.24027613852009164)
    print(e[2] - 0.51006797826488071)

    print("EA energies... (left eigenvector)")
    e,lv = mycc.eaccsd(nroots=3,left=True,partition=part)
    print(e)
    print(e[0] - 0.16737886338859731)
    print(e[1] - 0.24027613852009164)
    print(e[2] - 0.51006797826488071)

    mycc.eaccsd_star(e,v,lv)
