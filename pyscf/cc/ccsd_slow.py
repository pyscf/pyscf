#!/usr/bin/env python

import numpy
from pyscf import ao2mo
from pyscf import lib
from pyscf.lib import logger


def kernel(cc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6,
           verbose=logger.INFO):
    if verbose is None:
        verbose = cc.verbose
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cc.stdout, verbose)

    if t1 is None and t2 is None:
        t1, t2 = cc.get_init_guess(eris)[1:]
    elif t1 is None:
        t1 = numpy.zeros((nocc,nvir))
    elif t2 is None:
        t2 = cc.get_init_guess(eris)[2]

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
        foo, fov, fvv = make_inter_F(cc, t1, t2, eris)
        t1new = update_amp_t1(cc, t1, t2, eris, foo, fov, fvv)
        t2new = update_amp_t2(cc, t1, t2, eris, foo, fov, fvv)
        normt = numpy.linalg.norm(t1new-t1) + numpy.linalg.norm(t2new-t2)
        t1, t2 = t1new, t2new
        t1new = t2new = None
        if cc.diis:
            t1, t2 = cc.diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, cc.energy(t1, t2, eris)
        log.info('istep = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep, eccsd, eccsd - eold, normt)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    return conv, eccsd, t1, t2


def make_inter_F(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock

    tau = t2 + numpy.einsum('ia,jb->iajb', t1, t1*.5)
    theta = tau * 2 - tau.transpose(2,1,0,3)

    foo = fock[:nocc,:nocc].copy()
    foo[range(nocc),range(nocc)] = 0
    foo += .5 * numpy.einsum('ia,ja->ij', fock[:nocc,nocc:], t1)
    g2 = 2 * numpy.asarray(eris.ovoo)
    g2 -= numpy.asarray(eris.ovoo).transpose(2,1,0,3)
    foo += numpy.einsum('kc,kcij', t1, g2)
    foo += numpy.einsum('iakb,jakb->ij', eris.ovov, theta)
    g2 = None

    g2 = 2 * numpy.asarray(eris.ovov)
    g2 -= numpy.asarray(eris.ovov).transpose(2,1,0,3)
    fov = fock[:nocc,nocc:] + numpy.einsum('kc,iakc->ia', t1, g2)
    g2 = None

    fvv = fock[nocc:,nocc:].copy()
    fvv[range(nvir),range(nvir)] = 0
    fvv -= .5 * numpy.einsum('ia,ib->ab', t1, fock[:nocc,nocc:])
    g2 = 2 * numpy.asarray(eris.ovvv)
    g2 -= numpy.asarray(eris.ovvv).transpose(0,3,2,1)
    fvv += numpy.einsum('kc,kcab->ab', t1, g2)
    fvv -= numpy.einsum('icja,icjb->ab', theta, eris.ovov)
    return foo, fov, fvv

def update_amp_t1(cc, t1, t2, eris, foo, fov, fvv):
    nocc, nvir = t1.shape
    fock = eris.fock

    g2 = 2 * numpy.asarray(eris.ovov)
    g2 -= numpy.asarray(eris.oovv).transpose(1,2,0,3)
    t1new = fock[:nocc,nocc:] \
          + numpy.einsum('ib,ab->ia', t1, fvv) \
          - numpy.einsum('ja,ji->ia', t1, foo) \
          + numpy.einsum('jb,iajb->ia', t1, g2)
    theta = numpy.asarray(t2) * 2
    theta -= numpy.asarray(t2).transpose(2,1,0,3)
    t1new += numpy.einsum('jb,iajb->ia', fov, theta)
    t1new += numpy.einsum('ibjc,jcba->ia', theta, eris.ovvv)
    t1new -= numpy.einsum('jbki,jbka->ia', eris.ovoo, theta)
    mo_e = fock.diagonal()
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    t1new /= eia
    return t1new

def update_amp_t2(cc, t1, t2, eris, foo, fov, fvv):
    nocc, nvir = t1.shape
    fock = eris.fock
    t2new = numpy.copy(eris.ovov)

    ft_ij = foo + numpy.einsum('ja,ia->ij', .5*t1, fov)
    ft_ab = fvv - numpy.einsum('ia,ib->ab', .5*t1, fov)
    tmp1 = numpy.einsum('bc,iajc->iajb', ft_ab, t2)
    tmp2 = numpy.einsum('kj,kbia->iajb', ft_ij, t2)
    tw = tmp1 - tmp2
    tmp1 = tmp2 = None

    tmp1 = numpy.einsum('ic,kcjb->kijb', t1, eris.ovov)
    tw += numpy.einsum('ka,kijb->iajb', -t1, tmp1)
    tmp1 = numpy.einsum('ic,kjac->kjai', t1, eris.oovv)
    tw -= numpy.einsum('kb,kjai->iajb', t1, tmp1)
    #** wovOV, woVoV = make_inter_wovvo(cc, t1, t2, eris)
    theta = t2 * 2 - t2.transpose(2,1,0,3)
    tau = theta - numpy.einsum('jc,ka->jakc', t1, t1*2)

    wovOV = numpy.copy(eris.ovov)
    wovOV += .5 * numpy.einsum('jakc,ibkc->ibja', tau, eris.ovov)
    wovOV -= .5 * numpy.einsum('jakc,ickb->ibja', t2, eris.ovov)
    wovOV += numpy.einsum('jc,ibac->ibja', t1, eris.ovvv)
    wovOV -= numpy.einsum('ka,ibkj->ibja', t1, eris.ovoo)

    tau = .5*t2 + numpy.einsum('ia,jb->iajb', t1, t1)
    woVoV = -numpy.asarray(eris.oovv).transpose(1,2,0,3)
    woVoV += numpy.einsum('jcka,ickb->ibja', tau, eris.ovov)
    woVoV -= numpy.einsum('jc,icab->ibja', t1, eris.ovvv)
    woVoV += numpy.einsum('ka,kbij->ibja', t1, eris.ovoo)
    #** end make_inter_wovvo
    tw += numpy.einsum('iakc,kcjb->iajb', t2, woVoV)
    tw += numpy.einsum('icka,kcjb->ibja', t2, woVoV)
    tw += numpy.einsum('iakc,kcjb->iajb', theta, wovOV)
    tau = theta = None

    tw += numpy.einsum('jc,iacb->jbia', t1, eris.ovvv) \
        - numpy.einsum('ka,jbik->jbia', t1, eris.ovoo)
    t2new += tw + tw.transpose(2,3,0,1)
    tmp1 = tw = None
    #** woooo = make_inter_Woooo(cc, t1, t2, eris)
    tau = t2 + numpy.einsum('ia,jb->iajb', t1, t1)
    tmp1 = numpy.einsum('la,jaik->jlik', t1, eris.ovoo)
    tw = eris.oooo + tmp1 + tmp1.transpose(2,3,0,1)
    tw += numpy.einsum('iajb,kalb->ikjl', eris.ovov, tau)
    #** end make_inter_Woooo
    t2new += numpy.einsum('kilj,kalb->iajb', tw, tau)
    tmp1 = numpy.einsum('icjd,kcbd->ikjb', tau, eris.ovvv)
    tmp1 = numpy.einsum('ka,ikjb->iajb', t1, tmp1)
    t2new -= tmp1 + tmp1.transpose(2,3,0,1)
    t2new += cc.add_wvVvV(t1, t2, eris)

    mo_e = fock.diagonal()
    eia = (mo_e[:nocc,None] - mo_e[None,nocc:]).reshape(-1)
    t2new /= (eia[:,None] + eia[None,:]).reshape(nocc,nvir,nocc,nvir)
    return t2new


def energy(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    tau = t2 + numpy.einsum('ia,jb->iajb', t1, t1)
    theta = tau * 2 - tau.transpose(2,1,0,3)
    fock = eris.fock
    e = numpy.einsum('ia,ia', fock[:nocc,nocc:], t1) * 2 \
      + numpy.einsum('iajb,iajb', eris.ovov, theta)
    return e


class CCSD(lib.StreamObject):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.max_cycle = 50
        self.conv_tol = 1e-7
        self.conv_tol_normt = 1e-5
        self.diis_space = 6
        self.diis_file = None
        self.diis_start_cycle = 1
        self.diis_start_energy_diff = 1e-2

        self.frozen = frozen

##################################################
# don't modify the following attributes, they are not input options
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._conv = False
        self.emp2 = None
        self.ecc = None
        self.e_corr = None
        self.t1 = None
        self.t2 = None
        self.l1 = None
        self.l2 = None

    @property
    def nocc(self):
        self._nocc = int(self.mo_occ.sum()) // 2
        return self._nocc

    @property
    def nmo(self):
        self._nmo = len(self.mo_occ)
        return self._nmo

    def get_init_guess(self, eris=None):
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        return self.init_amps(eris)
    def init_amps(self, eris):
        nocc = self.nocc
        nvir = self.nmo - nocc
        mo_e = eris.fock.diagonal()
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        t2 = numpy.empty((nocc,nvir,nocc,nvir))
        self.emp2 = 0
        for i in range(nocc):
            gi = eris.ovov[i]
            t2[i] = gi/lib.direct_sum('a,jb->ajb', eia[i], eia)
            theta = gi*2 - gi.transpose(2,1,0)
            self.emp2 += numpy.einsum('ajb,ajb->', t2[i], theta)
        lib.logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        t1 = eris.fock[:nocc,nocc:] / eia
        return self.emp2, t1, t2

    energy = energy

    def kernel(self, t1=None, t2=None):
        eris = self.ao2mo()
        self._conv, self.ecc, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose)
        return self.ecc, self.t1, self.t2

    def ao2mo(self):
        return _ERIS(self)

    def add_wvVvV(self, t1, t2, eris):
        tau = t2 + numpy.einsum('ia,jb->iajb', t1, t1)
        return numpy.einsum('icjd,acbd->iajb', tau, eris.vvvv)

    def diis(self, t1, t2, istep, normt, de, adiis):
        if (istep > self.diis_start_cycle and
            abs(de) < self.diis_start_energy_diff):
            t1t2 = numpy.hstack((t1.ravel(),t2.ravel()))
            t1t2 = adiis.update(t1t2)
            t1 = t1t2[:t1.size].reshape(t1.shape)
            t2 = t1t2[t1.size:].reshape(t2.shape)
            logger.debug(self, 'DIIS for step %d', istep)
        return t1, t2

CC = CCSD

class _ERIS:
    def __init__(self, cc):
        nocc = cc.nocc
        nmo = cc.nmo
        mo_coeff = cc.mo_coeff
        eri1 = ao2mo.incore.full(cc._scf._eri, mo_coeff)
        eri1 = ao2mo.restore(1, eri1, nmo)
        self.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
        self.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
        self.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
        self.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
        self.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
        self.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
        self.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
        self.fock = numpy.diag(cc._scf.mo_energy)

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_h2o'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    rhf = scf.RHF(mol)
    rhf.scf() # -76.0267656731

    mcc = CCSD(rhf)
    eris = mcc.ao2mo()
    emp2, t1, t2 = mcc.init_amps(eris)
    #print emp2, 'emp2'
    print(abs(t2).sum() - 4.95565712182)
    foo, fov, fvv = make_inter_F(mcc, t1, t2, eris)
    print(abs(foo).sum() - 0.220378654203)
    print(abs(fov).sum() - 0.0           )
    print(abs(fvv).sum() - 0.526264584497)
    t1 = update_amp_t1(mcc, t1, t2, eris, foo, fov, fvv)
    print(abs(t1).sum() - 0.0475038989126)

    foo, fov, fvv = make_inter_F(mcc, t1, t2, eris)
    t2 = update_amp_t2(mcc, t1, t2, eris, foo, fov, fvv)
    print(abs(t2).sum()-5.4154348608834315)

    mcc.kernel()
    print(mcc.ecc - -0.21334318254)
    print(mcc.ecc - -0.213343234198275)
    print(abs(mcc.t2-mcc.t2.transpose(2,3,0,1)).sum())

