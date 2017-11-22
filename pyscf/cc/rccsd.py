'''
Restricted CCSD for complex integrals

note MO integrals are treated in chemist's notation

Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)
'''

import time
import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import rintermediates as imd
from pyscf.lib import linalg_helper

def update_amps(cc, t1, t2, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    assert(isinstance(eris, ccsd._ChemistsERIs))
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:].copy()
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()

    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    # Move energy terms to the other side
    Foo -= np.diag(np.diag(foo))
    Fvv -= np.diag(np.diag(fvv))

    # T1 equation
    t1new = np.asarray(fov).conj().copy()
    t1new +=-2*np.einsum('kc,ka,ic->ia', fov, t1, t1)
    t1new +=   np.einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -np.einsum('ki,ka->ia', Foo, t1)
    t1new += 2*np.einsum('kc,kica->ia', Fov, t2)
    t1new +=  -np.einsum('kc,ikca->ia', Fov, t2)
    t1new +=   np.einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += 2*np.einsum('aikc,kc->ia', eris.voov, t1)
    t1new +=  -np.einsum('acki,kc->ia', eris.vvoo, t1)
    eris_ovvv = np.asarray(eris.vovv).conj().transpose(1,0,3,2)
    t1new += 2*lib.einsum('kdac,ikcd->ia', eris_ovvv, t2)
    t1new +=  -lib.einsum('kcad,ikcd->ia', eris_ovvv, t2)
    t1new += 2*lib.einsum('kdac,kd,ic->ia', eris_ovvv, t1, t1)
    t1new +=  -lib.einsum('kcad,kd,ic->ia', eris_ovvv, t1, t1)
    eris_ooov = _cp(eris.vooo).conj().transpose(3,2,1,0)
    t1new +=-2*lib.einsum('kilc,klac->ia', eris_ooov, t2)
    t1new +=   lib.einsum('likc,klac->ia', eris_ooov, t2)
    t1new +=-2*lib.einsum('kilc,lc,ka->ia', eris_ooov, t1, t1)
    t1new +=   lib.einsum('likc,lc,ka->ia', eris_ooov, t1, t1)

    # T2 equation
    t2new = np.asarray(eris.vovo).transpose(1,3,0,2).copy()
    if cc.cc2:
        Woooo2 = np.asarray(eris.oooo).transpose(0,2,1,3).copy()
        Woooo2 += lib.einsum('kilc,jc->klij', eris_ooov, t1)
        Woooo2 += lib.einsum('ljkc,ic->klij', eris_ooov, t1)
        eris_ovov = eris.vovo.conj().transpose(1,0,3,2)
        Woooo2 += lib.einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
        t2new += lib.einsum('klij,ka,lb->ijab', Woooo2, t1, t1)
        Wvvvv = lib.einsum('kcbd,ka->abcd', eris_ovvv, -t1)
        Wvvvv = Wvvvv + Wvvvv.transpose(1,0,3,2)
        Wvvvv += np.asarray(eris.vvvv).transpose(0,2,1,3)
        t2new += lib.einsum('abcd,ic,jd->ijab', Wvvvv, t1, t1)
        Lvv2 = fvv - np.einsum('kc,ka->ac', fov, t1)
        Lvv2 -= np.diag(np.diag(fvv))
        tmp = lib.einsum('ac,ijcb->ijab', Lvv2, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        Loo2 = foo + np.einsum('kc,ic->ki', fov, t1)
        Loo2 -= np.diag(np.diag(foo))
        tmp = lib.einsum('ki,kjab->ijab', Loo2, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
    else:
        Loo = imd.Loo(t1, t2, eris)
        Lvv = imd.Lvv(t1, t2, eris)
        Loo -= np.diag(np.diag(foo))
        Lvv -= np.diag(np.diag(fvv))
        Woooo = imd.cc_Woooo(t1, t2, eris)
        Wvoov = imd.cc_Wvoov(t1, t2, eris)
        Wvovo = imd.cc_Wvovo(t1, t2, eris)
        Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
        tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
        t2new += lib.einsum('klij,klab->ijab', Woooo, tau)
        t2new += lib.einsum('abcd,ijcd->ijab', Wvvvv, tau)
        tmp = lib.einsum('ac,ijcb->ijab', Lvv, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        tmp = lib.einsum('ki,kjab->ijab', Loo, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
        tmp  = 2*lib.einsum('akic,kjcb->ijab', Wvoov, t2)
        tmp -=   lib.einsum('akci,kjcb->ijab', Wvovo, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        tmp = lib.einsum('akic,kjbc->ijab', Wvoov, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
        tmp = lib.einsum('bkci,kjac->ijab', Wvovo, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))

    tmp2  = lib.einsum('bcki,ka->abic', eris.vvoo, -t1)
    tmp2 += np.asarray(eris.vovv).transpose(0,2,1,3)
    tmp = lib.einsum('abic,jc->ijab', tmp2, t1)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp2  = lib.einsum('aikc,jc->akij', eris.voov, t1)
    tmp2 += eris_ooov.transpose(3,1,2,0).conj()
    tmp = lib.einsum('akij,kb->ijab', tmp2, t1)
    t2new -= (tmp + tmp.transpose(1,0,3,2))

    mo_e = eris.fock.diagonal().real
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new


def energy(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    e = 2*np.einsum('ia,ia', fock[:nocc,nocc:], t1)
    tau = np.einsum('ia,jb->ijab',t1,t1)
    tau += t2
    eris_ovov = np.asarray(eris.vovo).conj().transpose(1,0,3,2)
    e += 2*np.einsum('ijab,iajb', tau, eris_ovov)
    e +=  -np.einsum('ijab,ibja', tau, eris_ovov)
    return e.real


class RCCSD(ccsd.CCSD):
    '''restricted CCSD with IP-EOM, EA-EOM, EE-EOM, and SF-EOM capabilities

    Ground-state CCSD is performed in optimized ccsd.CCSD and EOM is performed here.
    '''
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.max_space = 20
        self._keys = self._keys.union(['max_space'])

    def init_amps(self, eris):
        nocc = self.nocc
        nvir = self.nmo - nocc
        mo_e = eris.fock.diagonal().real
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
        t1 = eris.fock[:nocc,nocc:].conj() / eia
        eris_vovo = np.asarray(eris.vovo)
        t2 = eris_vovo.transpose(1,3,0,2) / eijab
        eris_ovov = eris_vovo.conj().transpose(1,0,3,2)
        self.emp2  = 2*np.einsum('ijab,iajb', t2, eris_ovov)
        self.emp2 -=   np.einsum('ijab,ibja', t2, eris_ovov)
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, t1, t2

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2)
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
        if mbpt2:
            #cctyp = 'MBPT2'
            #self.e_corr, self.t1, self.t2 = self.init_amps(eris)
            raise NotImplementedError

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        return ccsd.CCSD.ccsd(self, t1, t2, eris)

    def ao2mo(self, mo_coeff=None):
        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff)

        elif hasattr(self._scf, 'with_df'):
            logger.warn(self, 'CCSD detected DF being used in the HF object. '
                        'MO integrals are computed based on the DF 3-index tensors.\n'
                        'It\'s recommended to use dfccsd.CCSD for the '
                        'DF-CCSD calculations')
            raise NotImplementedError
            #return _make_df_eris_outcore(self, mo_coeff)

        else:
            return _make_eris_outcore(self, mo_coeff)

    energy = energy
    update_amps = update_amps

    def _add_vvvv(self, t1, t2, eris, out=None):
        tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
        t2new = np.ndarray(t2.shape, t2.dtype, buffer=out)

        nocc, nvir = t1.shape
        max_memory = max(2000, self.max_memory-lib.current_memory()[0])
        unit = nvir**3*2 + nocc**2*nvir
        blksize = min(nvir, max(ccsd.BLKMIN, int(max_memory*1e6/8/unit)))

        for p0,p1 in lib.prange(0, nvir, blksize):
            t2new[:,:,p0:p1] = lib.einsum('ijcd,acbd->ijab', tau, eris.vvvv[p0:p1])
        return t2new
    _add_vvvv_full = _add_vvvv

def _make_eris_incore(mycc, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    eris = ccsd._ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc

    eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff)
    eris.vooo = np.empty((nvir,nocc,nocc,nocc))
    eris.voov = np.empty((nvir,nocc,nocc,nvir))
    eris.vovo = np.empty((nvir,nocc,nvir,nocc))
    eris.vovv = np.empty((nvir,nocc,nvir,nvir))
    eris.vvoo = np.empty((nvir,nvir,nocc,nocc))
    eris.vvvv = np.empty((nvir,nvir,nvir,nvir))

    nocc_pair = nocc*(nocc+1)//2
    eris.oooo = ao2mo.restore(1, eri1[:nocc_pair,:nocc_pair], nocc)

    outbuf = np.empty((nmo,nmo,nmo))
    p1 = nocc*(nocc+1)//2
    for i in range(nocc,nmo):
        p0, p1 = p1, p1 + i + 1
        buf = lib.unpack_tril(eri1[p0:p1], out=outbuf)
        eris.vooo[i-nocc] = buf[:nocc,:nocc,:nocc]
        eris.voov[i-nocc] = buf[:nocc,:nocc,nocc:]
        eris.vovo[i-nocc] = buf[:nocc,nocc:,:nocc]
        eris.vovv[i-nocc] = buf[:nocc,nocc:,nocc:]
        eris.vvoo[i-nocc,:i+1-nocc] = buf[nocc:i+1,:nocc,:nocc]
        eris.vvvv[i-nocc,:i+1-nocc] = buf[nocc:i+1,nocc:,nocc:]
        if i > nocc:
            eris.vvoo[:i-nocc,i-nocc] = buf[nocc:i,:nocc,:nocc]
            eris.vvvv[:i-nocc,i-nocc] = buf[nocc:i,nocc:,nocc:]
    logger.timer(mycc, 'CCSD integral transformation', *cput0)
    return eris

def _make_eris_outcore(mycc, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = ccsd._ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mol = mycc.mol
    mo_coeff = eris.mo_coeff
    nocc = eris.nocc
    nao, nmo = mo_coeff.shape
    nvir = nmo - nocc
    orbo = mo_coeff[:,:nocc]
    orbv = mo_coeff[:,nocc:]
    nvpair = nvir * (nvir+1) // 2
    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.vooo = eris.feri1.create_dataset('vooo', (nvir,nocc,nocc,nocc), 'f8')
    eris.voov = eris.feri1.create_dataset('voov', (nvir,nocc,nocc,nvir), 'f8')
    eris.vovo = eris.feri1.create_dataset('vovo', (nvir,nocc,nvir,nocc), 'f8')
    eris.vovv = eris.feri1.create_dataset('vovv', (nvir,nocc,nvir,nvir), 'f8')
    eris.vvoo = eris.feri1.create_dataset('vvoo', (nvir,nvir,nocc,nocc), 'f8')
    eris.vvvv = eris.feri1.create_dataset('vvvv', (nvir,nvir,nvir,nvir), 'f8')
    max_memory = max(2000, mycc.max_memory-lib.current_memory()[0])

    ftmp = lib.H5TmpFile()
    ao2mo.full(mol, mo_coeff, ftmp, max_memory=max_memory, verbose=log)
    eri = ftmp['eri_mo']

    nocc_pair = nocc*(nocc+1)//2
    eris.oooo[:] = ao2mo.restore(1, _cp(eri[:nocc_pair,:nocc_pair]), nocc)

    nmo_pair = nmo * (nmo+1) // 2
    tril2sq = lib.unpack_tril(np.arange(nmo_pair))
    blksize = min(nvir, max(2, int(max_memory*1e6/8/nmo**3/2)))
    for p0, p1 in lib.prange(0, nvir, blksize):
        q0, q1 = p0+nocc, p1+nocc
        off0 = q0*(q0+1)//2
        off1 = q1*(q1+1)//2
        buf = lib.unpack_tril(_cp(eri[off0:off1]))

        tmp = buf[ tril2sq[q0:q1,:nocc] - off0 ]
        eris.vooo[p0:p1] = tmp[:,:,:nocc,:nocc]
        eris.voov[p0:p1] = tmp[:,:,:nocc,nocc:]
        eris.vovo[p0:p1] = tmp[:,:,nocc:,:nocc]
        eris.vovv[p0:p1] = tmp[:,:,nocc:,nocc:]

        tmp = buf[ tril2sq[q0:q1,nocc:q1] - off0 ]
        eris.vvoo[p0:p1,:p1] = tmp[:,:,:nocc,:nocc]
        eris.vvvv[p0:p1,:p1] = tmp[:,:,nocc:,nocc:]
        if p0 > 0:
            eris.vvoo[:p0,p0:p1] = tmp[:,:p0,:nocc,:nocc].transpose(1,0,2,3)
            eris.vvvv[:p0,p0:p1] = tmp[:,:p0,nocc:,nocc:].transpose(1,0,2,3)
        buf = tmp = None
    log.timer('CCSD integral transformation', *cput0)
    return eris


def _cp(a):
    return np.array(a, copy=False, order='C')


if __name__ == '__main__':
    from functools import reduce
    from pyscf import scf
    from pyscf import gto

    mol = gto.M()
    nocc, nvir = 5, 12
    nmo = nocc + nvir
    nmo_pair = nmo*(nmo+1)//2
    mf = scf.RHF(mol)
    np.random.seed(12)
    mf._eri = np.random.random(nmo_pair*(nmo_pair+1)//2)
    mf.mo_coeff = np.random.random((nmo,nmo))
    mf.mo_energy = np.arange(0., nmo)
    mf.mo_occ = np.zeros(nmo)
    mf.mo_occ[:nocc] = 2
    vhf = mf.get_veff(mol, mf.make_rdm1())
    cinv = np.linalg.inv(mf.mo_coeff)
    mf.get_hcore = lambda *args: (reduce(np.dot, (cinv.T*mf.mo_energy, cinv)) - vhf)
    mycc = RCCSD(mf)
    eris = mycc.ao2mo()
    a = np.random.random((nmo,nmo)) * .1
    eris.fock += a + a.T.conj()
    t1 = np.random.random((nocc,nvir)) * .1
    t2 = np.random.random((nocc,nocc,nvir,nvir)) * .1
    t2 = t2 + t2.transpose(1,0,3,2)

    mycc.cc2 = False
    t1a, t2a = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1a) - -106360.5276951083)
    print(lib.finger(t2a) - 66540.100267798145)
    mycc.cc2 = True
    t1a, t2a = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1a) - -106360.5276951083)
    print(lib.finger(t2a) - -1517.9391800662809)

    eri1 = np.random.random((nmo,nmo,nmo,nmo)) + np.random.random((nmo,nmo,nmo,nmo))*1j
    eri1 = eri1.transpose(0,2,1,3)
    eri1 = eri1 + eri1.transpose(1,0,3,2).conj()
    eri1 = eri1 + eri1.transpose(2,3,0,1)
    eri1 *= .1
    eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    eris.vooo = eri1[nocc:,:nocc,:nocc,:nocc].copy()
    eris.vovo = eri1[nocc:,:nocc,nocc:,:nocc].copy()
    eris.vvoo = eri1[nocc:,nocc:,:nocc,:nocc].copy()
    eris.voov = eri1[nocc:,:nocc,:nocc,nocc:].copy()
    eris.vovv = eri1[nocc:,:nocc,nocc:,nocc:].copy()
    eris.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
    a = np.random.random((nmo,nmo)) * .1j
    eris.fock = eris.fock + a + a.T.conj()

    t1 = t1 + np.random.random((nocc,nvir)) * .1j
    t2 = t2 + np.random.random((nocc,nocc,nvir,nvir)) * .1j
    t2 = t2 + t2.transpose(1,0,3,2)
    mycc.cc2 = False
    t1a, t2a = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1a) - (-13.32050019680894-1.8825765910430254j))
    print(lib.finger(t2a) - (9.2521062044785189+29.999480274811873j))
    mycc.cc2 = True
    t1a, t2a = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1a) - (-13.32050019680894-1.8825765910430254j))
    print(lib.finger(t2a) - (-0.056223856104895858+0.025472249329733986j))

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    #mol.basis = '3-21G'
    mol.verbose = 0
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)

    mycc = RCCSD(mf)
    mycc.max_memory = 0
    eris = mycc.ao2mo()
    emp2, t1, t2 = mycc.init_amps(eris)
    print(lib.finger(t2) - 0.044540097905897198)
    np.random.seed(1)
    t1 = np.random.random(t1.shape)*.1
    t2 = np.random.random(t2.shape)*.1
    t2 = t2 + t2.transpose(1,0,3,2)
    t1, t2 = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1) - 0.25118555558133576)
    print(lib.finger(t2) - 0.02352137419932243)

    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.21334326214236796)

    e, v = mycc.ipccsd(nroots=3)
    print(e[0] - 0.43356041409195489)
    print(e[1] - 0.51876598058509493)
    print(e[2] - 0.6782879569941862 )

    e, v = mycc.eeccsd(nroots=4)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)
