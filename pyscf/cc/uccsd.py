'''
UCCSD with spatial integrals
'''

import time
import tempfile
from functools import reduce
import numpy
import numpy as np
import h5py

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import rccsd
from pyscf.lib import linalg_helper
from pyscf.cc import uintermediates as imd
from pyscf.cc.addons import spatial2spin, spin2spatial

#einsum = np.einsum
einsum = lib.einsum

# This is unrestricted (U)CCSD, i.e. spin-orbital form.

def kernel(cc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6,
           verbose=logger.INFO):
    """Exactly the same as pyscf.cc.ccsd.kernel, which calls a
    *local* energy() function."""
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cc.stdout, verbose)

    r1, r2 = cc.init_amps(eris)[1:]
    if t1 is None:
        t1 = r1
    if t2 is None:
        t2 = r2
    r1 = r2 = None

    cput1 = cput0 = (time.clock(), time.time())
    eold = 0
    eccsd = 0
    if cc.diis:
        adiis = lib.diis.DIIS(cc, cc.diis_file)
        adiis.space = cc.diis_space

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = cc.update_amps(t1, t2, eris)
        vec = cc.amplitudes_to_vector(t1new, t2new)
        normt = np.linalg.norm(vec - cc.amplitudes_to_vector(t1, t2))
        t1, t2 = t1new, t2new
        t1new = t2new = None
        if cc.diis:
            t1, t2 = cc.diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, cc.energy(t1, t2, eris)
        log.info('istep = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2

def update_amps(cc, t1, t2, eris):
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape

    fooa = eris.focka[:nocca,:nocca]
    foob = eris.fockb[:noccb,:noccb]
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]
    fvva = eris.focka[nocca:,nocca:]
    fvvb = eris.fockb[noccb:,noccb:]

    u1a = np.zeros_like(t1a)
    u1b = np.zeros_like(t1b)
    u2aa = np.zeros_like(t2aa)
    u2ab = np.zeros_like(t2ab)
    u2bb = np.zeros_like(t2bb)

    tauaa, tauab, taubb = make_tau(t2, t1, t1)

    Fooa  = fooa - np.diag(np.diag(fooa))
    Foob  = foob - np.diag(np.diag(foob))
    Fvva  = fvva - np.diag(np.diag(fvva))
    Fvvb  = fvvb - np.diag(np.diag(fvvb))
    Fooa += .5 * lib.einsum('me,ie->mi', fova, t1a)
    Foob += .5 * lib.einsum('me,ie->mi', fovb, t1b)
    Fvva -= .5 * lib.einsum('me,ma->ae', fova, t1a)
    Fvvb -= .5 * lib.einsum('me,ma->ae', fovb, t1b)
    wovvo = np.zeros((nocca,nvira,nvira,nocca))
    wOVVO = np.zeros((noccb,nvirb,nvirb,noccb))
    woVvO = np.zeros((nocca,nvirb,nvira,noccb))
    woVVo = np.zeros((nocca,nvirb,nvirb,nocca))
    wOvVo = np.zeros((noccb,nvira,nvirb,nocca))
    wOvvO = np.zeros((noccb,nvira,nvira,noccb))

    mem_now = lib.current_memory()[0]
    max_memory = lib.param.MAX_MEMORY - mem_now
    blksize = max(int(max_memory*1e6/8/(nvira**3*3+1)), 2)
    for p0,p1 in lib.prange(0, nocca, blksize):
        ovvv = np.asarray(eris.ovvv[p0:p1]).reshape((p1-p0)*nvira,nvira*(nvira+1)//2)
        ovvv = lib.unpack_tril(ovvv).reshape(p1-p0,nvira,nvira,nvira)
        ovvv = ovvv - ovvv.transpose(0,3,2,1)
        Fvva += np.einsum('mf,mfae->ae', t1a[p0:p1], ovvv)
        wovvo[p0:p1] += einsum('jf,mebf->mbej', t1a, ovvv)
        u1a += 0.5*lib.einsum('mief,meaf->ia', t2aa[p0:p1], ovvv)
        u2aa[:,p0:p1] += lib.einsum('ie,mbea->imab', t1a, ovvv.conj())
        tmp1aa = lib.einsum('ijef,mebf->ijmb', tauaa, ovvv)
        u2aa -= lib.einsum('ijmb,ma->ijab', tmp1aa, t1a[p0:p1]*.5)
        ovvv = tmp1aa = None

    blksize = max(int(max_memory*1e6/8/(nvirb**3*3+1)), 2)
    for p0,p1 in lib.prange(0, noccb, blksize):
        OVVV = np.asarray(eris.OVVV[p0:p1]).reshape((p1-p0)*nvirb,nvirb*(nvirb+1)//2)
        OVVV = lib.unpack_tril(OVVV).reshape(p1-p0,nvirb,nvirb,nvirb)
        OVVV = OVVV - OVVV.transpose(0,3,2,1)
        Fvvb += np.einsum('mf,mfae->ae', t1b[p0:p1], OVVV)
        wOVVO[p0:p1] = einsum('jf,mebf->mbej', t1b, OVVV)
        u1b += 0.5*lib.einsum('MIEF,MEAF->IA', t2bb[p0:p1], OVVV)
        u2bb[:,p0:p1] += lib.einsum('ie,mbea->imab', t1b, OVVV.conj())
        tmp1bb = lib.einsum('ijef,mebf->ijmb', taubb, OVVV)
        u2bb -= lib.einsum('ijmb,ma->ijab', tmp1bb, t1b[p0:p1]*.5)
        OVVV = tmp1bb = None

    blksize = max(int(max_memory*1e6/8/(nvira*nvirb**2*3+1)), 2)
    for p0,p1 in lib.prange(0, nocca, blksize):
        ovVV = np.asarray(eris.ovVV[p0:p1]).reshape((p1-p0)*nvira,nvirb*(nvirb+1)//2)
        ovVV = lib.unpack_tril(ovVV).reshape(p1-p0,nvira,nvirb,nvirb)
        Fvvb += np.einsum('mf,mfAE->AE', t1a[p0:p1], ovVV)
        woVvO[p0:p1] = einsum('JF,meBF->mBeJ', t1b, ovVV)
        woVVo[p0:p1] = einsum('jf,mfBE->mBEj',-t1a, ovVV)
        u1b += lib.einsum('mIeF,meAF->IA', t2ab[p0:p1], ovVV)
        u2ab[p0:p1] += lib.einsum('IE,maEB->mIaB', t1b, ovVV.conj())
        tmp1ab = lib.einsum('iJeF,meBF->iJmB', tauab, ovVV)
        u2ab -= lib.einsum('iJmB,ma->iJaB', tmp1ab, t1a[p0:p1])
        ovVV = tmp1ab = None

    blksize = max(int(max_memory*1e6/8/(nvirb*nocca**2*3+1)), 2)
    for p0,p1 in lib.prange(0, noccb, blksize):
        OVvv = np.asarray(eris.OVvv[p0:p1]).reshape((p1-p0)*nvirb,nvira*(nvira+1)//2)
        OVvv = lib.unpack_tril(OVvv).reshape(p1-p0,nvirb,nvira,nvira)
        Fvva += np.einsum('MF,MFae->ae', t1b[p0:p1], OVvv)
        wOvVo[p0:p1] = einsum('jf,MEbf->MbEj', t1a, OVvv)
        wOvvO[p0:p1] = einsum('JF,MFbe->MbeJ',-t1b, OVvv)
        u1a += lib.einsum('iMfE,MEaf->ia', t2ab[:,p0:p1], OVvv)
        u2ab[:,p0:p1] += lib.einsum('ie,MBea->iMaB', t1a, OVvv.conj())
        tmp1abba = lib.einsum('iJeF,MFbe->iJbM', tauab, OVvv)
        u2ab -= lib.einsum('iJbM,MA->iJbA', tmp1abba, t1b[p0:p1])
        OVvv = tmp1abba = None

    eris_ovov = np.asarray(eris.ovov)
    eris_ooov = np.asarray(eris.ooov)
    Woooo = lib.einsum('je,mine->mnij', t1a, eris_ooov)
    Woooo = Woooo - Woooo.transpose(0,1,3,2)
    Woooo += np.asarray(eris.oooo).transpose(0,2,1,3)
    Woooo += lib.einsum('ijef,menf->mnij', tauaa, eris_ovov) * .5
    u2aa += lib.einsum('mnab,mnij->ijab', tauaa, Woooo*.5)
    Woooo = tauaa = None
    ooov = eris_ooov - eris_ooov.transpose(2,1,0,3)
    Fooa += np.einsum('ne,mine->mi', t1a, ooov)
    u1a += 0.5*lib.einsum('mnae,nime->ia', t2aa, ooov)
    wovvo += einsum('nb,mjne->mbej', t1a, ooov)
    ooov = eris_ooov = None

    tilaa = make_tau_aa(t2[0], t1a, t1a, fac=0.5)
    ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
    Fvva -= .5 * einsum('mnaf,menf->ae', tilaa, ovov)
    Fooa += .5 * einsum('inef,menf->mi', tilaa, ovov)
    Fova = np.einsum('nf,menf->me',t1a, ovov)
    u2aa += ovov.conj().transpose(0,2,1,3) * .5
    wovvo -= 0.5*einsum('jnfb,menf->mbej', t2aa, ovov)
    woVvO += 0.5*einsum('nJfB,menf->mBeJ', t2ab, ovov)
    tmpaa = einsum('jf,menf->mnej', t1a, ovov)
    wovvo -= einsum('nb,mnej->mbej', t1a, tmpaa)
    eirs_ovov = ovov = tmpaa = tilaa = None

    eris_OVOV = np.asarray(eris.OVOV)
    eris_OOOV = np.asarray(eris.OOOV)
    WOOOO = lib.einsum('je,mine->mnij', t1b, eris_OOOV)
    WOOOO = WOOOO - WOOOO.transpose(0,1,3,2)
    WOOOO += np.asarray(eris.OOOO).transpose(0,2,1,3)
    WOOOO += lib.einsum('ijef,menf->mnij', taubb, eris_OVOV) * .5
    u2bb += lib.einsum('mnab,mnij->ijab', taubb, WOOOO*.5)
    WOOOO = taubb = None
    OOOV = eris_OOOV - eris_OOOV.transpose(2,1,0,3)
    Foob += np.einsum('ne,mine->mi', t1b, OOOV)
    u1b += 0.5*lib.einsum('mnae,nime->ia', t2bb, OOOV)
    wOVVO += einsum('nb,mjne->mbej', t1b, OOOV)
    OOOV = eris_OOOV = None

    tilbb = make_tau_aa(t2[2], t1b, t1b, fac=0.5)
    OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
    Fvvb -= .5 * einsum('MNAF,MENF->AE', tilbb, OVOV)
    Foob += .5 * einsum('inef,menf->mi', tilbb, OVOV)
    Fovb = np.einsum('nf,menf->me',t1b, OVOV)
    u2bb += OVOV.conj().transpose(0,2,1,3) * .5
    wOVVO -= 0.5*einsum('jnfb,menf->mbej', t2bb, OVOV)
    wOvVo += 0.5*einsum('jNbF,MENF->MbEj', t2ab, OVOV)
    tmpbb = einsum('jf,menf->mnej', t1b, OVOV)
    wOVVO -= einsum('nb,mnej->mbej', t1b, tmpbb)
    eris_OVOV = OVOV = tmpbb = tilbb = None

    eris_ooOV = np.asarray(eris.ooOV)
    eris_OOov = np.asarray(eris.OOov)
    Fooa += np.einsum('NE,miNE->mi', t1b, eris_ooOV)
    u1a -= lib.einsum('nMaE,niME->ia', t2ab, eris_ooOV)
    wOvVo -= einsum('nb,njME->MbEj', t1a, eris_ooOV)
    woVVo += einsum('NB,mjNE->mBEj', t1b, eris_ooOV)
    Foob += np.einsum('ne,MIne->MI', t1a, eris_OOov)
    u1b -= lib.einsum('mNeA,NIme->IA', t2ab, eris_OOov)
    woVvO -= einsum('NB,NJme->mBeJ', t1b, eris_OOov)
    wOvvO += einsum('nb,MJne->MbeJ', t1a, eris_OOov)
    WoOoO = lib.einsum('JE,miNE->mNiJ', t1b, eris_ooOV)
    WoOoO+= lib.einsum('je,MIne->nMjI', t1a, eris_OOov)
    WoOoO += np.asarray(eris.ooOO).transpose(0,2,1,3)
    eris_ooOV = eris_OOov = None

    eris_ovOV = np.asarray(eris.ovOV)
    WoOoO += lib.einsum('iJeF,meNF->mNiJ', tauab, eris_ovOV)
    u2ab += lib.einsum('mNaB,mNiJ->iJaB', tauab, WoOoO)
    WoOoO = None

    tilab = make_tau_ab(t2[1], t1 , t1 , fac=0.5)
    Fvva -= einsum('mNaF,meNF->ae', tilab, eris_ovOV)
    Fvvb -= einsum('nMfA,nfME->AE', tilab, eris_ovOV)
    Fooa += einsum('iNeF,meNF->mi', tilab, eris_ovOV)
    Foob += einsum('nIfE,nfME->MI', tilab, eris_ovOV)
    Fova+= np.einsum('NF,meNF->me',t1b, eris_ovOV)
    Fovb+= np.einsum('nf,nfME->ME',t1a, eris_ovOV)
    u2ab += eris_ovOV.conj().transpose(0,2,1,3)
    wovvo += 0.5*einsum('jNbF,meNF->mbej', t2ab, eris_ovOV)
    wOVVO += 0.5*einsum('nJfB,nfME->MBEJ', t2ab, eris_ovOV)
    wOvVo -= 0.5*einsum('jnfb,nfME->MbEj', t2aa, eris_ovOV)
    woVvO -= 0.5*einsum('JNFB,meNF->mBeJ', t2bb, eris_ovOV)
    woVVo += 0.5*einsum('jNfB,mfNE->mBEj', t2ab, eris_ovOV)
    wOvvO += 0.5*einsum('nJbF,neMF->MbeJ', t2ab, eris_ovOV)
    tmpabab = einsum('JF,meNF->mNeJ', t1b, eris_ovOV)
    tmpbaba = einsum('jf,nfME->MnEj', t1a, eris_ovOV)
    woVvO -= einsum('NB,mNeJ->mBeJ', t1b, tmpabab)
    wOvVo -= einsum('nb,MnEj->MbEj', t1a, tmpbaba)
    woVVo += einsum('NB,NmEj->mBEj', t1b, tmpbaba)
    wOvvO += einsum('nb,nMeJ->MbeJ', t1a, tmpabab)
    tmpabab = tmpbaba = tilab = None

    u1a += fova.conj()
    u1a += np.einsum('ie,ae->ia',t1a,Fvva)
    u1a -= np.einsum('ma,mi->ia',t1a,Fooa)
    u1a -= np.einsum('imea,me->ia', t2aa, Fova)
    u1a += np.einsum('iMaE,ME->ia', t2ab, Fovb)
    u1b += fovb.conj()
    u1b += np.einsum('ie,ae->ia',t1b,Fvvb)
    u1b -= np.einsum('ma,mi->ia',t1b,Foob)
    u1b -= np.einsum('imea,me->ia', t2bb, Fovb)
    u1b += np.einsum('mIeA,me->IA', t2ab, Fova)

    eris_oovv = np.asarray(eris.oovv)
    eris_ovvo = np.asarray(eris.ovvo)
    wovvo -= eris_oovv.transpose(0,2,3,1)
    wovvo += eris_ovvo.transpose(0,2,1,3)
    oovv = eris_oovv - eris_ovvo.transpose(0,3,2,1)
    u1a-= np.einsum('nf,niaf->ia', t1a,      oovv)
    tmp1aa = lib.einsum('ie,mjbe->mbij', t1a,      oovv)
    u2aa += 2*lib.einsum('ma,mbij->ijab', t1a, tmp1aa)
    eris_ovvo = eris_oovv = oovv = tmp1aa = None

    eris_OOVV = np.asarray(eris.OOVV)
    eris_OVVO = np.asarray(eris.OVVO)
    wOVVO -= eris_OOVV.transpose(0,2,3,1)
    wOVVO += eris_OVVO.transpose(0,2,1,3)
    OOVV = eris_OOVV - eris_OVVO.transpose(0,3,2,1)
    u1b-= np.einsum('nf,niaf->ia', t1b,      OOVV)
    tmp1bb = lib.einsum('ie,mjbe->mbij', t1b,      OOVV)
    u2bb += 2*lib.einsum('ma,mbij->ijab', t1b, tmp1bb)
    eris_OVVO = eris_OOVV = OOVV = None

    eris_ooVV = np.asarray(eris.ooVV)
    eris_ovVO = np.asarray(eris.ovVO)
    woVVo -= eris_ooVV.transpose(0,2,3,1)
    woVvO += eris_ovVO.transpose(0,2,1,3)
    u1b+= np.einsum('nf,nfAI->IA', t1a, eris_ovVO)
    tmp1ab = lib.einsum('ie,meBJ->mBiJ', t1a, eris_ovVO)
    tmp1ab+= lib.einsum('IE,mjBE->mBjI', t1b, eris_ooVV)
    u2ab -= lib.einsum('ma,mBiJ->iJaB', t1a, tmp1ab)
    eris_ooVV = eris_ovVo = tmp1ab = None

    eris_OOvv = np.asarray(eris.OOvv)
    eris_OVvo = np.asarray(eris.OVvo)
    wOvvO -= eris_OOvv.transpose(0,2,3,1)
    wOvVo += eris_OVvo.transpose(0,2,1,3)
    u1a+= np.einsum('NF,NFai->ia', t1b, eris_OVvo)
    tmp1ba = lib.einsum('IE,MEbj->MbIj', t1b, eris_OVvo)
    tmp1ba+= lib.einsum('ie,MJbe->MbJi', t1a, eris_OOvv)
    u2ab -= lib.einsum('MA,MbIj->jIbA', t1b, tmp1ba)
    eris_OOvv = eris_OVvO = tmp1ba = None

    u2aa += 2*lib.einsum('imae,mbej->ijab', t2aa, wovvo)
    u2aa += 2*lib.einsum('iMaE,MbEj->ijab', t2ab, wOvVo)
    u2bb += 2*lib.einsum('imae,mbej->ijab', t2bb, wOVVO)
    u2bb += 2*lib.einsum('mIeA,mBeJ->IJAB', t2ab, woVvO)
    u2ab += lib.einsum('imae,mBeJ->iJaB', t2aa, woVvO)
    u2ab += lib.einsum('iMaE,MBEJ->iJaB', t2ab, wOVVO)
    u2ab += lib.einsum('iMeA,MbeJ->iJbA', t2ab, wOvvO)
    u2ab += lib.einsum('IMAE,MbEj->jIbA', t2bb, wOvVo)
    u2ab += lib.einsum('mIeA,mbej->jIbA', t2ab, wovvo)
    u2ab += lib.einsum('mIaE,mBEj->jIaB', t2ab, woVVo)
    wovvo = wOVVO = woVvO = wOvVo = woVVo = wOvvO = None

    Ftmpa = Fvva - .5*lib.einsum('mb,me->be',t1a,Fova)
    Ftmpb = Fvvb - .5*lib.einsum('mb,me->be',t1b,Fovb)
    u2aa += lib.einsum('ijae,be->ijab', t2aa, Ftmpa)
    u2bb += lib.einsum('ijae,be->ijab', t2bb, Ftmpb)
    u2ab += lib.einsum('iJaE,BE->iJaB', t2ab, Ftmpb)
    u2ab += lib.einsum('iJeA,be->iJbA', t2ab, Ftmpa)
    Ftmpa = Fooa + 0.5*lib.einsum('je,me->mj', t1a, Fova)
    Ftmpb = Foob + 0.5*lib.einsum('je,me->mj', t1b, Fovb)
    u2aa -= lib.einsum('imab,mj->ijab', t2aa, Ftmpa)
    u2bb -= lib.einsum('imab,mj->ijab', t2bb, Ftmpb)
    u2ab -= lib.einsum('iMaB,MJ->iJaB', t2ab, Ftmpb)
    u2ab -= lib.einsum('mIaB,mj->jIaB', t2ab, Ftmpa)

    #:eris_vvvv = ao2mo.restore(1, np.asarray(eris.vvvv), nvirb)
    #:eris_VVVV = ao2mo.restore(1, np.asarray(eris.VVVV), nvirb)
    #:eris_vvVV = _restore(np.asarray(eris.vvVV), nvira, nvirb)
    #:u2aa += lib.einsum('ijef,aebf->ijab', tauaa, eris_vvvv) * .5
    #:u2bb += lib.einsum('ijef,aebf->ijab', taubb, eris_VVVV) * .5
    #:u2ab += lib.einsum('iJeF,aeBF->iJaB', tauab, eris_vvVV)
    tauaa, tauab, taubb = make_tau(t2, t1, t1)
    _add_vvvv_(cc, (tauaa,tauab,taubb), eris, (u2aa,u2ab,u2bb))

    eris_oovo = numpy.asarray(eris.oovo)
    eris_OOVO = numpy.asarray(eris.OOVO)
    eris_ooVO = numpy.asarray(eris.ooVO)
    eris_OOvo = numpy.asarray(eris.OOvo)
    oovo = eris_oovo - eris_oovo.transpose(0,3,2,1)
    OOVO = eris_OOVO - eris_OOVO.transpose(0,3,2,1)
    u2aa -= lib.einsum('ma,mibj->ijab', t1a, oovo)
    u2bb -= lib.einsum('ma,mibj->ijab', t1b, OOVO)
    u2ab -= lib.einsum('ma,miBJ->iJaB', t1a, eris_ooVO)
    u2ab -= lib.einsum('MA,MJbi->iJbA', t1b, eris_OOvo)
    eris_oovo = eris_ooVO = eris_OOVO = eris_OOvo = None

    u2aa *= .5
    u2bb *= .5
    u2aa = u2aa - u2aa.transpose(0,1,3,2)
    u2aa = u2aa - u2aa.transpose(1,0,2,3)
    u2bb = u2bb - u2bb.transpose(0,1,3,2)
    u2bb = u2bb - u2bb.transpose(1,0,2,3)

    eia_a = lib.direct_sum('i-a->ia', fooa.diagonal(), fvva.diagonal())
    eia_b = lib.direct_sum('i-a->ia', foob.diagonal(), fvvb.diagonal())
    u1a /= eia_a
    u1b /= eia_b

    u2aa /= lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
    u2ab /= lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
    u2bb /= lib.direct_sum('ia+jb->ijab', eia_b, eia_b)

    time0 = log.timer_debug1('update t1 t2', *time0)
    t1new = u1a, u1b
    t2new = u2aa, u2ab, u2bb
    return t1new, t2new


def get_nocc(mycc):
    if mycc._nocc is not None:
        return mycc._nocc
    if isinstance(mycc.frozen, (int, numpy.integer)):
        nocca = int(mycc.mo_occ[0].sum()) - mycc.frozen
        noccb = int(mycc.mo_occ[1].sum()) - mycc.frozen
        #assert(nocca > 0 and noccb > 0)
    else:
        frozen = mycc.frozen
        if len(frozen) > 0 and isinstance(frozen[0], (int, numpy.integer)):
# The same frozen orbital indices for alpha and beta orbitals
            frozen = [frozen, frozen]
        occidxa = mycc.mo_occ[0] > 0
        occidxa[list(frozen[0])] = False
        occidxb = mycc.mo_occ[1] > 0
        occidxb[list(frozen[1])] = False
        nocca = np.count_nonzero(occidxa)
        noccb = np.count_nonzero(occidxb)
    return nocca, noccb

def get_nmo(mycc):
    if mycc._nmo is not None:
        return mycc._nmo
    elif isinstance(mycc.frozen, (int, numpy.integer)):
        nmoa = mycc.mo_occ[0].size - mycc.frozen
        nmob = mycc.mo_occ[1].size - mycc.frozen
    elif isinstance(mycc.frozen[0], (int, numpy.integer)):
        nmoa = mycc.mo_occ[0].size - len(mycc.frozen)
        nmob = mycc.mo_occ[1].size - len(mycc.frozen)
    else:
        nmoa = len(mycc.mo_occ[0]) - len(mycc.frozen[0])
        nmob = len(mycc.mo_occ[1]) - len(mycc.frozen[1])
    return nmoa, nmob


def energy(cc, t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]
    e  = np.einsum('ia,ia', fova, t1a)
    e += np.einsum('ia,ia', fovb, t1b)
    e += 0.25*np.einsum('ijab,iajb',t2aa,eris_ovov)
    e -= 0.25*np.einsum('ijab,ibja',t2aa,eris_ovov)
    e += 0.25*np.einsum('ijab,iajb',t2bb,eris_OVOV)
    e -= 0.25*np.einsum('ijab,ibja',t2bb,eris_OVOV)
    e +=      np.einsum('iJaB,iaJB',t2ab,eris_ovOV)
    e += 0.5*np.einsum('ia,jb,iajb',t1a,t1a,eris_ovov)
    e -= 0.5*np.einsum('ia,jb,ibja',t1a,t1a,eris_ovov)
    e += 0.5*np.einsum('ia,jb,iajb',t1b,t1b,eris_OVOV)
    e -= 0.5*np.einsum('ia,jb,ibja',t1b,t1b,eris_OVOV)
    e +=     np.einsum('ia,jb,iajb',t1a,t1b,eris_ovOV)
    return e.real



class UCCSD(rccsd.RCCSD):

# argument frozen can be
# * An integer : The same number of inner-most alpha and beta orbitals are frozen
# * One list : Same alpha and beta orbital indices to be frozen
# * A pair of list : First list is the orbital indices to be frozen for alpha
#       orbitals, second list is for beta orbitals
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        rccsd.RCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        # Spin-orbital CCSD needs a stricter tolerance than spatial-orbital
        self.conv_tol_normt = 1e-6
        if getattr(mf, 'mo_energy', None) is not None:
            self.orbspin = orbspin_of_sorted_mo_energy(mf.mo_energy, self.mo_occ)
        else:
            self.orbspin = None
        self._keys = self._keys.union(['orbspin'])

    def build(self):
        '''Initialize integrals and orbspin'''
        self.orbspin = None

    @property
    def nocc(self):
        nocca, noccb = self.get_nocc()
        return nocca + noccb

    @property
    def nmo(self):
        nmoa, nmob = self.get_nmo()
        return nmoa + nmob

    get_nocc = get_nocc
    get_nmo = get_nmo

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        nocca, noccb = self.get_nocc()

        fooa = eris.focka[:nocca,:nocca]
        foob = eris.fockb[:noccb,:noccb]
        fova = eris.focka[:nocca,nocca:]
        fovb = eris.fockb[:noccb,noccb:]
        fvva = eris.focka[nocca:,nocca:]
        fvvb = eris.fockb[noccb:,noccb:]
        eia_a = lib.direct_sum('i-a->ia', fooa.diagonal(), fvva.diagonal())
        eia_b = lib.direct_sum('i-a->ia', foob.diagonal(), fvvb.diagonal())

        t1a = fova.conj() / eia_a
        t1b = fovb.conj() / eia_b

        eris_ovov = np.asarray(eris.ovov)
        eris_OVOV = np.asarray(eris.OVOV)
        eris_ovOV = np.asarray(eris.ovOV)
        t2aa = eris_ovov.transpose(0,2,1,3) / lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
        t2ab = eris_ovOV.transpose(0,2,1,3) / lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
        t2bb = eris_OVOV.transpose(0,2,1,3) / lib.direct_sum('ia+jb->ijab', eia_b, eia_b)
        t2aa = t2aa - t2aa.transpose(0,1,3,2)
        t2bb = t2bb - t2bb.transpose(0,1,3,2)
        e  =      np.einsum('iJaB,iaJB', t2ab, eris_ovOV)
        e += 0.25*np.einsum('ijab,iajb', t2aa, eris_ovov)
        e -= 0.25*np.einsum('ijab,ibja', t2aa, eris_ovov)
        e += 0.25*np.einsum('ijab,iajb', t2bb, eris_OVOV)
        e -= 0.25*np.einsum('ijab,ibja', t2bb, eris_OVOV)
        self.emp2 = e.real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, (t1a,t1b), (t2aa,t2ab,t2bb)

    energy = energy

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        return self.ccsd(t1, t2, eris, mbpt2)
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state unrestricted (U)CCSD.

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
            cctyp = 'UCCSD'
            self.converged, self.e_corr, self.t1, self.t2 = \
                    kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                           tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                           verbose=self.verbose)
            if self.converged:
                logger.info(self, 'UCCSD converged')
            else:
                logger.note(self, 'UCCSD not converged')
        if self._scf.e_tot == 0:
            logger.note(self, 'E_corr = %.16g', self.e_corr)
        else:
            logger.note(self, 'E(%s) = %.16g  E_corr = %.16g',
                        cctyp, self.e_tot, self.e_corr)
        return self.e_corr, self.t1, self.t2

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None):
        from pyscf.cc import uccsd_lambda_slow as uccsd_lambda
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                uccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                    max_cycle=self.max_cycle,
                                    tol=self.conv_tol_normt,
                                    verbose=self.verbose)
        return self.l1, self.l2

    def ccsd_t(self, t1=None, t2=None, eris=None):
        from pyscf.cc import uccsd_t
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        return uccsd_t.kernel(self, eris, t1, t2, self.verbose)
    uccsd_t = ccsd_t

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None):
        '''Un-relaxed 1-particle density matrix in MO space

        Returns:
            dm1a, dm1b
        '''
        from pyscf.cc import uccsd_rdm_slow as uccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return uccsd_rdm.make_rdm1(self, t1, t2, l1, l2)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None):
        '''2-particle density matrix in spin-oribital basis.
        '''
        from pyscf.cc import uccsd_rdm_slow as uccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return uccsd_rdm.make_rdm2(self, t1, t2, l1, l2)

    def ao2mo(self, mo_coeff=None):
        return _ERIS(self, mo_coeff)

    def update_amps(self, t1, t2, eris):
        return update_amps(self, t1, t2, eris)

    def nip(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        self._nip = nocc + nocc*(nocc-1)//2*nvir
        return self._nip

    def nea(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        self._nea = nvir + nocc*nvir*(nvir-1)//2
        return self._nea

    def nee(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        self._nee = nocc*nvir + nocc*(nocc-1)//2*nvir*(nvir-1)//2
        return self._nee

    def ipccsd_matvec(self, vector):
        # Ref: Tu, Wang, and Li, J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.eris.__dict__.update(_ERISspin(self).__dict__)
            self.imds.make_ip()
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ip(vector)
        nocc, nvir = r2.shape[1:]
        eris = self.eris

        # Eq. (8)
        Hr1  = np.einsum('me,mie->i',imds.Fov,r2)
        Hr1 -= np.einsum('mi,m->i',imds.Foo,r1)
        Hr1 -= 0.5*np.einsum('nmie,mne->i',imds.Wooov,r2)

        # Eq. (9)
        Hr2 = lib.einsum('ae,ije->ija',imds.Fvv,r2)
        tmp1 = lib.einsum('mi,mja->ija',imds.Foo,r2)
        Hr2 -= tmp1 - tmp1.transpose(1,0,2)
        Hr2 -= np.einsum('maji,m->ija',imds.Wovoo,r1)
        Hr2 += 0.5*lib.einsum('mnij,mna->ija',imds.Woooo,r2)
        tmp2 = lib.einsum('maei,mje->ija',imds.Wovvo,r2)
        Hr2 += tmp2 - tmp2.transpose(1,0,2)

        eris_ovov = np.asarray(eris.ovov)
        tmp = 0.5*np.einsum('menf,mnf->e', eris_ovov, r2)
        tmp-= 0.5*np.einsum('mfne,mnf->e', eris_ovov, r2)
        t2 = spatial2spin(self.t2, eris.orbspin)
        Hr2 += np.einsum('e,ijae->ija', tmp, t2)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def ipccsd_diag(self):
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.eris.__dict__.update(_ERISspin(self).__dict__)
            self.imds.make_ip()
        imds = self.imds

        t1, t2, eris = self.t1, self.t2, self.eris
        t1 = spatial2spin(t1, eris.orbspin)
        t2 = spatial2spin(t2, eris.orbspin)
        nocc, nvir = t1.shape

        Fo = np.diagonal(imds.Foo)
        Fv = np.diagonal(imds.Fvv)
        Hr1 = -Fo
        Hr2  = lib.direct_sum('-i-j+a->ija', Fo, Fo, Fv)

        Woooo = np.asarray(imds.Woooo)
        Woo = np.zeros((nocc,nocc), dtype=t1.dtype)
        Woo += np.einsum('ijij->ij', Woooo)
        Woo -= np.einsum('ijji->ij', Woooo)
        Hr2 += Woo.reshape(nocc,nocc,-1) * .5
        Wov = np.einsum('iaai->ia', imds.Wovvo)
        Hr2 += Wov
        Hr2 += Wov.reshape(nocc,1,nvir)
        eris_ovov = np.asarray(eris.ovov)
        Hr2 -= np.einsum('iajb,ijab->ija', eris_ovov, t2)
        Hr2 -= np.einsum('iajb,ijab->ijb', eris_ovov, t2)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ip(self,vector):
        nocc = self.nocc
        nvir = self.nmo - nocc
        r1 = vector[:nocc].copy()
        r2 = np.zeros((nocc**2,nvir), vector.dtype)
        otril = np.tril_indices(nocc, k=-1)
        r2_tril = vector[nocc:].reshape(-1,nvir)
        lib.takebak_2d(r2, r2_tril, otril[0]*nocc+otril[1], np.arange(nvir))
        lib.takebak_2d(r2,-r2_tril, otril[1]*nocc+otril[0], np.arange(nvir))
        return r1,r2.reshape(nocc,nocc,nvir)

    def amplitudes_to_vector_ip(self,r1,r2):
        nocc = self.nocc
        nvir = self.nmo - nocc
        size = nocc + nocc*(nocc-1)//2*nvir
        vector = np.empty(size, r1.dtype)
        vector[:nocc] = r1.copy()
        otril = np.tril_indices(nocc, k=-1)
        lib.take_2d(r2.reshape(-1,nvir), otril[0]*nocc+otril[1],
                    np.arange(nvir), out=vector[nocc:])
        return vector

    def eaccsd_matvec(self,vector):
        # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.eris.__dict__.update(_ERISspin(self).__dict__)
            self.imds.make_ea()
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ea(vector)
        t1, t2, eris = self.t1, self.t2, self.eris
        t1 = spatial2spin(t1, eris.orbspin)
        t2 = spatial2spin(t2, eris.orbspin)

        Hr1 = np.einsum('ac,c->a',imds.Fvv,r1)
        Hr1 += np.einsum('ld,lad->a',imds.Fov,r2)
        tmp1 = lib.einsum('ac,jcb->jab',imds.Fvv,r2)
        Hr2  = (tmp1 - tmp1.transpose(0,2,1))
        Hr2 -= lib.einsum('lj,lab->jab',imds.Foo,r2)

        eris_ovvv = np.asarray(eris.ovvv)
        Hr1 -= 0.5*np.einsum('lcad,lcd->a',eris_ovvv,r2)
        Hr1 += 0.5*np.einsum('ldac,lcd->a',eris_ovvv,r2)
        tau2 = r2 + np.einsum('jd,c->jcd', t1, r1) * 2
        tau2 = tau2 - tau2.transpose(0,2,1)
        tmp  = lib.einsum('mcad,jcd->maj', eris_ovvv, tau2)
        tmp = lib.einsum('mb,maj->jab', t1, tmp)
        Hr2 += .5 * (tmp - tmp.transpose(0,2,1))

        eris_ovov = np.asarray(eris.ovov)
        tau = imd.make_tau(t2, t1, t1)
        tmp = lib.einsum('menf,jef->mnj', eris_ovov, tau2)
        Hr2 += .25*lib.einsum('mnab,mnj->jab', tau, tmp)

        eris_ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
        tmp  = np.einsum('ndlc,lcd->n', eris_ovov, r2)
        Hr1 += .5 * np.einsum('na,n->a', t1, tmp)

        tmp = np.einsum('kcld,lcd->k', eris_ovov, r2)
        t2 = spatial2spin(self.t2, eris.orbspin)
        Hr2 -= 0.5 * np.einsum('k,kjab->jab', tmp, t2)

        tmp = lib.einsum('lbdj,lad->jab', imds.Wovvo, r2)
        Hr2 += tmp - tmp.transpose(0,2,1)

        Hr2 += np.einsum('abcj,c->jab', imds.Wvvvo, r1)

        eris_vvvv = np.asarray(eris.vvvv)
        Hr2 += 0.5*einsum('acbd,jcd->jab',eris_vvvv,tau2)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def eaccsd_diag(self):
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.eris.__dict__.update(_ERISspin(self).__dict__)
            self.imds.make_ea()
        imds = self.imds

        t1, t2, eris = self.t1, self.t2, self.eris
        t1 = spatial2spin(t1, eris.orbspin)
        t2 = spatial2spin(t2, eris.orbspin)
        nocc, nvir = t1.shape

        Fo = np.diagonal(imds.Foo)
        Fv = np.diagonal(imds.Fvv)
        Hr1 = Fv
        Hr2  = lib.direct_sum('-j+a+b->jab', Fo, Fv, Fv)
        Wov = np.einsum('iaai->ia', imds.Wovvo)
        Hr2 += Wov.reshape(nocc,nvir,1)
        Hr2 += Wov.reshape(nocc,1,nvir)
        eris_ovov = np.asarray(eris.ovov)
        Hr2 -= np.einsum('iajb,ijab->jab', eris_ovov, t2)
        Hr2 -= np.einsum('iajb,ijab->iab', eris_ovov, t2)

        eris_ovvv = np.asarray(eris.ovvv)
        Wvv  = einsum('mb,maab->ab', t1, eris_ovvv)
        Wvv -= einsum('mb,mbaa->ab', t1, eris_ovvv)
        Wvv = Wvv + Wvv.T
        eris_vvvv = np.asarray(eris.vvvv)
        Wvv += np.einsum('aabb->ab', eris_vvvv)
        Wvv -= np.einsum('abba->ab', eris_vvvv)
        tau = imd.make_tau(t2, t1, t1)
        Wvv += 0.5*np.einsum('mnab,manb->ab', tau, eris_ovov)
        Wvv -= 0.5*np.einsum('mnab,mbna->ab', tau, eris_ovov)
        Hr2 += Wvv

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ea(self,vector):
        nocc = self.nocc
        nvir = self.nmo - nocc
        r1 = vector[:nvir].copy()
        r2 = np.zeros((nocc,nvir*nvir), vector.dtype)
        vtril = np.tril_indices(nvir, k=-1)
        r2_tril = vector[nvir:].reshape(nocc,-1)
        lib.takebak_2d(r2, r2_tril, np.arange(nocc), vtril[0]*nvir+vtril[1])
        lib.takebak_2d(r2,-r2_tril, np.arange(nocc), vtril[1]*nvir+vtril[0])
        return r1,r2.reshape(nocc,nvir,nvir)

    def amplitudes_to_vector_ea(self,r1,r2):
        nocc = self.nocc
        nvir = self.nmo - nocc
        size = nvir + nvir*(nvir-1)//2*nocc
        vector = np.empty(size, r1.dtype)
        vector[:nvir] = r1.copy()
        vtril = np.tril_indices(nvir, k=-1)
        lib.take_2d(r2.reshape(nocc,-1), np.arange(nocc),
                    vtril[0]*nvir+vtril[1], out=vector[nvir:])
        return vector

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

        if hasattr(self,'imds') and (self.imds.made_ip_imds or self.imds.made_ea_imds):
            self.orbspin = orbspin_of_sorted_mo_energy(self._scf.mo_energy, self.mo_occ)
            self.eris = self.ao2mo(self.mo_coeff)
            self.imds = _IMDS(self)

        diag_ee, diag_sf = self.eeccsd_diag()
        guess_ee = []
        guess_sf = []
        if guess and guess[0].size == spinvec_size:
            for g in guess:
                r1, r2 = self.vector_to_amplitudes_ee(g)
                g = self.amplitudes_to_vector(self.spin2spatial(r1, self.orbspin),
                                              self.spin2spatial(r2, self.orbspin))
                if np.linalg.norm(g) > 1e-7:
                    guess_ee.append(g)
                else:
                    r1 = self.spin2spatial(r1, self.orbspin)
                    r2 = self.spin2spatial(r2, self.orbspin)
                    g = self.amplitudes_to_vector_eomsf(r1, r2)
                    guess_sf.append(g)
                r1 = r2 = None
            nroots_ee = len(guess_ee)
            nroots_sf = len(guess_sf)
        elif guess:
            for g in guess:
                if g.size == diag_ee.size:
                    guess_ee.append(g)
                else:
                    guess_sf.append(g)
            nroots_ee = len(guess_ee)
            nroots_sf = len(guess_sf)
        else:
            dee = np.sort(diag_ee)[:nroots]
            dsf = np.sort(diag_sf)[:nroots]
            dmax = np.sort(np.hstack([dee,dsf]))[nroots-1]
            nroots_ee = np.count_nonzero(dee <= dmax)
            nroots_sf = np.count_nonzero(dsf <= dmax)
            guess_ee = guess_sf = None

        e0 = e1 = []
        v0 = v1 = []
        if nroots_ee > 0:
            e0, v0 = self.eomee_ccsd(nroots_ee, koopmans, guess_ee, diag_ee)
            if nroots_ee == 1:
                e0, v0 = [e0], [v0]
        if nroots_sf > 0:
            e1, v1 = self.eomsf_ccsd(nroots_sf, koopmans, guess_sf, diag_sf)
            if nroots_sf == 1:
                e1, v1 = [e1], [v1]
        e = np.hstack([e0,e1])
        v = v0 + v1
        if nroots == 1:
            return e[0], v[0]
        else:
            idx = e.argsort()
            return e[idx], [v[x] for x in idx]


    def eomee_ccsd(self, nroots=1, koopmans=False, guess=None, diag=None):
        cput0 = (time.clock(), time.time())
        if diag is None:
            diag = self.eeccsd_diag()[0]
        nocca, noccb = self.get_nocc()
        nmoa, nmob = self.get_nmo()
        nvira, nvirb = nmoa-nocca, nmob-noccb

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
                    t1, t2 = self.vector_to_amplitudes(g, (nmoa,nmob), (nocca,noccb))
                    if np.linalg.norm(t1[0]) > .9 or np.linalg.norm(t1[1]) > .9:
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
        matvec = lambda xs: [self.eomee_ccsd_matvec(x) for x in xs]
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
            t1, t2 = self.vector_to_amplitudes(vn, (nmoa,nmob), (nocca,noccb))
            qpwt = np.linalg.norm(t1[0])**2 + np.linalg.norm(t1[1])**2
            logger.info(self, 'EOM-EE root %d E = %.16g  qpwt = %.6g  conv = %s',
                        n, en, qpwt, convn)
        logger.timer(self, 'EOM-EE-CCSD', *cput0)
        if nroots == 1:
            return eee[0], evecs[0]
        else:
            return eee, evecs

    def eomsf_ccsd(self, nroots=1, koopmans=False, guess=None, diag=None):
        cput0 = (time.clock(), time.time())
        if diag is None:
            diag = self.eeccsd_diag()[1]
        nocca, noccb = self.get_nocc()
        nmoa, nmob = self.get_nmo()
        nvira, nvirb = nmoa-nocca, nmob-noccb

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
                    t1, t2 = self.vector_to_amplitudes_eomsf(g, (nocca,noccb), (nvira,nvirb))
                    if np.linalg.norm(t1[0]) > .9 or np.linalg.norm(t1[1]) > .9:
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

        for n, en, vn, convn in zip(range(nroots), eee, evecs, conv):
            t1, t2 = self.vector_to_amplitudes_eomsf(vn, (nocca,noccb), (nvira,nvirb))
            qpwt = np.linalg.norm(t1[0])**2 + np.linalg.norm(t1[1])**2
            logger.info(self, 'EOM-SF root %d E = %.16g  qpwt = %.6g  conv = %s',
                        n, en, qpwt, convn)
        logger.timer(self, 'EOM-SF-CCSD', *cput0)
        if nroots == 1:
            return eee[0], evecs[0]
        else:
            return eee, evecs

    # Ref: Wang, Tu, and Wang, J. Chem. Theory Comput. 10, 5567 (2014) Eqs.(9)-(10)
    # Note: Last line in Eq. (10) is superfluous.
    # See, e.g. Gwaltney, Nooijen, and Barlett, Chem. Phys. Lett. 248, 189 (1996)
    def eomee_ccsd_matvec(self, vector):
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ee_imds:
            self.imds.make_ee()
        imds = self.imds

        r1, r2 = self.vector_to_amplitudes(vector)
        r1a, r1b = r1
        r2aa, r2ab, r2bb = r2
        t1, t2, eris = self.t1, self.t2, self.eris
        t1a, t1b = t1
        t2aa, t2ab, t2bb = t2
        nocca, noccb, nvira, nvirb = t2ab.shape

        Hr1a  = lib.einsum('ae,ie->ia', imds.Fvva, r1a)
        Hr1a -= lib.einsum('mi,ma->ia', imds.Fooa, r1a)
        Hr1a += np.einsum('me,imae->ia',imds.Fova, r2aa)
        Hr1a += np.einsum('ME,iMaE->ia',imds.Fovb, r2ab)
        Hr1b  = lib.einsum('ae,ie->ia', imds.Fvvb, r1b)
        Hr1b -= lib.einsum('mi,ma->ia', imds.Foob, r1b)
        Hr1b += np.einsum('me,imae->ia',imds.Fovb, r2bb)
        Hr1b += np.einsum('me,mIeA->IA',imds.Fova, r2ab)

        Hr2aa = lib.einsum('mnij,mnab->ijab', imds.woooo, r2aa) * .25
        Hr2bb = lib.einsum('mnij,mnab->ijab', imds.wOOOO, r2bb) * .25
        Hr2ab = lib.einsum('mNiJ,mNaB->iJaB', imds.woOoO, r2ab)
        Hr2aa+= lib.einsum('be,ijae->ijab', imds.Fvva, r2aa)
        Hr2bb+= lib.einsum('be,ijae->ijab', imds.Fvvb, r2bb)
        Hr2ab+= lib.einsum('BE,iJaE->iJaB', imds.Fvvb, r2ab)
        Hr2ab+= lib.einsum('be,iJeA->iJbA', imds.Fvva, r2ab)
        Hr2aa-= lib.einsum('mj,imab->ijab', imds.Fooa, r2aa)
        Hr2bb-= lib.einsum('mj,imab->ijab', imds.Foob, r2bb)
        Hr2ab-= lib.einsum('MJ,iMaB->iJaB', imds.Foob, r2ab)
        Hr2ab-= lib.einsum('mj,mIaB->jIaB', imds.Fooa, r2ab)

        #:tau2aa, tau2ab, tau2bb = make_tau(r2, r1, t1, 2)
        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
        #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
        #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
        #:eris_OVVV = lib.unpack_tril(np.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
        #:Hr1a += lib.einsum('mfae,imef->ia', eris_ovvv, r2aa)
        #:tmpaa = lib.einsum('meaf,ijef->maij', eris_ovvv, tau2aa)
        #:Hr2aa+= lib.einsum('mb,maij->ijab', t1a, tmpaa)
        #:tmpa = lib.einsum('mfae,me->af', eris_ovvv, r1a)
        #:tmpa-= lib.einsum('meaf,me->af', eris_ovvv, r1a)

        #:Hr1b += lib.einsum('mfae,imef->ia', eris_OVVV, r2bb)
        #:tmpbb = lib.einsum('meaf,ijef->maij', eris_OVVV, tau2bb)
        #:Hr2bb+= lib.einsum('mb,maij->ijab', t1b, tmpbb)
        #:tmpb = lib.einsum('mfae,me->af', eris_OVVV, r1b)
        #:tmpb-= lib.einsum('meaf,me->af', eris_OVVV, r1b)

        #:Hr1b += lib.einsum('mfAE,mIfE->IA', eris_ovVV, r2ab)
        #:tmpab = lib.einsum('meAF,iJeF->mAiJ', eris_ovVV, tau2ab)
        #:Hr2ab-= lib.einsum('mb,mAiJ->iJbA', t1a, tmpab)
        #:tmpb-= lib.einsum('meAF,me->AF', eris_ovVV, r1a)

        #:Hr1a += lib.einsum('MFae,iMeF->ia', eris_OVvv, r2ab)
        #:tmpba =-lib.einsum('MEaf,iJfE->MaiJ', eris_OVvv, tau2ab)
        #:Hr2ab+= lib.einsum('MB,MaiJ->iJaB', t1b, tmpba)
        #:tmpa-= lib.einsum('MEaf,ME->af', eris_OVvv, r1b)
        tau2aa = make_tau_aa(r2aa, r1a, t1a, 2)
        mem_now = lib.current_memory()[0]
        max_memory = lib.param.MAX_MEMORY - mem_now
        tmpa = np.zeros((nvira,nvira))
        tmpb = np.zeros((nvirb,nvirb))
        blksize = max(int(max_memory*1e6/8/(nvira**3*3)), 2)
        for p0, p1 in lib.prange(0, nocca, blksize):
            ovvv = np.asarray(eris.ovvv[p0:p1]).reshape((p1-p0)*nvira,-1)
            ovvv = lib.unpack_tril(ovvv).reshape(-1,nvira,nvira,nvira)
            Hr1a += lib.einsum('mfae,imef->ia', ovvv, r2aa[:,p0:p1])
            tmpaa = lib.einsum('meaf,ijef->maij', ovvv, tau2aa)
            Hr2aa+= lib.einsum('mb,maij->ijab', t1a[p0:p1], tmpaa)
            tmpa+= lib.einsum('mfae,me->af', ovvv, r1a[p0:p1])
            tmpa-= lib.einsum('meaf,me->af', ovvv, r1a[p0:p1])
            ovvv = tmpaa = None
        tau2aa = None

        tau2bb = make_tau_aa(r2bb, r1b, t1b, 2)
        blksize = max(int(max_memory*1e6/8/(nvirb**3*3)), 2)
        for p0, p1 in lib.prange(0, noccb, blksize):
            OVVV = np.asarray(eris.OVVV[p0:p1]).reshape((p1-p0)*nvirb,-1)
            OVVV = lib.unpack_tril(OVVV).reshape(-1,nvirb,nvirb,nvirb)
            Hr1b += lib.einsum('mfae,imef->ia', OVVV, r2bb[:,p0:p1])
            tmpbb = lib.einsum('meaf,ijef->maij', OVVV, tau2bb)
            Hr2bb+= lib.einsum('mb,maij->ijab', t1b[p0:p1], tmpbb)
            tmpb+= lib.einsum('mfae,me->af', OVVV, r1b[p0:p1])
            tmpb-= lib.einsum('meaf,me->af', OVVV, r1b[p0:p1])
            OVVV = tmpbb = None
        tau2bb = None

        tau2ab = make_tau_ab(r2ab, r1 , t1 , 2)
        blksize = max(int(max_memory*1e6/8/(nvira*nvirb**2*3)), 2)
        for p0, p1 in lib.prange(0, nocca, blksize):
            ovVV = np.asarray(eris.ovVV[p0:p1]).reshape((p1-p0)*nvira,-1)
            ovVV = lib.unpack_tril(ovVV).reshape(-1,nvira,nvirb,nvirb)
            Hr1b += lib.einsum('mfAE,mIfE->IA', ovVV, r2ab[p0:p1])
            tmpab = lib.einsum('meAF,iJeF->mAiJ', ovVV, tau2ab)
            Hr2ab-= lib.einsum('mb,mAiJ->iJbA', t1a[p0:p1], tmpab)
            tmpb-= lib.einsum('meAF,me->AF', ovVV, r1a[p0:p1])
            ovVV = tmpab = None

        blksize = max(int(max_memory*1e6/8/(nvirb*nvira**2*3)), 2)
        for p0, p1 in lib.prange(0, noccb, blksize):
            OVvv = np.asarray(eris.OVvv[p0:p1]).reshape((p1-p0)*nvirb,-1)
            OVvv = lib.unpack_tril(OVvv).reshape(-1,nvirb,nvira,nvira)
            Hr1a += lib.einsum('MFae,iMeF->ia', OVvv, r2ab[:,p0:p1])
            tmpba = lib.einsum('MEaf,iJfE->MaiJ', OVvv, tau2ab)
            Hr2ab-= lib.einsum('MB,MaiJ->iJaB', t1b[p0:p1], tmpba)
            tmpa-= lib.einsum('MEaf,ME->af', OVvv, r1b[p0:p1])
            OVvv = tmpba = None
        tau2ab = None

        Hr2aa-= lib.einsum('af,ijfb->ijab', tmpa, t2aa)
        Hr2bb-= lib.einsum('af,ijfb->ijab', tmpb, t2bb)
        Hr2ab-= lib.einsum('af,iJfB->iJaB', tmpa, t2ab)
        Hr2ab-= lib.einsum('AF,iJbF->iJbA', tmpb, t2ab)

        eris_ovov = np.asarray(eris.ovov)
        eris_OVOV = np.asarray(eris.OVOV)
        eris_ovOV = np.asarray(eris.ovOV)
        tau2aa = make_tau_aa(r2aa, r1a, t1a, 2)
        tauaa = make_tau_aa(t2aa, t1a, t1a)
        tmpaa = lib.einsum('menf,ijef->mnij', eris_ovov, tau2aa)
        Hr2aa += lib.einsum('mnij,mnab->ijab', tmpaa, tauaa) * 0.25
        tau2aa = tauaa = None

        tau2bb = make_tau_aa(r2bb, r1b, t1b, 2)
        taubb = make_tau_aa(t2bb, t1b, t1b)
        tmpbb = lib.einsum('menf,ijef->mnij', eris_OVOV, tau2bb)
        Hr2bb += lib.einsum('mnij,mnab->ijab', tmpbb, taubb) * 0.25
        tau2bb = taubb = None

        tau2ab = make_tau_ab(r2ab, r1 , t1 , 2)
        tauab = make_tau_ab(t2ab, t1 , t1)
        tmpab = lib.einsum('meNF,iJeF->mNiJ', eris_ovOV, tau2ab)
        Hr2ab += lib.einsum('mNiJ,mNaB->iJaB', tmpab, tauab)
        tau2ab = tauab = None

        tmpa = lib.einsum('menf,imef->ni', eris_ovov, r2aa)
        tmpa-= lib.einsum('neMF,iMeF->ni', eris_ovOV, r2ab)
        tmpb = lib.einsum('menf,imef->ni', eris_OVOV, r2bb)
        tmpb-= lib.einsum('mfNE,mIfE->NI', eris_ovOV, r2ab)
        Hr1a += lib.einsum('na,ni->ia', t1a, tmpa)
        Hr1b += lib.einsum('na,ni->ia', t1b, tmpb)
        Hr2aa+= lib.einsum('mj,imab->ijab', tmpa, t2aa)
        Hr2bb+= lib.einsum('mj,imab->ijab', tmpb, t2bb)
        Hr2ab+= lib.einsum('MJ,iMaB->iJaB', tmpb, t2ab)
        Hr2ab+= lib.einsum('mj,mIaB->jIaB', tmpa, t2ab)

        tmp1a = np.einsum('menf,mf->en', eris_ovov, r1a)
        tmp1a-= np.einsum('mfne,mf->en', eris_ovov, r1a)
        tmp1a-= np.einsum('neMF,MF->en', eris_ovOV, r1b)
        tmp1b = np.einsum('menf,mf->en', eris_OVOV, r1b)
        tmp1b-= np.einsum('mfne,mf->en', eris_OVOV, r1b)
        tmp1b-= np.einsum('mfNE,mf->EN', eris_ovOV, r1a)
        tmpa = np.einsum('en,nb->eb', tmp1a, t1a)
        tmpa+= lib.einsum('menf,mnfb->eb', eris_ovov, r2aa)
        tmpa-= lib.einsum('meNF,mNbF->eb', eris_ovOV, r2ab)
        tmpb = np.einsum('en,nb->eb', tmp1b, t1b)
        tmpb+= lib.einsum('menf,mnfb->eb', eris_OVOV, r2bb)
        tmpb-= lib.einsum('nfME,nMfB->EB', eris_ovOV, r2ab)
        Hr2aa+= lib.einsum('eb,ijae->ijab', tmpa, t2aa)
        Hr2bb+= lib.einsum('eb,ijae->ijab', tmpb, t2bb)
        Hr2ab+= lib.einsum('EB,iJaE->iJaB', tmpb, t2ab)
        Hr2ab+= lib.einsum('eb,iJeA->iJbA', tmpa, t2ab)
        eirs_ovov = eris_ovOV = eris_OVOV = None

        Hr2aa-= lib.einsum('mbij,ma->ijab', imds.wovoo, r1a)
        Hr2bb-= lib.einsum('mbij,ma->ijab', imds.wOVOO, r1b)
        Hr2ab-= lib.einsum('mBiJ,ma->iJaB', imds.woVoO, r1a)
        Hr2ab-= lib.einsum('MbJi,MA->iJbA', imds.wOvOo, r1b)

        Hr1a-= 0.5*lib.einsum('mnie,mnae->ia', imds.wooov, r2aa)
        Hr1a-=     lib.einsum('mNiE,mNaE->ia', imds.woOoV, r2ab)
        Hr1b-= 0.5*lib.einsum('mnie,mnae->ia', imds.wOOOV, r2bb)
        Hr1b-=     lib.einsum('MnIe,nMeA->IA', imds.wOoOv, r2ab)
        tmpa = lib.einsum('mnie,me->ni', imds.wooov, r1a)
        tmpa-= lib.einsum('nMiE,ME->ni', imds.woOoV, r1b)
        tmpb = lib.einsum('mnie,me->ni', imds.wOOOV, r1b)
        tmpb-= lib.einsum('NmIe,me->NI', imds.wOoOv, r1a)
        Hr2aa+= lib.einsum('ni,njab->ijab', tmpa, t2aa)
        Hr2bb+= lib.einsum('ni,njab->ijab', tmpb, t2bb)
        Hr2ab+= lib.einsum('ni,nJaB->iJaB', tmpa, t2ab)
        Hr2ab+= lib.einsum('NI,jNaB->jIaB', tmpb, t2ab)
        for p0, p1 in lib.prange(0, nvira, nocca):
            Hr2aa+= lib.einsum('ejab,ie->ijab', imds.wvovv[p0:p1], r1a[:,p0:p1])
            Hr2ab+= lib.einsum('eJaB,ie->iJaB', imds.wvOvV[p0:p1], r1a[:,p0:p1])
        for p0, p1 in lib.prange(0, nvirb, noccb):
            Hr2bb+= lib.einsum('ejab,ie->ijab', imds.wVOVV[p0:p1], r1b[:,p0:p1])
            Hr2ab+= lib.einsum('EjBa,IE->jIaB', imds.wVoVv[p0:p1], r1b[:,p0:p1])

        Hr1a += np.einsum('maei,me->ia',imds.wovvo,r1a)
        Hr1a += np.einsum('MaEi,ME->ia',imds.wOvVo,r1b)
        Hr1b += np.einsum('maei,me->ia',imds.wOVVO,r1b)
        Hr1b += np.einsum('mAeI,me->IA',imds.woVvO,r1a)
        Hr2aa+= lib.einsum('mbej,imae->ijab', imds.wovvo, r2aa) * 2
        Hr2aa+= lib.einsum('MbEj,iMaE->ijab', imds.wOvVo, r2ab) * 2
        Hr2bb+= lib.einsum('mbej,imae->ijab', imds.wOVVO, r2bb) * 2
        Hr2bb+= lib.einsum('mBeJ,mIeA->IJAB', imds.woVvO, r2ab) * 2
        Hr2ab+= lib.einsum('mBeJ,imae->iJaB', imds.woVvO, r2aa)
        Hr2ab+= lib.einsum('MBEJ,iMaE->iJaB', imds.wOVVO, r2ab)
        Hr2ab+= lib.einsum('mBEj,mIaE->jIaB', imds.woVVo, r2ab)
        Hr2ab+= lib.einsum('mbej,mIeA->jIbA', imds.wovvo, r2ab)
        Hr2ab+= lib.einsum('MbEj,IMAE->jIbA', imds.wOvVo, r2bb)
        Hr2ab+= lib.einsum('MbeJ,iMeA->iJbA', imds.wOvvO, r2ab)

        #:eris_vvvv = ao2mo.restore(1, np.asarray(eris.vvvv), nvirb)
        #:eris_VVVV = ao2mo.restore(1, np.asarray(eris.VVVV), nvirb)
        #:eris_vvVV = _restore(np.asarray(eris.vvVV), nvira, nvirb)
        #:Hr2aa += lib.einsum('ijef,aebf->ijab', tau2aa, eris_vvvv) * .5
        #:Hr2bb += lib.einsum('ijef,aebf->ijab', tau2bb, eris_VVVV) * .5
        #:Hr2ab += lib.einsum('iJeF,aeBF->iJaB', tau2ab, eris_vvVV)
        tau2aa, tau2ab, tau2bb = make_tau(r2, r1, t1, 2)
        _add_vvvv_(self, (tau2aa,tau2ab,tau2bb), eris, (Hr2aa,Hr2ab,Hr2bb))
        Hr2aa *= .5
        Hr2bb *= .5
        Hr2aa = Hr2aa - Hr2aa.transpose(0,1,3,2)
        Hr2aa = Hr2aa - Hr2aa.transpose(1,0,2,3)
        Hr2bb = Hr2bb - Hr2bb.transpose(0,1,3,2)
        Hr2bb = Hr2bb - Hr2bb.transpose(1,0,2,3)

        vector = self.amplitudes_to_vector((Hr1a,Hr1b), (Hr2aa,Hr2ab,Hr2bb))
        return vector

    def eomsf_ccsd_matvec(self, vector):
        '''Spin flip EOM-CCSD'''
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ee_imds:
            self.imds.make_ee()
        imds = self.imds

        t1, t2, eris = self.t1, self.t2, self.eris
        t1a, t1b = t1
        t2aa, t2ab, t2bb = t2
        nocca, noccb, nvira, nvirb = t2ab.shape
        r1, r2 = self.vector_to_amplitudes_eomsf(vector, (nocca,noccb), (nvira,nvirb))
        r1ab, r1ba = r1
        r2baaa, r2aaba, r2abbb, r2bbab = r2

        Hr1ab  = np.einsum('ae,ie->ia', imds.Fvvb, r1ab)
        Hr1ab -= np.einsum('mi,ma->ia', imds.Fooa, r1ab)
        Hr1ab += np.einsum('me,imae->ia', imds.Fovb, r2abbb)
        Hr1ab += np.einsum('me,imae->ia', imds.Fova, r2aaba)
        Hr1ba  = np.einsum('ae,ie->ia', imds.Fvva, r1ba)
        Hr1ba -= np.einsum('mi,ma->ia', imds.Foob, r1ba)
        Hr1ba += np.einsum('me,imae->ia', imds.Fova, r2baaa)
        Hr1ba += np.einsum('me,imae->ia', imds.Fovb, r2bbab)
        Hr2baaa = .5 *lib.einsum('nMjI,Mnab->Ijab', imds.woOoO, r2baaa)
        Hr2aaba = .25*lib.einsum('mnij,mnAb->ijAb', imds.woooo, r2aaba)
        Hr2abbb = .5 *lib.einsum('mNiJ,mNAB->iJAB', imds.woOoO, r2abbb)
        Hr2bbab = .25*lib.einsum('MNIJ,MNaB->IJaB', imds.wOOOO, r2bbab)
        Hr2baaa += lib.einsum('be,Ijae->Ijab', imds.Fvva   , r2baaa)
        Hr2baaa -= lib.einsum('mj,imab->ijab', imds.Fooa*.5, r2baaa)
        Hr2baaa -= lib.einsum('MJ,Miab->Jiab', imds.Foob*.5, r2baaa)
        Hr2bbab -= lib.einsum('mj,imab->ijab', imds.Foob   , r2bbab)
        Hr2bbab += lib.einsum('BE,IJaE->IJaB', imds.Fvvb*.5, r2bbab)
        Hr2bbab += lib.einsum('be,IJeA->IJbA', imds.Fvva*.5, r2bbab)
        Hr2aaba -= lib.einsum('mj,imab->ijab', imds.Fooa   , r2aaba)
        Hr2aaba += lib.einsum('be,ijAe->ijAb', imds.Fvva*.5, r2aaba)
        Hr2aaba += lib.einsum('BE,ijEa->ijBa', imds.Fvvb*.5, r2aaba)
        Hr2abbb += lib.einsum('BE,iJAE->iJAB', imds.Fvvb   , r2abbb)
        Hr2abbb -= lib.einsum('mj,imab->ijab', imds.Foob*.5, r2abbb)
        Hr2abbb -= lib.einsum('mj,mIAB->jIAB', imds.Fooa*.5, r2abbb)

        tau2baaa = np.einsum('ia,jb->ijab', r1ba, t1a)
        tau2baaa = tau2baaa - tau2baaa.transpose(0,1,3,2)
        tau2abbb = np.einsum('ia,jb->ijab', r1ab, t1b)
        tau2abbb = tau2abbb - tau2abbb.transpose(0,1,3,2)
        tau2aaba = np.einsum('ia,jb->ijab', r1ab, t1a)
        tau2aaba = tau2aaba - tau2aaba.transpose(1,0,2,3)
        tau2bbab = np.einsum('ia,jb->ijab', r1ba, t1b)
        tau2bbab = tau2bbab - tau2bbab.transpose(1,0,2,3)
        tau2baaa += r2baaa
        tau2bbab += r2bbab
        tau2abbb += r2abbb
        tau2aaba += r2aaba
        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
        #:Hr1ba += einsum('mfae,Imef->Ia', eris_ovvv, r2baaa)
        #:tmp1aaba = lib.einsum('meaf,Ijef->maIj', eris_ovvv, tau2baaa)
        #:Hr2baaa += lib.einsum('mb,maIj->Ijab', t1a   , tmp1aaba)
        mem_now = lib.current_memory()[0]
        max_memory = lib.param.MAX_MEMORY - mem_now
        blksize = max(int(max_memory*1e6/8/(nvira**3*3)), 2)
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = np.asarray(eris.ovvv[p0:p1]).reshape((p1-p0)*nvira,-1)
            ovvv = lib.unpack_tril(ovvv).reshape(-1,nvira,nvira,nvira)
            Hr1ba += einsum('mfae,Imef->Ia', ovvv, r2baaa[:,p0:p1])
            tmp1aaba = lib.einsum('meaf,Ijef->maIj', ovvv, tau2baaa)
            Hr2baaa += lib.einsum('mb,maIj->Ijab', t1a[p0:p1], tmp1aaba)
            ovvv = tmp1aaba = None

        #:eris_OVVV = lib.unpack_tril(np.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
        #:Hr1ab += einsum('MFAE,iMEF->iA', eris_OVVV, r2abbb)
        #:tmp1bbab = lib.einsum('MEAF,iJEF->MAiJ', eris_OVVV, tau2abbb)
        #:Hr2abbb += lib.einsum('MB,MAiJ->iJAB', t1b   , tmp1bbab)
        blksize = max(int(max_memory*1e6/8/(nvirb**3*3)), 2)
        for p0, p1 in lib.prange(0, noccb, blksize):
            OVVV = np.asarray(eris.OVVV[p0:p1]).reshape((p1-p0)*nvirb,-1)
            OVVV = lib.unpack_tril(OVVV).reshape(-1,nvirb,nvirb,nvirb)
            Hr1ab += einsum('MFAE,iMEF->iA', OVVV, r2abbb[:,p0:p1])
            tmp1bbab = lib.einsum('MEAF,iJEF->MAiJ', OVVV, tau2abbb)
            Hr2abbb += lib.einsum('MB,MAiJ->iJAB', t1b[p0:p1], tmp1bbab)
            OVVV = tmp1bbab = None

        #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
        #:Hr1ab += einsum('mfAE,imEf->iA', eris_ovVV, r2aaba)
        #:tmp1abaa = lib.einsum('meAF,ijFe->mAij', eris_ovVV, tau2aaba)
        #:tmp1abbb = lib.einsum('meAF,IJeF->mAIJ', eris_ovVV, tau2bbab)
        #:tmp1ba = lib.einsum('mfAE,mE->Af', eris_ovVV, r1ab)
        #:Hr2bbab -= lib.einsum('mb,mAIJ->IJbA', t1a*.5, tmp1abbb)
        #:Hr2aaba -= lib.einsum('mb,mAij->ijAb', t1a*.5, tmp1abaa)
        tmp1ba = np.zeros((nvirb,nvira))
        blksize = max(int(max_memory*1e6/8/(nvira*nvirb**2*3)), 2)
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = np.asarray(eris.ovVV[p0:p1]).reshape((p1-p0)*nvira,-1)
            ovVV = lib.unpack_tril(ovVV).reshape(-1,nvira,nvirb,nvirb)
            Hr1ab += einsum('mfAE,imEf->iA', ovVV, r2aaba[:,p0:p1])
            tmp1abaa = lib.einsum('meAF,ijFe->mAij', ovVV, tau2aaba)
            tmp1abbb = lib.einsum('meAF,IJeF->mAIJ', ovVV, tau2bbab)
            tmp1ba += lib.einsum('mfAE,mE->Af', ovVV, r1ab[p0:p1])
            Hr2bbab -= lib.einsum('mb,mAIJ->IJbA', t1a[p0:p1]*.5, tmp1abbb)
            Hr2aaba -= lib.einsum('mb,mAij->ijAb', t1a[p0:p1]*.5, tmp1abaa)

        #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
        #:Hr1ba += einsum('MFae,IMeF->Ia', eris_OVvv, r2bbab)
        #:tmp1baaa = lib.einsum('MEaf,ijEf->Maij', eris_OVvv, tau2aaba)
        #:tmp1babb = lib.einsum('MEaf,IJfE->MaIJ', eris_OVvv, tau2bbab)
        #:tmp1ab = lib.einsum('MFae,Me->aF', eris_OVvv, r1ba)
        #:Hr2aaba -= lib.einsum('MB,Maij->ijBa', t1b*.5, tmp1baaa)
        #:Hr2bbab -= lib.einsum('MB,MaIJ->IJaB', t1b*.5, tmp1babb)
        tmp1ab = np.zeros((nvira,nvirb))
        blksize = max(int(max_memory*1e6/8/(nvirb*nvira**2*3)), 2)
        for p0, p1 in lib.prange(0, noccb, blksize):
            OVvv = np.asarray(eris.OVvv[p0:p1]).reshape((p1-p0)*nvirb,-1)
            OVvv = lib.unpack_tril(OVvv).reshape(-1,nvirb,nvira,nvira)
            Hr1ba += einsum('MFae,IMeF->Ia', OVvv, r2bbab[:,p0:p1])
            tmp1baaa = lib.einsum('MEaf,ijEf->Maij', OVvv, tau2aaba)
            tmp1babb = lib.einsum('MEaf,IJfE->MaIJ', OVvv, tau2bbab)
            tmp1ab+= lib.einsum('MFae,Me->aF', OVvv, r1ba[p0:p1])
            Hr2aaba -= lib.einsum('MB,Maij->ijBa', t1b[p0:p1]*.5, tmp1baaa)
            Hr2bbab -= lib.einsum('MB,MaIJ->IJaB', t1b[p0:p1]*.5, tmp1babb)

        Hr2baaa += lib.einsum('aF,jIbF->Ijba', tmp1ab   , t2ab)
        Hr2bbab -= lib.einsum('aF,IJFB->IJaB', tmp1ab*.5, t2bb)
        Hr2abbb += lib.einsum('Af,iJfB->iJBA', tmp1ba   , t2ab)
        Hr2aaba -= lib.einsum('Af,ijfb->ijAb', tmp1ba*.5, t2aa)
        Hr2baaa -= lib.einsum('MbIj,Ma->Ijab', imds.wOvOo, r1ba   )
        Hr2bbab -= lib.einsum('MBIJ,Ma->IJaB', imds.wOVOO, r1ba*.5)
        Hr2abbb -= lib.einsum('mBiJ,mA->iJAB', imds.woVoO, r1ab   )
        Hr2aaba -= lib.einsum('mbij,mA->ijAb', imds.wovoo, r1ab*.5)

        Hr1ab -= 0.5*lib.einsum('mnie,mnAe->iA', imds.wooov, r2aaba)
        Hr1ab -=     lib.einsum('mNiE,mNAE->iA', imds.woOoV, r2abbb)
        Hr1ba -= 0.5*lib.einsum('MNIE,MNaE->Ia', imds.wOOOV, r2bbab)
        Hr1ba -=     lib.einsum('MnIe,Mnae->Ia', imds.wOoOv, r2baaa)
        tmp1ab = lib.einsum('MnIe,Me->nI', imds.wOoOv, r1ba)
        tmp1ba = lib.einsum('mNiE,mE->Ni', imds.woOoV, r1ab)
        Hr2baaa += lib.einsum('nI,njab->Ijab', tmp1ab*.5, t2aa)
        Hr2bbab += lib.einsum('nI,nJaB->IJaB', tmp1ab   , t2ab)
        Hr2abbb += lib.einsum('Ni,NJAB->iJAB', tmp1ba*.5, t2bb)
        Hr2aaba += lib.einsum('Ni,jNbA->ijAb', tmp1ba   , t2ab)
        for p0, p1 in lib.prange(0, nvira, nocca):
            Hr2baaa += lib.einsum('ejab,Ie->Ijab', imds.wvovv[p0:p1], r1ba[:,p0:p1]*.5)
            Hr2bbab += lib.einsum('eJaB,Ie->IJaB', imds.wvOvV[p0:p1], r1ba[:,p0:p1]   )
        for p0, p1 in lib.prange(0, nvirb, noccb):
            Hr2abbb += lib.einsum('EJAB,iE->iJAB', imds.wVOVV[p0:p1], r1ab[:,p0:p1]*.5)
            Hr2aaba += lib.einsum('EjAb,iE->ijAb', imds.wVoVv[p0:p1], r1ab[:,p0:p1]   )

        Hr1ab += np.einsum('mAEi,mE->iA', imds.woVVo, r1ab)
        Hr1ba += np.einsum('MaeI,Me->Ia', imds.wOvvO, r1ba)
        Hr2baaa += lib.einsum('mbej,Imae->Ijab', imds.wovvo, r2baaa)
        Hr2baaa += lib.einsum('MbeJ,Miae->Jiab', imds.wOvvO, r2baaa)
        Hr2baaa += lib.einsum('MbEj,IMaE->Ijab', imds.wOvVo, r2bbab)
        Hr2bbab += lib.einsum('MBEJ,IMaE->IJaB', imds.wOVVO, r2bbab)
        Hr2bbab += lib.einsum('MbeJ,IMeA->IJbA', imds.wOvvO, r2bbab)
        Hr2bbab += lib.einsum('mBeJ,Imae->IJaB', imds.woVvO, r2baaa)
        Hr2aaba += lib.einsum('mbej,imAe->ijAb', imds.wovvo, r2aaba)
        Hr2aaba += lib.einsum('mBEj,imEa->ijBa', imds.woVVo, r2aaba)
        Hr2aaba += lib.einsum('MbEj,iMAE->ijAb', imds.wOvVo, r2abbb)
        Hr2abbb += lib.einsum('MBEJ,iMAE->iJAB', imds.wOVVO, r2abbb)
        Hr2abbb += lib.einsum('mBEj,mIAE->jIAB', imds.woVVo, r2abbb)
        Hr2abbb += lib.einsum('mBeJ,imAe->iJAB', imds.woVvO, r2aaba)

        eris_ovov = np.asarray(eris.ovov)
        eris_OVOV = np.asarray(eris.OVOV)
        eris_ovOV = np.asarray(eris.ovOV)
        tauaa, tauab, taubb = make_tau(t2, t1, t1)
        tmp1baaa = lib.einsum('nfME,ijEf->Mnij', eris_ovOV, tau2aaba)
        tmp1aaba = lib.einsum('menf,Ijef->mnIj', eris_ovov, tau2baaa)
        tmp1abbb = lib.einsum('meNF,IJeF->mNIJ', eris_ovOV, tau2bbab)
        tmp1bbab = lib.einsum('MENF,iJEF->MNiJ', eris_OVOV, tau2abbb)
        Hr2baaa += 0.5*.5*lib.einsum('mnIj,mnab->Ijab', tmp1aaba, tauaa)
        Hr2bbab +=     .5*lib.einsum('nMIJ,nMaB->IJaB', tmp1abbb, tauab)
        Hr2aaba +=     .5*lib.einsum('Nmij,mNbA->ijAb', tmp1baaa, tauab)
        Hr2abbb += 0.5*.5*lib.einsum('MNiJ,MNAB->iJAB', tmp1bbab, taubb)
        tauaa = tauab = taubb = None

        tmpab  = lib.einsum('menf,Imef->nI', eris_ovov, r2baaa)
        tmpab -= lib.einsum('nfME,IMfE->nI', eris_ovOV, r2bbab)
        tmpba  = lib.einsum('MENF,iMEF->Ni', eris_OVOV, r2abbb)
        tmpba -= lib.einsum('meNF,imFe->Ni', eris_ovOV, r2aaba)
        Hr1ab += np.einsum('NA,Ni->iA', t1b, tmpba)
        Hr1ba += np.einsum('na,nI->Ia', t1a, tmpab)
        Hr2baaa -= lib.einsum('mJ,imab->Jiab', tmpab*.5, t2aa)
        Hr2bbab -= lib.einsum('mJ,mIaB->IJaB', tmpab*.5, t2ab) * 2
        Hr2aaba -= lib.einsum('Mj,iMbA->ijAb', tmpba*.5, t2ab) * 2
        Hr2abbb -= lib.einsum('Mj,IMAB->jIAB', tmpba*.5, t2bb)

        tmp1ab = np.einsum('meNF,mF->eN', eris_ovOV, r1ab)
        tmp1ba = np.einsum('nfME,Mf->En', eris_ovOV, r1ba)
        tmpab = np.einsum('eN,NB->eB', tmp1ab, t1b)
        tmpba = np.einsum('En,nb->Eb', tmp1ba, t1a)
        tmpab -= lib.einsum('menf,mnBf->eB', eris_ovov, r2aaba)
        tmpab += lib.einsum('meNF,mNFB->eB', eris_ovOV, r2abbb)
        tmpba -= lib.einsum('MENF,MNbF->Eb', eris_OVOV, r2bbab)
        tmpba += lib.einsum('nfME,Mnfb->Eb', eris_ovOV, r2baaa)
        Hr2baaa -= lib.einsum('Eb,jIaE->Ijab', tmpba*.5, t2ab) * 2
        Hr2bbab -= lib.einsum('Eb,IJAE->IJbA', tmpba*.5, t2bb)
        Hr2aaba -= lib.einsum('eB,ijae->ijBa', tmpab*.5, t2aa)
        Hr2abbb -= lib.einsum('eB,iJeA->iJAB', tmpab*.5, t2ab) * 2
        eris_ovov = eris_OVOV = eris_ovOV = None

        #:eris_vvvv = ao2mo.restore(1, np.asarray(eris.vvvv), nvirb)
        #:eris_VVVV = ao2mo.restore(1, np.asarray(eris.VVVV), nvirb)
        #:eris_vvVV = _restore(np.asarray(eris.vvVV), nvira, nvirb)
        #:Hr2baaa += .5*lib.einsum('Ijef,aebf->Ijab', tau2baaa, eris_vvvv)
        #:Hr2abbb += .5*lib.einsum('iJEF,AEBF->iJAB', tau2abbb, eris_VVVV)
        #:Hr2bbab += .5*lib.einsum('IJeF,aeBF->IJaB', tau2bbab, eris_vvVV)
        #:Hr2aaba += .5*lib.einsum('ijEf,bfAE->ijAb', tau2aaba, eris_vvVV)
        tau2baaa *= .5
        rccsd._add_vvvv1_(self, tau2baaa, eris, Hr2baaa)
        fakeri = lambda:None
        fakeri.vvvv = eris.VVVV
        tau2abbb *= .5
        rccsd._add_vvvv1_(self, tau2abbb, fakeri, Hr2abbb)
        fakeri.vvvv = eris.vvVV
        tau2bbab *= .5
        rccsd._add_vvvv1_(self, tau2bbab, fakeri, Hr2bbab)
        fakeri = None
        for i in range(nvira):
            i0 = i*(i+1)//2
            vvv = lib.unpack_tril(np.asarray(eris.vvVV[i0:i0+i+1]))
            Hr2aaba[:,:,:,i ] += .5*lib.einsum('ijef,fae->ija', tau2aaba[:,:,:,:i+1], vvv)
            Hr2aaba[:,:,:,:i] += .5*lib.einsum('ije,bae->ijab', tau2aaba[:,:,:,i], vvv[:i])
            vvv = None

        Hr2baaa = Hr2baaa - Hr2baaa.transpose(0,1,3,2)
        Hr2bbab = Hr2bbab - Hr2bbab.transpose(1,0,2,3)
        Hr2abbb = Hr2abbb - Hr2abbb.transpose(0,1,3,2)
        Hr2aaba = Hr2aaba - Hr2aaba.transpose(1,0,2,3)
        vector = self.amplitudes_to_vector_eomsf((Hr1ab, Hr1ba), (Hr2baaa,Hr2aaba,Hr2abbb,Hr2bbab))
        return vector

    def eeccsd_diag(self):
        if not hasattr(self,'imds'):
            self.imds = _IMDS(self)
        if not self.imds.made_ee_imds:
            self.imds.make_ee()
        imds = self.imds

        eris = self.eris
        t1, t2 = self.t1, self.t2
        t1a, t1b = t1
        t2aa, t2ab, t2bb = t2
        tauaa, tauab, taubb = make_tau(t2, t1, t1)
        nocca, noccb, nvira, nvirb = t2ab.shape

        Foa = imds.Fooa.diagonal()
        Fob = imds.Foob.diagonal()
        Fva = imds.Fvva.diagonal()
        Fvb = imds.Fvvb.diagonal()
        Wovaa = np.einsum('iaai->ia', imds.wovvo)
        Wovbb = np.einsum('iaai->ia', imds.wOVVO)
        Wovab = np.einsum('iaai->ia', imds.woVVo)
        Wovba = np.einsum('iaai->ia', imds.wOvvO)

        Hr1aa = lib.direct_sum('-i+a->ia', Foa, Fva)
        Hr1bb = lib.direct_sum('-i+a->ia', Fob, Fvb)
        Hr1ab = lib.direct_sum('-i+a->ia', Foa, Fvb)
        Hr1ba = lib.direct_sum('-i+a->ia', Fob, Fva)
        Hr1aa += Wovaa
        Hr1bb += Wovbb
        Hr1ab += Wovab
        Hr1ba += Wovba

        eris_ovov = np.asarray(eris.ovov)
        eris_OVOV = np.asarray(eris.OVOV)
        eris_ovOV = np.asarray(eris.ovOV)
        ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
        OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
        Wvvaa = .5*np.einsum('mnab,manb->ab', tauaa, eris_ovov)
        Wvvbb = .5*np.einsum('mnab,manb->ab', taubb, eris_OVOV)
        Wvvab =    np.einsum('mNaB,maNB->aB', tauab, eris_ovOV)
        ijb = np.einsum('iejb,ijbe->ijb',      ovov, t2aa)
        IJB = np.einsum('iejb,ijbe->ijb',      OVOV, t2bb)
        iJB =-np.einsum('ieJB,iJeB->iJB', eris_ovOV, t2ab)
        Ijb =-np.einsum('jbIE,jIbE->Ijb', eris_ovOV, t2ab)
        iJb =-np.einsum('ibJE,iJbE->iJb', eris_ovOV, t2ab)
        IjB =-np.einsum('jeIB,jIeB->IjB', eris_ovOV, t2ab)
        jab = np.einsum('kajb,jkab->jab',      ovov, t2aa)
        JAB = np.einsum('kajb,jkab->jab',      OVOV, t2bb)
        jAb =-np.einsum('jbKA,jKbA->jAb', eris_ovOV, t2ab)
        JaB =-np.einsum('kaJB,kJaB->JaB', eris_ovOV, t2ab)
        jaB =-np.einsum('jaKB,jKaB->jaB', eris_ovOV, t2ab)
        JAb =-np.einsum('kbJA,kJbA->JAb', eris_ovOV, t2ab)
        eris_ovov = eris_ovOV = eris_OVOV = ovov = OVOV = None
        Hr2aa = lib.direct_sum('ijb+a->ijba', ijb, Fva)
        Hr2bb = lib.direct_sum('ijb+a->ijba', IJB, Fvb)
        Hr2ab = lib.direct_sum('iJb+A->iJbA', iJb, Fvb)
        Hr2ab+= lib.direct_sum('iJB+a->iJaB', iJB, Fva)
        Hr2aa+= lib.direct_sum('-i+jab->ijab', Foa, jab)
        Hr2bb+= lib.direct_sum('-i+jab->ijab', Fob, JAB)
        Hr2ab+= lib.direct_sum('-i+JaB->iJaB', Foa, JaB)
        Hr2ab+= lib.direct_sum('-I+jaB->jIaB', Fob, jaB)
        Hr2aa = Hr2aa + Hr2aa.transpose(0,1,3,2)
        Hr2aa = Hr2aa + Hr2aa.transpose(1,0,2,3)
        Hr2bb = Hr2bb + Hr2bb.transpose(0,1,3,2)
        Hr2bb = Hr2bb + Hr2bb.transpose(1,0,2,3)
        Hr2aa *= .5
        Hr2bb *= .5
        Hr2baaa = lib.direct_sum('Ijb+a->Ijba', Ijb, Fva)
        Hr2aaba = lib.direct_sum('ijb+A->ijAb', ijb, Fvb)
        Hr2aaba+= Fva.reshape(1,1,1,-1)
        Hr2abbb = lib.direct_sum('iJB+A->iJBA', iJB, Fvb)
        Hr2bbab = lib.direct_sum('IJB+a->IJaB', IJB, Fva)
        Hr2bbab+= Fvb.reshape(1,1,1,-1)
        Hr2baaa = Hr2baaa + Hr2baaa.transpose(0,1,3,2)
        Hr2abbb = Hr2abbb + Hr2abbb.transpose(0,1,3,2)
        Hr2baaa+= lib.direct_sum('-I+jab->Ijab', Fob, jab)
        Hr2baaa-= Foa.reshape(1,-1,1,1)
        tmpaaba = lib.direct_sum('-i+jAb->ijAb', Foa, jAb)
        Hr2abbb+= lib.direct_sum('-i+JAB->iJAB', Foa, JAB)
        Hr2abbb-= Fob.reshape(1,-1,1,1)
        tmpbbab = lib.direct_sum('-I+JaB->IJaB', Fob, JaB)
        Hr2aaba+= tmpaaba + tmpaaba.transpose(1,0,2,3)
        Hr2bbab+= tmpbbab + tmpbbab.transpose(1,0,2,3)
        tmpaaba = tmpbbab = None
        Hr2aa += Wovaa.reshape(1,nocca,1,nvira)
        Hr2aa += Wovaa.reshape(nocca,1,1,nvira)
        Hr2aa += Wovaa.reshape(nocca,1,nvira,1)
        Hr2aa += Wovaa.reshape(1,nocca,nvira,1)
        Hr2ab += Wovbb.reshape(1,noccb,1,nvirb)
        Hr2ab += Wovab.reshape(nocca,1,1,nvirb)
        Hr2ab += Wovaa.reshape(nocca,1,nvira,1)
        Hr2ab += Wovba.reshape(1,noccb,nvira,1)
        Hr2bb += Wovbb.reshape(1,noccb,1,nvirb)
        Hr2bb += Wovbb.reshape(noccb,1,1,nvirb)
        Hr2bb += Wovbb.reshape(noccb,1,nvirb,1)
        Hr2bb += Wovbb.reshape(1,noccb,nvirb,1)
        Hr2baaa += Wovaa.reshape(1,nocca,1,nvira)
        Hr2baaa += Wovba.reshape(noccb,1,1,nvira)
        Hr2baaa += Wovba.reshape(noccb,1,nvira,1)
        Hr2baaa += Wovaa.reshape(1,nocca,nvira,1)
        Hr2aaba += Wovaa.reshape(1,nocca,1,nvira)
        Hr2aaba += Wovaa.reshape(nocca,1,1,nvira)
        Hr2aaba += Wovab.reshape(nocca,1,nvirb,1)
        Hr2aaba += Wovab.reshape(1,nocca,nvirb,1)
        Hr2abbb += Wovbb.reshape(1,noccb,1,nvirb)
        Hr2abbb += Wovab.reshape(nocca,1,1,nvirb)
        Hr2abbb += Wovab.reshape(nocca,1,nvirb,1)
        Hr2abbb += Wovbb.reshape(1,noccb,nvirb,1)
        Hr2bbab += Wovbb.reshape(1,noccb,1,nvirb)
        Hr2bbab += Wovbb.reshape(noccb,1,1,nvirb)
        Hr2bbab += Wovba.reshape(noccb,1,nvira,1)
        Hr2bbab += Wovba.reshape(1,noccb,nvira,1)

        Wooaa  = np.einsum('ijij->ij', imds.woooo).copy()
        Wooaa -= np.einsum('ijji->ij', imds.woooo)
        Woobb  = np.einsum('ijij->ij', imds.wOOOO).copy()
        Woobb -= np.einsum('ijji->ij', imds.wOOOO)
        Wooab = np.einsum('ijij->ij', imds.woOoO)
        Wooba = Wooab.T
        Wooaa *= .5
        Woobb *= .5
        Hr2aa += Wooaa.reshape(nocca,nocca,1,1)
        Hr2ab += Wooab.reshape(nocca,noccb,1,1)
        Hr2bb += Woobb.reshape(noccb,noccb,1,1)
        Hr2baaa += Wooba.reshape(noccb,nocca,1,1)
        Hr2aaba += Wooaa.reshape(nocca,nocca,1,1)
        Hr2abbb += Wooab.reshape(nocca,noccb,1,1)
        Hr2bbab += Woobb.reshape(noccb,noccb,1,1)

        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
        #:Wvvaa += np.einsum('mb,maab->ab', t1a, eris_ovvv)
        #:Wvvaa -= np.einsum('mb,mbaa->ab', t1a, eris_ovvv)
        mem_now = lib.current_memory()[0]
        max_memory = lib.param.MAX_MEMORY - mem_now
        blksize = max(int(max_memory*1e6/8/(nvira**3*3)), 2)
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = np.asarray(eris.ovvv[p0:p1]).reshape((p1-p0)*nvira,-1)
            ovvv = lib.unpack_tril(ovvv).reshape(-1,nvira,nvira,nvira)
            Wvvaa += np.einsum('mb,maab->ab', t1a[p0:p1], ovvv)
            Wvvaa -= np.einsum('mb,mbaa->ab', t1a[p0:p1], ovvv)
            ovvv = None
        #:eris_OVVV = lib.unpack_tril(np.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
        #:Wvvbb += np.einsum('mb,maab->ab', t1b, eris_OVVV)
        #:Wvvbb -= np.einsum('mb,mbaa->ab', t1b, eris_OVVV)
        blksize = max(int(max_memory*1e6/8/(nvirb**3*3)), 2)
        for p0, p1 in lib.prange(0, noccb, blksize):
            OVVV = np.asarray(eris.OVVV[p0:p1]).reshape((p1-p0)*nvirb,-1)
            OVVV = lib.unpack_tril(OVVV).reshape(-1,nvirb,nvirb,nvirb)
            Wvvbb += np.einsum('mb,maab->ab', t1b[p0:p1], OVVV)
            Wvvbb -= np.einsum('mb,mbaa->ab', t1b[p0:p1], OVVV)
            OVVV = None
        #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
        #:Wvvab -= np.einsum('mb,mbaa->ba', t1a, eris_ovVV)
        blksize = max(int(max_memory*1e6/8/(nvira*nvirb**2*3)), 2)
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = np.asarray(eris.ovVV[p0:p1]).reshape((p1-p0)*nvira,-1)
            ovVV = lib.unpack_tril(ovVV).reshape(-1,nvira,nvirb,nvirb)
            Wvvab -= np.einsum('mb,mbaa->ba', t1a[p0:p1], ovVV)
            ovVV = None
        blksize = max(int(max_memory*1e6/8/(nvirb*nvira**2*3)), 2)
        #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
        #:Wvvab -= np.einsum('mb,mbaa->ab', t1b, eris_OVvv)
        idxa = np.arange(nvira)
        idxa = idxa*(idxa+1)//2+idxa
        for p0, p1 in lib.prange(0, noccb, blksize):
            OVvv = np.asarray(eris.OVvv[p0:p1])
            Wvvab -= np.einsum('mb,mba->ab', t1b[p0:p1], OVvv[:,:,idxa])
            OVvv = None
        Wvvaa = Wvvaa + Wvvaa.T
        Wvvbb = Wvvbb + Wvvbb.T
        #:eris_vvvv = ao2mo.restore(1, np.asarray(eris.vvvv), nvirb)
        #:eris_VVVV = ao2mo.restore(1, np.asarray(eris.VVVV), nvirb)
        #:eris_vvVV = _restore(np.asarray(eris.vvVV), nvira, nvirb)
        #:Wvvaa += np.einsum('aabb->ab', eris_vvvv) - np.einsum('abba->ab', eris_vvvv)
        #:Wvvbb += np.einsum('aabb->ab', eris_VVVV) - np.einsum('abba->ab', eris_VVVV)
        #:Wvvab += np.einsum('aabb->ab', eris_vvVV)
        for i in range(nvira):
            i0 = i*(i+1)//2
            vvv = lib.unpack_tril(np.asarray(eris.vvvv[i0:i0+i+1]))
            tmp = np.einsum('bb->b', vvv[i])
            Wvvaa[i] += tmp
            tmp = np.einsum('bb->b', vvv[:,:i+1,i])
            Wvvaa[i,:i+1] -= tmp
            Wvvaa[:i  ,i] -= tmp[:i]
            vvv = lib.unpack_tril(np.asarray(eris.vvVV[i0:i0+i+1]))
            Wvvab[i] += np.einsum('bb->b', vvv[i])
            vvv = None
        for i in range(nvirb):
            i0 = i*(i+1)//2
            vvv = lib.unpack_tril(np.asarray(eris.VVVV[i0:i0+i+1]))
            tmp = np.einsum('bb->b', vvv[i])
            Wvvbb[i] += tmp
            tmp = np.einsum('bb->b', vvv[:,:i+1,i])
            Wvvbb[i,:i+1] -= tmp
            Wvvbb[:i  ,i] -= tmp[:i]
            vvv = None
        Wvvba = Wvvab.T
        Hr2aa += Wvvaa.reshape(1,1,nvira,nvira)
        Hr2ab += Wvvab.reshape(1,1,nvira,nvirb)
        Hr2bb += Wvvbb.reshape(1,1,nvirb,nvirb)
        Hr2baaa += Wvvaa.reshape(1,1,nvira,nvira)
        Hr2aaba += Wvvba.reshape(1,1,nvirb,nvira)
        Hr2abbb += Wvvbb.reshape(1,1,nvirb,nvirb)
        Hr2bbab += Wvvab.reshape(1,1,nvira,nvirb)

        vec_ee = self.amplitudes_to_vector((Hr1aa,Hr1bb), (Hr2aa,Hr2ab,Hr2bb))
        vec_sf = self.amplitudes_to_vector_eomsf((Hr1ab,Hr1ba), (Hr2baaa,Hr2aaba,Hr2abbb,Hr2bbab))
        return vec_ee, vec_sf

    def amplitudes_to_vector_ee(self, t1, t2, out=None):
        return self.amplitudes_to_vector_s4(t1, t2, out)

    def vector_to_amplitudes_ee(self, vector, nocc=None, nvir=None):
        return self.vector_to_amplitudes_s4(vector, nocc, nvir)

    def amplitudes_to_vector(self, t1, t2, out=None):
        nocca, nvira = t1[0].shape
        noccb, nvirb = t1[1].shape
        sizea = nocca * nvira + nocca*(nocca-1)//2*nvira*(nvira-1)//2
        sizeb = noccb * nvirb + noccb*(noccb-1)//2*nvirb*(nvirb-1)//2
        sizeab = nocca * noccb * nvira * nvirb
        vector = np.ndarray(sizea+sizeb+sizeab, t2[0].dtype, buffer=out)
        self.amplitudes_to_vector_ee(t1[0], t2[0], out=vector[:sizea])
        self.amplitudes_to_vector_ee(t1[1], t2[2], out=vector[sizea:])
        vector[sizea+sizeb:] = t2[1].ravel()
        return vector

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nocc is None:
            nocca, noccb = self.get_nocc()
        else:
            nocca, noccb = nocc
        if nmo is None:
            nmoa, nmob = self.get_nmo()
        else:
            nmoa, nmob = nmo
        nvira, nvirb = nmoa-nocca, nmob-noccb
        nocc = nocca + noccb
        nvir = nvira + nvirb
        nov = nocc * nvir
        size = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
        if vector.size == size:
            return self.vector_to_amplitudes_ee(vector, nocc, nvir)
        else:
            size = vector.size
            sizea = nocca * nvira + nocca*(nocca-1)//2*nvira*(nvira-1)//2
            sizeb = noccb * nvirb + noccb*(noccb-1)//2*nvirb*(nvirb-1)//2
            sizeab = nocca * noccb * nvira * nvirb
            t1a, t2aa = self.vector_to_amplitudes_ee(vector[:sizea], nocca, nvira)
            t1b, t2bb = self.vector_to_amplitudes_ee(vector[sizea:sizea+sizeb], noccb, nvirb)
            t2ab = vector[size-sizeab:].copy().reshape(nocca,noccb,nvira,nvirb)
            return (t1a,t1b), (t2aa,t2ab,t2bb)

    def amplitudes_from_rccsd(self, t1, t2):
        '''Convert spatial orbital T1,T2 to spin-orbital T1,T2'''
        orbspin = self.orbspin
        t2aa = t2 - t2.transpose(0,1,3,2)
        return (spatial2spin((t1, t1), orbspin),
                spatial2spin((t2aa, t2, t2aa), orbspin))

    def spatial2spin(self, tx, orbspin=None):
        if orbspin is None: orbspin = self.orbspin
        return spatial2spin(tx, orbspin)

    def spin2spatial(self, tx, orbspin=None):
        if orbspin is None: orbspin = self.orbspin
        return spin2spatial(tx, orbspin)

    def amplitudes_to_vector_eomsf(self, t1, t2, out=None):
        t1ab, t1ba = t1
        t2baaa, t2aaba, t2abbb, t2bbab = t2
        nocca, nvirb = t1ab.shape
        noccb, nvira = t1ba.shape

        nbaaa = noccb*nocca*nvira*(nvira-1)//2
        naaba = nocca*(nocca-1)//2*nvirb*nvira
        nabbb = nocca*noccb*nvirb*(nvirb-1)//2
        nbbab = noccb*(noccb-1)//2*nvira*nvirb
        size = t1ab.size + t1ba.size + nbaaa + naaba + nabbb + nbbab
        vector = numpy.ndarray(size, t2baaa.dtype, buffer=out)
        vector[:t1ab.size] = t1ab.ravel()
        vector[t1ab.size:t1ab.size+t1ba.size] = t1ba.ravel()
        pvec = vector[t1ab.size+t1ba.size:]

        t2baaa = t2baaa.reshape(noccb*nocca,nvira*nvira)
        t2aaba = t2aaba.reshape(nocca*nocca,nvirb*nvira)
        t2abbb = t2abbb.reshape(nocca*noccb,nvirb*nvirb)
        t2bbab = t2bbab.reshape(noccb*noccb,nvira*nvirb)
        otrila = numpy.tril_indices(nocca, k=-1)
        otrilb = numpy.tril_indices(noccb, k=-1)
        vtrila = numpy.tril_indices(nvira, k=-1)
        vtrilb = numpy.tril_indices(nvirb, k=-1)
        oidxab = np.arange(nocca*noccb, dtype=numpy.int32)
        vidxab = np.arange(nvira*nvirb, dtype=numpy.int32)
        lib.take_2d(t2baaa, oidxab, vtrila[0]*nvira+vtrila[1], out=pvec)
        lib.take_2d(t2aaba, otrila[0]*nocca+otrila[1], vidxab, out=pvec[nbaaa:])
        lib.take_2d(t2abbb, oidxab, vtrilb[0]*nvirb+vtrilb[1], out=pvec[nbaaa+naaba:])
        lib.take_2d(t2bbab, otrilb[0]*noccb+otrilb[1], vidxab, out=pvec[nbaaa+naaba+nabbb:])
        return vector

    def vector_to_amplitudes_eomsf(self, vector, nocc=None, nvir=None):
        if nocc is None:
            nocca, noccb = self.get_nocc()
        else:
            nocca, noccb = nocc
        if nvir is None:
            nmoa, nmob = self.get_nmo()
            nvira, nvirb = nmoa-nocca, nmob-noccb
        else:
            nvira, nvirb = nvir

        t1ab = vector[:nocca*nvirb].reshape(nocca,nvirb).copy()
        t1ba = vector[nocca*nvirb:nocca*nvirb+noccb*nvira].reshape(noccb,nvira).copy()
        pvec = vector[t1ab.size+t1ba.size:]

        nbaaa = noccb*nocca*nvira*(nvira-1)//2
        naaba = nocca*(nocca-1)//2*nvirb*nvira
        nabbb = nocca*noccb*nvirb*(nvirb-1)//2
        nbbab = noccb*(noccb-1)//2*nvira*nvirb
        t2baaa = np.zeros((noccb*nocca,nvira*nvira), dtype=vector.dtype)
        t2aaba = np.zeros((nocca*nocca,nvirb*nvira), dtype=vector.dtype)
        t2abbb = np.zeros((nocca*noccb,nvirb*nvirb), dtype=vector.dtype)
        t2bbab = np.zeros((noccb*noccb,nvira*nvirb), dtype=vector.dtype)
        otrila = numpy.tril_indices(nocca, k=-1)
        otrilb = numpy.tril_indices(noccb, k=-1)
        vtrila = numpy.tril_indices(nvira, k=-1)
        vtrilb = numpy.tril_indices(nvirb, k=-1)
        oidxab = np.arange(nocca*noccb, dtype=numpy.int32)
        vidxab = np.arange(nvira*nvirb, dtype=numpy.int32)

        v = pvec[:nbaaa].reshape(noccb*nocca,-1)
        lib.takebak_2d(t2baaa, v, oidxab, vtrila[0]*nvira+vtrila[1])
        lib.takebak_2d(t2baaa,-v, oidxab, vtrila[1]*nvira+vtrila[0])
        v = pvec[nbaaa:nbaaa+naaba].reshape(-1,nvirb*nvira)
        lib.takebak_2d(t2aaba, v, otrila[0]*nocca+otrila[1], vidxab)
        lib.takebak_2d(t2aaba,-v, otrila[1]*nocca+otrila[0], vidxab)
        v = pvec[nbaaa+naaba:nbaaa+naaba+nabbb].reshape(nocca*noccb,-1)
        lib.takebak_2d(t2abbb, v, oidxab, vtrilb[0]*nvirb+vtrilb[1])
        lib.takebak_2d(t2abbb,-v, oidxab, vtrilb[1]*nvirb+vtrilb[0])
        v = pvec[nbaaa+naaba+nabbb:].reshape(-1,nvira*nvirb)
        lib.takebak_2d(t2bbab, v, otrilb[0]*noccb+otrilb[1], vidxab)
        lib.takebak_2d(t2bbab,-v, otrilb[1]*noccb+otrilb[0], vidxab)
        t2baaa = t2baaa.reshape(noccb,nocca,nvira,nvira)
        t2aaba = t2aaba.reshape(nocca,nocca,nvirb,nvira)
        t2abbb = t2abbb.reshape(nocca,noccb,nvirb,nvirb)
        t2bbab = t2bbab.reshape(noccb,noccb,nvira,nvirb)
        return (t1ab,t1ba), (t2baaa, t2aaba, t2abbb, t2bbab)

    def spatial2spin_eomsf(self, rx, orbspin):
        '''Convert EOM spatial R1,R2 to spin-orbital R1,R2'''
        if len(rx) == 2:  # r1
            r1ab, r1ba = rx
            nocca, nvirb = r1ab.shape
            noccb, nvira = r1ba.shape
        else:
            r2baaa,r2aaba,r2abbb,r2bbab = rx
            noccb, nocca, nvira = r2baaa.shape[:3]
            nvirb = r2aaba.shape[2]

        nocc = nocca + noccb
        nvir = nvira + nvirb
        idxoa = np.where(orbspin[:nocc] == 0)[0]
        idxob = np.where(orbspin[:nocc] == 1)[0]
        idxva = np.where(orbspin[nocc:] == 0)[0]
        idxvb = np.where(orbspin[nocc:] == 1)[0]

        if len(rx) == 2:  # r1
            r1 = np.zeros((nocc,nvir), dtype=r1ab.dtype)
            lib.takebak_2d(r1, r1ab, idxoa, idxvb)
            lib.takebak_2d(r1, r1ba, idxob, idxva)
            return r1

        else:
            r2 = np.zeros((nocc**2,nvir**2), dtype=r2aaba.dtype)
            idxoaa = idxoa[:,None] * nocc + idxoa
            idxoab = idxoa[:,None] * nocc + idxob
            idxoba = idxob[:,None] * nocc + idxoa
            idxobb = idxob[:,None] * nocc + idxob
            idxvaa = idxva[:,None] * nvir + idxva
            idxvab = idxva[:,None] * nvir + idxvb
            idxvba = idxvb[:,None] * nvir + idxva
            idxvbb = idxvb[:,None] * nvir + idxvb
            r2baaa = r2baaa.reshape(noccb*nocca,nvira*nvira)
            r2aaba = r2aaba.reshape(nocca*nocca,nvirb*nvira)
            r2abbb = r2abbb.reshape(nocca*noccb,nvirb*nvirb)
            r2bbab = r2bbab.reshape(noccb*noccb,nvira*nvirb)
            lib.takebak_2d(r2, r2baaa, idxoba.ravel(), idxvaa.ravel())
            lib.takebak_2d(r2, r2aaba, idxoaa.ravel(), idxvba.ravel())
            lib.takebak_2d(r2, r2abbb, idxoab.ravel(), idxvbb.ravel())
            lib.takebak_2d(r2, r2bbab, idxobb.ravel(), idxvab.ravel())
            lib.takebak_2d(r2, r2baaa, idxoab.T.ravel(), idxvaa.T.ravel())
            lib.takebak_2d(r2, r2aaba, idxoaa.T.ravel(), idxvab.T.ravel())
            lib.takebak_2d(r2, r2abbb, idxoba.T.ravel(), idxvbb.T.ravel())
            lib.takebak_2d(r2, r2bbab, idxobb.T.ravel(), idxvba.T.ravel())
            return r2.reshape(nocc,nocc,nvir,nvir)

    def spin2spatial_eomsf(self, rx, orbspin):
        '''Convert EOM spin-orbital R1,R2 to spatial R1,R2'''
        if rx.ndim == 2:  # r1
            nocc, nvir = rx.shape
        else:
            nocc, nvir = rx.shape[1:3]

        idxoa = np.where(orbspin[:nocc] == 0)[0]
        idxob = np.where(orbspin[:nocc] == 1)[0]
        idxva = np.where(orbspin[nocc:] == 0)[0]
        idxvb = np.where(orbspin[nocc:] == 1)[0]
        nocca = len(idxoa)
        noccb = len(idxob)
        nvira = len(idxva)
        nvirb = len(idxvb)

        if rx.ndim == 2:
            r1ab = lib.take_2d(rx, idxoa, idxvb)
            r1ba = lib.take_2d(rx, idxob, idxva)
            return r1ab, r1ba
        else:
            idxoaa = idxoa[:,None] * nocc + idxoa
            idxoab = idxoa[:,None] * nocc + idxob
            idxoba = idxob[:,None] * nocc + idxoa
            idxobb = idxob[:,None] * nocc + idxob
            idxvaa = idxva[:,None] * nvir + idxva
            idxvab = idxva[:,None] * nvir + idxvb
            idxvba = idxvb[:,None] * nvir + idxva
            idxvbb = idxvb[:,None] * nvir + idxvb
            r2 = rx.reshape(nocc**2,nvir**2)
            r2baaa = lib.take_2d(r2, idxoba.ravel(), idxvaa.ravel())
            r2aaba = lib.take_2d(r2, idxoaa.ravel(), idxvba.ravel())
            r2abbb = lib.take_2d(r2, idxoab.ravel(), idxvbb.ravel())
            r2bbab = lib.take_2d(r2, idxobb.ravel(), idxvab.ravel())
            r2baaa = r2baaa.reshape(noccb,nocca,nvira,nvira)
            r2aaba = r2aaba.reshape(nocca,nocca,nvirb,nvira)
            r2abbb = r2abbb.reshape(nocca,noccb,nvirb,nvirb)
            r2bbab = r2bbab.reshape(noccb,noccb,nvira,nvirb)
            return r2baaa,r2aaba,r2abbb,r2bbab


class _ERISspin:
    def __init__(self, cc, mo_coeff=None, method='incore',
                 ao2mofn=ao2mo.outcore.general_iofree):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(cc.stdout, cc.verbose)
        moidx = get_umoidx(cc)
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = [cc.mo_coeff[0][:,moidx[0]],
                                        cc.mo_coeff[1][:,moidx[1]]]
        else:
            self.mo_coeff = mo_coeff = [mo_coeff[0][:,moidx[0]],
                                        mo_coeff[1][:,moidx[1]]]

        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = rccsd._mem_usage(nocc, nvir)
        mem_now = lib.current_memory()[0]

        self.fock, so_coeff, self.orbspin = uspatial2spin(cc, moidx, mo_coeff)
        if (cc.orbspin is None or cc.orbspin.size != self.orbspin.size or
            any(cc.orbspin != self.orbspin)):
            log.warn('Overwrite cc.orbspin by _ERIS.')
            cc.orbspin = self.orbspin

        self.feri = lib.H5TmpFile()
        if (method == 'incore' and cc._scf._eri is not None and
            (mem_incore+mem_now < cc.max_memory) or cc.mol.incore_anyway):
            idxa = self.orbspin == 0
            idxb = self.orbspin == 1
            moa = so_coeff[:,idxa]
            mob = so_coeff[:,idxb]
            nmoa = moa.shape[1]
            nmob = mob.shape[1]
            maska = numpy.where((idxa.reshape(-1,1) & idxa).ravel())[0]
            maskb = numpy.where((idxb.reshape(-1,1) & idxb).ravel())[0]
            eri = numpy.zeros((nmo*nmo,nmo*nmo))

            eri_aa = ao2mo.restore(1, ao2mo.full(cc._scf._eri, moa), nmoa)
            lib.takebak_2d(eri, eri_aa.reshape(nmoa**2,-1), maska, maska)
            eri_bb = ao2mo.restore(1, ao2mo.full(cc._scf._eri, mob), nmob)
            lib.takebak_2d(eri, eri_bb.reshape(nmob**2,-1), maskb, maskb)

            eri_ab = ao2mo.general(cc._scf._eri, (moa,moa,mob,mob), compact=False)
            eri_ba = lib.transpose(eri_ab)
            lib.takebak_2d(eri, eri_ab, maska, maskb)
            lib.takebak_2d(eri, eri_ba, maskb, maska)
            eri = eri.reshape(nmo,nmo,nmo,nmo)

            self.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
            self.ooov = eri[:nocc,:nocc,:nocc,nocc:].copy()
            self.ovoo = eri[:nocc,nocc:,:nocc,:nocc].copy()
            self.oovo = eri[:nocc,:nocc,nocc:,:nocc].copy()
            self.ovov = eri[:nocc,nocc:,:nocc,nocc:].copy()
            self.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
            self.ovvo = eri[:nocc,nocc:,nocc:,:nocc].copy()
            self.ovvv = eri[:nocc,nocc:,nocc:,nocc:].copy()
            self.vvvv = eri[nocc:,nocc:,nocc:,nocc:].copy()

        elif hasattr(cc._scf, 'with_df') and cc._scf.with_df:
            raise NotImplementedError

        else:
            orbo = so_coeff[:,:nocc]
            orbv = so_coeff[:,nocc:]
            self.dtype = so_coeff.dtype
            ds_type = so_coeff.dtype.char
            self.oooo = self.feri.create_dataset('oooo', (nocc,nocc,nocc,nocc), ds_type)
            self.ooov = self.feri.create_dataset('ooov', (nocc,nocc,nocc,nvir), ds_type)
            self.ovoo = self.feri.create_dataset('ovoo', (nocc,nvir,nocc,nocc), ds_type)
            self.oovo = self.feri.create_dataset('oovo', (nocc,nocc,nvir,nocc), ds_type)
            self.ovov = self.feri.create_dataset('ovov', (nocc,nvir,nocc,nvir), ds_type)
            self.oovv = self.feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), ds_type)
            self.ovvo = self.feri.create_dataset('ovvo', (nocc,nvir,nvir,nocc), ds_type)
            self.ovvv = self.feri.create_dataset('ovvv', (nocc,nvir,nvir,nvir), ds_type)
            self.vvvv = self.feri.create_dataset('vvvv', (nvir,nvir,nvir,nvir), ds_type)

            idxoa = self.orbspin[:nocc] == 0
            idxob = self.orbspin[:nocc] == 1
            idxva = self.orbspin[nocc:] == 0
            idxvb = self.orbspin[nocc:] == 1
            idxa = self.orbspin == 0
            idxb = self.orbspin == 1
            orbo_a = orbo[:,idxoa]
            orbo_b = orbo[:,idxob]
            orbv_a = orbv[:,idxva]
            orbv_b = orbv[:,idxvb]
            moa = so_coeff[:,idxa]
            mob = so_coeff[:,idxb]
            nocca = orbo_a.shape[1]
            noccb = orbo_b.shape[1]
            nvira = orbv_a.shape[1]
            nvirb = orbv_b.shape[1]
            nmoa = moa.shape[1]
            nmob = mob.shape[1]

            cput1 = time.clock(), time.time()
            # <ij||pq> = <ij|pq> - <ij|qp> = (ip|jq) - (iq|jp)
            tmpfile2 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            ao2mo.general(cc.mol, (orbo_a,moa,moa,moa), tmpfile2.name, 'aa')
            ao2mo.general(cc.mol, (orbo_a,moa,mob,mob), tmpfile2.name, 'ab')
            ao2mo.general(cc.mol, (orbo_b,mob,moa,moa), tmpfile2.name, 'ba')
            ao2mo.general(cc.mol, (orbo_b,mob,mob,mob), tmpfile2.name, 'bb')
            with h5py.File(tmpfile2.name) as f:
                maska1 = numpy.where(idxa)[0]
                maskb1 = numpy.where(idxb)[0]
                maska2 = numpy.where((idxa.reshape(-1,1) & idxa).ravel())[0]
                maskb2 = numpy.where((idxb.reshape(-1,1) & idxb).ravel())[0]
                bufv = numpy.empty((nmo*nmo*nmo))
                for i in range(nocc):
                    buf = numpy.zeros((nmo,nmo*nmo))
                    if self.orbspin[i] == 0:  # alpha
                        ia = numpy.count_nonzero(idxoa[:i])
                        v1 = f['aa'][ia*nmoa:ia*nmoa+nmoa]
                        v1 = lib.unpack_tril(v1, out=bufv).reshape(nmoa,-1)
                        lib.takebak_2d(buf, v1, maska1, maska2)
                        v1 = f['ab'][ia*nmoa:ia*nmoa+nmoa]
                        v1 = lib.unpack_tril(v1, out=bufv).reshape(nmoa,-1)
                        lib.takebak_2d(buf, v1, maska1, maskb2)
                    else:
                        ib = numpy.count_nonzero(idxob[:i])
                        v1 = f['ba'][ib*nmob:ib*nmob+nmob]
                        v1 = lib.unpack_tril(v1, out=bufv).reshape(nmob,-1)
                        lib.takebak_2d(buf, v1, maskb1, maska2)
                        v1 = f['bb'][ib*nmob:ib*nmob+nmob]
                        v1 = lib.unpack_tril(v1, out=bufv).reshape(nmob,-1)
                        lib.takebak_2d(buf, v1, maskb1, maskb2)
                    buf = buf.reshape(nmo,nmo,nmo)
                    self.oooo[i] = buf[:nocc,:nocc,:nocc]
                    self.ooov[i] = buf[:nocc,:nocc,nocc:]
                    self.ovoo[i] = buf[nocc:,:nocc,:nocc]
                    self.ovov[i] = buf[nocc:,:nocc,nocc:]
                    self.oovo[i] = buf[:nocc,nocc:,:nocc]
                    self.oovv[i] = buf[:nocc,nocc:,nocc:]
                    self.ovvo[i] = buf[nocc:,nocc:,:nocc]
                    self.ovvv[i] = buf[nocc:,nocc:,nocc:]
                    buf = None
                bufv = None

            cput1 = log.timer_debug1('transforming oopq, ovpq', *cput1)

            ao2mo.full(cc.mol, orbv_a, tmpfile2.name, 'aa', compact=False)
            ao2mo.full(cc.mol, orbv_b, tmpfile2.name, 'bb', compact=False)
            ao2mo.general(cc.mol, (orbv_a,orbv_a,orbv_b,orbv_b), tmpfile2.name, 'ab', compact=False)
            ao2mo.general(cc.mol, (orbv_b,orbv_b,orbv_a,orbv_a), tmpfile2.name, 'ba', compact=False)
            with h5py.File(tmpfile2.name) as f:
                maska1 = numpy.where(idxva)[0]
                maskb1 = numpy.where(idxvb)[0]
                maska2 = numpy.where((idxva.reshape(-1,1) & idxva).ravel())[0]
                maskb2 = numpy.where((idxvb.reshape(-1,1) & idxvb).ravel())[0]
                for i in range(nvir):
                    buf = numpy.zeros((nvir,nvir*nvir))
                    if idxva[i]:  # alpha
                        ia = numpy.count_nonzero(idxva[:i])
                        v1 = f['aa'][ia*nvira:ia*nvira+nvira]
                        lib.takebak_2d(buf, v1, maska1, maska2)
                        v1 = f['ab'][ia*nvira:ia*nvira+nvira]
                        lib.takebak_2d(buf, v1, maska1, maskb2)
                    else:
                        ib = numpy.count_nonzero(idxvb[:i])
                        v1 = f['ba'][ib*nvirb:ib*nvirb+nvirb]
                        lib.takebak_2d(buf, v1, maskb1, maska2)
                        v1 = f['bb'][ib*nvirb:ib*nvirb+nvirb]
                        lib.takebak_2d(buf, v1, maskb1, maskb2)
                    buf = buf.reshape(nvir,nvir,nvir)
                    self.vvvv[i] = buf
                    buf = None

            cput1 = log.timer_debug1('transforming vvvv', *cput1)

        log.timer('CCSD integral transformation', *cput0)

class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore',
                 ao2mofn=ao2mo.outcore.general_iofree):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(cc.stdout, cc.verbose)
        moidx = get_umoidx(cc)
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = [cc.mo_coeff[0][:,moidx[0]],
                                        cc.mo_coeff[1][:,moidx[1]]]
        else:
            self.mo_coeff = mo_coeff = [mo_coeff[0][:,moidx[0]],
                                        mo_coeff[1][:,moidx[1]]]

        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = rccsd._mem_usage(nocc, nvir)
        mem_now = lib.current_memory()[0]

        fock, so_coeff, self.orbspin = uspatial2spin(cc, moidx, mo_coeff)
        idxa = self.orbspin == 0
        idxb = self.orbspin == 1
        self.focka = fock[idxa][:,idxa]
        self.fockb = fock[idxb][:,idxb]
        if (cc.orbspin is None or cc.orbspin.size != self.orbspin.size or
            any(cc.orbspin != self.orbspin)):
            log.warn('Overwrite cc.orbspin by _ERIS.')
            cc.orbspin = self.orbspin

        if (method == 'incore' and cc._scf._eri is not None and
            (mem_incore+mem_now < cc.max_memory) or cc.mol.incore_anyway):
            moa = so_coeff[:,idxa]
            mob = so_coeff[:,idxb]
            nmoa = moa.shape[1]
            nmob = mob.shape[1]

            eri_aa = ao2mo.restore(1, ao2mo.full(cc._scf._eri, moa), nmoa)
            eri_bb = ao2mo.restore(1, ao2mo.full(cc._scf._eri, mob), nmob)
            eri_ab = ao2mo.general(cc._scf._eri, (moa,moa,mob,mob), compact=False)
            eri_ba = lib.transpose(eri_ab)

            self.nocca = nocca = np.count_nonzero(self.orbspin[:nocc] == 0)
            self.noccb = noccb = np.count_nonzero(self.orbspin[:nocc] == 1)
            nvira = np.count_nonzero(self.orbspin[nocc:] == 0)
            nvirb = np.count_nonzero(self.orbspin[nocc:] == 1)
            nmoa = nocca + nvira
            nmob = noccb + nvirb
            eri_aa = eri_aa.reshape(nmoa,nmoa,nmoa,nmoa)
            eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
            eri_ba = eri_ba.reshape(nmob,nmob,nmoa,nmoa)
            eri_bb = eri_bb.reshape(nmob,nmob,nmob,nmob)
            self.oooo = eri_aa[:nocca,:nocca,:nocca,:nocca].copy()
            self.ooov = eri_aa[:nocca,:nocca,:nocca,nocca:].copy()
            self.ovoo = eri_aa[:nocca,nocca:,:nocca,:nocca].copy()
            self.oovo = eri_aa[:nocca,:nocca,nocca:,:nocca].copy()
            self.ovov = eri_aa[:nocca,nocca:,:nocca,nocca:].copy()
            self.oovv = eri_aa[:nocca,:nocca,nocca:,nocca:].copy()
            self.ovvo = eri_aa[:nocca,nocca:,nocca:,:nocca].copy()
            ovvv = eri_aa[:nocca,nocca:,nocca:,nocca:].reshape(nocca*nvira,nvira,nvira)
            self.ovvv = lib.pack_tril(ovvv).reshape(nocca,nvira,nvira*(nvira+1)//2)
            ovvv = None
            self.vvvv = ao2mo.restore(4, eri_aa[nocca:,nocca:,nocca:,nocca:].copy(), nvira)
            self.OOOO = eri_bb[:noccb,:noccb,:noccb,:noccb].copy()
            self.OOOV = eri_bb[:noccb,:noccb,:noccb,noccb:].copy()
            self.OVOO = eri_bb[:noccb,noccb:,:noccb,:noccb].copy()
            self.OOVO = eri_bb[:noccb,:noccb,noccb:,:noccb].copy()
            self.OVOV = eri_bb[:noccb,noccb:,:noccb,noccb:].copy()
            self.OOVV = eri_bb[:noccb,:noccb,noccb:,noccb:].copy()
            self.OVVO = eri_bb[:noccb,noccb:,noccb:,:noccb].copy()
            OVVV = eri_bb[:noccb,noccb:,noccb:,noccb:].reshape(noccb*nvirb,nvirb,nvirb)
            self.OVVV = lib.pack_tril(OVVV).reshape(noccb,nvirb,nvirb*(nvirb+1)//2)
            OVVV = None
            self.VVVV = ao2mo.restore(4, eri_bb[noccb:,noccb:,noccb:,noccb:].copy(), nvirb)
            self.ooOO = eri_ab[:nocca,:nocca,:noccb,:noccb].copy()
            self.ooOV = eri_ab[:nocca,:nocca,:noccb,noccb:].copy()
            self.ovOO = eri_ab[:nocca,nocca:,:noccb,:noccb].copy()
            self.ooVO = eri_ab[:nocca,:nocca,noccb:,:noccb].copy()
            self.ovOV = eri_ab[:nocca,nocca:,:noccb,noccb:].copy()
            self.ooVV = eri_ab[:nocca,:nocca,noccb:,noccb:].copy()
            self.ovVO = eri_ab[:nocca,nocca:,noccb:,:noccb].copy()
            ovVV = eri_ab[:nocca,nocca:,noccb:,noccb:].reshape(nocca*nvira,nvirb,nvirb)
            self.ovVV = lib.pack_tril(ovVV).reshape(nocca,nvira,nvirb*(nvirb+1)//2)
            ovVV = None
            vvVV = eri_ab[nocca:,nocca:,noccb:,noccb:].reshape(nvira**2,nvirb**2)
            idxa = np.tril_indices(nvira)
            idxb = np.tril_indices(nvirb)
            self.vvVV = lib.take_2d(vvVV, idxa[0]*nvira+idxa[1], idxb[0]*nvirb+idxb[1])
            #self.OOoo = eri_ba[:noccb,:noccb,:nocca,:nocca].copy()
            self.OOov = eri_ba[:noccb,:noccb,:nocca,nocca:].copy()
            self.OVoo = eri_ba[:noccb,noccb:,:nocca,:nocca].copy()
            self.OOvo = eri_ba[:noccb,:noccb,nocca:,:nocca].copy()
            #self.OVov = eri_ba[:noccb,noccb:,:nocca,nocca:].copy()
            self.OOvv = eri_ba[:noccb,:noccb,nocca:,nocca:].copy()
            self.OVvo = eri_ba[:noccb,noccb:,nocca:,:nocca].copy()
            #self.OVvv = eri_ba[:noccb,noccb:,nocca:,nocca:].copy()
            OVvv = eri_ba[:noccb,noccb:,nocca:,nocca:].reshape(noccb*nvirb,nvira,nvira)
            self.OVvv = lib.pack_tril(OVvv).reshape(noccb,nvirb,nvira*(nvira+1)//2)
            OVvv = None
            #self.VVvv = eri_ba[noccb:,noccb:,nocca:,nocca:].copy()
        elif hasattr(cc._scf, 'with_df') and cc._scf.with_df:
            raise NotImplementedError
        else:
            if cc.direct:
                raise NotImplementedError
            moa = so_coeff[:,idxa]
            mob = so_coeff[:,idxb]
            nmoa = moa.shape[1]
            nmob = mob.shape[1]
            self.nocca = nocca = int(cc.mo_occ[0][moidx[0]].sum())
            self.noccb = noccb = int(cc.mo_occ[1][moidx[1]].sum())
            nvira = nmoa - nocca
            nvirb = nmob - noccb

            orboa = moa[:,:nocca]
            orbob = mob[:,:noccb]
            orbva = moa[:,nocca:]
            orbvb = mob[:,noccb:]
            self.dtype = so_coeff.dtype
            ds_type = so_coeff.dtype.char
            self.feri = lib.H5TmpFile()
            self.oooo = self.feri.create_dataset('oooo', (nocca,nocca,nocca,nocca), ds_type)
            self.ooov = self.feri.create_dataset('ooov', (nocca,nocca,nocca,nvira), ds_type)
            self.ovoo = self.feri.create_dataset('ovoo', (nocca,nvira,nocca,nocca), ds_type)
            self.oovo = self.feri.create_dataset('oovo', (nocca,nocca,nvira,nocca), ds_type)
            self.ovov = self.feri.create_dataset('ovov', (nocca,nvira,nocca,nvira), ds_type)
            self.oovv = self.feri.create_dataset('oovv', (nocca,nocca,nvira,nvira), ds_type)
            self.ovvo = self.feri.create_dataset('ovvo', (nocca,nvira,nvira,nocca), ds_type)
            self.ovvv = self.feri.create_dataset('ovvv', (nocca,nvira,nvira*(nvira+1)//2), ds_type)
            #self.vvvv = self.feri.create_dataset('vvvv', (nvira,nvira,nvira,nvira), ds_type)
            self.OOOO = self.feri.create_dataset('OOOO', (noccb,noccb,noccb,noccb), ds_type)
            self.OOOV = self.feri.create_dataset('OOOV', (noccb,noccb,noccb,nvirb), ds_type)
            self.OVOO = self.feri.create_dataset('OVOO', (noccb,nvirb,noccb,noccb), ds_type)
            self.OOVO = self.feri.create_dataset('OOVO', (noccb,noccb,nvirb,noccb), ds_type)
            self.OVOV = self.feri.create_dataset('OVOV', (noccb,nvirb,noccb,nvirb), ds_type)
            self.OOVV = self.feri.create_dataset('OOVV', (noccb,noccb,nvirb,nvirb), ds_type)
            self.OVVO = self.feri.create_dataset('OVVO', (noccb,nvirb,nvirb,noccb), ds_type)
            self.OVVV = self.feri.create_dataset('OVVV', (noccb,nvirb,nvirb*(nvirb+1)//2), ds_type)
            #self.VVVV = self.feri.create_dataset('VVVV', (nvirb,nvirb,nvirb,nvirb), ds_type)
            self.ooOO = self.feri.create_dataset('ooOO', (nocca,nocca,noccb,noccb), ds_type)
            self.ooOV = self.feri.create_dataset('ooOV', (nocca,nocca,noccb,nvirb), ds_type)
            self.ovOO = self.feri.create_dataset('ovOO', (nocca,nvira,noccb,noccb), ds_type)
            self.ooVO = self.feri.create_dataset('ooVO', (nocca,nocca,nvirb,noccb), ds_type)
            self.ovOV = self.feri.create_dataset('ovOV', (nocca,nvira,noccb,nvirb), ds_type)
            self.ooVV = self.feri.create_dataset('ooVV', (nocca,nocca,nvirb,nvirb), ds_type)
            self.ovVO = self.feri.create_dataset('ovVO', (nocca,nvira,nvirb,noccb), ds_type)
            self.ovVV = self.feri.create_dataset('ovVV', (nocca,nvira,nvirb*(nvirb+1)//2), ds_type)
            #self.vvVV = self.feri.create_dataset('vvVV', (nvira,nvira,nvirb,nvirb), ds_type)
            self.OOov = self.feri.create_dataset('OOov', (noccb,noccb,nocca,nvira), ds_type)
            self.OVoo = self.feri.create_dataset('OVoo', (noccb,nvirb,nocca,nocca), ds_type)
            self.OOvo = self.feri.create_dataset('OOvo', (noccb,noccb,nvira,nocca), ds_type)
            self.OOvv = self.feri.create_dataset('OOvv', (noccb,noccb,nvira,nvira), ds_type)
            self.OVvo = self.feri.create_dataset('OVvo', (noccb,nvirb,nvira,nocca), ds_type)
            self.OVvv = self.feri.create_dataset('OVvv', (noccb,nvirb,nvira*(nvira+1)//2), ds_type)

            cput1 = time.clock(), time.time()
            # <ij||pq> = <ij|pq> - <ij|qp> = (ip|jq) - (iq|jp)
            tmpfile2 = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            ao2mo.general(cc.mol, (orboa,moa,moa,moa), tmpfile2.name, 'aa')
            with h5py.File(tmpfile2.name) as f:
                buf = numpy.empty((nmoa,nmoa,nmoa))
                for i in range(nocca):
                    lib.unpack_tril(f['aa'][i*nmoa:(i+1)*nmoa], out=buf)
                    self.oooo[i] = buf[:nocca,:nocca,:nocca]
                    self.ooov[i] = buf[:nocca,:nocca,nocca:]
                    self.ovoo[i] = buf[nocca:,:nocca,:nocca]
                    self.ovov[i] = buf[nocca:,:nocca,nocca:]
                    self.oovo[i] = buf[:nocca,nocca:,:nocca]
                    self.oovv[i] = buf[:nocca,nocca:,nocca:]
                    self.ovvo[i] = buf[nocca:,nocca:,:nocca]
                    self.ovvv[i] = lib.pack_tril(buf[nocca:,nocca:,nocca:])
                del(f['aa'])
                buf = None

            ao2mo.general(cc.mol, (orbob,mob,mob,mob), tmpfile2.name, 'bb')
            with h5py.File(tmpfile2.name) as f:
                buf = numpy.empty((nmob,nmob,nmob))
                for i in range(noccb):
                    lib.unpack_tril(f['bb'][i*nmob:(i+1)*nmob], out=buf)
                    self.OOOO[i] = buf[:noccb,:noccb,:noccb]
                    self.OOOV[i] = buf[:noccb,:noccb,noccb:]
                    self.OVOO[i] = buf[noccb:,:noccb,:noccb]
                    self.OVOV[i] = buf[noccb:,:noccb,noccb:]
                    self.OOVO[i] = buf[:noccb,noccb:,:noccb]
                    self.OOVV[i] = buf[:noccb,noccb:,noccb:]
                    self.OVVO[i] = buf[noccb:,noccb:,:noccb]
                    self.OVVV[i] = lib.pack_tril(buf[noccb:,noccb:,noccb:])
                del(f['bb'])
                buf = None

            ao2mo.general(cc.mol, (orboa,moa,mob,mob), tmpfile2.name, 'ab')
            with h5py.File(tmpfile2.name) as f:
                buf = numpy.empty((nmoa,nmob,nmob))
                for i in range(nocca):
                    lib.unpack_tril(f['ab'][i*nmoa:(i+1)*nmoa], out=buf)
                    self.ooOO[i] = buf[:nocca,:noccb,:noccb]
                    self.ooOV[i] = buf[:nocca,:noccb,noccb:]
                    self.ovOO[i] = buf[nocca:,:noccb,:noccb]
                    self.ovOV[i] = buf[nocca:,:noccb,noccb:]
                    self.ooVO[i] = buf[:nocca,noccb:,:noccb]
                    self.ooVV[i] = buf[:nocca,noccb:,noccb:]
                    self.ovVO[i] = buf[nocca:,noccb:,:noccb]
                    self.ovVV[i] = lib.pack_tril(buf[nocca:,noccb:,noccb:])
                del(f['ab'])
                buf = None

            ao2mo.general(cc.mol, (orbob,mob,moa,moa), tmpfile2.name, 'ba')
            with h5py.File(tmpfile2.name) as f:
                buf = numpy.empty((nmob,nmoa,nmoa))
                for i in range(noccb):
                    lib.unpack_tril(f['ba'][i*nmob:(i+1)*nmob], out=buf)
                    self.OOov[i] = buf[:noccb,:nocca,nocca:]
                    self.OVoo[i] = buf[noccb:,:nocca,:nocca]
                    self.OOvo[i] = buf[:noccb,nocca:,:nocca]
                    self.OOvv[i] = buf[:noccb,nocca:,nocca:]
                    self.OVvo[i] = buf[noccb:,nocca:,:nocca]
                    self.OVvv[i] = lib.pack_tril(buf[noccb:,nocca:,nocca:])
                del(f['ba'])
                buf = None

            cput1 = log.timer_debug1('transforming oopq, ovpq', *cput1)

            ao2mo.full(cc.mol, orbva, self.feri, dataname='vvvv')
            ao2mo.full(cc.mol, orbvb, self.feri, dataname='VVVV')
            ao2mo.general(cc.mol, (orbva,orbva,orbvb,orbvb), self.feri, dataname='vvVV')
            self.vvvv = self.feri['vvvv']
            self.VVVV = self.feri['VVVV']
            self.vvVV = self.feri['vvVV']

            cput1 = log.timer_debug1('transforming vvvv', *cput1)

        log.timer('CCSD integral transformation', *cput0)


def get_umoidx(cc):
    '''Get MO boolean indices for unrestricted reference, accounting for frozen orbs.'''
    moidxa = numpy.ones(cc.mo_occ[0].size, dtype=bool)
    moidxb = numpy.ones(cc.mo_occ[1].size, dtype=bool)
    if isinstance(cc.frozen, (int, numpy.integer)):
        moidxa[:cc.frozen] = False
        moidxb[:cc.frozen] = False
#        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
#        fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
#        eab = list()
#        for a in range(2):
#            eab.append( np.diag(reduce(numpy.dot, (cc.mo_coeff[a].T, fockao[a], cc.mo_coeff[a]))) )
#        eab = np.array(eab)
#        #FIXME: if occ-energy > vir-energy, vir orbitals may be appeared in occ set and may be frozen
#        idxs = np.column_stack(np.unravel_index(np.argsort(eab.ravel()), (2, eab.shape[1])))
#        frozen = [[],[]]
#        for n, idx in zip(range(cc.frozen), idxs):
#            frozen[idx[0]].append(idx[1])
    else:
        frozen = cc.frozen
        if len(frozen) > 0 and isinstance(frozen[0], (int, numpy.integer)):
            frozen = [frozen,frozen]
        moidxa[list(frozen[0])] = False
        moidxb[list(frozen[1])] = False

    return moidxa,moidxb

def orbspin_of_sorted_mo_energy(mo_energy, mo_occ=None):
    if isinstance(mo_energy, np.ndarray) and mo_energy.ndim == 1:
        # RHF orbitals
        orbspin = np.zeros(mo_energy.size*2, dtype=int)
        orbspin[1::2] = 1
    else:  # UHF orbitals
        if mo_occ is None:
            mo_occ = np.zeros_like(mo_energy)
        idxo = np.hstack([mo_energy[0][mo_occ[0]==1],
                          mo_energy[1][mo_occ[1]==1]]).argsort()
        idxv = np.hstack([mo_energy[0][mo_occ[0]==0],
                          mo_energy[1][mo_occ[1]==0]]).argsort()
        nocca = np.count_nonzero(mo_occ[0]==1)
        nvira = np.count_nonzero(mo_occ[0]==0)
        occspin = np.zeros(idxo.size, dtype=int)
        occspin[nocca:] = 1  # label beta orbitals
        virspin = np.zeros(idxv.size, dtype=int)
        virspin[nvira:] = 1
        orbspin = np.hstack([occspin[idxo], virspin[idxv]])
    return orbspin

def uspatial2spin(cc, moidx, mo_coeff):
    '''Convert the results of an unrestricted mean-field calculation to spin-orbital form.

    Spin-orbital ordering is determined by orbital energy without regard for spin.

    Returns:
        fock : (nso,nso) ndarray
            The Fock matrix in the basis of spin-orbitals
        so_coeff : (nao, nso) ndarray
            The matrix of spin-orbital coefficients in the AO basis
        spin : (nso,) ndarary
            The spin (0 or 1) of each spin-orbital
    '''
# Note: Always recompute the fock matrix in UCCSD because the mf object may be
# converted from ROHF object in which orbital energies are eigenvalues of
# Roothaan Fock rather than the true alpha, beta orbital energies.
    dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
    fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
    fockab = [reduce(numpy.dot, (mo_coeff[0].T, fockao[0], mo_coeff[0])),
              reduce(numpy.dot, (mo_coeff[1].T, fockao[1], mo_coeff[1]))]

    mo_energy = [fockab[0].diagonal(), fockab[1].diagonal()]
    mo_occa = cc.mo_occ[0][moidx[0]]
    mo_occb = cc.mo_occ[1][moidx[1]]
    spin = orbspin_of_sorted_mo_energy(mo_energy, (mo_occa,mo_occb))

    sorta = np.hstack([np.where(mo_occa!=0)[0], np.where(mo_occa==0)[0]])
    sortb = np.hstack([np.where(mo_occb!=0)[0], np.where(mo_occb==0)[0]])
    idxa = np.where(spin == 0)[0]
    idxb = np.where(spin == 1)[0]

    nao = mo_coeff[0].shape[0]
    nmo = mo_coeff[0].shape[1] + mo_coeff[1].shape[1]
    fock = np.zeros((nmo,nmo), dtype=fockab[0].dtype)
    lib.takebak_2d(fock, lib.take_2d(fockab[0], sorta, sorta), idxa, idxa)
    lib.takebak_2d(fock, lib.take_2d(fockab[1], sortb, sortb), idxb, idxb)

    so_coeff = np.zeros((nao, nmo), dtype=mo_coeff[0].dtype)
    so_coeff[:,idxa] = mo_coeff[0][:,sorta]
    so_coeff[:,idxb] = mo_coeff[1][:,sortb]

    return fock, so_coeff, spin


class _IMDS:
    # Exactly the same as RCCSD IMDS except
    # -- rintermediates --> uintermediates
    # -- Loo, Lvv, cc_Fov --> Foo, Fvv, Fov
    # -- One less 2-virtual intermediate
    def __init__(self, cc):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.t1 = cc.t1
        self.t2 = cc.t2
        self.eris = cc.eris
        self._made_shared = False
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.made_ee_imds = False

    def _make_shared(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        t1 = spatial2spin(t1, eris.orbspin)
        t2 = spatial2spin(t2, eris.orbspin)
        nocc, nvir = t1.shape

        fov = eris.fock[:nocc,nocc:]
        foo = eris.fock[:nocc,:nocc]
        fvv = eris.fock[nocc:,nocc:]

        eris_ovvv = np.asarray(eris.ovvv)
        Fvv  = np.einsum('mf,mfae->ae', t1, eris_ovvv)
        Fvv -= np.einsum('mf,meaf->ae', t1, eris_ovvv)
        Wmbej  = lib.einsum('jf,mebf->mbej', t1, eris_ovvv)
        Wmbej -= lib.einsum('jf,mfbe->mbej', t1, eris_ovvv)
        eris_ovvv = None

        tau_tilde = imd.make_tau(t2,t1,t1,fac=0.5)
        tau = t2 + np.einsum('jf,nb->jnfb', t1, t1)
        eris_ovov = np.asarray(eris.ovov)
        eris_ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
        self.Fov = np.einsum('nf,menf->me',t1,eris_ovov)
        tau_tilde = imd.make_tau(t2,t1,t1,fac=0.5)
        Foo  = 0.5*einsum('inef,menf->mi',tau_tilde,eris_ovov)
        Fvv -= 0.5*einsum('mnaf,menf->ae',tau_tilde,eris_ovov)
        Wmbej -= einsum('jnfb,menf->mbej', tau, eris_ovov)
        eris_ovov = None

        eris_ooov = np.asarray(eris.ooov)
        Foo += np.einsum('ne,mine->mi',t1,eris_ooov)
        Foo -= np.einsum('ne,nime->mi',t1,eris_ooov)
        Wmbej += einsum('nb,mjne->mbej',t1,eris_ooov)
        Wmbej -= einsum('nb,njme->mbej',t1,eris_ooov)
        eris_ooov = None

        Foo += foo + 0.5*einsum('me,ie->mi',fov,t1)
        Foo += 0.5*einsum('me,ie->mi',self.Fov,t1)
        self.Foo = Foo
        Fvv += fvv - 0.5*einsum('me,ma->ae',fov,t1)
        Fvv -= 0.5*einsum('ma,me->ae',t1,self.Fov)
        self.Fvv = Fvv

        Wmbej += np.asarray(eris.ovvo).transpose(0,2,1,3)
        Wmbej -= np.asarray(eris.oovv).transpose(0,2,3,1)
        self.Wovvo = Wmbej

        self._made_shared = True
        log.timer('EOM-CCSD shared intermediates', *cput0)

    def make_ip(self):
        if self._made_shared is False:
            self._make_shared()

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris
        t1 = spatial2spin(t1, eris.orbspin)
        t2 = spatial2spin(t2, eris.orbspin)
        nocc, nvir = t1.shape
        tau = imd.make_tau(t2,t1,t1)

        eris_ooov = np.asarray(eris.ooov)
        eris_ooov = eris_ooov - eris_ooov.transpose(2,1,0,3)
        Woooo = lib.einsum('je,mine->mnij', t1, eris_ooov)
        Wmbij = lib.einsum('mine,jnbe->mbij', eris_ooov, t2)
        self.Wooov = eris_ooov.transpose(0,2,1,3).copy()
        eris_ooov = None

        Woooo += np.asarray(eris.oooo).transpose(0,2,1,3)
        self.Woooo = Woooo - Woooo.transpose(0,1,3,2)

        eris_ovov = np.asarray(eris.ovov)
        eris_ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
        self.Woooo += 0.5*einsum('ijef,menf->mnij', tau, eris_ovov)

        self.Wooov += lib.einsum('if,mfne->mnie', t1, eris_ovov)

        tmp  = lib.einsum('njbf,menf->mbej', t2, eris_ovov)
        Wmbij -= einsum('ie,mbej->mbij', t1, tmp)
        Wmbij += np.asarray(eris.oovo).transpose(0,2,1,3)
        eris_ovov = None
        Wmbij = Wmbij - Wmbij.transpose(0,1,3,2)

        eris_ovvo = np.asarray(eris.ovvo)
        eris_oovv = np.asarray(eris.oovv)
        tmp = lib.einsum('ie,mebj->mbij',t1, eris_ovvo)
        tmp-= lib.einsum('ie,mjbe->mbij',t1, eris_oovv)
        Wmbij += tmp - tmp.transpose(0,1,3,2)
        eris_oovv = eris_ovvo = None
        Wmbij -= lib.einsum('me,ijbe->mbij', self.Fov, t2)
        Wmbij -= lib.einsum('nb,mnij->mbij', t1, self.Woooo)

        eris_ovvv = np.asarray(eris.ovvv)
        Wmbij += 0.5 * einsum('mebf,ijef->mbij', eris_ovvv, tau)
        Wmbij -= 0.5 * einsum('mfbe,ijef->mbij', eris_ovvv, tau)
        self.Wovoo = Wmbij
        self.made_ip_imds = True
        log.timer('EOM-CCSD IP intermediates', *cput0)

    def make_ea(self):
        if self._made_shared is False:
            self._make_shared()

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris
        t1 = spatial2spin(t1, eris.orbspin)
        t2 = spatial2spin(t2, eris.orbspin)
        nocc, nvir = t1.shape
        tau = imd.make_tau(t2,t1,t1)

        eris_ooov = np.asarray(eris.ooov)
        Wabei = einsum('nime,mnab->abei',eris_ooov,tau)
        eris_ooov = None

        eris_ovov = np.asarray(eris.ovov)
        eris_ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
        Wabei -= np.einsum('me,miab->abei', self.Fov, t2)
        tmp = einsum('nibf,menf->mbei', t2, eris_ovov)
        tmp = einsum('ma,mbei->abei', t1, tmp)
        eris_ovov = None
        eris_ovvo = np.asarray(eris.ovvo)
        eris_oovv = np.asarray(eris.oovv)
        tmp += einsum('ma,mibe->abei', t1, eris_oovv)
        tmp -= einsum('ma,mebi->abei', t1, eris_ovvo)
        eris_oovv = eris_ovvo = None
        Wabei += tmp - tmp.transpose(1,0,2,3)

        eris_ovvv = np.asarray(eris.ovvv)
        eris_ovvv = eris_ovvv - eris_ovvv.transpose(0,3,2,1)
        Wabei += eris_ovvv.transpose(3,1,2,0).conj()
        tmp1 = lib.einsum('mebf,miaf->abei', eris_ovvv, t2)
        Wabei -= tmp1 - tmp1.transpose(1,0,2,3)
        self.Wvvvo = Wabei
        self.made_ea_imds = True
        log.timer('EOM-CCSD EA intermediates', *cput0)

    def make_ee(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris
        t1a, t1b = t1
        t2aa, t2ab, t2bb = t2
        nocca, noccb, nvira, nvirb = t2ab.shape

        fooa = eris.focka[:nocca,:nocca]
        foob = eris.fockb[:noccb,:noccb]
        fova = eris.focka[:nocca,nocca:]
        fovb = eris.fockb[:noccb,noccb:]
        fvva = eris.focka[nocca:,nocca:]
        fvvb = eris.fockb[noccb:,noccb:]

        self.Fooa = numpy.zeros((nocca,nocca))
        self.Foob = numpy.zeros((noccb,noccb))
        self.Fvva = numpy.zeros((nvira,nvira))
        self.Fvvb = numpy.zeros((nvirb,nvirb))

        wovvo = np.zeros((nocca,nvira,nvira,nocca))
        wOVVO = np.zeros((noccb,nvirb,nvirb,noccb))
        woVvO = np.zeros((nocca,nvirb,nvira,noccb))
        woVVo = np.zeros((nocca,nvirb,nvirb,nocca))
        wOvVo = np.zeros((noccb,nvira,nvirb,nocca))
        wOvvO = np.zeros((noccb,nvira,nvira,noccb))

        wovoo = np.zeros((nocca,nvira,nocca,nocca))
        wOVOO = np.zeros((noccb,nvirb,noccb,noccb))
        woVoO = np.zeros((nocca,nvirb,nocca,noccb))
        wOvOo = np.zeros((noccb,nvira,noccb,nocca))

        tauaa, tauab, taubb = make_tau(t2, t1, t1)
        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
        #:ovvv = eris_ovvv - eris_ovvv.transpose(0,3,2,1)
        #:self.Fvva  = np.einsum('mf,mfae->ae', t1a, ovvv)
        #:self.wovvo = lib.einsum('jf,mebf->mbej', t1a, ovvv)
        #:self.wovoo  = 0.5 * einsum('mebf,ijef->mbij', eris_ovvv, tauaa)
        #:self.wovoo -= 0.5 * einsum('mfbe,ijef->mbij', eris_ovvv, tauaa)
        mem_now = lib.current_memory()[0]
        max_memory = lib.param.MAX_MEMORY - mem_now
        blksize = max(int(max_memory*1e6/8/(nvira**3*3)), 2)
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovvv = np.asarray(eris.ovvv[p0:p1]).reshape((p1-p0)*nvira,-1)
            ovvv = lib.unpack_tril(ovvv).reshape(-1,nvira,nvira,nvira)
            ovvv = ovvv - ovvv.transpose(0,3,2,1)
            self.Fvva += np.einsum('mf,mfae->ae', t1a[p0:p1], ovvv)
            wovvo[p0:p1] = lib.einsum('jf,mebf->mbej', t1a, ovvv)
            wovoo[p0:p1] = 0.5 * einsum('mebf,ijef->mbij', ovvv, tauaa)
            ovvv = None

        #:eris_OVVV = lib.unpack_tril(np.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
        #:OVVV = eris_OVVV - eris_OVVV.transpose(0,3,2,1)
        #:self.Fvvb  = np.einsum('mf,mfae->ae', t1b, OVVV)
        #:self.wOVVO = lib.einsum('jf,mebf->mbej', t1b, OVVV)
        #:self.wOVOO  = 0.5 * einsum('mebf,ijef->mbij', OVVV, taubb)
        blksize = max(int(max_memory*1e6/8/(nvirb**3*3)), 2)
        for p0, p1 in lib.prange(0, noccb, blksize):
            OVVV = np.asarray(eris.OVVV[p0:p1]).reshape((p1-p0)*nvirb,-1)
            OVVV = lib.unpack_tril(OVVV).reshape(-1,nvirb,nvirb,nvirb)
            OVVV = OVVV - OVVV.transpose(0,3,2,1)
            self.Fvvb += np.einsum('mf,mfae->ae', t1b[p0:p1], OVVV)
            wOVVO[p0:p1] = lib.einsum('jf,mebf->mbej', t1b, OVVV)
            wOVOO[p0:p1] = 0.5 * einsum('mebf,ijef->mbij', OVVV, taubb)
            OVVV = None

        #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
        #:self.Fvvb += np.einsum('mf,mfAE->AE', t1a, eris_ovVV)
        #:self.woVvO = lib.einsum('JF,meBF->mBeJ', t1b, eris_ovVV)
        #:self.woVVo = lib.einsum('jf,mfBE->mBEj',-t1a, eris_ovVV)
        #:self.woVoO  = 0.5 * einsum('meBF,iJeF->mBiJ', eris_ovVV, tauab)
        #:self.woVoO += 0.5 * einsum('mfBE,iJfE->mBiJ', eris_ovVV, tauab)
        blksize = max(int(max_memory*1e6/8/(nvira*nvirb**2*3)), 2)
        for p0,p1 in lib.prange(0, nocca, blksize):
            ovVV = np.asarray(eris.ovVV[p0:p1]).reshape((p1-p0)*nvira,-1)
            ovVV = lib.unpack_tril(ovVV).reshape(-1,nvira,nvirb,nvirb)
            self.Fvvb += np.einsum('mf,mfAE->AE', t1a[p0:p1], ovVV)
            woVvO[p0:p1] = lib.einsum('JF,meBF->mBeJ', t1b, ovVV)
            woVVo[p0:p1] = lib.einsum('jf,mfBE->mBEj',-t1a, ovVV)
            woVoO[p0:p1] = 0.5 * einsum('meBF,iJeF->mBiJ', ovVV, tauab)
            woVoO[p0:p1]+= 0.5 * einsum('mfBE,iJfE->mBiJ', ovVV, tauab)
            ovVV = None

        #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
        #:self.Fvva += np.einsum('MF,MFae->ae', t1b, eris_OVvv)
        #:self.wOvVo = lib.einsum('jf,MEbf->MbEj', t1a, eris_OVvv)
        #:self.wOvvO = lib.einsum('JF,MFbe->MbeJ',-t1b, eris_OVvv)
        #:self.wOvOo  = 0.5 * einsum('MEbf,jIfE->MbIj', eris_OVvv, tauab)
        #:self.wOvOo += 0.5 * einsum('MFbe,jIeF->MbIj', eris_OVvv, tauab)
        blksize = max(int(max_memory*1e6/8/(nvirb*nvira**2*3)), 2)
        for p0, p1 in lib.prange(0, noccb, blksize):
            OVvv = np.asarray(eris.OVvv[p0:p1]).reshape((p1-p0)*nvirb,-1)
            OVvv = lib.unpack_tril(OVvv).reshape(-1,nvirb,nvira,nvira)
            self.Fvva += np.einsum('MF,MFae->ae', t1b[p0:p1], OVvv)
            wOvVo[p0:p1] = lib.einsum('jf,MEbf->MbEj', t1a, OVvv)
            wOvvO[p0:p1] = lib.einsum('JF,MFbe->MbeJ',-t1b, OVvv)
            wOvOo[p0:p1] = 0.5 * einsum('MEbf,jIfE->MbIj', OVvv, tauab)
            wOvOo[p0:p1]+= 0.5 * einsum('MFbe,jIeF->MbIj', OVvv, tauab)
            OVvv = None

        eris_ovov = np.asarray(eris.ovov)
        eris_OVOV = np.asarray(eris.OVOV)
        eris_ovOV = np.asarray(eris.ovOV)
        ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
        OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
        self.Fova = np.einsum('nf,menf->me', t1a,      ovov)
        self.Fova+= np.einsum('NF,meNF->me', t1b, eris_ovOV)
        self.Fovb = np.einsum('nf,menf->me', t1b,      OVOV)
        self.Fovb+= np.einsum('nf,nfME->ME', t1a, eris_ovOV)
        tilaa, tilab, tilbb = make_tau(t2,t1,t1,fac=0.5)
        self.Fooa  = einsum('inef,menf->mi', tilaa, eris_ovov)
        self.Fooa += einsum('iNeF,meNF->mi', tilab, eris_ovOV)
        self.Foob  = einsum('inef,menf->mi', tilbb, eris_OVOV)
        self.Foob += einsum('nIfE,nfME->MI', tilab, eris_ovOV)
        self.Fvva -= einsum('mnaf,menf->ae', tilaa, eris_ovov)
        self.Fvva -= einsum('mNaF,meNF->ae', tilab, eris_ovOV)
        self.Fvvb -= einsum('mnaf,menf->ae', tilbb, eris_OVOV)
        self.Fvvb -= einsum('nMfA,nfME->AE', tilab, eris_ovOV)
        wovvo -= einsum('jnfb,menf->mbej', t2aa,      ovov)
        wovvo += einsum('jNbF,meNF->mbej', t2ab, eris_ovOV)
        wOVVO -= einsum('jnfb,menf->mbej', t2bb,      OVOV)
        wOVVO += einsum('nJfB,nfME->MBEJ', t2ab, eris_ovOV)
        woVvO += einsum('nJfB,menf->mBeJ', t2ab,      ovov)
        woVvO -= einsum('JNFB,meNF->mBeJ', t2bb, eris_ovOV)
        wOvVo -= einsum('jnfb,nfME->MbEj', t2aa, eris_ovOV)
        wOvVo += einsum('jNbF,MENF->MbEj', t2ab,      OVOV)
        woVVo += einsum('jNfB,mfNE->mBEj', t2ab, eris_ovOV)
        wOvvO += einsum('nJbF,neMF->MbeJ', t2ab, eris_ovOV)

        eris_ooov = np.asarray(eris.ooov)
        eris_OOOV = np.asarray(eris.OOOV)
        eris_ooOV = np.asarray(eris.ooOV)
        eris_OOov = np.asarray(eris.OOov)
        self.Fooa += np.einsum('ne,mine->mi', t1a, eris_ooov)
        self.Fooa -= np.einsum('ne,nime->mi', t1a, eris_ooov)
        self.Fooa += np.einsum('NE,miNE->mi', t1b, eris_ooOV)
        self.Foob += np.einsum('ne,mine->mi', t1b, eris_OOOV)
        self.Foob -= np.einsum('ne,nime->mi', t1b, eris_OOOV)
        self.Foob += np.einsum('ne,MIne->MI', t1a, eris_OOov)
        eris_ooov = eris_ooov + np.einsum('jf,nfme->njme', t1a, eris_ovov)
        eris_OOOV = eris_OOOV + np.einsum('jf,nfme->njme', t1b, eris_OVOV)
        eris_ooOV = eris_ooOV + np.einsum('jf,nfme->njme', t1a, eris_ovOV)
        eris_OOov = eris_OOov + np.einsum('jf,menf->njme', t1b, eris_ovOV)
        ooov = eris_ooov - eris_ooov.transpose(2,1,0,3)
        OOOV = eris_OOOV - eris_OOOV.transpose(2,1,0,3)
        wovvo += lib.einsum('nb,mjne->mbej', t1a,      ooov)
        wOVVO += lib.einsum('nb,mjne->mbej', t1b,      OOOV)
        woVvO -= lib.einsum('NB,NJme->mBeJ', t1b, eris_OOov)
        wOvVo -= lib.einsum('nb,njME->MbEj', t1a, eris_ooOV)
        woVVo += lib.einsum('NB,mjNE->mBEj', t1b, eris_ooOV)
        wOvvO += lib.einsum('nb,MJne->MbeJ', t1a, eris_OOov)
        eris_ooov = eris_OOOV = eris_OOov = eris_ooOV = None

        self.Fooa += fooa + 0.5*einsum('me,ie->mi', self.Fova+fova, t1a)
        self.Foob += foob + 0.5*einsum('me,ie->mi', self.Fovb+fovb, t1b)
        self.Fvva += fvva - 0.5*einsum('me,ma->ae', self.Fova+fova, t1a)
        self.Fvvb += fvvb - 0.5*einsum('me,ma->ae', self.Fovb+fovb, t1b)

        # 0 or 1 virtuals
        eris_ooov = np.asarray(eris.ooov)
        eris_OOOV = np.asarray(eris.OOOV)
        eris_ooOV = np.asarray(eris.ooOV)
        eris_OOov = np.asarray(eris.OOov)
        ooov = eris_ooov - eris_ooov.transpose(2,1,0,3)
        OOOV = eris_OOOV - eris_OOOV.transpose(2,1,0,3)
        woooo = lib.einsum('je,mine->mnij', t1a,      ooov)
        wOOOO = lib.einsum('je,mine->mnij', t1b,      OOOV)
        woOoO = lib.einsum('JE,miNE->mNiJ', t1b, eris_ooOV)
        woOOo = lib.einsum('je,NIme->mNIj',-t1a, eris_OOov)
        tmpaa = lib.einsum('mine,jnbe->mbij',      ooov, t2aa)
        tmpaa+= lib.einsum('miNE,jNbE->mbij', eris_ooOV, t2ab)
        tmpbb = lib.einsum('mine,jnbe->mbij',      OOOV, t2bb)
        tmpbb+= lib.einsum('MIne,nJeB->MBIJ', eris_OOov, t2ab)
        woVoO += lib.einsum('mine,nJeB->mBiJ',      ooov, t2ab)
        woVoO += lib.einsum('miNE,JNBE->mBiJ', eris_ooOV, t2bb)
        woVoO -= lib.einsum('NIme,jNeB->mBjI', eris_OOov, t2ab)
        wOvOo += lib.einsum('MINE,jNbE->MbIj',      OOOV, t2ab)
        wOvOo += lib.einsum('MIne,jnbe->MbIj', eris_OOov, t2aa)
        wOvOo -= lib.einsum('niME,nJbE->MbJi', eris_ooOV, t2ab)
        wovoo += tmpaa - tmpaa.transpose(0,1,3,2)
        wOVOO += tmpbb - tmpbb.transpose(0,1,3,2)
        self.wooov =       ooov.transpose(0,2,1,3).copy()
        self.wOOOV =       OOOV.transpose(0,2,1,3).copy()
        self.woOoV = eris_ooOV.transpose(0,2,1,3).copy()
        self.wOoOv = eris_OOov.transpose(0,2,1,3).copy()
        self.wOooV =-eris_ooOV.transpose(2,0,1,3).copy()
        self.woOOv =-eris_OOov.transpose(2,0,1,3).copy()
        eris_ooov = eris_OOOV = eris_OOov = eris_ooOV = None

        woooo += np.asarray(eris.oooo).transpose(0,2,1,3)
        wOOOO += np.asarray(eris.OOOO).transpose(0,2,1,3)
        woOoO += np.asarray(eris.ooOO).transpose(0,2,1,3)
        self.woooo = woooo - woooo.transpose(0,1,3,2)
        self.wOOOO = wOOOO - wOOOO.transpose(0,1,3,2)
        self.woOoO = woOoO - woOOo.transpose(0,1,3,2)

        eris_ovov = np.asarray(eris.ovov)
        eris_OVOV = np.asarray(eris.OVOV)
        eris_ovOV = np.asarray(eris.ovOV)
        ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
        OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
        tauaa, tauab, taubb = make_tau(t2,t1,t1)
        self.woooo += 0.5*lib.einsum('ijef,menf->mnij', tauaa,      ovov)
        self.wOOOO += 0.5*lib.einsum('ijef,menf->mnij', taubb,      OVOV)
        self.woOoO +=     lib.einsum('iJeF,meNF->mNiJ', tauab, eris_ovOV)

        self.wooov += lib.einsum('if,mfne->mnie', t1a,      ovov)
        self.wOOOV += lib.einsum('if,mfne->mnie', t1b,      OVOV)
        self.woOoV += lib.einsum('if,mfNE->mNiE', t1a, eris_ovOV)
        self.wOoOv += lib.einsum('IF,neMF->MnIe', t1b, eris_ovOV)
        self.wOooV -= lib.einsum('if,nfME->MniE', t1a, eris_ovOV)
        self.woOOv -= lib.einsum('IF,meNF->mNIe', t1b, eris_ovOV)

        tmp1aa = lib.einsum('njbf,menf->mbej', t2aa,      ovov)
        tmp1aa-= lib.einsum('jNbF,meNF->mbej', t2ab, eris_ovOV)
        tmp1bb = lib.einsum('njbf,menf->mbej', t2bb,      OVOV)
        tmp1bb-= lib.einsum('nJfB,nfME->MBEJ', t2ab, eris_ovOV)
        tmp1ab = lib.einsum('NJBF,meNF->mBeJ', t2bb, eris_ovOV)
        tmp1ab-= lib.einsum('nJfB,menf->mBeJ', t2ab,      ovov)
        tmp1ba = lib.einsum('njbf,nfME->MbEj', t2aa, eris_ovOV)
        tmp1ba-= lib.einsum('jNbF,MENF->MbEj', t2ab,      OVOV)
        tmp1abba =-lib.einsum('jNfB,mfNE->mBEj', t2ab, eris_ovOV)
        tmp1baab =-lib.einsum('nJbF,neMF->MbeJ', t2ab, eris_ovOV)
        tmpaa = einsum('ie,mbej->mbij', t1a, tmp1aa)
        tmpbb = einsum('ie,mbej->mbij', t1b, tmp1bb)
        tmpab = einsum('ie,mBeJ->mBiJ', t1a, tmp1ab)
        tmpab-= einsum('IE,mBEj->mBjI', t1b, tmp1abba)
        tmpba = einsum('IE,MbEj->MbIj', t1b, tmp1ba)
        tmpba-= einsum('ie,MbeJ->MbJi', t1a, tmp1baab)
        wovoo -= tmpaa - tmpaa.transpose(0,1,3,2)
        wOVOO -= tmpbb - tmpbb.transpose(0,1,3,2)
        woVoO -= tmpab
        wOvOo -= tmpba
        eris_ovov = eris_OVOV = eris_ovOV = None
        eris_oovo = numpy.asarray(eris.oovo)
        eris_OOVO = numpy.asarray(eris.OOVO)
        eris_OOvo = numpy.asarray(eris.OOvo)
        eris_ooVO = numpy.asarray(eris.ooVO)
        wovoo += eris_oovo.transpose(0,2,1,3) - eris_oovo.transpose(0,2,3,1)
        wOVOO += eris_OOVO.transpose(0,2,1,3) - eris_OOVO.transpose(0,2,3,1)
        woVoO += eris_ooVO.transpose(0,2,1,3)
        wOvOo += eris_OOvo.transpose(0,2,1,3)
        eris_oovo = eris_OOVO = eris_OOvo = eris_ooVO = None

        eris_ovvo = np.asarray(eris.ovvo)
        eris_OVVO = np.asarray(eris.OVVO)
        eris_OVvo = np.asarray(eris.OVvo)
        eris_ovVO = np.asarray(eris.ovVO)
        eris_oovv = np.asarray(eris.oovv)
        eris_OOVV = np.asarray(eris.OOVV)
        eris_OOvv = np.asarray(eris.OOvv)
        eris_ooVV = np.asarray(eris.ooVV)
        wovvo += eris_ovvo.transpose(0,2,1,3)
        wOVVO += eris_OVVO.transpose(0,2,1,3)
        woVvO += eris_ovVO.transpose(0,2,1,3)
        wOvVo += eris_OVvo.transpose(0,2,1,3)
        wovvo -= eris_oovv.transpose(0,2,3,1)
        wOVVO -= eris_OOVV.transpose(0,2,3,1)
        woVVo -= eris_ooVV.transpose(0,2,3,1)
        wOvvO -= eris_OOvv.transpose(0,2,3,1)

        tmpaa = lib.einsum('ie,mebj->mbij', t1a, eris_ovvo)
        tmpbb = lib.einsum('ie,mebj->mbij', t1b, eris_OVVO)
        tmpaa-= lib.einsum('ie,mjbe->mbij', t1a, eris_oovv)
        tmpbb-= lib.einsum('ie,mjbe->mbij', t1b, eris_OOVV)
        woVoO += lib.einsum('ie,meBJ->mBiJ', t1a, eris_ovVO)
        woVoO -= lib.einsum('IE,mjBE->mBjI',-t1b, eris_ooVV)
        wOvOo += lib.einsum('IE,MEbj->MbIj', t1b, eris_OVvo)
        wOvOo -= lib.einsum('ie,MJbe->MbJi',-t1a, eris_OOvv)
        wovoo += tmpaa - tmpaa.transpose(0,1,3,2)
        wOVOO += tmpbb - tmpbb.transpose(0,1,3,2)
        wovoo -= lib.einsum('me,ijbe->mbij', self.Fova, t2aa)
        wOVOO -= lib.einsum('me,ijbe->mbij', self.Fovb, t2bb)
        woVoO += lib.einsum('me,iJeB->mBiJ', self.Fova, t2ab)
        wOvOo += lib.einsum('ME,jIbE->MbIj', self.Fovb, t2ab)
        wovoo -= lib.einsum('nb,mnij->mbij', t1a, self.woooo)
        wOVOO -= lib.einsum('nb,mnij->mbij', t1b, self.wOOOO)
        woVoO -= lib.einsum('NB,mNiJ->mBiJ', t1b, self.woOoO)
        wOvOo -= lib.einsum('nb,nMjI->MbIj', t1a, self.woOoO)
        eris_ovvo = eris_OVVO = eris_OVvo = eris_ovVO = None
        eris_oovv = eris_OOVV = eris_OOvv = eris_ooVV = None

        self.saved = lib.H5TmpFile()
        self.saved['ovvo'] = wovvo
        self.saved['OVVO'] = wOVVO
        self.saved['oVvO'] = woVvO
        self.saved['OvVo'] = wOvVo
        self.saved['oVVo'] = woVVo
        self.saved['OvvO'] = wOvvO
        self.wovvo = self.saved['ovvo']
        self.wOVVO = self.saved['OVVO']
        self.woVvO = self.saved['oVvO']
        self.wOvVo = self.saved['OvVo']
        self.woVVo = self.saved['oVVo']
        self.wOvvO = self.saved['OvvO']
        self.saved['ovoo'] = wovoo
        self.saved['OVOO'] = wOVOO
        self.saved['oVoO'] = woVoO
        self.saved['OvOo'] = wOvOo
        self.wovoo = self.saved['ovoo']
        self.wOVOO = self.saved['OVOO']
        self.woVoO = self.saved['oVoO']
        self.wOvOo = self.saved['OvOo']

        self.wvovv = self.saved.create_dataset('vovv', (nvira,nocca,nvira,nvira), t1a.dtype.char)
        self.wVOVV = self.saved.create_dataset('VOVV', (nvirb,noccb,nvirb,nvirb), t1a.dtype.char)
        self.wvOvV = self.saved.create_dataset('vOvV', (nvira,noccb,nvira,nvirb), t1a.dtype.char)
        self.wVoVv = self.saved.create_dataset('VoVv', (nvirb,nocca,nvirb,nvira), t1a.dtype.char)

        # 3 or 4 virtuals
        eris_ooov = np.asarray(eris.ooov)
        eris_ovov = np.asarray(eris.ovov)
        eris_ovOV = np.asarray(eris.ovOV)
        ovov = eris_ovov - eris_ovov.transpose(0,3,2,1)
        eris_oovv = np.asarray(eris.oovv)
        eris_ovvo = np.asarray(eris.ovvo)
        oovv = eris_oovv - eris_ovvo.transpose(0,3,2,1)
        eris_oovv = eris_ovvo = None
        #:wvovv  = .5 * lib.einsum('nime,mnab->eiab', eris_ooov, tauaa)
        #:wvovv -= .5 * lib.einsum('me,miab->eiab', self.Fova, t2aa)
        #:tmp1aa = lib.einsum('nibf,menf->mbei', t2aa,      ovov)
        #:tmp1aa-= lib.einsum('iNbF,meNF->mbei', t2ab, eris_ovOV)
        #:wvovv+= lib.einsum('ma,mbei->eiab', t1a, tmp1aa)
        #:wvovv+= einsum('ma,mibe->eiab', t1a,      oovv)
        for p0, p1 in lib.prange(0, nvira, nocca):
            wvovv  = .5*lib.einsum('nime,mnab->eiab', eris_ooov[:,:,:,p0:p1], tauaa)
            wvovv -= .5*lib.einsum('me,miab->eiab', self.Fova[:,p0:p1], t2aa)

            tmp1aa = lib.einsum('nibf,menf->mbei', t2aa, ovov[:,p0:p1])
            tmp1aa-= lib.einsum('iNbF,meNF->mbei', t2ab, eris_ovOV[:,p0:p1])
            wvovv += lib.einsum('ma,mbei->eiab', t1a, tmp1aa)
            wvovv += einsum('ma,mibe->eiab', t1a, oovv[:,:,:,p0:p1])
            self.wvovv[p0:p1] = wvovv
            tmp1aa = None
        eris_ovov = eris_ooov = eris_ovOV = None

        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
        #:ovvv = eris_ovvv - eris_ovvv.transpose(0,3,2,1)
        #:wvovv += lib.einsum('mebf,miaf->eiab',      ovvv, t2aa)
        #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
        #:wvovv += lib.einsum('MFbe,iMaF->eiab', eris_OVvv, t2ab)
        #:wvovv += eris_ovvv.transpose(2,0,3,1).conj()
        #:self.wvovv -= wvovv - wvovv.transpose(0,1,3,2)
        mem_now = lib.current_memory()[0]
        max_memory = lib.param.MAX_MEMORY - mem_now
        blksize = max(int(max_memory*1e6/8/(nvira**3*6)), 2)
        for i0,i1 in lib.prange(0, nocca, blksize):
            wvovv = self.wvovv[:,i0:i1]
            for p0,p1 in lib.prange(0, noccb, blksize):
                OVvv = np.asarray(eris.OVvv[p0:p1]).reshape((p1-p0)*nvirb,-1)
                OVvv = lib.unpack_tril(OVvv).reshape(-1,nvirb,nvira,nvira)
                wvovv -= lib.einsum('MFbe,iMaF->eiab', OVvv, t2ab[i0:i1,p0:p1])
                OVvv = None
            for p0,p1 in lib.prange(0, nocca, blksize):
                ovvv = np.asarray(eris.ovvv[p0:p1]).reshape((p1-p0)*nvira,-1)
                ovvv = lib.unpack_tril(ovvv).reshape(-1,nvira,nvira,nvira)
                if p0 == i0:
                    wvovv += ovvv.transpose(2,0,3,1).conj()
                ovvv = ovvv - ovvv.transpose(0,3,2,1)
                wvovv -= lib.einsum('mebf,miaf->eiab', ovvv, t2aa[p0:p1,i0:i1])
                ovvv = None
            wvovv = wvovv - wvovv.transpose(0,1,3,2)
            self.wvovv[:,i0:i1] = wvovv

        eris_OOOV = np.asarray(eris.OOOV)
        eris_OVOV = np.asarray(eris.OVOV)
        eris_ovOV = np.asarray(eris.ovOV)
        OVOV = eris_OVOV - eris_OVOV.transpose(0,3,2,1)
        eris_OOVV = np.asarray(eris.OOVV)
        eris_OVVO = np.asarray(eris.OVVO)
        OOVV = eris_OOVV - eris_OVVO.transpose(0,3,2,1)
        eris_OOVV = eris_OVVO = None
        #:wVOVV  = .5*lib.einsum('nime,mnab->eiab', eris_OOOV, taubb)
        #:wVOVV -= .5*lib.einsum('me,miab->eiab', self.Fovb, t2bb)
        #:tmp1bb = lib.einsum('nibf,menf->mbei', t2bb,      OVOV)
        #:tmp1bb-= lib.einsum('nIfB,nfME->MBEI', t2ab, eris_ovOV)
        #:wVOVV += lib.einsum('ma,mbei->eiab', t1b, tmp1bb)
        #:wVOVV += einsum('ma,mibe->eiab', t1b,      OOVV)
        for p0, p1 in lib.prange(0, nvirb, noccb):
            wVOVV  = .5*lib.einsum('nime,mnab->eiab', eris_OOOV[:,:,:,p0:p1], taubb)
            wVOVV -= .5*lib.einsum('me,miab->eiab', self.Fovb[:,p0:p1], t2bb)

            tmp1bb = lib.einsum('nibf,menf->mbei', t2bb, OVOV[:,p0:p1])
            tmp1bb-= lib.einsum('nIfB,nfME->MBEI', t2ab, eris_ovOV[:,:,:,p0:p1])
            wVOVV += lib.einsum('ma,mbei->eiab', t1b, tmp1bb)
            wVOVV += einsum('ma,mibe->eiab', t1b, OOVV[:,:,:,p0:p1])
            self.wVOVV[p0:p1] = wVOVV
            tmp1bb = None
        eris_OVOV = eris_OOOV = eris_ovOV = None

        #:eris_OVVV = lib.unpack_tril(np.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
        #:OVVV = eris_OVVV - eris_OVVV.transpose(0,3,2,1)
        #:wVOVV -= lib.einsum('MEBF,MIAF->EIAB',      OVVV, t2bb)
        #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
        #:wVOVV -= lib.einsum('mfBE,mIfA->EIAB', eris_ovVV, t2ab)
        #:wVOVV += eris_OVVV.transpose(2,0,3,1).conj()
        #:self.wVOVV += wVOVV - wVOVV.transpose(0,1,3,2)
        blksize = max(int(max_memory*1e6/8/(nvirb**3*6)), 2)
        for i0,i1 in lib.prange(0, noccb, blksize):
            wVOVV = self.wVOVV[:,i0:i1]
            for p0,p1 in lib.prange(0, nocca, blksize):
                ovVV = np.asarray(eris.ovVV[p0:p1]).reshape((p1-p0)*nvira,-1)
                ovVV = lib.unpack_tril(ovVV).reshape(-1,nvira,nvirb,nvirb)
                wVOVV -= lib.einsum('mfBE,mIfA->EIAB', ovVV, t2ab[p0:p1,i0:i1])
                ovVV = None
            for p0,p1 in lib.prange(0, noccb, blksize):
                OVVV = np.asarray(eris.OVVV[p0:p1]).reshape((p1-p0)*nvirb,-1)
                OVVV = lib.unpack_tril(OVVV).reshape(-1,nvirb,nvirb,nvirb)
                if p0 == i0:
                    wVOVV += OVVV.transpose(2,0,3,1).conj()
                OVVV = OVVV - OVVV.transpose(0,3,2,1)
                wVOVV -= lib.einsum('mebf,miaf->eiab', OVVV, t2bb[p0:p1,i0:i1])
                OVVV = None
            wVOVV = wVOVV - wVOVV.transpose(0,1,3,2)
            self.wVOVV[:,i0:i1] = wVOVV

        eris_ovOV = np.asarray(eris.ovOV)
        eris_OOov = np.asarray(eris.OOov)
        eris_OOvv = np.asarray(eris.OOvv)
        eris_ovVO = np.asarray(eris.ovVO)
        #:self.wvOvV = einsum('NIme,mNaB->eIaB', eris_OOov, tauab)
        #:self.wvOvV -= lib.einsum('me,mIaB->eIaB', self.Fova, t2ab)
        #:tmp1ab = lib.einsum('NIBF,meNF->mBeI', t2bb, eris_ovOV)
        #:tmp1ab-= lib.einsum('nIfB,menf->mBeI', t2ab,      ovov)
        #:tmp1baab = lib.einsum('nIbF,neMF->MbeI', t2ab, eris_ovOV)
        #:tmpab = lib.einsum('ma,mBeI->eIaB', t1a, tmp1ab)
        #:tmpab+= lib.einsum('MA,MbeI->eIbA', t1b, tmp1baab)
        #:tmpab-= einsum('MA,MIbe->eIbA', t1b, eris_OOvv)
        #:tmpab-= einsum('ma,meBI->eIaB', t1a, eris_ovVO)
        #:self.wvOvV += tmpab
        for p0, p1 in lib.prange(0, nvira, nocca):
            wvOvV  = einsum('NIme,mNaB->eIaB', eris_OOov[:,:,:,p0:p1], tauab)
            wvOvV -= lib.einsum('me,mIaB->eIaB', self.Fova[:,p0:p1], t2ab)
            tmp1ab = lib.einsum('NIBF,meNF->mBeI', t2bb, eris_ovOV[:,p0:p1])
            tmp1ab-= lib.einsum('nIfB,menf->mBeI', t2ab, ovov[:,p0:p1])
            wvOvV+= lib.einsum('ma,mBeI->eIaB', t1a, tmp1ab)
            tmp1ab = None
            tmp1baab = lib.einsum('nIbF,neMF->MbeI', t2ab, eris_ovOV[:,p0:p1])
            wvOvV+= lib.einsum('MA,MbeI->eIbA', t1b, tmp1baab)
            tmp1baab = None
            wvOvV-= einsum('MA,MIbe->eIbA', t1b, eris_OOvv[:,:,:,p0:p1])
            wvOvV-= einsum('ma,meBI->eIaB', t1a, eris_ovVO[:,p0:p1])
            self.wvOvV[p0:p1] = wvOvV
        eris_ovOV = eris_OOov = eris_OOvv = eris_ovVO = None

        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvira,nvira)
        #:ovvv = eris_ovvv - eris_ovvv.transpose(0,3,2,1)
        #:self.wvOvV -= lib.einsum('mebf,mIfA->eIbA',      ovvv, t2ab)
        #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
        #:self.wvOvV -= lib.einsum('meBF,mIaF->eIaB', eris_ovVV, t2ab)
        #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
        #:self.wvOvV -= lib.einsum('MFbe,MIAF->eIbA', eris_OVvv, t2bb)
        #:self.wvOvV += eris_OVvv.transpose(2,0,3,1).conj()
        blksize = max(int(max_memory*1e6/8/(nvira**3*6)), 2)
        for i0,i1 in lib.prange(0, nocca, blksize):
            wvOvV = self.wvOvV[:,i0:i1]
            for p0,p1 in lib.prange(0, nocca, blksize):
                ovVV = np.asarray(eris.ovVV[p0:p1]).reshape((p1-p0)*nvira,-1)
                ovVV = lib.unpack_tril(ovVV).reshape(-1,nvira,nvirb,nvirb)
                wvOvV -= lib.einsum('meBF,mIaF->eIaB', ovVV, t2ab[p0:p1,i0:i1])
                ovVV = None
            for p0,p1 in lib.prange(0, nocca, blksize):
                ovvv = np.asarray(eris.ovvv[p0:p1]).reshape((p1-p0)*nvira,-1)
                ovvv = lib.unpack_tril(ovvv).reshape(-1,nvira,nvira,nvira)
                ovvv = ovvv - ovvv.transpose(0,3,2,1)
                wvOvV -= lib.einsum('mebf,mIfA->eIbA',ovvv, t2ab[p0:p1,i0:i1])
                ovvv = None
            self.wvOvV[:,i0:i1] = wvOvV

        blksize = max(int(max_memory*1e6/8/(nvirb*nvira**2*3)), 2)
        for i0,i1 in lib.prange(0, nocca, blksize):
            wvOvV = self.wvOvV[:,i0:i1]
            for p0,p1 in lib.prange(0, nocca, blksize):
                OVvv = np.asarray(eris.OVvv[p0:p1]).reshape((p1-p0)*nvirb,-1)
                OVvv = lib.unpack_tril(OVvv).reshape(-1,nvirb,nvira,nvira)
                if p0 == i0:
                    wvOvV += OVvv.transpose(2,0,3,1).conj()
                wvOvV -= lib.einsum('MFbe,MIAF->eIbA', OVvv, t2bb[p0:p1,i0:i1])
                OVvv = None
            self.wvOvV[:,i0:i1] = wvOvV

        eris_ovOV = np.asarray(eris.ovOV)
        eris_ooOV = np.asarray(eris.ooOV)
        eris_ooVV = np.asarray(eris.ooVV)
        eris_OVvo = np.asarray(eris.OVvo)
        #:self.wVoVv = einsum('niME,nMbA->EiAb', eris_ooOV, tauab)
        #:self.wVoVv -= lib.einsum('ME,iMbA->EiAb', self.Fovb, t2ab)
        #:tmp1ba = lib.einsum('nibf,nfME->MbEi', t2aa, eris_ovOV)
        #:tmp1ba-= lib.einsum('iNbF,MENF->MbEi', t2ab,      OVOV)
        #:tmp1abba = lib.einsum('iNfB,mfNE->mBEi', t2ab, eris_ovOV)
        #:tmpba = lib.einsum('MA,MbEi->EiAb', t1b, tmp1ba)
        #:tmpba+= lib.einsum('ma,mBEi->EiBa', t1a, tmp1abba)
        #:tmpba-= einsum('ma,miBE->EiBa', t1a, eris_ooVV)
        #:tmpba-= einsum('MA,MEbi->EiAb', t1b, eris_OVvo)
        #:self.wVoVv += tmpba
        for p0, p1 in lib.prange(0, nvirb, noccb):
            wVoVv  = einsum('niME,nMbA->EiAb', eris_ooOV[:,:,:,p0:p1], tauab)
            wVoVv -= lib.einsum('ME,iMbA->EiAb', self.Fovb[:,p0:p1], t2ab)
            tmp1ba = lib.einsum('nibf,nfME->MbEi', t2aa, eris_ovOV[:,:,:,p0:p1])
            tmp1ba-= lib.einsum('iNbF,MENF->MbEi', t2ab, OVOV[:,p0:p1])
            wVoVv += lib.einsum('MA,MbEi->EiAb', t1b, tmp1ba)
            tmp1ba = None
            tmp1abba = lib.einsum('iNfB,mfNE->mBEi', t2ab, eris_ovOV[:,:,:,p0:p1])
            wVoVv += lib.einsum('ma,mBEi->EiBa', t1a, tmp1abba)
            tmp1abba = None
            wVoVv -= einsum('ma,miBE->EiBa', t1a, eris_ooVV[:,:,:,p0:p1])
            wVoVv -= einsum('MA,MEbi->EiAb', t1b, eris_OVvo[:,p0:p1])
            self.wVoVv[p0:p1] = wVoVv
        eris_ovOV = eris_ooOV = eris_ooVV = eris_OVvo = None

        #:eris_OVVV = lib.unpack_tril(np.asarray(eris.OVVV).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvirb,nvirb)
        #:OVVV = eris_OVVV - eris_OVVV.transpose(0,3,2,1)
        #:self.wVoVv -= lib.einsum('MEBF,iMaF->EiBa',      OVVV, t2ab)
        #:eris_OVvv = lib.unpack_tril(np.asarray(eris.OVvv).reshape(noccb*nvirb,-1)).reshape(noccb,nvirb,nvira,nvira)
        #:self.wVoVv -= lib.einsum('MEbf,iMfA->EiAb', eris_OVvv, t2ab)
        #:eris_ovVV = lib.unpack_tril(np.asarray(eris.ovVV).reshape(nocca*nvira,-1)).reshape(nocca,nvira,nvirb,nvirb)
        #:self.wVoVv -= lib.einsum('mfBE,miaf->EiBa', eris_ovVV, t2aa)
        #:self.wVoVv += eris_ovVV.transpose(2,0,3,1).conj()
        blksize = max(int(max_memory*1e6/8/(nvirb**3*6)), 2)
        for i0,i1 in lib.prange(0, noccb, blksize):
            wVoVv = self.wVoVv[:,i0:i1]
            for p0,p1 in lib.prange(0, noccb, blksize):
                OVvv = np.asarray(eris.OVvv[p0:p1]).reshape((p1-p0)*nvirb,-1)
                OVvv = lib.unpack_tril(OVvv).reshape(-1,nvirb,nvira,nvira)
                wVoVv -= lib.einsum('MEbf,iMfA->EiAb', OVvv, t2ab[i0:i1,p0:p1])
                OVvv = None
            for p0,p1 in lib.prange(0, noccb, blksize):
                OVVV = np.asarray(eris.OVVV[p0:p1]).reshape((p1-p0)*nvirb,-1)
                OVVV = lib.unpack_tril(OVVV).reshape(-1,nvirb,nvirb,nvirb)
                OVVV = OVVV - OVVV.transpose(0,3,2,1)
                wVoVv -= lib.einsum('MEBF,iMaF->EiBa', OVVV, t2ab[i0:i1,p0:p1])
                OVVV = None
            self.wVoVv[:,i0:i1] = wVoVv

        blksize = max(int(max_memory*1e6/8/(nvira*nvirb**2*3)), 2)
        for i0,i1 in lib.prange(0, noccb, blksize):
            wVoVv = self.wVoVv[:,i0:i1]
            for p0,p1 in lib.prange(0, noccb, blksize):
                ovVV = np.asarray(eris.ovVV[p0:p1]).reshape((p1-p0)*nvira,-1)
                ovVV = lib.unpack_tril(ovVV).reshape(-1,nvira,nvirb,nvirb)
                if p0 == i0:
                    wVoVv += ovVV.transpose(2,0,3,1).conj()
                wVoVv -= lib.einsum('mfBE,miaf->EiBa', ovVV, t2aa[p0:p1,i0:i1])
                ovVV = None
            self.wVoVv[:,i0:i1] = wVoVv

        self.made_ee_imds = True
        log.timer('EOM-CCSD EE intermediates', *cput0)


def make_tau(t2, t1, r1, fac=1, out=None):
    t1a, t1b = t1
    r1a, r1b = r1
    tau1aa = make_tau_aa(t2[0], t1a, r1a, fac, out)
    tau1bb = make_tau_aa(t2[2], t1b, r1b, fac, out)
    tau1ab = make_tau_ab(t2[1], t1, r1, fac, out)
    return tau1aa, tau1ab, tau1bb

def make_tau_aa(t2aa, t1a, r1a, fac=1, out=None):
    tau1aa = np.einsum('ia,jb->ijab', t1a, r1a)
    tau1aa-= np.einsum('ia,jb->jiab', t1a, r1a)
    tau1aa = tau1aa - tau1aa.transpose(0,1,3,2)
    tau1aa *= fac * .5
    tau1aa += t2aa
    return tau1aa

def make_tau_ab(t2ab, t1, r1, fac=1, out=None):
    t1a, t1b = t1
    r1a, r1b = r1
    tau1ab = np.einsum('ia,jb->ijab', t1a, r1b)
    tau1ab+= np.einsum('ia,jb->ijab', r1a, t1b)
    tau1ab *= fac * .5
    tau1ab += t2ab
    return tau1ab

def _add_vvvv_(cc, t2, eris, Ht2):
    t2aa, t2ab, t2bb = t2
    u2aa, u2ab, u2bb = Ht2
    rccsd._add_vvvv_(cc, t2aa, eris, u2aa)
    fakeri = lambda:None
    fakeri.vvvv = eris.VVVV
    rccsd._add_vvvv_(cc, t2bb, fakeri, u2bb)
    fakeri.vvvv = eris.vvVV
    rccsd._add_vvvv1_(cc, t2ab, fakeri, u2ab)
    return (u2aa,u2ab,u2bb)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol)
    print(mf.scf())
    # Freeze 1s electrons
    frozen = [[0,1], [0,1]]
    # also acceptable
    #frozen = 4
    ucc = UCCSD(mf, frozen=frozen)
    ecc, t1, t2 = ucc.kernel()
    print(ecc - -0.3486987472235819)

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.build()
    mf = scf.UHF(mol)
    print(mf.scf())

    mycc = UCCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.2133432712431435)
    print(mycc.ccsd_t() - -0.003060021865720902)

    e,v = mycc.ipccsd(nroots=8)
    print(e[0] - 0.4335604332073799)
    print(e[2] - 0.5187659896045407)
    print(e[4] - 0.6782876002229172)

    e,v = mycc.eaccsd(nroots=8)
    print(e[0] - 0.16737886338859731)
    print(e[2] - 0.24027613852009164)
    print(e[4] - 0.51006797826488071)

    e,v = mycc.eeccsd(nroots=4)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)
