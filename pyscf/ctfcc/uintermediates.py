from symtensor.sym_ctf import einsum

def make_tau(t2, t1, t1p, fac=1.):
    t2aa, t2ab, t2bb = t2
    t1a, t1b = t1
    t1pa, t1pb = t1p
    tauaa  = t2aa + einsum('ia,jb->ijab', fac*.5*t1a, t1pa)
    tauaa -= einsum('ib,ja->ijab', fac*.5*t1a, t1pa)
    tauaa -= einsum('ja,ib->ijab', fac*.5*t1a, t1pa)
    tauaa += einsum('jb,ia->ijab', fac*.5*t1a, t1pa)

    taubb  = t2bb + einsum('ia,jb->ijab', fac*.5*t1b, t1pb)
    taubb -= einsum('ib,ja->ijab', fac*.5*t1b, t1pb)
    taubb -= einsum('ja,ib->ijab', fac*.5*t1b, t1pb)
    taubb += einsum('jb,ia->ijab', fac*.5*t1b, t1pb)

    tauab  = t2ab+einsum('ia,jb->ijab', fac*.5*t1a, t1pb)
    tauab += einsum('jb,ia->ijab', fac*.5*t1b, t1pa)
    return tauaa, tauab, taubb

def make_tau2(t2, t1, t1p, fac=1.):
    t2aa, t2ab, t2bb = t2
    t1a, t1b = t1
    t1pa, t1pb = t1p

    tauaa  = t2aa + einsum('ia,jb->ijab', fac*.5*t1a, t1pa)
    tauaa += einsum('jb,ia->ijab', fac*.5*t1a, t1pa)

    taubb  = t2bb + einsum('ia,jb->ijab', fac*.5*t1b, t1pb)
    taubb += einsum('jb,ia->ijab', fac*.5*t1b, t1pb)

    tauab  = t2ab + einsum('ia,jb->ijab', fac*.5*t1a, t1pb)
    tauab += einsum('jb,ia->ijab', fac*.5*t1b, t1pa)
    return tauaa, tauab, taubb

def cc_Fvv(t1, t2, eris):
    t1a, t1b = t1

    tau_tildeaa,tau_tildeab,tau_tildebb = make_tau(t2,t1,t1,fac=0.5)
    fa  = eris.fvv - 0.5*einsum('me,ma->ae',eris.fov,t1a)
    fb  = eris.fVV - 0.5*einsum('me,ma->ae',eris.fOV,t1b)
    fa += einsum('mf,fmea->ae',t1a, eris.vovv.conj())
    fa -= einsum('mf,emfa->ae',t1a, eris.vovv.conj())
    fa += einsum('mf,fmea->ae',t1b, eris.VOvv.conj())

    fb += einsum('mf,fmea->ae',t1b, eris.VOVV.conj())
    fb -= einsum('mf,emfa->ae',t1b, eris.VOVV.conj())
    fb += einsum('mf,fmea->ae',t1a, eris.voVV.conj())

    tmp = eris.ovov - eris.ovov.transpose(0,3,2,1)
    fa -= einsum('mnaf,menf->ae', tau_tildeaa, tmp) * .5
    fa -= einsum('mnaf,menf->ae', tau_tildeab, eris.ovOV)

    tmp = eris.OVOV - eris.OVOV.transpose(0,3,2,1)
    fb -= einsum('mnaf,menf->ae', tau_tildebb, tmp) * .5
    fb -= einsum('mnfa,mfne->ae', tau_tildeab, eris.ovOV)

    return fa,fb


def cc_Foo(t1, t2, eris):
    t1a, t1b = t1

    tau_tildeaa,tau_tildeab,tau_tildebb= make_tau(t2,t1,t1,fac=0.5)

    fa  = eris.foo + 0.5*einsum('me,ne->mn',eris.fov,t1a)
    fb  = eris.fOO + 0.5*einsum('me,ne->mn',eris.fOV,t1b)

    fa +=einsum('oa,mnoa->mn',t1a,eris.ooov)
    fa +=einsum('oa,mnoa->mn',t1b,eris.ooOV)
    fa -=einsum('oa,onma->mn',t1a,eris.ooov)

    fb +=einsum('oa,mnoa->mn',t1b,eris.OOOV)
    fb +=einsum('oa,mnoa->mn',t1a,eris.OOov)
    fb -=einsum('oa,onma->mn',t1b,eris.OOOV)

    tmp = eris.ovov - eris.ovov.transpose(0,3,2,1)
    fa += einsum('inef,menf->mi', tau_tildeaa, tmp) * .5
    fa += einsum('inef,menf->mi',tau_tildeab,eris.ovOV)

    tmp = eris.OVOV - eris.OVOV.transpose(0,3,2,1)
    fb += einsum('inef,menf->mi',tau_tildebb, tmp) * .5
    fb += einsum('nief,nemf->mi',tau_tildeab,eris.ovOV)

    return fa,fb


def cc_Fov(t1, t2, eris):
    t1a, t1b = t1

    fa  = eris.fov + einsum('nf,menf->me',t1a,eris.ovov)
    fa +=einsum('nf,menf->me',t1b,eris.ovOV)
    fa -=einsum('nf,mfne->me',t1a,eris.ovov)
    fb  = eris.fOV + einsum('nf,menf->me',t1b,eris.OVOV)
    fb +=einsum('nf,nfme->me',t1a,eris.ovOV)
    fb -=einsum('nf,mfne->me',t1b,eris.OVOV)

    return fa,fb

def cc_Woooo(t1, t2, eris):
    t1a, t1b = t1

    tmp_aaaaJ = einsum('je,mine->minj', t1a, eris.ooov)
    tmp_aaaaJ-= tmp_aaaaJ.transpose(0,3,2,1)
    tmp_bbbbJ = einsum('je,mine->minj', t1b, eris.OOOV)
    tmp_bbbbJ-= tmp_bbbbJ.transpose(0,3,2,1)

    tmp_aabbJ = einsum('je,mine->minj', t1b, eris.ooOV)
    tmp_baabJ =-einsum('ie,mjne->minj', t1a, eris.OOov)

    Woooo = eris.oooo + tmp_aaaaJ
    WOOOO = eris.OOOO + tmp_bbbbJ
    WooOO = eris.ooOO + tmp_aabbJ
    WooOO-= tmp_baabJ.transpose(2,1,0,3)
    del tmp_aaaaJ, tmp_bbbbJ, tmp_aabbJ, tmp_baabJ,
    tau_aa, tau_ab, tau_bb = make_tau(t2, t1, t1)
    Woooo+= 0.25*einsum('ijef,menf->minj', tau_aa, eris.ovov)
    WOOOO+= 0.25*einsum('ijef,menf->minj', tau_bb, eris.OVOV)
    WooOO+= 0.5 *einsum('ijef,menf->minj', tau_ab, eris.ovOV)

    Woooo = Woooo - Woooo.transpose(2,1,0,3)
    WOOOO = WOOOO - WOOOO.transpose(2,1,0,3)
    return Woooo, WooOO, WOOOO


def cc_Wvvvv(t1, t2, eris):
    t1a, t1b = t1

    Wvvvv = eris.vvvv + einsum('mb,emfa->aebf', t1a, eris.vovv.conj())
    Wvvvv-= einsum('mb,fmea->aebf', t1a, eris.vovv.conj())
    Wvvvv-= Wvvvv.transpose(2,1,0,3)

    WvvVV = eris.vvVV - einsum('ma,emfb->aebf', t1a, eris.voVV.conj())
    WvvVV-= einsum('mb,fmea->aebf', t1b, eris.VOvv.conj())

    WVVVV = eris.VVVV + einsum('mb,emfa->aebf', t1b, eris.VOVV.conj())
    WVVVV-= einsum('mb,fmea->aebf', t1b, eris.VOVV.conj())
    WVVVV-= WVVVV.transpose(2,1,0,3)
    return Wvvvv, WvvVV, WVVVV


def cc_Wvvvv_half(t1, t2, eris):
    '''Similar to cc_Wvvvv, without anti-symmetrization'''
    t1a, t1b = t1
    Wvvvv = eris.vvvv + einsum('mb,emfa->aebf', t1a, eris.vovv.conj())
    Wvvvv-= einsum('mb,fmea->aebf', t1a, eris.vovv.conj())

    WvvVV = eris.vvVV - einsum('ma,emfb->aebf', t1a, eris.voVV.conj())
    WvvVV-= einsum('mb,fmea->aebf', t1b, eris.VOvv.conj())

    WVVVV = eris.VVVV + einsum('mb,emfa->aebf', t1b, eris.VOVV.conj())
    WVVVV-= einsum('mb,fmea->aebf', t1b, eris.VOVV.conj())
    return Wvvvv, WvvVV, WVVVV

def Wvvvv(t1, t2, eris):
    tauaa, tauab, taubb = make_tau(t2, t1, t1)
    Wvvvv, WvvVV, WVVVV = cc_Wvvvv(t1, t2, eris)
    Wvvvv += einsum('mnab,menf->aebf', tauaa, eris.ovov)
    WvvVV += einsum('mnab,menf->aebf', tauab, eris.ovOV)
    WVVVV += einsum('mnab,menf->aebf', taubb, eris.OVOV)
    return Wvvvv, WvvVV, WVVVV

def get_Wvvvv(t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    vvvv  = einsum('emfa,mb->aebf', eris.vovv.conj(), t1a)
    vvvv -= einsum('fmea,mb->aebf', eris.vovv.conj(), t1a)
    vvvv -= einsum('emfb,ma->aebf', eris.vovv.conj(), t1a)
    vvvv += einsum('fmeb,ma->aebf', eris.vovv.conj(), t1a)
    vvvv += eris.vvvv
    vvvv -= eris.vvvv.transpose(2,1,0,3)

    vvvv += einsum('mcnf,ma,nb->acbf', eris.ovov, t1a, t1a)
    vvvv -= einsum('mcnf,mb,na->acbf', eris.ovov, t1a, t1a)
    vvVV  = einsum('emfb,ma->aebf', eris.voVV.conj(),-t1a)
    vvVV += einsum('fmea,mb->aebf', eris.VOvv.conj(),-t1b)
    vvVV += einsum('mcnf,ma,nb->acbf', eris.ovOV, t1a, t1b)
    vvVV += eris.vvVV

    VVVV  = einsum('emfa,mb->aebf', eris.VOVV.conj(), t1b)
    VVVV -= einsum('fmea,mb->aebf', eris.VOVV.conj(), t1b)
    VVVV -= einsum('emfb,ma->aebf', eris.VOVV.conj(), t1b)
    VVVV += einsum('fmeb,ma->aebf', eris.VOVV.conj(), t1b)
    VVVV += eris.VVVV
    VVVV -= eris.VVVV.transpose(2,1,0,3)
    VVVV += einsum('mcnf,ma,nb->acbf', eris.OVOV, t1b, t1b)
    VVVV -= einsum('mcnf,mb,na->acbf', eris.OVOV, t1b, t1b)

    vvvv += einsum('mnab,mcnf->acbf', t2aa, eris.ovov)
    vvVV += einsum('mnab,mcnf->acbf', t2ab, eris.ovOV)
    VVVV += einsum('mnab,mcnf->acbf', t2bb, eris.OVOV)
    return vvvv, vvVV, VVVV

def cc_Wovvo(t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    Wovvo  = eris.voov.conj().transpose(1,0,3,2)
    WovVO  = eris.voOV.conj().transpose(1,0,3,2)
    WOVvo  = eris.voOV.transpose(2,3,0,1)
    WOVVO  = eris.VOOV.conj().transpose(1,0,3,2)

    Wovvo -= eris.oovv.transpose(0,3,2,1)
    WOVVO -= eris.OOVV.transpose(0,3,2,1)
    WoVVo  =-eris.ooVV.transpose(0,3,2,1)
    WOvvO  =-eris.OOvv.transpose(0,3,2,1)

    tauaa, tauab, taubb = make_tau2(t2, t1, t1,fac=2.0)

    Wovvo += einsum('jf,emfb->mebj', t1a, eris.vovv.conj())
    WOVVO += einsum('jf,emfb->mebj', t1b, eris.VOVV.conj())
    WovVO += einsum('jf,emfb->mebj', t1b, eris.voVV.conj())
    WOVvo += einsum('jf,emfb->mebj', t1a, eris.VOvv.conj())

    Wovvo -= einsum('je,emfb->mfbj', t1a, eris.vovv.conj())
    WOVVO -= einsum('je,emfb->mfbj', t1b, eris.VOVV.conj())
    WOvvO -= einsum('je,emfb->mfbj', t1b, eris.VOvv.conj())
    WoVVo -= einsum('je,emfb->mfbj', t1a, eris.voVV.conj())

    WOVvo -= einsum('nb,njme->mebj', t1a, eris.ooOV)
    WovVO -= einsum('nb,njme->mebj', t1b, eris.OOov)

    WOvvO += einsum('nb,mjne->mebj', t1a, eris.OOov)
    WoVVo += einsum('nb,mjne->mebj', t1b, eris.ooOV)

    ooov_temp = eris.ooov - eris.ooov.transpose(2,1,0,3)
    Wovvo -= einsum('nb,njme->mebj', t1a, ooov_temp)
    ooov_temp = None
    OOOV_temp = eris.OOOV - eris.OOOV.transpose(2,1,0,3)
    WOVVO -= einsum('nb,njme->mebj', t1b, OOOV_temp)
    OOOV_temp = None

    Wovvo += 0.5*einsum('jnbf,menf->mebj', t2ab, eris.ovOV)
    WOvvO += 0.5*einsum('njbf,nemf->mebj', tauab, eris.ovOV)
    WovVO -= 0.5*einsum('njbf,menf->mebj', taubb, eris.ovOV)

    WOVVO += 0.5*einsum('njfb,nfme->mebj', t2ab, eris.ovOV)
    WOVvo -= 0.5*einsum('njbf,nfme->mebj', tauaa, eris.ovOV)
    WoVVo += 0.5*einsum('jnfb,mfne->mebj', tauab, eris.ovOV)

    temp_OVOV = eris.OVOV - eris.OVOV.transpose(2,1,0,3)
    WOVVO -= 0.5*einsum('njbf,menf->mebj', taubb, temp_OVOV)
    WOVvo += 0.5*einsum('jnbf,menf->mebj', t2ab, temp_OVOV)
    temp_OVOV = None

    temp_ovov = eris.ovov - eris.ovov.transpose(2,1,0,3)
    Wovvo += 0.5*einsum('njbf,nemf->mebj', tauaa, temp_ovov)
    WovVO -= 0.5*einsum('njfb,nemf->mebj', t2ab, temp_ovov)
    temp_ovov = None

    return Wovvo, WovVO, WOVvo, WOVVO, WoVVo, WOvvO

def _cc_Wovvo_k0k2(t1, t2, eris, k0, k2):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    Wovvo = eris.voov.conj().transpose(1,0,3,2)
    WovVO = eris.voOV.conj().transpose(1,0,3,2)
    WOVvo = eris.voOV.transpose(2,3,0,1)
    WOVVO = eris.VOOV.conj().transpose(1,0,3,2)

    vovv = eris.vovv.conj()
    VOVV = eris.VOVV.conj()
    voVV = eris.voVV.conj()
    VOvv = eris.VOvv.conj()

    Wovvo-= eris.oovv.transpose(0,3,2,1)
    WOVVO-= eris.OOVV.transpose(0,3,2,1)
    WoVVo =-eris.ooVV.transpose(0,3,2,1)
    WOvvO =-eris.OOvv.transpose(0,3,2,1)

    Wovvo += einsum('jf,emfb->mebj', t1a, vovv)
    WOVVO += einsum('jf,emfb->mebj', t1b, VOVV)
    WovVO += einsum('jf,emfb->mebj', t1b, voVV)
    WOVvo += einsum('jf,emfb->mebj', t1a, VOvv)

    Wovvo -= einsum('je,emfb->mfbj', t1a, vovv)
    WOVVO -= einsum('je,emfb->mfbj', t1b, VOVV)
    WOvvO -= einsum('je,emfb->mfbj', t1b, VOvv)
    WoVVo -= einsum('je,emfb->mfbj', t1a, voVV)

    vovv = VOVV = VOvv = voVV = None

    Wovvo -= einsum('nb,njme->mebj', t1a, eris.ooov)
    WOVvo -= einsum('nb,njme->mebj', t1a, eris.ooOV)
    WOVVO -= einsum('nb,njme->mebj', t1b, eris.OOOV)
    WovVO -= einsum('nb,njme->mebj', t1b, eris.OOov)

    Wovvo += einsum('nb,mjne->mebj', t1a, eris.ooov)
    WOVVO += einsum('nb,mjne->mebj', t1b, eris.OOOV)
    WoVVo += einsum('nb,mjne->mebj', t1b, eris.ooOV)
    WOvvO += einsum('nb,mjne->mebj', t1a, eris.OOov)

    tmp = eris.ovov - eris.ovov.transpose(2,1,0,3)
    Wovvo -= 0.5*einsum('jnfb,menf->mebj', t2aa, tmp)
    Wovvo += 0.5*einsum('jnbf,menf->mebj', t2ab, eris.ovOV)
    tmp = eris.OVOV - eris.OVOV.transpose(2,1,0,3)
    WOVVO -= 0.5*einsum('jnfb,menf->mebj', t2bb, tmp)
    WOVVO += 0.5*einsum('njfb,nfme->mebj', t2ab, eris.ovOV)
    tmp = eris.ovov - eris.ovov.transpose(2,1,0,3)
    WovVO += 0.5*einsum('njfb,menf->mebj', t2ab, tmp)
    WovVO -= 0.5*einsum('jnfb,menf->mebj', t2bb, eris.ovOV)
    tmp = eris.OVOV - eris.OVOV.transpose(2,1,0,3)
    WOVvo += 0.5*einsum('jnbf,menf->mebj', t2ab, tmp)
    WOVvo -= 0.5*einsum('jnfb,nfme->mebj', t2aa, eris.ovOV)
    WoVVo += 0.5*einsum('jnfb,mfne->mebj', t2ab, eris.ovOV)
    WOvvO += 0.5*einsum('njbf,nemf->mebj', t2ab, eris.ovOV)

    tmp = einsum('menf,jf->menj', eris.ovov, t1a)
    tmp-= einsum('nemf,jf->menj', eris.ovov, t1a)
    Wovvo -= einsum('nb,menj->mebj', t1a, tmp)
    tmp = einsum('menf,jf->menj', eris.OVOV, t1b)
    tmp-= einsum('nemf,jf->menj', eris.OVOV, t1b)
    WOVVO -= einsum('nb,menj->mebj', t1b, tmp)

    WovVO -= einsum('jf,nb,menf->mebj',t1b,t1b, eris.ovOV)
    WOVvo -= einsum('jf,nb,nfme->mebj',t1a,t1a, eris.ovOV)
    WoVVo += einsum('jf,nb,mfne->mebj',t1a,t1b, eris.ovOV)
    WOvvO += einsum('jf,nb,nemf->mebj',t1b,t1a, eris.ovOV)

    return Wovvo, WovVO, WOVvo, WOVVO, WoVVo, WOvvO


def Foo(t1,t2,eris):
    t1a, t1b = t1
    Fova, Fovb = cc_Fov(t1,t2,eris)
    Fooa, Foob = cc_Foo(t1,t2,eris)
    Fooa += 0.5*einsum('ie,me->mi',t1a, Fova)
    Foob += 0.5*einsum('ie,me->mi',t1b, Fovb)
    return Fooa, Foob

def Fvv(t1,t2,eris):
    t1a, t1b = t1
    Fova, Fovb = cc_Fov(t1,t2,eris)
    Fvva, Fvvb = cc_Fvv(t1,t2,eris)
    Fvva -= 0.5*einsum('me,ma->ae', Fova, t1a)
    Fvvb -= 0.5*einsum('me,ma->ae', Fovb, t1b)
    return Fvva, Fvvb

def Fov(t1,t2,eris):
    Fme = cc_Fov(t1,t2,eris)
    return Fme

def Wvvov(t1,t2,eris):
    t1a, t1b = t1

    Wvvov = eris.vovv.transpose(3,2,1,0).conj() - eris.vovv.transpose(3,0,1,2).conj()
    WVVov = eris.voVV.transpose(3,2,1,0).conj()
    WvvOV = eris.VOvv.transpose(3,2,1,0).conj()
    WVVOV = eris.VOVV.transpose(3,2,1,0).conj() - eris.VOVV.transpose(3,0,1,2).conj()

    ovov = eris.ovov - eris.ovov.transpose(0,3,2,1)
    OVOV = eris.OVOV - eris.OVOV.transpose(0,3,2,1)

    Wvvov += -einsum('na,nemf->aemf',t1a,ovov)
    WvvOV += -einsum('na,nemf->aemf',t1a,eris.ovOV)
    WVVov += -einsum('na,nemf->aemf',t1b,eris.OVov)
    WVVOV += -einsum('na,nemf->aemf',t1b,OVOV)

    return Wvvov, WvvOV, WVVov, WVVOV

def Wvvvo(t1,t2,eris):

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    fova, fovb = cc_Fov(t1, t2, eris)

    ovvv = eris.vovv.transpose(1,0,3,2).conj() - eris.vovv.transpose(1,2,3,0).conj()
    OVvv = eris.VOvv.transpose(1,0,3,2).conj()
    ovVV = eris.voVV.transpose(1,0,3,2).conj()
    OVVV = eris.VOVV.transpose(1,0,3,2).conj() - eris.VOVV.transpose(1,2,3,0).conj()

    aebi = einsum('mebf,miaf->aebi',ovvv,t2aa)
    aebi += einsum('mfbe,imaf->aebi',eris.VOvv.transpose(1,0,3,2).conj(),t2ab)
    Wvvvo = -aebi + aebi.transpose(2,1,0,3)

    WVVvo = -einsum('mebf,imfa->aebi',OVvv,t2ab)
    WvvVO = -einsum('mebf,miaf->aebi',ovVV,t2ab)

    AEBI = einsum('mebf,miaf->aebi',OVVV,t2bb)
    AEBI += einsum('mfbe,mifa->aebi',ovVV,t2ab)
    WVVVO = -AEBI +AEBI.transpose(2,1,0,3)

    WVVvo -= einsum('mfae,mibf->aebi', ovVV, t2aa)
    WVVvo -= einsum('meaf,imbf->aebi', OVVV, t2ab)
    WvvVO -= einsum('mfae,mibf->aebi', OVvv, t2bb)
    WvvVO -= einsum('meaf,mifb->aebi', ovvv, t2ab)
    del ovvv, OVvv, ovVV, OVVV, aebi, AEBI

    ovvo = eris.voov.transpose(1,0,3,2).conj() - eris.oovv.transpose(0,3,2,1)
    OVvo = eris.VOov.transpose(1,0,3,2).conj()
    ovVO = eris.voOV.transpose(1,0,3,2).conj()
    OVVO = eris.VOOV.transpose(1,0,3,2).conj() - eris.OOVV.transpose(0,3,2,1)

    ovov = eris.ovov - eris.ovov.transpose(0,3,2,1)
    OVov = eris.OVov
    ovOV = eris.ovOV
    OVOV = eris.OVOV - eris.OVOV.transpose(0,3,2,1)

    tmp1aa = -einsum('nibf,menf->mebi',t2aa, ovov)
    tmp1aa += einsum('inbf,menf->mebi',t2ab, ovOV)

    tmp1ab = einsum('nifb,menf->mebi',t2ab, ovov)
    tmp1ab -= einsum('nibf,menf->mebi',t2bb, ovOV)

    tmp1ba = einsum('inbf,menf->mebi',t2ab, OVOV)
    tmp1ba -= einsum('nibf,menf->mebi',t2aa, OVov)

    tmp1bb = -einsum('nibf,menf->mebi',t2bb, OVOV)
    tmp1bb += einsum('nifb,menf->mebi',t2ab, OVov)

    Wvvvo -= einsum('ma,mebi->aebi',t1a,ovvo+tmp1aa)
    WVVvo -= einsum('ma,mebi->aebi',t1b,OVvo+tmp1ba)
    WvvVO -= einsum('ma,mebi->aebi',t1a,ovVO+tmp1ab)
    WVVVO -= einsum('ma,mebi->aebi',t1b,OVVO+tmp1bb)

    tmp1aa =-einsum('niaf,menf->meai',t2aa, ovov)
    tmp1aa+= einsum('inaf,menf->meai',t2ab, ovOV)
    tmp1ab = einsum('niaf,mfne->meai',t2ab, OVov)
    tmp1ba = einsum('infa,mfne->meai',t2ab, ovOV)
    tmp1bb =-einsum('niaf,menf->meai',t2bb, OVOV)
    tmp1bb+= einsum('nifa,menf->meai',t2ab, OVov)

    del ovov, ovOV, OVov, OVOV

    Wvvvo += einsum('mb,meai->aebi',t1a,(ovvo+tmp1aa))
    WVVvo += einsum('mb,meai->aebi',t1a, -eris.ooVV.transpose(0,3,2,1)+tmp1ba)
    WvvVO += einsum('mb,meai->aebi',t1b, -eris.OOvv.transpose(0,3,2,1)+tmp1ab)
    WVVVO += einsum('mb,meai->aebi',t1b,(OVVO+tmp1bb))

    del ovvo, OVVO, tmp1aa, tmp1ab, tmp1ba, tmp1bb
    # Remaining terms
    Wvvvo += eris.vovv.transpose(2,3,0,1) - eris.vovv.transpose(0,3,2,1)
    WVVvo += eris.voVV.transpose(2,3,0,1)
    WvvVO += eris.VOvv.transpose(2,3,0,1)
    WVVVO += eris.VOVV.transpose(2,3,0,1) - eris.VOVV.transpose(0,3,2,1)

    Wvvvo -= einsum('me,miab->aebi',fova,t2aa)
    WVVvo -= einsum('me,imba->aebi',fovb,t2ab)
    WvvVO -= einsum('me,miab->aebi',fova,t2ab)
    WVVVO -= einsum('me,miab->aebi',fovb,t2bb)

    Wvvvv, WvvVV, WVVVV = get_Wvvvv(t1, t2, eris)
    Wvvvo += einsum('if,aebf->aebi', t1a, Wvvvv)
    WVVvo += einsum('ie,aebf->bfai', t1a, WvvVV)
    WvvVO += einsum('if,aebf->aebi', t1b, WvvVV)
    WVVVO += einsum('if,aebf->aebi', t1b, WVVVV)
    del Wvvvv, WvvVV, WVVVV

    ovoo = eris.ooov.transpose(2,3,0,1) - eris.ooov.transpose(0,3,2,1)
    ovOO = eris.OOov.transpose(2,3,0,1)
    ooOV = eris.ooOV
    OOov = eris.OOov
    OVoo = eris.ooOV.transpose(2,3,0,1)
    OVOO = eris.OOOV.transpose(2,3,0,1) - eris.OOOV.transpose(0,3,2,1)

    tauaa, tauab, taubb = make_tau(t2, t1, t1)
    Wvvvo += 0.5*einsum('meni,mnab->aebi',ovoo,tauaa)
    WVVvo += 0.5*einsum('meni,nmba->aebi',OVoo,tauab)
    WVVvo += 0.5*einsum('mine,mnba->aebi',ooOV,tauab)
    WvvVO += 0.5*einsum('meni,mnab->aebi',ovOO,tauab)
    WvvVO += 0.5*einsum('mine,nmab->aebi',OOov,tauab)
    WVVVO += 0.5*einsum('meni,mnab->aebi',OVOO,taubb)

    return Wvvvo, WvvVO, WVVvo, WVVVO


def Woooo(t1,t2,eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    tmp_aaaaJ = einsum('je,mine->minj', t1a, eris.ooov)
    tmp_aaaaJ-= einsum('ie,mjne->minj', t1a, eris.ooov)
    tmp_bbbbJ = einsum('je,mine->minj', t1b, eris.OOOV)
    tmp_bbbbJ-= einsum('ie,mjne->minj', t1b, eris.OOOV)
    tmp_baabJ =-einsum('ie,mjne->minj', t1a, eris.OOov)
    tmp_aabbJ = einsum('je,mine->minj', t1b, eris.ooOV)

    Woooo = eris.oooo + tmp_aaaaJ
    WooOO = eris.ooOO + tmp_aabbJ - tmp_baabJ.transpose(2,1,0,3)
    WOOOO = eris.OOOO + tmp_bbbbJ

    tmp_aaaaJ  = tmp_aabbJ = tmp_baabJ = tmp_bbbbJ

    Woooo = Woooo - Woooo.transpose(2,1,0,3)
    WOOOO = WOOOO - WOOOO.transpose(2,1,0,3)

    ovov = eris.ovov - eris.ovov.transpose(0,3,2,1)
    OVOV = eris.OVOV - eris.OVOV.transpose(0,3,2,1)

    tau_aa, tau_ab, tau_bb = make_tau(t2, t1, t1)
    Woooo += 0.5*einsum('ijef,menf->minj', tau_aa,      ovov)
    WOOOO += 0.5*einsum('ijef,menf->minj', tau_bb,      OVOV)
    WooOO +=     einsum('ijef,menf->minj', tau_ab, eris.ovOV)

    WOOoo = None
    return Woooo, WooOO, WOOoo, WOOOO

def Woovo(t1,t2,eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    Woovo = eris.ooov.transpose(1,0,3,2).conj() - eris.ooov.transpose(1,2,3,0).conj()
    WooVO = eris.ooOV.transpose(1,0,3,2).conj()
    WOOvo = eris.OOov.transpose(1,0,3,2).conj()
    WOOVO = eris.OOOV.transpose(1,0,3,2).conj() - eris.OOOV.transpose(1,2,3,0).conj()
    ooov = eris.ooov - eris.ooov.transpose(2,1,0,3)
    OOOV = eris.OOOV - eris.OOOV.transpose(2,1,0,3)

    Woovo += einsum('mine,jnbe->mibj', ooov, t2aa) + einsum('mine,jnbe->mibj', eris.ooOV, t2ab)
    WooVO += einsum('mine,njeb->mibj', ooov, t2ab) + einsum('mine,jnbe->mibj', eris.ooOV, t2bb)
    WOOvo += einsum('mine,jnbe->mibj', OOOV, t2ab) + einsum('mine,jnbe->mibj', eris.OOov, t2aa)
    WOOVO += einsum('mine,jnbe->mibj', OOOV, t2bb) + einsum('mine,njeb->mibj', eris.OOov, t2ab)

    Woovo -= einsum('mjne,inbe->mibj', ooov, t2aa) + einsum('mjne,inbe->mibj', eris.ooOV, t2ab)
    WooVO -= einsum('njme,ineb->mibj', eris.OOov, t2ab)
    WOOvo -= einsum('njme,nibe->mibj', eris.ooOV, t2ab)
    WOOVO -= einsum('mjne,inbe->mibj', OOOV, t2bb) + einsum('mjne,nieb->mibj', eris.OOov, t2ab)

    ovvo = eris.voov.transpose(1,0,3,2).conj() - eris.oovv.transpose(0,3,2,1)
    OVVO = eris.VOOV.transpose(1,0,3,2).conj() - eris.OOVV.transpose(0,3,2,1)
    ovVO = eris.voOV.transpose(1,0,3,2).conj()
    OVvo = eris.VOov.transpose(1,0,3,2).conj()
    Woovo += einsum('ie,mebj->mibj', t1a, ovvo)
    WooVO += einsum('ie,mebj->mibj', t1a, ovVO)
    WOOvo += einsum('ie,mebj->mibj', t1b, OVvo)
    WOOVO += einsum('ie,mebj->mibj', t1b, OVVO)
        #P(ij)

    Woovo -= einsum('je,mebi->mibj', t1a, ovvo)
    WooVO -= -einsum('je,mibe->mibj', t1b, eris.ooVV)
    WOOvo -= -einsum('je,mibe->mibj', t1a, eris.OOvv)
    WOOVO -= einsum('je,mebi->mibj', t1b, OVVO)

    ovov = eris.ovov - eris.ovov.transpose(0,3,2,1)
    OVOV = eris.OVOV - eris.OVOV.transpose(0,3,2,1)
    Woovo -= einsum('ie,njbf,menf->mibj', t1a, t2aa, ovov) - einsum('ie,jnbf,menf->mibj', t1a, t2ab, eris.ovOV)
    WooVO -= -einsum('ie,njfb,menf->mibj', t1a, t2ab, ovov) + einsum('ie,njbf,menf->mibj', t1a, t2bb, eris.ovOV)
    WOOvo -= -einsum('ie,jnbf,menf->mibj', t1b, t2ab, OVOV) + einsum('ie,njbf,menf->mibj', t1b, t2aa, eris.OVov)
    WOOVO -= einsum('ie,njbf,menf->mibj', t1b, t2bb, OVOV) - einsum('ie,njfb,menf->mibj', t1b, t2ab, eris.OVov)
    #P(ij)
    ovov = eris.ovov - eris.ovov.transpose(0,3,2,1)
    OVOV = eris.OVOV - eris.OVOV.transpose(0,3,2,1)
    Woovo += einsum('je,nibf,menf->mibj', t1a, t2aa, ovov) - einsum('je,inbf,menf->mibj', t1a, t2ab, eris.ovOV)
    WooVO += -einsum('je,infb,mfne->mibj', t1b, t2ab, eris.ovOV)
    WOOvo += -einsum('je,nibf,mfne->mibj', t1a, t2ab, eris.OVov)
    WOOVO += einsum('je,nibf,menf->mibj', t1b, t2bb, OVOV) - einsum('je,nifb,menf->mibj', t1b, t2ab, eris.OVov)

    Fme, FME = Fov(t1, t2, eris)
    Wminj, WmiNJ, WMInj, WMINJ = Woooo(t1,t2,eris)
    tauaa, tauab, taubb = make_tau(t2, t1, t1, fac=1.)

    Woovo -= einsum('me,ijbe->mibj', Fme, t2aa)
    WooVO += einsum('me,ijeb->mibj', Fme, t2ab)
    WOOvo += einsum('me,jibe->mibj', FME, t2ab)
    WOOVO -= einsum('me,ijbe->mibj', FME, t2bb)

    Woovo -= einsum('nb,minj->mibj', t1a, Wminj)
    WooVO -= einsum('nb,minj->mibj', t1b, WmiNJ)
    WOOvo -= einsum('nb,njmi->mibj', t1a, WmiNJ)
    WOOVO -= einsum('nb,minj->mibj', t1b, WMINJ)

    ovvv = eris.vovv.transpose(1,0,3,2).conj() - eris.vovv.transpose(1,2,3,0).conj()
    OVVV = eris.VOVV.transpose(1,0,3,2).conj() - eris.VOVV.transpose(1,2,3,0).conj()
    ovVV = eris.voVV.transpose(1,0,3,2).conj()
    OVvv = eris.VOvv.transpose(1,0,3,2).conj()

    Woovo += 0.5 * einsum('mebf,ijef->mibj', ovvv, tauaa)
    WooVO += einsum('mebf,ijef->mibj', ovVV, tauab)
    WOOvo += einsum('mebf,jife->mibj', OVvv, tauab)
    WOOVO += 0.5 * einsum('mebf,ijef->mibj', OVVV, taubb)

    return Woovo, WooVO, WOOvo, WOOVO


def Wooov(t1, t2, eris):
    t1a, t1b = t1
    Wooov = eris.ooov - eris.ooov.transpose(2,1,0,3)
    WOOOV = eris.OOOV - eris.OOOV.transpose(2,1,0,3)

    Wooov += einsum('if,mfne->mine', t1a, eris.ovov) - einsum('if,nfme->mine', t1a, eris.ovov)
    WooOV = eris.ooOV + einsum('if,mfne->mine', t1a, eris.ovOV)
    WOOov = eris.OOov + einsum('if,mfne->mine', t1b, eris.OVov)
    WOOOV += einsum('if,mfne->mine', t1b, eris.OVOV) - einsum('if, nfme->mine', t1b, eris.OVOV)

    return Wooov, WooOV, WOOov, WOOOV

def Wovvo(t1, t2, eris):
    t2aa, t2ab, t2bb = t2

    Wovvo, WovVO, WOVvo, WOVVO, WoVVo, WOvvO = cc_Wovvo(t1,t2,eris)
    Wovvo += 0.5 * einsum('jnbf,menf->mebj', t2aa, eris.ovov)
    Wovvo -= 0.5 * einsum('jnbf,mfne->mebj', t2aa, eris.ovov)
    Wovvo += 0.5 * einsum('jnbf,menf->mebj', t2ab, eris.ovOV)

    WOVvo += 0.5 * einsum('jnbf,menf->mebj', t2ab, eris.OVOV)
    WOVvo -= 0.5 * einsum('jnbf,mfne->mebj', t2ab, eris.OVOV)
    WOVvo += 0.5 * einsum('jnbf,menf->mebj', t2aa, eris.OVov)

    WovVO += 0.5 * einsum('njfb,menf->mebj', t2ab, eris.ovov)
    WovVO -= 0.5 * einsum('njfb,mfne->mebj', t2ab, eris.ovov)
    WovVO += 0.5 * einsum('jnbf,menf->mebj', t2bb, eris.ovOV)

    WOVVO += 0.5 * einsum('jnbf,menf->mebj', t2bb, eris.OVOV)
    WOVVO -= 0.5 * einsum('jnbf,mfne->mebj', t2bb, eris.OVOV)
    WOVVO += 0.5 * einsum('njfb,menf->mebj', t2ab, eris.OVov)

    return Wovvo, WovVO, WOVvo, WOVVO

def W1oovv(t1, t2, eris):
    t2aa, t2ab, t2bb = t2
    Woovv = eris.oovv - eris.voov.transpose(2,1,0,3)
    WooVV = eris.ooVV
    WOOvv = eris.OOvv
    WOOVV = eris.OOVV - eris.VOOV.transpose(2,1,0,3)

    Woovv -= einsum('lckd,ilbc->kibd', eris.ovov, t2aa)
    Woovv += einsum('ldkc,ilbc->kibd', eris.ovov, t2aa)
    Woovv -= einsum('lckd,ilbc->kibd', eris.OVov, t2ab)

    WooVV -= einsum('kcld,ilcb->kibd', eris.ovOV, t2ab)
    WOOvv -= einsum('kcld,libc->kibd', eris.OVov, t2ab)

    WOOVV -= einsum('lckd,ilbc->kibd', eris.OVOV, t2bb)
    WOOVV += einsum('ldkc,ilbc->kibd', eris.OVOV, t2bb)
    WOOVV -= einsum('lckd,licb->kibd', eris.ovOV, t2ab)

    return Woovv, WooVV, WOOvv, WOOVV

def W2oovv(t1, t2, eris):
    t1a, t1b = t1

    WWooov, WWooOV, WWOOov, WWOOOV = Wooov(t1, t2, eris)

    Woovv = einsum('kild,lb->kibd',WWooov,-t1a)
    WooVV = einsum('kild,lb->kibd',WWooOV,-t1b)
    WOOvv = einsum('kild,lb->kibd',WWOOov,-t1a)
    WOOVV = einsum('kild,lb->kibd',WWOOOV,-t1b)
    del WWooov, WWooOV, WWOOov, WWOOOV

    Woovv += einsum('ckdb,ic->kibd', eris.vovv.conj(), t1a)
    Woovv -= einsum('dkcb,ic->kibd', eris.vovv.conj(), t1a)

    WooVV += einsum('ckdb,ic->kibd', eris.voVV.conj(), t1a)
    WOOvv += einsum('ckdb,ic->kibd', eris.VOvv.conj(), t1b)

    WOOVV += einsum('ckdb,ic->kibd', eris.VOVV.conj(), t1b)
    WOOVV -= einsum('dkcb,ic->kibd', eris.VOVV.conj(), t1b)

    return Woovv, WooVV, WOOvv, WOOVV

def Woovv(t1, t2, eris):
    Woovv, WooVV, WOOvv, WOOVV = W1oovv(t1, t2, eris)
    WWoovv, WWooVV, WWOOvv, WWOOVV = W2oovv(t1, t2, eris)
    Woovv = Woovv + WWoovv
    WooVV = WooVV + WWooVV
    WOOvv = WOOvv + WWOOvv
    WOOVV = WOOVV + WWOOVV
    return Woovv, WooVV, WOOvv, WOOVV
