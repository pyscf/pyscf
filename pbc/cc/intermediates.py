import numpy as np
from pyscf.pbc import lib as pbclib

#einsum = np.einsum
einsum = pbclib.einsum

#################################################
# FOLLOWING:                                    #
# J. Gauss and J. F. Stanton,                   #
# J. Chem. Phys. 103, 3561 (1995) Table III     #
#################################################

### Section (a)

def make_tau(t2, t1a, t1b, fac=1, out=None):
    tmp = einsum('ia,jb->ijab',t1a,t1b)
    t1t1 = tmp - tmp.transpose(1,0,2,3) - tmp.transpose(0,1,3,2) + tmp.transpose(1,0,3,2)
    tau1 = t2 + fac*0.50*t1t1
    return tau1

def cc_Fvv(t1,t2,eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    fov = fock[:nocc,nocc:].copy()
    fvv = fock[nocc:,nocc:].copy()
    eris_vovv = eris.ovvv.transpose(1,0,3,2)
    eris_oovv = eris.oovv.copy()
    tau_tilde = make_tau(t2,t1,t1,fac=0.5)
    Fae = ( fvv - 0.5*einsum('me,ma->ae',fov,t1)
            + einsum('mf,amef->ae',t1,eris_vovv)
            - 0.5*einsum('mnaf,mnef->ae',tau_tilde,eris_oovv) )
    return Fae

def cc_Foo(t1,t2,eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    fov = fock[:nocc,nocc:].copy()
    foo = fock[:nocc,:nocc].copy()
    eris_ooov = eris.ooov.copy()
    eris_oovv = eris.oovv.copy()
    tau_tilde = make_tau(t2,t1,t1,fac=0.5)
    Fmi = ( foo + 0.5*einsum('me,ie->mi',fov,t1) 
            + einsum('ne,mnie->mi',t1,eris_ooov)
            + 0.5*einsum('inef,mnef->mi',tau_tilde,eris_oovv) )
    return Fmi

def cc_Fov(t1,t2,eris):
    nocc, nvir = t1.shape
    eris_oovv = eris.oovv.copy()
    Fme = einsum('nf,mnef->me',t1,eris_oovv)
    return Fme

def cc_Woooo(t1,t2,eris):
    nocc, nvir = t1.shape
    eris_ooov = eris.ooov.copy()
    eris_oooo = eris.oooo.copy()
    eris_oovv = eris.oovv.copy()
    tau = make_tau(t2,t1,t1)
    tmp = einsum('je,mnie->mnij',t1,eris_ooov)
    Wmnij = eris_oooo + tmp - tmp.transpose(0,1,3,2)
    Wmnij += 0.25*einsum('ijef,mnef->mnij',tau,eris_oovv)
    return Wmnij

def cc_Wvvvv(t1,t2,eris):
    nocc, nvir = t1.shape
    eris_vovv = eris.ovvv.transpose(1,0,3,2)
    eris_vvvv = eris.vvvv.copy()
    eris_oovv = eris.oovv.copy()
    tau = make_tau(t2,t1,t1)
    tmp = einsum('mb,amef->abef',t1,eris_vovv)
    Wabef = eris_vvvv - tmp + tmp.transpose(1,0,2,3)
    Wabef += 0.25*einsum('mnab,mnef->abef',tau,eris_oovv)
    return Wabef

def cc_Wovvo(t1,t2,eris):
    nocc, nvir = t1.shape
    eris_oovv = eris.oovv.copy()
    eris_ovvo = -eris.ovov.transpose(0,1,3,2)
    eris_ovvv = eris.ovvv.copy()
    eris_oovo = -eris.ooov.transpose(0,1,3,2)
    Wmbej = eris_ovvo.copy() 
    Wmbej +=   einsum('jf,mbef->mbej',t1,eris_ovvv)
    Wmbej += - einsum('nb,mnej->mbej',t1,eris_oovo)
    Wmbej += - 0.5*einsum('jnfb,mnef->mbej',t2,eris_oovv)
    Wmbej += - einsum('jf,nb,mnef->mbej',t1,t1,eris_oovv)
    return Wmbej

### Section (b)

def Fvv(cc):
    Fae = cc_Fvv(cc) - 0.5*einsum('ma,me->ae',cc.t1,cc_Fov(cc))
    return Fae

def Foo(cc):
    Fmi = cc_Foo(cc) + 0.5*einsum('ie,me->mi',cc.t1,cc_Fov(cc))
    return Fmi

def Fov(cc):
    Fme = cc_Fov(cc)
    return Fme

def Woooo(cc):
    nocc = cc.nocc
    nvir = cc.nvir
    Wmnij = cc_Woooo(cc) 
    # Exactly the same as Wmnij in cc_Woooo:
    #Wmnij += 0.25*einsum('ijef,mnef->mnij',cc.tau,cc.w_oovv)
    Wmnij += 0.25*np.dot(cc.w_oovv.reshape(nocc*nocc,-1),
                         cc.tau.transpose(2,3,0,1).reshape(nvir*nvir,-1)).reshape((nocc,)*4)
    return Wmnij

def Wvvvv(cc):
    nocc = cc.nocc
    nvir = cc.nvir
    Wabef = cc_Wvvvv(cc) 
    # Exactly the same as Wabef in cc_Wvvvv:
    #Wabef += 0.25*einsum('mnab,mnef->abef',cc.tau,cc.w_oovv)
    Wabef += 0.25*np.dot(cc.tau.reshape(nocc*nocc,-1).T,
                         cc.w_oovv.reshape(nocc*nocc,-1)).reshape((nvir,)*4)
    return Wabef

def Wovvo(cc):
    nocc = cc.nocc
    nvir = cc.nvir
    Wmbej = cc_Wovvo(cc) 
    # Exactly the same as Wmbej in cc_Wovvo:
    #Wmbej += - 0.5*einsum('jnfb,mnef->mbej',cc.t2,cc.w_oovv)
    # --> Wmbej += - 0.5*einsum('menf,nfjb->mejb',cc.w_oovv.transpose(0,2,1,3),cc.t2.transpose(1,2,0,3)).transpose(0,3,1,2)
    Wmbej += - 0.5*np.dot(cc.w_oovv.transpose(0,2,1,3).reshape(nocc*nvir,-1),
                          cc.t2.transpose(1,2,0,3).reshape(nocc*nvir,-1)).reshape(nocc,nvir,nocc,nvir).transpose(0,3,1,2)
    return Wmbej

# Indices in the following can be safely permuted.

def Wooov(cc):
    Wmnie = cc.w_ooov + einsum('if,mnfe->mnie',cc.t1,cc.w_oovv)
    return Wmnie

def Wvovv(cc):
    Wamef = cc.w_vovv - einsum('na,nmef->amef',cc.t1,cc.w_oovv)
    return Wamef

def Wovoo(cc):
    nocc = cc.nocc
    nvir = cc.nvir
    #tmp1 = einsum('mnie,jnbe->mbij',cc.w_ooov,cc.t2)
    # --> tmp1 = einsum('mine,nejb->mijb',cc.w_ooov.transpose(0,2,1,3),cc.t2.transpose(1,3,0,2)).transpose(0,3,1,2)
    tmp1 = np.dot(cc.w_ooov.transpose(0,2,1,3).reshape(nocc*nocc,-1),
                  cc.t2.transpose(1,3,0,2).reshape(nocc*nvir,-1)).reshape(nocc,nocc,nocc,nvir).transpose(0,3,1,2)
    tmp2 = einsum('ie,mbej->mbij',cc.t1,cc.w_ovvo)
    #tmp2 += - einsum('ie,njbf,mnef->mbij',cc.t1,cc.t2,cc.w_oovv)
    #t2w = einsum('njbf,mnef->jbme',cc.t2,cc.w_oovv)
    # --> t2w = einsum('jbnf,nfme->jbme',cc.t2.transpose(1,2,0,3),cc.w_oovv.transpose(1,3,0,2))
    t2w = np.dot(cc.t2.transpose(1,2,0,3).reshape(nocc*nvir,-1), 
                 cc.w_oovv.transpose(1,3,0,2).reshape(nocc*nvir,-1)).reshape(nocc,nvir,nocc,nvir)
    tmp2 += - einsum('ie,jbme->mbij',cc.t1,t2w)
    Wmbij = cc.w_ovoo.copy()
    Wmbij += - einsum('me,ijbe->mbij',Fov(cc),cc.t2)
    woooo = Woooo(cc)
    Wmbij += - einsum('nb,mnij->mbij',cc.t1,woooo)
    #Wmbij += 0.5 * einsum('mbef,ijef->mbij',cc.w_ovvv,cc.tau)
    Wmbij += 0.5 * np.dot(cc.w_ovvv.reshape(nocc*nvir,-1),
                          cc.tau.reshape(nocc*nocc,-1).T).reshape(nocc,nvir,nocc,nocc)
    Wmbij += ( tmp1 - tmp1.transpose(0,1,3,2) )
    Wmbij += ( tmp2 - tmp2.transpose(0,1,3,2) )
    return Wmbij

def Wvvvo(cc):
    nocc = cc.nocc
    nvir = cc.nvir
    #tmp1 = einsum('mbef,miaf->abei',cc.w_ovvv,cc.t2)
    # --> tmp1 = einsum('bemf,mfia->beia',cc.w_ovvv.transpose(1,2,0,3),cc.t2.transpose(0,3,1,2)).transpose(3,0,1,2)
    tmp1 = np.dot(cc.w_ovvv.transpose(1,2,0,3).reshape(nvir*nvir,-1),
                  cc.t2.transpose(0,3,1,2).reshape(nocc*nvir,-1)).reshape(nvir,nvir,nocc,nvir).transpose(3,0,1,2)
    tmp2 = einsum('ma,mbei->abei',cc.t1,cc.w_ovvo)
    #tmp2 += - einsum('ma,nibf,mnef->abei',cc.t1,cc.t2,cc.w_oovv)
    t2w = einsum('nibf,mnef->ibme',cc.t2,cc.w_oovv)
    tmp2 += - einsum('ma,ibme->abei',cc.t1,t2w)
    Wabei = cc.w_vvvo.copy()
    Wabei += - einsum('me,miab->abei',Fov(cc),cc.t2)
    wvvvv = Wvvvv(cc)
    Wabei += einsum('if,abef->abei',cc.t1,wvvvv)
    #Wabei += 0.5 * einsum('mnei,mnab->abei',cc.w_oovo,cc.tau)
    Wabei += 0.5 * np.dot(cc.tau.reshape(nocc*nocc,-1).T,
                          cc.w_oovo.reshape(nocc*nocc,-1)).reshape(nvir,nvir,nvir,nocc)
    Wabei += ( - tmp1 + tmp1.transpose(1,0,2,3) )
    Wabei += ( - tmp2 + tmp2.transpose(1,0,2,3) )
    return Wabei

### Section (d)

def tau(cc):
    tmp = einsum('ia,jb->ijab',cc.t1,cc.t1)
    t1t1 = tmp - tmp.transpose(1,0,2,3) - tmp.transpose(0,1,3,2) + tmp.transpose(1,0,3,2)
    tau1 = cc.t2 + 0.50*t1t1
    return tau1

def tau_tilde(cc):
    tmp = einsum('ia,jb->ijab',cc.t1,cc.t1)
    t1t1 = tmp - tmp.transpose(1,0,2,3) - tmp.transpose(0,1,3,2) + tmp.transpose(1,0,3,2)
    tau1 = cc.t2 + 0.25*t1t1
    return tau1

