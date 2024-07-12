

#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Ning Zhang <ningzhang1024@gmail.com>
#

import pyscf.pbc.df.isdf.thc_cc_helper._einsum_holder as einsum_holder
from pyscf.lib import logger, module_method

einsum = einsum_holder.thc_einsum_sybolic

import pyscf.pbc.df.isdf.thc_cc_helper.thc_rintermediates as imd

class _IMDS_symbolic:
    def __init__(self, cc, eris=None, MRPT2=True):
        #print("IMDS is constructed")
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        if MRPT2: # J. Chem. Phys. 102 (4), 22 January 1995
            self.t1 = None
            self.t2 = cc._t2_expr
            self.MRPT2_approx = True
        else:
            self.t1 = cc._t1_expr
            self.t2 = cc._t2_expr
            self.MRPT2_approx = False
        self.eris = eris
        self._made_shared_2e = False
        self.t1_val = cc.t1
        self.t2_val = cc.t2
    
    def _make_shared_1e(self):
        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = self.t1, self.t2, self.eris
        self.Loo = imd.Loo(t1, t2, eris)    # return symbol
        self.Lvv = imd.Lvv(t1, t2, eris)    # return symbol
        self.Fov = imd.cc_Fov(t1, t2, eris) # return symbol

        logger.timer_debug1(self, 'EOM-CCSD shared one-electron '
                            'intermediates', *cput0)
        return self

    def _make_shared_2e(self):
        
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        # 2 virtuals
        self.Wovov = imd.Wovov(t1, t2, eris)
        self.Wovvo = imd.Wovvo(t1, t2, eris)
        if eris is None:
            eris_ovov = einsum_holder._thc_eri_ovov()
        else:
            eris_ovov = eris.ovov
        self.Woovv = eris_ovov.transpose((0,2,1,3))

        self._made_shared_2e = True
        log.timer_debug1('EOM-CCSD shared two-electron intermediates', *cput0)
        return self
    
    def make_ip(self, ip_partition=None):
        self._make_shared_1e()
        if not self._made_shared_2e and ip_partition != 'mp':
            self._make_shared_2e()

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris

        #print("ip_partition = ", ip_partition)

        # 0 or 1 virtuals
        if ip_partition != 'mp':
            self.Woooo = imd.Woooo(t1, t2, eris)
        self.Wooov = imd.Wooov(t1, t2, eris)
        self.Wovoo = imd.Wovoo(t1, t2, eris)
        log.timer_debug1('EOM-CCSD IP intermediates', *cput0)
        return self
    
    def make_ea(self, ea_partition=None):
        self._make_shared_1e()
        if not self._made_shared_2e and ea_partition != 'mp':
            self._make_shared_2e()

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1, t2, eris)
        if ea_partition == 'mp':
            self.Wvvvo = imd.Wvvvo(t1, t2, eris)
        else:
            self.Wvvvv = imd.Wvvvv(t1, t2, eris)
            self.Wvvvo = imd.Wvvvo(t1, t2, eris, self.Wvvvv)
        log.timer_debug1('EOM-CCSD EA intermediates', *cput0)
        return self
    
    
    def make_t3p2_ip(self, cc, ip_partition=None):
        raise NotImplementedError

    def make_t3p2_ea(self, cc, ea_partition=None):
        raise NotImplementedError

    def make_ee(self):
        raise NotImplementedError

################## MVP for IP/EA ##################

def ipccsd_matvec(eom, imds=None, support_multi_root=False, diag=None, thc_scheduler:einsum_holder.THC_scheduler=None):
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    #nmo = eom.nmo
    #r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)
    
    multiroots = eom._nroots > 1 and support_multi_root
    #print("multiroots = ", multiroots)
    
    r1 = einsum_holder._expr_r1_ip(multiroots)
    r2 = einsum_holder._expr_r2_ip(multiroots)

    ###### register foo, fvv ######
    fock = imds.eris.fock
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()
    fov = fock[:nocc,nocc:].copy()
    thc_scheduler.add_input("foo", foo)
    thc_scheduler.add_input("fvv", fvv)
    thc_scheduler.add_input("fov", fov)
    foo = einsum_holder.to_expr_holder(einsum_holder._expr_foo())
    fvv = einsum_holder.to_expr_holder(einsum_holder._expr_fvv())
    fov = einsum_holder.to_expr_holder(einsum_holder._expr_fov())
    ###############################

    ###### register intermediates ######
    thc_scheduler.register_intermediates("FOV", imds.Fov)
    thc_scheduler.register_intermediates("LOO", imds.Loo)
    thc_scheduler.register_intermediates("LVV", imds.Lvv)
    ####################################

    if not multiroots:
        # 1h-1h blocks
        Hr1 = -einsum('ki,k->i', imds.Loo, r1)
        # 1h-2h1p block
        Hr1 += 2*einsum('ld,ild->i', imds.Fov, r2)
        Hr1 +=  -einsum('kd,kid->i', imds.Fov, r2)
        Hr1 +=-2*einsum('klid,kld->i', imds.Wooov, r2)
        Hr1 +=   einsum('lkid,kld->i', imds.Wooov, r2)
    else:
        # 1h-1h block
        Hr1 = -einsum('ki,pk->pi', imds.Loo, r1)
        # 1h-2h1p block
        Hr1 += 2*einsum('ld,pild->pi', imds.Fov, r2)
        Hr1 +=  -einsum('kd,pkid->pi', imds.Fov, r2)
        Hr1 +=-2*einsum('klid,pkld->pi', imds.Wooov, r2)
        Hr1 +=   einsum('lkid,pkld->pi', imds.Wooov, r2)
        

    if not multiroots:
        # 2h1p-1h block
        Hr2 = -einsum('kbij,k->ijb', imds.Wovoo, r1)
    else:
        # 2h1p-1h block
        Hr2 = -einsum('kbij,pk->pijb', imds.Wovoo, r1)
    
    if not multiroots:
        # 2h1p-2h1p block
        if eom.partition == 'mp':
            #fock = imds.eris.fock
            #foo = fock[:nocc,:nocc]
            #fvv = fock[nocc:,nocc:]
            Hr2 +=  einsum('bd,ijd->ijb', fvv, r2)
            Hr2 += -einsum('ki,kjb->ijb', foo, r2)
            Hr2 += -einsum('lj,ilb->ijb', foo, r2)
        elif eom.partition == 'full':
            raise NotImplementedError
            #diag_matrix2 = eom.vector_to_amplitudes(diag, nmo, nocc)[1]
            #Hr2 += diag_matrix2 * r2
        else:
            Hr2 +=  einsum('bd,ijd->ijb', imds.Lvv, r2)
            Hr2 += -einsum('ki,kjb->ijb', imds.Loo, r2)
            Hr2 += -einsum('lj,ilb->ijb', imds.Loo, r2)
            Hr2 +=  einsum('klij,klb->ijb', imds.Woooo, r2)
            Hr2 +=2*einsum('lbdj,ild->ijb', imds.Wovvo, r2)
            Hr2 += -einsum('kbdj,kid->ijb', imds.Wovvo, r2)
            Hr2 += -einsum('lbjd,ild->ijb', imds.Wovov, r2) #typo in Ref
            Hr2 += -einsum('kbid,kjd->ijb', imds.Wovov, r2)
            tmp = 2*einsum('lkdc,kld->c', imds.Woovv, r2)
            tmp += -einsum('kldc,kld->c', imds.Woovv, r2)
            Hr2 += -einsum('c,ijcb->ijb', tmp, imds.t2)
    else:
        # 2h1p-2h1p block
        if eom.partition == 'mp':
            #fock = imds.eris.fock
            #foo = fock[:nocc,:nocc]
            #fvv = fock[nocc:,nocc:]
            Hr2 +=  einsum('bd,pijd->pijb', fvv, r2)
            Hr2 += -einsum('ki,pkjb->pijb', foo, r2)
            Hr2 += -einsum('lj,pilb->pijb', foo, r2)
        elif eom.partition == 'full':
            raise NotImplementedError
            #diag_matrix2 = eom.vector_to_amplitudes(diag, nmo, nocc)[1]
            #Hr2 += diag_matrix2 * r2
        else:
            Hr2 +=  einsum('bd,pijd->pijb', imds.Lvv, r2)
            Hr2 += -einsum('ki,pkjb->pijb', imds.Loo, r2)
            Hr2 += -einsum('lj,pilb->pijb', imds.Loo, r2)
            Hr2 +=  einsum('klij,pklb->pijb', imds.Woooo, r2)
            Hr2 +=2*einsum('lbdj,pild->pijb', imds.Wovvo, r2)
            Hr2 += -einsum('kbdj,pkid->pijb', imds.Wovvo, r2)
            Hr2 += -einsum('lbjd,pild->pijb', imds.Wovov, r2) #typo in Ref
            Hr2 += -einsum('kbid,pkjd->pijb', imds.Wovov, r2)
            tmp = 2*einsum('lkdc,pkld->pc', imds.Woovv, r2)
            tmp += -einsum('kldc,pkld->pc', imds.Woovv, r2)
            Hr2 += -einsum('pc,ijcb->pijb', tmp, imds.t2)
    
    thc_scheduler.register_expr(einsum_holder.THC_scheduler.ip_hr1_r_name, Hr1)
    thc_scheduler.register_expr(einsum_holder.THC_scheduler.ip_hr2_r_name, Hr2)

def lipccsd_matvec(eom, imds=None, support_multi_root=False, diag=None, thc_scheduler:einsum_holder.THC_scheduler=None):
    '''For left eigenvector'''
    # Note this is not the same left EA equations used by Nooijen and Bartlett.
    # Small changes were made so that the same type L2 basis was used for both the
    # left EA and left IP equations.  You will note more similarity for these
    # equations to the left IP equations than for the left EA equations by Nooijen.
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    #nmo = eom.nmo
    #r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)

    multiroots = eom._nroots > 1 and support_multi_root
    
    r1 = einsum_holder._expr_r1_ip(multiroots)
    r2 = einsum_holder._expr_r2_ip(multiroots)
    
    ###### register foo, fvv ######
    fock = imds.eris.fock
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()
    fov = fock[:nocc,nocc:].copy()
    thc_scheduler.add_input("foo", foo)
    thc_scheduler.add_input("fvv", fvv)
    thc_scheduler.add_input("fov", fov)
    foo = einsum_holder.to_expr_holder(einsum_holder._expr_foo())
    fvv = einsum_holder.to_expr_holder(einsum_holder._expr_fvv())
    fov = einsum_holder.to_expr_holder(einsum_holder._expr_fov())
    ###############################

    ###### register intermediates ######
    thc_scheduler.register_intermediates("FOV", imds.Fov)
    thc_scheduler.register_intermediates("LOO", imds.Loo)
    thc_scheduler.register_intermediates("LVV", imds.Lvv)
    ####################################
    
    if not multiroots:
        # 1h-1h block
        Hr1 =  -einsum('ki,i->k', imds.Loo, r1)
        # 1h-2h1p block
        Hr1 += -einsum('kbij,ijb->k', imds.Wovoo, r2)
    else:
        # 1h-1h block
        Hr1 =  -einsum('ki,pi->pk', imds.Loo, r1)
        # 1h-2h1p block
        Hr1 += -einsum('kbij,pijb->pk', imds.Wovoo, r2)

    if not multiroots:
        # 2h1p-1h block
        Hr2  =   -einsum('kd,l->kld', imds.Fov, r1)
        Hr2 += 2.*einsum('ld,k->kld', imds.Fov, r1)
        #Hr2 +=   -einsum('klid,i->kld', 2.*imds.Wooov-imds.Wooov.transpose((1,0,2,3)), r1)
        _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Wooov)
        _expr_tmp2 = einsum_holder.to_expr_holder(imds.Wooov.transpose((1,0,2,3)))
        _expr_tmp = _expr_tmp1 - _expr_tmp2
        Hr2 +=   -einsum('klid,i->kld', _expr_tmp, r1)
    else:
        # 2h1p-1h block
        Hr2  =   -einsum('kd,pl->pkld', imds.Fov, r1)
        Hr2 += 2.*einsum('ld,pk->pkld', imds.Fov, r1)
        #Hr2 +=   -einsum('klid,pi->pkld', 2.*imds.Wooov-imds.Wooov.transpose((1,0,2,3)), r1)
        _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Wooov)
        _expr_tmp2 = einsum_holder.to_expr_holder(imds.Wooov.transpose((1,0,2,3)))
        _expr_tmp = _expr_tmp1 - _expr_tmp2
        Hr2 +=   -einsum('klid,pi->pkld', _expr_tmp, r1)
    
    if not multiroots:
        # 2h1p-2h1p block
        if eom.partition == 'mp':
            #fock = imds.eris.fock
            #foo = fock[:nocc,:nocc]
            #fvv = fock[nocc:,nocc:]
            Hr2 +=  einsum('bd,klb->kld', fvv, r2)
            Hr2 += -einsum('ki,ild->kld', foo, r2)
            Hr2 += -einsum('lj,kjd->kld', foo, r2)
        elif eom.partition == 'full':
            raise NotImplementedError
            #diag_matrix2 = eom.vector_to_amplitudes(diag, nmo, nocc)[1]
            #Hr2 += diag_matrix2 * r2
        else:
            Hr2 +=  einsum('bd,klb->kld', imds.Lvv, r2)
            Hr2 += -einsum('ki,ild->kld', imds.Loo, r2)
            Hr2 += -einsum('lj,kjd->kld', imds.Loo, r2)
            #Hr2 +=  einsum('lbdj,kjb->kld', 2.*imds.Wovvo-imds.Wovov.transpose((0,1,3,2)), r2)
            _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Wovvo)
            _expr_tmp2 = einsum_holder.to_expr_holder(imds.Wovov.transpose((0,1,3,2)))
            _expr_tmp = _expr_tmp1 - _expr_tmp2
            Hr2 +=  einsum('lbdj,kjb->kld', _expr_tmp, r2)
            Hr2 += -einsum('kbdj,ljb->kld', imds.Wovvo, r2)
            Hr2 +=  einsum('klij,ijd->kld', imds.Woooo, r2)
            Hr2 += -einsum('kbid,ilb->kld', imds.Wovov, r2)
            tmp =   einsum('ijcb,ijb->c', imds.t2, r2)
            #Hr2 += -einsum('lkdc,c->kld', 2.*imds.Woovv-imds.Woovv.transpose((1,0,2,3)), tmp)
            _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Woovv)
            _expr_tmp2 = einsum_holder.to_expr_holder(imds.Woovv.transpose((1,0,2,3)))
            _expr_tmp = _expr_tmp1 - _expr_tmp2
            Hr2 += -einsum('lkdc,c->kld', _expr_tmp, tmp)
    else:
        # 2h1p-2h1p block
        if eom.partition == 'mp':
            #fock = imds.eris.fock
            #foo = fock[:nocc,:nocc]
            #fvv = fock[nocc:,nocc:]
            Hr2 +=  einsum('bd,pklb->pkld', fvv, r2)
            Hr2 += -einsum('ki,pild->pkld', foo, r2)
            Hr2 += -einsum('lj,pkjd->pkld', foo, r2)
        elif eom.partition == 'full':
            raise NotImplementedError
            #diag_matrix2 = eom.vector_to_amplitudes(diag, nmo, nocc)[1]
            #Hr2 += diag_matrix2 * r2
        else:
            Hr2 +=  einsum('bd,pklb->pkld', imds.Lvv, r2)
            Hr2 += -einsum('ki,pild->pkld', imds.Loo, r2)
            Hr2 += -einsum('lj,pkjd->pkld', imds.Loo, r2)
            _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Wovvo)
            _expr_tmp2 = einsum_holder.to_expr_holder(imds.Wovov.transpose((0,1,3,2)))
            _expr_tmp = _expr_tmp1 - _expr_tmp2
            Hr2 +=  einsum('lbdj,pkjb->pkld', _expr_tmp, r2)
            Hr2 += -einsum('kbdj,pljb->pkld', imds.Wovvo, r2)
            Hr2 +=  einsum('klij,pijd->pkld', imds.Woooo, r2)
            Hr2 += -einsum('kbid,pilb->pkld', imds.Wovov, r2)
            tmp =   einsum('ijcb,pijb->pc', imds.t2, r2)
            _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Woovv)
            _expr_tmp2 = einsum_holder.to_expr_holder(imds.Woovv.transpose((1,0,2,3)))
            _expr_tmp = _expr_tmp1 - _expr_tmp2
            Hr2 += -einsum('lkdc,pc->pkld', _expr_tmp, tmp)
    
    thc_scheduler.register_expr(einsum_holder.THC_scheduler.ip_hr1_l_name, Hr1)
    thc_scheduler.register_expr(einsum_holder.THC_scheduler.ip_hr2_l_name, Hr2)

def eaccsd_matvec(eom, imds=None, support_multi_root=False, diag=None, thc_scheduler:einsum_holder.THC_scheduler=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1995) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    #nmo = eom.nmo
    #nvir = nmo - nocc
    #r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)

    multiroots = eom._nroots > 1 and support_multi_root
    
    r1 = einsum_holder._expr_r1_ea(multiroots)
    r2 = einsum_holder._expr_r2_ea(multiroots)

    ###### register foo, fvv ######
    fock = imds.eris.fock
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()
    fov = fock[:nocc,nocc:].copy()
    thc_scheduler.add_input("foo", foo)
    thc_scheduler.add_input("fvv", fvv)
    thc_scheduler.add_input("fov", fov)
    foo = einsum_holder.to_expr_holder(einsum_holder._expr_foo())
    fvv = einsum_holder.to_expr_holder(einsum_holder._expr_fvv())
    fov = einsum_holder.to_expr_holder(einsum_holder._expr_fov())
    ###############################

    ###### register intermediates ######
    thc_scheduler.register_intermediates("FOV", imds.Fov)
    thc_scheduler.register_intermediates("LOO", imds.Loo)
    thc_scheduler.register_intermediates("LVV", imds.Lvv)
    ####################################

    if not multiroots:
        # Eq. (37)
        # 1p-1p block
        Hr1 =  einsum('ac,c->a', imds.Lvv, r1)
        # 1p-2p1h block
        Hr1 += 2.0*einsum('ld,lad->a', imds.Fov, r2)
        Hr1 -= einsum('ld,lda->a', imds.Fov, r2)
        #Hr1 += einsum('alcd,lcd->a', 2.*imds.Wvovv-imds.Wvovv.transpose(0,1,3,2), r2)
        _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Wvovv)
        _expr_tmp2 = einsum_holder.to_expr_holder(imds.Wvovv.transpose((0,1,3,2)))
        _expr_tmp = _expr_tmp1 - _expr_tmp2
        Hr1 += einsum('alcd,lcd->a', _expr_tmp, r2)
        # Eq. (38)
        # 2p1h-1p block
        Hr2 = einsum('abcj,c->jab', imds.Wvvvo, r1)
    else:
        # Eq. (37)
        # 1p-1p block
        Hr1 =  einsum('ac,pc->pa', imds.Lvv, r1)
        # 1p-2p1h block
        Hr1 += 2.*einsum('ld,plad->pa', imds.Fov, r2)
        Hr1 -= einsum('ld,plda->pa',    imds.Fov, r2)
        _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Wvovv)
        _expr_tmp2 = einsum_holder.to_expr_holder(imds.Wvovv.transpose((0,1,3,2)))
        _expr_tmp = _expr_tmp1 - _expr_tmp2
        Hr1 += einsum('alcd,plcd->pa', _expr_tmp, r2)
        # Eq. (38)
        # 2p1h-1p block
        Hr2 = einsum('abcj,pc->pjab', imds.Wvvvo, r1)
    
    if not multiroots:
        # 2p1h-2p1h block
        if eom.partition == 'mp':
            #fock = imds.eris.fock
            #foo = fock[:nocc,:nocc]
            #fvv = fock[nocc:,nocc:]
            Hr2 +=  einsum('ac,jcb->jab', fvv, r2)
            Hr2 +=  einsum('bd,jad->jab', fvv, r2)
            Hr2 += -einsum('lj,lab->jab', foo, r2)
        elif eom.partition == 'full':
            #diag_matrix2 = eom.vector_to_amplitudes(diag, nmo, nocc)[1]
            #Hr2 += diag_matrix2 * r2
            raise NotImplementedError
        else:
            Hr2 +=  einsum('ac,jcb->jab', imds.Lvv, r2)
            Hr2 +=  einsum('bd,jad->jab', imds.Lvv, r2)
            Hr2 += -einsum('lj,lab->jab', imds.Loo, r2)
            #Hr2 += einsum('lbdj,lad->jab', 2.*imds.Wovvo-imds.Wovov.transpose(0,1,3,2), r2)
            _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Wovvo)
            _expr_tmp2 = einsum_holder.to_expr_holder(imds.Wovov.transpose((0,1,3,2)))
            _expr_tmp = _expr_tmp1 - _expr_tmp2
            Hr2 += einsum('lbdj,lad->jab', _expr_tmp, r2)
            Hr2 += -einsum('lajc,lcb->jab', imds.Wovov, r2)
            Hr2 += -einsum('lbcj,lca->jab', imds.Wovvo, r2)
            Hr2 += einsum('abcd,jcd->jab', imds.Wvvvv, r2)
            #tmp = np.einsum('klcd,lcd->k', 2.*imds.Woovv-imds.Woovv.transpose(0,1,3,2), r2)
            _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Woovv)
            _expr_tmp2 = einsum_holder.to_expr_holder(imds.Woovv.transpose((0,1,3,2)))
            _expr_tmp = _expr_tmp1 - _expr_tmp2
            tmp = einsum('klcd,lcd->k', _expr_tmp, r2)
            Hr2 += -einsum('k,kjab->jab', tmp, imds.t2)
    else:
        # 2p1h-2p1h block
        if eom.partition == 'mp':
            #fock = imds.eris.fock
            #foo = fock[:nocc,:nocc]
            #fvv = fock[nocc:,nocc:]
            Hr2 +=  einsum('ac,pjcb->pjab', fvv, r2)
            Hr2 +=  einsum('bd,pjad->pjab', fvv, r2)
            Hr2 += -einsum('lj,plab->pjab', foo, r2)
        elif eom.partition == 'full':
            #diag_matrix2 = eom.vector_to_amplitudes(diag, nmo, nocc)[1]
            #Hr2 += diag_matrix2 * r2
            raise NotImplementedError
        else:
            Hr2 += einsum('ac,pjcb->pjab', imds.Lvv, r2)
            Hr2 += einsum('bd,pjad->pjab', imds.Lvv, r2)
            Hr2 += -einsum('lj,plab->pjab', imds.Loo, r2)
            #Hr2 += einsum('lbdj,plad->pjab', 2.*imds.Wovvo-imds.Wovov.transpose(0,1,3,2), r2)
            _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Wovvo)
            _expr_tmp2 = einsum_holder.to_expr_holder(imds.Wovov.transpose((0,1,3,2)))
            _expr_tmp = _expr_tmp1 - _expr_tmp2
            Hr2 += einsum('lbdj,plad->pjab', _expr_tmp, r2)
            Hr2 += -einsum('lajc,plcb->pjab', imds.Wovov, r2)
            Hr2 += -einsum('lbcj,plca->pjab', imds.Wovvo, r2)
            Hr2 += einsum('abcd,pjcd->pjab', imds.Wvvvv, r2)
            #tmp = np.einsum('klcd,pkld->pc', 2.*imds.Woovv-imds.Woovv.transpose(0,1,3,2), r2)
            _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Woovv)
            _expr_tmp2 = einsum_holder.to_expr_holder(imds.Woovv.transpose((0,1,3,2)))
            _expr_tmp = _expr_tmp1 - _expr_tmp2
            tmp = einsum('klcd,plcd->pk', _expr_tmp, r2)
            Hr2+= -einsum('pk,kjab->pjab', tmp, imds.t2)
            
    thc_scheduler.register_expr(einsum_holder.THC_scheduler.ea_hr1_r_name, Hr1)
    thc_scheduler.register_expr(einsum_holder.THC_scheduler.ea_hr2_r_name, Hr2)

def leaccsd_matvec(eom, imds=None, support_multi_root=False, diag=None, thc_scheduler:einsum_holder.THC_scheduler=None):
    # Note this is not the same left EA equations used by Nooijen and Bartlett.
    # Small changes were made so that the same type L2 basis was used for both the
    # left EA and left IP equations.  You will note more similarity for these
    # equations to the left IP equations than for the left EA equations by Nooijen.
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    #nmo = eom.nmo
    #nvir = nmo - nocc
    #r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)

    multiroots = eom._nroots > 1 and support_multi_root
    
    r1 = einsum_holder._expr_r1_ea(multiroots)
    r2 = einsum_holder._expr_r2_ea(multiroots)

    ###### register foo, fvv ######
    fock = imds.eris.fock
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()
    fov = fock[:nocc,nocc:].copy()
    thc_scheduler.add_input("foo", foo)
    thc_scheduler.add_input("fvv", fvv)
    thc_scheduler.add_input("fov", fov)
    foo = einsum_holder.to_expr_holder(einsum_holder._expr_foo())
    fvv = einsum_holder.to_expr_holder(einsum_holder._expr_fvv())
    fov = einsum_holder.to_expr_holder(einsum_holder._expr_fov())
    ###############################

    ###### register intermediates ######
    thc_scheduler.register_intermediates("FOV", imds.Fov)
    thc_scheduler.register_intermediates("LOO", imds.Loo)
    thc_scheduler.register_intermediates("LVV", imds.Lvv)
    ####################################
    
    if not multiroots:
        # Eq. (30)
        # 1p-1p block
        Hr1 = einsum('ac,a->c', imds.Lvv, r1)
        # 1p-2p1h block
        Hr1 += einsum('abcj,jab->c', imds.Wvvvo, r2)
        # Eq. (31)
        # 2p1h-1p block
        Hr2 = 2.*einsum('c,ld->lcd', r1, imds.Fov)
        Hr2 +=  -einsum('d,lc->lcd', r1, imds.Fov)
        #Hr2 += einsum('a,alcd->lcd', r1, 2.*imds.Wvovv-imds.Wvovv.transpose(0,1,3,2))
        _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Wvovv)
        _expr_tmp2 = einsum_holder.to_expr_holder(imds.Wvovv.transpose((0,1,3,2)))
        _expr_tmp = _expr_tmp1 - _expr_tmp2
        Hr2 += einsum('a,alcd->lcd', r1, _expr_tmp)
    else:
        # Eq. (30)
        # 1p-1p block
        Hr1 = einsum('ac,pa->pc', imds.Lvv, r1)
        # 1p-2p1h block
        Hr1 += einsum('abcj,pjab->pc', imds.Wvvvo, r2)
        # Eq. (31)
        # 2p1h-1p block
        Hr2 = 2.*einsum('pc,ld->plcd', r1, imds.Fov)
        Hr2 +=  -einsum('pd,lc->plcd', r1, imds.Fov)
        _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Wvovv)
        _expr_tmp2 = einsum_holder.to_expr_holder(imds.Wvovv.transpose((0,1,3,2)))
        _expr_tmp = _expr_tmp1 - _expr_tmp2
        Hr2 += einsum('pa,alcd->plcd', r1, _expr_tmp)
    
    # 2p1h-2p1h block
    
    if not multiroots:
        if eom.partition == 'mp':
            #fock = imds.eris.fock
            #foo = fock[:nocc,:nocc]
            #fvv = fock[nocc:,nocc:]
            Hr2 += einsum('lad,ac->lcd', r2, fvv)
            Hr2 += einsum('lcb,bd->lcd', r2, fvv)
            Hr2 += -einsum('jcd,lj->lcd', r2, foo)
        elif eom.partition == 'full':
            #diag_matrix2 = eom.vector_to_amplitudes(diag, nmo, nocc)[1]
            #Hr2 += diag_matrix2 * r2
            raise NotImplementedError
        else:
            Hr2 += einsum('lad,ac->lcd', r2, imds.Lvv)
            Hr2 += einsum('lcb,bd->lcd', r2, imds.Lvv)
            Hr2 += -einsum('jcd,lj->lcd', r2, imds.Loo)
            #Hr2 += einsum('jcb,lbdj->lcd', r2, 2.*imds.Wovvo-imds.Wovov.transpose(0,1,3,2))
            _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Wovvo)
            _expr_tmp2 = einsum_holder.to_expr_holder(imds.Wovov.transpose((0,1,3,2)))
            _expr_tmp = _expr_tmp1 - _expr_tmp2
            Hr2 += einsum('jcb,lbdj->lcd', r2, _expr_tmp)
            Hr2 += -einsum('lajc,jab->lcb', imds.Wovov, r2)
            Hr2 += -einsum('lbcj,jab->lca', imds.Wovvo, r2)
            #for a in range(nvir):
            #    Hr2 += einsum('lb,bcd->lcd', r2[:,a,:], imds.Wvvvv[a])
            Hr2 += einsum('lab,abcd->lcd', r2, imds.Wvvvv)
            tmp = einsum('ijcb,ibc->j', imds.t2, r2)
            _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Woovv)
            _expr_tmp2 = einsum_holder.to_expr_holder(imds.Woovv.transpose((0,1,3,2)))
            _expr_tmp = _expr_tmp1 - _expr_tmp2
            Hr2 += -einsum('kjfe,j->kef',_expr_tmp,tmp)
    else:
        if eom.partition == 'mp':
            #fock = imds.eris.fock
            #foo = fock[:nocc,:nocc]
            #fvv = fock[nocc:,nocc:]
            Hr2 += einsum('plad,ac->plcd', r2, fvv)
            Hr2 += einsum('plcb,bd->plcd', r2, fvv)
            Hr2 += -einsum('pjcd,lj->plcd', r2, foo)
        elif eom.partition == 'full':
            #diag_matrix2 = eom.vector_to_amplitudes(diag, nmo, nocc)[1]
            #Hr2 += diag_matrix2 * r2
            raise NotImplementedError
        else:
            Hr2 += einsum('plad,ac->plcd', r2, imds.Lvv)
            Hr2 += einsum('plcb,bd->plcd', r2, imds.Lvv)
            Hr2 += -einsum('pjcd,lj->plcd', r2, imds.Loo)
            _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Wovvo)
            _expr_tmp2 = einsum_holder.to_expr_holder(imds.Wovov.transpose((0,1,3,2)))
            _expr_tmp = _expr_tmp1 - _expr_tmp2
            Hr2 += einsum('pjcb,lbdj->plcd', r2, _expr_tmp)
            Hr2 += -einsum('lajc,pjab->plcb', imds.Wovov, r2)
            Hr2 += -einsum('lbcj,pjab->plca', imds.Wovvo, r2)
            Hr2 += einsum('plab,abcd->plcd', r2, imds.Wvvvv)
            tmp = einsum('ijcb,pibc->pj', imds.t2, r2)
            _expr_tmp1 = einsum_holder.to_expr_holder(2.*imds.Woovv)
            _expr_tmp2 = einsum_holder.to_expr_holder(imds.Woovv.transpose((0,1,3,2)))
            _expr_tmp = _expr_tmp1 - _expr_tmp2
            Hr2 += -einsum('kjfe,pj->pkef',_expr_tmp,tmp)

    thc_scheduler.register_expr(einsum_holder.THC_scheduler.ea_hr1_l_name, Hr1)
    thc_scheduler.register_expr(einsum_holder.THC_scheduler.ea_hr2_l_name, Hr2)


################## used only for testing ##################

from pyscf.pbc.df.isdf.thc_cc_helper._thc_rccsd import _fake_eris, _fake_eris_full

class _fake_cc:
    def __init__(self, nocc, nvir, cc2=False):
        self.nocc = nocc
        self.nvir = nvir
        self.level_shift = 0.1
        self.cc2 = cc2
        self.verbose = 10
        import sys
        self.stdout = sys.stdout
        
        self._t1_expr = einsum_holder._expr_t1()
        self._t2_expr = einsum_holder._expr_ccsd_t2()
        self._t2_expr = self._t2_expr.transpose((0,2,1,3))

class _fake_eom_ip:
    def __init__(self, cc, partition=None, nroots=1, eris=None):
        self.cc = cc
        self.partition = partition
        self._nroots = nroots
        self.verbose = 10
        import sys
        self.stdout = sys.stdout
        self.eris = eris
        nocc = eris.nocc
        nvir = eris.nvir
        self.nocc = nocc
        self.nvir = nvir
        self.nmo = nocc + nvir
        import numpy as np
        if nroots == 1:
            self.r1 = np.random.rand(nocc) * 0.5
            self.r2 = np.random.rand(nocc, nocc, nvir) * 0.5
        else:
            self.r1 = np.random.rand(nroots, nocc) * 0.5
            self.r2 = np.random.rand(nroots, nocc, nocc, nvir) * 0.5
    
    from pyscf.cc.eom_rccsd import vector_to_amplitudes_ip, amplitudes_to_vector_ip
    from pyscf.lib import logger, module_method
    amplitudes_to_vector = staticmethod(amplitudes_to_vector_ip)
    vector_to_amplitudes = module_method(vector_to_amplitudes_ip,
                                         absences=['nmo', 'nocc'])
    
    def make_imds(self):
        return _IMDS_symbolic(self.cc, eris=self.eris, MRPT2=False)

class _fake_eom_ea:
    def __init__(self, cc, partition=None, nroots=1, eris=None):
        self.cc = cc
        self.partition = partition
        self._nroots = nroots
        self.verbose = 10
        import sys
        self.stdout = sys.stdout
        self.eris = eris
        nocc = eris.nocc
        nvir = eris.nvir
        self.nocc = nocc
        self.nvir = nvir
        self.nmo = nocc + nvir
        import numpy as np
        if nroots == 1:
            self.r1 = np.random.rand(nvir) * 0.5
            self.r2 = np.random.rand(nocc, nvir, nvir) * 0.5
        else:
            self.r1 = np.random.rand(nroots, nvir) * 0.5
            self.r2 = np.random.rand(nroots, nocc, nvir, nvir) * 0.5
    
    from pyscf.cc.eom_rccsd import vector_to_amplitudes_ea, amplitudes_to_vector_ea
    from pyscf.lib import logger, module_method
    amplitudes_to_vector = staticmethod(amplitudes_to_vector_ea)
    vector_to_amplitudes = module_method(vector_to_amplitudes_ea,
                                         absences=['nmo', 'nocc'])
    
    def make_imds(self):
        return _IMDS_symbolic(self.cc, eris=self.eris, MRPT2=False)

class _fake_eom_imds:
    def __init__(self):
        pass

if __name__ == "__main__":
    
    ### generate random input ### 
    
    nocc = 4
    nvir = 5
    nthc = 17
    nlaplace = 7
    
    import numpy as np
    
    Xo    = np.random.rand(nocc, nthc) * 0.5
    Xv    = np.random.rand(nvir, nthc) * 0.5
    Tau_o = np.random.rand(nocc, nlaplace) * 0.5
    Tau_v = np.random.rand(nvir, nlaplace) * 0.5
    THC_INT = np.random.rand(nthc, nthc) * 0.5
    THC_INT+= THC_INT.T
    Xo_T2 = np.random.rand(nocc, nthc) * 0.5
    Xv_T2 = np.random.rand(nvir, nthc) * 0.5
    PROJ  = np.random.rand(nthc, nthc) * 0.5
    PROJ += PROJ.T
        
    THC_T2  = np.random.rand(nthc, nthc) * 0.5
    THC_T2 += THC_T2.T
    T1      = np.random.rand(nocc, nvir) * 0.5
    
    ################### test IP ################
    
    scheduler = einsum_holder.THC_scheduler(
        X_O=Xo,
        X_V=Xv,
        TAU_O=Tau_o,
        TAU_V=Tau_v,
        THC_INT=THC_INT,
        T1=T1,
        THC_T2=THC_T2,
        XO_T2=Xo_T2,
        XV_T2=Xv_T2,
        PROJECTOR=PROJ,
        use_torch=False,
        with_gpu=False
    )
    
    eris = _fake_eris(nocc, nvir)
    eris_full = _fake_eris_full(eris.fock, Xo, Xv, THC_INT)
    cc = _fake_cc(nocc, nvir)
    cc.t1 = T1
    cc.t2 = THC_T2
    NROOTS = 3
    eom = _fake_eom_ip(cc,eris=eris, nroots=NROOTS)
    
    imds = eom.make_imds()
    imds.make_ip()
    imds.make_ea()
    
    ipccsd_matvec(eom, imds, thc_scheduler=scheduler)
    lipccsd_matvec(eom, imds, thc_scheduler=scheduler)
    scheduler.register_expr("Wooov_test", imds.Wooov)
    scheduler.register_expr("Wvovv_test", imds.Wvovv)
    scheduler.register_expr("Wovvo_test", imds.Wovvo)
    scheduler.register_expr("Wovov_test", imds.Wovov)
    scheduler.register_expr("Woooo_test", imds.Woooo)
    scheduler.register_expr("Wvvvv_test", imds.Wvvvv)
    scheduler.register_expr("Wvvvo_test", imds.Wvvvo)
    scheduler.register_expr("Wovoo_test", imds.Wovoo)
    scheduler._build_expression()
    print(scheduler)
    scheduler.update_r1(eom.r1)
    scheduler.update_r2(eom.r2)
    scheduler._build_contraction(backend="opt_einsum")
    scheduler.evaluate_all()
    
    ########## judge the intermediates terms by terms ##########
    
    t1 = T1
    t2_full = np.einsum("iP,aP,PQ,jQ,bQ,iT,jT,aT,bT->ijab", Xo_T2, Xv_T2, THC_T2, Xo_T2, Xv_T2, Tau_o, Tau_o, Tau_v, Tau_v, optimize=True)
    
    import pyscf.cc.rintermediates as rind
    
    Wooov_bench = rind.Wooov(t1, t2_full, eris_full)
    Wooov_test = scheduler.get_tensor("Wooov_test")
    print("diff Wooov = ", np.linalg.norm(Wooov_bench - Wooov_test))
    assert np.allclose(Wooov_bench, Wooov_test)

    Wvovv_bench = rind.Wvovv(t1, t2_full, eris_full)
    Wvovv_test = scheduler.get_tensor("Wvovv_test")
    print("diff Wvovv = ", np.linalg.norm(Wvovv_bench - Wvovv_test))
    assert np.allclose(Wvovv_bench, Wvovv_test)
    
    Wovvo_bench = rind.Wovvo(t1, t2_full, eris_full)
    Wovvo_test = scheduler.get_tensor("Wovvo_test")
    print("diff Wovvo = ", np.linalg.norm(Wovvo_bench - Wovvo_test))
    assert np.allclose(Wovvo_bench, Wovvo_test)
    
    Wovov_bench = rind.Wovov(t1, t2_full, eris_full)
    Wovov_test = scheduler.get_tensor("Wovov_test")
    print("diff Wovov = ", np.linalg.norm(Wovov_bench - Wovov_test))
    assert np.allclose(Wovov_bench, Wovov_test)
    
    Woooo_bench = rind.Woooo(t1, t2_full, eris_full)
    Woooo_test = scheduler.get_tensor("Woooo_test")
    print("diff Woooo = ", np.linalg.norm(Woooo_bench - Woooo_test))
    assert np.allclose(Woooo_bench, Woooo_test)
    
    Wvvvv_bench = rind.Wvvvv(t1, t2_full, eris_full)
    Wvvvv_test = scheduler.get_tensor("Wvvvv_test")
    #print("Wvvvv[0]", Wvvvv_bench[0,0,:,:])
    #print("Wvvvv[0]", Wvvvv_test[0,0,:,:])
    print("diff Wvvvv = ", np.max(np.abs((Wvvvv_bench - Wvvvv_test))))
    assert np.allclose(Wvvvv_bench, Wvvvv_test)
    
    Wvvvo_bench = rind.Wvvvo(t1, t2_full, eris_full)
    Wvvvo_test = scheduler.get_tensor("Wvvvo_test")
    #print("Wvvvo[0]", Wvvvo_bench[0,0,:,:])
    #print("Wvvvo[0]", Wvvvo_test[0,0,:,:])
    #print("diff Wvvvo = ", np.linalg.norm(Wvvvo_bench - Wvvvo_test))
    print("diff Wvvvo = ", np.max(np.abs((Wvvvo_bench - Wvvvo_test))))
    assert np.allclose(Wvvvo_bench, Wvvvo_test)
    
    Wovoo_bench = rind.Wovoo(t1, t2_full, eris_full)
    Wovoo_test = scheduler.get_tensor("Wovoo_test")
    print("diff Wovoo = ", np.linalg.norm(Wovoo_bench - Wovoo_test))
    assert np.allclose(Wovoo_bench, Wovoo_test)
    
    
    eom_imds = _fake_eom_imds()
    
    Loo = rind.Loo(t1, t2_full, eris_full)
    Loo_test = scheduler.get_tensor("LOO")
    print("diff Loo = ", np.linalg.norm(Loo - Loo_test))
    assert np.allclose(Loo, Loo_test)
    
    Fov = rind.cc_Fov(t1, t2_full, eris_full)
    Fov_test = scheduler.get_tensor("FOV")
    print("diff Fov = ", np.linalg.norm(Fov - Fov_test))
    assert np.allclose(Fov, Fov_test)
    
    Lvv = rind.Lvv(t1, t2_full, eris_full)
    Lvv_test = scheduler.get_tensor("LVV")
    print("diff Lvv = ", np.linalg.norm(Lvv - Lvv_test))
    assert np.allclose(Lvv, Lvv_test)
    
    eom_imds.Fov = Fov
    eom_imds.Loo = Loo
    eom_imds.Lvv = Lvv
    eom_imds.Wooov = Wooov_test
    eom_imds.Wvovv = Wvovv_test
    eom_imds.Wovvo = Wovvo_test
    eom_imds.Wovov = Wovov_test
    eom_imds.Woooo = Woooo_test
    eom_imds.Wvvvv = Wvvvv_test
    eom_imds.Wvvvo = Wvvvo_test
    eom_imds.Wovoo = Wovoo_test
    eom_imds.Woovv = eris_full.ovov.transpose((0,2,1,3))
    eom_imds.t2    = t2_full
    
    from pyscf.cc.eom_rccsd import vector_to_amplitudes_ip, amplitudes_to_vector_ip
    import pyscf.cc.eom_rccsd as eom_rccsd

    r1_test = scheduler.get_tensor(einsum_holder.THC_scheduler.ip_hr1_r_name)
    r2_test = scheduler.get_tensor(einsum_holder.THC_scheduler.ip_hr2_r_name)
    l1_test = scheduler.get_tensor(einsum_holder.THC_scheduler.ip_hr1_l_name)
    l2_test = scheduler.get_tensor(einsum_holder.THC_scheduler.ip_hr2_l_name)

    ########### test IP R ###########

    for i in range(NROOTS):
        if NROOTS == 1:
            vector         = amplitudes_to_vector_ip(eom.r1, eom.r2)
            vector_hr      = eom_rccsd.ipccsd_matvec(eom, vector, eom_imds)
            vector_hr_test = amplitudes_to_vector_ip(r1_test, r2_test)
            print("diff vector_hr = ", np.linalg.norm(vector_hr - vector_hr_test))
            assert np.allclose(vector_hr, vector_hr_test)
        else:
            vector         = amplitudes_to_vector_ip(eom.r1[i], eom.r2[i])
            vector_hr      = eom_rccsd.ipccsd_matvec(eom, vector, eom_imds)
            vector_hr_test = amplitudes_to_vector_ip(r1_test[i], r2_test[i])
            print("diff vector_hr = ", np.linalg.norm(vector_hr - vector_hr_test))
            assert np.allclose(vector_hr, vector_hr_test)
    
    ########### test IP L ###########
    
    for i in range(NROOTS):
        if NROOTS == 1:
            vector         = amplitudes_to_vector_ip(eom.r1, eom.r2)
            vector_hl      = eom_rccsd.lipccsd_matvec(eom, vector, eom_imds)
            vector_hl_test = amplitudes_to_vector_ip(l1_test, l2_test)
            print("diff vector_hl = ", np.linalg.norm(vector_hl - vector_hl_test))
            assert np.allclose(vector_hl, vector_hl_test)
        else:
            vector         = amplitudes_to_vector_ip(eom.r1[i], eom.r2[i])
            vector_hl      = eom_rccsd.lipccsd_matvec(eom, vector, eom_imds)
            vector_hl_test = amplitudes_to_vector_ip(l1_test[i], l2_test[i])
            print("diff vector_hl = ", np.linalg.norm(vector_hl - vector_hl_test))
            assert np.allclose(vector_hl, vector_hl_test)
    
    
    ################### test EA ################
    
    scheduler = einsum_holder.THC_scheduler(
        X_O=Xo,
        X_V=Xv,
        TAU_O=Tau_o,
        TAU_V=Tau_v,
        THC_INT=THC_INT,
        T1=T1,
        THC_T2=THC_T2,
        XO_T2=Xo_T2,
        XV_T2=Xv_T2,
        PROJECTOR=PROJ,
        use_torch=False,
        with_gpu=False
    )
    
    eom = _fake_eom_ea(cc,eris=eris, nroots=NROOTS)
    
    imds = eom.make_imds()
    imds.make_ip()
    imds.make_ea()
    
    eaccsd_matvec(eom, imds, thc_scheduler=scheduler)
    leaccsd_matvec(eom, imds, thc_scheduler=scheduler)
    scheduler._build_expression()
    print(scheduler)
    scheduler.update_r1(eom.r1)
    scheduler.update_r2(eom.r2)
    scheduler._build_contraction(backend="opt_einsum")
    scheduler.evaluate_all()
    
    from pyscf.cc.eom_rccsd import vector_to_amplitudes_ea, amplitudes_to_vector_ea
    import pyscf.cc.eom_rccsd as eom_rccsd
    
    r1_test = scheduler.get_tensor(einsum_holder.THC_scheduler.ea_hr1_r_name)
    r2_test = scheduler.get_tensor(einsum_holder.THC_scheduler.ea_hr2_r_name)
    l1_test = scheduler.get_tensor(einsum_holder.THC_scheduler.ea_hr1_l_name)
    l2_test = scheduler.get_tensor(einsum_holder.THC_scheduler.ea_hr2_l_name) 
    
    Fov = rind.cc_Fov(t1, t2_full, eris_full)
    Fov_test = scheduler.get_tensor("FOV")
    print("diff Fov = ", np.linalg.norm(Fov - Fov_test))
    assert np.allclose(Fov, Fov_test)
    
    eom_imds.Fov = Fov
    
    ########### test EA R ###########

    for i in range(NROOTS):
        if NROOTS == 1:
            vector         = amplitudes_to_vector_ea(eom.r1, eom.r2)
            vector_hr      = eom_rccsd.eaccsd_matvec(eom, vector, eom_imds)
            vector_hr_test = amplitudes_to_vector_ea(r1_test, r2_test)
            print("diff vector_hr = ", np.linalg.norm(vector_hr - vector_hr_test))
            assert np.allclose(vector_hr, vector_hr_test)
        else:
            vector         = amplitudes_to_vector_ea(eom.r1[i], eom.r2[i])
            vector_hr      = eom_rccsd.eaccsd_matvec(eom, vector, eom_imds)
            vector_hr_test = amplitudes_to_vector_ea(r1_test[i], r2_test[i])
            print("diff vector_hr = ", np.linalg.norm(vector_hr - vector_hr_test))
            assert np.allclose(vector_hr, vector_hr_test)
    
    ########### test EA L ###########
    
    # exit(1)
    
    for i in range(NROOTS):
        if NROOTS == 1:
            vector         = amplitudes_to_vector_ea(eom.r1, eom.r2)
            vector_hl      = eom_rccsd.leaccsd_matvec(eom, vector, eom_imds)
            vector_hl_test = amplitudes_to_vector_ea(l1_test, l2_test)
            print("diff vector_hl = ", np.linalg.norm(vector_hl - vector_hl_test))
            assert np.allclose(vector_hl, vector_hl_test)
        else:
            vector         = amplitudes_to_vector_ea(eom.r1[i], eom.r2[i])
            vector_hl      = eom_rccsd.leaccsd_matvec(eom, vector, eom_imds)
            vector_hl_test = amplitudes_to_vector_ea(l1_test[i], l2_test[i])
            print("diff vector_hl = ", np.linalg.norm(vector_hl - vector_hl_test))
            assert np.allclose(vector_hl, vector_hl_test)
    