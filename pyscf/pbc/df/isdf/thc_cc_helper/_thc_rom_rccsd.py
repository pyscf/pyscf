

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
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        if MRPT2: # J. Chem. Phys. 102 (4), 22 January 1995
            self.t1 = None
            self.t2 = einsum_holder._expr_mp2_t2()
            self.MRPT2_approx = True
        else:
            self.t1 = einsum_holder._expr_t1()
            self.t2 = einsum_holder._expr_t2()
            self.MRPT2_approx = False
        self.eris = eris
        self._made_shared_2e = False
    
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
        self.Woovv = asarray(eris.ovov).transpose(0,2,1,3)

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

def ipccsd_matvec(eom, imds=None, diag=None, thc_scheduler:einsum_holder.THC_scheduler=None):
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    #nocc = eom.nocc
    #nmo = eom.nmo
    #r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)
    
    r1 = einsum_holder._expr_r1_ip()
    r2 = einsum_holder._expr_r2_ip()

    ###### register foo, fvv ######
    fock = imds.eris.fock
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()
    thc_scheduler.add_input("foo", foo)
    thc_scheduler.add_input("fvv", fvv)
    foo = einsum_holder._expr_foo()
    fvv = einsum_holder._expr_fvv()
    ###############################

    ###### register intermediates ######
    thc_scheduler.register_intermediates("FOV", imds.Fov)
    thc_scheduler.register_intermediates("LOO", imds.Loo)
    thc_scheduler.register_intermediates("LVV", imds.Lvv)
    ####################################

    # 1h-1h block
    Hr1 = -einsum('ki,k->i', imds.Loo, r1)
    #1h-2h1p block
    Hr1 += 2*einsum('ld,ild->i', imds.Fov, r2)
    Hr1 +=  -einsum('kd,kid->i', imds.Fov, r2)
    Hr1 +=-2*einsum('klid,kld->i', imds.Wooov, r2)
    Hr1 +=   einsum('lkid,kld->i', imds.Wooov, r2)

    # 2h1p-1h block
    Hr2 = -einsum('kbij,k->ijb', imds.Wovoo, r1)
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

    #vector = eom.amplitudes_to_vector(Hr1, Hr2)
    #return vector
    
    thc_scheduler.register_expr(einsum_holder.THC_scheduler.ip_hr1_r_name, Hr1)
    thc_scheduler.register_expr(einsum_holder.THC_scheduler.ip_hr2_r_name, Hr2)

def lipccsd_matvec(eom, imds=None, diag=None, thc_scheduler:einsum_holder.THC_scheduler=None):
    '''For left eigenvector'''
    # Note this is not the same left EA equations used by Nooijen and Bartlett.
    # Small changes were made so that the same type L2 basis was used for both the
    # left EA and left IP equations.  You will note more similarity for these
    # equations to the left IP equations than for the left EA equations by Nooijen.
    if imds is None: imds = eom.make_imds()
    #nocc = eom.nocc
    #nmo = eom.nmo
    #r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)

    r1 = einsum_holder._expr_r1_ip()
    r2 = einsum_holder._expr_r2_ip()
    
    ###### register foo, fvv ######
    fock = imds.eris.fock
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nocc:,nocc:].copy()
    thc_scheduler.add_input("foo", foo)
    thc_scheduler.add_input("fvv", fvv)
    foo = einsum_holder._expr_foo()
    fvv = einsum_holder._expr_fvv()
    ###############################

    ###### register intermediates ######
    thc_scheduler.register_intermediates("FOV", imds.Fov)
    thc_scheduler.register_intermediates("LOO", imds.Loo)
    thc_scheduler.register_intermediates("LVV", imds.Lvv)
    ####################################
    
    # 1h-1h block
    Hr1 =  -einsum('ki,i->k', imds.Loo, r1)
    # 1h-2h1p block
    Hr1 += -einsum('kbij,ijb->k', imds.Wovoo, r2)

    # 2h1p-1h block
    Hr2  =   -einsum('kd,l->kld', imds.Fov, r1)
    Hr2 += 2.*einsum('ld,k->kld', imds.Fov, r1)
    Hr2 +=   -einsum('klid,i->kld', 2.*imds.Wooov-imds.Wooov.transpose(1,0,2,3), r1)
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
        Hr2 +=  einsum('lbdj,kjb->kld', 2.*imds.Wovvo-imds.Wovov.transpose((0,1,3,2)), r2)
        Hr2 += -einsum('kbdj,ljb->kld', imds.Wovvo, r2)
        Hr2 +=  einsum('klij,ijd->kld', imds.Woooo, r2)
        Hr2 += -einsum('kbid,ilb->kld', imds.Wovov, r2)
        tmp =   einsum('ijcb,ijb->c', imds.t2, r2)
        Hr2 += -einsum('lkdc,c->kld', 2.*imds.Woovv-imds.Woovv.transpose((1,0,2,3)), tmp)

    #vector = eom.amplitudes_to_vector(Hr1, Hr2)
    #return vector
    
    thc_scheduler.register_expr(einsum_holder.THC_scheduler.ip_hr1_l_name, Hr1)
    thc_scheduler.register_expr(einsum_holder.THC_scheduler.ip_hr2_l_name, Hr2)