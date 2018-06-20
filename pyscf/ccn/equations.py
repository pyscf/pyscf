#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
# Author: Artem Pulkin
#

from .util import e, p


def eq_gs_s(oo, ov, vo, vv, oovo, oovv, ovvo, ovvv, t1):
    hE = 0
    hE += e("ba,ia->ib", vv, t1)  # d0_ov
    hE -= e("ij,ia->ja", oo, t1)  # d1_ov
    hE += e("ai->ia", vo)  # d2_ov
    hE -= e("ia,ib,ja->jb", ov, t1, t1)  # d3_ov
    hE -= e("jacb,jb,ic->ia", ovvv, t1, t1)  # d4_ov
    hE += e("jabi,jb->ia", ovvo, t1)  # d5_ov
    hE -= e("jkbc,jb,ka,ic->ia", oovv, t1, t1, t1)  # d6_ov
    hE -= e("jiak,ja,ib->kb", oovo, t1, t1)  # d7_ov
    return hE


def energy_gs_s(ov, oovv, t1):
    scalar = 0
    scalar += e("ia,ia", ov, t1)  # s0
    scalar += 1./2 * e("jiba,ia,jb", oovv, t1, t1)  # s1
    return scalar


def eq_gs_sd(oo, ov, vo, vv, oooo, oovo, oovv, ovoo, ovvo, ovvv, vvoo, vvvo, vvvv, t1, t2):
    hE = hhEE = 0
    hE += e("ba,ia->ib", vv, t1)  # d0_ov
    hhEE += p("..ab", e("ba,ijac->ijbc", vv, t2))  # d1_oovv
    hE -= e("ij,ia->ja", oo, t1)  # d2_ov
    hhEE -= p("ab..", e("ik,ijba->kjba", oo, t2))  # d3_oovv
    hE += e("ai->ia", vo)  # d4_ov
    hE -= e("jb,ja,ib->ia", ov, t1, t1)  # d5_ov
    hhEE -= p("ab..", e("ic,kc,ijba->kjba", ov, t1, t2))  # d6_oovv
    hhEE -= p("..ab", e("ia,ib,jkac->jkbc", ov, t1, t2))  # d7_oovv
    hE += e("ia,ijab->jb", ov, t2)  # d8_ov
    hhEE += 1./2 * p("ab..", e("cbad,ja,id->jicb", vvvv, t1, t1))  # d9_oovv
    hhEE += 1./2 * e("dcba,ijba->ijdc", vvvv, t2)  # d10_oovv
    hhEE -= p("ab..", e("baci,jc->ijba", vvvo, t1))  # d11_oovv
    hhEE += e("baij->ijba", vvoo)  # d12_oovv
    hhEE -= 1./2 * p("ab..", p("..ab", e("icdb,ia,jb,kd->kjac", ovvv, t1, t1, t1)))  # d13_oovv
    hE += e("jbca,ia,jc->ib", ovvv, t1, t1)  # d14_ov
    hhEE += p("ab..", p("..ab", e("kacd,jd,kicb->ijba", ovvv, t1, t2)))  # d15_oovv
    hhEE += 1./2 * p("..ab", e("kcba,kd,ijba->ijcd", ovvv, t1, t2))  # d16_oovv
    hhEE -= p("..ab", e("kadc,kc,ijbd->ijba", ovvv, t1, t2))  # d17_oovv
    hE += 1./2 * e("icba,ijba->jc", ovvv, t2)  # d18_ov
    hhEE += p("ab..", p("..ab", e("kcaj,ia,kb->ijcb", ovvo, t1, t1)))  # d19_oovv
    hE += e("jabi,jb->ia", ovvo, t1)  # d20_ov
    hhEE += p("ab..", p("..ab", e("ibaj,ikac->kjcb", ovvo, t2)))  # d21_oovv
    hhEE += p("..ab", e("kbij,ka->ijba", ovoo, t1))  # d22_oovv
    hhEE += 1./4 * p("ab..", p("..ab", e("jlca,ia,jb,kc,ld->kibd", oovv, t1, t1, t1, t1)))  # d23_oovv
    hhEE -= 1./4 * p("ab..", e("jkda,ia,ld,jkcb->ilcb", oovv, t1, t1, t2))  # d24_oovv
    hE += e("jkca,ia,jb,kc->ib", oovv, t1, t1, t1)  # d25_ov
    hhEE -= p("ab..", p("..ab", e("kldc,jc,la,kidb->ijba", oovv, t1, t1, t2)))  # d26_oovv
    hhEE += p("ab..", e("licd,lc,kd,ijba->jkba", oovv, t1, t1, t2))  # d27_oovv
    hE -= 1./2 * e("jkbc,ic,jkba->ia", oovv, t1, t2)  # d28_ov
    hhEE -= 1./4 * p("..ab", e("klba,kc,ld,ijba->ijdc", oovv, t1, t1, t2))  # d29_oovv
    hhEE += p("..ab", e("lkcd,ka,lc,ijdb->ijba", oovv, t1, t1, t2))  # d30_oovv
    hE -= 1./2 * e("kjcb,ja,kicb->ia", oovv, t1, t2)  # d31_ov
    hE += e("jiba,jb,ikac->kc", oovv, t1, t2)  # d32_ov
    hhEE -= 1./2 * p("..ab", e("ijab,klbc,ijad->kldc", oovv, t2, t2))  # d33_oovv
    hhEE += 1./4 * e("ijba,klba,ijdc->kldc", oovv, t2, t2)  # d34_oovv
    hhEE += 1./2 * p("ab..", e("jiba,ikba,jldc->kldc", oovv, t2, t2))  # d35_oovv
    hhEE += 1./2 * p("ab..", p("..ab", e("ljdb,jiba,lkdc->ikac", oovv, t2, t2)))  # d36_oovv
    hhEE -= 1./2 * p("ab..", p("..ab", e("ijcl,ia,jb,kc->lkab", oovo, t1, t1, t1)))  # d37_oovv
    hhEE -= 1./2 * p("ab..", e("ijak,la,ijcb->klcb", oovo, t1, t2))  # d38_oovv
    hE += e("jkbi,kb,ja->ia", oovo, t1, t1)  # d39_ov
    hhEE -= p("ab..", p("..ab", e("jkai,kc,jlab->ilcb", oovo, t1, t2)))  # d40_oovv
    hhEE += p("ab..", e("ilck,lc,ijba->kjba", oovo, t1, t2))  # d41_oovv
    hE -= 1./2 * e("ijak,ijab->kb", oovo, t2)  # d42_ov
    hhEE -= 1./2 * p("..ab", e("klij,lb,ka->ijba", oooo, t1, t1))  # d43_oovv
    hhEE += 1./2 * e("klij,klba->ijba", oooo, t2)  # d44_oovv
    return hE, hhEE


def energy_gs_sd(ov, oovv, t1, t2):
    scalar = 0
    scalar += e("ia,ia", ov, t1)  # s0
    scalar += 1./2 * e("jiba,ia,jb", oovv, t1, t1)  # s1
    scalar += 1./4 * e("ijba,ijba", oovv, t2)  # s2
    return scalar


def eq_gs_d(oo, vv, oooo, oovv, ovvo, vvoo, vvvv, t2):
    hhEE = 0
    hhEE += p("..ab", e("ab,ijbc->ijac", vv, t2))  # d0_oovv
    hhEE -= p("ab..", e("ki,kjba->ijba", oo, t2))  # d1_oovv
    hhEE += 1./2 * e("dcba,ijba->ijdc", vvvv, t2)  # d2_oovv
    hhEE += e("baij->ijba", vvoo)  # d3_oovv
    hhEE += p("ab..", p("..ab", e("icak,ijab->kjcb", ovvo, t2)))  # d4_oovv
    hhEE -= 1./2 * p("..ab", e("klad,ijab,kldc->ijbc", oovv, t2, t2))  # d5_oovv
    hhEE += 1./4 * e("kldc,ijdc,klba->ijba", oovv, t2, t2)  # d6_oovv
    hhEE -= 1./2 * p("ab..", e("ildc,ijba,lkdc->jkba", oovv, t2, t2))  # d7_oovv
    hhEE += 1./2 * p("ab..", p("..ab", e("jiba,jkbc,ilad->klcd", oovv, t2, t2)))  # d8_oovv
    hhEE += 1./2 * e("ijkl,ijba->klba", oooo, t2)  # d9_oovv
    return hhEE


def energy_gs_d(oovv, t2):
    scalar = 0
    scalar += 1./4 * e("ijba,ijba", oovv, t2)  # s0
    return scalar


def eq_gs_sdt(oo, ov, vo, vv, oooo, oovo, oovv, ovoo, ovvo, ovvv, vvoo, vvvo, vvvv, t1, t2, t3):
    hE = hhEE = hhhEEE = 0
    hE += e("ba,ia->ib", vv, t1)  # d0_ov
    hhEE += p("..ab", e("ac,ijcb->ijab", vv, t2))  # d1_oovv
    hhhEEE += p("...abb", e("ba,jkiadc->jkibdc", vv, t3))  # d2_ooovvv
    hE -= e("ij,ia->ja", oo, t1)  # d3_ov
    hhEE -= p("ab..", e("ij,ikba->jkba", oo, t2))  # d4_oovv
    hhhEEE -= p("abb...", e("il,ikjcba->lkjcba", oo, t3))  # d5_ooovvv
    hE += e("ai->ia", vo)  # d6_ov
    hE -= e("ia,ja,ib->jb", ov, t1, t1)  # d7_ov
    hhEE -= p("ab..", e("kc,ic,kjba->ijba", ov, t1, t2))  # d8_oovv
    hhhEEE -= p("abb...", e("ld,kd,ljicba->kjicba", ov, t1, t3))  # d9_ooovvv
    hhEE -= p("..ab", e("kc,ka,ijcb->ijab", ov, t1, t2))  # d10_oovv
    hhhEEE -= p("...abb", e("la,ld,jkiacb->jkidcb", ov, t1, t3))  # d11_ooovvv
    hhhEEE -= p("aab...", p("...abb", e("id,ijba,klcd->kljcba", ov, t2, t2)))  # d12_ooovvv
    hE += e("ia,ijab->jb", ov, t2)  # d13_ov
    hhEE += e("ia,ikjacb->kjcb", ov, t3)  # d14_oovv
    hhEE += 1./2 * p("ab..", e("cbad,ia,jd->ijcb", vvvv, t1, t1))  # d15_oovv
    hhhEEE += p("abb...", p("...aab", e("dcea,ke,ijab->kijdcb", vvvv, t1, t2)))  # d16_ooovvv
    hhEE += 1./2 * e("dcba,ijba->ijdc", vvvv, t2)  # d17_oovv
    hhhEEE += 1./2 * p("...abb", e("badc,jkidce->jkieba", vvvv, t3))  # d18_ooovvv
    hhEE += p("ab..", e("cbaj,ia->ijcb", vvvo, t1))  # d19_oovv
    hhhEEE -= p("aab...", p("...abb", e("dcak,ijab->ijkbdc", vvvo, t2)))  # d20_ooovvv
    hhEE += e("baij->ijba", vvoo)  # d21_oovv
    hhEE -= 1./2 * p("ab..", p("..ab", e("kcba,ia,jb,kd->ijcd", ovvv, t1, t1, t1)))  # d22_oovv
    hhhEEE += 1./2 * p("abc...", p("...abb", e("kcba,ia,jb,kled->iljced", ovvv, t1, t1, t2)))  # d23_ooovvv
    hhhEEE -= p("aab...", p("...abc", e("ibca,ja,id,klce->kljbde", ovvv, t1, t1, t2)))  # d24_ooovvv
    hE -= e("ibca,ia,jc->jb", ovvv, t1, t1)  # d25_ov
    hhEE += p("ab..", p("..ab", e("jcdb,kd,jiba->ikca", ovvv, t1, t2)))  # d26_oovv
    hhhEEE -= p("aab...", p("...abb", e("icba,lb,ijkaed->jklced", ovvv, t1, t3)))  # d27_ooovvv
    hhEE += 1./2 * p("..ab", e("kadc,kb,ijdc->ijab", ovvv, t1, t2))  # d28_oovv
    hhhEEE -= 1./2 * p("...abc", e("ibed,ia,kljedc->kljbca", ovvv, t1, t3))  # d29_ooovvv
    hhEE -= p("..ab", e("icba,ia,jkbd->jkcd", ovvv, t1, t2))  # d30_oovv
    hhhEEE -= p("...abb", e("iabc,ic,kljbed->kljaed", ovvv, t1, t3))  # d31_ooovvv
    hhhEEE -= p("abb...", p("...abc", e("ibac,jkcd,ilae->ljkbed", ovvv, t2, t2)))  # d32_ooovvv
    hhhEEE += 1./2 * p("abb...", p("...abb", e("iacb,iled,kjcb->lkjaed", ovvv, t2, t2)))  # d33_ooovvv
    hE += 1./2 * e("icba,ijba->jc", ovvv, t2)  # d34_ov
    hhEE += 1./2 * p("..ab", e("kadc,kijdcb->ijab", ovvv, t3))  # d35_oovv
    hhEE += p("ab..", p("..ab", e("kcaj,ia,kb->ijcb", ovvo, t1, t1)))  # d36_oovv
    hhhEEE += p("abc...", p("...abb", e("jbai,ka,jldc->kilbdc", ovvo, t1, t2)))  # d37_ooovvv
    hhhEEE += p("abb...", p("...abc", e("icbj,ia,lkbd->jlkacd", ovvo, t1, t2)))  # d38_ooovvv
    hE += e("ibaj,ia->jb", ovvo, t1)  # d39_ov
    hhEE += p("ab..", p("..ab", e("kbcj,kica->ijab", ovvo, t2)))  # d40_oovv
    hhhEEE += p("abb...", p("...abb", e("jabi,jlkbdc->ilkadc", ovvo, t3)))  # d41_ooovvv
    hhEE += p("..ab", e("iajk,ib->jkab", ovoo, t1))  # d42_oovv
    hhhEEE += p("aab...", p("...abb", e("ickl,ijba->kljcba", ovoo, t2)))  # d43_ooovvv
    hhEE += 1./4 * p("ab..", p("..ab", e("lidc,ia,jc,kd,lb->kjba", oovv, t1, t1, t1, t1)))  # d44_oovv
    hhhEEE -= 1./2 * p("abc...", p("...aab", e("ijbc,ia,kb,lc,jmed->mlkeda", oovv, t1, t1, t1, t2)))  # d45_ooovvv
    hhEE -= 1./4 * p("ab..", e("ijba,kb,la,ijdc->lkdc", oovv, t1, t1, t2))  # d46_oovv
    hhhEEE += 1./4 * p("abc...", e("lmde,kd,je,lmicba->ikjcba", oovv, t1, t1, t3))  # d47_ooovvv
    hhhEEE -= 1./2 * p("aab...", p("...abc", e("mled,id,la,mc,jkeb->jkibca", oovv, t1, t1, t1, t2)))  # d48_ooovvv
    hE -= e("jiab,ka,jc,ib->kc", oovv, t1, t1, t1)  # d49_ov
    hhEE -= p("ab..", p("..ab", e("klcd,lb,jd,kica->ijab", oovv, t1, t1, t2)))  # d50_oovv
    hhhEEE -= p("aab...", p("...aab", e("mled,ma,ke,lijdcb->ijkcba", oovv, t1, t1, t3)))  # d51_ooovvv
    hhEE -= p("ab..", e("jlcd,lc,kd,jiba->ikba", oovv, t1, t1, t2))  # d52_oovv
    hhhEEE -= p("aab...", e("miba,ia,jb,mkledc->kljedc", oovv, t1, t1, t3))  # d53_ooovvv
    hhhEEE -= 1./2 * p("abb...", p("...aab", e("jkba,ia,mlbe,jkdc->imldce", oovv, t1, t2, t2)))  # d54_ooovvv
    hhhEEE -= p("abc...", p("...abb", e("lmde,ie,ljda,mkcb->jikacb", oovv, t1, t2, t2)))  # d55_ooovvv
    hE -= 1./2 * e("jkba,ia,jkbc->ic", oovv, t1, t2)  # d56_ov
    hhEE -= 1./2 * p("ab..", e("klcd,id,kljcba->ijba", oovv, t1, t3))  # d57_oovv
    hhEE += 1./4 * p("..ab", e("ijba,jc,id,klba->kldc", oovv, t1, t1, t2))  # d58_oovv
    hhhEEE -= 1./4 * p("...abc", e("mled,lc,ma,jkibed->jkibca", oovv, t1, t1, t3))  # d59_ooovvv
    hhEE -= p("..ab", e("klcd,ld,ka,ijcb->ijab", oovv, t1, t1, t2))  # d60_oovv
    hhhEEE += p("...abb", e("jiab,ia,je,lmkbdc->lmkedc", oovv, t1, t1, t3))  # d61_ooovvv
    hhhEEE += p("abb...", p("...abc", e("klda,kc,ijab,lmde->mijecb", oovv, t1, t2, t2)))  # d62_ooovvv
    hhhEEE += 1./2 * p("abb...", p("...aab", e("ilcb,ia,jkcb,lmed->mjkeda", oovv, t1, t2, t2)))  # d63_ooovvv
    hE += 1./2 * e("ikcb,ia,kjcb->ja", oovv, t1, t2)  # d64_ov
    hhEE -= 1./2 * p("..ab", e("ilba,ld,ikjbac->kjdc", oovv, t1, t3))  # d65_oovv
    hhhEEE += p("abb...", p("...aab", e("lmae,me,ijab,lkdc->kijdcb", oovv, t1, t2, t2)))  # d66_ooovvv
    hE += e("ikac,ia,kjcb->jb", oovv, t1, t2)  # d67_ov
    hhEE += e("ijab,jb,ikladc->kldc", oovv, t1, t3)  # d68_oovv
    hhEE += 1./2 * p("..ab", e("klac,ijab,klcd->ijdb", oovv, t2, t2))  # d69_oovv
    hhhEEE += 1./2 * p("abb...", p("...aab", e("kjce,mled,ikjbac->imlbad", oovv, t2, t3)))  # d70_ooovvv
    hhEE += 1./4 * e("ijba,ijdc,klba->kldc", oovv, t2, t2)  # d71_oovv
    hhhEEE += 1./4 * p("abb...", e("ijed,mled,ijkcba->kmlcba", oovv, t2, t3))  # d72_ooovvv
    hhEE -= 1./2 * p("ab..", e("ikba,ijba,kldc->jldc", oovv, t2, t2))  # d73_oovv
    hhhEEE += 1./2 * p("aab...", p("...abb", e("lmed,mkcb,lijeda->ijkacb", oovv, t2, t3)))  # d74_ooovvv
    hhEE += 1./2 * p("ab..", p("..ab", e("ijab,jkbc,ilad->klcd", oovv, t2, t2)))  # d75_oovv
    hhhEEE += p("aab...", p("...aab", e("mled,lkdc,mijeba->ijkbac", oovv, t2, t3)))  # d76_ooovvv
    hhhEEE += 1./2 * p("aab...", e("ijba,jmba,ikledc->klmedc", oovv, t2, t3))  # d77_ooovvv
    hhhEEE += 1./4 * p("...abb", e("lmcb,lmed,jkicba->jkiaed", oovv, t2, t3))  # d78_ooovvv
    hhhEEE += 1./2 * p("...aab", e("ijba,ijac,lmkbed->lmkedc", oovv, t2, t3))  # d79_ooovvv
    hE += 1./4 * e("ijba,ijkbac->kc", oovv, t3)  # d80_ov
    hhEE += 1./2 * p("ab..", p("..ab", e("kiaj,ib,la,kc->ljcb", oovo, t1, t1, t1)))  # d81_oovv
    hhhEEE -= p("abc...", p("...aab", e("mldi,la,jd,kmcb->jkicba", oovo, t1, t1, t2)))  # d82_ooovvv
    hhEE += 1./2 * p("ab..", e("klcj,ic,klba->ijba", oovo, t1, t2))  # d83_oovv
    hhhEEE += 1./2 * p("abc...", e("lmak,ia,lmjdcb->ikjdcb", oovo, t1, t3))  # d84_ooovvv
    hhhEEE += 1./2 * p("abb...", p("...abc", e("ijak,ib,jc,mlad->kmlbdc", oovo, t1, t1, t2)))  # d85_ooovvv
    hE += e("ijak,ib,ja->kb", oovo, t1, t1)  # d86_ov
    hhEE += p("ab..", p("..ab", e("ikbj,ia,klbc->jlac", oovo, t1, t2)))  # d87_oovv
    hhhEEE += p("abb...", p("...abb", e("ikbj,ia,kmlbdc->jmladc", oovo, t1, t3)))  # d88_ooovvv
    hhEE -= p("ab..", e("ijal,ia,jkcb->lkcb", oovo, t1, t2))  # d89_oovv
    hhhEEE -= p("abb...", e("lmdk,ld,mjicba->kjicba", oovo, t1, t3))  # d90_ooovvv
    hhhEEE += 1./2 * p("aab...", p("...abb", e("lmak,ijba,lmdc->ijkbdc", oovo, t2, t2)))  # d91_ooovvv
    hhhEEE -= p("abc...", p("...aab", e("lmdk,liba,mjdc->ikjbac", oovo, t2, t2)))  # d92_ooovvv
    hE -= 1./2 * e("ijak,ijab->kb", oovo, t2)  # d93_ov
    hhEE += 1./2 * p("ab..", e("ijal,ijkacb->klcb", oovo, t3))  # d94_oovv
    hhEE += 1./2 * p("..ab", e("iljk,ia,lb->jkab", oooo, t1, t1))  # d95_oovv
    hhhEEE += p("aab...", p("...aab", e("iljk,ia,lmcb->jkmcba", oooo, t1, t2)))  # d96_ooovvv
    hhEE += 1./2 * e("ijkl,ijba->klba", oooo, t2)  # d97_oovv
    hhhEEE += 1./2 * p("aab...", e("ijkl,ijmcba->klmcba", oooo, t3))  # d98_ooovvv
    return hE, hhEE, hhhEEE


def energy_gs_sdt(ov, oovv, t1, t2):
    scalar = 0
    scalar += e("ia,ia", ov, t1)  # s0
    scalar += 1./2 * e("ijab,jb,ia", oovv, t1, t1)  # s1
    scalar += 1./4 * e("ijba,ijba", oovv, t2)  # s2
    return scalar


def eq_lambda_s(oo, ov, vv, oovo, oovv, ovvo, ovvv, a1, t1):
    He = 0
    He += e("ba,ib->ia", vv, a1)  # d0_ov
    He -= e("ji,ia->ja", oo, a1)  # d1_ov
    He += e("ia->ia", ov)  # d2_ov
    He -= e("jb,ia,ja->ib", ov, a1, t1)  # d3_ov
    He -= e("jb,ia,ib->ja", ov, a1, t1)  # d4_ov
    He -= e("jabc,ia,ib->jc", ovvv, a1, t1)  # d5_ov
    He += e("jabc,ia,jb->ic", ovvv, a1, t1)  # d6_ov
    He += e("jabi,ia->jb", ovvo, a1)  # d7_ov
    He += e("ijab,ia->jb", oovv, t1)  # d8_ov
    He += e("kiab,jc,ia,kc->jb", oovv, a1, t1, t1)  # d9_ov
    He += e("ikba,jc,ia,jb->kc", oovv, a1, t1, t1)  # d10_ov
    He -= e("jkac,ib,ia,jb->kc", oovv, a1, t1, t1)  # d11_ov
    He -= e("jkbi,ia,jb->ka", oovo, a1, t1)  # d12_ov
    He -= e("kibj,ja,ia->kb", oovo, a1, t1)  # d13_ov
    return He


def eq_lambda_sd(oo, ov, vo, vv, oooo, oovo, oovv, ovoo, ovvo, ovvv, vvvo, vvvv, a1, a2, t1, t2):
    He = HHee = 0
    He += e("ab,ia->ib", vv, a1)  # d0_ov
    HHee -= p("..ab", e("ac,ijab->ijbc", vv, a2))  # d1_oovv
    He += e("ac,ijab,ic->jb", vv, a2, t1)  # d2_ov
    He -= e("ji,ia->ja", oo, a1)  # d3_ov
    HHee -= p("ab..", e("ik,kjba->ijba", oo, a2))  # d4_oovv
    He -= e("jk,kiba,jb->ia", oo, a2, t1)  # d5_ov
    He += e("ai,ijab->jb", vo, a2)  # d6_ov
    He += e("ia->ia", ov)  # d7_ov
    HHee += p("..ab", p("ab..", e("jb,ia->ijab", ov, a1)))  # d8_oovv
    He -= e("ja,ib,jb->ia", ov, a1, t1)  # d9_ov
    He -= e("jb,ia,ib->ja", ov, a1, t1)  # d10_ov
    HHee += p("..ab", e("kc,ijba,kb->ijac", ov, a2, t1))  # d11_oovv
    He -= e("ka,ijcb,ia,kc->jb", ov, a2, t1, t1)  # d12_ov
    HHee -= p("ab..", e("jc,ikba,kc->ijba", ov, a2, t1))  # d13_oovv
    He += e("ia,jkbc,kica->jb", ov, a2, t2)  # d14_ov
    He -= 1./2 * e("jc,ikba,ijba->kc", ov, a2, t2)  # d15_ov
    He += 1./2 * e("ib,jkac,jkcb->ia", ov, a2, t2)  # d16_ov
    HHee += 1./2 * e("badc,ijba->ijdc", vvvv, a2)  # d17_oovv
    He -= 1./2 * e("cbad,jicb,jd->ia", vvvv, a2, t1)  # d18_ov
    He += 1./2 * e("bacj,ijba->ic", vvvo, a2)  # d19_ov
    HHee -= p("ab..", e("jacb,ia->ijcb", ovvv, a1))  # d20_oovv
    He -= e("jcab,ic,jb->ia", ovvv, a1, t1)  # d21_ov
    He -= e("jacb,ia,ic->jb", ovvv, a1, t1)  # d22_ov
    HHee -= p("..ab", p("ab..", e("icda,kjcb,kd->ijab", ovvv, a2, t1)))  # d23_oovv
    He += 1./2 * e("kbcd,ijba,ic,jd->ka", ovvv, a2, t1, t1)  # d24_ov
    He -= e("kbad,ijcb,ia,kc->jd", ovvv, a2, t1, t1)  # d25_ov
    He += e("jdbc,kida,jb,kc->ia", ovvv, a2, t1, t1)  # d26_ov
    HHee -= e("ibdc,jkab,ia->jkdc", ovvv, a2, t1)  # d27_oovv
    HHee += p("..ab", e("ibac,jkbd,ia->jkcd", ovvv, a2, t1))  # d28_oovv
    He += 1./2 * e("icba,jkdc,jkbd->ia", ovvv, a2, t2)  # d29_ov
    He += 1./4 * e("ibdc,jkba,jkdc->ia", ovvv, a2, t2)  # d30_ov
    He += e("iabc,jkda,ijbd->kc", ovvv, a2, t2)  # d31_ov
    He += 1./2 * e("jcba,ikcd,jiba->kd", ovvv, a2, t2)  # d32_ov
    He += e("jabi,ia->jb", ovvo, a1)  # d33_ov
    HHee += p("..ab", p("ab..", e("kaci,ijab->kjcb", ovvo, a2)))  # d34_oovv
    He -= e("jcak,kicb,ia->jb", ovvo, a2, t1)  # d35_ov
    He -= e("kbci,ijba,ka->jc", ovvo, a2, t1)  # d36_ov
    He += e("jabi,ikac,jb->kc", ovvo, a2, t1)  # d37_ov
    He += 1./2 * e("kaij,ijab->kb", ovoo, a2)  # d38_ov
    HHee += e("ijba->ijba", oovv)  # d39_oovv
    He += e("jiba,jb->ia", oovv, t1)  # d40_ov
    HHee += p("..ab", e("ijba,kc,kb->ijac", oovv, a1, t1))  # d41_oovv
    He -= e("ijab,kc,ka,ic->jb", oovv, a1, t1, t1)  # d42_ov
    He -= e("ijab,kc,ka,jb->ic", oovv, a1, t1, t1)  # d43_ov
    HHee += p("ab..", e("kiba,jc,kc->ijba", oovv, a1, t1))  # d44_oovv
    He -= e("jkba,ic,jb,kc->ia", oovv, a1, t1, t1)  # d45_ov
    HHee += p("..ab", p("ab..", e("ijab,kc,ia->jkbc", oovv, a1, t1)))  # d46_oovv
    He += e("ikac,jb,ijab->kc", oovv, a1, t2)  # d47_ov
    He -= 1./2 * e("ijba,kc,ikba->jc", oovv, a1, t2)  # d48_ov
    He -= 1./2 * e("ijab,kc,ijac->kb", oovv, a1, t2)  # d49_ov
    HHee -= p("ab..", e("ijab,lkdc,la,jb->ikdc", oovv, a2, t1, t1))  # d50_oovv
    He -= e("jiab,klcd,jc,ka,ib->ld", oovv, a2, t1, t1, t1)  # d51_ov
    HHee += p("..ab", p("ab..", e("jiab,klcd,ka,ic->jlbd", oovv, a2, t1, t1)))  # d52_oovv
    He += 1./2 * e("ijcb,lkad,ia,jd,lc->kb", oovv, a2, t1, t1, t1)  # d53_ov
    He -= 1./2 * e("jidc,lkab,ia,kc,ld->jb", oovv, a2, t1, t1, t1)  # d54_ov
    HHee += 1./2 * e("klad,ijcb,ia,jd->klcb", oovv, a2, t1, t1)  # d55_oovv
    He += 1./2 * e("ijab,lkdc,la,ijbd->kc", oovv, a2, t1, t2)  # d56_ov
    He += 1./4 * e("klab,ijdc,ia,kldc->jb", oovv, a2, t1, t2)  # d57_ov
    He += e("klcd,ijba,ic,ljdb->ka", oovv, a2, t1, t2)  # d58_ov
    He -= 1./2 * e("liba,jkdc,jb,lkdc->ia", oovv, a2, t1, t2)  # d59_ov
    HHee += p("..ab", e("ildc,jkab,ia,ld->jkcb", oovv, a2, t1, t1))  # d60_oovv
    HHee += 1./2 * e("ildc,jkab,ia,lb->jkdc", oovv, a2, t1, t1)  # d61_oovv
    He += 1./2 * e("lkdc,jiab,la,kjdc->ib", oovv, a2, t1, t2)  # d62_ov
    He += e("lkdc,jiab,la,kjcb->id", oovv, a2, t1, t2)  # d63_ov
    He += 1./4 * e("ijdc,klab,ia,kldc->jb", oovv, a2, t1, t2)  # d64_ov
    He += 1./2 * e("ildc,jkab,ia,jkcb->ld", oovv, a2, t1, t2)  # d65_ov
    He += e("lida,jkbc,ld,ijab->kc", oovv, a2, t1, t2)  # d66_ov
    He += 1./2 * e("lidc,jkba,ld,ijba->kc", oovv, a2, t1, t2)  # d67_ov
    He += 1./2 * e("lidb,jkca,ld,jkbc->ia", oovv, a2, t1, t2)  # d68_ov
    HHee -= 1./2 * p("..ab", e("ijab,kldc,klad->ijbc", oovv, a2, t2))  # d69_oovv
    HHee += 1./4 * e("klba,ijdc,ijba->kldc", oovv, a2, t2)  # d70_oovv
    HHee -= 1./2 * p("ab..", e("ijba,lkdc,ildc->jkba", oovv, a2, t2))  # d71_oovv
    HHee += p("..ab", p("ab..", e("ijab,klcd,ikac->jlbd", oovv, a2, t2)))  # d72_oovv
    HHee += 1./2 * p("ab..", e("ildc,kjba,lkdc->ijba", oovv, a2, t2))  # d73_oovv
    HHee += 1./4 * e("ijba,kldc,ijdc->klba", oovv, a2, t2)  # d74_oovv
    HHee -= 1./2 * p("..ab", e("ijac,klbd,ijab->klcd", oovv, a2, t2))  # d75_oovv
    HHee += p("..ab", e("jkbi,ia->jkab", oovo, a1))  # d76_oovv
    He -= e("ikaj,jb,ia->kb", oovo, a1, t1)  # d77_ov
    He -= e("ijak,kb,jb->ia", oovo, a1, t1)  # d78_ov
    HHee += e("klci,jiba,jc->klba", oovo, a2, t1)  # d79_oovv
    He += e("klcj,ijab,ka,ic->lb", oovo, a2, t1, t1)  # d80_ov
    HHee += p("..ab", p("ab..", e("ijbl,lkac,ia->jkbc", oovo, a2, t1)))  # d81_oovv
    He -= 1./2 * e("ijak,klbc,ib,jc->la", oovo, a2, t1, t1)  # d82_ov
    He += e("jkai,ilcb,jc,ka->lb", oovo, a2, t1, t1)  # d83_ov
    HHee -= p("ab..", e("kicl,ljba,kc->ijba", oovo, a2, t1))  # d84_oovv
    He -= 1./2 * e("ilck,jkba,ijba->lc", oovo, a2, t2)  # d85_ov
    He -= e("ikaj,ljbc,ilab->kc", oovo, a2, t2)  # d86_ov
    He -= 1./4 * e("jkal,licb,jkcb->ia", oovo, a2, t2)  # d87_ov
    He -= 1./2 * e("klbj,jica,klbc->ia", oovo, a2, t2)  # d88_ov
    HHee += 1./2 * e("ijkl,klba->ijba", oooo, a2)  # d89_oovv
    He -= 1./2 * e("iljk,jkba,lb->ia", oooo, a2, t1)  # d90_ov
    return He, HHee


def eq_lambda_d(oo, vv, oooo, oovv, ovvo, vvvv, a2, t2):
    HHee = 0
    HHee += p("..ab", e("ba,ijbc->ijac", vv, a2))  # d0_oovv
    HHee -= p("ab..", e("ij,jkba->ikba", oo, a2))  # d1_oovv
    HHee += 1./2 * e("dcba,ijdc->ijba", vvvv, a2)  # d2_oovv
    HHee += p("..ab", p("ab..", e("kbcj,jiba->kica", ovvo, a2)))  # d3_oovv
    HHee += e("ijba->ijba", oovv)  # d4_oovv
    HHee -= 1./2 * p("..ab", e("ijab,klcd,klac->ijbd", oovv, a2, t2))  # d5_oovv
    HHee += 1./4 * e("ijba,kldc,klba->ijdc", oovv, a2, t2)  # d6_oovv
    HHee -= 1./2 * p("ab..", e("ikdc,jlba,ijba->kldc", oovv, a2, t2))  # d7_oovv
    HHee += p("..ab", p("ab..", e("klcd,jiba,jkbc->lida", oovv, a2, t2)))  # d8_oovv
    HHee -= 1./2 * p("ab..", e("kidc,ljba,kldc->ijba", oovv, a2, t2))  # d9_oovv
    HHee += 1./4 * e("ijba,kldc,ijdc->klba", oovv, a2, t2)  # d10_oovv
    HHee -= 1./2 * p("..ab", e("ijab,kldc,ijad->klbc", oovv, a2, t2))  # d11_oovv
    HHee += 1./2 * e("ijkl,klba->ijba", oooo, a2)  # d12_oovv
    return HHee


def eq_lambda_sdt(oo, ov, vo, vv, oooo, oovo, oovv, ovoo, ovvo, ovvv, vvoo, vvvo, vvvv, a1, a2, a3, t1, t2, t3):
    He = HHee = HHHeee = 0
    He += e("ba,ib->ia", vv, a1)  # d0_ov
    HHee += p("..ab", e("ca,ijcb->ijab", vv, a2))  # d1_oovv
    He += e("ba,jibc,ja->ic", vv, a2, t1)  # d2_ov
    HHHeee += p("...abb", e("da,jkidcb->jkiacb", vv, a3))  # d3_ooovvv
    HHee += e("ab,ijkadc,ib->jkdc", vv, a3, t1)  # d4_oovv
    He += 1./2 * e("db,ijkdac,ijba->kc", vv, a3, t2)  # d5_ov
    He -= e("ji,ia->ja", oo, a1)  # d6_ov
    HHee -= p("ab..", e("ki,ijba->kjba", oo, a2))  # d7_oovv
    He -= e("ij,jkab,ia->kb", oo, a2, t1)  # d8_ov
    HHHeee -= p("abb...", e("lk,kjicba->ljicba", oo, a3))  # d9_ooovvv
    HHee -= e("li,ikjacb,la->kjcb", oo, a3, t1)  # d10_oovv
    He -= 1./2 * e("jl,likbac,jiba->kc", oo, a3, t2)  # d11_ov
    He += e("bj,ijab->ia", vo, a2)  # d12_ov
    HHee += e("ai,ijkacb->jkcb", vo, a3)  # d13_oovv
    He += e("ia->ia", ov)  # d14_ov
    HHee += p("..ab", p("ab..", e("jb,ia->jiba", ov, a1)))  # d15_oovv
    He -= e("ib,ja,jb->ia", ov, a1, t1)  # d16_ov
    He -= e("ja,ib,jb->ia", ov, a1, t1)  # d17_ov
    HHHeee += p("...abb", p("abb...", e("ia,jkcb->ijkacb", ov, a2)))  # d18_ooovvv
    HHee -= p("ab..", e("ic,kjba,kc->ijba", ov, a2, t1))  # d19_oovv
    He += e("ja,ikcb,ia,jb->kc", ov, a2, t1, t1)  # d20_ov
    HHee += p("..ab", e("ka,ijbc,kc->ijab", ov, a2, t1))  # d21_oovv
    He += 1./2 * e("ib,jkca,jkbc->ia", ov, a2, t2)  # d22_ov
    He += 1./2 * e("kc,ijba,kiba->jc", ov, a2, t2)  # d23_ov
    He += e("kc,ijab,kjcb->ia", ov, a2, t2)  # d24_ov
    HHHeee -= p("abb...", e("ja,ilkdcb,ia->jlkdcb", ov, a3, t1))  # d25_ooovvv
    HHee -= e("ia,jkldcb,ja,id->klcb", ov, a3, t1, t1)  # d26_oovv
    He -= 1./2 * e("ia,ljkdcb,ka,ildc->jb", ov, a3, t1, t2)  # d27_ov
    HHHeee -= p("...abb", e("ib,kljdca,ia->kljbdc", ov, a3, t1))  # d28_ooovvv
    He += 1./2 * e("jb,klidca,jc,klbd->ia", ov, a3, t1, t2)  # d29_ov
    HHee += 1./2 * p("ab..", e("ka,ijlbdc,ijab->kldc", ov, a3, t2))  # d30_oovv
    HHee += 1./2 * p("..ab", e("ic,jklbad,ijba->klcd", ov, a3, t2))  # d31_oovv
    HHee += e("ia,jklcbd,ilad->jkcb", ov, a3, t2)  # d32_oovv
    He -= 1./12 * e("ib,kljadc,kljbdc->ia", ov, a3, t3)  # d33_ov
    He -= 1./12 * e("ia,kljdcb,ikldcb->ja", ov, a3, t3)  # d34_ov
    He += 1./4 * e("ia,jlkbdc,ilkadc->jb", ov, a3, t3)  # d35_ov
    HHee += 1./2 * e("dcba,ijdc->ijba", vvvv, a2)  # d36_oovv
    He -= 1./2 * e("cbda,ijcb,jd->ia", vvvv, a2, t1)  # d37_ov
    HHHeee += 1./2 * p("...aab", e("badc,jkibae->jkidce", vvvv, a3))  # d38_ooovvv
    HHee += 1./2 * p("..ab", e("cbad,ijkcbe,ia->jkde", vvvv, a3, t1))  # d39_oovv
    He += 1./4 * e("dcbe,jkidca,jb,ke->ia", vvvv, a3, t1, t1)  # d40_ov
    He -= 1./4 * e("edac,ijkbed,ijab->kc", vvvv, a3, t2)  # d41_ov
    He += 1./8 * e("edba,ijkedc,ijba->kc", vvvv, a3, t2)  # d42_ov
    He -= 1./2 * e("baci,ijba->jc", vvvo, a2)  # d43_ov
    HHee += 1./2 * p("..ab", e("badi,ikjbac->kjcd", vvvo, a3))  # d44_oovv
    He -= 1./2 * e("badi,ijkbac,jd->kc", vvvo, a3, t1)  # d45_ov
    He += 1./4 * e("baij,ijkbac->kc", vvoo, a3)  # d46_ov
    HHee += p("ab..", e("icba,jc->ijba", ovvv, a1))  # d47_oovv
    He -= e("ibca,jb,jc->ia", ovvv, a1, t1)  # d48_ov
    He += e("ibac,jb,ia->jc", ovvv, a1, t1)  # d49_ov
    HHHeee += p("...abb", p("aab...", e("icba,jkcd->jkidba", ovvv, a2)))  # d50_ooovvv
    HHee += p("..ab", e("kadc,ijab,kc->ijbd", ovvv, a2, t1))  # d51_oovv
    He -= e("jcab,ikcd,ia,jb->kd", ovvv, a2, t1, t1)  # d52_ov
    HHee += e("kadc,ijab,kb->ijdc", ovvv, a2, t1)  # d53_oovv
    He += e("kbda,ijcb,jd,kc->ia", ovvv, a2, t1, t1)  # d54_ov
    HHee -= p("..ab", p("ab..", e("kbdc,ijab,jd->ikac", ovvv, a2, t1)))  # d55_oovv
    He -= 1./2 * e("iacb,jkad,jb,kc->id", ovvv, a2, t1, t1)  # d56_ov
    He += 1./2 * e("icba,kjcd,jiba->kd", ovvv, a2, t2)  # d57_ov
    He += e("jcbd,ikac,ijab->kd", ovvv, a2, t2)  # d58_ov
    He += 1./4 * e("iacb,jkad,jkcb->id", ovvv, a2, t2)  # d59_ov
    He += 1./2 * e("kadc,ijab,ijbd->kc", ovvv, a2, t2)  # d60_ov
    HHHeee -= p("...aab", e("lade,jkiacb,le->jkicbd", ovvv, a3, t1))  # d61_ooovvv
    HHee -= e("lcde,ijkbac,le,kd->ijba", ovvv, a3, t1, t1)  # d62_oovv
    He += 1./2 * e("lbed,jkicba,ld,jkec->ia", ovvv, a3, t1, t2)  # d63_ov
    HHHeee -= p("...abb", e("iedc,kljaeb,ia->kljbdc", ovvv, a3, t1))  # d64_ooovvv
    HHee += p("..ab", e("ldcb,kijaed,kc,le->ijab", ovvv, a3, t1, t1))  # d65_oovv
    He -= 1./2 * e("kdab,iljced,ia,jb,kc->le", ovvv, a3, t1, t1, t1)  # d66_ov
    He += 1./4 * e("ibdc,jlkeba,ia,lkdc->je", ovvv, a3, t1, t2)  # d67_ov
    He -= 1./2 * e("iacb,jlkaed,id,lkec->jb", ovvv, a3, t1, t2)  # d68_ov
    HHHeee -= p("...aab", p("aab...", e("lced,ijkbac,ke->ijlbad", ovvv, a3, t1)))  # d69_ooovvv
    HHee += 1./2 * p("ab..", e("leab,ikjdce,ia,jb->kldc", ovvv, a3, t1, t1))  # d70_oovv
    He += e("jebc,ilkade,kc,ijab->ld", ovvv, a3, t1, t2)  # d71_ov
    He += 1./2 * e("lbae,ikjbdc,je,kldc->ia", ovvv, a3, t1, t2)  # d72_ov
    He += 1./2 * e("iabc,kljade,jc,kleb->id", ovvv, a3, t1, t2)  # d73_ov
    HHee -= 1./2 * e("kedc,ijlbae,lkdc->ijba", ovvv, a3, t2)  # d74_oovv
    HHee += p("..ab", e("lbed,ijkabc,klce->ijad", ovvv, a3, t2))  # d75_oovv
    HHee -= 1./2 * e("lced,ikjbac,ilba->kjed", ovvv, a3, t2)  # d76_oovv
    HHee -= 1./4 * p("ab..", e("iacb,jlkaed,lkcb->jied", ovvv, a3, t2))  # d77_oovv
    HHee += 1./2 * p("..ab", p("ab..", e("lbed,jkibca,jkce->ilad", ovvv, a3, t2)))  # d78_oovv
    He -= 1./4 * e("lbed,ikjbac,kjlced->ia", ovvv, a3, t3)  # d79_ov
    He += 1./4 * e("laed,ikjacb,kjlcbe->id", ovvv, a3, t3)  # d80_ov
    He -= 1./12 * e("leba,jkidec,jkicba->ld", ovvv, a3, t3)  # d81_ov
    He += 1./12 * e("lcde,jkibac,jkibae->ld", ovvv, a3, t3)  # d82_ov
    He += e("ibaj,jb->ia", ovvo, a1)  # d83_ov
    HHee += p("..ab", p("ab..", e("ibaj,jkbc->ikac", ovvo, a2)))  # d84_oovv
    He -= e("icbk,kjca,jb->ia", ovvo, a2, t1)  # d85_ov
    He -= e("ibcj,jkba,ia->kc", ovvo, a2, t1)  # d86_ov
    He += e("kbcj,jiba,kc->ia", ovvo, a2, t1)  # d87_ov
    HHHeee += p("...abb", p("abb...", e("ibaj,jlkbdc->ilkadc", ovvo, a3)))  # d88_ooovvv
    HHee += p("ab..", e("ldak,ikjdcb,ia->ljcb", ovvo, a3, t1))  # d89_oovv
    He += e("icdl,kljcab,ia,kd->jb", ovvo, a3, t1, t1)  # d90_ov
    HHee += p("..ab", e("idbl,ljkadc,ia->jkbc", ovvo, a3, t1))  # d91_oovv
    HHee += e("lcdk,kjicba,ld->jiba", ovvo, a3, t1)  # d92_oovv
    He += 1./2 * e("jabi,ikladc,klbd->jc", ovvo, a3, t2)  # d93_ov
    He += 1./2 * e("jabi,ilkadc,jldc->kb", ovvo, a3, t2)  # d94_ov
    He += e("ibaj,jlkbdc,ilad->kc", ovvo, a3, t2)  # d95_ov
    He += 1./2 * e("kaij,ijab->kb", ovoo, a2)  # d96_ov
    HHee += 1./2 * p("ab..", e("laij,ijkacb->lkcb", ovoo, a3))  # d97_oovv
    He -= 1./2 * e("ickl,kljacb,ia->jb", ovoo, a3, t1)  # d98_ov
    HHee += e("ijba->ijba", oovv)  # d99_oovv
    He += e("ijab,jb->ia", oovv, t1)  # d100_ov
    HHHeee += p("...aab", p("aab...", e("jkcb,ia->jkicba", oovv, a1)))  # d101_ooovvv
    HHee -= p("..ab", e("ijac,kb,kc->ijab", oovv, a1, t1))  # d102_oovv
    He -= e("jkbc,ia,ja,ib->kc", oovv, a1, t1, t1)  # d103_ov
    He -= e("ijac,kb,ia,kc->jb", oovv, a1, t1, t1)  # d104_ov
    HHee -= p("ab..", e("ikba,jc,kc->ijba", oovv, a1, t1))  # d105_oovv
    He += e("ijba,kc,jb,ic->ka", oovv, a1, t1, t1)  # d106_ov
    HHee += p("..ab", p("ab..", e("jiba,kc,jb->ikac", oovv, a1, t1)))  # d107_oovv
    He += e("ijab,kc,ikac->jb", oovv, a1, t2)  # d108_ov
    He -= 1./2 * e("ikba,jc,ijba->kc", oovv, a1, t2)  # d109_ov
    He -= 1./2 * e("ijab,kc,ijac->kb", oovv, a1, t2)  # d110_ov
    HHHeee += p("...abb", p("aab...", e("kldc,ijba,id->kljcba", oovv, a2, t1)))  # d111_ooovvv
    HHee += 1./2 * e("ijcd,lkba,kd,lc->ijba", oovv, a2, t1, t1)  # d112_oovv
    He += 1./2 * e("jiba,klcd,ka,ic,lb->jd", oovv, a2, t1, t1, t1)  # d113_ov
    HHee -= p("..ab", p("ab..", e("lkcd,jiab,ka,jd->licb", oovv, a2, t1, t1)))  # d114_oovv
    He += 1./2 * e("licd,kjab,ia,lb,kd->jc", oovv, a2, t1, t1, t1)  # d115_ov
    He -= e("lkdc,ijab,kc,id,la->jb", oovv, a2, t1, t1, t1)  # d116_ov
    HHee += p("ab..", e("ijba,lkdc,ia,lb->jkdc", oovv, a2, t1, t1))  # d117_oovv
    He += 1./2 * e("jiab,kldc,la,jkdc->ib", oovv, a2, t1, t2)  # d118_ov
    He -= e("ijab,klcd,lb,ikac->jd", oovv, a2, t1, t2)  # d119_ov
    He += 1./4 * e("klab,ijdc,ia,kldc->jb", oovv, a2, t1, t2)  # d120_ov
    He += 1./2 * e("kldc,jiba,jd,klcb->ia", oovv, a2, t1, t2)  # d121_ov
    HHHeee += p("...aab", p("abb...", e("liba,kjdc,ld->ikjbac", oovv, a2, t1)))  # d122_ooovvv
    HHee += 1./2 * e("jiba,kldc,jd,ic->klba", oovv, a2, t1, t1)  # d123_oovv
    HHee -= p("..ab", e("ilad,jkcb,ia,lc->jkdb", oovv, a2, t1, t1))  # d124_oovv
    He += 1./2 * e("ijab,klcd,id,klac->jb", oovv, a2, t1, t2)  # d125_ov
    He += 1./4 * e("jiba,kldc,jd,klba->ic", oovv, a2, t1, t2)  # d126_ov
    He += e("kldc,jiab,kb,ljda->ic", oovv, a2, t1, t2)  # d127_ov
    He -= 1./2 * e("kldc,ijba,lb,kidc->ja", oovv, a2, t1, t2)  # d128_ov
    HHHeee += p("...abb", p("abb...", e("ijab,lkdc,ia->jlkbdc", oovv, a2, t1)))  # d129_ooovvv
    He -= 1./2 * e("lkcd,ijab,kc,ijda->lb", oovv, a2, t1, t2)  # d130_ov
    He -= 1./2 * e("ijba,lkdc,jb,ildc->ka", oovv, a2, t1, t2)  # d131_ov
    He += e("klcd,ijab,ld,kica->jb", oovv, a2, t1, t2)  # d132_ov
    HHee -= 1./2 * p("..ab", e("klac,ijbd,ijab->klcd", oovv, a2, t2))  # d133_oovv
    HHee += 1./4 * e("ijba,kldc,klba->ijdc", oovv, a2, t2)  # d134_oovv
    HHee -= 1./2 * p("ab..", e("ikdc,jlba,ijba->kldc", oovv, a2, t2))  # d135_oovv
    HHee += p("..ab", p("ab..", e("ijab,klcd,ikac->jlbd", oovv, a2, t2)))  # d136_oovv
    HHee -= 1./2 * p("ab..", e("jiba,lkdc,jlba->ikdc", oovv, a2, t2))  # d137_oovv
    HHee += 1./4 * e("ijba,kldc,ijdc->klba", oovv, a2, t2)  # d138_oovv
    HHee -= 1./2 * p("..ab", e("klca,ijdb,klcd->ijab", oovv, a2, t2))  # d139_oovv
    He += 1./4 * e("jiba,lkdc,jlkbdc->ia", oovv, a2, t3)  # d140_ov
    He += 1./4 * e("ilba,kjcd,ikjbac->ld", oovv, a2, t3)  # d141_ov
    He += 1./4 * e("ijab,kldc,ijkadc->lb", oovv, a2, t3)  # d142_ov
    He += 1./4 * e("kjcb,ilad,ikjacb->ld", oovv, a2, t3)  # d143_ov
    HHHeee -= 1./2 * p("aab...", e("jked,limcba,md,le->jkicba", oovv, a3, t1, t1))  # d144_ooovvv
    HHee += 1./2 * p("ab..", e("mlde,ikjcba,id,ke,mc->ljba", oovv, a3, t1, t1, t1))  # d145_oovv
    He += 1./4 * e("lmed,kijcab,jd,ke,lc,mb->ia", oovv, a3, t1, t1, t1, t1)  # d146_ov
    He += 1./4 * e("imed,ljkbac,le,kd,ijba->mc", oovv, a3, t1, t1, t2)  # d147_ov
    He -= 1./8 * e("ijab,kmldce,ka,lb,ijdc->me", oovv, a3, t1, t1, t2)  # d148_ov
    HHHeee -= p("...abb", p("abb...", e("lkec,mjidba,ld,me->kjicba", oovv, a3, t1, t1)))  # d149_ooovvv
    HHee -= 1./2 * p("..ab", e("imbe,ljkadc,ia,le,md->jkbc", oovv, a3, t1, t1, t1))  # d150_oovv
    HHee += e("kjba,imlced,ia,jb,kc->mled", oovv, a3, t1, t1, t1)  # d151_oovv
    He += 1./2 * e("klbd,ijmcae,kc,md,ijba->le", oovv, a3, t1, t1, t2)  # d152_ov
    He += 1./2 * e("mlae,kijbdc,je,mb,lkdc->ia", oovv, a3, t1, t1, t2)  # d153_ov
    He += e("mlde,ikjbac,mb,je,lida->kc", oovv, a3, t1, t1, t2)  # d154_ov
    HHHeee -= p("abb...", e("lide,kjmcba,ld,me->ikjcba", oovv, a3, t1, t1))  # d155_ooovvv
    He += 1./2 * e("ikac,mljedb,ia,lc,kmed->jb", oovv, a3, t1, t1, t2)  # d156_ov
    HHee += 1./2 * e("ijba,klmdce,kb,lmae->ijdc", oovv, a3, t1, t2)  # d157_oovv
    HHee += 1./2 * p("..ab", p("ab..", e("ikec,jmlbad,me,ijba->klcd", oovv, a3, t1, t2)))  # d158_oovv
    HHee -= p("ab..", e("mlde,ikjacb,ke,mida->ljcb", oovv, a3, t1, t2))  # d159_oovv
    HHee += 1./4 * p("..ab", e("jkad,ilmecb,ia,jkcb->lmde", oovv, a3, t1, t2))  # d160_oovv
    HHee += 1./2 * e("ijba,mkldce,mb,ijae->kldc", oovv, a3, t1, t2)  # d161_oovv
    He += 1./12 * e("mled,ijkcba,kd,ijmcba->le", oovv, a3, t1, t3)  # d162_ov
    He -= 1./4 * e("imae,kjlcbd,le,ikjacb->md", oovv, a3, t1, t3)  # d163_ov
    He += 1./12 * e("jkae,limdcb,ia,jkldcb->me", oovv, a3, t1, t3)  # d164_ov
    He += 1./4 * e("jkba,miledc,ia,jkmbed->lc", oovv, a3, t1, t3)  # d165_ov
    HHHeee -= 1./2 * p("...aab", e("imcb,kljade,ia,me->kljcbd", oovv, a3, t1, t1))  # d166_ooovvv
    He += 1./4 * e("jiab,klmced,jd,ie,klac->mb", oovv, a3, t1, t1, t2)  # d167_ov
    He -= 1./8 * e("ijba,lmkcde,ic,je,lmba->kd", oovv, a3, t1, t1, t2)  # d168_ov
    HHHeee += p("...abb", e("ijab,lmkced,jc,ib->lmkaed", oovv, a3, t1, t1))  # d169_ooovvv
    He += 1./2 * e("jiab,klmcde,ia,je,klbc->md", oovv, a3, t1, t1, t2)  # d170_ov
    HHee -= 1./2 * p("..ab", p("ab..", e("lmed,ijkacb,mb,ijae->lkdc", oovv, a3, t1, t2)))  # d171_oovv
    HHee += 1./4 * p("ab..", e("imed,kljacb,ia,kled->mjcb", oovv, a3, t1, t2))  # d172_oovv
    HHee += 1./2 * e("kied,jlmcba,ia,jkcb->lmed", oovv, a3, t1, t2)  # d173_oovv
    HHee -= p("..ab", e("lmed,ikjbac,ma,ilbe->kjdc", oovv, a3, t1, t2))  # d174_oovv
    HHee += 1./2 * e("jkba,ilmced,kc,ijba->lmed", oovv, a3, t1, t2)  # d175_oovv
    He -= 1./12 * e("lmae,jkicbd,ld,jkiacb->me", oovv, a3, t1, t3)  # d176_ov
    He += 1./12 * e("imed,kljbac,ia,kljedb->mc", oovv, a3, t1, t3)  # d177_ov
    He += 1./4 * e("ijba,mlkdce,ie,jmlbdc->ka", oovv, a3, t1, t3)  # d178_ov
    He -= 1./4 * e("ijba,lmkdce,je,ilmbad->kc", oovv, a3, t1, t3)  # d179_ov
    HHee -= 1./2 * p("ab..", e("klac,ijmbed,kc,ijab->lmed", oovv, a3, t1, t2))  # d180_oovv
    HHee += 1./2 * p("..ab", e("imce,jklbad,me,ijba->klcd", oovv, a3, t1, t2))  # d181_oovv
    HHee += e("jkbc,ilmaed,kc,ijab->lmed", oovv, a3, t1, t2)  # d182_oovv
    He -= 1./12 * e("jiab,lmkedc,ib,lmkaed->jc", oovv, a3, t1, t3)  # d183_ov
    He += 1./12 * e("kjba,lmiedc,jb,klmedc->ia", oovv, a3, t1, t3)  # d184_ov
    He += 1./4 * e("ijab,lmkedc,jb,ilmaed->kc", oovv, a3, t1, t3)  # d185_ov
    HHHeee -= 1./2 * p("...abb", p("aab...", e("lmed,ijkacb,ijea->lmkdcb", oovv, a3, t2)))  # d186_ooovvv
    He += 1./4 * e("lmed,kjicba,jida,mkcb->le", oovv, a3, t2, t2)  # d187_ov
    He -= 1./2 * e("mlae,kijdbc,ijab,mked->lc", oovv, a3, t2, t2)  # d188_ov
    He -= 1./8 * e("ijab,klmedc,klac,ijed->mb", oovv, a3, t2, t2)  # d189_ov
    He -= 1./4 * e("jkbd,lmicea,jkbc,lmde->ia", oovv, a3, t2, t2)  # d190_ov
    HHHeee += 1./4 * p("aab...", e("ijba,kmledc,mlba->ijkedc", oovv, a3, t2))  # d191_ooovvv
    He -= 1./8 * e("mlba,kijdce,ijba,mkdc->le", oovv, a3, t2, t2)  # d192_ov
    He += 1./16 * e("lmed,ijkbac,ijed,lmba->kc", oovv, a3, t2, t2)  # d193_ov
    HHHeee -= 1./2 * p("...aab", p("abb...", e("ikdc,jmlbae,ijba->kmldce", oovv, a3, t2)))  # d194_ooovvv
    He += 1./2 * e("ijba,mlkedc,jldc,imbe->ka", oovv, a3, t2, t2)  # d195_ov
    He -= 1./4 * e("jiba,mlkedc,iled,jmba->kc", oovv, a3, t2, t2)  # d196_ov
    HHHeee += p("...abb", p("abb...", e("ijab,mlkedc,imae->jlkbdc", oovv, a3, t2)))  # d197_ooovvv
    He += 1./2 * e("lmde,ikjacb,lkdc,miea->jb", oovv, a3, t2, t2)  # d198_ov
    HHHeee -= 1./2 * p("abb...", e("ijba,mlkedc,imba->jlkedc", oovv, a3, t2))  # d199_ooovvv
    HHHeee += 1./4 * p("...aab", e("ijba,lmkdce,ijdc->lmkbae", oovv, a3, t2))  # d200_ooovvv
    HHHeee += 1./2 * p("...abb", e("ijab,lmkdce,ijbe->lmkadc", oovv, a3, t2))  # d201_ooovvv
    HHee += 1./12 * p("..ab", e("ijca,lmkedb,lmkced->ijab", oovv, a3, t3))  # d202_oovv
    HHee += 1./12 * e("lmed,jkiacb,jkieda->lmcb", oovv, a3, t3)  # d203_oovv
    HHee += 1./12 * p("ab..", e("ijba,klmedc,ikledc->jmba", oovv, a3, t3))  # d204_oovv
    HHee += 1./4 * p("..ab", p("ab..", e("ijab,lmkedc,ilmaed->jkbc", oovv, a3, t3)))  # d205_oovv
    HHee += 1./4 * p("ab..", e("kmcb,jilaed,kjicba->mled", oovv, a3, t3))  # d206_oovv
    HHee += 1./12 * e("jked,ilmcba,jkicba->lmed", oovv, a3, t3)  # d207_oovv
    HHee -= 1./4 * p("..ab", e("klac,ijmbed,klmced->ijab", oovv, a3, t3))  # d208_oovv
    HHee += 1./4 * e("lmed,ikjacb,lmieda->kjcb", oovv, a3, t3)  # d209_oovv
    HHee -= p("..ab", e("jkai,ib->jkab", oovo, a1))  # d210_oovv
    He += e("ikaj,jb,ib->ka", oovo, a1, t1)  # d211_ov
    He += e("jiak,kb,ia->jb", oovo, a1, t1)  # d212_ov
    HHHeee -= p("...abb", p("aab...", e("jkai,ilcb->jklacb", oovo, a2)))  # d213_ooovvv
    HHee -= e("klci,ijba,jc->klba", oovo, a2, t1)  # d214_oovv
    He += e("jkal,licb,ia,kc->jb", oovo, a2, t1, t1)  # d215_ov
    HHee -= p("..ab", p("ab..", e("kjai,ilbc,jb->klac", oovo, a2, t1)))  # d216_oovv
    He += 1./2 * e("lick,kjab,ia,lb->jc", oovo, a2, t1, t1)  # d217_ov
    He += e("kjai,ilbc,kb,ja->lc", oovo, a2, t1, t1)  # d218_ov
    HHee -= p("ab..", e("kicl,ljba,kc->ijba", oovo, a2, t1))  # d219_oovv
    He += 1./2 * e("jkai,ilcb,jlcb->ka", oovo, a2, t2)  # d220_ov
    He -= e("klcj,jiab,lica->kb", oovo, a2, t2)  # d221_ov
    He -= 1./4 * e("jkai,ilcb,jkcb->la", oovo, a2, t2)  # d222_ov
    He -= 1./2 * e("klbj,jica,klbc->ia", oovo, a2, t2)  # d223_ov
    HHHeee += p("aab...", e("ijdl,lkmcba,md->ijkcba", oovo, a3, t1))  # d224_ooovvv
    HHee += p("ab..", e("mldj,jkicba,id,mc->lkba", oovo, a3, t1, t1))  # d225_oovv
    He += 1./2 * e("ijak,kmlcbd,ic,ma,jd->lb", oovo, a3, t1, t1, t1)  # d226_ov
    He -= 1./2 * e("jmal,lkicbd,ia,jkcb->md", oovo, a3, t1, t2)  # d227_ov
    He += 1./4 * e("klbj,jimdca,mb,kldc->ia", oovo, a3, t1, t2)  # d228_ov
    HHHeee += p("...abb", p("abb...", e("ildm,mkjacb,ia->lkjdcb", oovo, a3, t1)))  # d229_ooovvv
    HHee += 1./2 * p("..ab", e("imbl,ljkdac,ia,md->jkbc", oovo, a3, t1, t1))  # d230_oovv
    HHee += e("imdl,ljkacb,ia,md->jkcb", oovo, a3, t1, t1)  # d231_oovv
    He += 1./2 * e("lmdi,ikjacb,ma,kjdb->lc", oovo, a3, t1, t2)  # d232_ov
    He += 1./2 * e("kiaj,jlmdcb,ib,kmdc->la", oovo, a3, t1, t2)  # d233_ov
    He -= e("ikbj,jmldac,ia,klbc->md", oovo, a3, t1, t2)  # d234_ov
    HHHeee += p("abb...", e("ildm,mkjcba,ld->ikjcba", oovo, a3, t1))  # d235_ooovvv
    He += 1./2 * e("lmdi,ijkbac,md,ljba->kc", oovo, a3, t1, t2)  # d236_ov
    HHee -= 1./2 * e("lmdi,ikjbac,kjdc->lmba", oovo, a3, t2)  # d237_oovv
    HHee -= 1./2 * p("..ab", p("ab..", e("jiak,kmldcb,imdc->jlab", oovo, a3, t2)))  # d238_oovv
    HHee -= p("ab..", e("mldi,ikjcba,mjda->lkcb", oovo, a3, t2))  # d239_oovv
    HHee -= 1./4 * p("..ab", e("klam,mijbdc,kldc->ijab", oovo, a3, t2))  # d240_oovv
    HHee -= 1./2 * e("kldm,mijbac,kldc->ijba", oovo, a3, t2)  # d241_oovv
    He -= 1./12 * e("ijak,kmldcb,jmldcb->ia", oovo, a3, t3)  # d242_ov
    He -= 1./4 * e("mldk,kjicba,mjidba->lc", oovo, a3, t3)  # d243_ov
    He -= 1./12 * e("jkam,mildcb,jkldcb->ia", oovo, a3, t3)  # d244_ov
    He -= 1./4 * e("lmdj,jikbac,lmidba->kc", oovo, a3, t3)  # d245_ov
    HHee += 1./2 * e("ijkl,klba->ijba", oooo, a2)  # d246_oovv
    He -= 1./2 * e("lijk,jkab,lb->ia", oooo, a2, t1)  # d247_ov
    HHHeee += 1./2 * p("aab...", e("lmij,ijkcba->lmkcba", oooo, a3))  # d248_ooovvv
    HHee += 1./2 * p("ab..", e("mikl,kljcba,mc->ijba", oooo, a3, t1))  # d249_oovv
    He += 1./4 * e("imkl,kljacb,ia,mc->jb", oooo, a3, t1, t1)  # d250_ov
    He -= 1./4 * e("iklm,jlmbac,ijba->kc", oooo, a3, t2)  # d251_ov
    He += 1./8 * e("lmjk,jkicba,lmcb->ia", oooo, a3, t2)  # d252_ov
    return He, HHee, HHHeee


def eq_ip_s(oo, ov, oovo, oovv, r_ip1, t1):
    h = 0
    h -= e("ji,j->i", oo, r_ip1)  # d0_o
    h -= e("ja,j,ia->i", ov, r_ip1, t1)  # d1_o
    h -= e("kjba,j,ia,kb->i", oovv, r_ip1, t1, t1)  # d2_o
    h += e("ikaj,i,ka->j", oovo, r_ip1, t1)  # d3_o
    return h


def eq_ip_sd(oo, ov, vv, oooo, oovo, oovv, ovoo, ovvo, ovvv, r_ip1, r_ip2, t1, t2):
    h = hhE = 0
    hhE += e("ba,ija->ijb", vv, r_ip2)  # d0_oov
    h -= e("ij,i->j", oo, r_ip1)  # d1_o
    hhE += p("ab.", e("jk,jia->ika", oo, r_ip2))  # d2_oov
    h -= e("ia,i,ja->j", ov, r_ip1, t1)  # d3_o
    hhE -= e("kb,k,ijba->ija", ov, r_ip1, t2)  # d4_oov
    hhE -= e("kb,ijb,ka->ija", ov, r_ip2, t1)  # d5_oov
    hhE -= p("ab.", e("ia,ijb,ka->kjb", ov, r_ip2, t1))  # d6_oov
    h -= e("ia,ija->j", ov, r_ip2)  # d7_o
    hhE -= 1./2 * p("ab.", e("kcab,k,ia,jb->ijc", ovvv, r_ip1, t1, t1))  # d8_oov
    hhE -= 1./2 * e("kacb,k,ijcb->ija", ovvv, r_ip1, t2)  # d9_oov
    hhE += e("icab,jkb,ia->jkc", ovvv, r_ip2, t1)  # d10_oov
    hhE -= p("ab.", e("ibac,ikc,ja->jkb", ovvv, r_ip2, t1))  # d11_oov
    hhE += p("ab.", e("ibaj,i,ka->jkb", ovvo, r_ip1, t1))  # d12_oov
    hhE += p("ab.", e("kabi,kjb->ija", ovvo, r_ip2))  # d13_oov
    hhE -= e("kaij,k->ija", ovoo, r_ip1)  # d14_oov
    hhE -= 1./2 * p("ab.", e("jlac,l,ia,jb,kc->ikb", oovv, r_ip1, t1, t1, t1))  # d15_oov
    h -= e("ikab,k,ia,jb->j", oovv, r_ip1, t1, t1)  # d16_o
    hhE += p("ab.", e("klca,l,ia,kjcb->jib", oovv, r_ip1, t1, t2))  # d17_oov
    hhE += 1./2 * e("klba,k,lc,ijba->ijc", oovv, r_ip1, t1, t2)  # d18_oov
    hhE += e("ijab,j,ib,klac->klc", oovv, r_ip1, t1, t2)  # d19_oov
    h -= 1./2 * e("ijba,j,ikba->k", oovv, r_ip1, t2)  # d20_o
    hhE += e("libc,jkb,ia,lc->jka", oovv, r_ip2, t1, t1)  # d21_oov
    hhE += 1./2 * e("klac,ija,klcb->ijb", oovv, r_ip2, t2)  # d22_oov
    hhE -= p("ab.", e("lkbc,lia,kb,jc->ija", oovv, r_ip2, t1, t1))  # d23_oov
    hhE -= 1./2 * p("ab.", e("ijba,ikc,jlba->klc", oovv, r_ip2, t2))  # d24_oov
    hhE += p("ab.", e("jlac,jia,kc,lb->ikb", oovv, r_ip2, t1, t1))  # d25_oov
    h += e("kjba,ikb,ja->i", oovv, r_ip2, t1)  # d26_o
    hhE -= p("ab.", e("jlac,jia,lkcb->ikb", oovv, r_ip2, t2))  # d27_oov
    hhE += 1./4 * p("ab.", e("klac,klb,ia,jc->ijb", oovv, r_ip2, t1, t1))  # d28_oov
    hhE += 1./4 * e("klba,klc,ijba->ijc", oovv, r_ip2, t2)  # d29_oov
    h += 1./2 * e("ijab,ija,kb->k", oovv, r_ip2, t1)  # d30_o
    hhE += 1./2 * e("klcb,klc,ijba->ija", oovv, r_ip2, t2)  # d31_oov
    hhE += p("ab.", e("lkbi,k,jb,la->ija", oovo, r_ip1, t1, t1))  # d32_oov
    h += e("kiaj,k,ia->j", oovo, r_ip1, t1)  # d33_o
    hhE += p("ab.", e("ilbk,i,ljba->kja", oovo, r_ip1, t2))  # d34_oov
    hhE += p("ab.", e("jlbk,ija,lb->ika", oovo, r_ip2, t1))  # d35_oov
    hhE += p("ab.", e("klbj,kib,la->ija", oovo, r_ip2, t1))  # d36_oov
    hhE -= 1./2 * p("ab.", e("klbi,kla,jb->ija", oovo, r_ip2, t1))  # d37_oov
    h += 1./2 * e("ijak,ija->k", oovo, r_ip2)  # d38_o
    hhE -= e("lijk,i,la->jka", oooo, r_ip1, t1)  # d39_oov
    hhE += 1./2 * e("ijkl,ija->kla", oooo, r_ip2)  # d40_oov
    return h, hhE


def eq_ip_d(oo, ov, vv, oooo, oovo, oovv, ovvo, r_ip2, t2):
    h = hhE = 0
    hhE += e("ba,ija->ijb", vv, r_ip2)  # d0_oov
    hhE -= p("ab.", e("ij,ika->jka", oo, r_ip2))  # d1_oov
    h += e("ia,jia->j", ov, r_ip2)  # d2_o
    hhE += p("ab.", e("ibak,ija->kjb", ovvo, r_ip2))  # d3_oov
    hhE -= 1./2 * e("ijac,klc,ijab->klb", oovv, r_ip2, t2)  # d4_oov
    hhE -= 1./2 * p("ab.", e("ijba,jlc,ikba->klc", oovv, r_ip2, t2))  # d5_oov
    hhE += p("ab.", e("ijab,jlb,ikac->klc", oovv, r_ip2, t2))  # d6_oov
    hhE += 1./4 * e("ijcb,ija,klcb->kla", oovv, r_ip2, t2)  # d7_oov
    hhE += 1./2 * e("klbc,klc,ijab->ija", oovv, r_ip2, t2)  # d8_oov
    h += 1./2 * e("jkai,jka->i", oovo, r_ip2)  # d9_o
    hhE += 1./2 * e("klij,kla->ija", oooo, r_ip2)  # d10_oov
    return h, hhE


def eq_ip_sdt(oo, ov, vv, oooo, oovo, oovv, ovoo, ovvo, ovvv, vvvo, vvvv, r_ip1, r_ip2, r_ip3, t1, t2, t3):
    h = hhE = hhhEE = 0
    hhE += e("ba,ija->ijb", vv, r_ip2)  # d0_oov
    hhhEE += p("...ab", e("ca,jkiab->jkicb", vv, r_ip3))  # d1_ooovv
    h -= e("ij,i->j", oo, r_ip1)  # d2_o
    hhE += p("ab.", e("ij,ika->kja", oo, r_ip2))  # d3_oov
    hhhEE -= p("abb..", e("li,lkjba->ikjba", oo, r_ip3))  # d4_ooovv
    h -= e("ia,i,ja->j", ov, r_ip1, t1)  # d5_o
    hhE -= e("ib,i,jkba->jka", ov, r_ip1, t2)  # d6_oov
    hhhEE -= e("lc,l,jkicba->jkiba", ov, r_ip1, t3)  # d7_ooovv
    hhE -= e("ka,ija,kb->ijb", ov, r_ip2, t1)  # d8_oov
    hhhEE -= p("aab..", e("la,ija,lkcb->ijkcb", ov, r_ip2, t2))  # d9_ooovv
    hhE -= p("ab.", e("jb,ija,kb->ika", ov, r_ip2, t1))  # d10_oov
    hhhEE += p("abb..", p("...ab", e("ib,ija,klbc->jklac", ov, r_ip2, t2)))  # d11_ooovv
    h -= e("ia,ija->j", ov, r_ip2)  # d12_o
    hhhEE += p("...ab", e("ic,kljcb,ia->kljba", ov, r_ip3, t1))  # d13_ooovv
    hhhEE -= p("aab..", e("la,jklcb,ia->jkicb", ov, r_ip3, t1))  # d14_ooovv
    hhE -= e("ia,ikjab->kjb", ov, r_ip3)  # d15_oov
    hhhEE += p("abb..", e("bacd,jic,kd->kjiba", vvvv, r_ip2, t1))  # d16_ooovv
    hhhEE += 1./2 * e("badc,jkidc->jkiba", vvvv, r_ip3)  # d17_ooovv
    hhhEE += p("abb..", e("cbai,jka->ijkcb", vvvo, r_ip2))  # d18_ooovv
    hhE += 1./2 * p("ab.", e("icab,i,ja,kb->kjc", ovvv, r_ip1, t1, t1))  # d19_oov
    hhhEE -= p("aab..", p("...ab", e("lcda,l,kd,ijab->ijkcb", ovvv, r_ip1, t1, t2)))  # d20_ooovv
    hhE -= 1./2 * e("kacb,k,ijcb->ija", ovvv, r_ip1, t2)  # d21_oov
    hhhEE -= 1./2 * p("...ab", e("icba,i,kljbad->kljcd", ovvv, r_ip1, t3))  # d22_ooovv
    hhhEE -= p("abb..", p("...ab", e("iacb,jkb,lc,id->ljkad", ovvv, r_ip2, t1, t1)))  # d23_ooovv
    hhE += e("kabc,ijc,kb->ija", ovvv, r_ip2, t1)  # d24_oov
    hhhEE += p("abb..", p("...ab", e("lacd,jid,lkcb->kjiab", ovvv, r_ip2, t2)))  # d25_ooovv
    hhhEE -= 1./2 * p("abc..", p("...ab", e("idbc,ija,kb,lc->kljda", ovvv, r_ip2, t1, t1)))  # d26_ooovv
    hhhEE -= 1./2 * p("aab..", p("...ab", e("icba,ijd,klba->kljcd", ovvv, r_ip2, t2)))  # d27_ooovv
    hhE -= p("ab.", e("kbac,kjc,ia->ijb", ovvv, r_ip2, t1))  # d28_oov
    hhhEE -= p("aab..", p("...ab", e("ibad,ija,kldc->kljbc", ovvv, r_ip2, t2)))  # d29_ooovv
    hhhEE += p("...ab", e("ldca,jkiab,lc->jkidb", ovvv, r_ip3, t1))  # d30_ooovv
    hhhEE += 1./2 * p("...ab", e("ladc,jkidc,lb->jkiab", ovvv, r_ip3, t1))  # d31_ooovv
    hhhEE -= p("abb..", p("...ab", e("icda,ikjab,ld->lkjcb", ovvv, r_ip3, t1)))  # d32_ooovv
    hhE -= 1./2 * e("icba,ikjba->kjc", ovvv, r_ip3)  # d33_oov
    hhE += p("ab.", e("kabi,k,jb->ija", ovvo, r_ip1, t1))  # d34_oov
    hhhEE += p("abb..", p("...ab", e("ibcl,i,kjca->lkjba", ovvo, r_ip1, t2)))  # d35_ooovv
    hhhEE += p("abb..", p("...ab", e("lbci,kjc,la->ikjba", ovvo, r_ip2, t1)))  # d36_ooovv
    hhhEE += p("abc..", p("...ab", e("lbcj,lia,kc->jkiba", ovvo, r_ip2, t1)))  # d37_ooovv
    hhE += p("ab.", e("ibaj,ika->jkb", ovvo, r_ip2))  # d38_oov
    hhhEE += p("abb..", p("...ab", e("laci,lkjcb->ikjab", ovvo, r_ip3)))  # d39_ooovv
    hhE -= e("iajk,i->jka", ovoo, r_ip1)  # d40_oov
    hhhEE -= p("aab..", p("...ab", e("ibkl,ija->kljba", ovoo, r_ip2)))  # d41_ooovv
    hhE += 1./2 * p("ab.", e("lkcb,l,ka,jb,ic->ija", oovv, r_ip1, t1, t1, t1))  # d42_oov
    hhhEE -= 1./2 * p("abc..", e("ijab,i,kb,la,jmdc->lmkdc", oovv, r_ip1, t1, t1, t2))  # d43_ooovv
    hhhEE -= p("aab..", p("...ab", e("lmad,l,kd,mc,ijab->ijkcb", oovv, r_ip1, t1, t1, t2)))  # d44_ooovv
    h += e("ijab,i,kb,ja->k", oovv, r_ip1, t1, t1)  # d45_o
    hhE -= p("ab.", e("ilbc,i,kc,ljba->jka", oovv, r_ip1, t1, t2))  # d46_oov
    hhhEE += p("aab..", e("mlcd,m,kd,ijlbac->ijkba", oovv, r_ip1, t1, t3))  # d47_ooovv
    hhE += 1./2 * e("lkba,l,kc,ijba->ijc", oovv, r_ip1, t1, t2)  # d48_oov
    hhhEE -= 1./2 * p("...ab", e("mldc,m,lb,jkidca->jkiab", oovv, r_ip1, t1, t3))  # d49_ooovv
    hhE -= e("ijba,i,ja,klbc->klc", oovv, r_ip1, t1, t2)  # d50_oov
    hhhEE += e("ijba,i,jb,lmkadc->lmkdc", oovv, r_ip1, t1, t3)  # d51_ooovv
    hhhEE -= p("abb..", p("...ab", e("mjbd,m,ijab,lkdc->ilkac", oovv, r_ip1, t2, t2)))  # d52_ooovv
    hhhEE -= 1./2 * p("abb..", e("lmdc,l,jidc,kmba->kjiba", oovv, r_ip1, t2, t2))  # d53_ooovv
    h += 1./2 * e("kjba,k,jiba->i", oovv, r_ip1, t2)  # d54_o
    hhE += 1./2 * e("liba,l,ikjbac->kjc", oovv, r_ip1, t3)  # d55_oov
    hhhEE += 1./2 * p("aab..", p("...ab", e("lmdc,ijd,kc,la,mb->ijkab", oovv, r_ip2, t1, t1, t1)))  # d56_ooovv
    hhhEE += 1./2 * p("aab..", e("ijcd,klc,md,ijba->klmba", oovv, r_ip2, t1, t2))  # d57_ooovv
    hhE -= e("jibc,klc,ia,jb->kla", oovv, r_ip2, t1, t1)  # d58_oov
    hhhEE -= p("aab..", p("...ab", e("jmdb,kld,mc,ijab->kliac", oovv, r_ip2, t1, t2)))  # d59_ooovv
    hhhEE -= p("aab..", e("jiba,klb,ia,jmdc->klmdc", oovv, r_ip2, t1, t2))  # d60_ooovv
    hhE += 1./2 * e("klab,ija,klbc->ijc", oovv, r_ip2, t2)  # d61_oov
    hhhEE += 1./2 * p("aab..", e("ijab,kla,ijmbdc->klmdc", oovv, r_ip2, t3))  # d62_ooovv
    hhhEE += 1./2 * p("abc..", p("...ab", e("lmcd,lka,id,jc,mb->jkiab", oovv, r_ip2, t1, t1, t1)))  # d63_ooovv
    hhE -= p("ab.", e("ljcb,ija,kb,lc->ika", oovv, r_ip2, t1, t1))  # d64_oov
    hhhEE += p("abc..", p("...ab", e("miad,mkc,ld,ijab->kjlcb", oovv, r_ip2, t1, t2)))  # d65_ooovv
    hhhEE -= 1./2 * p("abb..", p("...ab", e("jkcb,jia,kd,mlcb->imlad", oovv, r_ip2, t1, t2)))  # d66_ooovv
    hhhEE += p("abb..", p("...ab", e("imdc,ija,mc,lkdb->jlkab", oovv, r_ip2, t1, t2)))  # d67_ooovv
    hhE += 1./2 * p("ab.", e("klcb,ika,ljcb->ija", oovv, r_ip2, t2))  # d68_oov
    hhhEE -= 1./2 * p("abb..", p("...ab", e("jmdc,jia,mlkdcb->ilkab", oovv, r_ip2, t3)))  # d69_ooovv
    hhE += p("ab.", e("ilab,ija,kb,lc->jkc", oovv, r_ip2, t1, t1))  # d70_oov
    hhhEE -= p("abc..", e("ljda,lmd,ia,jkcb->mkicb", oovv, r_ip2, t1, t2))  # d71_ooovv
    hhhEE -= p("abb..", p("...ab", e("lmcd,ilc,ma,kjbd->ikjba", oovv, r_ip2, t1, t2)))  # d72_ooovv
    h -= e("ikab,ija,kb->j", oovv, r_ip2, t1)  # d73_o
    hhE -= p("ab.", e("ikab,ija,klbc->jlc", oovv, r_ip2, t2))  # d74_oov
    hhhEE -= p("abb..", e("imad,ija,mlkdcb->jlkcb", oovv, r_ip2, t3))  # d75_ooovv
    hhE += 1./4 * p("ab.", e("klcb,kla,ib,jc->jia", oovv, r_ip2, t1, t1))  # d76_oov
    hhhEE += 1./2 * p("aab..", p("...ab", e("ijba,ijc,kb,lmad->lmkcd", oovv, r_ip2, t1, t2)))  # d77_ooovv
    hhE += 1./4 * e("klba,klc,ijba->ijc", oovv, r_ip2, t2)  # d78_oov
    hhhEE += 1./4 * p("...ab", e("ijba,ijc,lmkbad->lmkcd", oovv, r_ip2, t3))  # d79_ooovv
    h += 1./2 * e("jkba,jkb,ia->i", oovv, r_ip2, t1)  # d80_o
    hhE += 1./2 * e("ijab,ija,klbc->klc", oovv, r_ip2, t2)  # d81_oov
    hhhEE += 1./2 * e("lmcd,lmc,jkidba->jkiba", oovv, r_ip2, t3)  # d82_ooovv
    hhhEE -= p("...ab", e("lmad,jkiab,ld,mc->jkibc", oovv, r_ip3, t1, t1))  # d83_ooovv
    hhhEE += 1./2 * p("...ab", e("lmcd,jkiac,lmdb->jkiab", oovv, r_ip3, t2))  # d84_ooovv
    hhhEE += 1./4 * p("...ab", e("jiba,lmkba,ic,jd->lmkdc", oovv, r_ip3, t1, t1))  # d85_ooovv
    hhhEE += 1./4 * e("lmdc,jkidc,lmba->jkiba", oovv, r_ip3, t2)  # d86_ooovv
    hhhEE -= p("aab..", e("imad,klmcb,ia,jd->kljcb", oovv, r_ip3, t1, t1))  # d87_ooovv
    hhhEE += 1./2 * p("aab..", e("mldc,jkmba,lidc->jkiba", oovv, r_ip3, t2))  # d88_ooovv
    hhhEE += p("aab..", p("...ab", e("lmcd,mijcb,kd,la->ijkab", oovv, r_ip3, t1, t1)))  # d89_ooovv
    hhE -= e("klbc,kijba,lc->ija", oovv, r_ip3, t1)  # d90_oov
    hhhEE -= p("aab..", p("...ab", e("ijab,iklac,jmbd->klmcd", oovv, r_ip3, t2)))  # d91_ooovv
    hhE += 1./2 * e("ijba,iklba,jc->klc", oovv, r_ip3, t1)  # d92_oov
    hhhEE += 1./2 * p("aab..", e("kmba,kjiba,mldc->jildc", oovv, r_ip3, t2))  # d93_ooovv
    hhhEE += 1./4 * p("abc..", e("lmdc,lmiba,jc,kd->ikjba", oovv, r_ip3, t1, t1))  # d94_ooovv
    hhhEE += 1./4 * p("abb..", e("ijdc,ijkba,mldc->kmlba", oovv, r_ip3, t2))  # d95_ooovv
    hhE += 1./2 * p("ab.", e("lkca,jlkbc,ia->jib", oovv, r_ip3, t1))  # d96_oov
    hhhEE -= 1./2 * p("abb..", p("...ab", e("klca,klmcd,ijab->mijdb", oovv, r_ip3, t2)))  # d97_ooovv
    h += 1./4 * e("ijba,ijkba->k", oovv, r_ip3)  # d98_o
    hhE -= p("ab.", e("jibk,j,ia,lb->kla", oovo, r_ip1, t1, t1))  # d99_oov
    hhhEE -= p("abc..", e("ijak,i,la,jmcb->klmcb", oovo, r_ip1, t1, t2))  # d100_ooovv
    hhhEE -= p("abb..", p("...ab", e("ijck,i,ja,mlcb->kmlab", oovo, r_ip1, t1, t2)))  # d101_ooovv
    h -= e("ikaj,k,ia->j", oovo, r_ip1, t1)  # d102_o
    hhE -= p("ab.", e("ikaj,k,ilab->jlb", oovo, r_ip1, t2))  # d103_oov
    hhhEE -= p("abb..", e("ikaj,k,imlacb->jmlcb", oovo, r_ip1, t3))  # d104_ooovv
    hhhEE -= 1./2 * p("abb..", p("...ab", e("mlci,kjc,lb,ma->ikjba", oovo, r_ip2, t1, t1)))  # d105_ooovv
    hhhEE += 1./2 * p("abb..", e("lmak,ija,lmcb->kijcb", oovo, r_ip2, t2))  # d106_ooovv
    hhhEE -= p("abc..", p("...ab", e("jiak,ilb,ma,jc->klmcb", oovo, r_ip2, t1, t1)))  # d107_ooovv
    hhE += p("ab.", e("ilbk,ija,lb->kja", oovo, r_ip2, t1))  # d108_oov
    hhhEE += p("abc..", p("...ab", e("imck,ija,mlcb->kjlab", oovo, r_ip2, t2)))  # d109_ooovv
    hhE += p("ab.", e("jkai,kla,jb->ilb", oovo, r_ip2, t1))  # d110_oov
    hhhEE -= p("abc..", e("jkai,jma,klcb->imlcb", oovo, r_ip2, t2))  # d111_ooovv
    hhE -= 1./2 * p("ab.", e("klbi,kla,jb->ija", oovo, r_ip2, t1))  # d112_oov
    hhhEE -= 1./2 * p("abb..", p("...ab", e("jkai,jkb,mlac->imlbc", oovo, r_ip2, t2)))  # d113_ooovv
    h += 1./2 * e("ijak,ija->k", oovo, r_ip2)  # d114_o
    hhhEE += p("abb..", e("imcl,ikjba,mc->lkjba", oovo, r_ip3, t1))  # d115_ooovv
    hhhEE += p("abb..", p("...ab", e("micl,mkjcb,ia->lkjba", oovo, r_ip3, t1)))  # d116_ooovv
    hhhEE += 1./2 * p("abc..", e("jkai,jkmcb,la->imlcb", oovo, r_ip3, t1))  # d117_ooovv
    hhE += 1./2 * p("ab.", e("ijak,ijlab->klb", oovo, r_ip3))  # d118_oov
    hhE -= e("klij,l,ka->ija", oooo, r_ip1, t1)  # d119_oov
    hhhEE -= p("aab..", e("lmij,m,lkba->ijkba", oooo, r_ip1, t2))  # d120_ooovv
    hhhEE -= p("aab..", p("...ab", e("lijk,lmb,ia->jkmba", oooo, r_ip2, t1)))  # d121_ooovv
    hhE += 1./2 * e("klij,kla->ija", oooo, r_ip2)  # d122_oov
    hhhEE += 1./2 * p("aab..", e("lmij,lmkba->ijkba", oooo, r_ip3))  # d123_ooovv
    return h, hhE, hhhEE


def eq_ea_s(ov, vv, oovv, ovvv, r_ea1, t1):
    E = 0
    E += e("ab,b->a", vv, r_ea1)  # d0_v
    E -= e("ib,b,ia->a", ov, r_ea1, t1)  # d1_v
    E -= e("ibac,a,ic->b", ovvv, r_ea1, t1)  # d2_v
    E -= e("jica,a,ib,jc->b", oovv, r_ea1, t1, t1)  # d3_v
    return E


def eq_ea_sd(oo, ov, vv, oovo, oovv, ovvo, ovvv, vvvo, vvvv, r_ea1, r_ea2, t1, t2):
    E = hEE = 0
    E += e("ba,a->b", vv, r_ea1)  # d0_v
    hEE -= p(".ab", e("ca,iab->ibc", vv, r_ea2))  # d1_ovv
    hEE -= e("ij,iba->jba", oo, r_ea2)  # d2_ovv
    E -= e("ib,b,ia->a", ov, r_ea1, t1)  # d3_v
    hEE += e("jc,c,ijba->iba", ov, r_ea1, t2)  # d4_ovv
    hEE -= p(".ab", e("jb,iab,jc->iac", ov, r_ea2, t1))  # d5_ovv
    hEE -= e("ia,icb,ja->jcb", ov, r_ea2, t1)  # d6_ovv
    E -= e("ib,iba->a", ov, r_ea2)  # d7_v
    hEE += e("dcab,a,ib->idc", vvvv, r_ea1, t1)  # d8_ovv
    hEE += 1./2 * e("badc,idc->iba", vvvv, r_ea2)  # d9_ovv
    hEE += e("baci,c->iba", vvvo, r_ea1)  # d10_ovv
    hEE -= p(".ab", e("icba,a,jb,id->jcd", ovvv, r_ea1, t1, t1))  # d11_ovv
    E += e("icab,b,ia->c", ovvv, r_ea1, t1)  # d12_v
    hEE -= p(".ab", e("jcbd,d,ijab->iac", ovvv, r_ea1, t2))  # d13_ovv
    hEE += p(".ab", e("icda,jdb,ia->jbc", ovvv, r_ea2, t1))  # d14_ovv
    hEE += 1./2 * p(".ab", e("jcba,iba,jd->icd", ovvv, r_ea2, t1))  # d15_ovv
    hEE -= p(".ab", e("icad,iab,jd->jbc", ovvv, r_ea2, t1))  # d16_ovv
    E -= 1./2 * e("iacb,icb->a", ovvv, r_ea2)  # d17_v
    hEE += p(".ab", e("ibaj,a,ic->jbc", ovvo, r_ea1, t1))  # d18_ovv
    hEE += p(".ab", e("icaj,iab->jcb", ovvo, r_ea2))  # d19_ovv
    hEE += 1./2 * p(".ab", e("jkad,a,id,jc,kb->icb", oovv, r_ea1, t1, t1, t1))  # d20_ovv
    hEE += 1./2 * e("jkba,b,ia,jkdc->idc", oovv, r_ea1, t1, t2)  # d21_ovv
    E -= e("ijbc,c,ib,ja->a", oovv, r_ea1, t1, t1)  # d22_v
    hEE += p(".ab", e("jkcd,d,kb,jica->iab", oovv, r_ea1, t1, t2))  # d23_ovv
    hEE -= e("jida,d,ia,jkcb->kcb", oovv, r_ea1, t1, t2)  # d24_ovv
    E += 1./2 * e("ijca,c,ijab->b", oovv, r_ea1, t2)  # d25_v
    hEE -= p(".ab", e("jiab,kad,ic,jb->kdc", oovv, r_ea2, t1, t1))  # d26_ovv
    hEE -= 1./2 * p(".ab", e("jkad,iab,jkdc->ibc", oovv, r_ea2, t2))  # d27_ovv
    hEE += 1./4 * p(".ab", e("ijba,kba,jd,ic->kcd", oovv, r_ea2, t1, t1))  # d28_ovv
    hEE += 1./4 * e("ijba,kba,ijdc->kdc", oovv, r_ea2, t2)  # d29_ovv
    hEE += e("jkdc,jba,ic,kd->iba", oovv, r_ea2, t1, t1)  # d30_ovv
    hEE += 1./2 * e("ikdc,iba,kjdc->jba", oovv, r_ea2, t2)  # d31_ovv
    hEE += p(".ab", e("jkca,jcb,ia,kd->ibd", oovv, r_ea2, t1, t1))  # d32_ovv
    E -= e("ijac,iab,jc->b", oovv, r_ea2, t1)  # d33_v
    hEE -= p(".ab", e("ikbd,iba,kjdc->jac", oovv, r_ea2, t2))  # d34_ovv
    E += 1./2 * e("ijba,iba,jc->c", oovv, r_ea2, t1)  # d35_v
    hEE += 1./2 * e("ikba,iba,kjdc->jdc", oovv, r_ea2, t2)  # d36_ovv
    hEE += 1./2 * p(".ab", e("ikaj,a,ib,kc->jbc", oovo, r_ea1, t1, t1))  # d37_ovv
    hEE += 1./2 * e("ijck,c,ijba->kba", oovo, r_ea1, t2)  # d38_ovv
    hEE -= e("ikaj,kcb,ia->jcb", oovo, r_ea2, t1)  # d39_ovv
    hEE += p(".ab", e("ikcj,kcb,ia->jab", oovo, r_ea2, t1))  # d40_ovv
    return E, hEE


def eq_ea_d(oo, ov, vv, oovv, ovvo, ovvv, vvvv, r_ea2, t2):
    E = hEE = 0
    hEE -= p(".ab", e("bc,ica->iab", vv, r_ea2))  # d0_ovv
    hEE -= e("ij,iba->jba", oo, r_ea2)  # d1_ovv
    E -= e("ia,iab->b", ov, r_ea2)  # d2_v
    hEE += 1./2 * e("badc,idc->iba", vvvv, r_ea2)  # d3_ovv
    E -= 1./2 * e("icba,iba->c", ovvv, r_ea2)  # d4_v
    hEE += p(".ab", e("icaj,iab->jcb", ovvo, r_ea2))  # d5_ovv
    hEE += 1./2 * p(".ab", e("jkbd,iab,jkdc->iac", oovv, r_ea2, t2))  # d6_ovv
    hEE += 1./4 * e("ijba,kba,ijdc->kdc", oovv, r_ea2, t2)  # d7_ovv
    hEE += 1./2 * e("jiba,jdc,ikba->kdc", oovv, r_ea2, t2)  # d8_ovv
    hEE += p(".ab", e("kjdc,kad,jicb->iab", oovv, r_ea2, t2))  # d9_ovv
    hEE += 1./2 * e("ikba,iba,kjdc->jdc", oovv, r_ea2, t2)  # d10_ovv
    return E, hEE


def eq_ea_sdt(oo, ov, vv, oooo, oovo, oovv, ovoo, ovvo, ovvv, vvvo, vvvv, r_ea1, r_ea2, r_ea3, t1, t2, t3):
    E = hEE = hhEEE = 0
    E += e("ab,b->a", vv, r_ea1)  # d0_v
    hEE += p(".ab", e("bc,iac->iab", vv, r_ea2))  # d1_ovv
    hhEEE += p("..aab", e("da,ijacb->ijcbd", vv, r_ea3))  # d2_oovvv
    hEE -= e("ij,iba->jba", oo, r_ea2)  # d3_ovv
    hhEEE -= p("ab...", e("jk,jicba->kicba", oo, r_ea3))  # d4_oovvv
    E -= e("ia,a,ib->b", ov, r_ea1, t1)  # d5_v
    hEE -= e("ja,a,jicb->icb", ov, r_ea1, t2)  # d6_ovv
    hhEEE -= e("ka,a,kijdcb->ijdcb", ov, r_ea1, t3)  # d7_oovvv
    hEE -= p(".ab", e("ia,jac,ib->jbc", ov, r_ea2, t1))  # d8_ovv
    hhEEE -= p("ab...", p("..aab", e("ka,iab,kjdc->jidcb", ov, r_ea2, t2)))  # d9_oovvv
    hEE -= e("jc,jba,ic->iba", ov, r_ea2, t1)  # d10_ovv
    hhEEE -= p("..abb", e("ia,icb,jkad->jkdcb", ov, r_ea2, t2))  # d11_oovvv
    E -= e("ia,iab->b", ov, r_ea2)  # d12_v
    hhEEE -= p("..abb", e("ka,ijacb,kd->ijdcb", ov, r_ea3, t1))  # d13_oovvv
    hhEEE -= p("ab...", e("ka,kjdcb,ia->ijdcb", ov, r_ea3, t1))  # d14_oovvv
    hEE -= e("ia,ijacb->jcb", ov, r_ea3)  # d15_ovv
    hEE -= e("cbda,a,id->icb", vvvv, r_ea1, t1)  # d16_ovv
    hhEEE -= p("..abb", e("bacd,d,ijce->ijeba", vvvv, r_ea1, t2))  # d17_oovvv
    hhEEE -= p("ab...", p("..aab", e("dcab,jbe,ia->ijdce", vvvv, r_ea2, t1)))  # d18_oovvv
    hEE += 1./2 * e("dcba,iba->idc", vvvv, r_ea2)  # d19_ovv
    hhEEE += 1./2 * p("..aab", e("edba,ijbac->ijedc", vvvv, r_ea3))  # d20_oovvv
    hEE += e("cbai,a->icb", vvvo, r_ea1)  # d21_ovv
    hhEEE += p("ab...", p("..aab", e("dcaj,iab->jidcb", vvvo, r_ea2)))  # d22_oovvv
    hEE -= p(".ab", e("jbad,a,id,jc->icb", ovvv, r_ea1, t1, t1))  # d23_ovv
    hhEEE -= p("ab...", p("..aab", e("ibca,c,ja,iked->kjedb", ovvv, r_ea1, t1, t2)))  # d24_oovvv
    hhEEE -= p("..abc", e("kcae,e,kd,ijab->ijbcd", ovvv, r_ea1, t1, t2))  # d25_oovvv
    E -= e("iabc,b,ic->a", ovvv, r_ea1, t1)  # d26_v
    hEE -= p(".ab", e("jcad,a,jidb->icb", ovvv, r_ea1, t2))  # d27_ovv
    hhEEE -= p("..abb", e("kced,e,kijdba->ijcba", ovvv, r_ea1, t3))  # d28_oovvv
    hhEEE -= p("ab...", p("..abc", e("jdca,kce,ia,jb->kibed", ovvv, r_ea2, t1, t1)))  # d29_oovvv
    hEE += p(".ab", e("jdac,iab,jc->ibd", ovvv, r_ea2, t1))  # d30_ovv
    hhEEE += p("ab...", p("..abc", e("idea,kec,ijab->kjcdb", ovvv, r_ea2, t2)))  # d31_oovvv
    hEE += 1./2 * p(".ab", e("icba,jba,id->jcd", ovvv, r_ea2, t1))  # d32_ovv
    hhEEE += 1./2 * p("ab...", p("..abb", e("icba,jba,iked->jkced", ovvv, r_ea2, t2)))  # d33_oovvv
    hhEEE -= 1./2 * p("ab...", p("..aab", e("kdea,kcb,ia,je->jicbd", ovvv, r_ea2, t1, t1)))  # d34_oovvv
    hhEEE -= 1./2 * p("..aab", e("iacb,ied,jkcb->jkeda", ovvv, r_ea2, t2))  # d35_oovvv
    hEE += p(".ab", e("icbd,iab,jd->jac", ovvv, r_ea2, t1))  # d36_ovv
    hhEEE -= p("..abc", e("icbe,iba,jked->jkacd", ovvv, r_ea2, t2))  # d37_oovvv
    E -= 1./2 * e("icba,iba->c", ovvv, r_ea2)  # d38_v
    hhEEE -= p("..aab", e("ieda,jkdcb,ia->jkcbe", ovvv, r_ea3, t1))  # d39_oovvv
    hhEEE += 1./2 * p("..abc", e("keba,ijbac,kd->ijced", ovvv, r_ea3, t1))  # d40_oovvv
    hhEEE += p("ab...", p("..aab", e("kade,ikcbd,je->ijcba", ovvv, r_ea3, t1)))  # d41_oovvv
    hEE += 1./2 * p(".ab", e("idba,ijbac->jcd", ovvv, r_ea3))  # d42_ovv
    hEE += p(".ab", e("jabi,b,jc->iac", ovvo, r_ea1, t1))  # d43_ovv
    hhEEE += p("ab...", p("..abb", e("kbai,a,kjdc->ijbdc", ovvo, r_ea1, t2)))  # d44_oovvv
    hhEEE -= p("ab...", p("..abc", e("jabi,kbc,jd->ikacd", ovvo, r_ea2, t1)))  # d45_oovvv
    hhEEE += p("ab...", p("..abb", e("kdaj,kcb,ia->jidcb", ovvo, r_ea2, t1)))  # d46_oovvv
    hEE += p(".ab", e("ibaj,iac->jbc", ovvo, r_ea2))  # d47_ovv
    hhEEE += p("ab...", p("..abb", e("idak,ijacb->kjdcb", ovvo, r_ea3)))  # d48_oovvv
    hhEEE -= p("..abb", e("iajk,icb->jkacb", ovoo, r_ea2))  # d49_oovvv
    hEE -= 1./2 * p(".ab", e("kjad,d,ia,kb,jc->ibc", oovv, r_ea1, t1, t1, t1))  # d50_ovv
    hhEEE -= p("ab...", p("..abb", e("lied,d,ia,je,lkcb->kjacb", oovv, r_ea1, t1, t1, t2)))  # d51_oovvv
    hEE += 1./2 * e("jkcd,c,id,jkba->iba", oovv, r_ea1, t1, t2)  # d52_ovv
    hhEEE -= 1./2 * p("ab...", e("ijab,a,lb,ijkedc->kledc", oovv, r_ea1, t1, t3))  # d53_oovvv
    hhEEE -= 1./2 * p("..abc", e("jidc,c,ia,jb,klde->kleba", oovv, r_ea1, t1, t1, t2))  # d54_oovvv
    E -= e("ijba,b,ja,ic->c", oovv, r_ea1, t1, t1)  # d55_v
    hEE -= p(".ab", e("jkcd,d,ja,kicb->iba", oovv, r_ea1, t1, t2))  # d56_ovv
    hhEEE -= p("..aab", e("ijab,a,ie,jklbdc->kldce", oovv, r_ea1, t1, t3))  # d57_oovvv
    hEE += e("kjcd,c,jd,ikba->iba", oovv, r_ea1, t1, t2)  # d58_ovv
    hhEEE += e("liab,b,ia,ljkedc->jkedc", oovv, r_ea1, t1, t3)  # d59_oovvv
    hhEEE += 1./2 * p("..aab", e("ijed,e,ijba,kldc->klbac", oovv, r_ea1, t2, t2))  # d60_oovvv
    hhEEE -= p("ab...", p("..abb", e("ilde,e,ijba,lkdc->kjcba", oovv, r_ea1, t2, t2)))  # d61_oovvv
    E -= 1./2 * e("ijab,b,ijac->c", oovv, r_ea1, t2)  # d62_v
    hEE += 1./2 * e("jkdc,d,jkicba->iba", oovv, r_ea1, t3)  # d63_ovv
    hhEEE += 1./2 * p("ab...", p("..abc", e("klab,jbc,ia,kd,le->ijedc", oovv, r_ea2, t1, t1, t1)))  # d64_oovvv
    hhEEE -= 1./2 * p("ab...", p("..aab", e("klae,jeb,ia,kldc->ijdcb", oovv, r_ea2, t1, t2)))  # d65_oovvv
    hEE += p(".ab", e("jkcb,iba,jd,kc->ida", oovv, r_ea2, t1, t1))  # d66_ovv
    hhEEE += p("ab...", p("..abc", e("lkde,jec,lb,kida->ijbac", oovv, r_ea2, t1, t2)))  # d67_oovvv
    hhEEE -= p("ab...", p("..aab", e("ljea,iab,le,jkdc->kidcb", oovv, r_ea2, t1, t2)))  # d68_oovvv
    hEE += 1./2 * p(".ab", e("ijda,kdc,ijab->kbc", oovv, r_ea2, t2))  # d69_ovv
    hhEEE += 1./2 * p("ab...", p("..aab", e("ijba,lbe,ijkadc->kldce", oovv, r_ea2, t3)))  # d70_oovvv
    hEE += 1./4 * p(".ab", e("ikdc,jdc,ia,kb->jab", oovv, r_ea2, t1, t1))  # d71_ovv
    hhEEE += 1./2 * p("ab...", p("..aab", e("kled,jed,lc,kiba->ijbac", oovv, r_ea2, t1, t2)))  # d72_oovvv
    hEE += 1./4 * e("jkba,iba,jkdc->idc", oovv, r_ea2, t2)  # d73_ovv
    hhEEE -= 1./4 * p("ab...", e("ijba,lba,ijkedc->kledc", oovv, r_ea2, t3))  # d74_oovvv
    hhEEE += 1./2 * p("ab...", p("..abb", e("ljea,ldc,ia,jb,ke->kibdc", oovv, r_ea2, t1, t1, t1)))  # d75_oovvv
    hEE -= e("ikad,kcb,ia,jd->jcb", oovv, r_ea2, t1, t1)  # d76_ovv
    hhEEE -= p("ab...", p("..abb", e("iled,iba,kd,ljec->jkcba", oovv, r_ea2, t1, t2)))  # d77_oovvv
    hhEEE += 1./2 * p("..abb", e("klba,ked,lc,ijba->ijced", oovv, r_ea2, t1, t2))  # d78_oovvv
    hhEEE += p("..abb", e("lkde,kcb,le,ijda->ijacb", oovv, r_ea2, t1, t2))  # d79_oovvv
    hEE -= 1./2 * e("ijba,jdc,ikba->kdc", oovv, r_ea2, t2)  # d80_ovv
    hhEEE -= 1./2 * p("..abb", e("ilba,led,ikjbac->kjced", oovv, r_ea2, t3))  # d81_oovvv
    hEE += p(".ab", e("jiab,jbc,ka,id->kdc", oovv, r_ea2, t1, t1))  # d82_ovv
    hhEEE -= p("ab...", p("..aab", e("lkae,keb,ia,ljdc->jidcb", oovv, r_ea2, t1, t2)))  # d83_oovvv
    hhEEE -= p("..abc", e("lked,kda,lc,ijeb->ijbca", oovv, r_ea2, t1, t2))  # d84_oovvv
    E -= e("jiba,iac,jb->c", oovv, r_ea2, t1)  # d85_v
    hEE += p(".ab", e("jkcd,jcb,kida->iab", oovv, r_ea2, t2))  # d86_ovv
    hhEEE -= p("..aab", e("ijbc,iba,jklced->kleda", oovv, r_ea2, t3))  # d87_oovvv
    E -= 1./2 * e("jiba,iba,jc->c", oovv, r_ea2, t1)  # d88_v
    hEE += 1./2 * e("ijba,iba,jkdc->kdc", oovv, r_ea2, t2)  # d89_ovv
    hhEEE -= 1./2 * e("iled,led,ikjcba->kjcba", oovv, r_ea2, t3)  # d90_oovvv
    hhEEE -= p("..aab", e("ijeb,kledc,ia,jb->kldca", oovv, r_ea3, t1, t1))  # d91_oovvv
    hhEEE -= 1./2 * p("..aab", e("ijae,kledc,ijab->kldcb", oovv, r_ea3, t2))  # d92_oovvv
    hhEEE -= 1./4 * p("..abc", e("ijba,klbad,je,ic->klcde", oovv, r_ea3, t1, t1))  # d93_oovvv
    hhEEE += 1./4 * p("..abb", e("ijba,klbac,ijed->klced", oovv, r_ea3, t2))  # d94_oovvv
    hhEEE += p("ab...", e("klde,licba,je,kd->ijcba", oovv, r_ea3, t1, t1))  # d95_oovvv
    hhEEE += 1./2 * p("ab...", e("jkba,kledc,jiba->liedc", oovv, r_ea3, t2))  # d96_oovvv
    hhEEE += p("ab...", p("..abb", e("lked,kjdcb,ie,la->jiacb", oovv, r_ea3, t1, t1)))  # d97_oovvv
    hEE -= e("jiba,ikadc,jb->kdc", oovv, r_ea3, t1)  # d98_ovv
    hhEEE -= p("ab...", p("..aab", e("ijab,jkbdc,ilae->kldce", oovv, r_ea3, t2)))  # d99_oovvv
    hEE += 1./2 * p(".ab", e("kiba,ijbac,kd->jcd", oovv, r_ea3, t1))  # d100_ovv
    hhEEE += 1./2 * p("ab...", p("..abb", e("liba,ijbac,lked->jkced", oovv, r_ea3, t2)))  # d101_oovvv
    hhEEE += 1./4 * p("ab...", e("klab,kledc,ia,jb->ijedc", oovv, r_ea3, t1, t1))  # d102_oovvv
    hhEEE += 1./4 * e("kled,klcba,ijed->ijcba", oovv, r_ea3, t2)  # d103_oovvv
    hEE -= 1./2 * e("jkad,jkdcb,ia->icb", oovv, r_ea3, t1)  # d104_ovv
    hhEEE += 1./2 * p("..aab", e("ijad,ijacb,klde->klcbe", oovv, r_ea3, t2))  # d105_oovvv
    E += 1./4 * e("ijba,ijbac->c", oovv, r_ea3)  # d106_v
    hEE += 1./2 * p(".ab", e("kjai,a,jc,kb->ibc", oovo, r_ea1, t1, t1))  # d107_ovv
    hhEEE -= p("ab...", p("..aab", e("jlck,c,ld,jiba->kibad", oovo, r_ea1, t1, t2)))  # d108_oovvv
    hEE += 1./2 * e("jkci,c,jkba->iba", oovo, r_ea1, t2)  # d109_ovv
    hhEEE += 1./2 * p("ab...", e("ijdl,d,ijkcba->lkcba", oovo, r_ea1, t3))  # d110_oovvv
    hhEEE += 1./2 * p("ab...", p("..abc", e("ikaj,lad,ib,kc->jlbcd", oovo, r_ea2, t1, t1)))  # d111_oovvv
    hhEEE += 1./2 * p("ab...", p("..aab", e("kldi,jdc,klba->ijbac", oovo, r_ea2, t2)))  # d112_oovvv
    hhEEE -= p("ab...", p("..abb", e("ilck,iba,jc,ld->kjdba", oovo, r_ea2, t1, t1)))  # d113_oovvv
    hEE -= e("jkci,kba,jc->iba", oovo, r_ea2, t1)  # d114_ovv
    hhEEE += p("ab...", p("..abb", e("ildj,iba,lkdc->jkcba", oovo, r_ea2, t2)))  # d115_oovvv
    hEE += p(".ab", e("jkci,kcb,ja->iab", oovo, r_ea2, t1))  # d116_ovv
    hhEEE += p("ab...", p("..aab", e("ijak,iab,jldc->kldcb", oovo, r_ea2, t2)))  # d117_oovvv
    hhEEE += p("ab...", e("ildk,ijcba,ld->kjcba", oovo, r_ea3, t1))  # d118_oovvv
    hhEEE -= p("ab...", p("..aab", e("jiak,jlacb,id->klcbd", oovo, r_ea3, t1)))  # d119_oovvv
    hhEEE -= 1./2 * p("ab...", e("ijak,ijdcb,la->kldcb", oovo, r_ea3, t1))  # d120_oovvv
    hEE += 1./2 * e("ijak,ijacb->kcb", oovo, r_ea3)  # d121_ovv
    hhEEE -= p("..abb", e("klij,lcb,ka->ijacb", oooo, r_ea2, t1))  # d122_oovvv
    hhEEE += 1./2 * e("klij,klcba->ijcba", oooo, r_ea3)  # d123_oovvv
    return E, hEE, hhEEE
