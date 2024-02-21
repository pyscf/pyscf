import copy
from pyscf import gto
from pyscf.lib import logger
from pyscf import lib
from pyscf import mp

import numpy
from numpy import einsum
import scipy.linalg
import scipy.sparse.linalg

def _vec_to_rotmat(vec, nocc=0, nvir=0):
    assert nocc > 0 and nvir > 0, f"The number of occupied and virtual orbitals must both be positive)"

    nocc = int(nocc)
    nvir = int(nvir)
    nocc_param = int(nocc * (nocc - 1) // 2)
    nvir_param = int(nvir * (nvir - 1) // 2)
    assert len(vec) == nocc_param + \
               nvir_param, f"Invalid Vector length, received {len(vec)}, required {nocc_param + nvir_param}"

    occ_triu_idx = numpy.triu_indices(nocc, k=1)
    vir_triu_idx = numpy.triu_indices(nvir, k=1)
    occ_mat = numpy.zeros((nocc, nocc))
    vir_mat = numpy.zeros((nvir, nvir))
    occ_mat[occ_triu_idx] = vec[:nocc_param]
    vir_mat[vir_triu_idx] = vec[nocc_param:]

    mat = numpy.zeros((nocc+nvir, nocc+nvir))
    mat[:nocc,:nocc] = occ_mat
    mat[nocc:,nocc:] = vir_mat
    mat -= mat.T
    return mat

def _extract_eri_delta(mp, mo_coeff, mo_energy, n_occ, n_vir):
    # e_ijab = e_i + e_j - e_a - e_b
    e_ijab = mo_energy[:n_occ,None,None,None] + mo_energy[None,:n_occ,None,None] - \
        mo_energy[None,None,n_occ:,None] - mo_energy[None,None,None,n_occ:]

    D_vvoo = einsum('ijab->abij', -1/e_ijab)

    eris = mp.ao2mo(mo_coeff).ovov
    v_oovv = None
    if isinstance(eris, numpy.ndarray) and eris.ndim == 4:
        v_oovv = eris
    else:
        v_oovv = numpy.zeros((n_occ, n_occ, n_vir, n_vir))
        for i in range(n_occ):
            gi = numpy.asarray(eris[i*n_vir:(i+1)*n_vir])
            v_oovv[i] = gi.reshape(n_vir, n_occ, n_vir).transpose(1,0,2)

    return v_oovv, D_vvoo

def _wrap_gradient(g_oo, g_vv, n_occ, n_vir):
    n_occ_p = int(n_occ * (n_occ - 1) // 2)
    n_vir_p = int(n_vir * (n_vir - 1) // 2)
    n_param = n_occ_p + n_vir_p

    occ_idx = numpy.array(numpy.triu_indices(n_occ, k=1))
    vir_idx = numpy.array(numpy.triu_indices(n_vir, k=1))
    grad = numpy.zeros(n_param)

    grad_o = g_oo[occ_idx[0], occ_idx[1]]
    grad_v = g_vv[vir_idx[0], vir_idx[1]]
    grad[:n_occ_p] = grad_o
    grad[n_occ_p:] = grad_v

    return grad

def _wrap_hessian(h_oooo, h_oovv, h_vvvv, n_occ, n_vir):
    n_occ_p = int(n_occ * (n_occ - 1) // 2)
    n_vir_p = int(n_vir * (n_vir - 1) // 2)
    n_param = n_occ_p + n_vir_p

    h_oo = numpy.zeros((n_occ_p, n_occ_p))
    h_ov = numpy.zeros((n_occ_p, n_vir_p))
    h_vv = numpy.zeros((n_vir_p, n_vir_p))
    hess = numpy.zeros((n_param, n_param))
    occ_idx = numpy.array(numpy.triu_indices(n_occ, k=1))
    vir_idx = numpy.array(numpy.triu_indices(n_vir, k=1))

    for hi, oidx0 in enumerate(occ_idx.T):
        for hj, oidx1 in enumerate(occ_idx.T):
            h_oo[hi,hj] = h_oooo[oidx0[0],oidx0[1],oidx1[0],oidx1[1]]
        for ha, vidx1 in enumerate(vir_idx.T):
            h_ov[hi,ha] = h_oovv[oidx0[0],oidx0[1],vidx1[0],vidx1[1]]

    for ha, vidx0 in enumerate(vir_idx.T):
        for hb, vidx1 in enumerate(vir_idx.T):
            h_vv[ha,hb] = h_vvvv[vidx0[0],vidx0[1],vidx1[0],vidx1[1]]

    hess[:n_occ_p,:n_occ_p] = h_oo
    hess[n_occ_p:,n_occ_p:] = h_vv
    hess[:n_occ_p,n_occ_p:] = h_ov
    hess[n_occ_p:,:n_occ_p] = h_ov.T

    return hess

def _eri_sum_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    g_oovv = _extract_eri_delta(ordmp._mp2, mo_coeff, mo_energy, n_occ, n_vir)[0]

    d_oo = numpy.eye(n_occ)
    d_vv = numpy.eye(n_vir)

    # ====================================================================================================
    # Occupied Gradient:
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij}) to:
    itmd0 = (+ 1 * einsum('ac,bd,im,jk,ln,mncd->ij', d_vv, d_vv, d_oo, d_oo, d_oo, g_oovv)   # N^10: O^6V^4 / N^4: O^2V^2
    + 1 * einsum('ac,bd,in,jl,km,mncd->ij', d_vv, d_vv, d_oo, d_oo, d_oo, g_oovv))  # N^10: O^6V^4 / N^4: O^2V^2

    itmd0 = itmd0 - einsum('ij->ji', itmd0)
    # ====================================================================================================
    # Virtual Gradient:
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ab}) to:
    itmd1 = (+ 1 * einsum('ae,bc,df,ik,jl,klef->ab', d_vv, d_vv, d_vv, d_oo, d_oo, g_oovv)   # N^10: O^4V^6 / N^4: O^2V^2
    + 1 * einsum('af,bd,ce,ik,jl,klef->ab', d_vv, d_vv, d_vv, d_oo, d_oo, g_oovv))  # N^10: O^4V^6 / N^4: O^2V^2

    itmd1 = itmd1 - einsum('ab->ba', itmd1)

    return _wrap_gradient(itmd0, itmd1, n_occ, n_vir)

def _eri_sum_hess(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    v_oovv = _extract_eri_delta(ordmp._mp2, mo_coeff, mo_energy, n_occ, n_vir)[0]

    d_oo = numpy.eye(n_occ)
    d_vv = numpy.eye(n_vir)
    # ====================================================================================================
    # Occupied-Occupied Hessian
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij} - P_{kl} + P_{ij}P_{kl} + P_{ik}P_{jl} + P_{il}P_{jk}
    # - P_{ij}P_{ik}P_{jl} - P_{ij}P_{il}P_{jk}) to:
    itmd0 = (+ 1 * einsum('im,kn,jlab->ijkl', d_oo, d_oo, v_oovv)   # N^8: O^6V^2 / N^4: O^2V^2
    + 0.5 * einsum('il,jm,knab->ijkl', d_oo, d_oo, v_oovv)   # N^8: O^6V^2 / N^4: O^2V^2
    + 0.5 * einsum('il,jn,mkab->ijkl', d_oo, d_oo, v_oovv))  # N^8: O^6V^2 / N^4: O^2V^2

    h_oooo = itmd0 - einsum('ijkl->jikl',
    itmd0) - einsum('ijkl->ijlk',
    itmd0) + einsum('ijkl->jilk',
    itmd0) + einsum('ijkl->klij',
    itmd0) + einsum('ijkl->lkji',
    itmd0) - einsum('ijkl->lkij',
    itmd0) - einsum('ijkl->klji',
     itmd0)
    del itmd0

    # ====================================================================================================
    # Occupied-Virtual Hessian
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{cd} - P_{ij} + P_{ij}P_{cd}) to:
    itmd2 = (+ 1 * einsum('ac,ik,jldb->ijcd', d_vv, d_oo, v_oovv)   # N^8: O^4V^4 / N^4: O^2V^2
    + 1 * einsum('ac,il,kjdb->ijcd', d_vv, d_oo, v_oovv)   # N^8: O^4V^4 / N^4: O^2V^2
    + 1 * einsum('bc,ik,jlad->ijcd', d_vv, d_oo, v_oovv)   # N^8: O^4V^4 / N^4: O^2V^2
    + 1 * einsum('bc,il,kjad->ijcd', d_vv, d_oo, v_oovv))  # N^8: O^4V^4 / N^4: O^2V^2

    h_oovv = itmd2 - einsum('ijcd->ijdc', itmd2) - einsum('ijcd->jicd', itmd2) + einsum('ijcd->jidc', itmd2)
    del itmd2
    # ====================================================================================================
    # Virtual-Virtual Hessian
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ab} - P_{cd} + P_{ab}P_{cd} + P_{ac}P_{bd} + P_{ad}P_{bc}
    # - P_{ab}P_{ac}P_{bd} - P_{ab}P_{ad}P_{bc}) to:
    itmd3 = (+ 1 * einsum('ae,cf,ijbd->abcd', d_vv, d_vv, v_oovv)   # N^8: O^2V^6 / N^4: V^4
    + 0.5 * einsum('ad,be,ijcf->abcd', d_vv, d_vv, v_oovv)   # N^8: O^2V^6 / N^4: V^4
    + 0.5 * einsum('ad,bf,ijec->abcd', d_vv, d_vv, v_oovv))  # N^8: O^2V^6 / N^4: V^4

    h_vvvv = itmd3 - einsum('abcd->bacd',
    itmd3) - einsum('abcd->abdc',
    itmd3) + einsum('abcd->badc',
    itmd3) + einsum('abcd->cdab',
    itmd3) + einsum('abcd->dcba',
    itmd3) - einsum('abcd->dcab',
    itmd3) - einsum('abcd->cdba',
     itmd3)
    del itmd3

    return _wrap_hessian(h_oooo, h_oovv, h_vvvv, n_occ, n_vir)

def _max_energy_grad(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    v_oovv, D_vvoo = _extract_eri_delta(ordmp._mp2, mo_coeff, mo_energy, n_occ, n_vir)

    # ====================================================================================================
    # Occupied Gradient:
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij}) to:
    itmd0 = (- 4 * einsum('ikab,jkab,abjk->ij', v_oovv, v_oovv, D_vvoo)   # N^5: O^3V^2 / N^4: O^2V^2
    - 4 * einsum('kiab,kjab,abjk->ij', v_oovv, v_oovv, D_vvoo)   # N^5: O^3V^2 / N^4: O^2V^2
    - 2 * einsum('ikab,jkba,abik->ij', v_oovv, v_oovv, D_vvoo)   # N^5: O^3V^2 / N^4: O^2V^2
    - 2 * einsum('kiab,kjba,abik->ij', v_oovv, v_oovv, D_vvoo))  # N^5: O^3V^2 / N^4: O^2V^2

    itmd0 = itmd0 - einsum('ij->ji', itmd0)

    # ====================================================================================================
    # Virtual Gradient:
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ab}) to:
    itmd1 = (- 4 * einsum('ijac,ijbc,bcij->ab', v_oovv, v_oovv, D_vvoo)   # N^5: O^2V^3 / N^4: O^2V^2
    - 4 * einsum('ijca,ijcb,bcij->ab', v_oovv, v_oovv, D_vvoo)   # N^5: O^2V^3 / N^4: O^2V^2
    - 2 * einsum('ijac,ijcb,acij->ab', v_oovv, v_oovv, D_vvoo)   # N^5: O^2V^3 / N^4: O^2V^2
    - 2 * einsum('ijbc,ijca,acij->ab', v_oovv, v_oovv, D_vvoo))  # N^5: O^2V^3 / N^4: O^2V^2

    itmd1 = itmd1 - einsum('ab->ba', itmd1)

    return _wrap_gradient(itmd0, itmd1, n_occ, n_vir)

def _max_energy_hess(ordmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
    v_oovv, D_vvoo = _extract_eri_delta(ordmp._mp2, mo_coeff, mo_energy, n_occ, n_vir)

    d_oo = numpy.eye(n_occ)
    d_vv = numpy.eye(n_vir)

    # ====================================================================================================
    # Occupied-Occupied Hessian
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij} - P_{kl} + P_{ij}P_{kl} + P_{ik}P_{jl} + P_{il}P_{jk}
    # - P_{ij}P_{ik}P_{jl} - P_{ij}P_{il}P_{jk}) to:
    itmd0 = (- 4 * einsum('ikab,jlab,abik->ijkl', v_oovv, v_oovv, D_vvoo)   # N^6: O^4V^2 / N^4: O^2V^2
    - 4 * einsum('ilab,jkab,abik->ijkl', v_oovv, v_oovv, D_vvoo)   # N^6: O^4V^2 / N^4: O^2V^2
    - 2 * einsum('ikab,jlba,abil->ijkl', v_oovv, v_oovv, D_vvoo)   # N^6: O^4V^2 / N^4: O^2V^2
    - 2 * einsum('ilab,jkba,abil->ijkl', v_oovv, v_oovv, D_vvoo)   # N^6: O^4V^2 / N^4: O^2V^2
    + 1 * einsum('il,jmab,kmba,abjm->ijkl', d_oo, v_oovv, v_oovv, D_vvoo)   # N^7: O^5V^2 / N^4: O^2V^2
    + 1 * einsum('il,mjab,mkba,abjm->ijkl', d_oo, v_oovv, v_oovv, D_vvoo)   # N^7: O^5V^2 / N^4: O^2V^2
    + 2 * einsum('ik,jmab,lmab,abjm->ijkl', d_oo, v_oovv, v_oovv, D_vvoo)   # N^7: O^5V^2 / N^4: O^2V^2
    + 2 * einsum('ik,mjab,mlab,abjm->ijkl', d_oo, v_oovv, v_oovv, D_vvoo))  # N^7: O^5V^2 / N^4: O^2V^2

    itmd0 = itmd0 - einsum('ijkl->jikl',
    itmd0) - einsum('ijkl->ijlk',
    itmd0) + einsum('ijkl->jilk',
    itmd0) + einsum('ijkl->klij',
    itmd0) + einsum('ijkl->lkji',
    itmd0) - einsum('ijkl->lkij',
    itmd0) - einsum('ijkl->klji',
     itmd0)

    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ij} - P_{kl} + P_{ij}P_{kl}) to:
    itmd1 = (- 4 * einsum('ik,jmab,lmab,abim->ijkl', d_oo, v_oovv, v_oovv, D_vvoo)   # N^7: O^5V^2 / N^4: O^2V^2
    - 4 * einsum('ik,mjab,mlab,abim->ijkl', d_oo, v_oovv, v_oovv, D_vvoo)   # N^7: O^5V^2 / N^4: O^2V^2
    - 2 * einsum('il,jmab,kmba,abim->ijkl', d_oo, v_oovv, v_oovv, D_vvoo)   # N^7: O^5V^2 / N^4: O^2V^2
    - 2 * einsum('il,mjab,mkba,abim->ijkl', d_oo, v_oovv, v_oovv, D_vvoo))  # N^7: O^5V^2 / N^4: O^2V^2

    itmd1 = itmd1 - einsum('ijkl->jikl', itmd1) - einsum('ijkl->ijlk', itmd1) + einsum('ijkl->jilk', itmd1)

    h_oooo = itmd0 + itmd1
    del itmd0, itmd1

    # ====================================================================================================
    # Occupied-Virtual Hessian
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{cd} - P_{ij} + P_{ij}P_{cd}) to:
    itmd2 = (- 4 * einsum('ikac,jkad,acik->ijcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikad,jkac,acik->ijcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikca,jkda,acik->ijcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('ikda,jkca,acik->ijcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('kiac,kjad,acik->ijcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('kiad,kjac,acik->ijcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('kica,kjda,acik->ijcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^3V^3 / N^4: O^2V^2
    - 4 * einsum('kida,kjca,acik->ijcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,jkda,acjk->ijcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikac,jkda,adik->ijcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikad,jkca,acjk->ijcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('ikad,jkca,adik->ijcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('kiac,kjda,acjk->ijcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('kiac,kjda,adik->ijcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('kiad,kjca,acjk->ijcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^3V^3 / N^4: O^2V^2
    - 2 * einsum('kiad,kjca,adik->ijcd', v_oovv, v_oovv, D_vvoo))  # N^6: O^3V^3 / N^4: O^2V^2

    h_oovv = itmd2 - einsum('ijcd->ijdc', itmd2) - einsum('ijcd->jicd', itmd2) + einsum('ijcd->jidc', itmd2)
    del itmd2

    # ====================================================================================================
    # Virtual-Virtual Hessian
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ab} - P_{cd} + P_{ab}P_{cd} + P_{ac}P_{bd} + P_{ad}P_{bc}
    # - P_{ab}P_{ac}P_{bd} - P_{ab}P_{ad}P_{bc}) to:
    itmd3 = (- 4 * einsum('ijad,ijbc,acij->abcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^2V^4 / N^4: V^4
    + 1 * einsum('ad,ijbe,ijec,beij->abcd', d_vv, v_oovv, v_oovv, D_vvoo)   # N^7: O^2V^5 / N^4: V^4
    + 1 * einsum('ad,ijbe,ijec,ceij->abcd', d_vv, v_oovv, v_oovv, D_vvoo)   # N^7: O^2V^5 / N^4: V^4
    + 2 * einsum('ac,ijbe,ijde,beij->abcd', d_vv, v_oovv, v_oovv, D_vvoo)   # N^7: O^2V^5 / N^4: V^4
    + 2 * einsum('ac,ijeb,ijed,beij->abcd', d_vv, v_oovv, v_oovv, D_vvoo))  # N^7: O^2V^5 / N^4: V^4

    itmd3 = itmd3 - einsum('abcd->bacd',
    itmd3) - einsum('abcd->abdc',
    itmd3) + einsum('abcd->badc',
    itmd3) + einsum('abcd->cdab',
    itmd3) + einsum('abcd->dcba',
    itmd3) - einsum('abcd->dcab',
    itmd3) - einsum('abcd->cdba',
     itmd3)

    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 + P_{ab}P_{cd} + P_{ac}P_{bd} + P_{ad}P_{bc}) to:
    itmd4 = (- 2 * einsum('ijac,ijbd,acij->abcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^2V^4 / N^4: V^4
    + 2 * einsum('ijad,ijcb,adij->abcd', v_oovv, v_oovv, D_vvoo))  # N^6: O^2V^4 / N^4: V^4

    itmd4 = itmd4 + einsum('abcd->badc', itmd4) + einsum('abcd->cdab', itmd4) + einsum('abcd->dcba', itmd4)

    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    # Apply (1 - P_{ab} - P_{cd} + P_{ab}P_{cd}) to:
    itmd5 = (- 2 * einsum('ijac,ijdb,adij->abcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^2V^4 / N^4: V^4
    - 2 * einsum('ijac,ijdb,bcij->abcd', v_oovv, v_oovv, D_vvoo)   # N^6: O^2V^4 / N^4: V^4
    - 4 * einsum('ac,ijbe,ijde,aeij->abcd', d_vv, v_oovv, v_oovv, D_vvoo)   # N^7: O^2V^5 / N^4: V^4
    - 4 * einsum('ac,ijeb,ijed,aeij->abcd', d_vv, v_oovv, v_oovv, D_vvoo)   # N^7: O^2V^5 / N^4: V^4
    - 2 * einsum('ad,ijbe,ijec,aeij->abcd', d_vv, v_oovv, v_oovv, D_vvoo)   # N^7: O^2V^5 / N^4: V^4
    - 2 * einsum('ad,ijce,ijeb,aeij->abcd', d_vv, v_oovv, v_oovv, D_vvoo))  # N^7: O^2V^5 / N^4: V^4

    itmd5 = itmd5 - einsum('abcd->bacd', itmd5) - einsum('abcd->abdc', itmd5) + einsum('abcd->badc', itmd5)

    h_vvvv = itmd3 + itmd4 + itmd5
    del itmd3, itmd4, itmd5

    return _wrap_hessian(h_oooo, h_oovv, h_vvvv, n_occ, n_vir)

class ORDMP2(lib.StreamObject):
    """
    Orbital rotation dependent Möller-Plesset Theory that utilises an MP backend.
    """

    def __init__(self, mf, optimality='max_energy'):

        self._mf = mf
        self._mp2 = mp.MP2(mf)
        self._mp2.verbose = 0
        self.u_mat = None
        self.opt = optimality
        self.mo_coeff = None

        # Internal variables
        self.max_cycles = 200
        self.stepsize = 5.0

    def make_rdm1(self, t2=None, eris=None, ao_repr=False):
        return self._mp2.make_rdm1(t2, eris, ao_repr)

    def make_rdm2(self, t2=None, eris=None, ao_repr=False):
        return self._mp2.make_rdm2(t2, eris, ao_repr)

    @property
    def e_corr(self):
        return self._mp2.e_corr

    @property
    def t2(self):
        return self._mp2.t2

    @property
    def e_tot(self):
        return self._mp2.e_tot

    def criterion(self, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
        if n_occ <= 0 or n_vir <= 0:
            n_occ = numpy.sum(mo_occ) // 2
            n_vir = len(mo_occ) - n_occ
        n_param = int((n_occ * (n_occ - 1) + n_vir * (n_vir - 1)) // 2)

        if vec is None:
            vec = numpy.zeros(n_param)

        if self.opt == 'max_energy':
            return self.max_energy(vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        elif self.opt == 'min_energy':
            return self.min_energy(vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        elif self.opt == 'eri_sum':
            return self.eri_sum(vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        else:
            raise NotImplementedError(f"The optimality criterion {opt} is not available")

    def crit_grad(self, mo_coeff, mo_energy, mo_occ, n_occ=0, n_vir=0, force_numeric=False, epsilon=10**-5):
        if force_numeric:
            return self.crit_grad_num(mo_coeff, mo_energy, mo_occ, epsilon, n_occ, n_vir)

        if n_occ <= 0 or n_vir <= 0:
            n_occ = int(numpy.sum(mo_occ) // 2)
            n_vir = int(len(mo_occ) - n_occ)

        if self.opt == 'max_energy':
            return _max_energy_grad(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        if self.opt == 'min_energy':
            return - _max_energy_grad(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        if self.opt == 'eri_sum':
            return _eri_sum_grad(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

        return self.crit_grad_num(self, mo_coeff, mo_energy, mo_occ, epsilon, n_occ, n_vir)

    def crit_hess(self, mo_coeff, mo_energy, mo_occ, n_occ=0, n_vir=0, force_numeric=False, epsilon=10**-5):
        if force_numeric:
            return self.crit_hess_num(mo_coeff, mo_energy, mo_occ, epsilon, n_occ, n_vir)
        if n_occ <= 0 or n_vir <= 0:
            n_occ = int(numpy.sum(mo_occ) // 2)
            n_vir = int(len(mo_occ) - n_occ)

        if self.opt == 'max_energy':
            return _max_energy_hess(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        if self.opt == 'min_energy':
            return - _max_energy_hess(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        if self.opt == 'eri_sum':
            return _eri_sum_hess(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

        return self.crit_hess_num(mo_coeff, mo_energy, mo_occ, epsilon, n_occ, n_vir)

    def crit_grad_num(self, mo_coeff, mo_energy, mo_occ, epsilon=10**-5, n_occ=0, n_vir=0):
        if n_occ <= 0 or n_vir <= 0:
            n_occ = numpy.sum(mo_occ) // 2
            n_vir = len(mo_occ) - n_occ
        n_param = int((n_occ * (n_occ - 1) + n_vir * (n_vir - 1)) // 2)

        grad = numpy.zeros(n_param)
        for i in range(n_param):
            vp = numpy.zeros(n_param)
            vm = numpy.zeros(n_param)
            vp[i] += epsilon
            vm[i] -= epsilon
            grad[i] = (self.criterion(vp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir) -
                       self.criterion(vm, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)) / (2 * epsilon)

        return grad

    def crit_hess_num(self, mo_coeff, mo_energy, mo_occ, epsilon=10**-8, n_occ=0, n_vir=0):
        if n_occ <= 0 or n_vir <= 0:
            n_occ = numpy.sum(mo_occ) // 2
            n_vir = len(mo_occ) - n_occ
        n_param = int((n_occ * (n_occ - 1) + n_vir * (n_vir - 1)) // 2)

        rel_indices = numpy.array(numpy.triu_indices(n_param, k=0))
        hess = numpy.zeros((n_param, n_param))
        for p in rel_indices.T:
            i, j = p[0], p[1]
            #print(f"Calculating Hessian: {i}, {j}")
            vpp = numpy.zeros(n_param)
            vpm = numpy.zeros(n_param)
            vmp = numpy.zeros(n_param)
            vmm = numpy.zeros(n_param)
            vpp[i] += epsilon
            vpp[j] += epsilon
            vpm[i] += epsilon
            vpm[j] -= epsilon
            vmp[i] -= epsilon
            vmp[j] += epsilon
            vmm[i] -= epsilon
            vmm[j] -= epsilon

            c_pp = self.criterion(vpp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
            c_pm = self.criterion(vpm, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
            c_mp = self.criterion(vmp, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
            c_mm = self.criterion(vmm, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

            hess[i,j] = (c_pp + c_mm - c_pm - c_mp) / (4 * epsilon**2)

        return hess + hess.T - numpy.diag(numpy.diag(hess))

    def gen_new_vec(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
        self._mp2.kernel(mo_coeff=mo_coeff, mo_energy=mo_energy, with_t2=True)
        print(f"Calculating Gradient")
        grad = self.crit_grad(mo_coeff, mo_energy, mo_occ, n_occ=n_occ, n_vir=n_vir)
        print(f"Calculating Hessian")
        hess = self.crit_hess(mo_coeff, mo_energy, mo_occ, n_occ=n_occ, n_vir=n_vir)

        eigvals = numpy.linalg.eigvalsh(hess)

        hess_shifted = hess - numpy.eye(grad.shape[0]) * (1.25 * eigvals[0])
        print(f"Lowest Hessian Eigenvalue: {eigvals[0]}")
        print(f"Gradient Length: {numpy.linalg.norm(grad)}")
        step = numpy.linalg.solve(hess_shifted, -grad)

        stepnorm = numpy.linalg.norm(step)
        if stepnorm >= self.stepsize:
            step *= self.stepsize / stepnorm

        print(f"Stepsize: {stepnorm}")

        return step

    def check_convergence(self, mo_coeff, mo_energy, mo_occ, n_occ, n_vir, stepvec, checksize=0.1):
        grad = self.crit_grad(mo_coeff, mo_energy, mo_occ, n_occ=n_occ, n_vir=n_vir)
        if numpy.linalg.norm(grad) > 10**-5:
            return False, stepvec
        #if numpy.linalg.norm(stepvec) > 10**-5:
        #   return False, stepvec

        n_param = int((n_occ * (n_occ - 1) + n_vir * (n_vir - 1)) // 2)

        #checkvecs = numpy.eye(n_param) * checksize
        checkvecs = scipy.linalg.svd(self.crit_hess(mo_coeff, mo_energy, mo_occ, n_occ, n_vir))[0] * checksize
        current_val = self.criterion(numpy.zeros(n_param), mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

        print(f"Checking Manually for Saddlepoint...")

        for cv in checkvecs:
            c = self.criterion(cv, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
            if c - current_val < -10**-7:
                print(f"Found to be Saddlepoint, Dist: {c - current_val}.")
                return False, cv

        return True, None

    def kernel(self, optimality='max_energy'):
        mo_coeff = copy.copy(self._mf.mo_coeff)
        mo_energy = self._mf.mo_energy
        mo_occ = self._mf.mo_occ

        log = logger.new_logger(self)

        n_occ = int(numpy.sum(mo_occ) // 2)
        n_vir = int(len(mo_occ) - n_occ)
        n_param = int((n_occ * (n_occ - 1) + n_vir * (n_vir - 1)) // 2)

        print(f"Number of Orbitals: {n_occ} occupied, {n_vir} virtual.")
        print(f"Number of Parameters: {n_param}")

        '''
        numpy.set_printoptions(linewidth=500, precision=3, suppress=True)
        self._mp2.kernel(mo_coeff=mo_coeff, mo_energy=mo_energy, with_t2=True)
        print("Calculating anayltical Hessian")
        hess_anal = self.crit_hess(mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
        print("Calculating numerical Hessian")
        hess_num = self.crit_hess(mo_coeff, mo_energy, mo_occ, n_occ, n_vir, force_numeric=True)

        print(f"Exact MP2 energy: {self._mp2.e_corr}")

        print("Analytical")
        print(hess_anal)
        print("Numerical")
        print(hess_num)
        print("Division")
        #print((hess_num[n_occ:,n_occ:] / hess_anal[n_occ:,n_occ:]) * (abs(hess_anal[n_occ:,n_occ:]) > 0.001))
        print(hess_num - hess_anal)
        print(numpy.linalg.norm(hess_num - hess_anal))
        quit()
        '''

        for cycle in range(self.max_cycles):
            print(f"ORD-MP2 iteration: {cycle}")
            stepvec = self.gen_new_vec(mo_coeff, mo_energy, mo_occ, n_occ, n_vir)
            converged, stepvec = self.check_convergence(mo_coeff, mo_energy, mo_occ, n_occ, n_vir, stepvec)

            if converged:
                break

            rot_coeff = _vec_to_rotmat(stepvec, n_occ, n_vir)
            mo_coeff = mo_coeff @ scipy.linalg.expm(rot_coeff)

            print(f"Value: {self.criterion(numpy.zeros(stepvec.shape), mo_coeff, mo_energy, mo_occ, n_occ, n_vir)}")


        self.mo_coeff = mo_coeff
        ec = self._mp2.kernel(mo_coeff=mo_coeff, mo_energy=mo_energy)
        return ec

    def max_energy(self, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
        rotmat = scipy.linalg.expm(_vec_to_rotmat(vec, n_occ, n_vir))
        mc = mo_coeff @ rotmat
        e_corr = self._mp2.kernel(mo_energy=mo_energy, mo_coeff=mc)[0]
        return e_corr

    def min_energy(self, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
        return -self.max_energy(vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir)

    def eri_sum(self, vec, mo_coeff, mo_energy, mo_occ, n_occ, n_vir):
        n_occ_p = int(n_occ * (n_occ - 1) // 2)
        n_vir_p = int(n_vir * (n_vir - 1) // 2)
        n_param = n_occ_p + n_vir_p
        mp = self._mp2
        rotmat = scipy.linalg.expm(_vec_to_rotmat(vec, n_occ, n_vir))
        mc = mo_coeff @ rotmat
        eris = mp.ao2mo(mc).ovov
        g_oovv = None
        if isinstance(eris, numpy.ndarray) and eris.ndim == 4:
            g_oovv = eris
        else:
            g_oovv = numpy.zeros((n_occ, n_occ, n_vir, n_vir))
            for i in range(n_occ):
                gi = numpy.asarray(eris[i*n_vir:(i+1)*n_vir])
                g_oovv[i] = gi.reshape(n_vir, n_occ, n_vir).transpose(1,0,2)

        return einsum('ijab->', g_oovv)


if __name__ == '__main__':
    from pyscf import gto, scf, mp
    mol = gto.M(atom='Li 0.0 0.0 0.0; Li 0.0 0.0 0.78', basis='sto-3g', verbose=4)
    mf = scf.RHF(mol)
    mf.kernel()

    ordmp = ORDMP2(mf, optimality='max_energy')
    mp2 = mp.MP2(mf)

    ec = ordmp.kernel()
    ec0 = mp2.kernel()[0]
    print(f"ORD-MP2 energy: {ec[0]}")
    print(f"RMP2 energy: {ec0}")

