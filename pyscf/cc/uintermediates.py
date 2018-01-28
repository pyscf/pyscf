import numpy as np
from pyscf import lib
from pyscf import ao2mo
from pyscf.cc.rintermediates import _get_ovvv, _get_vvvv

# Ref: Gauss and Stanton, J. Chem. Phys. 103, 3561 (1995) Table III

# uccsd intermediates has been moved to gccsd intermediates

def make_tau(t2, t1a, t1b, fac=1, out=None):
    #:tmp = einsum('ia,jb->ijab',t1a,t1b)
    #:t1t1 = tmp - tmp.transpose(1,0,2,3) - tmp.transpose(0,1,3,2) + tmp.transpose(1,0,3,2)
    #:tau1 = t2 + fac*0.50*t1t1
    tau1  = np.einsum('ia,jb->ijab', t1a, t1b)
    tau1 -= np.einsum('ia,jb->jiab', t1a, t1b)
    tau1 = tau1 - tau1.transpose(0,1,3,2)
    tau1 *= fac * .5
    tau1 += t2
    return tau1

def _get_ovvv_base(ovvv, *slices):
    if len(ovvv.shape) == 3:  # DO NOT use .ndim here for h5py library
                              # backward compatbility
        ovw = np.asarray(ovvv[slices])
        nocc, nvir, nvir_pair = ovw.shape
        ovvv = lib.unpack_tril(ovw.reshape(nocc*nvir,nvir_pair))
        nvir1 = ovvv.shape[2]
        return ovvv.reshape(nocc,nvir,nvir1,nvir1)
    elif slices:
        return ovvv[slices]
    else:
        return ovvv

def _get_ovVV(eris, *slices):
    return _get_ovvv_base(eris.ovVV, *slices)

def _get_OVvv(eris, *slices):
    return _get_ovvv_base(eris.OVvv, *slices)

def _get_OVVV(eris, *slices):
    return _get_ovvv_base(eris.OVVV, *slices)

def _get_vvVV(eris):
    if eris.vvVV is None and hasattr(eris, 'VVL'):  # DF eris
        vvL = np.asarray(eris.vvL)
        VVL = np.asarray(eris.VVL)
        vvVV = lib.dot(vvL, VVL.T)
    elif len(eris.vvVV.shape) == 2:
        vvVV = np.asarray(eris.vvVV)
    else:
        return eris.vvVV

    nvira = int(np.sqrt(vvVV.shape[0]*2))
    nvirb = int(np.sqrt(vvVV.shape[1]*2))
    vvVV1 = np.zeros((nvira**2,nvirb**2))
    vtrila = np.tril_indices(nvira)
    vtrilb = np.tril_indices(nvirb)
    lib.takebak_2d(vvVV1, vvVV, vtrila[0]*nvira+vtrila[1], vtrilb[0]*nvirb+vtrilb[1])
    lib.takebak_2d(vvVV1, vvVV, vtrila[1]*nvira+vtrila[0], vtrilb[1]*nvirb+vtrilb[0])
    lib.takebak_2d(vvVV1, vvVV, vtrila[0]*nvira+vtrila[1], vtrilb[1]*nvirb+vtrilb[0])
    lib.takebak_2d(vvVV1, vvVV, vtrila[1]*nvira+vtrila[0], vtrilb[0]*nvirb+vtrilb[1])
    return vvVV1.reshape(nvira,nvira,nvirb,nvirb)

def _get_VVVV(eris):
    if eris.VVVV is None and hasattr(eris, 'VVL'):  # DF eris
        VVL = np.asarray(eris.VVL)
        nvir = int(np.sqrt(eris.VVL.shape[0]*2))
        return ao2mo.restore(1, lib.dot(VVL, VVL.T), nvir)
    elif len(eris.VVVV.shape) == 2:
        nvir = int(np.sqrt(eris.VVVV.shape[0]*2))
        return ao2mo.restore(1, np.asarray(eris.VVVV), nvir)
    else:
        return eris.VVVV
