from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.linalg import blas
from pyscf.nao.m_sparsetools import csr_matvec, csc_matvec, csc_matvecs

try:
    import numba
    from pyscf.nao.m_iter_div_eigenenergy_numba import div_eigenenergy_numba
    use_numba = True
except:
    use_numba = False

def chi0_mv(v, xocc, xvrt, ksn2e, ksn2f, cc_da, v_dab, no, nfermi, nprod, vstart,
        comega=1j*0.0, dtype=np.float32, cdtype=np.complex64, gemm=blas.sgemm):
    """
        Apply the non-interacting response function to a vector
    """

    if dtype == np.float64:
        gemm = blas.dgemm

    if v.dtype == cdtype:
        vext = np.zeros((v.shape[0], 2), dtype = dtype, order="F")
        vext[:, 0] = v.real
        vext[:, 1] = v.imag

        # real part
        vdp = csr_matvec(cc_da, vext[:, 0])
        sab = (vdp*v_dab).reshape([no,no])

        nb2v = gemm(1.0, xocc, sab)
        nm2v_re = gemm(1.0, nb2v, xvrt.T)

        # imaginary part
        vdp = csr_matvec(cc_da, vext[:, 1])
        sab = (vdp*v_dab).reshape([no,no])

        nb2v = gemm(1.0, xocc, sab)
        nm2v_im = gemm(1.0, nb2v, xvrt.T)
    else:
        vext = np.zeros((v.shape[0], 2), dtype = dtype, order="F")
        vext[:, 0] = v.real

        vdp = csr_matvec(cc_da, vext[:, 0])
        sab = (vdp*v_dab).reshape([no,no])

        nb2v = gemm(1.0, xocc, sab)
        nm2v_re = gemm(1.0, nb2v, xvrt.T)

        nm2v_im = np.zeros(nm2v_re.shape, dtype = dtype)

    if use_numba:
        div_eigenenergy_numba(ksn2e, ksn2f, nfermi, vstart, comega, 
                nm2v_re, nm2v_im, no)
    else:
        for n,[en,fn] in enumerate(zip(ksn2e[0:nfermi], ksn2f[0:nfermi])):
            for j,[em,fm] in enumerate(zip(ksn2e[n+1:],ksn2f[n+1:])):
                m = j+n+1-vstart
                nm2v = nm2v_re[n, m] + 1.0j*nm2v_im[n, m]
                nm2v = nm2v * (fn-fm) *\
                        ( 1.0 / (comega - (em - en)) - 1.0 / (comega + (em - en)) )
                nm2v_re[n, m] = nm2v.real
                nm2v_im[n, m] = nm2v.imag

    # real part
    nb2v = gemm(1.0, nm2v_re, xvrt)
    ab2v = gemm(1.0, xocc.T, nb2v).reshape(no*no)
    vdp = csr_matvec(v_dab, ab2v)
    
    chi0_re = vdp*cc_da

    # imag part
    nb2v = gemm(1.0, nm2v_im, xvrt)
    ab2v = gemm(1.0, xocc.T, nb2v).reshape(no*no)
    vdp = csr_matvec(v_dab, ab2v)
    
    chi0_im = vdp*cc_da

    return chi0_re + 1.0j*chi0_im

def chi0_mv_gpu(tddft_iter_gpu, v, cc_da, v_dab, no,
        comega=1j*0.0, dtype=np.float32, cdtype=np.complex64):
    """
        Apply the non-interacting response function to a vector using gpu for
        matrix-matrix multiplication
    """

    if dtype != np.float32:
        print(dtype)
        raise ValueError("GPU version only with single precision")

    vext = np.zeros((v.shape[0], 2), dtype = dtype, order="F")
    vext[:, 0] = v.real
    vext[:, 1] = v.imag

    # real part
    vdp = csr_matvec(cc_da, vext[:, 0])
    sab = (vdp*v_dab).reshape([no,no])

    tddft_iter_gpu.cpy_sab_to_device(sab)
    tddft_iter_gpu.calc_nm2v_real()

    # imaginary part
    vdp = csr_matvec(cc_da, vext[:, 1])
    sab = (vdp*v_dab).reshape([no,no])

    tddft_iter_gpu.cpy_sab_to_device(sab)
    tddft_iter_gpu.calc_nm2v_imag()

    tddft_iter_gpu.div_eigenenergy_gpu(comega)

    # real part
    tddft_iter_gpu.calc_sab_real()
    tddft_iter_gpu.cpy_sab_to_host(sab, Async = 1)
    
    # start calc_ imag to overlap with cpu calculations
    tddft_iter_gpu.calc_sab_imag()

    vdp = csr_matvec(v_dab, sab)
    chi0_re = vdp*cc_da

    # imag part
    tddft_iter_gpu.cpy_sab_to_host(sab)

    vdp = csr_matvec(v_dab, sab)
    chi0_im = vdp*cc_da

    return chi0_re + 1.0j*chi0_im

