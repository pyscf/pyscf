# Copyright (C) 2003  CAMP
# Copyright (C) 2010  Argonne National Laboratory
# Please see the accompanying LICENSE file for further information.
# original file from https://gitlab.com/gpaw/gpaw/blob/master/gpaw/utilities/scalapack.py
# modified the 19-09-2017 to use with pyscf by Marc Barbry

"""
Python wrapper functions for the C and Fortran packages:
Basic Linear Algebra Communication Subprogramcs (BLACS)
ScaLAPACK

See also:
http://www.netlib.org/blacs
and
http://www.netlib.org/scalapack
"""

import warnings

from pyscf.lib import misc
libmpi = misc.load_library("libmpi_wp")

switch_lu = {'U': 'L', 'L': 'U'}


def scalapack_zero(desca, a, uplo, ia=1, ja=1):
    """Zero the upper or lower half of a square matrix."""
    assert desca.gshape[0] == desca.gshape[1]
    p = desca.gshape[0] - 1
    if uplo == 'L':
        ia = ia + 1
    else:
        ja = ja + 1
    scalapack_set(desca, a, 0.0, 0.0, uplo, p, p, ia, ja)


def scalapack_set(desca, a, alpha, beta, uplo, m=None, n=None, ia=1, ja=1):
    """Set the diagonal and upper/lower triangular part of a.

    Set the upper or lower triangular part of a to alpha, and the diagonal
    of a to beta, where alpha and beta are real or complex numbers."""
    desca.checkassert(a)
    assert uplo in ['L', 'U']
    if m is None:
        m = desca.gshape[0]
    if n is None:
        n = desca.gshape[1]
    if not desca.blacsgrid.is_active():
        return
    libmpi.scalapack_set(a, desca.asarray(), alpha, beta, 
                        switch_lu[uplo], n, m, ja, ia)


def scalapack_diagonalize_dc(desca, a, z, w, uplo):
    """Diagonalize symmetric matrix using the divide & conquer algorithm.
    Orthogonal eigenvectors not guaranteed; no warning is provided.
 
    Solve the eigenvalue equation::
    
      A_nn Z_nn = w_N Z_nn

    Diagonalizes A_nn and writes eigenvectors to Z_nn.  Both A_nn
    and Z_nn must be compatible with desca descriptor.  Values in
    A_nn will be overwritten.
    
    Eigenvalues are written to the global array w_N in ascending order.

    The `uplo` flag can be either 'L' or 'U', meaning that the
    matrices are taken to be upper or lower triangular respectively.
    """
    desca.checkassert(a)
    desca.checkassert(z)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    assert desca.gshape[0] == len(w)
    info = libmpi.scalapack_diagonalize_dc(a, desca.asarray(), 
                                          switch_lu[uplo], z, w)
    if info != 0:
        raise RuntimeError('scalapack_diagonalize_dc error: %d' % info)

 
def scalapack_diagonalize_ex(desca, a, z, w, uplo, iu=None):
    """Diagonalize symmetric matrix using the bisection and inverse
    iteration algorithm. Re-orthogonalization of eigenvectors 
    is an issue for tightly clustered eigenvalue problems; it 
    requires substantial memory and is not scalable. See ScaLAPACK
    pdsyevx.f routine for more information.
 
    Solve the eigenvalue equation::
    
      A_nn Z_nn = w_N Z_nn

    Diagonalizes A_nn and writes eigenvectors to Z_nn.  Both A_nn
    and Z_nn must be compatible with desca descriptor.  Values in
    A_nn will be overwritten.
    
    Eigenvalues are written to the global array w_N in ascending order.

    The `uplo` flag can be either 'L' or 'U', meaning that the
    matrices are taken to be upper or lower triangular respectively.

    The `iu` specifies how many eigenvectors and eigenvalues to compute.
    """
    desca.checkassert(a)
    desca.checkassert(z)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    if iu is None: # calculate all eigenvectors and eigenvalues
        iu = desca.gshape[0]
    assert 1 < iu <= desca.gshape[0]
    # still need assert for eigenvalues
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    assert desca.gshape[0] == len(w)
    if (desca.blacsgrid.myrow, desca.blacsgrid.mycol) == (0, 0):
        message = 'scalapack_diagonalize_ex may have a buffer ' \
            'overflow, use scalapack_diagonalize_dc instead'
        warnings.warn(message, RuntimeWarning)
    info = libmpi.scalapack_diagonalize_ex(a, desca.asarray(), 
                                          switch_lu[uplo], 
                                          iu, z, w)
    if info != 0:
        # 0 means you are OK
        raise RuntimeError('scalapack_diagonalize_ex error: %d' % info)

def scalapack_diagonalize_mr3(desca, a, z, w, uplo, iu=None):
    """Diagonalize symmetric matrix using the MRRR algorithm.
 
    Solve the eigenvalue equation::
    
      A_nn Z_nn = w_N Z_nn

    Diagonalizes A_nn and writes eigenvectors to Z_nn.  Both A_nn
    and Z_nn must be compatible with this desca descriptor.  Values in
    A_nn will be overwritten.
    
    Eigenvalues are written to the global array w_N in ascending order.

    The `uplo` flag can be either 'L' or 'U', meaning that the
    matrices are taken to be upper or lower triangular respectively.

    The `iu` specifies how many eigenvectors and eigenvalues to compute.
    """
    desca.checkassert(a)
    desca.checkassert(z)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    if iu is None: # calculate all eigenvectors and eigenvalues
        iu = desca.gshape[0]
    assert 1 < iu <= desca.gshape[0]
    # stil need assert for eigenvalues
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    assert desca.gshape[0] == len(w)
    info = libmpi.scalapack_diagonalize_mr3(a, desca.asarray(), 
                                           switch_lu[uplo], 
                                           iu, z, w)
    if info != 0:
        raise RuntimeError('scalapack_diagonalize_mr3 error: %d' % info)

def scalapack_general_diagonalize_dc(desca, a, b, z, w, uplo):
    """Diagonalize symmetric matrix using the divide & conquer algorithm.
    Orthogonal eigenvectors not guaranteed; no warning is provided.
 
    Solve the generalized eigenvalue equation::
    
      A_nn Z_nn = w_N B_nn Z_nn

    B_nn is assumed to be positivde definite. Eigenvectors written to Z_nn. 
    Both A_nn, B_nn and Z_nn must be compatible with desca descriptor.
    Values in A_nn and B_nn will be overwritten.
    
    Eigenvalues are written to the global array w_N in ascending order.

    The `uplo` flag can be either 'L' or 'U', meaning that the
    matrices are taken to be upper or lower triangular respectively.
    """
    desca.checkassert(a)
    desca.checkassert(b)
    desca.checkassert(z)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1] 
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    assert desca.gshape[0] == len(w)
    info = libmpi.scalapack_general_diagonalize_dc(a, desca.asarray(), 
                                          switch_lu[uplo], b, z, w)
    if info != 0:
        raise RuntimeError('scalapack_general_diagonalize_dc error: %d' % info)

def scalapack_general_diagonalize_ex(desca, a, b, z, w, uplo, iu=None):
    """Diagonalize symmetric matrix using the bisection and inverse
    iteration algorithm. Re-orthogonalization of eigenvectors 
    is an issue for tightly clustered eigenvalue problems; it 
    requires substantial memory and is not scalable. See ScaLAPACK
    pdsyevx.f routine for more information.
 
    Solves the eigenvalue equation::
    
      A_nn Z_nn = w_N B_nn Z_nn

    B_nn is assumed to be positivde definite. Eigenvectors written to Z_nn. 
    Both A_nn, B_nn and Z_nn must be compatible with desca descriptor.
    Values in A_nn and B_nn will be overwritten.
    
    Eigenvalues are written to the global array w_N in ascending order.

    The `uplo` flag can be either 'L' or 'U', meaning that the
    matrices are taken to be upper or lower triangular respectively.

    The `iu` specifies how many eigenvectors and eigenvalues to compute.
    """
    desca.checkassert(a)
    desca.checkassert(b)
    desca.checkassert(z)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    if iu is None: # calculate all eigenvectors and eigenvalues
        iu = desca.gshape[0]
    assert 1 < iu <= desca.gshape[0]
    # still need assert for eigenvalues
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    assert desca.gshape[0] == len(w)
    if (desca.blacsgrid.myrow, desca.blacsgrid.mycol) == (0, 0):
        message = 'scalapack_general_diagonalize_ex may have a buffer ' \
            'overflow, use scalapack_general_diagonalize_dc instead'
        warnings.warn(message, RuntimeWarning)
    info = libmpi.scalapack_general_diagonalize_ex(a, desca.asarray(), 
                                                  switch_lu[uplo], 
                                                  iu, b, z, w)
    if info != 0:
        # 0 means you are OK
        raise RuntimeError('scalapack_general_diagonalize_ex error: %d' % info)

def scalapack_general_diagonalize_mr3(desca, a, b, z, w, uplo, iu=None):
    """Diagonalize symmetric matrix using the MRRR algorithm.

    Solve the generalized eigenvalue equation::
    
      A_nn Z_nn = w_N B_nn Z_nn

    B_nn is assumed to be positivde definite. Eigenvectors written to Z_nn. 
    Both A_nn, B_nn and Z_nn must be compatible with desca descriptor.
    Values in A_nn and B_nn will be overwritten.
    
    Eigenvalues are written to the global array w_N in ascending order.

    The `uplo` flag can be either 'L' or 'U', meaning that the
    matrices are taken to be upper or lower triangular respectively.

    The `iu` specifies how many eigenvectors and eigenvalues to compute.
    """
    desca.checkassert(a)
    desca.checkassert(b)
    desca.checkassert(z)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    if iu is None: # calculate all eigenvectors and eigenvalues
        iu = desca.gshape[0]
    assert 1 < iu <= desca.gshape[0]
    # still need assert for eigenvalues
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    assert desca.gshape[0] == len(w)
    info = libmpi.scalapack_general_diagonalize_mr3(a, desca.asarray(), 
                                                   switch_lu[uplo], 
                                                   iu, b, z, w)
    if info != 0:
        raise RuntimeError('scalapack_general_diagonalize_mr3 error: %d' % info)

def scalapack_inverse_cholesky(desca, a, uplo):
    """Perform Cholesky decomposin followed by an inversion
    of the resulting triangular matrix.

    Only the upper or lower half of the matrix a will be
    modified; the other half is zeroed out.

    The `uplo` flag can be either 'L' or 'U', meaning that the
    matrices are taken to be upper or lower triangular respectively.
    """
    desca.checkassert(a)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    info = libmpi.scalapack_inverse_cholesky(a, desca.asarray(),
                                            switch_lu[uplo])
    if info != 0:
        raise RuntimeError('scalapack_inverse_cholesky error: %d' % info)

def scalapack_inverse(desca, a, uplo):
    """Perform a hermitian matrix inversion.

    """
    desca.checkassert(a)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    assert uplo in ['L', 'U']
    if not desca.blacsgrid.is_active():
        return
    info = libmpi.scalapack_inverse(a, desca.asarray(), switch_lu[uplo])
    if info != 0:
        raise RuntimeError('scalapack_inverse error: %d' % info)

def scalapack_solve(desca, descb, a, b):
    """Perform general matrix solution to Ax=b. Result will be replaces with b.
       Equivalent to numpy.linalg.solve(a, b.T.conjugate()).T.conjugate()

    """
    desca.checkassert(a)
    descb.checkassert(b)
    # only symmetric matrices
    assert desca.gshape[0] == desca.gshape[1]
    # valid equation
    assert desca.gshape[1] == descb.gshape[1]

    if not desca.blacsgrid.is_active():
        return
    info = libmpi.scalapack_solve(a.T, desca.asarray(), b.T, descb.asarray())
    if info != 0:
        raise RuntimeError('scalapack_solve error: %d' % info)

def pblas_tran(alpha, a_MN, beta, c_NM, desca, descc):
    desca.checkassert(a_MN)
    descc.checkassert(c_NM)
    M, N = desca.gshape
    assert N, M == descc.gshape
    libmpi.pblas_tran(N, M, alpha, a_MN, beta, c_NM,
                     desca.asarray(), descc.asarray())


def pblas_hemm(alpha, a_MK, b_KN, beta, c_MN, desca, descb, descc,
               side='L', uplo='L'):
    # Hermitean matrix multiply, only lower or upper diagonal of a_MK
    # is used. By default, C = beta*C + alpha*A*B
    # Executes PBLAS method pzhemm for complex and pdsymm for real matrices.
    desca.checkassert(a_MK)
    descb.checkassert(b_KN)
    descc.checkassert(c_MN)
    assert side in ['R','L'] and uplo in ['L','U']
    M, Ka = desca.gshape
    Kb, N = descb.gshape
    if side=='R':
        Kb, N = N, Kb

    if not desca.blacsgrid.is_active():
        return
    fortran_side = {'L':'R', 'R':'L'}
    fortran_uplo = {'U':'L', 'L':'U'}
    if side=='R':
        M, N = N, M

    libmpi.pblas_hemm(fortran_side[side], fortran_uplo[uplo], 
                     N, M, alpha, a_MK.T, b_KN.T, beta, c_MN.T,
                     desca.asarray(), descb.asarray(), descc.asarray())
    
def pblas_gemm(alpha, a_MK, b_KN, beta, c_MN, desca, descb, descc,
               transa='N', transb='N'):
    desca.checkassert(a_MK)
    descb.checkassert(b_KN)
    descc.checkassert(c_MN)
    assert transa in ['N', 'T', 'C'] and transb in ['N', 'T', 'C']
    M, Ka = desca.gshape
    Kb, N = descb.gshape

    if transa == 'T':
        M, Ka = Ka, M
    if transb == 'T':
        Kb, N = N, Kb
    Mc, Nc = descc.gshape

    assert Ka == Kb
    assert M == Mc
    assert N == Nc

    #trans = transa + transb

    """
    if transb == 'N':
        assert desca.gshape[1] == descb.gshape[0]
        assert desca.gshape[0] == descc.gshape[0]
        assert descb.gshape[1] == descc.gshape[1]
    if transb == 'T':
        N, Kb = Kb, N
        #assert desca.gshape[1] == descb.gshape[1]
        assert desca.gshape[0] == descc.gshape[0]
        assert descb.gshape[0] == descc.gshape[1]

    if trans == 'NN':
        assert desca.gshape[1] == descb.gshape[0]
        assert desca.gshape[0] == descc.gshape[0]
        assert descb.gshape[1] == descc.gshape[1]
    elif transa == 'T':
        M, Ka = Ka, M
        assert desca.gshape[1] == descc.gshape[0]
    if transb == 'N':
        assert descb.gshape[1] == descc.gshape[1]
    elif transb == 'T':
        assert descb.gshape[1] == descc.gshape[1]
    assert Ka == Kb
    #assert transa == 'N' # XXX remember to implement 'T'
    libmpi.pblas_gemm(N, M, Ka, alpha, b_KN.T, a_MK.T, beta, c_MN.T,
    """
    #assert transa == 'N' # XXX remember to implement 'T'
    if not desca.blacsgrid.is_active():
        return
    libmpi.pblas_gemm(N, M, Ka, alpha, b_KN.T, a_MK.T, beta, c_MN.T,
                     descb.asarray(), desca.asarray(), descc.asarray(),
                     transb, transa)


def pblas_simple_gemm(desca, descb, descc, a_MK, b_KN, c_MN, 
                      transa='N', transb='N'):
    alpha = 1.0
    beta = 0.0
    pblas_gemm(alpha, a_MK, b_KN, beta, c_MN, desca, descb, descc,
               transa, transb)

def pblas_simple_hemm(desca, descb, descc, a_MK, b_KN, c_MN, side='L', uplo='L'):
    alpha = 1.0
    beta = 0.0
    pblas_hemm(alpha, a_MK, b_KN, beta, c_MN, desca, descb, descc, side, uplo)

def pblas_gemv(alpha, a, x, beta, y, desca, descx, descy,
               transa='T'):
    desca.checkassert(a)
    descx.checkassert(x)
    descy.checkassert(y)
    M, N = desca.gshape
    # XXX transa = 'N' not implemented
    assert transa in ['T', 'C']
    assert desca.gshape[0] == descy.gshape[0]
    assert desca.gshape[1] == descx.gshape[0]
    assert descx.gshape[1] == descy.gshape[1]
    if not desca.blacsgrid.is_active():
        return
    libmpi.pblas_gemv(N, M, alpha,
                     a, x, beta, y,
                     desca.asarray(),
                     descx.asarray(),
                     descy.asarray(), 
                     transa)


def pblas_simple_gemv(desca, descx, descy, a, x, y):
    alpha = 1.0
    beta = 0.0
    pblas_gemv(alpha, a, x, beta, y, desca, descx, descy)


def pblas_r2k(alpha, a_NK, b_NK, beta, c_NN, desca, descb, descc,
                uplo='U'):
    if not desca.blacsgrid.is_active():
        return
    desca.checkassert(a_NK)
    descb.checkassert(b_NK)
    descc.checkassert(c_NN)
    assert descc.gshape[0] == descc.gshape[1] # symmetric matrix
    assert desca.gshape == descb.gshape # same shape
    assert uplo in ['L', 'U']
    N = descc.gshape[0] # order of C
    # K must take into account implicit tranpose due to C ordering
    K = desca.gshape[1] # number of columns of A and B
    libmpi.pblas_r2k(N, K, alpha, a_NK, b_NK, beta, c_NN,
                    desca.asarray(), 
                    descb.asarray(), 
                    descc.asarray(),
                    uplo)


def pblas_simple_r2k(desca, descb, descc, a, b, c, uplo='U'):
    alpha = 1.0
    beta = 0.0
    pblas_r2k(alpha, a, b, beta, c, 
                desca, descb, descc, uplo)


def pblas_rk(alpha, a_NK, beta, c_NN, desca, descc,
             uplo='U'):
    if not desca.blacsgrid.is_active():
        return
    desca.checkassert(a_NK)
    descc.checkassert(c_NN)
    assert descc.gshape[0] == descc.gshape[1] # symmetrix matrix
    assert uplo in ['L', 'U']
    N = descc.gshape[0] # order of C
    # K must take into account implicit tranpose due to C ordering
    K = desca.gshape[1] # number of columns of A
    libmpi.pblas_rk(N, K, alpha, a_NK, beta, c_NN,
                    desca.asarray(), 
                    descc.asarray(),
                    uplo)


def pblas_simple_rk(desca, descc, a, c):
    alpha = 1.0 
    beta = 0.0 
    pblas_rk(alpha, a, beta, c, 
             desca, descc)

