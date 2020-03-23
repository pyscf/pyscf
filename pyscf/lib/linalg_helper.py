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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Extension to scipy.linalg module
'''

import sys
import inspect
import warnings
from functools import reduce
import numpy
import scipy.linalg
from pyscf.lib import logger
from pyscf.lib import numpy_helper
from pyscf.lib import misc
from pyscf import __config__

SAFE_EIGH_LINDEP = getattr(__config__, 'lib_linalg_helper_safe_eigh_lindep', 1e-15)
DAVIDSON_LINDEP = getattr(__config__, 'lib_linalg_helper_davidson_lindep', 1e-14)
DSOLVE_LINDEP = getattr(__config__, 'lib_linalg_helper_dsolve_lindep', 1e-15)
MAX_MEMORY = getattr(__config__, 'lib_linalg_helper_davidson_max_memory', 2000)  # 2GB

# sort by similarity has problem which flips the ordering of eigenvalues when
# the initial guess is closed to excited state.  In this situation, function
# _sort_by_similarity may mark the excited state as the first eigenvalue and
# freeze the first eigenvalue.
SORT_EIG_BY_SIMILARITY = \
        getattr(__config__, 'lib_linalg_helper_davidson_sort_eig_by_similiarity', False)
# Projecting out converged eigenvectors has problems when conv_tol is loose.
# In this situation, the converged eigenvectors may be updated in the
# following iterations.  Projecting out the converged eigenvectors may lead to
# large errors to the yet converged eigenvectors.
PROJECT_OUT_CONV_EIGS = \
        getattr(__config__, 'lib_linalg_helper_davidson_project_out_eigs', False)

FOLLOW_STATE = getattr(__config__, 'lib_linalg_helper_davidson_follow_state', False)


def safe_eigh(h, s, lindep=SAFE_EIGH_LINDEP):
    '''Solve generalized eigenvalue problem  h v = w s v.

    .. note::
        The number of eigenvalues and eigenvectors might be less than the
        matrix dimension if linear dependency is found in metric s.

    Args:
        h, s : 2D array
            Complex Hermitian or real symmetric matrix.

    Kwargs:
        lindep : float
            Linear dependency threshold.  By diagonalizing the metric s, we
            consider the eigenvectors are linearly dependent subsets if their
            eigenvalues are smaller than this threshold.

    Returns:
        w, v, seig.  w is the eigenvalue vector; v is the eigenfunction array;
        seig is the eigenvalue vector of the metric s.
    '''
    seig, t = scipy.linalg.eigh(s)
    try:
        w, v = scipy.linalg.eigh(h, s)
    except numpy.linalg.LinAlgError:
        idx = seig >= lindep
        t = t[:,idx] * (1/numpy.sqrt(seig[idx]))
        if t.size > 0:
            heff = reduce(numpy.dot, (t.T.conj(), h, t))
            w, v = scipy.linalg.eigh(heff)
            v = numpy.dot(t, v)
        else:
            w = numpy.zeros((0,))
            v = t
    return w, v, seig

def eigh_by_blocks(h, s=None, labels=None):
    '''Solve an ordinary or generalized eigenvalue problem for diagonal blocks.
    The diagonal blocks are extracted based on the given basis "labels".  The
    rows and columns which have the same labels are put in the same block.
    One common scenario one needs the block-wise diagonalization is to
    diagonalize the matrix in symmetry adapted basis, in which "labels" is the
    irreps of each basis.

    Args:
        h, s : 2D array
            Complex Hermitian or real symmetric matrix.

    Kwargs:
        labels : list

    Returns:
        w, v.  w is the eigenvalue vector; v is the eigenfunction array;
        seig is the eigenvalue vector of the metric s.

    Examples:

    >>> from pyscf import lib
    >>> a = numpy.ones((4,4))
    >>> a[0::3,0::3] = 0
    >>> a[1::3,1::3] = 2
    >>> a[2::3,2::3] = 4
    >>> labels = ['a', 'b', 'c', 'a']
    >>> lib.eigh_by_blocks(a, labels)
    (array([ 0.,  0.,  2.,  4.]),
     array([[ 1.,  0.,  0.,  0.],
            [ 0.,  0.,  1.,  0.],
            [ 0.,  0.,  0.,  1.],
            [ 0.,  1.,  0.,  0.]]))
    >>> numpy.linalg.eigh(a)
    (array([ -8.82020545e-01,  -1.81556477e-16,   1.77653793e+00,   5.10548262e+00]),
     array([[  6.40734630e-01,  -7.07106781e-01,   1.68598330e-01,   -2.47050070e-01],
            [ -3.80616542e-01,   9.40505244e-17,   8.19944479e-01,   -4.27577008e-01],
            [ -1.84524565e-01,   9.40505244e-17,  -5.20423152e-01,   -8.33732828e-01],
            [  6.40734630e-01,   7.07106781e-01,   1.68598330e-01,   -2.47050070e-01]]))

    >>> from pyscf import gto, lib, symm
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1', basis='ccpvdz', symmetry=True)
    >>> c = numpy.hstack(mol.symm_orb)
    >>> vnuc_so = reduce(numpy.dot, (c.T, mol.intor('int1e_nuc_sph'), c))
    >>> orbsym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, c)
    >>> lib.eigh_by_blocks(vnuc_so, labels=orbsym)
    (array([-4.50766885, -1.80666351, -1.7808565 , -1.7808565 , -1.74189134,
            -0.98998583, -0.98998583, -0.40322226, -0.30242374, -0.07608981]),
     ...)
    '''
    if labels is None:
        return scipy.linalg.eigh(h, s)

    labels = numpy.asarray(labels)
    es = []
    cs = numpy.zeros_like(h)
    if s is None:
        p0 = 0
        for label in set(labels):
            idx = labels == label
            e, c = scipy.linalg.eigh(h[idx][:,idx])
            cs[idx,p0:p0+e.size] = c
            es.append(e)
            p0 = p0 + e.size
    else:
        p0 = 0
        for label in set(labels):
            idx = labels == label
            e, c = scipy.linalg.eigh(h[idx][:,idx], s[idx][:,idx])
            cs[idx,p0:p0+e.size] = c
            es.append(e)
            p0 = p0 + e.size
    es = numpy.hstack(es)
    idx = numpy.argsort(es)
    return es[idx], cs[:,idx]

def davidson(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
             lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
             dot=numpy.dot, callback=None,
             nroots=1, lessio=False, pick=None, verbose=logger.WARN,
             follow_state=FOLLOW_STATE):
    '''Davidson diagonalization method to solve  a c = e c.  Ref
    [1] E.R. Davidson, J. Comput. Phys. 17 (1), 87-94 (1975).
    [2] http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter11.pdf

    Note: This function has an overhead of memory usage ~4*x0.size*nroots

    Args:
        aop : function(x) => array_like_x
            aop(x) to mimic the matrix vector multiplication :math:`\sum_{j}a_{ij}*x_j`.
            The argument is a 1D array.  The returned value is a 1D array.
        x0 : 1D array or a list of 1D array
            Initial guess.  The initial guess vector(s) are just used as the
            initial subspace bases.  If the subspace is smaller than "nroots",
            eg 10 roots and one initial guess, all eigenvectors are chosen as
            the eigenvectors during the iterations.  The first iteration has
            one eigenvector, the next iteration has two, the third iterastion
            has 4, ..., until the subspace size > nroots.
        precond : diagonal elements of the matrix or  function(dx, e, x0) => array_like_dx
            Preconditioner to generate new trial vector.
            The argument dx is a residual vector ``a*x0-e*x0``; e is the current
            eigenvalue; x0 is the current eigenvector.

    Kwargs:
        tol : float
            Convergence tolerance.
        max_cycle : int
            max number of iterations.
        max_space : int
            space size to hold trial vectors.
        lindep : float
            Linear dependency threshold.  The function is terminated when the
            smallest eigenvalue of the metric of the trial vectors is lower
            than this threshold.
        max_memory : int or float
            Allowed memory in MB.
        dot : function(x, y) => scalar
            Inner product
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            envrionment.
        nroots : int
            Number of eigenvalues to be computed.  When nroots > 1, it affects
            the shape of the return value
        lessio : bool
            How to compute a*x0 for current eigenvector x0.  There are two
            ways to compute a*x0.  One is to assemble the existed a*x.  The
            other is to call aop(x0).  The default is the first method which
            needs more IO and less computational cost.  When IO is slow, the
            second method can be considered.
        pick : function(w,v,nroots) => (e[idx], w[:,idx], idx)
            Function to filter eigenvalues and eigenvectors.
        follow_state : bool
            If the solution dramatically changes in two iterations, clean the
            subspace and restart the iteration with the old solution.  It can
            help to improve numerical stability.  Default is False.

    Returns:
        e : float or list of floats
            Eigenvalue.  By default it's one float number.  If :attr:`nroots` > 1, it
            is a list of floats for the lowest :attr:`nroots` eigenvalues.
        c : 1D array or list of 1D arrays
            Eigenvector.  By default it's a 1D array.  If :attr:`nroots` > 1, it
            is a list of arrays for the lowest :attr:`nroots` eigenvectors.

    Examples:

    >>> from pyscf import lib
    >>> a = numpy.random.random((10,10))
    >>> a = a + a.T
    >>> aop = lambda x: numpy.dot(a,x)
    >>> precond = lambda dx, e, x0: dx/(a.diagonal()-e)
    >>> x0 = a[0]
    >>> e, c = lib.davidson(aop, x0, precond)
    '''
    e, x = davidson1(lambda xs: [aop(x) for x in xs],
                     x0, precond, tol, max_cycle, max_space, lindep,
                     max_memory, dot, callback, nroots, lessio, pick, verbose,
                     follow_state)[1:]
    if nroots == 1:
        return e[0], x[0]
    else:
        return e, x

def davidson1(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
             lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
             dot=numpy.dot, callback=None,
             nroots=1, lessio=False, pick=None, verbose=logger.WARN,
             follow_state=FOLLOW_STATE, tol_residual=None):
    '''Davidson diagonalization method to solve  a c = e c.  Ref
    [1] E.R. Davidson, J. Comput. Phys. 17 (1), 87-94 (1975).
    [2] http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter11.pdf

    Note: This function has an overhead of memory usage ~4*x0.size*nroots

    Args:
        aop : function([x]) => [array_like_x]
            Matrix vector multiplication :math:`y_{ki} = \sum_{j}a_{ij}*x_{jk}`.
        x0 : 1D array or a list of 1D arrays or a function to generate x0 array(s)
            Initial guess.  The initial guess vector(s) are just used as the
            initial subspace bases.  If the subspace is smaller than "nroots",
            eg 10 roots and one initial guess, all eigenvectors are chosen as
            the eigenvectors during the iterations.  The first iteration has
            one eigenvector, the next iteration has two, the third iterastion
            has 4, ..., until the subspace size > nroots.
        precond : diagonal elements of the matrix or  function(dx, e, x0) => array_like_dx
            Preconditioner to generate new trial vector.
            The argument dx is a residual vector ``a*x0-e*x0``; e is the current
            eigenvalue; x0 is the current eigenvector.

    Kwargs:
        tol : float
            Convergence tolerance.
        max_cycle : int
            max number of iterations.
        max_space : int
            space size to hold trial vectors.
        lindep : float
            Linear dependency threshold.  The function is terminated when the
            smallest eigenvalue of the metric of the trial vectors is lower
            than this threshold.
        max_memory : int or float
            Allowed memory in MB.
        dot : function(x, y) => scalar
            Inner product
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            envrionment.
        nroots : int
            Number of eigenvalues to be computed.  When nroots > 1, it affects
            the shape of the return value
        lessio : bool
            How to compute a*x0 for current eigenvector x0.  There are two
            ways to compute a*x0.  One is to assemble the existed a*x.  The
            other is to call aop(x0).  The default is the first method which
            needs more IO and less computational cost.  When IO is slow, the
            second method can be considered.
        pick : function(w,v,nroots) => (e[idx], w[:,idx], idx)
            Function to filter eigenvalues and eigenvectors.
        follow_state : bool
            If the solution dramatically changes in two iterations, clean the
            subspace and restart the iteration with the old solution.  It can
            help to improve numerical stability.  Default is False.

    Returns:
        conv : bool
            Converged or not
        e : list of floats
            The lowest :attr:`nroots` eigenvalues.
        c : list of 1D arrays
            The lowest :attr:`nroots` eigenvectors.

    Examples:

    >>> from pyscf import lib
    >>> a = numpy.random.random((10,10))
    >>> a = a + a.T
    >>> aop = lambda xs: [numpy.dot(a,x) for x in xs]
    >>> precond = lambda dx, e, x0: dx/(a.diagonal()-e)
    >>> x0 = a[0]
    >>> e, c = lib.davidson(aop, x0, precond, nroots=2)
    >>> len(e)
    2
    '''
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if tol_residual is None:
        toloose = numpy.sqrt(tol)
    else:
        toloose = tol_residual
    log.debug1('tol %g  toloose %g', tol, toloose)

    if not callable(precond):
        precond = make_diag_precond(precond)

    if callable(x0):  # lazy initialization to reduce memory footprint
        x0 = x0()
    if isinstance(x0, numpy.ndarray) and x0.ndim == 1:
        x0 = [x0]
    #max_cycle = min(max_cycle, x0[0].size)
    max_space = max_space + (nroots-1) * 3
    # max_space*2 for holding ax and xs, nroots*2 for holding axt and xt
    _incore = max_memory*1e6/x0[0].nbytes > max_space*2+nroots*3
    lessio = lessio and not _incore
    log.debug1('max_cycle %d  max_space %d  max_memory %d  incore %s',
               max_cycle, max_space, max_memory, _incore)
    dtype = None
    heff = None
    fresh_start = True
    e = 0
    v = None
    conv = [False] * nroots
    emin = None
    norm_min = 1

    for icyc in range(max_cycle):
        if fresh_start:
            if _incore:
                xs = []
                ax = []
            else:
                xs = _Xlist()
                ax = _Xlist()
            space = 0
# Orthogonalize xt space because the basis of subspace xs must be orthogonal
# but the eigenvectors x0 might not be strictly orthogonal
            xt = None
            x0len = len(x0)
            xt, x0 = _qr(x0, dot, lindep)[0], None
            if len(xt) != x0len:
                log.warn('QR decomposition removed %d vectors.  The davidson may fail.',
                         x0len - len(xt))
                if callable(pick):
                    log.warn('Check to see if `pick` function %s is providing '
                             'linear dependent vectors', pick.__name__)
            max_dx_last = 1e9
            if SORT_EIG_BY_SIMILARITY:
                conv = [False] * nroots
        elif len(xt) > 1:
            xt = _qr(xt, dot, lindep)[0]
            xt = xt[:40]  # 40 trial vectors at most

        axt = aop(xt)
        for k, xi in enumerate(xt):
            xs.append(xt[k])
            ax.append(axt[k])
        rnow = len(xt)
        head, space = space, space+rnow

        if dtype is None:
            try:
                dtype = numpy.result_type(axt[0], xt[0])
            except IndexError:
                dtype = numpy.result_type(ax[0].dtype, xs[0].dtype)
        if heff is None:  # Lazy initilize heff to determine the dtype
            heff = numpy.empty((max_space+nroots,max_space+nroots), dtype=dtype)
        else:
            heff = numpy.asarray(heff, dtype=dtype)

        elast = e
        vlast = v
        conv_last = conv
        for i in range(space):
            if head <= i < head+rnow:
                axi = axt[i-head].conj()
                for k in range(i-head+1):
                    heff[i,head+k] = dot(axi, xt[k])
                    heff[head+k,i] = heff[i,head+k].conj()
            else:
                axi = numpy.asarray(ax[i]).conj()
                for k in range(rnow):
                    heff[i,head+k] = dot(axi, xt[k])
                    heff[head+k,i] = heff[i,head+k].conj()
            axi = None
        xt = axt = None

        w, v = scipy.linalg.eigh(heff[:space,:space])
        if callable(pick):
            w, v, idx = pick(w, v, nroots, locals())
        if SORT_EIG_BY_SIMILARITY:
            e, v = _sort_by_similarity(w, v, nroots, conv, vlast, emin)
            if elast.size != e.size:
                de = e
            else:
                de = e - elast
        else:
            e = w[:nroots]
            v = v[:,:nroots]

        x0 = None
        x0 = _gen_x0(v, xs)
        if lessio:
            ax0 = aop(x0)
        else:
            ax0 = _gen_x0(v, ax)

        if SORT_EIG_BY_SIMILARITY:
            dx_norm = [0] * nroots
            xt = [None] * nroots
            for k, ek in enumerate(e):
                if not conv[k]:
                    xt[k] = ax0[k] - ek * x0[k]
                    dx_norm[k] = numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                    if abs(de[k]) < tol and dx_norm[k] < toloose:
                        log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                                  k, dx_norm[k], ek, de[k])
                        conv[k] = True
        else:
            elast, conv_last = _sort_elast(elast, conv_last, vlast, v,
                                           fresh_start, log)
            de = e - elast
            dx_norm = []
            xt = []
            conv = [False] * nroots
            for k, ek in enumerate(e):
                xt.append(ax0[k] - ek * x0[k])
                dx_norm.append(numpy.sqrt(dot(xt[k].conj(), xt[k]).real))
                conv[k] = abs(de[k]) < tol and dx_norm[k] < toloose
                if conv[k] and not conv_last[k]:
                    log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                              k, dx_norm[k], ek, de[k])
        ax0 = None
        max_dx_norm = max(dx_norm)
        ide = numpy.argmax(abs(de))
        if all(conv):
            log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, space, max_dx_norm, e, de[ide])
            break
        elif (follow_state and max_dx_norm > 1 and
              max_dx_norm/max_dx_last > 3 and space > nroots*1):
            log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                      icyc, space, max_dx_norm, e, de[ide], norm_min)
            log.debug('Large |r| detected, restore to previous x0')
            x0 = _gen_x0(vlast, xs)
            fresh_start = True
            continue

        if SORT_EIG_BY_SIMILARITY:
            if any(conv) and e.dtype == numpy.double:
                emin = min(e)

        # remove subspace linear dependency
        if any(((not conv[k]) and n**2>lindep) for k, n in enumerate(dx_norm)):
            for k, ek in enumerate(e):
                if (not conv[k]) and dx_norm[k]**2 > lindep:
                    xt[k] = precond(xt[k], e[0], x0[k])
                    xt[k] *= 1/numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                else:
                    xt[k] = None
        else:
            for k, ek in enumerate(e):
                if dx_norm[k]**2 > lindep:
                    xt[k] = precond(xt[k], e[0], x0[k])
                    xt[k] *= 1/numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                else:
                    xt[k] = None
                    log.debug1('Throwing out eigenvector %d with norm=%4.3g', k, dx_norm[k])
        xt = [xi for xi in xt if xi is not None]

        for i in range(space):
            xsi = numpy.asarray(xs[i])
            for xi in xt:
                xi -= xsi * dot(xsi.conj(), xi)
            xsi = None
        norm_min = 1
        for i,xi in enumerate(xt):
            norm = numpy.sqrt(dot(xi.conj(), xi).real)
            if norm**2 > lindep:
                xt[i] *= 1/norm
                norm_min = min(norm_min, norm)
            else:
                xt[i] = None
        xt = [xi for xi in xt if xi is not None]
        xi = None
        log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, space, max_dx_norm, e, de[ide], norm_min)
        if len(xt) == 0:
            log.debug('Linear dependency in trial subspace. |r| for each state %s',
                     dx_norm)
            conv = [conv[k] or (norm < toloose) for k,norm in enumerate(dx_norm)]
            break

        max_dx_last = max_dx_norm
        fresh_start = space+nroots > max_space

        if callable(callback):
            callback(locals())

    x0 = [x for x in x0]  # nparray -> list

    # Check whether the solver finds enough eigenvectors.
    h_dim = x0[0].size
    if len(x0) < min(h_dim, nroots):
        # Two possible reasons:
        # 1. All the initial guess are the eigenvectors. No more trial vectors
        # can be generated.
        # 2. The initial guess sits in the subspace which is smaller than the
        # required number of roots.
        msg = 'Not enough eigenvectors (len(x0)=%d, nroots=%d)' % (len(x0), nroots)
        warnings.warn(msg)

    return numpy.asarray(conv), e, x0


def make_diag_precond(diag, level_shift=0):
    '''Generate the preconditioner function with the diagonal function.'''
    def precond(dx, e, *args):
        diagd = diag - (e - level_shift)
        diagd[abs(diagd)<1e-8] = 1e-8
        return dx/diagd
    return precond


def eigh(a, *args, **kwargs):
    nroots = kwargs.get('nroots', 1)
    if isinstance(a, numpy.ndarray) and a.ndim == 2:
        e, v = scipy.linalg.eigh(a)
        if nroots == 1:
            return e[0], v[:,0]
        else:
            return e[:nroots], v[:,:nroots].T
    else:
        return davidson(a, *args, **kwargs)
dsyev = eigh


def pick_real_eigs(w, v, nroots, envs):
    '''This function searchs the real eigenvalues or eigenvalues with small
    imaginary component.
    '''
    threshold = 1e-3
    abs_imag = abs(w.imag)
    # Grab `nroots` number of e with small(est) imaginary components
    max_imag_tol = max(threshold, numpy.sort(abs_imag)[min(w.size,nroots)-1])
    real_idx = numpy.where((abs_imag <= max_imag_tol))[0]
    nbelow_thresh = numpy.count_nonzero(abs_imag[real_idx] < threshold)
    if nbelow_thresh < nroots and w.size >= nroots:
        warnings.warn('Only %d eigenvalues (out of %3d requested roots) with imaginary part < %4.3g.\n'
                      % (nbelow_thresh, min(w.size,nroots), threshold))

    # Guess whether the matrix to diagonalize is real or complex
    if envs.get('dtype') == numpy.double:
        w, v, idx = _eigs_cmplx2real(w, v, real_idx, real_eigenvectors=True)
    else:
        w, v, idx = _eigs_cmplx2real(w, v, real_idx, real_eigenvectors=False)
    return w, v, idx

def _eigs_cmplx2real(w, v, real_idx, real_eigenvectors=True):
    '''
    For non-hermitian diagonalization, this function transforms the complex
    eigenvectors to real eigenvectors.

    If the complex eigenvalue has small imaginary part, both the real part
    and the imaginary part of the eigenvector can approximately be used as
    the "real" eigen solutions.

    NOTE: If real_eigenvectors is set to True, this function can only be used
    for real matrix and real eigenvectors. It discards the imaginary part of
    the eigenvectors then returns only the real part of the eigenvectors.
    '''
    idx = real_idx[w[real_idx].real.argsort()]
    w = w[idx]
    v = v[:,idx]

    if real_eigenvectors:
        degen_idx = numpy.where(w.imag != 0)[0]
        if degen_idx.size > 0:
            # Take the imaginary part of the "degenerated" eigenvectors as an
            # independent eigenvector then discard the imaginary part of v
            v[:,degen_idx[1::2]] = v[:,degen_idx[1::2]].imag
        v = v.real
    return w.real, v, idx

def eig(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
        lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
        dot=numpy.dot, callback=None,
        nroots=1, lessio=False, left=False, pick=pick_real_eigs,
        verbose=logger.WARN, follow_state=FOLLOW_STATE):
    '''Davidson diagonalization to solve the non-symmetric eigenvalue problem

    Args:
        aop : function([x]) => [array_like_x]
            Matrix vector multiplication :math:`y_{ki} = \sum_{j}a_{ij}*x_{jk}`.
        x0 : 1D array or a list of 1D array
            Initial guess.  The initial guess vector(s) are just used as the
            initial subspace bases.  If the subspace is smaller than "nroots",
            eg 10 roots and one initial guess, all eigenvectors are chosen as
            the eigenvectors during the iterations.  The first iteration has
            one eigenvector, the next iteration has two, the third iterastion
            has 4, ..., until the subspace size > nroots.
        precond : diagonal elements of the matrix or  function(dx, e, x0) => array_like_dx
            Preconditioner to generate new trial vector.
            The argument dx is a residual vector ``a*x0-e*x0``; e is the current
            eigenvalue; x0 is the current eigenvector.

    Kwargs:
        tol : float
            Convergence tolerance.
        max_cycle : int
            max number of iterations.
        max_space : int
            space size to hold trial vectors.
        lindep : float
            Linear dependency threshold.  The function is terminated when the
            smallest eigenvalue of the metric of the trial vectors is lower
            than this threshold.
        max_memory : int or float
            Allowed memory in MB.
        dot : function(x, y) => scalar
            Inner product
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            envrionment.
        nroots : int
            Number of eigenvalues to be computed.  When nroots > 1, it affects
            the shape of the return value
        lessio : bool
            How to compute a*x0 for current eigenvector x0.  There are two
            ways to compute a*x0.  One is to assemble the existed a*x.  The
            other is to call aop(x0).  The default is the first method which
            needs more IO and less computational cost.  When IO is slow, the
            second method can be considered.
        left : bool
            Whether to calculate and return left eigenvectors.  Default is False.
        pick : function(w,v,nroots) => (e[idx], w[:,idx], idx)
            Function to filter eigenvalues and eigenvectors.
        follow_state : bool
            If the solution dramatically changes in two iterations, clean the
            subspace and restart the iteration with the old solution.  It can
            help to improve numerical stability.  Default is False.

    Returns:
        conv : bool
            Converged or not
        e : list of eigenvalues
            The eigenvalues can be sorted real or complex, depending on the
            return value of ``pick`` function.
        vl : list of 1D arrays
            Left eigenvectors. Only returned if ``left=True``.
        c : list of 1D arrays
            Right eigenvectors.

    Examples:

    >>> from pyscf import lib
    >>> a = numpy.random.random((10,10))
    >>> a = a
    >>> aop = lambda xs: [numpy.dot(a,x) for x in xs]
    >>> precond = lambda dx, e, x0: dx/(a.diagonal()-e)
    >>> x0 = a[0]
    >>> e, vl, vr = lib.davidson(aop, x0, precond, nroots=2, left=True)
    >>> len(e)
    2
    '''
    res = davidson_nosym1(lambda xs: [aop(x) for x in xs],
                          x0, precond, tol, max_cycle, max_space, lindep,
                          max_memory, dot, callback, nroots, lessio,
                          left, pick, verbose, follow_state)
    if left:
        e, vl, vr = res[1:]
        if nroots == 1:
            return e[0], vl[0], vr[0]
        else:
            return e, vl, vr
    else:
        e, x = res[1:]
        if nroots == 1:
            return e[0], x[0]
        else:
            return e, x
davidson_nosym = eig

def davidson_nosym1(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
                    lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
                    dot=numpy.dot, callback=None,
                    nroots=1, lessio=False, left=False, pick=pick_real_eigs,
                    verbose=logger.WARN, follow_state=FOLLOW_STATE,
                    tol_residual=None):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if tol_residual is None:
        toloose = numpy.sqrt(tol)
    else:
        toloose = tol_residual
    log.debug1('tol %g  toloose %g', tol, toloose)

    if not callable(precond):
        precond = make_diag_precond(precond)

    if callable(x0):
        x0 = x0()
    if isinstance(x0, numpy.ndarray) and x0.ndim == 1:
        x0 = [x0]
    #max_cycle = min(max_cycle, x0[0].size)
    max_space = max_space + (nroots-1) * 4
    # max_space*2 for holding ax and xs, nroots*2 for holding axt and xt
    _incore = max_memory*1e6/x0[0].nbytes > max_space*2+nroots*3
    lessio = lessio and not _incore
    log.debug1('max_cycle %d  max_space %d  max_memory %d  incore %s',
               max_cycle, max_space, max_memory, _incore)
    dtype = None
    heff = None
    fresh_start = True
    e = 0
    v = None
    conv = [False] * nroots
    emin = None
    norm_min = 1

    for icyc in range(max_cycle):
        if fresh_start:
            if _incore:
                xs = []
                ax = []
            else:
                xs = _Xlist()
                ax = _Xlist()
            space = 0
# Orthogonalize xt space because the basis of subspace xs must be orthogonal
# but the eigenvectors x0 might not be strictly orthogonal
            xt = None
            x0len = len(x0)
            xt, x0 = _qr(x0, dot, lindep)[0], None
            if len(xt) != x0len:
                log.warn('QR decomposition removed %d vectors.  The davidson may fail.'
                         'Check to see if `pick` function :%s: is providing linear dependent '
                         'vectors' % (x0len - len(xt), pick.__name__))
            max_dx_last = 1e9
            if SORT_EIG_BY_SIMILARITY:
                conv = [False] * nroots
        elif len(xt) > 1:
            xt = _qr(xt, dot, lindep)[0]
            xt = xt[:40]  # 40 trial vectors at most

        axt = aop(xt)
        for k, xi in enumerate(xt):
            xs.append(xt[k])
            ax.append(axt[k])
        rnow = len(xt)
        head, space = space, space+rnow

        if dtype is None:
            try:
                dtype = numpy.result_type(axt[0], xt[0])
            except IndexError:
                dtype = numpy.result_type(ax[0].dtype, xs[0].dtype)
        if heff is None:  # Lazy initilize heff to determine the dtype
            heff = numpy.empty((max_space+nroots,max_space+nroots), dtype=dtype)
        else:
            heff = numpy.asarray(heff, dtype=dtype)

        elast = e
        vlast = v
        conv_last = conv
        for i in range(rnow):
            for k in range(rnow):
                heff[head+k,head+i] = dot(xt[k].conj(), axt[i])
        for i in range(head):
            axi = numpy.asarray(ax[i])
            xi = numpy.asarray(xs[i])
            for k in range(rnow):
                heff[head+k,i] = dot(xt[k].conj(), axi)
                heff[i,head+k] = dot(xi.conj(), axt[k])
            axi = xi = None

        w, v = scipy.linalg.eig(heff[:space,:space])
        w, v, idx = pick(w, v, nroots, locals())
        if SORT_EIG_BY_SIMILARITY:
            e, v = _sort_by_similarity(w, v, nroots, conv, vlast, emin,
                                       heff[:space,:space])
            if e.size != elast.size:
                de = e
            else:
                de = e - elast
        else:
            e = w[:nroots]
            v = v[:,:nroots]

        x0 = _gen_x0(v, xs)
        if lessio:
            ax0 = aop(x0)
        else:
            ax0 = _gen_x0(v, ax)

        if SORT_EIG_BY_SIMILARITY:
            dx_norm = [0] * nroots
            xt = [None] * nroots
            for k, ek in enumerate(e):
                if not conv[k]:
                    xt[k] = ax0[k] - ek * x0[k]
                    dx_norm[k] = numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                    if abs(de[k]) < tol and dx_norm[k] < toloose:
                        log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                                  k, dx_norm[k], ek, de[k])
                        conv[k] = True
        else:
            elast, conv_last = _sort_elast(elast, conv_last, vlast, v,
                                           fresh_start, log)
            de = e - elast
            dx_norm = []
            xt = []
            for k, ek in enumerate(e):
                xt.append(ax0[k] - ek * x0[k])
                dx_norm.append(numpy.sqrt(dot(xt[k].conj(), xt[k]).real))
                if not conv_last[k] and abs(de[k]) < tol and dx_norm[k] < toloose:
                    log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                              k, dx_norm[k], ek, de[k])
            dx_norm = numpy.asarray(dx_norm)
            conv = (abs(de) < tol) & (dx_norm < toloose)
        ax0 = None
        max_dx_norm = max(dx_norm)
        ide = numpy.argmax(abs(de))
        if all(conv):
            log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, space, max_dx_norm, e, de[ide])
            break
        elif (follow_state and max_dx_norm > 1 and
              max_dx_norm/max_dx_last > 3 and space > nroots*3):
            log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                      icyc, space, max_dx_norm, e, de[ide], norm_min)
            log.debug('Large |r| detected, restore to previous x0')
            x0 = _gen_x0(vlast, xs)
            fresh_start = True
            continue

        if SORT_EIG_BY_SIMILARITY:
            if any(conv) and e.dtype == numpy.double:
                emin = min(e)

        # remove subspace linear dependency
        if any(((not conv[k]) and n**2>lindep) for k, n in enumerate(dx_norm)):
            for k, ek in enumerate(e):
                if (not conv[k]) and dx_norm[k]**2 > lindep:
                    xt[k] = precond(xt[k], e[0], x0[k])
                    xt[k] *= 1/numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                else:
                    xt[k] = None
                    log.debug1('Throwing out eigenvector %d with norm=%4.3g', k, dx_norm[k])
        else:
            for k, ek in enumerate(e):
                if dx_norm[k]**2 > lindep:
                    xt[k] = precond(xt[k], e[0], x0[k])
                    xt[k] *= 1/numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                else:
                    xt[k] = None
        xt = [xi for xi in xt if xi is not None]

        for i in range(space):
            xsi = numpy.asarray(xs[i])
            for xi in xt:
                xi -= xsi * dot(xsi.conj(), xi)
            xsi = None
        norm_min = 1
        for i,xi in enumerate(xt):
            norm = numpy.sqrt(dot(xi.conj(), xi).real)
            if norm**2 > lindep:
                xt[i] *= 1/norm
                norm_min = min(norm_min, norm)
            else:
                xt[i] = None
        xt = [xi for xi in xt if xi is not None]
        xi = None
        log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, space, max_dx_norm, e, de[ide], norm_min)
        if len(xt) == 0:
            log.debug('Linear dependency in trial subspace. |r| for each state %s',
                     dx_norm)
            conv = [conv[k] or (norm < toloose) for k,norm in enumerate(dx_norm)]
            break

        max_dx_last = max_dx_norm
        fresh_start = space+nroots > max_space

        if callable(callback):
            callback(locals())

    xnorm = numpy.array([numpy_helper.norm(x) for x in x0])
    enorm = xnorm < 1e-6
    if numpy.any(enorm):
        warnings.warn("{:d} davidson root{_s}: {} {_has} very small norm{_s}: {}".format(
            enorm.sum(),
            ", ".join("#{:d}".format(i) for i in numpy.argwhere(enorm)[:, 0]),
            ", ".join("{:.3e}".format(i) for i in xnorm[enorm]),
            _s='s' if enorm.sum() > 1 else "",
            _has="have" if enorm.sum() > 1 else "has a",
        ))

    if left:
        warnings.warn('Left eigenvectors from subspace diagonalization method may not be converged')
        w, vl, v = scipy.linalg.eig(heff[:space,:space], left=True)
        e, v, idx = pick(w, v, nroots, locals())
        xl = _gen_x0(vl[:,idx[:nroots]].conj(), xs)
        x0 = _gen_x0(v[:,:nroots], xs)
        xl = [x for x in xl]  # nparray -> list
        x0 = [x for x in x0]  # nparray -> list
        return numpy.asarray(conv), e[:nroots], xl, x0
    else:
        x0 = [x for x in x0]  # nparray -> list
        return numpy.asarray(conv), e, x0

def dgeev(abop, x0, precond, type=1, tol=1e-12, max_cycle=50, max_space=12,
          lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
          dot=numpy.dot, callback=None,
          nroots=1, lessio=False, verbose=logger.WARN):
    '''Davidson diagonalization method to solve  A c = e B c.

    Args:
        abop : function(x) => (array_like_x, array_like_x)
            abop applies two matrix vector multiplications and returns tuple (Ax, Bx)
        x0 : 1D array
            Initial guess
        precond : diagonal elements of the matrix or  function(dx, e, x0) => array_like_dx
            Preconditioner to generate new trial vector.
            The argument dx is a residual vector ``a*x0-e*x0``; e is the current
            eigenvalue; x0 is the current eigenvector.

    Kwargs:
        tol : float
            Convergence tolerance.
        max_cycle : int
            max number of iterations.
        max_space : int
            space size to hold trial vectors.
        lindep : float
            Linear dependency threshold.  The function is terminated when the
            smallest eigenvalue of the metric of the trial vectors is lower
            than this threshold.
        max_memory : int or float
            Allowed memory in MB.
        dot : function(x, y) => scalar
            Inner product
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            envrionment.
        nroots : int
            Number of eigenvalues to be computed.  When nroots > 1, it affects
            the shape of the return value
        lessio : bool
            How to compute a*x0 for current eigenvector x0.  There are two
            ways to compute a*x0.  One is to assemble the existed a*x.  The
            other is to call aop(x0).  The default is the first method which
            needs more IO and less computational cost.  When IO is slow, the
            second method can be considered.

    Returns:
        e : list of floats
            The lowest :attr:`nroots` eigenvalues.
        c : list of 1D arrays
            The lowest :attr:`nroots` eigenvectors.
    '''
    def map_abop(xs):
        ab = [abop(x) for x in xs]
        alst = [x[0] for x in ab]
        blst = [x[1] for x in ab]
        return alst, blst
    e, x = dgeev1(map_abop, x0, precond, type, tol, max_cycle, max_space, lindep,
                  max_memory, dot, callback, nroots, lessio, verbose)[1:]
    if nroots == 1:
        return e[0], x[0]
    else:
        return e, x

def dgeev1(abop, x0, precond, type=1, tol=1e-12, max_cycle=50, max_space=12,
          lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
          dot=numpy.dot, callback=None,
          nroots=1, lessio=False, verbose=logger.WARN, tol_residual=None):
    '''Davidson diagonalization method to solve  A c = e B c.

    Args:
        abop : function([x]) => ([array_like_x], [array_like_x])
            abop applies two matrix vector multiplications and returns tuple (Ax, Bx)
        x0 : 1D array
            Initial guess
        precond : diagonal elements of the matrix or  function(dx, e, x0) => array_like_dx
            Preconditioner to generate new trial vector.
            The argument dx is a residual vector ``a*x0-e*x0``; e is the current
            eigenvalue; x0 is the current eigenvector.

    Kwargs:
        tol : float
            Convergence tolerance.
        max_cycle : int
            max number of iterations.
        max_space : int
            space size to hold trial vectors.
        lindep : float
            Linear dependency threshold.  The function is terminated when the
            smallest eigenvalue of the metric of the trial vectors is lower
            than this threshold.
        max_memory : int or float
            Allowed memory in MB.
        dot : function(x, y) => scalar
            Inner product
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            envrionment.
        nroots : int
            Number of eigenvalues to be computed.  When nroots > 1, it affects
            the shape of the return value
        lessio : bool
            How to compute a*x0 for current eigenvector x0.  There are two
            ways to compute a*x0.  One is to assemble the existed a*x.  The
            other is to call aop(x0).  The default is the first method which
            needs more IO and less computational cost.  When IO is slow, the
            second method can be considered.

    Returns:
        conv : bool
            Converged or not
        e : list of floats
            The lowest :attr:`nroots` eigenvalues.
        c : list of 1D arrays
            The lowest :attr:`nroots` eigenvectors.
    '''
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if tol_residual is None:
        toloose = numpy.sqrt(tol) * 1e-2
    else:
        toloose = tol_residual

    if not callable(precond):
        precond = make_diag_precond(precond)

    if isinstance(x0, numpy.ndarray) and x0.ndim == 1:
        x0 = [x0]
    #max_cycle = min(max_cycle, x0[0].size)
    max_space = max_space + (nroots-1) * 3
    # max_space*3 for holding ax, bx and xs, nroots*3 for holding axt, bxt and xt
    _incore = max_memory*1e6/x0[0].nbytes > max_space*3+nroots*3
    lessio = lessio and not _incore
    heff = numpy.empty((max_space,max_space), dtype=x0[0].dtype)
    seff = numpy.empty((max_space,max_space), dtype=x0[0].dtype)
    fresh_start = True
    conv = False

    for icyc in range(max_cycle):
        if fresh_start:
            if _incore:
                xs = []
                ax = []
                bx = []
            else:
                xs = _Xlist()
                ax = _Xlist()
                bx = _Xlist()
            space = 0
# Orthogonalize xt space because the basis of subspace xs must be orthogonal
# but the eigenvectors x0 are very likely non-orthogonal when A is non-Hermitian.
            xt, x0 = _qr(x0, dot, lindep)[0], None
            e = numpy.zeros(nroots)
            fresh_start = False
        elif len(xt) > 1:
            xt = _qr(xt, dot, lindep)[0]
            xt = xt[:40]  # 40 trial vectors at most

        axt, bxt = abop(xt)
        if type > 1:
            axt = abop(bxt)[0]
        for k, xi in enumerate(xt):
            xs.append(xt[k])
            ax.append(axt[k])
            bx.append(bxt[k])
        rnow = len(xt)
        head, space = space, space+rnow

        if type == 1:
            for i in range(space):
                if head <= i < head+rnow:
                    for k in range(i-head+1):
                        heff[head+k,i] = dot(xt[k].conj(), axt[i-head])
                        heff[i,head+k] = heff[head+k,i].conj()
                        seff[head+k,i] = dot(xt[k].conj(), bxt[i-head])
                        seff[i,head+k] = seff[head+k,i].conj()
                else:
                    axi = numpy.asarray(ax[i])
                    bxi = numpy.asarray(bx[i])
                    for k in range(rnow):
                        heff[head+k,i] = dot(xt[k].conj(), axi)
                        heff[i,head+k] = heff[head+k,i].conj()
                        seff[head+k,i] = dot(xt[k].conj(), bxi)
                        seff[i,head+k] = seff[head+k,i].conj()
                axi = bxi = None
        else:
            for i in range(space):
                if head <= i < head+rnow:
                    for k in range(i-head+1):
                        heff[head+k,i] = dot(bxt[k].conj(), axt[i-head])
                        heff[i,head+k] = heff[head+k,i].conj()
                        seff[head+k,i] = dot(xt[k].conj(), bxt[i-head])
                        seff[i,head+k] = seff[head+k,i].conj()
                else:
                    axi = numpy.asarray(ax[i])
                    bxi = numpy.asarray(bx[i])
                    for k in range(rnow):
                        heff[head+k,i] = dot(bxt[k].conj(), axi)
                        heff[i,head+k] = heff[head+k,i].conj()
                        seff[head+k,i] = dot(xt[k].conj(), bxi)
                        seff[i,head+k] = seff[head+k,i].conj()
                axi = bxi = None

        w, v = scipy.linalg.eigh(heff[:space,:space], seff[:space,:space])
        if space < nroots or e.size != nroots:
            de = w[:nroots]
        else:
            de = w[:nroots] - e
        e = w[:nroots]

        x0 = _gen_x0(v[:,:nroots], xs)
        if lessio:
            ax0, bx0 = abop(x0)
            if type > 1:
                ax0 = abop(bx0)[0]
        else:
            ax0 = _gen_x0(v[:,:nroots], ax)
            bx0 = _gen_x0(v[:,:nroots], bx)

        ide = numpy.argmax(abs(de))
        if abs(de[ide]) < tol:
            log.debug('converged %d %d  e= %s  max|de|= %4.3g',
                      icyc, space, e, de[ide])
            conv = True
            break

        dx_norm = []
        xt = []
        for k, ek in enumerate(e):
            if type == 1:
                dxtmp = ax0[k] - ek * bx0[k]
            else:
                dxtmp = ax0[k] - ek * x0[k]
            xt.append(dxtmp)
            dx_norm.append(numpy_helper.norm(dxtmp))
        ax0 = bx0 = None

        if max(dx_norm) < toloose:
            log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, space, max(dx_norm), e, de[ide])
            conv = True
            break

        # remove subspace linear dependency
        for k, ek in enumerate(e):
            if dx_norm[k] > toloose:
                xt[k] = precond(xt[k], e[0], x0[k])
                xt[k] *= 1/numpy_helper.norm(xt[k])
            else:
                xt[k] = None
        xt = [xi for xi in xt if xi is not None]
        for i in range(space):
            xsi = numpy.asarray(xs[i])
            for xi in xt:
                xi -= xsi * numpy.dot(xi, xsi)
            xsi = None
        norm_min = 1
        for i,xi in enumerate(xt):
            norm = numpy_helper.norm(xi)
            if norm > toloose:
                xt[i] *= 1/norm
                norm_min = min(norm_min, norm)
            else:
                xt[i] = None
        xt = [xi for xi in xt if xi is not None]
        log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, space, max(dx_norm), e, de[ide], norm)
        if len(xt) == 0:
            log.debug('Linear dependency in trial subspace. |r| for each state %s',
                     dx_norm)
            conv = all(norm < toloose for norm in dx_norm)
            break

        fresh_start = fresh_start or (space+len(xt) > max_space)

        if callable(callback):
            callback(locals())

    if type == 3:
        for k in range(nroots):
            x0[k] = abop(x0[k])[1]

    x0 = [x for x in x0]  # nparray -> list
    return conv, e, x0


# TODO: new solver with Arnoldi iteration
# The current implementation fails in polarizability. see
# https://github.com/pyscf/pyscf/issues/507
def krylov(aop, b, x0=None, tol=1e-10, max_cycle=30, dot=numpy.dot,
           lindep=DSOLVE_LINDEP, callback=None, hermi=False,
           max_memory=MAX_MEMORY, verbose=logger.WARN):
    '''Krylov subspace method to solve  (1+a) x = b.  Ref:
    J. A. Pople et al, Int. J.  Quantum. Chem.  Symp. 13, 225 (1979).

    Args:
        aop : function(x) => array_like_x
            aop(x) to mimic the matrix vector multiplication :math:`\sum_{j}a_{ij} x_j`.
            The argument is a 1D array.  The returned value is a 1D array.
        b : a vector or a list of vectors

    Kwargs:
        x0 : 1D array
            Initial guess
        tol : float
            Tolerance to terminate the operation aop(x).
        max_cycle : int
            max number of iterations.
        lindep : float
            Linear dependency threshold.  The function is terminated when the
            smallest eigenvalue of the metric of the trial vectors is lower
            than this threshold.
        dot : function(x, y) => scalar
            Inner product
        callback : function(envs_dict) => None
            callback function takes one dict as the argument which is
            generated by the builtin function :func:`locals`, so that the
            callback function can access all local variables in the current
            envrionment.

    Returns:
        x : ndarray like b

    Examples:

    >>> from pyscf import lib
    >>> a = numpy.random.random((10,10)) * 1e-2
    >>> b = numpy.random.random(10)
    >>> aop = lambda x: numpy.dot(a,x)
    >>> x = lib.krylov(aop, b)
    >>> numpy.allclose(numpy.dot(a,x)+x, b)
    True
    '''
    if isinstance(aop, numpy.ndarray) and aop.ndim == 2:
        return numpy.linalg.solve(aop+numpy.eye(aop.shape[0]), b)

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if not (isinstance(b, numpy.ndarray) and b.ndim == 1):
        b = numpy.asarray(b)

    if x0 is None:
        x1 = b
    else:
        b = b - (x0 + aop(x0))
        x1 = b
    if x1.ndim == 1:
        x1 = x1.reshape(1, x1.size)
    nroots, ndim = x1.shape

    # Not exactly QR, vectors are orthogonal but not normalized
    x1, rmat = _qr(x1, dot, lindep)
    for i in range(len(x1)):
        x1[i] *= rmat[i,i]

    innerprod = [dot(xi.conj(), xi).real for xi in x1]
    max_innerprod = max(innerprod)
    if max_innerprod < lindep or max_innerprod < tol**2:
        if x0 is None:
            return numpy.zeros_like(b)
        else:
            return x0

    _incore = max_memory*1e6/b.nbytes > 14
    log.debug1('max_memory %d  incore %s', max_memory, _incore)
    if _incore:
        xs = []
        ax = []
    else:
        xs = _Xlist()
        ax = _Xlist()

    max_cycle = min(max_cycle, ndim)
    for cycle in range(max_cycle):
        axt = aop(x1)
        if axt.ndim == 1:
            axt = axt.reshape(1,ndim)
        xs.extend(x1)
        ax.extend(axt)
        if callable(callback):
            callback(cycle, xs, ax)

        x1 = axt.copy()
        for i in range(len(xs)):
            xsi = numpy.asarray(xs[i])
            for j, axj in enumerate(axt):
                x1[j] -= xsi * (dot(xsi.conj(), axj) / innerprod[i])
        axt = None

        max_innerprod = 0
        idx = []
        for i, xi in enumerate(x1):
            innerprod1 = dot(xi.conj(), xi).real
            max_innerprod = max(max_innerprod, innerprod1)
            if innerprod1 > lindep and innerprod1 > tol**2:
                idx.append(i)
                innerprod.append(innerprod1)
        log.debug('krylov cycle %d  r = %g', cycle, max_innerprod**.5)
        if max_innerprod < lindep or max_innerprod < tol**2:
            break

        x1 = x1[idx]

    nd = cycle + 1
    h = numpy.empty((nd,nd), dtype=x1.dtype)

    for i in range(nd):
        xi = numpy.asarray(xs[i])
        if hermi:
            for j in range(i+1):
                h[i,j] = dot(xi.conj(), ax[j])
                h[j,i] = h[i,j].conj()
        else:
            for j in range(nd):
                h[i,j] = dot(xi.conj(), ax[j])
        xi = None

    # Add the contribution of I in (1+a)
    for i in range(nd):
        h[i,i] += innerprod[i]

    g = numpy.zeros((nd,nroots), dtype=x1.dtype)
    if b.ndim == 1:
        g[0] = innerprod[0]
    else:
        # Restore the first nroots vectors, which are array b or b-(1+a)x0
        for i in range(min(nd, nroots)):
            xsi = numpy.asarray(xs[i])
            for j in range(nroots):
                g[i,j] = dot(xsi.conj(), b[j])

    c = numpy.linalg.solve(h, g)
    x = _gen_x0(c, xs)
    if b.ndim == 1:
        x = x[0]

    if x0 is not None:
        x += x0
    return x


def dsolve(aop, b, precond, tol=1e-12, max_cycle=30, dot=numpy.dot,
           lindep=DSOLVE_LINDEP, verbose=0, tol_residual=None):
    '''Davidson iteration to solve linear equation.  It works bad.
    '''

    if tol_residual is None:
        toloose = numpy.sqrt(tol)
    else:
        toloose = tol_residual

    assert callable(precond)

    xs = [precond(b)]
    ax = [aop(xs[-1])]

    dtype = numpy.result_type(ax[0], xs[0])
    aeff = numpy.zeros((max_cycle,max_cycle), dtype=dtype)
    beff = numpy.zeros((max_cycle), dtype=dtype)
    for istep in range(max_cycle):
        beff[istep] = dot(xs[istep], b)
        for i in range(istep+1):
            aeff[istep,i] = dot(xs[istep], ax[i])
            aeff[i,istep] = dot(xs[i], ax[istep])

        v = scipy.linalg.solve(aeff[:istep+1,:istep+1], beff[:istep+1])
        xtrial = dot(v, xs)
        dx = b - dot(v, ax)
        rr = numpy_helper.norm(dx)
        if verbose:
            print('davidson', istep, rr)
        if rr < toloose:
            break
        xs.append(precond(dx))
        ax.append(aop(xs[-1]))

    if verbose:
        print(istep)

    return xtrial


def cho_solve(a, b, strict_sym_pos=True):
    '''Solve ax = b, where a is a postive definite hermitian matrix

    Kwargs:
        strict_sym_pos (bool) : Whether to impose the strict positive definition
            on matrix a
    '''
    try:
        return scipy.linalg.solve(a, b, sym_pos=True)
    except numpy.linalg.LinAlgError:
        if strict_sym_pos:
            raise
        else:
            fname, lineno = inspect.stack()[1][1:3]
            warnings.warn('%s:%s: matrix a is not strictly postive definite' %
                          (fname, lineno))
            return scipy.linalg.solve(a, b)


def _qr(xs, dot, lindep=1e-14):
    '''QR decomposition for a list of vectors (for linearly independent vectors only).
    xs = (r.T).dot(qs)
    '''
    nvec = len(xs)
    dtype = xs[0].dtype
    qs = numpy.empty((nvec,xs[0].size), dtype=dtype)
    rmat = numpy.empty((nvec,nvec), order='F', dtype=dtype)

    nv = 0
    for i in range(nvec):
        xi = numpy.array(xs[i], copy=True)
        rmat[:,nv] = 0
        rmat[nv,nv] = 1
        for j in range(nv):
            prod = dot(qs[j].conj(), xi)
            xi -= qs[j] * prod
            rmat[:,nv] -= rmat[:,j] * prod
        innerprod = dot(xi.conj(), xi).real
        norm = numpy.sqrt(innerprod)
        if innerprod > lindep:
            qs[nv] = xi/norm
            rmat[:nv+1,nv] /= norm
            nv += 1
    return qs[:nv], numpy.linalg.inv(rmat[:nv,:nv])

def _gen_x0(v, xs):
    space, nroots = v.shape
    x0 = numpy.einsum('c,x->cx', v[space-1], numpy.asarray(xs[space-1]))
    for i in reversed(range(space-1)):
        xsi = numpy.asarray(xs[i])
        for k in range(nroots):
            x0[k] += v[i,k] * xsi
    return x0

def _sort_by_similarity(w, v, nroots, conv, vlast, emin=None, heff=None):
    if not any(conv) or vlast is None:
        return w[:nroots], v[:,:nroots]

    head, nroots = vlast.shape
    conv = numpy.asarray(conv[:nroots])
    ovlp = vlast[:,conv].T.conj().dot(v[:head])
    ovlp = numpy.einsum('ij,ij->j', ovlp, ovlp)
    nconv = numpy.count_nonzero(conv)
    nleft = nroots - nconv
    idx = ovlp.argsort()
    sorted_idx = numpy.zeros(nroots, dtype=int)
    sorted_idx[conv] = numpy.sort(idx[-nconv:])
    sorted_idx[~conv] = numpy.sort(idx[:-nconv])[:nleft]

    e = w[sorted_idx]
    c = v[:,sorted_idx]
    return e, c

def _sort_elast(elast, conv_last, vlast, v, fresh_start, log):
    '''
    Eigenstates may be flipped during the Davidson iterations.  Reorder the
    eigenvalues of last iteration to make them comparable to the eigenvalues
    of the current iterations.
    '''
    if fresh_start:
        return elast, conv_last
    head, nroots = vlast.shape
    ovlp = abs(numpy.dot(v[:head].conj().T, vlast))
    idx = numpy.argmax(ovlp, axis=1)

    if log.verbose >= logger.DEBUG:
        ordering_diff = (idx != numpy.arange(len(idx)))
        if numpy.any(ordering_diff):
            log.debug('Old state -> New state')
            for i in numpy.where(ordering_diff)[0]:
                log.debug('  %3d     ->   %3d ', idx[i], i)

    return [elast[i] for i in idx], [conv_last[i] for i in idx]


class _Xlist(list):
    def __init__(self):
        self.scr_h5 = misc.H5TmpFile()
        self.index = []

    def __getitem__(self, n):
        key = self.index[n]
        return self.scr_h5[str(key)]

    def append(self, x):
        length = len(self.index)
        key = length + 1
        index_set = set(self.index)
        if key in index_set:
            key = set(range(length)).difference(index_set).pop()
        self.index.append(key)

        self.scr_h5[str(key)] = x
        self.scr_h5.flush()

    def extend(self, x):
        for xi in x:
            self.append(xi)

    def __setitem__(self, n, x):
        key = self.index[n]
        self.scr_h5[str(key)][:] = x
        self.scr_h5.flush()

    def __len__(self):
        return len(self.index)

    def pop(self, index):
        key = self.index.pop(index)
        del(self.scr_h5[str(key)])

del(SAFE_EIGH_LINDEP, DAVIDSON_LINDEP, DSOLVE_LINDEP, MAX_MEMORY)


if __name__ == '__main__':
    a = numpy.random.random((9,5))+numpy.random.random((9,5))*1j
    q, r = _qr(a.T, numpy.dot)
    print(abs(r.T.dot(q)-a.T).max())

    numpy.random.seed(12)
    n = 1000
    #a = numpy.random.random((n,n))
    a = numpy.arange(n*n).reshape(n,n)
    a = numpy.sin(numpy.sin(a)) + a*1e-3j
    a = a + a.T.conj() + numpy.diag(numpy.random.random(n))*10

    e,u = scipy.linalg.eigh(a)
    #a = numpy.dot(u[:,:15]*e[:15], u[:,:15].T)
    print(e[0], u[0,0])

    def aop(x):
        return numpy.dot(a, x)

    def precond(r, e0, x0):
        idx = numpy.argwhere(abs(x0)>.1).ravel()
        #idx = numpy.arange(20)
        m = idx.size
        if m > 2:
            h0 = a[idx][:,idx] - numpy.eye(m)*e0
            h0x0 = x0 / (a.diagonal() - e0)
            h0x0[idx] = numpy.linalg.solve(h0, h0x0[idx])
            h0r = r / (a.diagonal() - e0)
            h0r[idx] = numpy.linalg.solve(h0, r[idx])
            e1 = numpy.dot(x0, h0r) / numpy.dot(x0, h0x0)
            x1 = (r - e1*x0) / (a.diagonal() - e0)
            x1[idx] = numpy.linalg.solve(h0, (r-e1*x0)[idx])
            return x1
        else:
            return r / (a.diagonal() - e0)

    x0 = [a[0]/numpy.linalg.norm(a[0]),
          a[1]/numpy.linalg.norm(a[1]),
          a[2]/numpy.linalg.norm(a[2]),
          a[3]/numpy.linalg.norm(a[3])]
    e0,x0 = dsyev(aop, x0, precond, max_cycle=30, max_space=12,
                  max_memory=.0001, verbose=5, nroots=4, follow_state=True)
    print(e0[0] - e[0])
    print(e0[1] - e[1])
    print(e0[2] - e[2])
    print(e0[3] - e[3])

##########
    a = a + numpy.diag(numpy.random.random(n)+1.1)* 10
    b = numpy.random.random(n)
    def aop(x):
        return numpy.dot(a,x)
    def precond(x, *args):
        return x / a.diagonal()
    x = numpy.linalg.solve(a, b)
    x1 = dsolve(aop, b, precond, max_cycle=50)
    print(abs(x - x1).sum())
    a_diag = a.diagonal()
    log = logger.Logger(sys.stdout, 5)
    aop = lambda x: numpy.dot(a-numpy.diag(a_diag), x.ravel())/a_diag
    x1 = krylov(aop, b/a_diag, max_cycle=50, verbose=log)
    print(abs(x - x1).sum())
    x1 = krylov(aop, b/a_diag, None, max_cycle=10, verbose=log)
    x1 = krylov(aop, b/a_diag, x1, max_cycle=30, verbose=log)
    print(abs(x - x1).sum())

##########
    numpy.random.seed(12)
    n = 500
    #a = numpy.random.random((n,n))
    a = numpy.arange(n*n).reshape(n,n)
    a = numpy.sin(numpy.sin(a))
    a = a + a.T + numpy.diag(numpy.random.random(n))*10
    b = numpy.random.random((n,n))
    b = numpy.dot(b,b.T) + numpy.eye(n)*5

    def abop(x):
        return numpy.dot(numpy.asarray(x), a.T), numpy.dot(numpy.asarray(x), b.T)

    def precond(r, e0, x0):
        return r / (a.diagonal() - e0)

    e,u = scipy.linalg.eigh(a, b)
    x0 = [a[0]/numpy.linalg.norm(a[0]),
          a[1]/numpy.linalg.norm(a[1]),]
    e0,x0 = dgeev1(abop, x0, precond, type=1, max_cycle=100, max_space=18,
                   verbose=5, nroots=4)[1:]
    print(e0[0] - e[0])
    print(e0[1] - e[1])
    print(e0[2] - e[2])
    print(e0[3] - e[3])


    e,u = scipy.linalg.eigh(a, b, type=2)
    x0 = [a[0]/numpy.linalg.norm(a[0]),
          a[1]/numpy.linalg.norm(a[1]),]
    e0,x0 = dgeev1(abop, x0, precond, type=2, max_cycle=100, max_space=18,
                   verbose=5, nroots=4)[1:]
    print(e0[0] - e[0])
    print(e0[1] - e[1])
    print(e0[2] - e[2])
    print(e0[3] - e[3])

    e,u = scipy.linalg.eigh(a, b, type=2)
    x0 = [a[0]/numpy.linalg.norm(a[0]),
          a[1]/numpy.linalg.norm(a[1]),]
    abdiag = numpy.dot(a,b).diagonal().copy()
    def abop(x):
        x = numpy.asarray(x).T
        return numpy.dot(a, numpy.dot(b, x)).T.copy()
    def precond(r, e0, x0):
        return r / (abdiag-e0)
    e0, x0 = eig(abop, x0, precond, max_cycle=100, max_space=30, verbose=5,
                 nroots=4, pick=pick_real_eigs)
    print(e0[0] - e[0])
    print(e0[1] - e[1])
    print(e0[2] - e[2])
    print(e0[3] - e[3])

    e, ul, u = scipy.linalg.eig(numpy.dot(a, b), left=True)
    idx = numpy.argsort(e)
    e = e[idx]
    ul = ul[:,idx]
    u  = u [:,idx]
    u  /= numpy.linalg.norm(u, axis=0)
    ul /= numpy.linalg.norm(ul, axis=0)
    x0 = [a[0]/numpy.linalg.norm(a[0]),
          a[1]/numpy.linalg.norm(a[1]),]
    abdiag = numpy.dot(a,b).diagonal().copy()
    e0, vl, vr = eig(abop, x0, precond, max_cycle=100, max_space=30, verbose=5,
                     nroots=4, pick=pick_real_eigs, left=True)
    print(e0[0] - e[0])
    print(e0[1] - e[1])
    print(e0[2] - e[2])
    print(e0[3] - e[3])
    print((abs(vr[0]) - abs(u[:,0])).sum())
    print((abs(vr[1]) - abs(u[:,1])).sum())
    print((abs(vr[2]) - abs(u[:,2])).sum())
    print((abs(vr[3]) - abs(u[:,3])).sum())
#    print((abs(vl[0]) - abs(ul[:,0])).max())
#    print((abs(vl[1]) - abs(ul[:,1])).max())
#    print((abs(vl[2]) - abs(ul[:,2])).max())
#    print((abs(vl[3]) - abs(ul[:,3])).max())

##########
    N = 200
    neig = 4
    A = numpy.zeros((N,N))
    k = N/2
    for ii in range(N):
        i = ii+1
        for jj in range(N):
            j = jj+1
            if j <= k:
                A[ii,jj] = i*(i==j)-(i-j-k**2)
            else:
                A[ii,jj] = i*(i==j)+(i-j-k**2)
    def matvec(x):
        return numpy.dot(A,x)

    def precond(r, e0, x0):
        return (r+e0*x0) / A.diagonal()  # Converged
        #return (r+e0*x0) / (A.diagonal()-e0)  # Does not converge
        #return r / (A.diagonal()-e0)  # Does not converge
    e, c = eig(matvec, A[:,0], precond, nroots=4, verbose=5,
                   max_cycle=200,max_space=40, tol=1e-5)
    print("# davidson evals =", e)

