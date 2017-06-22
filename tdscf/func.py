import numpy as np
import os,sys
import ctypes
from pyscf import lib

OCCDROP = 1e-12
BLKSIZE = 96
SWITCH_SIZE = 800


def load_library(libname):
    if '1.6' in np.__version__:
        if (sys.platform.startswith('linux') or
            sys.platform.startswith('gnukfreebsd')):
            so_ext = '.so'
        elif sys.platform.startswith('darwin'):
            so_ext = '.dylib'
        elif sys.platform.startswith('win'):
            so_ext = '.dll'
        else:
            raise OSError('Unknown platform')
        libname_so = libname + so_ext
        return ctypes.CDLL(os.path.join(os.path.dirname(__file__), libname_so))
    else:
        _loaderpath = os.path.dirname(__file__)
        return np.ctypeslib.load_library(libname, _loaderpath)

libdft = lib.load_library('libdft')

def eval_rhoc(mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
              verbose=None):

#print ao
#   print ao.flags.c_contiguous
#    assert(ao.flags.c_contiguous)
    xctype = xctype.upper()
    if xctype == 'LDA':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = np.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=np.int8)

    pos = mo_occ.real > OCCDROP
    cpos = np.einsum('ij,j->ij', mo_coeff[:,pos], np.sqrt(mo_occ[pos]))
    if pos.sum() > 0:
        if xctype == 'LDA':
            c0 = Z_dot_ao_dm(mol, ao, cpos, nao, ngrids, non0tab)
            rho = np.einsum('pi,pi->p', c0, c0.conj())
        elif xctype == 'GGA':
            rho = np.empty((4,ngrids)).astype(complex)
            c0 = Z_dot_ao_dm(mol, ao[0], cpos, nao, ngrids, non0tab)
            rho[0] = np.einsum('pi,pi->p', c0, c0.conj())
            for i in range(1, 4):
                c1 = Z_dot_ao_dm(mol, ao[i], cpos, nao, ngrids, non0tab)
                rho[i] = np.einsum('pi,pi->p', c0, c1.conj()) * 2 # *2 for +c.c.
        else: # meta-GGA
            rho = np.empty((6,ngrids))
            c0 = Z_dot_ao_dm(mol, ao[0], cpos, nao, ngrids, non0tab)
            rho[0] = np.einsum('pi,pi->p', c0, c0.conj())
            rho[5] = 0
            for i in range(1, 4):
                c1 = Z_dot_ao_dm(mol, ao[i], cpos, nao, ngrids, non0tab)
                rho[i] = np.einsum('pi,pi->p', c0, c1.conj()) * 2 # *2 for +c.c.
                rho[5] += np.einsum('pi,pi->p', c1, c1.conj())
            XX, YY, ZZ = 4, 7, 9
            ao2 = ao[XX] + ao[YY] + ao[ZZ]
            c1 = Z_dot_ao_dm(mol, ao2.real, cpos.real, nao, ngrids, non0tab)
            rho[4] = np.einsum('pi,pi->p', c0, c1.conj())
            rho[4] += rho[5]
            rho[4] *= 2

            rho[5] *= .5
    else:
        if xctype == 'LDA':
            rho = np.zeros(ngrids)
        elif xctype == 'GGA':
            rho = np.zeros((4,ngrids))
        else:
            rho = np.zeros((6,ngrids))

    neg = mo_occ.real < -OCCDROP
    if neg.sum() > 0:
        cneg = np.einsum('ij,j->ij', mo_coeff[:,neg], np.sqrt(-mo_occ[neg]))
        if xctype == 'LDA':
            c0 = Z_dot_ao_dm(mol, ao, cneg, nao, ngrids, non0tab)
            rho -= np.einsum('pi,pi->p', c0, c0.conj())
        elif xctype == 'GGA':
            c0 = Z_dot_ao_dm(mol, ao[0], cneg, nao, ngrids, non0tab)
            rho[0] -= np.einsum('pi,pi->p', c0, c0.conj())
            for i in range(1, 4):
                c1 = Z_dot_ao_dm(mol, ao[i], cneg, nao, ngrids, non0tab)
                rho[i] -= np.einsum('pi,pi->p', c0, c1.conj()) * 2 # *2 for +c.c.
        else:
            c0 = Z_dot_ao_dm(mol, ao[0], cneg, nao, ngrids, non0tab)
            rho[0] -= np.einsum('pi,pi->p', c0, c0.conj())
            rho5 = 0
            for i in range(1, 4):
                c1 = Z_dot_ao_dm(mol, ao[i], cneg, nao, ngrids, non0tab)
                rho[i] -= np.einsum('pi,pi->p', c0, c1.conj()) * 2 # *2 for +c.c.
                rho5 -= np.einsum('pi,pi->p', c1, c1.conj())
            XX, YY, ZZ = 4, 7, 9
            ao2 = ao[XX] + ao[YY] + ao[ZZ]
            c1 = Z_dot_ao_dm(mol, ao2, cneg, nao, ngrids, non0tab)
            rho[4] -= np.einsum('pi,pi->p', c0, c1.conj()) * 2
            rho[4] -= rho5 * 2

            rho[5] -= rho5 * .5
    return rho.real

def TransMat(M,U,inv = 1):
    if inv == 1:
        # U.t() * M * U
        Mtilde = np.dot(np.dot(U.T.conj(),M),U)
    elif inv == -1:
        # U * M * U.t()
        Mtilde = np.dot(np.dot(U,M),U.T.conj())
    return Mtilde

def TrDot(A,B):
    C = np.trace(np.dot(A,B))
    return C

def MatrixPower(A,p,PrintCondition=False):
	''' Raise a Hermitian Matrix to a possibly fractional power. '''
	u,s,v = np.linalg.svd(A)
	if (PrintCondition):
		print "MatrixPower: Minimal Eigenvalue =", np.min(s)
	for i in range(len(s)):
		if (abs(s[i]) < np.power(10.0,-14.0)):
			s[i] = np.power(10.0,-14.0)
	return np.dot(u,np.dot(np.diag(np.power(s,p)),v))

#def _dot_ao_ao(mol, ao1, ao2, nao, ngrids, non0tab):
#    '''return numpy.dot(ao1.T, ao2) Copied from dft'''
#    natm = ctypes.c_int(mol._atm.shape[0])
#    nbas = ctypes.c_int(mol.nbas)
#    ao1 = np.asarray(ao1, order='C')
#    ao2 = np.asarray(ao2, order='C')
#    vv = np.empty((nao,nao))
#    libdft.VXCdot_ao_ao(vv.ctypes.data_as(ctypes.c_void_p),
#                        ao1.ctypes.data_as(ctypes.c_void_p),
#                        ao2.ctypes.data_as(ctypes.c_void_p),
#                        ctypes.c_int(nao), ctypes.c_int(ngrids),
#                        ctypes.c_int(BLKSIZE),
#                        non0tab.ctypes.data_as(ctypes.c_void_p),
#                        mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
#                        mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
#                        mol._env.ctypes.data_as(ctypes.c_void_p))
#    return vv

def _dot_ao_ao(mol, ao1, ao2, non0tab, shls_slice, ao_loc, hermi=0):
    '''return numpy.dot(ao1.T, ao2)'''
    ngrids, nao = ao1.shape
    if nao < SWITCH_SIZE:
        return lib.dot(ao1.T.conj(), ao2)
    if not ao1.flags.f_contiguous:
        ao1 = lib.transpose(ao1)
    if not ao2.flags.f_contiguous:
        ao2 = lib.transpose(ao2)
    if ao1.dtype == ao2.dtype == numpy.double:
        fn = libdft.VXCdot_ao_ao
    else:
        fn = libdft.VXCzdot_ao_ao
        ao1 = numpy.asarray(ao1, numpy.complex128)
        ao2 = numpy.asarray(ao2, numpy.complex128)
    vv = numpy.empty((nao,nao), dtype=ao1.dtype)
    fn(vv.ctypes.data_as(ctypes.c_void_p),
        ao1.ctypes.data_as(ctypes.c_void_p),
        ao2.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao), ctypes.c_int(ngrids),
        ctypes.c_int(mol.nbas), ctypes.c_int(hermi),
        non0tab.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int*2)(*shls_slice),
        ao_loc.ctypes.data_as(ctypes.c_void_p))
    return vv


def Z_dot_ao_dm(mol, ao, dm, nao, ngrids, non0tab):
    '''return numpy.dot(ao, dm) complex'''
    natm = ctypes.c_int(mol._atm.shape[0])
    nbas = ctypes.c_int(mol.nbas)
    vm = np.empty((ngrids,dm.shape[1])).astype(complex)
    ao = np.asarray(ao, order='C').astype(complex)
    dm = np.asarray(dm, order='C').astype(complex)
    libdft.Z_ao_dm(vm.ctypes.data_as(ctypes.c_void_p),
                        ao.ctypes.data_as(ctypes.c_void_p),
                        dm.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(nao), ctypes.c_int(dm.shape[1]),
                        ctypes.c_int(ngrids), ctypes.c_int(BLKSIZE),
                        non0tab.ctypes.data_as(ctypes.c_void_p),
                        mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                        mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                        mol._env.ctypes.data_as(ctypes.c_void_p))
    return vm
