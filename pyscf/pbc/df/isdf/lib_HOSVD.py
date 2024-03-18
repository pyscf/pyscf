import copy
from functools import reduce
import numpy as np
from pyscf import lib
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member
from pyscf.gto.mole import *

from pyscf.pbc.df.isdf.isdf_k import build_supercell
from pyscf.pbc.df.isdf.isdf_eval_gto import ISDF_eval_gto
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time

from itertools import permutations

import numpy
import numpy as np

from copy import deepcopy

class HOSVD_4D:
    
    def __init__(self, 
                 # A:np.ndarray, 
                 U:list[np.ndarray],
                 S:list[np.ndarray],
                 B:np.ndarray,
                 maxnorm_elmt=None):
        
        self.U = deepcopy(U)
        self.S = deepcopy(S)
        self.B = B.copy()
        self.shape = (U[0].shape[0], U[1].shape[0], U[2].shape[0], U[3].shape[0])
        self.Bshape = B.shape
        self.maxnorm_elmt = maxnorm_elmt
        
        assert len(U) == 4
        assert len(S) == 4
        assert B.ndim == 4
    
    def getFullMat(self): 
        
        # depends on whether U is complex
        
        raise NotImplementedError

    def getFullMat_after_fft(self):
        
        # U must be complex 
        
        raise NotImplementedError   

    def rfft(self):
        U = deepcopy(self.U)
        S = deepcopy(self.S)
        B = self.B.copy()

        assert U[1].dtype == np.float64
        assert U[2].dtype == np.float64
        assert U[3].dtype == np.float64
        
        U[1] = np.fft.fft(U[1],  axis=0)
        U[2] = np.fft.fft(U[2],  axis=0)
        U[3] = np.fft.rfft(U[3], axis=0)
        
        return HOSVD_4D(U=U, S=S, B=B)
        
    def fft(self):
        U = deepcopy(self.U)
        S = deepcopy(self.S)
        B = self.B.copy()
        
        U[1] = np.fft.fft(U[1], axis=0, norm='backward')
        U[2] = np.fft.fft(U[2], axis=0, norm='backward')
        U[3] = np.fft.fft(U[3], axis=0, norm='backward')
        
        return HOSVD_4D(U=U, S=S, B=B)

    def ifft(self):
        U = deepcopy(self.U)
        S = deepcopy(self.S)
        B = self.B.copy()
        
        U[1] = np.fft.ifft(U[1], axis=0, norm='backward')
        U[2] = np.fft.ifft(U[2], axis=0, norm='backward')
        U[3] = np.fft.ifft(U[3], axis=0, norm='backward')
        
        return HOSVD_4D(U=U, S=S, B=B)

    def irfft(self, n=None):
        U = deepcopy(self.U)
        S = deepcopy(self.S)
        B = self.B.copy()
        
        U[1] = np.fft.ifft(U[1],axis=0)
        U[2] = np.fft.ifft(U[2],axis=0)
        U[3] = np.fft.irfft(U[3], n=n[3], axis=0)
        
        return HOSVD_4D(U=U, S=S, B=B)

def HOSVD(A:np.ndarray, diff_scale=1e5, diff_scale_cutoff=1e-10, cutoff=1e-12, rela_cutoff=1e-8, buf=None):
    
    num_threads = lib.num_threads()
    
    A_maxsize_dim = np.max(A.shape)
    
    size_buf = max(A_maxsize_dim * A_maxsize_dim, A.size) * (num_threads + 2)
    
    if A.dtype == np.complex128:
        size_buf *= 2
    
    if buf is None:
        buf = np.zeros(size_buf, dtype=A.dtype)
    else:
        if buf.size < size_buf:
            if A.dtype == np.complex128:
                buf = np.zeros(size_buf//2, dtype=A.dtype)
            else:
                buf = np.zeros(size_buf, dtype=A.dtype)
    
    Res = {
        'U':[],
        'S':[],
        'B':[]
    }
    
    ndim = A.ndim
    
    for i in range(ndim):
        
        ## move the i-th axis to the left 
        
        tmp = np.moveaxis(A, i, 0)
        tmp = tmp.reshape(tmp.shape[0], -1)
        
        # tmp = numpy.dot(tmp, tmp.conj().T)
        
        if tmp.shape[0] == 1:
            
            tmp = tmp.reshape(-1)
            tmp = np.dot(tmp, tmp.conj().T)
            Res['U'].append(np.array([[1.0]], dtype=tmp.dtype)) # 1x1
            Res['S'].append(np.sqrt(np.array([tmp], dtype=tmp.dtype))) # 1
            assert Res['S'][-1].shape == (1,) 

            continue
        
        out = np.array((tmp.shape[0], tmp.shape[0]), dtype=tmp.dtype, buffer=buf)
        offset = 0
        offset += tmp.shape[0] * tmp.shape[0] * tmp.itemsize
        ddot_buf = np.array((tmp.shape[0], tmp.shape[0], num_threads), dtype=tmp.dtype, buffer=buf, offset=offset)
        
        if tmp.dtype == np.complex128:
            lib.zdot(tmp, tmp.conj().T, c=out)
        else:
            lib.ddot_withbuffer(tmp, tmp.T, c=out, buf=ddot_buf)
        
        ddot_buf = None
        tmp = None
        tmp = out
        
        with lib.threadpool_controller.limit(limits=lib.num_threads(), user_api='blas'):
            e, h = scipy.linalg.eigh(tmp)
        
        ## cutoff ## 
        
        cutoff_now = min(cutoff, rela_cutoff * e[-1])
        
        dim_tmp = len(e)
        
        loc_i = None
        
        for i in range(dim_tmp-1, -1, -1):
            if e[i] < cutoff_now:
                loc_i = i + 1
                break
            else:
                if i!=0:
                    if e[i]/e[i-1] > diff_scale_cutoff and e[i-1] < diff_scale_cutoff:
                        loc_i = i
                        break
        
        if loc_i is None:
            loc_i = 0
        
        Res['U'].append(h[:, loc_i:])
        Res['S'].append(np.sqrt(e[loc_i:]))
    
    offset_B_tmp1 = 0
    offset_B_tmp2 = A.size * A.itemsize   
    offset_ddot_buf = offset_B_tmp2 + A.size * A.itemsize
    
    assert ndim == 4
    
    # B = np.einsum('ijkl,ia->ajkl', B, Res['U'][0])
    # B = np.einsum('ajkl,jb->abkl', B, Res['U'][1])
    # B = np.einsum('abkl,kc->abcl', B, Res['U'][2])
    # B = np.einsum('abcl,ld->abcd', B, Res['U'][3])

    # contract 3 first to save memory
    
    B_tmp1 = np.ndarray(A.shape, dtype=A.dtype, buffer=buf, offset=offset_B_tmp1)
    B_tmp1.ravel()[:] = A.ravel()[:]
    
    if B.shape[3] > 1:
        B_tmp1 = B_tmp1.reshape(-1, B_tmp1.shape[3])
        res_shape = (B_tmp1.shape[0], Res['U'][3].shape[1])
        B_tmp2 = np.ndarray(res_shape, dtype=A.dtype, buffer=buf, offset=offset_B_tmp2)
        if B_tmp2.dtype == np.float64:
            ddot_buf = np.ndarray((B_tmp2.shape[0], B_tmp2.shape[1], num_threads), dtype=A.dtype, buffer=buf, offset=offset_ddot_buf)
            lib.ddot_withbuffer(B_tmp1, Res['U'][3], c=B_tmp2, buf=ddot_buf)
            ddot_buf = None
        else:
            lib.zdot(B_tmp1, Res['U'][3], c=B_tmp2)
        B_tmp1 = B_tmp2
        B_tmp2 = None
        offset_B_tmp1, offset_B_tmp2 = offset_B_tmp2, offset_B_tmp1

    if B.shape[2] > 1:
        B_tmp1 = B_tmp1.transpose(0, 1, 3, 2)
        B_tmp1 = B_tmp1.reshape(-1, B_tmp1.shape[3])
        res_shape = (B_tmp1.shape[0], Res['U'][2].shape[1])
        B_tmp2 = np.ndarray(res_shape, dtype=A.dtype, buffer=buf, offset=offset_B_tmp2)
        if B_tmp2.dtype == np.float64:
            ddot_buf = np.ndarray((B_tmp2.shape[0], B_tmp2.shape[1], num_threads), dtype=A.dtype, buffer=buf, offset=offset_ddot_buf)
            lib.ddot_withbuffer(B_tmp1, Res['U'][2], c=B_tmp2, buf=ddot_buf)
            ddot_buf = None
        else:
            lib.zdot(B_tmp1, Res['U'][2], c=B_tmp2)
        B_tmp1 = B_tmp2
        B_tmp2 = None
        B_tmp1 = B_tmp1.reshape(0, 1, 3, 2)

    if B.shape[1] > 1:
        B_tmp1 = B_tmp1.transpose(0, 2, 3, 1)
        B_tmp1 = B_tmp1.reshape(-1, B_tmp1.shape[3])
        res_shape = (B_tmp1.shape[0], Res['U'][1].shape[1])
        B_tmp2 = np.ndarray(res_shape, dtype=A.dtype, buffer=buf, offset=offset_B_tmp2)
        if B_tmp2.dtype == np.float64:
            ddot_buf = np.ndarray((B_tmp2.shape[0], B_tmp2.shape[1], num_threads), dtype=A.dtype, buffer=buf, offset=offset_ddot_buf)
            lib.ddot_withbuffer(B_tmp1, Res['U'][1], c=B_tmp2, buf=ddot_buf)
            ddot_buf = None
        else:
            lib.zdot(B_tmp1, Res['U'][1], c=B_tmp2)
        B_tmp1 = B_tmp2
        B_tmp2 = None
        B_tmp1 = B_tmp1.reshape(0, 3, 1, 2)

    if B.shape[0] > 1: # likely
        B_tmp1 = B_tmp1.transpose(1, 2, 3, 0)
        B_tmp1 = B_tmp1.reshape(-1, B_tmp1.shape[3])
        res_shape = (B_tmp1.shape[0], Res['U'][0].shape[1])
        B_tmp2 = np.ndarray(res_shape, dtype=A.dtype, buffer=buf, offset=offset_B_tmp2)
        if B_tmp2.dtype == np.float64:
            ddot_buf = np.ndarray((B_tmp2.shape[0], B_tmp2.shape[1], num_threads), dtype=A.dtype, buffer=buf, offset=offset_ddot_buf)
            lib.ddot_withbuffer(B_tmp1, Res['U'][0], c=B_tmp2, buf=ddot_buf)
            ddot_buf = None
        else:
            lib.zdot(B_tmp1, Res['U'][0], c=B_tmp2)
        B_tmp1 = B_tmp2
        B_tmp2 = None
        B_tmp1 = B_tmp1.reshape(3, 0, 1, 2)

    B = B_tmp1

    if B.dtype == np.complex128:
        B_imag = B.imag

        if np.linalg.norm(B_imag) > 1e-10:
            print("warning: B_imag = ", np.linalg.norm(B_imag))

        Res['B'] = B.real
    else:
        Res['B'] = B
    
    return HOSVD_4D(U=Res['U'], S=Res['S'], B=Res['B'])