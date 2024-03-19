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

## just todo write a demo ##

TRACE_GLOPS = True

def _get_flops(description):
    res = 0
    split = description.split('\n')
    for line in split:
        if "Optimized FLOP count" in line:
            res = float(line.split(":")[-1])
            return res
    return res

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
        
    def getFullMat_Naive(self):
        
        path, description = np.einsum_path('ijkl,pi,qj,rk,sl->pqrs', self.B, self.U[0],
                                           self.U[1], self.U[2], self.U[3],
                                           optimize='optimal')
        
        if TRACE_GLOPS:
            glops = _get_flops(description)
            return np.einsum('ijkl,pi,qj,rk,sl->pqrs', self.B, self.U[0], self.U[1], self.U[2], self.U[3], optimize=path), glops
        else:
            return np.einsum('ijkl,pi,qj,rk,sl->pqrs', self.B, self.U[0], self.U[1], self.U[2], self.U[3], optimize=path)
    
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

def _contract012(A:HOSVD_4D, B:HOSVD_4D):
    
    U0A = A.U[0]
    U1A = A.U[1]
    U2A = A.U[2]
    U0B = B.U[0]
    U1B = B.U[1]
    U2B = B.U[2]
    
    # optimize the contraction order ijkl, ai, bj, ck, pqrs, ap, bq, cr -> ls
    
    U0 = np.dot(U0A.T, U0B)
    U1 = np.dot(U1A.T, U1B)
    U2 = np.dot(U2A.T, U2B)
    
    path, description = np.einsum_path('ijkl,ip,jq,kr,pqrs->ls', A.B, U0, U1, U2, B.B, optimize='greedy')

    res = np.einsum('ijkl,ip,jq,kr,pqrs->ls', A.B, U0, U1, U2, B.B, optimize=path)
    U0  = None
    U1  = None
    U2  = None
    
    glops = _get_flops(description)
    glops += U0A.size * U0B.shape[1] + U1A.size * U1B.shape[1] + U2A.size * U2B.shape[1]
    
    U3A = A.U[3]
    U3B = B.U[3]
    
    res = np.dot(U3A, res)
    glops += res.size * U3A.shape[1]
    res = np.dot(res, U3B.T)
    glops += res.size * U3B.shape[1]
    
    return res, glops

def _contract012_orth_grid(A:HOSVD_4D):
    
    U0A = A.U[0]
    U0B = A.U[0]
    
    U0 = np.dot(U0A.T, U0B)
    
    path, description = np.einsum_path('ijkl,ip,pjks->ls', A.B, U0, A.B, optimize='greedy')

    glops = _get_flops(description)
    glops += U0.size * U0B.shape[0]
    
    res = np.einsum('ijkl,ip,pjks->ls', A.B, U0, A.B, optimize=path)
    
    U3A = A.U[3]
    U3B = A.U[3]
    
    res = np.dot(U3A, res)
    glops += res.size * U3A.shape[1]
    res = np.dot(res, U3B.T)
    glops += res.size * U3B.shape[1]
    
    return res, glops

def _contract013(A:HOSVD_4D, B:HOSVD_4D):
    
    U0A = A.U[0]
    U1A = A.U[1]
    U3A = A.U[3]
    U0B = B.U[0]
    U1B = B.U[1]
    U3B = B.U[3]
    
    # optimize the contraction order ijkl, ai, bj, cl, pqrs, ap, bq, cs -> kr

    U0 = np.dot(U0A.T, U0B)
    U1 = np.dot(U1A.T, U1B)
    U3 = np.dot(U3A.T, U3B)
    
    path, description = np.einsum_path('ijkl,ip,jq,ls,pqrs->kr', A.B, U0, U1, U3, B.B, optimize='greedy')

    glops = _get_flops(description)
    glops += U0.size * U0B.shape[1] + U1.size * U1B.shape[1] + U3.size * U3B.shape[1]
    
    res = np.einsum('ijkl,ip,jq,ls,pqrs->kr', A.B, U0, U1, U3, B.B, optimize=path)
    
    U2A = A.U[2]
    U2B = B.U[2]
    
    res = np.dot(U2A, res)
    glops += res.size * U2A.shape[1]
    res = np.dot(res, U2B.T)
    glops += res.size * U2B.shape[1]
    
    return res, glops

def _contract013_orth_grid(A:HOSVD_4D):
    
    U0A = A.U[0]
    U0B = A.U[0]
    
    U0 = np.dot(U0A.T, U0B)
    
    path, description = np.einsum_path('ijkl,ip,pjrl->kr', A.B, U0, A.B, optimize='greedy')

    glops = _get_flops(description)
    glops += U0.size * U0B.shape[0]
    
    res = np.einsum('ijkl,ip,pjrl->kr', A.B, U0, A.B, optimize=path)
    
    U2A = A.U[2]
    U2B = A.U[2]
    
    res = np.dot(U2A, res)
    glops += res.size * U2A.shape[1]
    res = np.dot(res, U2B.T)
    glops += res.size * U2B.shape[1]
    
    return res, glops

def _contract123(A:HOSVD_4D, B:HOSVD_4D):

    U1A = A.U[1]
    U2A = A.U[2]
    U3A = A.U[3]
    U1B = B.U[1]
    U2B = B.U[2]
    U3B = B.U[3]
    
    # optimize the contraction order ijkl, aj, bk, cl, pqrs, aq, br, cs -> ip 
    
    U1 = np.dot(U1A.T, U1B)
    U2 = np.dot(U2A.T, U2B)
    U3 = np.dot(U3A.T, U3B)
    
    path, description = np.einsum_path('ijkl,jq,kr,ls,pqrs->ip', A.B, U1, U2, U3, B.B, optimize='greedy')

    glops = _get_flops(description)
    glops += U1.size * U1B.shape[0] + U2.size * U2B.shape[0] + U3.size * U3B.shape[0]
    
    # print(description)

    res = np.einsum('ijkl,jq,kr,ls,pqrs->ip', A.B, U1, U2, U3, B.B, optimize=path)
    U1  = None
    U2  = None
    U3  = None
    
    U0A = A.U[0]
    U0B = B.U[0]
    
    res = np.dot(U0A, res)
    glops += res.size * U0A.shape[1]
    res = np.dot(res, U0B.T)
    glops += res.size * U0B.shape[1]
    
    return res, glops

def _contract123_orth_grid(A:HOSVD_4D):
    
    A_B = A.B.reshape(A.B.shape[0], -1)
    
    res = np.dot(A_B, A_B.T)
    
    glops = res.size * A_B.shape[1]
    
    U1  = None
    U2  = None
    U3  = None
    
    U0A = A.U[0]
    U0B = A.U[0]
    
    res = np.dot(U0A, res)
    glops += res.size * U0A.shape[1]
    res = np.dot(res, U0B.T)
    glops += res.size * U0B.shape[1]
    
    return res, glops

def _contract023(A:HOSVD_4D, B:HOSVD_4D):
    
    U0A = A.U[0]
    U2A = A.U[2]
    U3A = A.U[3]
    U0B = B.U[0]
    U2B = B.U[2]
    U3B = B.U[3]
    
    # optimize the contraction order ijkl, ai, bk, cl, pqrs, ap, br, cs -> jq
    
    U0 = np.dot(U0A.T, U0B)
    U2 = np.dot(U2A.T, U2B)
    U3 = np.dot(U3A.T, U3B)
    
    path, description = np.einsum_path('ijkl,ip,kr,ls,pqrs->jq', A.B, U0, U2, U3, B.B, optimize='greedy')
    
    glops = _get_flops(description)
    glops += U0.size * U0B.shape[0] + U2.size * U2B.shape[0] + U3.size * U3B.shape[0]

    res = np.einsum('ijkl,ip,kr,ls,pqrs->jq', A.B, U0, U2, U3, B.B, optimize=path)
    
    U1A = A.U[1]
    U1B = B.U[1]
    
    res = np.dot(U1A, res)
    glops += res.size * U1A.shape[1]
    res = np.dot(res, U1B.T)
    glops += res.size * U1B.shape[1]
    
    return res, glops

def _contract023_orth_grid(A:HOSVD_4D):
    
    U0A = A.U[0]
    U0B = A.U[0]
    
    U0 = np.dot(U0A.T, U0B)
    
    path, description = np.einsum_path('ijkl,ip,pqkl->jq', A.B, U0, A.B, optimize='greedy')

    glops = _get_flops(description)
    glops += U0.size * U0B.shape[0]
    
    res = np.einsum('ijkl,ip,pqkl->jq', A.B, U0, A.B, optimize=path)
    
    U1A = A.U[1]
    U1B = A.U[1]
    
    res = np.dot(U1A, res)
    glops += res.size * U1A.shape[1]
    res = np.dot(res, U1B.T)
    glops += res.size * U1B.shape[1]
    
    return res, glops

def add(a:HOSVD_4D, b:HOSVD_4D, cutoff=1e-10, rela_cutoff=1e-8):
    '''
    assume that indx 1 2 3 is orthogonal
    '''

    cost = 0 # trace tensor contraction cost

    
    

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
        
        out = np.ndarray((tmp.shape[0], tmp.shape[0]), dtype=tmp.dtype, buffer=buf)
        offset = 0
        offset += tmp.shape[0] * tmp.shape[0] * tmp.itemsize
        ddot_buf = np.ndarray((tmp.shape[0], tmp.shape[0], num_threads), dtype=tmp.dtype, buffer=buf, offset=offset)
        
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
    
    if B_tmp1.shape[3] > 1:
        shape_now = B_tmp1.shape
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
        B_tmp1 = B_tmp1.reshape(shape_now[0], shape_now[1], shape_now[2], -1)

    if B_tmp1.shape[2] > 1:
        B_tmp1 = B_tmp1.transpose(0, 1, 3, 2)
        shape_now = B_tmp1.shape
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
        B_tmp1 = B_tmp1.reshape(shape_now[0], shape_now[1], shape_now[2], -1)
        B_tmp1 = B_tmp1.transpose(0, 1, 3, 2)

    if B_tmp1.shape[1] > 1:
        B_tmp1 = B_tmp1.transpose(0, 2, 3, 1)
        shape_now = B_tmp1.shape
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
        B_tmp1 = B_tmp1.reshape(shape_now[0], shape_now[1], shape_now[2], -1)
        B_tmp1 = B_tmp1.transpose(0, 3, 1, 2)

    if B_tmp1.shape[0] > 1: # likely
        B_tmp1 = B_tmp1.transpose(1, 2, 3, 0)
        shape_now = B_tmp1.shape
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
        B_tmp1 = B_tmp1.reshape(shape_now[0], shape_now[1], shape_now[2], -1)
        B_tmp1 = B_tmp1.transpose(3, 0, 1, 2)

    B = B_tmp1

    if B.dtype == np.complex128:
        B_imag = B.imag

        if np.linalg.norm(B_imag) > 1e-10:
            print("warning: B_imag = ", np.linalg.norm(B_imag))

        Res['B'] = B.real
    else:
        Res['B'] = B
    
    return HOSVD_4D(U=Res['U'], S=Res['S'], B=Res['B'])




if __name__ == "__main__":
    
    boxlen = 3.5668
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    KE_CUTOFF = 256
    atm = [
        ['C', (0.     , 0.     , 0.    )],
        ['C', (0.8917 , 0.8917 , 0.8917)],
        ['C', (1.7834 , 1.7834 , 0.    )],
        ['C', (2.6751 , 2.6751 , 0.8917)],
        ['C', (1.7834 , 0.     , 1.7834)],
        ['C', (2.6751 , 0.8917 , 2.6751)],
        ['C', (0.     , 1.7834 , 1.7834)],
        ['C', (0.8917 , 2.6751 , 2.6751)],
    ]
    
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF)
    prim_mesh = prim_cell.mesh
    print("prim_mesh = ", prim_mesh)
    
    Ls = [3, 3, 3]
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    
    cell = build_supercell(atm, prim_a, Ls=Ls, ke_cutoff=KE_CUTOFF, mesh=mesh)
    
    from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridFFTDF2, _eval_rhoG

    df_tmp = MultiGridFFTDF2(cell)
    grids  = df_tmp.grids
    coords = np.asarray(grids.coords).reshape(-1,3)
    
    for i in range(cell.natm):
        print('%s %s  charge %f  xyz %s' % (cell.atom_symbol(i),
                                        cell.atom_pure_symbol(i),
                                        cell.atom_charge(i),
                                        cell.atom_coord(i)))

    print("Atoms' charges in a vector\n%s" % cell.atom_charges())
    print("Atoms' coordinates in an array\n%s" % cell.atom_coords())
    
    ## extract the range of bas belong to the first atm ##
    
    shl_loc_begin = None
    shl_loc_end = None
    
    for i in range(cell.nbas):
        if cell.bas_atom(i) == 0:
            if shl_loc_begin is None:
                shl_loc_begin = i
            shl_loc_end = i+1
    
    print("shl_loc_begin = ", shl_loc_begin)
    print("shl_loc_end = ", shl_loc_end)
    
    aoR = ISDF_eval_gto(cell=cell, coords=coords, shls_slice=(shl_loc_begin, shl_loc_end))
    
    print("aoR.shape = ", aoR.shape)
    print("mesh = ", mesh)
    
    aoR = aoR.reshape(-1, *mesh)
    
    ## do the same thing for the second atm ##
    
    shl_loc_begin = None
    shl_loc_end = None
    
    for i in range(cell.nbas):
        if cell.bas_atom(i) == 1:
            if shl_loc_begin is None:
                shl_loc_begin = i
            shl_loc_end = i+1
    
    aoR1 = ISDF_eval_gto(cell=cell, coords=coords, shls_slice=(shl_loc_begin, shl_loc_end))
    
    aoR1 = aoR1.reshape(-1, *mesh)
    
    HOSVD1 = HOSVD(aoR, cutoff=1e-10, rela_cutoff=1e-8)
    HOSVD2 = HOSVD(aoR1, cutoff=1e-10, rela_cutoff=1e-8)
    
    print("HOSVD1.Bshape = ", HOSVD1.Bshape)
    print("HOSVD2.Bshape = ", HOSVD2.Bshape)
    
    aoR1_full, _ = HOSVD1.getFullMat_Naive()
    aoR2_full, _ = HOSVD2.getFullMat_Naive()
    
    diff1 = np.linalg.norm(aoR1_full - aoR) / np.sqrt(np.prod(aoR.shape))
    diff2 = np.linalg.norm(aoR2_full - aoR1) / np.sqrt(np.prod(aoR1.shape))

    print("diff1 = ", diff1)
    print("diff2 = ", diff2)

    ########## test contract 123 ##########

    print(" ---------------- test contract 123 -------------------- ")
    
    t1 = time.time()
    res, glops = _contract123(HOSVD1, HOSVD2)
    t2 = time.time()
    print("contract time = ", t2 - t1)
    
    print("glops = ", glops)
    
    aoR = aoR.reshape(aoR.shape[0], -1)
    aoR1 = aoR1.reshape(aoR1.shape[0], -1)
    
    t1 = time.time()
    benchmark = np.dot(aoR, aoR1.T)
    t2 = time.time()
    print("benchmark time = ", t2 - t1)
    
    glops_naive = aoR.size * aoR1.shape[0]
    
    diff = np.linalg.norm(res - benchmark) / np.sqrt(np.prod(res.shape))
    print("diff = ", diff)
    print("reduce = %15.8f" % (glops_naive / glops))
    
    ########## test contract 123 orth grid ##########
    
    print(" ---------------- test contract 123 orth grid ---------------- ")
    
    t1 = time.time()
    res, glops = _contract123_orth_grid(HOSVD1)
    t2 = time.time()
    print("contract time = ", t2 - t1)
    
    print("glops = ", glops)
    
    t1 = time.time()
    benchmark = np.dot(aoR, aoR.T)
    t2 = time.time()
    
    print("benchmark time = ", t2 - t1)
    
    glops_naive = aoR.size * aoR.shape[0]
    
    print("reduce = %15.8f" % (glops_naive / glops))
    
    diff = np.linalg.norm(res - benchmark) / np.sqrt(np.prod(res.shape))
    
    print("diff = ", diff)
    
    ####### test contract 012 ########
    
    print(" ---------------- test contract 012 ---------------- ")
    
    t1 = time.time()
    res, glops = _contract012(HOSVD1, HOSVD2)
    t2 = time.time()
    print("contract time = ", t2 - t1)
    print("glops = ", glops)
    
    aoR = aoR.reshape(-1, *mesh)
    aoR1 = aoR1.reshape(-1, *mesh)
    
    t1 = time.time()
    benchmark = np.einsum('ijkl,ijks->ls', aoR, aoR1)
    t2 = time.time()
    
    print("benchmark time = ", t2 - t1)
    
    glops_naive = aoR.size * aoR1.shape[1]
    
    print("reduce = %15.8f" % (glops_naive / glops))
    
    diff = np.linalg.norm(res - benchmark) / np.sqrt(np.prod(res.shape))
    print("diff = ", diff)
    
    ######## test contract 012 orth grid ########
    
    print(" ---------------- test contract 012 orth grid ---------------- ")
    
    t1 = time.time()
    res, glops = _contract012_orth_grid(HOSVD1)
    t2 = time.time()
    
    print("contract time = ", t2 - t1)
    print("glops = ", glops)
    
    t1 = time.time()
    benchmark = np.einsum('ijkl,ijks->ls', aoR, aoR)
    t2 = time.time()
    
    print("benchmark time = ", t2 - t1)
    
    glops_naive = aoR.size * aoR.shape[1]
    
    print("reduce = %15.8f" % (glops_naive / glops))
    
    diff = np.linalg.norm(res - benchmark) / np.sqrt(np.prod(res.shape))
    print("diff = ", diff)
    
    