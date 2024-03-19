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

C = 2
M = 5

######## use python's code as the reference ########

## to do : implement the C code part 

## to do : write a flat_HOSVD to handle the matrix 

## a class to describe the HOSVD for a 4-order tensor ##

import numpy
import numpy as np

from copy import deepcopy

from itertools import permutations

class HOSVD_4D:
    
    def __init__(self, 
                 # A:np.ndarray, 
                 U:list[np.ndarray],
                 S:list[np.ndarray],
                 B:np.ndarray):
        
        self.U = deepcopy(U)
        self.S = deepcopy(S)
        self.B = B.copy()
        self.shape = (U[0].shape[0], U[1].shape[0], U[2].shape[0], U[3].shape[0])
        self.Bshape = B.shape
        
        assert len(U) == 4
        assert len(S) == 4
        assert B.ndim == 4
    
    def size(self):
        return self.B.size + sum([np.prod(data.shape) for data in self.U])
    
    def _analysis_GetFullMat_cost(self):
        
        ### GENERATE ALL POSSIBLE PATHS ###
        
        perms = list(permutations([0,1,2,3]))
        all_path = []
        for perm in perms:
            all_path.append(list(perm)) 
        
        ## analysis all possible path ##
        
        min_cost = None
        min_path = None
        min_path_storage = 0
        
        for path in all_path:
            shape_now = deepcopy(self.Bshape)
            cost = 0    
            storage = 0
            for idx in path:
                shape_now = list(shape_now)
                shape_now[idx] = self.U[idx].shape[0]
                shape_now = tuple(shape_now)
                cost += np.prod(shape_now) * self.Bshape[idx]
                storage = max(storage, np.prod(shape_now))
            
            if min_cost is None:
                min_cost = cost
                min_path = path
                min_path_storage = storage
            else:
                if cost < min_cost:
                    min_cost = cost
                    min_path = path
                    min_path_storage = storage
        
        return min_cost, min_path, min_path_storage
        
    
    def getFullMat(self): ## for testing, extremply slow !
        U = self.U
        S = self.S
        B = self.B

        A = B.copy()

        _, path, _ = self._analysis_GetFullMat_cost()
        
        for idx in path:
            if idx == 0:
                A = np.einsum('ijkl,ia->ajkl', A, U[0].conj().T)
            elif idx == 1:
                A = np.einsum('ajkl,jb->abkl', A, U[1].conj().T)
            elif idx == 2:
                A = np.einsum('abkl,kc->abcl', A, U[2].conj().T)
            elif idx == 3:
                A = np.einsum('abcl,ld->abcd', A, U[3].conj().T)
            else:
                raise RuntimeError("idx error")

        return A

    def getFullMat_after_fft(self):
        U = self.U
        S = self.S
        B = self.B

        A = B.copy()

        assert B.dtype == np.float64

        A = np.einsum('abcd,dl->abcl', A, U[3].T)
        A = np.einsum('abcl,ck->abkl', A, U[2].T)
        A = np.einsum('abkl,bj->ajkl', A, U[1].T)
        A = np.einsum('ajkl,ai->ijkl', A, U[0].T)

        return A

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
        
def _analysis_contract_cost(A:HOSVD_4D, B:HOSVD_4D, path:list[str]):
    assert len(path) <= 4
    
    contr_indx = []
    for data in path:
        contr_indx.append(int(data[1:]))
    
    cost_init_mm = 0
    for indx in contr_indx:
        assert A.shape[indx] == B.shape[indx]
        cost_init_mm += A.shape[indx] * A.Bshape[indx] * B.Bshape[indx]
    
    ##### middle mm #####
    
    shape_A_now = A.Bshape
    shape_B_now = B.Bshape
        
    cost_middle_mm = 0
        
    for data in path:
        idx = int(data[1:])
        if data[0] == "L":
            shape_A_now = list(shape_A_now)
            shape_A_now[idx] = B.Bshape[idx]
            shape_A_now = tuple(shape_A_now)
            cost_middle_mm += np.prod(shape_A_now) * A.Bshape[idx]
        else:
            shape_B_now = list(shape_B_now)
            shape_B_now[idx] = A.Bshape[idx]
            shape_B_now = tuple(shape_B_now)
            cost_middle_mm += np.prod(shape_B_now) * B.Bshape[idx]
    
    ##### final mm #####
    
    cost_final_mm = 1
    
    for i in range(4):
        if i not in contr_indx:
            cost_final_mm *= A.Bshape[i] * B.Bshape[i]
        else:
            assert shape_A_now[i] == shape_B_now[i]
            cost_final_mm *= shape_A_now[i]
    
    cost = cost_init_mm + cost_middle_mm + cost_final_mm
    # print("cost = ", cost)
    return cost

def _analysis_uncont_indx_trans(A:HOSVD_4D, B:HOSVD_4D, unctr_indx):
    intermediate_shape = []
    for i in unctr_indx:
        intermediate_shape.append(A.Bshape[i])
    unctr_indx_reverse = deepcopy(unctr_indx)
    unctr_indx_reverse.reverse()
    path = []
    for i in unctr_indx:
        intermediate_shape.append(A.Bshape[i])
        path.append("L%d" % i)
    for i in unctr_indx_reverse:
        intermediate_shape.append(B.Bshape[i])
        path.append("R%d" % i)
    
    intermediate_shape = tuple(intermediate_shape)
    
    perms = list(permutations(path))
    all_path = []
    for perm in perms:
        all_path.append(list(perm)) 
    
    ##### analysis all possible path #####
    
    min_cost = None
    min_path = None
    
    for path in all_path:
        shape_now = deepcopy(intermediate_shape)
        cost = 0
        for data in path:
            idx = int(data[1:])
            if data[0] == "L":
                shape_now = list(shape_now)
                shape_now[idx] = A.U[idx].shape[0]
                shape_now = tuple(shape_now)
                cost += np.prod(shape_now) * A.U[idx].shape[1]
            else:
                shape_now = list(shape_now)
                shape_now[idx] = B.U[idx].shape[0]
                shape_now = tuple(shape_now)
                cost += np.prod(shape_now) * B.U[idx].shape[1]
        
        if min_cost is None:
            min_cost = cost
            min_path = path
        else:
            if cost < min_cost:
                min_cost = cost
                min_path = path
    
    return min_path
    
        
def _analysis_contract_grid_cost(A:HOSVD_4D, B:HOSVD_4D, verbose=False):
    '''
    the same as contract 1 2 3
    '''

    cost_naive = A.shape[0] * (A.shape[1] * A.shape[2] * A.shape[3]) * B.shape[0]
    
    ### first loop over all the possible contraction path to determine the one with the smallest cost ### 

    ### the cost for transpotation is omitted ### 

    path = [
    ]
    
    path.append(["L1", "L2", "L3", _analysis_contract_cost(A, B, ["L1", "L2", "L3"])])
    path.append(["L2", "L3", "L1", _analysis_contract_cost(A, B, ["L2", "L3", "L1"])])
    path.append(["L3", "L1", "L2", _analysis_contract_cost(A, B, ["L3", "L1", "L2"])])
    path.append(["L1", "L3", "L2", _analysis_contract_cost(A, B, ["L1", "L3", "L2"])])
    path.append(["L2", "L1", "L3", _analysis_contract_cost(A, B, ["L2", "L1", "L3"])])
    path.append(["L3", "L2", "L1", _analysis_contract_cost(A, B, ["L3", "L2", "L1"])])
    path.append(["L1", "L2", "R3", _analysis_contract_cost(A, B, ["L1", "L2", "R3"])])
    path.append(["L2", "L1", "R3", _analysis_contract_cost(A, B, ["L2", "L1", "R3"])])
    path.append(["L1", "L3", "R2", _analysis_contract_cost(A, B, ["L1", "L3", "R2"])])
    path.append(["L3", "L1", "R2", _analysis_contract_cost(A, B, ["L3", "L1", "R2"])])
    path.append(["R2", "R3", "L1", _analysis_contract_cost(A, B, ["R2", "R3", "L1"])])
    path.append(["R3", "R2", "L1", _analysis_contract_cost(A, B, ["R3", "R2", "L1"])])
    path.append(["L2", "L3", "R1", _analysis_contract_cost(A, B, ["L2", "L3", "R1"])])    
    path.append(["L3", "L2", "R1", _analysis_contract_cost(A, B, ["L3", "L2", "R1"])])
    path.append(["R1", "R3", "L2", _analysis_contract_cost(A, B, ["R1", "R3", "L2"])])
    path.append(["R3", "R1", "L2", _analysis_contract_cost(A, B, ["R3", "R1", "L2"])])
    path.append(["R1", "R2", "L3", _analysis_contract_cost(A, B, ["R1", "R2", "L3"])])
    path.append(["R2", "R1", "L3", _analysis_contract_cost(A, B, ["R2", "R1", "L3"])])
    path.append(["R1", "R2", "R3", _analysis_contract_cost(A, B, ["R1", "R2", "R3"])])
    path.append(["R2", "R1", "R3", _analysis_contract_cost(A, B, ["R2", "R1", "R3"])])
    path.append(["R3", "R1", "R2", _analysis_contract_cost(A, B, ["R3", "R1", "R2"])])
    path.append(["R1", "R3", "R2", _analysis_contract_cost(A, B, ["R1", "R3", "R2"])])
    path.append(["R2", "R3", "R1", _analysis_contract_cost(A, B, ["R2", "R3", "R1"])]) 
    path.append(["R3", "R2", "R1", _analysis_contract_cost(A, B, ["R3", "R2", "R1"])])

    min_cost = None
    max_cost = None
    min_path = None
    max_path = None
    
    for i, path_now in enumerate(path):
        if verbose:
            print(path_now)
            print("checking path ", i, " cost = ", path_now[-1])
        if i == 0:
            min_cost = path_now[-1]
            max_cost = path_now[-1]
            min_path = path_now
            max_path = path_now
        else:
            if path_now[-1] < min_cost:
                min_cost = path_now[-1]
                min_path = path_now
            if path_now[-1] > max_cost:
                max_cost = path_now[-1]
                max_path = path_now
    
    if verbose:
        print("min_cost = ", min_cost)
        print("max_cost = ", max_cost)
        print("min_path = ", min_path)
        print("max_path = ", max_path)
        print("cost_naive = ", cost_naive)
        print("reduction %15.8f ~ %15.8f" % (cost_naive/max_cost, cost_naive/min_cost)) 
    
    return min_path

def contract(A:HOSVD_4D, B:HOSVD_4D, indx, path = None, verbose=False):
    
    if isinstance(indx, str):
        indx = [int(indx)]
    elif isinstance(indx, list):
        indx.sort()
    elif isinstance(indx, int):
        indx = [indx]
    else:
        raise RuntimeError("indx type error")
    
    def generate_all_path(contr_indx):
        all_path = [["L%d" % contr_indx[0]], ["R%d" % contr_indx[0]]]
        # print("contr_indx = ", contr_indx)
        for i in contr_indx[1:]:
            all_path_new = []
            for data in all_path:
                data_L = deepcopy(data)
                data_L.append("L%d" % i)
                data_R = deepcopy(data)
                data_R.append("R%d" % i)
                all_path_new.append(data_L)
                all_path_new.append(data_R)
            all_path = all_path_new
        ### for all the path generate all permutation ###
        
        Res = []
        
        for path in all_path:
            L = []
            R = []
            for data in path:
                if data[0] == "L":
                    L.append(int(data[1:]))
                else:
                    R.append(int(data[1:]))

            L.sort()
            R.sort()
            
            if len(L) <=1:
                L = [L]
            else:
                ### all permutation of L ###
                perms = list(permutations(L))
                L = []
                for perm in perms:
                    L.append(list(perm))
            
            if len(R) <=1:
                R = [R]
            else:
                ### all permutation of R ###
                perms = list(permutations(R))
                R = []
                for perm in perms:
                    R.append(list(perm))
            
            for l in L:
                for r in R:
                    path_tmp = []
                    for id_l in l:
                        path_tmp.append("L%d" % id_l)
                    for id_r in r:
                        path_tmp.append("R%d" % id_r)
                    Res.append(path_tmp)
        
        return Res
    
    cost_naive = 1
    
    indx_not_contr = []
    for i in range(4):
        if i not in indx:
            indx_not_contr.append(i)
            cost_naive *= A.shape[i] * B.shape[i]
        else:
            cost_naive *= A.shape[i]
    
    indx_not_contr_reverse = deepcopy(indx_not_contr)
    indx_not_contr_reverse.reverse()
    
    # print("indx_not_contr = ", indx_not_contr)
    # print("indx_not_contr_reverse = ", indx_not_contr_reverse)
    
    if path is None:
        
        path_all = generate_all_path(indx)
        
        # print("path_all = ", path_all)
        
        min_path = None
        min_cost = None
        
        for path_now in path_all:
            cost = _analysis_contract_cost(A, B, path_now)
            if min_cost is None:
                min_cost = cost
                min_path = path_now
            else:
                if cost < min_cost:
                    min_cost = cost
                    min_path = path_now
        
        path = min_path
    
    # print("path = ", path)
    
    assert path is not None
    
    tmp = {
        0:None,
        1:None,
        2:None,
        3:None
    }
    
    cost = 0
    
    for idx in indx:
        # print("idx = ", idx)
        # print("A.U[idx] = ", A.U[idx][:,0])
        # print("B.U[idx] = ", B.U[idx][:,0])
        tmp[idx] = np.dot(A.U[idx].conj().T, B.U[idx])
        # print("tmp[idx].shape = ", tmp[idx].shape)
        # print(tmp[idx])
        cost += np.prod(tmp[idx].shape) * B.U[idx].shape[0]    
    
    # print(A.B)
    tmp_A = deepcopy(A.B)
    tmp_B = deepcopy(B.B)
    
    ### perform un contracted 
    
    for data in path:
        # print("data = ", data)
        if data == "L0":
            tmp_A = np.einsum('ijkl,ia->ajkl', tmp_A.conj(), tmp[0])
            cost += np.prod(tmp_A.shape) * tmp[0].shape[0]
        elif data == "R0":
            tmp_B = np.einsum('ijkl,ai->ajkl', tmp_B, tmp[0])   
            cost += np.prod(tmp_B.shape) * tmp[0].shape[1] 
        elif data == "L1":
            tmp_A = np.einsum('ajkl,jb->abkl', tmp_A.conj(), tmp[1])
            cost += np.prod(tmp_A.shape) * tmp[1].shape[0]
        elif data == "R1":
            tmp_B = np.einsum('ajkl,bj->abkl', tmp_B, tmp[1])
            cost += np.prod(tmp_B.shape) * tmp[1].shape[1]
        elif data == "L2":
            tmp_A = np.einsum('abkl,kc->abcl', tmp_A.conj(), tmp[2])
            cost += np.prod(tmp_A.shape) * tmp[2].shape[0]
        elif data == "R2":
            tmp_B = np.einsum('abkl,ck->abcl', tmp_B, tmp[2])
            cost += np.prod(tmp_B.shape) * tmp[2].shape[1]
        elif data == "L3":
            tmp_A = np.einsum('abcl,ld->abcd', tmp_A.conj(), tmp[3])
            cost += np.prod(tmp_A.shape) * tmp[3].shape[0]
        elif data == "R3":
            tmp_B = np.einsum('abcl,dl->abcd', tmp_B, tmp[3])
            cost += np.prod(tmp_B.shape) * tmp[3].shape[1]

    if len(indx) == 4:
        cost += np.prod(tmp_A.shape)
        if verbose:
            print("cost = ", cost)
            print("cost_naive = ", cost_naive)
            print("reduction %15.8f" % (cost_naive/cost))
        return np.dot(tmp_A.ravel().conj(), tmp_B.ravel())
    else:
        ### transpose 

        # final_shape = list(tmp_A.shape[indx_not_contr])
        final_shape = []
        for i in indx_not_contr:
            final_shape.append(tmp_A.shape[i])
        # final_shape.extend(list(tmp_B.shape[indx_not_contr_reverse]))
        for i in indx_not_contr_reverse:
            final_shape.append(tmp_B.shape[i])
        final_shape = tuple(final_shape)

        # print("transpose A", (*indx_not_contr, *indx))
        # print("trnaspose B", (*indx, *indx_not_contr_reverse))

        tmp_A = np.transpose(tmp_A, (*indx_not_contr, *indx))
        tmp_B = np.transpose(tmp_B, (*indx, *indx_not_contr_reverse))

        tmp_A = tmp_A.reshape(-1, np.prod(tmp_A.shape[-len(indx):]))
        tmp_B = tmp_B.reshape(np.prod(tmp_B.shape[:len(indx)]), -1)

        cost += np.prod(tmp_A.shape) * tmp_B.shape[1]

        # print("final_shape = ", final_shape)

        res = np.dot(tmp_A.conj(), tmp_B)
        
        # res = np.einsum("ia,ab,jb->ij", A.U[0], res, B.U[0])

        # print("res.shape = ", res.shape)
        
        res = res.reshape(*final_shape)
        
        # final transformation # 
        
        assert len(indx_not_contr) == 1
        
        min_path_trans = _analysis_uncont_indx_trans(A, B, indx_not_contr)
        
        # print("min_path_trans = ", min_path_trans)  
        
        path0 = min_path_trans[0]
        
        if path0[0] == "L":
            res = np.dot(A.U[indx_not_contr[0]], res)
            cost += np.prod(res.shape) * A.U[indx_not_contr[0]].shape[1]
            res = np.dot(res, B.U[indx_not_contr[0]].T)
            cost += np.prod(res.shape) * B.U[indx_not_contr[0]].shape[0]
        elif path0[0] == "R":
            res = np.dot(res, B.U[indx_not_contr[0]].T)
            cost += np.prod(res.shape) * B.U[indx_not_contr[0]].shape[0]
            res = np.dot(A.U[indx_not_contr[0]], res)
            cost += np.prod(res.shape) * A.U[indx_not_contr[0]].shape[1]
        else:
            raise RuntimeError("path0[0] error")

        if verbose:
            print("cost = ", cost)
            print("cost_naive = ", cost_naive)
            print("reduction %15.8f" % (cost_naive/cost))
        
        return res
        

def HOSVD(A:np.ndarray, diff_scale=1e5, diff_scale_cutoff=1e-10, cutoff=1e-12, rela_cutoff=1e-8):
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
        tmp = numpy.dot(tmp, tmp.conj().T)
        e, h = np.linalg.eigh(tmp)
        
        # cutoff_now = max(cutoff, e[-1]*rela_cutoff)
        
        if e[-1] *rela_cutoff < cutoff:
            cutoff_now = e[-1] *rela_cutoff
        
        # print("e = ", e)
        
        ## cutoff ## 
        
        dim_tmp = len(e)
        
        loc_i = None
        
        for i in range(dim_tmp-1, -1, -1):
            if e[i] < cutoff:
                loc_i = i + 1
                break
            else:
                if i!=0:
                    if e[i]/e[i-1] > diff_scale and e[i-1] < diff_scale_cutoff:
                        print("e[i] = ", e[i])
                        print("e[i-1] = ", e[i-1])
                        loc_i = i
                        break
        
        if loc_i is None:
            loc_i = 0
        
        Res['U'].append(h[:, loc_i:])
        Res['S'].append(np.sqrt(e[loc_i:]))
    
    B = A.copy()
    
    assert ndim == 4
    
    B = np.einsum('ijkl,ia->ajkl', B, Res['U'][0])
    B = np.einsum('ajkl,jb->abkl', B, Res['U'][1])
    B = np.einsum('abkl,kc->abcl', B, Res['U'][2])
    B = np.einsum('abcl,ld->abcd', B, Res['U'][3])

    B_imag = B.imag

    if np.linalg.norm(B_imag) > 1e-10:
        print("warning: B_imag = ", np.linalg.norm(B_imag))

    Res['B'] = B.real
    
    return HOSVD_4D(U=Res['U'], S=Res['S'], B=Res['B'])

def HOSVD_5D(A:np.ndarray, diff_scale=1e5, diff_scale_cutoff=1e-10, cutoff=1e-12, rela_cutoff=1e-8):
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
        tmp = numpy.dot(tmp, tmp.conj().T)
        e, h = np.linalg.eigh(tmp)
        
        # cutoff_now = max(cutoff, e[-1]*rela_cutoff)
        
        if e[-1] *rela_cutoff < cutoff:
            cutoff_now = e[-1] *rela_cutoff
        
        # print("e = ", e)
        
        ## cutoff ## 
        
        dim_tmp = len(e)
        
        loc_i = None
        
        for i in range(dim_tmp-1, -1, -1):
            if e[i] < cutoff:
                loc_i = i + 1
                break
            else:
                if i!=0:
                    if e[i]/e[i-1] > diff_scale and e[i-1] < diff_scale_cutoff:
                        print("e[i] = ", e[i])
                        print("e[i-1] = ", e[i-1])
                        loc_i = i
                        break
        
        if loc_i is None:
            loc_i = 0
        
        Res['U'].append(h[:, loc_i:])
        Res['S'].append(np.sqrt(e[loc_i:]))
    
    B = A.copy()
    
    assert ndim == 5
    
    
    
    B = np.einsum('ijklm,me->ijkle', B, Res['U'][4])
    B = np.einsum('ijkle,ld->ijkde', B, Res['U'][3])
    B = np.einsum('ijkde,kc->ijcde', B, Res['U'][2])
    B = np.einsum('ijcde,jb->ibcde', B, Res['U'][1])
    B = np.einsum('ibcde,ia->abcde', B, Res['U'][0])
    
    B_imag = B.imag

    if np.linalg.norm(B_imag) > 1e-10:
        print("warning: B_imag = ", np.linalg.norm(B_imag))

    Res['B'] = B.real
    
    return Res

def Check_HOSVD(A:np.ndarray, Res:HOSVD_4D):
    A1 = Res.getFullMat()
    
    diff = np.linalg.norm(A1 - A) / np.sqrt(np.prod(A.shape))
    
    print("diff = ", diff)
    
    assert diff < 1e-10

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
    
    #### do the same for atm 1 #### 
    
    shl_loc_begin = None
    shl_loc_end = None
    
    for i in range(cell.nbas):
        if cell.bas_atom(i) == 1:
            if shl_loc_begin is None:
                shl_loc_begin = i
            shl_loc_end = i+1
    
    print("shl_loc_begin = ", shl_loc_begin)
    print("shl_loc_end = ", shl_loc_end)
    
    aoR1 = ISDF_eval_gto(cell=cell, coords=coords, shls_slice=(shl_loc_begin, shl_loc_end))
    aoR1 = aoR1.reshape(-1, *mesh)
    aoR2 = aoR1[0:1,:,:,:]
    
    
    print("aoR.shape  = ", aoR.shape)
    print("aoR1.shape = ", aoR1.shape)
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    Res  = HOSVD(aoR)
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1, t2, "HOSVD")
    Res1 = HOSVD(aoR1)
    Res2 = HOSVD(aoR2)
    
    ### print sigma ###
    
    # for iblock, data in enumerate(Res.S):
    #     print("iblock = ", iblock)
    #     print("data = ", data)
    
    # print("Res['B'] = ", Res.B)
    
    # 计算 Res.B 里头绝对值大于 1e-14 的元素的个数
    
    large_elmt_count = np.sum(np.abs(Res.B) > 1e-14)
    print("size of B = ", Res.B.size)   
    print("B.shape = ", Res.B.shape)
    print("B.shape = ", Res1.B.shape)
    print("large_elmt_count = ", large_elmt_count)
    
    ### check the accurary of HOSVD ###
    
    Check_HOSVD(aoR, Res)
    
    ### implement fft and ifft ###
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    Res_fft = Res.fft()
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1, t2, "HOSVD fft")
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    Res_fft = Res.fft().getFullMat_after_fft()
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1, t2, "fft+getFullMat_after_fft")
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    benchmark = np.fft.fftn(aoR, axes=[1,2,3])
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1, t2, "numpy fft")
    print("Res_fft.shape = ", Res_fft.shape)
    print("benchmark.shape = ", benchmark.shape)
    diff = np.linalg.norm(Res_fft - benchmark) / np.sqrt(np.prod(Res_fft.shape))
    print("diff = ", diff)
    if diff > 1e-10:
        print("warning impl of fft is not correct: diff = ", diff)
    # print(Res_fft/benchmark)
    # print(Res_fft[0,0,1,:10])
    # print(benchmark[0,0,1,:10])
    
    for data, bench in zip(Res_fft.ravel(), benchmark.ravel()):
        if abs(data-bench) > 1e-8:
            print("warning: data = ", data, "bench = ", bench)
    
    ##### whether the ifft is correct #####
    
    Res_fft = Res.fft()
    Res_ifft = Res_fft.ifft()
    Res_full = Res_ifft.getFullMat_after_fft()
    Res_full_real = Res_full.real
    Res_full_imag = Res_full.imag
    norm_imag = np.linalg.norm(Res_full_imag) / np.sqrt(np.prod(Res_full_imag.shape))
    print("norm_imag = ", norm_imag)
    if norm_imag > 1e-10:
        print("warning: norm_imag = ", norm_imag)
    
    diff = Res_full_real - aoR 
    
    norm_diff = np.linalg.norm(diff) / np.sqrt(np.prod(diff.shape))
    print("norm_diff = ", norm_diff)
    if norm_diff > 1e-10:
        print("warning: norm_diff = ", norm_diff)
    
    ##### the ifft 
    
    Check_HOSVD(aoR, Res)
    
    aoR_fft = np.fft.fftn(aoR, axes=[1,2,3])
    aoR_back = np.fft.ifftn(aoR_fft, axes=[1,2,3])
    
    aoR_back_imag = aoR_back.imag
    norm_imag = np.linalg.norm(aoR_back_imag) / np.sqrt(np.prod(aoR_back_imag.shape))
    # print("norm_imag = ", norm_imag)
    if norm_imag > 1e-10:
        print("warning: norm_imag = ", norm_imag)
    
    aoR_back_real = aoR_back.real
    diff = aoR_back_real - aoR
    norm_diff = np.linalg.norm(diff) / np.sqrt(np.prod(diff.shape))
    print("norm_diff = ", norm_diff)
    
    # exit(1)
    
    Res_ifft = Res.ifft().getFullMat_after_fft()
    benchmark = np.fft.ifftn(aoR, axes=[1,2,3])
    diff = np.linalg.norm(Res_ifft - benchmark) / np.sqrt(np.prod(Res_ifft.shape))
    print("diff = ", diff)
    if diff > 1e-10:
        print("warning impl of ifft is not correct: diff = ", diff)
    
    for data, bench in zip(Res_ifft.ravel(), benchmark.ravel()):
        if abs(data-bench) > 1e-8:
            print("warning: data = ", data, "bench = ", bench, data/bench)
    
    ##### whether the rfft and irfft is correct ##### 
    
    Res_rfft = Res.rfft()
    Res_irfft = Res_rfft.irfft(n=aoR.shape)
    Res_full = Res_irfft.getFullMat_after_fft()
    Res_full_real = Res_full.real
    Res_full_imag = Res_full.imag
    norm_imag = np.linalg.norm(Res_full_imag) / np.sqrt(np.prod(Res_full_imag.shape))
    print("norm_imag = ", norm_imag)
    
    if norm_imag > 1e-10:
        print("warning: norm_imag = ", norm_imag)
    
    diff = Res_full_real - aoR
    
    norm_diff = np.linalg.norm(diff) / np.sqrt(np.prod(diff.shape))
    
    print("norm_diff = ", norm_diff)
    if norm_diff > 1e-10:
        print("warning: norm_diff = ", norm_diff)
    
    
    ##### implement tensor contraction #####
    
    _analysis_contract_grid_cost(Res, Res1, verbose=True)
    _analysis_contract_grid_cost(Res, Res2, verbose=True)    
    
    Res  = HOSVD(aoR)
    Res1 = HOSVD(aoR1)
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    ovlp = contract(Res, Res1, indx = [1,2,3], verbose=True)    
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1, t2, "contract")
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    ovlp_bench_mark = np.dot(aoR.reshape(-1, np.prod(mesh)), aoR1.reshape(-1, np.prod(mesh)).T)
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    _benchmark_time(t1, t2, "contract_bench_mark")
    
    aoR1_full = Res.getFullMat()
    aoR1_full = aoR1_full.reshape(-1, np.prod(mesh))
    aoR2_full = Res1.getFullMat()
    aoR2_full = aoR2_full.reshape(-1, np.prod(mesh))
    
    ovlp2 = np.dot(aoR1_full, aoR2_full.T)
    
    diff = np.linalg.norm(ovlp - ovlp_bench_mark) / np.sqrt(np.prod(ovlp.shape))
    print("diff = ", diff)
    if diff > 1e-10:
        print("warning: diff = ", diff)
    
    diff2 = np.linalg.norm(ovlp_bench_mark - ovlp2) / np.sqrt(np.prod(ovlp.shape))
    print("diff2 = ", diff2)
    
    # for data, bench in zip(ovlp.ravel(), ovlp_bench_mark.ravel()):
    #     if abs(data-bench) > 1e-8:
    #         print("warning: data = ", data, "bench = ", bench, data/bench)