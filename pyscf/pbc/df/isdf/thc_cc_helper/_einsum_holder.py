

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
# Author: Ning Zhang <ningzhang1024@gmail.com>
#

import copy
import numpy as np
from functools import partial
from pyscf.lib import logger

SUPPORTED_INPUT_NAME = [
    "T1", "XO", "XV", "TAUO", "TAUV", "THC_INT", "THC_T2",  ## NOTE: only these terms are involved in THC!
    "XO_T2", "XV_T2", "PROJ",                               ## used in THC-CCSD, "PROJ" stands for PROJ_THC
    "PROJ_ROBUST", "PROJ_MP2", "PROJ_MP3",
    "R1", "R2"
]

OCC_INDICES = ["i", "j", "k", "l", "m", "n", "o"]
VIR_INDICES = ["a", "b", "c", "d", "e", "f", "g"]

####### TORCH BACKEND #######

FOUND_TORCH = False
GPU_SUPPORTED = False

try:
    import torch
    FOUND_TORCH = True
except ImportError:
    pass

if FOUND_TORCH:
    if torch.cuda.is_available():
        GPU_SUPPORTED = True

##############################

def _is_same_type(ind_a, ind_b):
    
    is_a_occ = ind_a in OCC_INDICES
    is_b_occ = ind_b in OCC_INDICES
    is_a_vir = ind_a in VIR_INDICES
    is_b_vir = ind_b in VIR_INDICES
    
    if is_a_occ and is_b_occ:
        return True
    if is_a_vir and is_b_vir:
        return True
    if (not is_a_occ and not is_a_vir) and (not is_b_occ and not is_b_vir):
        return True
    
    return False
    
### einsum term holder, expressing contraction only ###

##### every term will be parsed until the args are all strings #####

class _einsum_term:
    
    POSSIBLE_NEW_DUMMY_INDICES = "hqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    def __init__(self, name=None, einsum_str:str=None, factor=1.0, args=None):
        
        assert isinstance(einsum_str, str)
        
        einsum_str = einsum_str.replace(" ", "")
        
        self.einsum_str = einsum_str
        self.args       = args  ## can be either str or a class with .name attribute
        self.factor     = factor
        
        self.name = name
        if name is None:
            self.name = "intermediate"
        
        #### additional info ####
        
        self._output_indices = None
        self._dummy_indices  = None
        self._conj           = False
        
        ### check consistency ###

        if "->" not in einsum_str:
            self._is_skeleton = True
            assert len(args) == 1
        else:
            self._is_skeleton = False
            
            left = einsum_str.split("->")[0]
            left = left.split(",")
            assert len(left) == len(args)
        
        self._contract_fn = None
        self._diagonal    = False
        
        self.build()
    
    def set_diagonal(self):
        assert self.is_skeleton
        assert len(self._output_indices) == 2
        self._diagonal = True
    
    def __str__(self):
        
        res  = "********** _einsum_term  ********** \n"
        res += "Name            : {}\n".format(self.name)
        res += "Factor          : {}\n".format(self.factor)
        if self.is_parsed:
            res += "Args            : {}\n".format(self.args)
        res += "Einsum string   : {}\n".format(self.einsum_str)
        res += "_output_indices : {}\n".format(self._output_indices)
        res += "_dummy_indices  : {}\n".format(self._dummy_indices)
        res += "_conj           : {}\n".format(self._conj)
        res += "is_skeleton     : {}\n".format(self.is_skeleton)
        res += "*********************************** \n"
        
        return res
    
    def build(self):
        
        if self.is_skeleton:
            self._output_indices = list(self.einsum_str)
            nocc_indices = len([i for i in self._output_indices if i in OCC_INDICES])
            nvir_indices = len([i for i in self._output_indices if i in VIR_INDICES])
            assert nocc_indices + nvir_indices <= 4 # only support 4 indices for skeleton
            self._dummy_indices = []
            #if self._contract_fn is not None:
            self._contract_fn = None
        else:
            self._output_indices = list(self.einsum_str.split("->")[1].strip())
            self._dummy_indices  = self.einsum_str.split("->")[0].strip().replace(",", "")
            self._dummy_indices  = list(set([i for i in self._dummy_indices if i not in self._output_indices]))
            #assert self._contract is None
            self._contract_fn = None

    @property
    def n_index(self):
        return len(self._output_indices)

    @property
    def is_skeleton(self):
        return self._is_skeleton

    @property
    def is_parsed(self):
        return all([isinstance(arg, str) for arg in self.args])

    @property
    def occvir_str(self):
        res = ""
        for i in self._output_indices:
            if i in OCC_INDICES:
                res += "o"
            else:
                res += "v"
        return res

    ######################## perform algebraic operations ########################

    def transpose(self, order:tuple):
        
        order = tuple(order)
        
        assert not self.is_skeleton
        assert len(order) == len(self._output_indices)
        assert len(set(order)) == len(order)
        assert all([i >= 0 and i < len(self._output_indices) for i in order])
        
        # reordering the output indices #
        
        new_output_indices = [self._output_indices[i] for i in order]
        
        LHS = self.einsum_str.split("->")[0]
        RHS = "".join(new_output_indices)
        einsum_str = LHS + "->" + RHS
        
        res = _einsum_term(self.name, einsum_str, self.factor, self.args)
        
        if self._conj:
            res.conj()
        
        return res
    
    def transpose_(self, order:tuple):
        
        order = tuple(order)
        
        assert not self.is_skeleton
        assert len(order) == len(self._output_indices)
        assert len(set(order)) == len(order)
        assert all([i >= 0 and i < len(self._output_indices) for i in order])
        
        # reordering the output indices #
        
        new_output_indices = [self._output_indices[i] for i in order]
        
        LHS = self.einsum_str.split("->")[0]
        RHS = "".join(new_output_indices)
        einsum_str = LHS + "->" + RHS
        
        self.einsum_str = einsum_str
        self.build()
        
    def conj(self):
        res = _einsum_term(self.name, self.einsum_str, self.factor, self.args)
        res._conj = not self._conj
        return res
    
    def conj_(self):
        self._conj = not self._conj
        return self
    
    def _map_indices(self, _indices_map):  ### change the indices of the output tensor
        
        if isinstance(_indices_map, list):
            assert len(_indices_map) == len(self._output_indices)
            indices_map = dict(zip(self._output_indices, _indices_map))
        else:
            indices_map = copy.deepcopy(_indices_map)
        
        keys = list(indices_map.keys())
        keys = [k for k in keys if k in self._output_indices]
        values = [indices_map[k] for k in keys]
        # the actual mapping #
        indices_map_real = dict(zip(keys, values)) 
        dummy_indices_affected = [k for k in self._dummy_indices if k in values]
        for d_k in dummy_indices_affected:
            if d_k in OCC_INDICES:
                for _id_ in OCC_INDICES:
                    if _id_ not in indices_map_real.values() and _id_ != d_k and _id_ not in self._dummy_indices:
                        indices_map_real[d_k] = _id_
                        break
            elif d_k in VIR_INDICES:
                for _id_ in VIR_INDICES:
                    if _id_ not in indices_map_real.values() and _id_ != d_k and _id_ not in self._dummy_indices:
                        indices_map_real[d_k] = _id_
                        break
            else:
                for _id_ in _einsum_term.POSSIBLE_NEW_DUMMY_INDICES:
                    if _id_ not in indices_map_real.values() and _id_ != d_k and _id_ not in self._dummy_indices:
                        indices_map_real[d_k] = _id_
                        break
        #print("dummy_indices = ", self._dummy_indices)
        #print("dummy_indices_affected = ", dummy_indices_affected)
        #print("indices_map_real: ", indices_map_real)
        # check the sanity of indices_map #
        for key in indices_map_real:
            if key in OCC_INDICES:
                assert indices_map_real[key] in OCC_INDICES
            if key in VIR_INDICES:
                assert indices_map_real[key] in VIR_INDICES
        # mapping the output indices #
        new_output_indices = [indices_map_real[i] if i in indices_map_real else i for i in self._output_indices]
        new_einsum_str = [indices_map_real[i] if i in indices_map_real else i for i in self.einsum_str]
        new_einsum_str = "".join(new_einsum_str)
        self.einsum_str = new_einsum_str
        self.build()

    def _relabel_dummy_indices(self, indices_forbidden):
        indices_forbidden = list(set(indices_forbidden))
        indices_forbidden.extend(list(set(self._output_indices)))
        new_dummy_indices = []
        for _id_ in _einsum_term.POSSIBLE_NEW_DUMMY_INDICES:
            if _id_ not in indices_forbidden:
                new_dummy_indices.append(_id_)
        dummy_indices_map = {}
        for _id_ in self._dummy_indices:
            if _id_ not in indices_forbidden:
                dummy_indices_map[_id_] = _id_
                # remove _id_ in new_dummy_indices #
                if _id_ in new_dummy_indices:
                    new_dummy_indices.remove(_id_)
        for _id_ in self._dummy_indices:
            if _id_ in indices_forbidden:
                dummy_indices_map[_id_] = new_dummy_indices.pop()
        
        einsum_str_new = ""
        for c in self.einsum_str:
            if c in dummy_indices_map:
                einsum_str_new += dummy_indices_map[c]
            else:
                einsum_str_new += c
        
        self.einsum_str = einsum_str_new
        self.build()

    #### rewrite operators #### 
    
    @classmethod
    def _conj_scalar(cls, other):
        if isinstance(other, (int, float)):
            return other
        elif isinstance(other, complex):
            return other.conjugate()
        else:
            raise ValueError("Conjugation with type {} not supported.".format(type(other)))
    
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            if self._conj:
                other = _einsum_term._conj_scalar(other)
            res = _einsum_term(self.name, self.einsum_str, self.factor * other, self.args)
            if self._conj:
                res = res.conj()
            return res
        else:
            # 如果 other 是其他类型，抛出 ValueError
            raise ValueError("Multiplication with type {} not supported.".format(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def __itruediv__(self, other):
        return self.__imul__(1.0 / other)
        
    def __imul__(self, other):
        if isinstance(other, (int, float, complex)):
            if self._conj:
                other = _einsum_term._conj_scalar(other)
            self.factor *= other
            return self

    def __neg__(self):
        return self.__mul__(-1.0)
    
    def __add__(self, other):
        
        expr1 = to_expr_holder(self)
        expr2 = to_expr_holder(other)
        
        return expr1 + expr2

    #################################################################################
    
    ######################## contraction ########################

    #### constrauct path ####
    
    def _build_path_numpy(self, scheduler, **kwargs):
                
        optimize = kwargs.pop("optimize", True)
        
        subscripts = self.einsum_str
        tensors    = [scheduler.get_tensor(arg) for arg in self.args]

        import numpy as np

        contract_path = np.einsum_path(subscripts, *tensors, optimize=optimize)
        
        self._contract_fn = partial(np.einsum,
                                    subscripts = subscripts,
                                    optimize   = contract_path[0])
        
    def _build_path_opt_einsum(self, scheduler, **kwargs):
        
        optimize = kwargs.pop("optimize", True)
        memory   = kwargs.pop("memory", 2**28)
        tensors = [scheduler.get_tensor(arg) for arg in self.args]

        import opt_einsum as oe
        
        #print("einsum_str: ", self.einsum_str)
        
        if isinstance(optimize, bool):
            #for arg, ts in zip(tensors, self.args):
            #    print("arg %10s of shape %s" % (ts, arg.shape))
            path = oe.contract_path(self.einsum_str, *tensors, memory_limit=memory)
        else:
            path = oe.contract_path(self.einsum_str, *tensors, memory_limit=memory, optimize=optimize)
        
        #print(path[1])
        self._contract_fn = partial(oe.contract,
                                    #subscripts   = self.einsum_str,
                                    optimize     = path[0],
                                    memory_limit = memory)
        #print("term %s path is finished" % self.name)
        #assert self._contract_fn is not None
        
        print("the contraction path of term %s is " % (self.name))
        print(path[1])
        
    @classmethod
    def _size_dict_cotengra(cls, inputs, *tensors):

        size_dict = {}
        for idx, tensor in zip(inputs, tensors):
        
            #assert isinstance(tensor, np.ndarray)
            #assert tensor.ndim == len(idx)

            for _idx_ in idx:
                if _idx_ not in size_dict:
                    size_dict[_idx_] = tensor.shape[idx.index(_idx_)]
                else:
                    assert size_dict[_idx_] == tensor.shape[idx.index(_idx_)]

        return size_dict

    def _build_path_cotengra(self, scheduler, **kwargs):
        
        import cotengra as ctg
        
        inputs, output = ctg.utils.eq_to_inputs_output(self.einsum_str)
        tensors        = [scheduler.get_tensor(arg) for arg in self.args]
        size_dict      = _einsum_term._size_dict_cotengra(inputs, *tensors)
    
        memory = kwargs.pop("memory", 2**28) 
        
        opt = ctg.HyperOptimizer(
            slicing_reconf_opts=dict(
                target_size=memory,
            ),
            progbar=False,
        )
        
        self._cotengra_tree = opt.search(inputs, output, size_dict)
        self._contract_fn   = self._cotengra_tree.contract
        
        print("the contraction path of term %s is " % (self.name))
        print(self._cotengra_tree.print_contractions())
    
    def _contract_path(self, scheduler, backend, **kwargs):
        
        #print("term %s path is building" % self.name)
        #print("backend = ", backend)
        #print("kwargs  = ", kwargs)
        
        if self.is_skeleton:
            self._contract_fn = None
            return None  ### no need to build path for skeleton
    
        if hasattr(self, "_contract_fn"):     ### no need to rebuild
            if self._contract_fn is not None:
                return self
        
        #assert self._is_skeleton == False
        assert all([isinstance(arg, str) for arg in self.args])
        assert isinstance(backend, str)
        backend = backend.lower()
        assert backend in ["cotengra", "opt_einsum", "numpy"]
        self._backend = backend
        
        if backend == "numpy":
            self._build_path_numpy(scheduler, **kwargs)
        elif backend == "opt_einsum":
            self._build_path_opt_einsum(scheduler, **kwargs)
        else:
            assert backend == "cotengra"
            self._build_path_cotengra(scheduler, **kwargs)
        
        return self

    def _contract(self, scheduler):
        
        if self._contract_fn is None:
            #raise ValueError("Contract path not built.")
            #print("evaluate skeleton term: ", self.name) 
            assert self.is_skeleton
            res = scheduler.get_tensor(self.args[0]) * self.factor
        
        else:
            #print("evaluate term: ", self.name, " with factor ", self.factor)
            #print("args = ", self.args)
            tensors    = [scheduler.get_tensor(arg) for arg in self.args]

            #for tensor, name in zip(tensors, self.args):
            #    print("arg %10s = " % name, tensor)
        
            if self._backend == "cotengra":
                res = self._contract_fn(tensors) * self.factor
            else:
                res = self._contract_fn(self.einsum_str, *tensors) * self.factor
        
        if self._conj:
            res = res.conj()
        
        if self._diagonal:
            res = np.diag(np.diag(res))
        
        return res
        
    #############################################################
        
##### different types of THC einsum terms #####

class _thc_eri_oooo(_einsum_term):
    def __init__(self):
        super().__init__("eri_oooo", "iP,jP,PQ,kQ,lQ->ijkl", args=["XO", "XO", "THC_INT", "XO", "XO"])

class _thc_eri_ovoo(_einsum_term):
    def __init__(self):
        super().__init__("eri_ovoo", "iP,aP,PQ,kQ,lQ->iakl", args=["XO", "XV", "THC_INT", "XO", "XO"])

class _thc_eri_ovov(_einsum_term):
    def __init__(self):
        super().__init__("eri_ovov", "iP,aP,PQ,jQ,bQ->iajb", args=["XO", "XV", "THC_INT", "XO", "XV"])

class _thc_eri_ooov(_einsum_term):
    def __init__(self):
        super().__init__("eri_ooov", "iP,jP,PQ,kQ,aQ->ijka", args=["XO", "XO", "THC_INT", "XO", "XV"])

class _thc_eri_ovvv(_einsum_term):
    def __init__(self):
        super().__init__("eri_ovvv", "iP,aP,PQ,bQ,cQ->iabc", args=["XO", "XV", "THC_INT", "XV", "XV"])

class _thc_eri_ovvo(_einsum_term):
    def __init__(self):
        super().__init__("eri_ovvo", "iP,aP,PQ,bQ,kQ->iabk", args=["XO", "XV", "THC_INT", "XV", "XO"])

class _thc_eri_oovv(_einsum_term):
    def __init__(self):
        super().__init__("eri_oovv", "iP,jP,PQ,aQ,bQ->ijab", args=["XO", "XO", "THC_INT", "XV", "XV"])

class _thc_eri_vvvv(_einsum_term):
    def __init__(self):
        super().__init__("eri_vvvv", "aP,bP,PQ,cQ,dQ->abcd", args=["XV", "XV", "THC_INT", "XV", "XV"])

class _energy_denominator(_einsum_term):
    def __init__(self):
        super().__init__("ene_deno", "iT,jT,aT,bT->ijab", args=["TAUO", "TAUO", "TAUV", "TAUV"])

class _expr_mp2_t2(_einsum_term):
    def __init__(self):
        super().__init__("MP2_T2", "iP,aP,PQ,jQ,bQ,iT,jT,aT,bT->iajb", args=["XO", "XV", "THC_INT", "XO", "XV", "TAUO", "TAUO", "TAUV", "TAUV"])

class _expr_t1(_einsum_term):
    def __init__(self):
        super().__init__("T1", "ia", args=["T1"])

class _expr_t2(_einsum_term):
    def __init__(self):
        super().__init__("T2", "iP,aP,PQ,jQ,bQ->iajb", args=["XO_T2", "XV_T2", "THC_T2", "XO_T2", "XV_T2"])

class _expr_ccsd_t2(_einsum_term):
    def __init__(self):
        super().__init__("MP2_T2", "iP,aP,PQ,jQ,bQ,iT,jT,aT,bT->iajb", args=["XO_T2", "XV_T2", "THC_T2", "XO_T2", "XV_T2", "TAUO", "TAUO", "TAUV", "TAUV"])
        
class _expr_t2_thc_robust(_einsum_term):
    def __init__(self):
        super().__init__("T2", "iP,aP,PA,AB,BQ,jQ,bQ->iajb", args=["XO_T2", "XV_T2", "PROJ_ROBUST", "THC_T2", "PROJ_ROBUST", "XO_T2", "XV_T2"])

class _expr_r1_ip(_einsum_term):
    def __init__(self, multiroots=False):
        if not multiroots:
            super().__init__("R1", "i", args=["R1"])
        else:
            super().__init__("R1", "pi", args=["R1"])

class _expr_r2_ip(_einsum_term):
    def __init__(self, multiroots=False):
        if not multiroots:
            super().__init__("R2", "ija", args=["R2"])
        else:
            super().__init__("R2", "pija", args=["R2"])

class _expr_r1_ea(_einsum_term):
    def __init__(self, multiroots=False):
        if not multiroots:
            super().__init__("R1", "a", args=["R1"])
        else:
            super().__init__("R1", "pa", args=["R1"])
    
class _expr_r2_ea(_einsum_term):
    def __init__(self, multiroots=False):
        if not multiroots:
            super().__init__("R2", "iab", args=["R2"])
        else:
            super().__init__("R2", "piab", args=["R2"])

class _expr_foo(_einsum_term):
    def __init__(self):
        super().__init__("foo", "ij", args=["foo"])

class _expr_fvv(_einsum_term):
    def __init__(self):
        super().__init__("fvv", "ab", args=["fvv"])

class _expr_fov(_einsum_term):
    def __init__(self):
        super().__init__("fov", "ia", args=["fov"])

##### expression holder, can express + relation #####

### used in multiprocessing ### 

class _expr_holder:
    
    def __init__(self, name:str, indices:list[str], terms = None, 
                 cached=False,
                 remove_dg=False,
                 scheduler=None):
        
        self.name    = name
        self.indices = indices   ## the output indices of the expression
        self.terms   = terms
        if self.terms is None:
            self.terms = []
        self.cached  = cached     ## if cached, then this term is a terminal term
        
        if remove_dg:
            assert cached
        self._remove_dg = remove_dg
        
        ## a ptr to the scheduler ##
        
        self._scheduler = scheduler # only used when parsing the expression, because we have to determine whether there is intermediates of 2nd

    @property
    def n_index(self):
        return len(self.indices)

    @property
    def nocc(self):
        return len([i for i in self.indices if i in OCC_INDICES])
    
    @property
    def nvir(self):
        return len([i for i in self.indices if i in VIR_INDICES])

    @property
    def occvir_str(self):
        res = ""
        for i in self.indices:
            if i in OCC_INDICES:
                res += "o"
            else:
                res += "v"
        return res

    @property
    def is_parsed(self):
        return all([term.is_parsed for term in self.terms])

    @property
    def is_cached(self):
        return self.cached

    def remove_dg_(self):
        assert self.cached and len(self.indices) == 2
        self._remove_dg = True

    def rename_subterms(self):
        for _id_, term in enumerate(self.terms):
            term.name = self.name + "_" + "%d" % _id_

    #### __str__ ####
    
    def __str__(self):
        
        res = "********************* _expr_holder ********************* \n"
        res += "Name    : {}\n".format(self.name)
        res += "Indices : {}\n".format(self.indices)
        res += "Cached  : {}\n".format(self.cached)
        res += "Terms   : \n"
        for loc, term in enumerate(self.terms):
            res += "Term {} : \n".format(loc)
            res += str(term)
        res += "******************************************************** \n"

        return res
    
    #### map_indices ####
    
    def _map_indices(self, indices_map):
        
        for term in self.terms:
            term._map_indices(indices_map)

        if isinstance(indices_map, list):
            assert len(indices_map) == len(self.indices)
            self.indices = copy.deepcopy(indices_map)
        else:
            self.indices = [indices_map[i] if i in indices_map else i for i in self.indices]

    #### set property ####

    def set_scheduler(self, scheduler):
        self._scheduler = scheduler
    
    def set_cached(self, cached):
        self.cached = cached

    def conj_(self):
        for term in self.terms:
            term.conj_()
            #if isinstance(term, _einsum_term):
            #    term.conj_() # inplace
            #else:
            #    term.conj()
    
    def conj(self):
        res = copy.deepcopy(self)
        res.conj_()
        return res
    
    def transpose_(self, order:tuple):
        for term in self.terms:
            term.transpose_(order)
            #if isinstance(term, _einsum_term):
            #    term.transpose_(order) # inplace
            #else:
            #    term.transpose(order)

    def transpose(self, order:tuple):
        res = copy.deepcopy(self)
        res.transpose_(order)
        return res

    #### rewrite operations ####

    def __mul__(self, other):
        
        if isinstance(other, (int, float, complex)):
            res = copy.deepcopy(self)
            for term in res.terms:
                term *= other
            return res
        else:
            raise ValueError("Multiplication with type {} not supported.".format(type(other)))
    
    def __imul__(self, other):
        
        if isinstance(other, (int, float, complex)):
            for term in self.terms:
                term *= other
            return self
        else:
            raise ValueError("Multiplication with type {} not supported.".format(type(other)))
    
    def __irmul__(self, other):
        
        return self.__imul__(other)
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        return self.__mul__(1.0 / other)
    
    def __itruediv__(self, other):
        return self.__imul__(1.0 / other)

    def __add__(self, _other):
        
        if isinstance(_other, _einsum_term):
            other = to_expr_holder(_other)
            other.set_cached(self.cached)
        else:
            other = copy.deepcopy(_other)
            assert self._remove_dg == other._remove_dg
        
        other._map_indices(self.indices)
        
        cached = self.cached and other.cached
        scheduler = self._scheduler
        if scheduler is None:
            scheduler = other._scheduler
        name  = copy.deepcopy(self.name)
        terms = copy.deepcopy(self.terms) 
        terms.extend(copy.deepcopy(other.terms))
        
        res = _expr_holder(name, 
                           self.indices, 
                           terms, 
                           cached, 
                           self._remove_dg,
                           scheduler,
                           )
        
        return res

    def __iadd__(self, other):
        
        if isinstance(other, _einsum_term):
            other = to_expr_holder(other)
            other.set_cached(self.cached)
        else:
            other = copy.deepcopy(other)
            assert self._remove_dg == other._remove_dg
        
        other._map_indices(self.indices)
        
        self.cached = self.cached and other.cached
        self.terms.extend(copy.deepcopy(other.terms))
        
        if self._scheduler is None:
            self._scheduler = other._scheduler
        
        return self

    def __sub__(self, other):
        return self + (-1.0) * other

    def __isub__(self, other):
        return self.__iadd__(-1.0 * other)
    
    def __neg__(self):
        return self.__mul__(-1.0)
    
    #### constrauct path ####
    
    def process_term(self, term, scheduler, backend, **kwargs):
        term._contract_path(scheduler, backend, **kwargs)
        return term
        
    def build_contraction_path(self, scheduler, backend, **kwargs):
        
        t1 = (logger.process_clock(), logger.perf_counter())
        
        if scheduler is None:
            scheduler = self._scheduler
        else:
            self.set_scheduler(scheduler)
        
        if backend != "cotengra":
            
            if scheduler.with_gpu is False:
                from multiprocessing import Pool
                process_term_with_fixed_args = partial(self.process_term, scheduler=scheduler, backend=backend, **kwargs)
                terms = None
                with Pool() as pool:
                    #pool.map(self.process_term, [(term, scheduler, backend, kwargs) for term in self.terms])
                    terms = pool.map(process_term_with_fixed_args, self.terms)
                    #print("terms = ", terms)
                self.terms = terms
            else:
                for term in self.terms:
                    term._contract_path(scheduler, backend, **kwargs)
        else:
            
            for term in self.terms:
                term._contract_path(scheduler, backend, **kwargs)            

        t2 = (logger.process_clock(), logger.perf_counter())
        
        print("build contraction %s with time %.2f %.2f " % (self.name, t2[0]-t1[0], t2[1]-t1[1]))

    #### contraction ####
    
    def contract(self, scheduler):
        
        if scheduler is None:
            scheduler = self._scheduler
        else:
            self.set_scheduler(scheduler)
        
        res = None
        
        for term in self.terms:
            
            t1 = (logger.process_clock(), logger.perf_counter())
            
            if res is None:
                res = term._contract(scheduler)
            else:
                res += term._contract(scheduler)
            #print("after term %s, res = " % term.name, res)
            
            t2 = (logger.process_clock(), logger.perf_counter())
            
            print("contraction %s with time %.2f %.2f " % (term.name, t2[0]-t1[0], t2[1]-t1[1]))
        
        if self._remove_dg:
            res -= np.diag(np.diag(res))
        
        return res
    
def to_expr_holder(input, cached=False):
    if isinstance(input, _expr_holder):
        return input
    elif isinstance(input, _einsum_term):
        return _expr_holder(name   =input.name, 
                            indices=copy.deepcopy(input._output_indices), 
                            terms  =[copy.deepcopy(input)], 
                            cached =cached)
    else:
        raise ValueError("Unsupported type for conversion to _expr_holder.")

####### THC einsum sybolic #######

def thc_einsum_sybolic(subscripts, *tensors, **kwargs):
    
    assert "->" in subscripts ## do not support implicit
    
    #print("subscripts = ", subscripts)
    
    subscripts = subscripts.replace(" ", "")
    
    cached = kwargs.pop("cached", False)
    
    # special case #
    
    tensors = list(tensors)
    
    if len(tensors) == 0:
        raise ValueError("No tensors provided.")
    #if len(tensors) == 1:
    #    return tensors[0]
    
    # check sanity and aligned indices #
    
    tensors_indx_aligned = []
    
    _subscripts = subscripts.split("->")
    lhs        = _subscripts[0]
    rhs        = _subscripts[1]
    
    tensors_scripts = lhs.split(",")
    
    assert len(tensors_scripts) == len(tensors)
    
    for indices, tensor in zip(tensors_scripts, tensors):
        tensor_now = copy.deepcopy(tensor)
        tensor_now = to_expr_holder(tensor_now)
        assert tensor_now.n_index == len(indices)
        #print("indices = ", indices)
        ind_lst    = list(indices)        # lst the indices
        tensor_now._map_indices(ind_lst)  # change the indices
        tensors_indx_aligned.append(tensor_now)
    
    # build new expression holder #
    
    name = kwargs.pop("name", "intermediate")
    
    einsum_term = _einsum_term(
        name=name, 
        einsum_str=subscripts, 
        factor=1.0, 
        args=tensors_indx_aligned)
    
    return to_expr_holder(einsum_term, cached)    
    
####### schedule to evaluate expressions #######

def _parse_einsum_term(einsum_term:_einsum_term, scheduler):

    if einsum_term.is_parsed:
        return [einsum_term]
    
    name = einsum_term.name
    
    #print("Parsing term: ", name)
    
    NAME_FORMAT = name + "_sub_%d"
    
    #### parse the args iteratively until all are strings ####
    
    finish_parsing = False
    terms_now    = [einsum_term]
    terms_parsed = []

    #FACTOR = einsum_term.factor

    n_term_parsed = 0

    while not finish_parsing:
        
        finish_parsing = True
        terms_parsed   = []
        
        for term in terms_now:
            
            #print("---- Parsing term: ----\n", term)
            
            assert isinstance(term, _einsum_term)
            
            FACTOR_TERM = term.factor
            factor_absorbed = False
            
            if term.is_parsed:
                terms_parsed.append(term)
            else:
                finish_parsing = False
                
                subscripts = term.einsum_str.split("->")[0].split(",")
                RHS = term.einsum_str.split("->")[1]
                assert len(subscripts) == len(term.args)
                
                term_parse_found = False
                
                tensor_scripts = ""
                args           = []
                
                tensor_scripts_2 = []
                args_2           = []
                factor_2         = []
                
                indices_forbidden = ""
                for subscript in subscripts:
                    indices_forbidden += subscript
                indices_forbidden+= RHS
                indices_forbidden = set(list(indices_forbidden))
                
                assert len(term.args) == len(subscripts)
                
                for tr_script, arg in zip(subscripts, term.args): 
                    
                    if term_parse_found:
                        for _id_ in range(len(tensor_scripts_2)):
                            tensor_scripts_2[_id_] += tr_script + ","
                            arg_to_add = copy.deepcopy(arg)
                            if not isinstance(arg, str):
                                arg_to_add._map_indices(list(tr_script))
                            else:
                                if not factor_absorbed:
                                    factor_absorbed = True
                                    arg_to_add *= FACTOR_TERM
                            args_2[_id_].append(arg_to_add)
                    else:
                        if isinstance(arg, str):
                            tensor_scripts += tr_script + ","
                            args.append(arg)
                        else:
                            term_parse_found = True
                            assert isinstance(arg, _einsum_term) or isinstance(arg, _expr_holder)
                            if isinstance(arg, _einsum_term):
                                arg_to_add = copy.deepcopy(arg)
                                arg_to_add._map_indices(list(tr_script))
                                arg_to_add._relabel_dummy_indices(indices_forbidden)
                                arg_LHS = arg_to_add.einsum_str.split("->")[0].split(",")
                                for sub_tr_script, sub_arg in zip(arg_LHS, arg_to_add.args):
                                    tensor_scripts += sub_arg + ","
                                    args.append(sub_arg)
                                tensor_scripts_2.append(tensor_scripts)
                                args_2.append(args)
                                if factor_absorbed:
                                    #print("add factor = ", arg_to_add.factor)
                                    factor_2.append(arg_to_add.factor)
                                else:   
                                    #print("add factor = ", arg_to_add.factor * FACTOR_TERM)
                                    factor_2.append(arg_to_add.factor * FACTOR_TERM)
                                    factor_absorbed = True
                            else:
                                if arg.is_cached:
                                    arg_to_add = copy.deepcopy(arg)
                                    arg_to_add._map_indices(list(tr_script))
                                    scheduler.register_intermediates(arg.name, arg_to_add)
                                    tensor_scripts += tr_script + ","
                                    args.append(arg.name)
                                    tensor_scripts_2.append(tensor_scripts)
                                    args_2.append(args)
                                    factor_2.append(1.0)
                                else:
                                    #### looping over the terms ####
                                    
                                    # arg_to_add = copy.deepcopy(arg)
                                    # arg_to_add._map_indices(list(tr_script))
                                    # for sub_term in arg_to_add.terms:
                                    #     sub_term_to_add = copy.deepcopy(sub_term)
                                    #     #sub_term_to_add._map_indices(list(tr_script))
                                    #     sub_term_to_add._relabel_dummy_indices(indices_forbidden)
                                    #     arg_LHS = sub_term_to_add.einsum_str.split("->")[0].split(",")
                                    #     args_to_add = copy.deepcopy(args)
                                    #     tensor_scripts_to_add = tensor_scripts
                                    #     for sub_tr_script, sub_arg in zip(arg_LHS, sub_term_to_add.args):
                                    #         tensor_scripts_to_add += sub_tr_script + ","
                                    #         args_to_add.append(sub_arg)
                                    #     tensor_scripts_2.append(tensor_scripts_to_add)
                                    #     args_2.append(args_to_add)
                                    #     factor_2.append(sub_term_to_add.factor) 
                                    
                                    ### first parse the term ### 
                                    
                                    arg_to_add = copy.deepcopy(arg)
                                    arg_to_add._map_indices(list(tr_script))
                                    
                                    #print("arg_to_add: \n", arg_to_add)
                                    
                                    parsed_arg_to_add, _ = _parse_expression(arg_to_add, scheduler)
                                    
                                    if not factor_absorbed:
                                        parsed_arg_to_add *= FACTOR_TERM
                                        factor_absorbed = True
                                    
                                    #print("parsed_arg_to_add: \n", parsed_arg_to_add)
                                    
                                    for sub_term in parsed_arg_to_add.terms:
                                        sub_term_to_add = copy.deepcopy(sub_term)
                                        sub_term_to_add._relabel_dummy_indices(indices_forbidden)
                                        arg_LHS = sub_term_to_add.einsum_str.split("->")[0].split(",")
                                        args_to_add = copy.deepcopy(args)
                                        tensor_scripts_to_add = tensor_scripts
                                        for sub_tr_script, sub_arg in zip(arg_LHS, sub_term_to_add.args):
                                            tensor_scripts_to_add += sub_tr_script + ","
                                            args_to_add.append(sub_arg)
                                        tensor_scripts_2.append(tensor_scripts_to_add)
                                        args_2.append(args_to_add)
                                        #print("add factor = ", sub_term_to_add.factor)
                                        factor_2.append(sub_term_to_add.factor)

                assert term_parse_found
        
                #print("tensor_scripts_2: ", tensor_scripts_2)
                #print("factor_2        : ", factor_2)
        
                for _id_ in range(len(tensor_scripts_2)):
                    tensor_scripts_2[_id_] = tensor_scripts_2[_id_][:-1]
                    einsum_str = tensor_scripts_2[_id_] + "->" + RHS
                    factor = factor_2[_id_]
                    if not factor_absorbed:
                        factor *= FACTOR_TERM
                    if factor == 0.0:
                        continue
                    new_expr = _einsum_term(name       = NAME_FORMAT % n_term_parsed, 
                                            einsum_str = einsum_str, 
                                            factor     = factor, 
                                            args       = args_2[_id_])
                    #print("new_expr: \n", new_expr)
                    n_term_parsed += 1
                    terms_parsed.append(new_expr)
        
        terms_now = copy.deepcopy(terms_parsed)

    #for _id_ in range(len(terms_parsed)):
        #terms_parsed[_id_] *= FACTOR
        #print("factor = ", terms_parsed[_id_].factor)
        #print("term = \n", terms_parsed[_id_])

    return terms_parsed

def _parse_expression(expression_holder:_expr_holder, scheduler):
    
    #print("Parsing expression: ", expression_holder.name)
    
    expr_name = expression_holder.name

    terms_parsed = []

    for term in expression_holder.terms:
        #print("********************")
        #print("parse \n", term)
        #print("********************")
        #res = _parse_einsum_term(term, scheduler)
        #print(res)
        terms_parsed.extend(_parse_einsum_term(term, scheduler))
        #print("Parsed term: ", terms_parsed)
    
    #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    #for term in terms_parsed:
    #    print("Parsed term: \n", term)
    #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    
    dependence = []
    for term in terms_parsed:
        dependence.extend(term.args)
    dependence = list(set(dependence))
    
    return _expr_holder(name=expr_name, 
                        indices=expression_holder.indices, 
                        terms=terms_parsed, 
                        cached=expression_holder.cached, 
                        scheduler=scheduler), dependence

### expressions ###

##### TODO: write the parse function for THC-CCSD #####
    
from pyscf.pbc.df.isdf.thc_backend import *

_to_torch_tensor = to_torch
    
class THC_scheduler:
    
    ### static member name ### 
    
    t1_new_name       = "T1_NEW"
    t2_new_name       = "T2_NEW"
    ccsd_energy_name  = "CCSD_ENERGY"
    ip_hr1_r_name       = "IP_HR1_R"
    ip_hr2_r_name       = "IP_HR2_R"
    ea_hr1_r_name       = "EA_HR1_R"
    ea_hr2_r_name       = "EA_HR2_R"
    ip_hr1_l_name       = "IP_HR1_L"
    ip_hr2_l_name       = "IP_HR2_L"
    ea_hr1_l_name       = "EA_HR1_L"
    ea_hr2_l_name       = "EA_HR2_L"
    
    ### projector ###
    
    projector_allowed = ["thc", "thc_robust", "mp2", "thc_laplace"]
    thc_proj_name = "thc"
    thc_robust_proj_name = "thc_robust"
    mp2_proj_name = "mp2"
    thc_laplace_proj_name = "thc_laplace"
    
    def __init__(self, 
                 X_O:np.ndarray,
                 X_V:np.ndarray,
                 THC_INT:np.ndarray,
                 T1:np.ndarray,
                 XO_T2:np.ndarray,
                 XV_T2:np.ndarray,
                 THC_T2:np.ndarray,
                 TAU_O:np.ndarray,
                 TAU_V:np.ndarray,
                 # PROJECTOR:np.ndarray,
                 proj_type="thc",
                 grid_partition = None,
                 use_torch=False,
                 with_gpu =False,
                 verbose=4,
                 **kwargs):
        
        ###### holder the tensors ###### 
        
        self._xo = X_O
        self._xv = X_V
        self._thc_int = THC_INT
        self._tau_o = TAU_O
        self._tau_v = TAU_V
        self._xo_t2 = XO_T2
        self._xv_t2 = XV_T2
        # self._proj  = PROJECTOR
        
        self.proj_type = proj_type
        self.grid_partition = grid_partition
        
        ##### init projector ##### 
        
        assert proj_type in THC_scheduler.projector_allowed
        
        self.t1     = T1
        self.thc_t2 = THC_T2
        
        self.verbose = verbose
        
        self.use_torch = use_torch
        if with_gpu:
            assert use_torch
            if GPU_SUPPORTED is False:
                #import sys
                #self.stdout = sys.stdout 
                #logger.warn(self, "GPU is not supported, use CPU instead.")
                print("GPU is not supported, use CPU instead.")
                with_gpu = False
        self.with_gpu = with_gpu
        
        if use_torch:
            assert FOUND_TORCH
            self._xo = _to_torch_tensor(self._xo, self.with_gpu)
            self._xv = _to_torch_tensor(self._xv, self.with_gpu)
            self._thc_int = _to_torch_tensor(self._thc_int, self.with_gpu)
            self._tau_o = _to_torch_tensor(self._tau_o, self.with_gpu)
            self._tau_v = _to_torch_tensor(self._tau_v, self.with_gpu)
            self._xo_t2 = _to_torch_tensor(self._xo_t2, self.with_gpu)
            self._xv_t2 = _to_torch_tensor(self._xv_t2, self.with_gpu)
            
        if self.proj_type in [THC_scheduler.thc_proj_name, THC_scheduler.thc_robust_proj_name]:
            self.init_projector()
        elif self.proj_type == THC_scheduler.thc_laplace_proj_name:
            qr = kwargs.pop("qr_rela_cutoff", 1e-2)
            print("qr_rela_cutoff = ", qr)
            #self.init_laplace_projector()
            from pyscf.pbc.df.isdf.thc_cc_helper._thc_proj import _thc_proj_2
            XoT2, XvT2 = _thc_proj_2(
                THC_scheduler._tensor_to_cpu(self._xo),
                THC_scheduler._tensor_to_cpu(self._xv),
                THC_scheduler._tensor_to_cpu(self._tau_o),
                THC_scheduler._tensor_to_cpu(self._tau_v),
                partition=THC_scheduler._tensor_to_cpu(self.grid_partition),
                qr_cutoff=qr
            )
            self.init_projector(XoT2, XvT2)
        else:
            raise NotImplementedError("Projector type %s not implemented." % self.proj_type)
        
        ###### holder the expressions and intermediates ######
        
        self._input_tensor_name = ["XO", "XV", "THC_INT", "TAUO", "TAUV", "XO_T2", "XV_T2", # integrals and laplace 
                                   "PROJ", "PROJ_ROBUST", "PROJ_MP2",                       # projector 
                                   "T1", "THC_T2",                                          # t1, t2, CC 
                                   "R1", "R2"                                               # R1, R2, EOM
                                   ]
        self.tensor_name_to_attr_name = {
            "XO"    : "_xo",
            "XV"    : "_xv",
            "THC_INT" : "_thc_int",
            "TAUO"  : "_tau_o",
            "TAUV"  : "_tau_v",
            "XO_T2" : "_xo_t2",
            "XV_T2" : "_xv_t2",
            "PROJ"  : "_proj",
            "PROJ_ROBUST": "_proj_robust",
            "PROJ_MP2"   : "_proj_mp2",
            "T1"    : "t1",
            "THC_T2": "thc_t2",
            "R1"    : "r1",
            "R2"    : "r2"
        }
        self._registered_intermediates_name = []
        self._expr_intermediates            = []
        self._cached_intermediates          = [] # all np.ndarray
        
        self.expr = {}
        self._expr_cached = {}

        self._intermediates_hierarchy = [] # the first level determined by only the input tensors
                                           # the second level determined by the first level and the intermediates

    ########## __str__ for THC_scheduler ##########
    
    def __str__(self):
            
        res = " ********** THC_scheduler ********** \n"
        res += "Input tensors : \n\n"
        for name in self._input_tensor_name:
            res += "{} : \n".format(name)
        res += "\n"
        res += "Intermediates : \n\n"
        assert len(self._registered_intermediates_name) == len(self._expr_intermediates)
        for name, expr in zip(self._registered_intermediates_name, self._expr_intermediates):
            res += "Intermediate {} : \n".format(name)
            res += str(expr)
        res += "\n"
        res += "Expressions : \n"
        for name, expr in self.expr.items():
            res += "Expression {} : \n".format(name)
            res += str(expr)
        if hasattr(self, "_dependency"):
            res += "\n"
            res += "Dependency : \n"
            for name, dep in self._dependency.items():
                res += "{} : {}\n".format(name, dep)
        if len(self._intermediates_hierarchy) > 0:
            res += "\n"
            res += "Intermediates hierarchy : \n"
            for _id_, layer in enumerate(self._intermediates_hierarchy):
                res += "Layer %d : \n" % _id_
                res += "Intermediates : \n"
                for name in layer:
                    res += "{} ".format(name)
                res += "\n"
        res += " ********************************** \n"
        
        return res
    
    #############################################################################################################################################
    
    # initialization of different projector # 
    
    @classmethod
    def _evaluate_UDUT_mutliscale(cls,U,D):
        
        # D is addusmed to ascending
        
        res = None
        
        SCALE = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.01]

        for i in range(len(SCALE)-1):
            where = np.where((D>=D[0]*SCALE[i])&(D<D[0]*SCALE[i+1]))[0]
            if len(where) == 0:
                continue            
            U_Tmp = U[:,where]
            D_Tmp = D[where]
            TMP = U_Tmp @ np.diag(D_Tmp) @ U_Tmp.T
            if res is None:
                res = TMP
            else:
                res += TMP
        
        return res
    
    @classmethod
    def _tensor_to_cpu(cls, arg):
        try:
            import torch    
            if isinstance(arg, torch.Tensor):
                return arg.cpu().numpy()
            else:
                return arg
        except Exception as e:
            return arg
    
    def init_projector(self, X_o = None, X_v = None):
        
        time0 = logger.process_clock(), logger.perf_counter()
        
        if X_o is None:
            X_o = THC_scheduler._tensor_to_cpu(self._xo_t2)
        else:
            X_o = THC_scheduler._tensor_to_cpu(X_o)
            self._xo_t2 = X_o
            if self.use_torch:
                self._xo_t2 = _to_torch_tensor(self._xo_t2, self.with_gpu)
        
        if X_v is None:
            X_v = THC_scheduler._tensor_to_cpu(self._xv_t2)
        else:
            X_v = THC_scheduler._tensor_to_cpu(X_v)
            self._xv_t2 = X_v
            if self.use_torch:
                self._xv_t2 = _to_torch_tensor(self._xv_t2, self.with_gpu)
        
        PROJ_INV = np.einsum('iP,aP,iQ,aQ->PQ', X_o, X_v, X_o, X_v, optimize=True)
        print("PROJ_INV shape = ", PROJ_INV.shape)
        import scipy
        D_RR, U_RR = scipy.linalg.eigh(PROJ_INV)
        kept = D_RR > D_RR[-1]*1e-14
        
        print("condition number = ", D_RR[-1]/D_RR[0])    
        print("nkept = ", np.sum(kept))
        
        D_RR     = D_RR[kept]
        U_RR     = U_RR[:,kept].copy()
        D_RR_inv = (1.0/D_RR).copy()
        PROJ     = THC_scheduler._evaluate_UDUT_mutliscale(U_RR, D_RR_inv)  # have serious numerical problem, but still cannot be fixed by this subroutine
        #logger.timer(self, 'build projector', *time1)
        
        self.projector = PROJ
        
        D_RR_sqrt     = np.sqrt(D_RR)
        D_RR_inv_sqrt = 1.0/D_RR_sqrt
        
        self.projector_sqrt     = U_RR @ np.diag(D_RR_inv_sqrt) @ U_RR.T
        self.projector_inv_sqrt = U_RR @ np.diag(D_RR_sqrt) @ U_RR.T
        
        logger.timer(self, 'build thc projector', *time0)
        
        ### check tranpose ### 
        
        diff1 = np.linalg.norm(PROJ-PROJ.T) # numerical problem
        diff2 = np.linalg.norm(self.projector_sqrt-self.projector_sqrt.T)
        diff3 = np.linalg.norm(self.projector_inv_sqrt-self.projector_inv_sqrt.T)
        
        self.projector = (PROJ + PROJ.T) / 2.0
        self.projector_sqrt = (self.projector_sqrt + self.projector_sqrt.T) / 2.0
        self.projector_inv_sqrt = (self.projector_inv_sqrt + self.projector_inv_sqrt.T) / 2.0
        
        print("diff1 = ", diff1)
        print("diff2 = ", diff2)
        print("diff3 = ", diff3)
        
        if self.use_torch:
            self.projector = _to_torch_tensor(self.projector, self.with_gpu)
            self.projector_sqrt = _to_torch_tensor(self.projector_sqrt, self.with_gpu)
            self.projector_inv_sqrt = _to_torch_tensor(self.projector_inv_sqrt, self.with_gpu)
            self._proj = self.projector
            self._proj_robust = self.projector_sqrt
            self._proj_mp2 = None
        
        return PROJ

    def init_mp2_projector(self, mp2_cutoff=1e-3, df_obj=None):
        
        time0 = logger.process_clock(), logger.perf_counter()
        
        tau_o = _thc_rccsd_ind._tensor_to_cpu(self.tau_o)
        tau_v = _thc_rccsd_ind._tensor_to_cpu(self.tau_v)
        
        ### not positive definite, abandon this scheme ! 
        
        # Z   = _thc_rccsd_ind._tensor_to_cpu(self.Z)
        # D_Z, U_Z = np.linalg.eigh(Z)
        # print(D_Z)
        # kept = D_Z > D_Z[-1]*1e-14  ## check whether it is positive definite
        # U_Z = U_Z[:, kept]
        # D_Z = D_Z[kept]
        
        # build L_ia^C 
        
        if df_obj is None:
            if hasattr(self, "df_obj"):
                df_obj = self.df_obj
            else:
                ### build one with default settings ###
                
                from pyscf.pbc import scf
                
                time1 = logger.process_clock(), logger.perf_counter()
                _mf = scf.RHF(self.mol).density_fit()
                _mf.with_df.build() # build 
                df_obj = _mf.with_df
                logger.timer(self, 'build df obj', *time1)
        
        naux = df_obj.get_naoaux()
        nmo  = self.mo_coeff.shape[1]
        nocc = self.nocc
        nvir = nmo - nocc
        nlaplace = self.n_laplace
        mo = numpy.asarray(self.mo_coeff, order='F')
        
        print("naux ", naux)
        print("nmo  ", nmo)
        print("nocc ", nocc)
        print("nvir ", nvir)
        
        L_cW_ia = np.zeros((naux, nlaplace, nocc, nvir))
        
        ijslice = (0, nmo, 0, nmo)
        p1 = 0
        Lpq = None
        for k, eri1 in enumerate(df_obj.loop()):
            from pyscf.ao2mo import _ao2mo
            Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
            p0, p1 = p1, p1 + Lpq.shape[0]
            Lpq = Lpq.reshape(p1-p0,nmo,nmo)
            #Loo[p0:p1] = Lpq[:,:nocc,:nocc]
            Lov = Lpq[:,:nocc,nocc:]
            #Lvv = lib.pack_tril(Lpq[:,nocc:,nocc:])
            #eris.vvL[:,p0:p1] = Lvv.T
            L_cW_ia[p0:p1,:,:,:] = np.einsum("iW,aW,Cia->CWia", tau_o, tau_v, Lov, optimize=True)
        
        M_cW_cW = np.einsum("CWia,DXia->CWDX", L_cW_ia, L_cW_ia, optimize=True)
        M_cW_cW = M_cW_cW.reshape(nlaplace*naux, nlaplace*naux)
        TAU_V, V_CW_V = np.linalg.eigh(M_cW_cW)
        kept = TAU_V > TAU_V[-1]*mp2_cutoff
        print("condition number = ", TAU_V[-1]/TAU_V[0])
        print("nkept = ", np.sum(kept))
        #print("TAU_V = ", TAU_V)
        exit(1)
        
        logger.timer(self, 'build mp2 projector', *time0)

        #exit(1)
        
    #############################################################################################################################################

    def add_input(self, name, tensor):
        if name in self._input_tensor_name:
            self.update_input(name, tensor)
        else:
            setattr(self, "_"+name.lower(), tensor)
            self._input_tensor_name.append(name)
            self.tensor_name_to_attr_name[name] = "_"+name.lower()

    def update_input(self, name, tensor):
        assert name in self._input_tensor_name
        if self.use_torch:
            tensor = _to_torch_tensor(tensor, self.with_gpu)
        setattr(self, self.tensor_name_to_attr_name[name], tensor)
    
    def update_t1(self, t1):
        self.t1 = t1
        if self.use_torch and isinstance(self.t1, np.ndarray):
            self.t1 = _to_torch_tensor(self.t1, self.with_gpu)
    
    def update_t2(self, thc_t2):
        self.thc_t2 = thc_t2
        if self.use_torch and isinstance(self.thc_t2, np.ndarray):
            self.thc_t2 = _to_torch_tensor(self.thc_t2, self.with_gpu)

    def update_r1(self, r1):
        self.r1 = r1
        if self.use_torch and isinstance(self.r1, np.ndarray):
            self.r1 = _to_torch_tensor(self.r1, self.with_gpu)
    
    def update_r2(self, r2):
        self.r2 = r2
        if self.use_torch and isinstance(self.r2, np.ndarray):
            self.r2 = _to_torch_tensor(self.r2, self.with_gpu)

    #############################################################################################################################################
    
    def register_intermediates(self, name, expression):
        #print("Register intermediate: ", name)
        if name in self._input_tensor_name:
            return
        if name not in self._registered_intermediates_name:
            expression.name = name
            self._registered_intermediates_name.append(name)
            if hasattr(expression, "cached"):
                assert expression.cached == True
            if hasattr(expression, "name"):
                expression.name = name
            self._expr_intermediates.append(expression)
                
    def register_expr(self, name, expression):
        if name in self._registered_intermediates_name:
            raise ValueError("Expression with name {} already registered.".format(name))
        expression.name = name
        self.expr[name] = expression

        ### special treatment for t2 equations as we have to apply proj on this equation ###
        
        if name == THC_scheduler.t2_new_name: 
            
            ## first check the equation ## 
            
            assert expression.nocc == 2 and expression.nvir == 2
            occvir_str = expression.occvir_str
            indices_str = None
            if occvir_str == "oovv":
                expression._map_indices(["i", "j", "a", "b"])
                indices_str = "ijab"
            else:
                assert occvir_str == "ovov"
                expression._map_indices(["i", "a", "j", "b"])
                indices_str = "iajb"
            ## apply the projector ##
            _xo_expr = _einsum_term("XO_T2", "iP", 1.0, ["XO_T2"])
            _xv_expr = _einsum_term("XV_T2", "aP", 1.0, ["XV_T2"])
            if self.proj_type == "thc" or self.proj_type == "thc_laplace":
                _proj_expr = _einsum_term("PROJ", "PQ", 1.0, ["PROJ"])
            else:
                assert self.proj_type == "thc_robust"
                _proj_expr = _einsum_term("PROJ_ROBUST", "PQ", 1.0, ["PROJ_ROBUST"])
            expression_new = thc_einsum_sybolic("AP,iP,aP,%s,jQ,bQ,QB->AB" % indices_str, 
                                                    _proj_expr, 
                                                    _xo_expr, _xv_expr, 
                                                    expression, 
                                                    _xo_expr, _xv_expr, 
                                                    _proj_expr,
                                                    cached=True)
            expression_new.name = name
            self.expr[name] = expression_new

    #############################################################################################################################################

    def get_tensor(self, name):
        
        assert isinstance(name, str)
        
        if name in self._input_tensor_name:
            return getattr(self, self.tensor_name_to_attr_name[name])
        elif name in self._registered_intermediates_name:
            res =  self._cached_intermediates[self._registered_intermediates_name.index(name)]
            if res is None:
                #print("Evaluate intermediate: ", name)
                res = self._expr_intermediates[self._registered_intermediates_name.index(name)].contract(self)
                self._cached_intermediates[self._registered_intermediates_name.index(name)] = res
            return res
        else:
            if name in self._expr_cached and self._expr_cached[name] is not None:
                return self._expr_cached[name]
            else:
                #print("Evaluate expression: ", name)
                res = self.expr[name].contract(self)
                self._expr_cached[name] = res
                return res

    def _build_expression(self):
        
        t1 = (logger.process_clock(), logger.perf_counter())
        
        self._dependency = {}
        
        assert len(self._registered_intermediates_name) == len(self._expr_intermediates)
        
        for _id_ in range(len(self._expr_intermediates)):
            self._expr_intermediates[_id_], dep = _parse_expression(self._expr_intermediates[_id_], self)
            self._expr_intermediates[_id_].rename_subterms()
            self._dependency[self._expr_intermediates[_id_].name] = dep
        
        for name, expression in self.expr.items():
            self.expr[name], dep = _parse_expression(expression, self)
            self._dependency[name] = dep
            self.expr[name].rename_subterms()
        
        #### build the hierarchy ####

        self._intermediates_hierarchy = []
        
        all_intermediates = list(self.expr.keys())
        #print("all_intermediates: ", all_intermediates)
        all_intermediates.extend(self._registered_intermediates_name)
        #print("all_intermediates: ", all_intermediates)
        
        removed_intermediates = copy.deepcopy(self._input_tensor_name)
        
        nlayer = 0
        
        while True:
            intermediates_this_layer = []
            for name in all_intermediates:
                #print("name: ", name)
                #print("removed_intermediates: ", removed_intermediates)
                if all([dep in removed_intermediates for dep in self._dependency[name]]):
                    intermediates_this_layer.append(name)
            self._intermediates_hierarchy.append(intermediates_this_layer)
            for name in intermediates_this_layer:
                all_intermediates.remove(name)
            removed_intermediates.extend(intermediates_this_layer)
            nlayer += 1
            if len(all_intermediates) == 0:
                break
                
        self._cached_intermediates = [None] * len(self._registered_intermediates_name)
        
        t2 = (logger.process_clock(), logger.perf_counter())

        print("_build_expression time %.2f %.2f" % (t2[0]-t1[0],t2[1]-t1[1]))
    
    def _build_contraction(self, backend, **kwargs):
        
        assert len(self._intermediates_hierarchy) > 0
        assert len(self._registered_intermediates_name) == len(self._expr_intermediates)
        assert len(self._cached_intermediates) == len(self._registered_intermediates_name)

        NLAYER = len(self._intermediates_hierarchy)
        
        t1 = (logger.process_clock(), logger.perf_counter())
        
        if self.use_torch:
            import torch
            for name in self._input_tensor_name:
                if not hasattr(self, self.tensor_name_to_attr_name[name]):
                    setattr(self, self.tensor_name_to_attr_name[name], None)
                if isinstance(getattr(self, self.tensor_name_to_attr_name[name]), np.ndarray):
                    setattr(self, self.tensor_name_to_attr_name[name], _to_torch_tensor(getattr(self, self.tensor_name_to_attr_name[name]), self.with_gpu))
        
        for i in range(NLAYER):
            for name in self._intermediates_hierarchy[i]:
                if name in self._registered_intermediates_name:
                    self._expr_intermediates[self._registered_intermediates_name.index(name)].build_contraction_path(self, backend, **kwargs)
                else:
                    assert name in self.expr
                    self.expr[name].build_contraction_path(self, backend, **kwargs)
                if i < (NLAYER-1):
                    #print("name %s is finished" % name)
                    self._evaluate(name)

        t2 = (logger.process_clock(), logger.perf_counter())
    
        print("_build_contraction time %.2f %.2f" % (t2[0]-t1[0],t2[1]-t1[1]))
    
    #############################################################################################################################################
    
    def _evaluate(self, expr_name):
        
        if expr_name in self._input_tensor_name:
            return self.get_tensor(expr_name)
        elif expr_name in self._registered_intermediates_name:
            self._cached_intermediates[self._registered_intermediates_name.index(expr_name)] = None
            return self.get_tensor(expr_name)
        else:
            assert expr_name in self.expr
            self._expr_cached[expr_name] = None
            return self.get_tensor(expr_name)
        
    def evaluate_all(self):
        NLAYER = len(self._intermediates_hierarchy)
        for i in range(NLAYER):
            for name in self._intermediates_hierarchy[i]:
                self._evaluate(name)
    
    def evaluate_all_intermediates(self):
        NLAYER = len(self._intermediates_hierarchy)
        for i in range(NLAYER):
            for name in self._intermediates_hierarchy[i]:
                if name in self._registered_intermediates_name:
                    self._cached_intermediates[self._registered_intermediates_name.index(name)] = None
                    self._evaluate(name)
                else:
                    assert name in self.expr
                    self._expr_cached[name] = None

    def evaluate_t1_t2(self, t1, thc_t2, evaluate_ene=True, only_ene=False):
        
        NAME = [THC_scheduler.t1_new_name, THC_scheduler.t2_new_name, THC_scheduler.ccsd_energy_name]
        
        self.update_t1(t1)
        self.update_t2(thc_t2)

        #### start evaluation ####
        
        NLAYER = len(self._intermediates_hierarchy)
        
        for i in range(NLAYER):
            for name in self._intermediates_hierarchy[i]:
                if name != THC_scheduler.ccsd_energy_name:
                    if only_ene:
                        if name not in NAME:
                            self._evaluate(name)
                    else:
                        print("Evaluate: ", name)
                        self._evaluate(name)

        #### evaluate energy ####
        
        if evaluate_ene:
            self._evaluate(THC_scheduler.ccsd_energy_name)
            if self.use_torch:
                self._expr_cached[THC_scheduler.ccsd_energy_name] = self._expr_cached[THC_scheduler.ccsd_energy_name].item()
            if only_ene:
                return self._expr_cached[THC_scheduler.ccsd_energy_name]
            else:
                return self._expr_cached[THC_scheduler.ccsd_energy_name], self._expr_cached[THC_scheduler.t1_new_name], self._expr_cached[THC_scheduler.t2_new_name]
        else:
            if only_ene:
                return None, None, None
            else:
                return None, self._expr_cached[THC_scheduler.t1_new_name], self._expr_cached[THC_scheduler.t2_new_name]
    
    def energy(self, t1, thc_t2):
        return self.evaluate_t1_t2(t1, thc_t2, evaluate_ene=True, only_ene=True)

####### deal with thc einsum with expressions #######


if __name__ == "__main__":
    
    ######## test the basic functions of _einsum_term ########
    
    #print(_thc_eri_oooo())
    #print(_thc_eri_ovoo())
    #print(_thc_eri_ovov())
    
    #print("Done!")
    
    test = _thc_eri_ovoo()
    #print(test)
    #print("conj")
    test1 = test.conj()
    #print(test1)
    #print("transpose")
    test2 = test.transpose((0, 2, 1, 3))
    #print(test2)
    
    #test2._map_indices({"i": "P", "a": "Q"})
    test2._map_indices({"i": "j", "k": "i"})
    #print(test2)
    #print(test2 * 1.3)
    #print(2.0 * test2)
    #print(test2 * complex(1.0, 2.0))
    #print(test1)
    #print(test1 * complex(1.0, 2.0))
    
    #print(to_expr_holder(test2))
    
    #print(test1)
    
    test1 *=1.45
    
    #print(test1)
    
    #test1 /=1.45
    
    #print(test1/1.45)
    test1 /=1.45
    #print(test1)
    
    ######## test scheduler ########
    
    #print(list("ijab"))
    
    nocc = 8
    nvir = 8
    nthc = 32
    ntau = 9
    
    X_o = np.random.rand(nocc, nthc)
    X_v = np.random.rand(nvir, nthc)
    THC_INT = np.random.rand(nthc, nthc)
    Tau_o = np.random.rand(nocc, ntau)
    Tau_v = np.random.rand(nvir, ntau)
    X_o_T2 = np.random.rand(nocc, nthc)
    X_v_T2 = np.random.rand(nvir, nthc)
    Proj   = np.random.rand(nthc, nthc) 
    
    fov = np.random.rand(nocc, nvir)
    foo = np.random.rand(nocc, nocc)
    fvv = np.random.rand(nvir, nvir)
    
    scheduler_test = THC_scheduler(X_o, X_v, THC_INT, Tau_o, Tau_v, X_o_T2, X_v_T2, Proj)
    
    THC_scheduler.add_input(scheduler_test, "fov", fov)
    THC_scheduler.add_input(scheduler_test, "foo", foo)
    THC_scheduler.add_input(scheduler_test, "fvv", fvv)
    
    #### use one of the intermediates ####
    
    t1  = _expr_t1()
    t2  = _expr_t2()
    t2  = t2.transpose((0, 2, 1, 3)) # pyscf convention
    foo = _expr_foo()
    fvv = _expr_fvv()
    fov = _expr_fov()
    eris_ovov = _thc_eri_ovov()
    eris_ovvv = _thc_eri_ovvv()
    eris_ooov = _thc_eri_ooov()
    eris_ovvo = _thc_eri_ovvo()
    eris_oovv = _thc_eri_oovv()
    
    ### FOO 
    
    Fki  = 2*thc_einsum_sybolic('kcld,ilcd->ki', eris_ovov, t2, cached=True) 
    Fki -= thc_einsum_sybolic('idlc,jlcd->ij', eris_ovov, t2, cached=True)
    Fki += 2*thc_einsum_sybolic('kcld,ic,ld->ki', eris_ovov, t1, t1, cached=True)
    Fki -= thc_einsum_sybolic('kdlc,ic,ld->ki', eris_ovov, t1, t1, cached=True)
    Fki += foo
    Fki.name = "FOO"
    Foo = Fki
    
    ### FVV 
    
    Fac  = 2*thc_einsum_sybolic('kcld,klad->ac', eris_ovov, t2, cached=True)
    Fac += thc_einsum_sybolic('kdlc,klad->ac', eris_ovov, t2, cached=True)
    Fac -= 2*thc_einsum_sybolic('kcld,ka,ld->ac', eris_ovov, t1, t1, cached=True)
    Fac += thc_einsum_sybolic('kdlc,ka,ld->ac', eris_ovov, t1, t1, cached=True)
    Fac += fvv
    Fac.name = "FVV"
    Fvv = Fac
    
    ### FOV
    
    Fkc  = 2 * thc_einsum_sybolic('kcld,ld->kc', eris_ovov, t1, cached=True)
    Fkc -= thc_einsum_sybolic('kdlc,ld->kc', eris_ovov, t1, cached=True)
    Fkc += fov
    Fov = Fkc
    
    scheduler_test.register_intermediates("FOO", Fki)
    scheduler_test.register_intermediates("FVV", Fac)
    scheduler_test.register_intermediates("FOV", Fkc)
    
    ### T1 equation
    
    
    #print(Fki)
    #print(scheduler_test._expr_intermediates[0])
    
    scheduler_test._build_expression()
    
    t1new = to_expr_holder(fov)
    t1new.conj()
    
    t1new -= 2*thc_einsum_sybolic('kc,ka,ic->ia', fov, t1, t1, cached=True)
    t1new += thc_einsum_sybolic('ac,ic->ia', Fvv, t1, cached=True)
    t1new +=   thc_einsum_sybolic('ac,ic->ia', Fvv, t1, cached=True)
    t1new +=  -thc_einsum_sybolic('ki,ka->ia', Foo, t1, cached=True)
    t1new += 2*thc_einsum_sybolic('kc,kica->ia', Fov, t2, cached=True)
    t1new +=  -thc_einsum_sybolic('kc,ikca->ia', Fov, t2, cached=True)
    t1new +=   thc_einsum_sybolic('kc,ic,ka->ia', Fov, t1, t1, cached=True)
    t1new += 2*thc_einsum_sybolic('kcai,kc->ia', eris_ovvo, t1, cached=True)
    t1new +=  -thc_einsum_sybolic('kiac,kc->ia', eris_oovv, t1, cached=True)
    #eris_ovvv = np.asarray(eris.ovvv)
    t1new += 2*thc_einsum_sybolic('kdac,ikcd->ia', eris_ovvv, t2, cached=True)
    t1new +=  -thc_einsum_sybolic('kcad,ikcd->ia', eris_ovvv, t2, cached=True)
    t1new += 2*thc_einsum_sybolic('kdac,kd,ic->ia', eris_ovvv, t1, t1, cached=True)
    t1new +=  -thc_einsum_sybolic('kcad,kd,ic->ia', eris_ovvv, t1, t1, cached=True)
    t1new +=-2*thc_einsum_sybolic('kilc,klac->ia', eris_ooov, t2, cached=True)
    t1new +=   thc_einsum_sybolic('likc,klac->ia', eris_ooov, t2, cached=True)
    t1new +=-2*thc_einsum_sybolic('kilc,lc,ka->ia', eris_ooov, t1, t1, cached=True)
    t1new +=   thc_einsum_sybolic('likc,lc,ka->ia', eris_ooov, t1, t1, cached=True)
    
    scheduler_test.register_expr("T1_NEW", t1new)

    scheduler_test._build_expression()
    
    print(scheduler_test)