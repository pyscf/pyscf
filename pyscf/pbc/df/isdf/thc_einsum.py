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

import numpy
import numpy as np

### function to determine the bunchsize ### 

_numpy_einsum = np.einsum
_einsum_path  = numpy.einsum_path

OCC_INDICES      = ["i", "j", "k", "l", "m", "n"]
VIR_INDICES      = ["a", "b", "c", "d", "e", "f", "g", "h"]
THC_INDICES_LIST = ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
   
class thc_holder:
    
    def __init__(self, X_o, X_v, Z):
        self.X_o = X_o
        self.Z   = Z
        self.X_v = X_v
        
        assert self.X_o.shape[1] == self.Z.shape[0]
        assert self.X_o.shape[1] == self.Z.shape[1]
        assert self.X_o.shape[1] == self.X_v.shape[1]

    def get_eri(self, eri_subscript:str, thc_subscript:str):
        
        assert len(eri_subscript) == 4
        assert len(thc_subscript) == 2
        assert isinstance(eri_subscript, str)
        assert isinstance(thc_subscript, str)
        
        res_tensor = []
        res_tensor.append(self.Z)
        res_subscript = thc_subscript+","
        
        for i in range(4):
            if eri_subscript[i] in OCC_INDICES:
                res_tensor.append(self.X_o)
            elif eri_subscript[i] in VIR_INDICES:
                res_tensor.append(self.X_v)
            else:
                raise ValueError
            
            if i<2:
                res_subscript += eri_subscript[i]+thc_subscript[0]+","
            else:
                res_subscript += eri_subscript[i]+thc_subscript[1]+","
        
        return res_tensor, res_subscript
            

class energy_denomimator:
    
    def __init__(self, tau_o, tau_v):
        
        self.tau_o = tau_o
        self.tau_v = tau_v

        assert self.tau_o.shape[1] == self.tau_v.shape[1]

    def get_energy_denominator(self, subscript, laplace_index):
        
        assert len(subscript) == 4
        assert len(laplace_index) == 1
        assert isinstance(subscript, str)
        assert isinstance(laplace_index, str)
        
        res_tensor = []
        res_subscript = ""
        
        for i in range(4):
            if subscript[i] in OCC_INDICES:
                res_tensor.append(self.tau_o)
            elif subscript[i] in VIR_INDICES:
                res_tensor.append(self.tau_v)
            else:
                raise ValueError

            res_subscript += subscript[i]+laplace_index+","

        return res_tensor, res_subscript

##### helper func for cotengra backend ####

def _size_dict_cotengra(inputs, *tensors):

    size_dict = {}
    for idx, tensor in zip(inputs, tensors):
        
        assert isinstance(tensor, np.ndarray)
        assert tensor.ndim == len(idx)

        for _idx_ in idx:
            if _idx_ not in size_dict:
                size_dict[_idx_] = tensor.shape[idx.index(_idx_)]
            else:
                assert size_dict[_idx_] == tensor.shape[idx.index(_idx_)]

    return size_dict

def thc_einsum(subscripts, *tensors, **kwargs):
    
    contract   = kwargs.pop('_contract', _numpy_einsum)  ### we have to call for the numpy einsum function as we will deal with i,i->i
    subscripts = subscripts.replace(' ','')
    backend    = kwargs.pop('backend', None)
    return_path_only = kwargs.pop('return_path_only', False)
    memory           = kwargs.pop('memory', 2**28)
    
    if isinstance(backend, str):
        backend = backend.lower()
    else:
        assert backend is None
    
    use_cotengra = False
    if backend == "cotengra":
        import cotengra as ctg
        use_cotengra = True
    
    if len(tensors) <= 1 or '...' in subscripts:
        #out = _numpy_einsum(subscripts, *tensors, **kwargs)
        raise NotImplementedError
    elif len(tensors) <= 2:
        #out = _contract(subscripts, *tensors, **kwargs)
        raise NotImplementedError
    else:
        optimize = kwargs.pop('optimize', True)
        
        ### split subscripts ### 
        
        subscripts = subscripts.split('->')
        lhs        = subscripts[0]
        rhs        = subscripts[1]
        assert len(rhs)<=0  ## currently only support energy expression

        tensors = list(tensors)
        tensors_scripts = lhs.split(",")
        
        tensors_2         = []
        tensors_scripts_2 = ""
        
        n_THC_laplace_indices = 0
        
        for _tensor_, _script_ in zip(tensors, tensors_scripts):
            if isinstance(_tensor_, thc_holder):
                if n_THC_laplace_indices + 2 > len(THC_INDICES_LIST):
                    raise ValueError("number of thc and laplace indices exhausted")
                thc_indices                 = THC_INDICES_LIST[n_THC_laplace_indices] + THC_INDICES_LIST[n_THC_laplace_indices+1]
                n_THC_laplace_indices      += 2
                tmp_tensors, tmp_subscripts = _tensor_.get_eri(_script_, thc_indices)
                tensors_2.extend(tmp_tensors)
                tensors_scripts_2 += tmp_subscripts
            elif isinstance(_tensor_, energy_denomimator):
                if n_THC_laplace_indices + 1 > len(THC_INDICES_LIST):
                    raise ValueError("number of thc and laplace indices exhausted")
                laplace_index               = THC_INDICES_LIST[n_THC_laplace_indices]
                n_THC_laplace_indices      += 1
                tmp_tensors, tmp_subscripts = _tensor_.get_energy_denominator(_script_, laplace_index)
                tensors_2.extend(tmp_tensors)
                tensors_scripts_2 += tmp_subscripts
            else: ### normal tensor ###
                tensors_2.append(_tensor_)
                tensors_scripts_2 += _script_ + ","
        
        tensors_scripts_2 = tensors_scripts_2[:-1]
        subscripts_2      = tensors_scripts_2 + "->" + rhs
        
        ##### code deal with different backend 
        
        if use_cotengra:
            
            inputs, output = ctg.utils.eq_to_inputs_output(subscripts_2)
            size_dict      = _size_dict_cotengra(inputs, *tensors_2)
            
            # print(size_dict)
            
            # find a tree sliced to memory, (e.g. 2**28 (~8GB) unit?)
            
            opt = ctg.HyperOptimizer(
                slicing_reconf_opts=dict(
                    target_size=memory,
                ),
                progbar=False,
            )
            tree = opt.search(inputs, output, size_dict)
            
            if return_path_only:
                return tree            
            else:
                return tree.contract(tensors_2)
        
        else:
            contraction_list = _einsum_path(subscripts_2, *tensors_2, optimize=optimize,
                                            einsum_call=True)[1]
        
            if return_path_only:
                return contraction_list
        
            for contraction in contraction_list:
                inds, idx_rm, einsum_str, remaining = contraction[:4]
                tmp_operands = [tensors_2.pop(x) for x in inds]
                if len(tmp_operands) > 2:
                    out = _numpy_einsum(einsum_str, *tmp_operands)
                else:
                    out = contract(einsum_str, *tmp_operands)
                tensors_2.append(out)
    
    return out

if __name__ == "__main__":
    
    NOCC = 16
    NVIR = 16
    NTHC = 320
    N_LAPLACE = 9
    
    X_o = np.random.random((NOCC, NTHC))
    X_v = np.random.random((NVIR, NTHC))
    Z   = np.random.random((NTHC, NTHC))
    Z   = (Z+Z.T)/2
    
    tau_o = np.random.random((NOCC, N_LAPLACE))
    tau_v = np.random.random((NVIR, N_LAPLACE))
    
    eri      = thc_holder(X_o, X_v, Z)
    ene_deno = energy_denomimator(tau_o, tau_v)
    
    mp2_J  = thc_einsum("iajb,iajb,ijab->", eri, eri, ene_deno, backend="cotengra")
    mp2_K  = thc_einsum("iajb,ibja,ijab->", eri, eri, ene_deno, backend="cotengra")
    # mp2_J2 = thc_einsum("iajb,iajb,ijab->", eri, eri, ene_deno, optimize='greedy')
    # mp2_K  = thc_einsum("iajb,ibja,ijab->", eri, eri, ene_deno, optimize='greedy')
    # assert np.allclose(mp2_J, mp2_J2)
    
    print(-2*mp2_J+mp2_K)
    
    mp3_CX_1 = thc_einsum("iajb,jkbc,iakc,ijab,ikac->", eri, eri, eri, ene_deno, ene_deno, backend="cotengra")
    
    print(mp3_CX_1)