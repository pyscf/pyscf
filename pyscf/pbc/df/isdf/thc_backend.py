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

####### TORCH BACKEND #######

import numpy

FOUND_TORCH = False
GPU_SUPPORTED = False
GPU_DEVICE_ID = None


try:
    import torch
    FOUND_TORCH = True
except ImportError:
    pass

if FOUND_TORCH:
    if torch.cuda.is_available():
        GPU_SUPPORTED = True
        import os 
        GPU_DEVICE_ID = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if GPU_DEVICE_ID is None:
            GPU_DEVICE_ID = 0
        else:
            GPU_DEVICE_ID = int(GPU_DEVICE_ID)
        # check if the GPU is available
        if torch.cuda.device_count() <= GPU_DEVICE_ID:
            GPU_SUPPORTED = False
            GPU_DEVICE_ID = None
        torch.cuda.set_device(GPU_DEVICE_ID)

##############################

def to_scalar(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy().item()
    return x

def to_numpy_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    return x

def to_torch(x, with_gpu):
    if isinstance(x, torch.Tensor):
        if with_gpu:
            assert GPU_SUPPORTED
            return x.to('cuda:%d'%GPU_DEVICE_ID)
        else:
            return x    
    else:
        assert isinstance(x, numpy.ndarray)
        res = torch.from_numpy(x).detach()
        if with_gpu:
            assert GPU_SUPPORTED
            return res.to('cuda:%d'%GPU_DEVICE_ID)
        else:
            return res
    