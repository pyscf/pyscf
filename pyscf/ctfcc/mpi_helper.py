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
# Author: Yang Gao <younggao1994@gmail.com>

from mpi4py import MPI
import sys, os
import numpy
import ctf


'''helper functions for ctf'''
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank!=0:
    sys.stdout = open(os.devnull, 'w')    # supress printing

def static_partition(tasks):
    segsize = (len(tasks)+size-1) // size
    start = rank * segsize
    stop = min(len(tasks), start+segsize)
    return tasks[start:stop]

gather = lambda x: ctf.astensor(comm.bcast(x, root=0))

def all_gather(*vals):
    out = [gather(i) for i in vals]
    if len(vals) ==1:
        return out[0]
    else:
        return out

def sym_gather(*vals, sym=None):
    from symtensor.sym_ctf import tensor
    out = [tensor(gather(i),sym) for i in vals]
    if len(vals) ==1:
        return out[0]
    else:
        return out

def argsort(array, nroots=10, thresh=1e-8):
    ind, val = array.read_local()
    nvals = nroots * size
    mask = abs(val) > thresh
    ind, val = ind[mask], val[mask]
    args = numpy.argsort(val)[:nvals]
    ind = ind[args]
    vals = val[args]
    out = numpy.vstack([ind, vals])
    tmp = numpy.hstack(comm.allgather(out))
    ind, val = tmp
    args = numpy.argsort(val)[:nroots]
    return ind[args]

def unpack_idx(idx, *order):
    assert(len(order)>1)
    idx_lst = []
    tmp1 = idx
    for i in range(len(order)-1):
        mod = numpy.prod(order[i+1:])
        tmp, tmp1 = numpy.divmod(tmp1, mod)
        idx_lst.append(tmp)
    idx_lst.append(tmp1)
    return idx_lst

def tri_to_sqr(ind, norb, split=True):
    '''
    -  0  1  2                0  1  2  3
    -  -  3  4                4  5  6  7
    -  -  -  5      --->      8  9 10 11
    -  -  -  -               12 13 14 15
    if split, returns:
        aind: [0,0,0,1,1,2]
        bind: [1,2,3,2,3,3]
    else, return:
        [1,2,3,6,7,11]
    '''
    off = (numpy.arange(norb)+1) * (numpy.arange(norb)+2) //2
    tab = ind[:,None] + off[None,:]
    ref = (numpy.arange(norb)+1) * norb
    aind = norb - numpy.sum(tab<ref, axis=1)
    sqind = ind + (aind+1) * (aind+2)//2
    bind = numpy.divmod(sqind, norb)[1]
    if split:
        return aind, bind
    else:
        return sqind

def sqr_to_tri(idxa, idxb, norb):
    return idxa*norb + idxb - (idxa+1)*(idxa+2)//2
