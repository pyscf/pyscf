# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

from .mpi_blksize import get_max_blocksize_from_mem
from pyscf.lib.numpy_helper import cartesian_prod
import numpy
from mpi4py import MPI

MEM_SIZE = 0.5e9

def generate_task_list(chunk_size, array_size):
    segs = [range(int(numpy.ceil(array_size[i]*1./chunk_size[i]))) for i in range(len(array_size))]
    task_id = numpy.array(cartesian_prod(segs))
    task_ranges_lower = task_id * numpy.array(chunk_size)
    task_ranges_upper = numpy.minimum((task_id+1)*numpy.array(chunk_size),array_size)
    return list(numpy.dstack((task_ranges_lower,task_ranges_upper)))

def generate_max_task_list(array_size,blk_mem_size=16.,memory=MEM_SIZE,priority_list=None):
    shape = list(array_size)
    length = len(shape)
    if priority_list is None:
        priority_list = numpy.arange(length)[::-1]
    chunk_size = get_max_blocksize_from_mem(shape,blk_mem_size,memory,priority_list)
    return generate_task_list(chunk_size,shape)

def safeNormDiff(in_array1, in_array2):
    shape = in_array1.shape
    assert shape == in_array2.shape
    length = len(shape)
    # just use 16 for blocksize, size of complex(double)
    chunk_size = get_max_blocksize_from_mem(list(shape),16.,MEM_SIZE,priority_list=numpy.arange(length)[::-1])
    task_list = generate_task_list(chunk_size,shape)
    norm = 0.0
    for block in task_list:
        which_slice = [slice(*x) for x in block]
        norm += numpy.linalg.norm(in_array1[tuple(which_slice)] - in_array2[tuple(which_slice)])
    return norm

def safeAllreduceInPlace(comm, in_array):
    shape = in_array.shape
    length = len(shape)
    # just use 16 for blocksize, size of complex(double)
    chunk_size = get_max_blocksize_from_mem(list(shape),16.,MEM_SIZE,priority_list=numpy.arange(length)[::-1])
    task_list = generate_task_list(chunk_size,shape)
    for block in task_list:
        which_slice = [slice(*x) for x in block]
        tmp = in_array[tuple(which_slice)].copy()
        comm.Allreduce(MPI.IN_PLACE, tmp, op=MPI.SUM)
        in_array[tuple(which_slice)] = tmp

def safeBcastInPlace(comm, in_array, root=0):
    shape = in_array.shape
    length = len(shape)
    # just use 16 for blocksize, size of complex(double)
    chunk_size = get_max_blocksize_from_mem(list(shape),16.,MEM_SIZE,priority_list=numpy.arange(length)[::-1])
    task_list = generate_task_list(chunk_size,shape)
    for block in task_list:
        which_slice = [slice(*x) for x in block]
        tmp = in_array[tuple(which_slice)].copy()
        tmp = comm.bcast(tmp,root=0)
        in_array[tuple(which_slice)] = tmp
