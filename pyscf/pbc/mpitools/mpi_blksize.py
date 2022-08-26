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

import numpy
from mpi4py import MPI

comm = MPI.COMM_WORLD
nproc = comm.Get_size()

##########################################################################
# TODO: currently if the mem_per_blocksize is too
#       small then it will make everything just be one gigantic block...
#       which may be bad computationally in that if each processor is
#       responsible for work on a block, only one processor will be used
#       for work.
##########################################################################

def get_max_blocksize_from_mem(array_size, mem_per_block, mem, priority_list=None):
    '''
    Args:
        priority_list : list of importance of indices for the blocksizes,
            i.e. an index with priority 1 will be made to be the
            maximum it can be (according to array_size and mem),
            afterwhich an index with priority 2 will be made the maximum size
            it can be.  If two priorities are the same, the block size
            for those indices will be forced to be equal.
    '''
    #assert((priority_list is not None and hasattr(priority_list, '__iter__')) and
    #        "nchunks (int) or priority_list (iterable) must be specified.")
    #print("memory max = %.8e" % mem)
    nindices = len(array_size)
    if priority_list is None:
        _priority_list = [1]*nindices
    else:
        # still fails for a numpy.array(5), with shape == 0 and no len()
        _priority_list = priority_list
    cmem = mem/mem_per_block # current memory to distribute over blocks
    _priority_list = numpy.array(_priority_list)
    _array_size = numpy.array(array_size)
    idx = numpy.argsort(_priority_list)
    idxinv = numpy.argsort(idx) # maps sorted indices back to original
    _priority_list = _priority_list[idx]
    _array_size = _array_size[idx]
    iprior = 0
    chunksize = []
    loop = True
    while (loop):
        ib = _priority_list[iprior]
        len_b = 1
        for jprior in range(iprior+1,nindices):
            jb = _priority_list[jprior]
            if jb == ib:
                len_b += 1
            else:
                break
        jprior = iprior+len_b
        index_chunks = int(min(min(_array_size[iprior:jprior]),cmem**(1./len_b)))
        iprior = jprior
        index_chunks = max(index_chunks,1)
        for index in range(len_b):
            chunksize.append(index_chunks)
        cmem /= (index_chunks ** len_b)
        if iprior == nindices:
            loop = False
    chunksize = numpy.array(chunksize)[idxinv]
    #print("chunks = ", chunksize)
    #print("mem_per_chunk = %.8e" % (numpy.prod(numpy.asarray(chunksize))*mem_per_block))
    return tuple(chunksize)

