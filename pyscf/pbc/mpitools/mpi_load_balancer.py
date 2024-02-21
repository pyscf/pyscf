# Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
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

from mpi4py import MPI
import pyscf.lib
import numpy as np

import sys
from time import sleep

def enum(**enums):
    return type('Enum', (), enums)

tags = enum(WORK=1, WORK_DONE=2, KILL=3)

class load_balancer:
    def __init__(self,BLKSIZE):
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.size = MPI.COMM_WORLD.Get_size()
        self.COMM = MPI.COMM_WORLD
        self.BLKSIZE = BLKSIZE
        self.curr_block = None

    def set_ranges(self,inindices,BLKSIZE=None):
        rank = self.rank
#        if rank == 0:
#            print("Starting new mpi_load_balance...")
#        print("proc ", rank, " setting ranges ", inindices)
        if BLKSIZE is None:
            BLKSIZE = self.BLKSIZE
        if len(BLKSIZE) != len(inindices):
            print("BLKSIZE AND ININDICES MUST HAVE SAME SHAPE!!!!")
            print("rank, BLKSIZE, inindices")
            print(self.rank, BLKSIZE, inindices)
            sys.exit()
        self.nindices = len(inindices)
        ##print("nindices = ", self.nindices)
        ##print("inindices = ", inindices)
        ##print("BLKSIZE = ", BLKSIZE)
        outblocks = []
        for i in range(self.nindices):

            max_range = max(inindices[i])
            min_range = min(inindices[i])
            ranges_size = len(inindices[i])
            nblocks = int(np.ceil(ranges_size/(1.*BLKSIZE[i])))
            ##print("nblocks for range (%3d-%3d) = %3d" % (min_range, max_range, nblocks))
            segment_blocks = []
            for j in range(nblocks):
                block_j_min = min_range + BLKSIZE[i]*j
                block_j_max = min(max_range+1,min_range+BLKSIZE[i]*(j+1))
                segment_blocks.append(range(block_j_min,block_j_max))
#            if rank == 0:
#                print("segment blocks [segment %3d]: " % i)
#                print(segment_blocks)
            outblocks.append(segment_blocks)
        ###print("index 0 block 1")
        ###print(outblocks[0][1])
        self.outblocks = np.asarray(outblocks)
        #if rank == 0:
        #    ##print("final block structure")
        #    ##print(outblocks)
        if rank == 0:
            self.master()

    def master(self):
        status = MPI.Status()
        work = []
        for index in range(0,self.nindices):
            work.append(range(len(self.outblocks[index])))
        block_indices = pyscf.lib.cartesian_prod(work)
        nwork = len(block_indices)
        iwork = 0
        iwork_recieved = 0
        working_procs=[]
        for i in range(1,self.size):
            data = 0
            tag = tags.KILL
            if iwork < nwork:
                data = block_indices[iwork]
                tag = tags.WORK
                working_procs.append(i)
                #print("MASTER : sending out msg to processor ", i, data)
            self.COMM.isend(obj=data, dest=i, tag=tag)
            iwork += 1

        for i in range(iwork,nwork+1):
            self.COMM.Probe(MPI.ANY_SOURCE, tag=tags.WORK_DONE, status=status)
            recieved_from = status.Get_source()
            data = self.COMM.recv(source=recieved_from, tag=tags.WORK_DONE)
            iwork_recieved += 1
            #print("MASTER : just recieved work_done from processor ", recieved_from, " (",iwork_recieved,"/",nwork,")")
            if i == nwork:
                #print("MASTER : returning...")
                break

            data = block_indices[i]
            tag = tags.WORK
            #print("MASTER : sending out new work to processor ", recieved_from, data)
            self.COMM.isend(obj=data, dest=recieved_from, tag=tag)

        for i in range(iwork_recieved,nwork):
            #print("waiting on work...")
            self.COMM.Probe(MPI.ANY_SOURCE, tag=tags.WORK_DONE, status=status)
            recieved_from = status.Get_source()
            data = self.COMM.recv(source=recieved_from, tag=tags.WORK_DONE)

        # You only have to send a kill to the processors that actually did work, otherwise they were sent a kill
        # at the very first loop
        for i in working_procs:
            data = 0
            tag = tags.KILL
            #print("MASTER (ALL_WORK_DONE): sending kill out to rank ", i, data)
            self.COMM.isend(obj=data, dest=i, tag=tag)

        return

    def slave_set(self):
        if self.rank > 0:
            #print("SLAVE : ", self.rank, " starting...")
            status = MPI.Status()
            #print("SLAVE : ", self.rank, " probing for message...")
            self.COMM.Probe(0, MPI.ANY_TAG, status=status)
            #print("SLAVE : ", self.rank, " recieved a message... ", status.Get_tag())
            self.working = True
            if status.Get_tag() == tags.WORK:
                workingBlock = self.COMM.recv(source=0, tag=tags.WORK)
                #print("SLAVE : ", self.rank, " just recieved ", workingBlock)
                self.curr_block = workingBlock
                self.working = True
                return True, workingBlock
            else:
                self.working = False
                workingBlock = self.COMM.recv(source=0, tag=tags.KILL)
                #print("SLAVE : ", self.rank, " dying...")
                return False, 0
        else:
            return False, 0

    def get_blocks_from_data(self,data):
        nindices = len(data)
        if nindices == 1:
            index = 0
            block = data[index]
            ranges = self.outblocks[index][block]
            return ranges
        outranges = []
        for i in range(nindices):
            index = i
            block = data[i]
            ranges = self.outblocks[index][block]
            outranges.append(ranges)
        return outranges

    def slave_finished(self):
        if self.rank > 0:
            if self.working is True:
                ###print(self.rank, " just finished", self.curr_block)
                self.COMM.isend(obj=self.curr_block, dest=0, tag=tags.WORK_DONE)
#                print("SLAVE : ", self.rank, " just finished work on ", self.curr_block, ". sending tag...")
        else:
            return
