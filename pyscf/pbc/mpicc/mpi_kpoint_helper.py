#!/usr/bin/env python
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
#
# Authors: James D. McClain
#          Timothy Berkelbach <tim.berkelbach@gmail.com>
#

import numpy
import pyscf.pbc.ao2mo
import pyscf.lib
from pyscf.pbc.lib import kpts_helper

DEBUG = 0

class unique_pqr_list:
    #####################################################################################
    # The following only computes the integrals not related by permutational symmetries.
    # Wasn't sure how to do this 'cleanly', but it's fairly straightforward
    #####################################################################################
    def __init__(self,cell,kpts):
        kconserv = kpts_helper.get_kconserv(cell,kpts)
        nkpts = len(kpts)
        temp = range(0,nkpts)
        klist = pyscf.lib.cartesian_prod((temp,temp,temp))
        completed = numpy.zeros((nkpts,nkpts,nkpts),dtype=int)

        self.operations = numpy.zeros((nkpts,nkpts,nkpts),dtype=int)
        self.equivalentList = numpy.zeros((nkpts,nkpts,nkpts,3),dtype=int)
        self.nUnique = 0
        self.uniqueList = numpy.array([],dtype=int)

        ivec = 0
        not_done = True
        while (not_done):
            current_kvec = klist[ivec]
            # check to see if it's been done...
            kp = current_kvec[0]
            kq = current_kvec[1]
            kr = current_kvec[2]
            #print "computing ",kp,kq,kr
            if completed[kp,kq,kr] == 0:
                self.nUnique += 1
                self.uniqueList = numpy.append(self.uniqueList,current_kvec)
                ks = kconserv[kp,kq,kr]

                # Now find all equivalent kvectors by permuting it all possible ways...
                # and then storing how its related by symmetry
                completed[kp,kq,kr] = 1
                self.operations[kp,kq,kr] = 0
                self.equivalentList[kp,kq,kr] = current_kvec.copy()

                completed[kr,ks,kp] = 1
                self.operations[kr,ks,kp] = 1 #.transpose(2,3,0,1)
                self.equivalentList[kr,ks,kp] = current_kvec.copy()

                completed[kq,kp,ks] = 1
                self.operations[kq,kp,ks] = 2 #numpy.conj(.transpose(1,0,3,2))
                self.equivalentList[kq,kp,ks] = current_kvec.copy()

                completed[ks,kr,kq] = 1
                self.operations[ks,kr,kq] = 3 #numpy.conj(.transpose(3,2,1,0))
                self.equivalentList[ks,kr,kq] = current_kvec.copy()

            ivec += 1
            if ivec == len(klist):
                not_done = False

        self.uniqueList = self.uniqueList.reshape(self.nUnique,-1)
        if DEBUG == 1:
            print("::: kpoint helper :::")
            print("kvector list (in)")
            print("   shape = ", klist.shape)
            print("kvector list (out)")
            print("   shape  = ", self.uniqueList.shape)
            print("   unique list =")
            print(self.uniqueList)
            print("transformation =")
            for i in range(klist.shape[0]):
                pqr = klist[i]
                irr_pqr = self.equivalentList[pqr[0],pqr[1],pqr[2]]
                print("%3d %3d %3d   ->  %3d %3d %3d" % (pqr[0],pqr[1],pqr[2],
                                                         irr_pqr[0],irr_pqr[1],irr_pqr[2]))

    def get_uniqueList(self):
        return self.uniqueList

    def get_irrVec(self,kp,kq,kr):
        return self.equivalentList[kp,kq,kr]

    def get_transformation(self,kp,kq,kr):
        return self.operations[kp,kq,kr]

    ######################################################
    # for the invec created out of our unique list from
    # the irreducible brillouin zone, we transform it to
    # arbitrary kp,kq,kr
    ######################################################
    def transform_irr2full(self,invec,kp,kq,kr):
        operation = self.get_transformation(kp,kq,kr)
        if operation == 0:
            return invec
        if operation == 1:
            return invec.transpose(2,3,0,1)
        if operation == 2:
            return numpy.conj(invec.transpose(1,0,3,2))
        if operation == 3:
            return numpy.conj(invec.transpose(3,2,1,0))
