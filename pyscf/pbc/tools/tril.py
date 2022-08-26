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
import pyscf.lib

def tril_index(ki,kj):
    assert (numpy.array([ki<=kj])).all()
    return (kj*(kj+1))//2 + ki

# TODO: fairly messy and slow
def unpack_tril(in_array,nkpts,kp,kq,kr,ks):
    # We are only dealing with the case that one of kp,kq,kr is a list
    #
    if in_array.shape[0] == nkpts:
        return in_array[kp,kq,kr].copy()
    nints = sum([isinstance(x, (int, numpy.integer)) for x in (kp,kq,kr)])
    assert (nints>=2)

    kp,kq,kr,ks = [[x] if isinstance(x, (int, numpy.integer)) else x for x in (kp,kq,kr,ks)]
    kp,kq,kr,ks = [numpy.array(x) for x in (kp,kq,kr,ks)]
    indices = numpy.array(pyscf.lib.cartesian_prod((kp,kq,kr)))

    not_tril = numpy.array([x[0]>x[1] for x in indices])
    if sum(not_tril) == 0: # check if lower triangular
        indices = [tril_index(indices[:,0],indices[:,1]),indices[:,2]]
        tmp = in_array[indices].copy()
        if nints == 3:
            tmp = tmp.reshape(in_array.shape[2:7])
        return tmp
    # If not lower triangular, permute the k-point indices
    indices[not_tril] = indices[not_tril][:,[1,0,2]]
    indices[:,2][not_tril] = ks[not_tril]

    indices = [tril_index(indices[:,0],indices[:,1]),indices[:,2]]

    tmp = in_array[indices].copy()
    tmp[not_tril] = tmp[not_tril].transpose(0,2,1,4,3)
    if nints == 3:
        tmp = tmp.reshape(in_array.shape[2:7])
    return tmp
