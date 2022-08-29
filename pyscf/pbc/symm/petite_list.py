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
# Author: Xing Zhang <zhangxing.nju@gmail.com>
#

import numpy as np
from numpy.linalg import inv
import time
from pyscf import __config__
from pyscf import lib
from pyscf.pbc.symm import symmetry

LS_DIFF_TOL = getattr(__config__, 'pbc_symm_petite_list_ls_diff_tol', 1e-7)

def _compute_gv(cell, op, g, shl_cen1, shl_cen2):
    g_a = coord_rtoa(cell, g)
    op_dot_r = np.dot(inv(op), shl_cen1)
    op_dot_r_0 = np.mod(np.mod(op_dot_r, 1), 1)
    os1 = op_dot_r_0 - op_dot_r 
    #os1 = shl_cen1 - np.dot(inv(op), shl_cen1)
    op_dot_r = np.dot(inv(op), shl_cen2)                                        
    op_dot_r_0 = np.mod(np.mod(op_dot_r, 1), 1)
    os2 = op_dot_r_0 - op_dot_r
    #os2 = shl_cen2 - np.dot(inv(op), shl_cen2)
    gv_a = np.dot(inv(op), g_a) - os2 + os1
    gv = coord_ator(cell, gv_a)
    return gv

def get_Ls_2c(cell, Ls, op, shl_cen1, shl_cen2):
    gv = [_compute_gv(cell, op, g, shl_cen1, shl_cen2) for i, g in enumerate(Ls)]
    return np.asarray(gv)

def get_shl_centers(cell, nc):
    coords = cell.atom_coords()
    uniq_shlpr_cen_idx = get_shl_center_idx(cell, nc)
    shlpr_cen = []
    for i, idx in enumerate(uniq_shlpr_cen_idx):
        shlpr_cen.append([coords[idx[c]] for c in range(nc)])
    return shlpr_cen

def get_shl_center_idx(cell, nc):
    atm_idx = [cell._bas[i,0] for i in range(cell.nbas)]
    shlpr_cen_idx = lib.cartesian_prod([atm_idx,]*nc)
    uniq_shlpr_cen_idx = np.unique(shlpr_cen_idx, axis=0)
    return uniq_shlpr_cen_idx

def coord_rtoa(cell, coord):
    b = cell.reciprocal_vectors()
    coord_a = np.dot(coord, b.T)/(2.*np.pi)
    return coord_a

def coord_ator(cell, coord_a):
    return np.dot(coord_a, cell.lattice_vectors())

def Ls_rtoa(cell,Ls, tol=LS_DIFF_TOL):
    b = cell.reciprocal_vectors()
    Ls_a = np.dot(Ls, b.T)/(2.*np.pi)
    assert((np.absolute(Ls_a - Ls_a.round())<tol).all())
    return Ls_a.round().astype(int)

def get_equivalent_pairs(g):
    order = np.lexsort(g.T)
    g = g[order]
    diff_g = np.diff(g, axis=0)
    eq_pairs = np.array((diff_g == 0).all(1), dtype=bool)
    return eq_pairs,order

def map_Ls_2c(cell, g, ops, shl_cen1, shl_cen2, tol=LS_DIFF_TOL, nthreads=lib.num_threads()):
    '''
    Find symmetry relations between g and gv.
    '''
    lib.num_threads(nthreads)
    b = cell.reciprocal_vectors()
    nL = len(g)
    L2L = -np.ones([nL, len(ops)], dtype=np.int32)

    shl_cen1_a = coord_rtoa(cell, shl_cen1)                                     
    shl_cen2_a = coord_rtoa(cell, shl_cen2)
    for iop, op in enumerate(ops):
        gv = get_Ls_2c(cell,g, op, shl_cen1_a, shl_cen2_a)
        g_gv = np.concatenate([g, gv])
        g_gv = Ls_rtoa(cell, g_gv, tol)

        # Find the lexicographical order
        equivalentpairs_g, order = get_equivalent_pairs(g_gv)

        # Mapping array.
        orders = np.array([order[:-1][equivalentpairs_g],
                           order[1:][equivalentpairs_g]])

        # This has to be true.
        assert (orders[0] < nL).all()
        assert (orders[1] >= nL).all()
        L2L[orders[1] - nL, iop] = orders[0]
    return L2L

def get_t2(petite, cell=None, ops=None, Ls=None, L2L_Ls=None, tol=LS_DIFF_TOL):
    if cell is None: cell = petite.cell
    if ops is None: ops = petite.ops
    if Ls is None: Ls = petite.Ls
    if L2L_Ls is None: L2L_Ls = petite.L2L_Ls

    shlpr_cen = get_shl_centers(cell, 2)
    nL = len(Ls)
    nop = len(ops)
    t2_iL2L = []
    t2_L2iL = []
    t2_iLs  = []
    t2_sym_group = []
    t2_L_group = []

    t1 = (time.clock(), time.time())
    for i, shlpr in enumerate(shlpr_cen):
        _L2L = L2L_Ls[i]
        L2L = -np.ones(nL+1, dtype = np.int32)
        iL2L = []
        L_group = []
        sym_group = []
        for k in range(nL-1, -1, -1):
            if L2L[k] == -1:
                L2L[_L2L[k]] = k
                iL2L.append(k)
                L_idx, op_idx = np.unique(_L2L[k], return_index=True)
                if L_idx[0] == -1:
                    L_idx = L_idx[1:]
                    op_idx = op_idx[1:]
                L_group.append(L_idx)
                sym_group.append(op_idx)

        iL2L = np.array(iL2L[::-1])
        t2_iL2L.append(iL2L)
        t2_L_group.append(L_group[::-1])
        t2_sym_group.append(sym_group[::-1])
        
        L2L = L2L[:-1].copy()
        L2iL = np.empty(nL, dtype = np.int32)
        L2iL[iL2L] = np.arange(len(iL2L))
        L2iL = L2iL[L2L]
        t2_L2iL.append(L2iL)

        iLs = Ls[iL2L]
        t2_iLs.append(iLs)

        '''
        sym_group = []
        L_group = []
        for i, iL in enumerate(iLs):
            sym_group.append([])
            idx = np.where(L2iL == i)[0]
            L_group.append(idx)
            for j in range(idx.size):
                #L = Ls[idx[j]]
                L_idx = idx[j]
                L_star_idx = idx[-1]
                for io, op in enumerate(ops):
                    #if -1 in _L2L[:,io]: continue
                    #op_iL = _compute_gv(op, iL, shlpr[0], shlpr[1])
                    #diff = L - op_iL
                    #if (np.absolute(diff) < tol).all():
                    if _L2L[L_star_idx,io] == L_idx:
                        sym_group[i].append(io)
                        break
            sym_group[i] = np.asarray(sym_group[i])
        t2_sym_group.append(sym_group)
        t2_L_group.append(L_group)
        '''
    t1 = lib.logger.timer_debug1(petite, 'get_t2:', *t1)
    return t2_iLs, t2_iL2L, t2_L2iL, t2_L_group, t2_sym_group

def get_shlpr_idx(idx_table, shlpr_idx):
    res = (idx_table == tuple(shlpr_idx)).all(axis=1).nonzero()[0]
    assert(res.size == 1)
    return res[0]

def get_t3(petite, cell=None, ops=None, Ls=None, L2L_Ls=None, buf=None, tol=LS_DIFF_TOL):
    if cell is None: cell = petite.cell
    if ops is None: ops = petite.ops
    if Ls is None: Ls = petite.Ls
    if L2L_Ls is None: L2L_Ls = petite.L2L_Ls
    if buf is None: buf = petite.buf

    shlpr_cen_idx = get_shl_center_idx(cell, 2)
    shltrip_cen_idx = get_shl_center_idx(cell, 3)
    nshltrip = len(shltrip_cen_idx)
    nL = len(Ls)
    nL2 = nL * nL
    nop = len(ops)
    '''
    t3_iL2L = []
    t3_L2iL = []
    t3_iLs_idx  = []
    t3_sym_group = []
    t3_L_group = []
    '''

    t1 = (time.clock(), time.time())
#    for i, shltrip in enumerate(shltrip_cen_idx):
    def _get_iL(idx,shltrip,buf,nthreads):
        lib.num_threads(nthreads)
        t2 = (time.clock(), time.time())
        idx_i = get_shlpr_idx(shlpr_cen_idx, (shltrip[0],shltrip[1]))
        idx_j = get_shlpr_idx(shlpr_cen_idx, (shltrip[0],shltrip[2]))

        L2L_LsLs_T = np.empty([nop, nL2], dtype=np.int32)
        for iop, op in enumerate(ops):
            tmp = lib.cartesian_prod((L2L_Ls[idx_i,:,iop], L2L_Ls[idx_j,:,iop]))
            idx_throw = np.unique(np.where(tmp == -1)[0])
            L2L_LsLs_T[iop] = tmp[:,0] * nL + tmp[:,1]
            L2L_LsLs_T[iop, idx_throw] = -1

        _L2L = L2L_LsLs_T.T
        L2L = -np.ones(nL2+1, dtype=np.int32)
        iL2L = []
        _L_group = []
        _sym_group = []
        _group_size = []
        for k in range(nL2-1, -1, -1):
            if L2L[k] == -1:
                L2L[_L2L[k]] = k
                iL2L.append(k)
                L_idx, op_idx = np.unique(_L2L[k], return_index=True)
                if L_idx[0] == -1:
                    L_idx = L_idx[1:]
                    op_idx = op_idx[1:]
                _group_size.append(op_idx.size)
                _L_group.append(L_idx)
                _sym_group.append(op_idx)

        iL2L = np.array(iL2L[::-1])
        lib.logger.debug(petite, 'niL = %s', len(iL2L))
        group_size = np.concatenate(_group_size[::-1], axis=None)
        L_group = np.concatenate(_L_group[::-1], axis=None)
        sym_group = np.concatenate(_sym_group[::-1], axis=None)

        L2L = L2L[:-1].copy()
        L2iL = np.empty(nL2, dtype = np.int32)
        L2iL[iL2L] = np.arange(len(iL2L))
        L2iL = L2iL[L2L]

        #idx_i = (iL2L[L2iL] // nL).reshape((-1,1))
        #idx_j = (iL2L[L2iL] % nL).reshape((-1,1))
        Lop = np.empty(nL2, dtype = np.int32)
        Lop[L_group] = sym_group
        res = np.hstack((iL2L[L2iL].reshape(-1,1), Lop.reshape(-1,1)))
        buf[idx] = res

        #idx_i = (iL2L // nL).reshape((-1,1))
        #idx_j = (iL2L % nL).reshape((-1,1))
        #idx_ij = np.hstack((idx_i, idx_j))

        t2 = lib.logger.timer_debug1(petite, '_get_iL:', *t2)
        #return iL2L, L_group, sym_group, L2iL, idx_ij
        return None

    try:
        from joblib import Parallel, delayed
        with lib.with_multiproc_nproc(nshltrip) as mpi:
            Parallel(n_jobs = mpi.nproc)(delayed(_get_iL)(i, shltrip, buf, lib.num_threads()) for i, shltrip in enumerate(shltrip_cen_idx))
    except:
        for i, shltrip in enumerate(shltrip_cen_idx):
            _get_iL(i, shltrip, buf)

    t1 = lib.logger.timer_debug1(petite, 'get_t3:', *t1)
    #return t3_iLs_idx, t3_iL2L, t3_L2iL, t3_L_group, t3_sym_group
    return None

def build_L2L_Ls(petite, cell=None, ops=None, Ls=None, tol=LS_DIFF_TOL):
    if cell is None: cell = petite.cell
    if ops is None: ops = petite.ops_a
    if Ls is None: Ls = petite.Ls

    shlpr_cen = get_shl_centers(cell, 2)
    nshlpr = len(shlpr_cen)
    nL = len(Ls)
    nop = len(ops)

    t1 = (time.clock(), time.time())
    try:
        from joblib import Parallel, delayed
        with lib.with_multiproc_nproc(nshlpr) as mpi:
            res = Parallel(n_jobs = mpi.nproc)(delayed(map_Ls_2c)(cell,Ls,ops,shlpr[0], shlpr[1], tol, lib.num_threads()) for i, shlpr in enumerate(shlpr_cen))
        L2L_Ls = np.asarray(res)
        #for i in range(nshlpr):
        #    L2L_Ls[i] = res[i]
    except:
        L2L_Ls = np.empty([nshlpr,nL,nop], dtype=np.int32)
        for i, shlpr in enumerate(shlpr_cen):
            L2L_Ls[i] = map_Ls_2c(cell, Ls, ops, shlpr[0], shlpr[1], tol, lib.num_threads())
    t1 = lib.logger.timer_debug1(petite, 'build_L2L_Ls:', *t1)
    return L2L_Ls

class Petite_List(lib.StreamObject):
    '''
    Petite list for integral lattice summation
    '''
    def __init__(self, cell, Ls, auxcell=None):
        self.cell = cell
        self.Ls = Ls
        self.pg_symm = symmetry.Symmetry(cell)
        self.pg_symm.build(auxcell=auxcell)
        self.ops = [op.a2r(self.cell).rot for op in self.pg_symm.ops]  #in cartesian coordinate
        self.ops_a = [op.rot for op in self.pg_symm.ops] #in scaled lattice vector coordinate

        self.Dmats = self.pg_symm.Dmats
        self.l_max = self.pg_symm.l_max
        self.verbose = self.cell.verbose
        self.L2L_Ls = None

        self.t2_iLs = None 
        self.t2_iL2L = None 
        self.t2_L2iL = None 
        self.t2_L_group = None 
        self.t2_sym_group = None

        '''
        self.t3_iLs_idx = None
        self.t3_iL2L = None
        self.t3_L2iL = None
        self.t3_L_group = None
        self.t3_sym_group = None
        '''

        self.shltrip_cen_idx = get_shl_center_idx(cell, 3)
        self.nL = len(Ls)
        self.nL2 = self.nL * self.nL
        lib.logger.debug(self, 'nL2 = %s', self.nL2)
        nshltrip = len(self.shltrip_cen_idx)
        self.buf = np.memmap('petite_list', dtype=np.int32, shape=(nshltrip,self.nL2,2), mode='w+')

    build_L2L_Ls = build_L2L_Ls
    get_t2 = get_t2
    get_t3 = get_t3

    def kernel(self):
        self.L2L_Ls = self.build_L2L_Ls()
        #self.t2_iLs, self.t2_iL2L, self.t2_L2iL, self.t2_L_group, self.t2_sym_group = self.get_t2()
        #self.t3_iLs_idx, self.t3_iL2L, self.t3_L2iL, self.t3_L_group, self.t3_sym_group = self.get_t3()
        self.get_t3()

if __name__ == "__main__":
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.atom = """
        Si  0.0 0.0 0.0
        Si  1.3467560987 1.3467560987 1.3467560987
    """
    cell.a = [[0.0, 2.6935121974, 2.6935121974],
              [2.6935121974, 0.0, 2.6935121974],
              [2.6935121974, 2.6935121974, 0.0]]
    cell.verbose = 5
    cell.build()
    Ls = cell.get_lattice_Ls()
    Petite_List(cell, Ls).kernel()
