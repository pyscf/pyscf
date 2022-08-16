#!/usr/bin/env python

import unittest
import ctypes
import numpy as np
from pyscf.dft.numint import libdft
from pyscf import lib

BLKSIZE = 56

class KnownValues(unittest.TestCase):
    def test_dot_ao_dm_sparse_case1(self):
        np.random.seed(1)
        ngrids = BLKSIZE*128 + 8
        dims = np.array([1,3,1,1,1,3,3,6,6,5,7,1,3,1,3,1,1,3,1,1,1,1,3,3,1,1,3,3]*3)
        ao_loc = np.append(0, np.cumsum(dims)).astype(np.int32)
        nbas = len(dims)
        nao = ao_loc[-1]
        ao = np.exp(-12 - 20 * np.random.rand(nao, ngrids)).T
        vm = np.ones((nao, ngrids)).T
        dm = np.random.rand(nao, nao)
        nrow = (ngrids+BLKSIZE-1)//BLKSIZE
        mask = np.asarray(np.random.rand(nrow, nbas) < .2).astype(np.uint8)
        mask[-40:,:75] = 0
        mask[-80:,:40] = 0
        mask[:40 ,10:] = 0
        mask[:80 ,40:] = 0
        nbins = 20
        s_index = _make_screen_index(ao, ao_loc, mask, nbins)
        pair_mask = np.ones((nbas, nbas), dtype=np.uint8)
        ao = _fill_zero_blocks(ao, ao_loc, mask)
        ref = ao.dot(dm)
        ref = _fill_zero_blocks(ref, ao_loc, mask)

        libdft.VXCdot_ao_dm_sparse(
            vm.ctypes.data_as(ctypes.c_void_p),
            ao.ctypes.data_as(ctypes.c_void_p),
            dm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(ngrids), ctypes.c_int(nbas),
            ctypes.c_int(nbins), s_index.ctypes.data_as(ctypes.c_void_p),
            pair_mask.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p))
        vm = _fill_zero_blocks(vm, ao_loc, mask)
        self.assertAlmostEqual(abs(ref*ao - vm*ao).max(), 0, 24)

    def test_dot_ao_dm_sparse_case2(self):
        np.random.seed(1)
        ngrids = BLKSIZE*10 + 8
        dims = np.array([1,3,1,1,1,3,3,6,6,5,7,1,3,1,3,1,1,3,1,1,1,1,3,3,1,1,3,3]*8)
        ao_loc = np.append(0, np.cumsum(dims)).astype(np.int32)
        nbas = len(dims)
        nao = ao_loc[-1]
        ao = np.exp(-12 - 20 * np.random.rand(nao, ngrids)).T
        vm = np.ones((nao, ngrids)).T
        dm = np.random.rand(nao, nao)
        nrow = (ngrids+BLKSIZE-1)//BLKSIZE
        mask = np.asarray(np.random.rand(nrow, nbas) < .2).astype(np.uint8)
        mask[:2,nbas//4:] = 0
        mask[:4,nbas//3:] = 0
        mask[:6,nbas//2:] = 0
        mask[4:,:nbas//4] = 0
        mask[8:,:nbas//2] = 0
        nbins = 20
        s_index = _make_screen_index(ao, ao_loc, mask, nbins)
        ao = _fill_zero_blocks(ao, ao_loc, mask)
        ref = ao.dot(dm)
        ref = _fill_zero_blocks(ref, ao_loc, mask)
        pair_mask = np.ones((nbas, nbas), dtype=np.uint8)

        libdft.VXCdot_ao_dm_sparse(
            vm.ctypes.data_as(ctypes.c_void_p),
            ao.ctypes.data_as(ctypes.c_void_p),
            dm.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(ngrids), ctypes.c_int(nbas),
            ctypes.c_int(nbins), s_index.ctypes.data_as(ctypes.c_void_p),
            pair_mask.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p))
        vm = _fill_zero_blocks(vm, ao_loc, mask)
        self.assertAlmostEqual(abs(ref*ao - vm*ao).max(), 0, 24)

    def test_contract_rho_case1(self):
        np.random.seed(1)
        ngrids = BLKSIZE*128 + 8
        dims = np.array([1,3,1,1,1,3,3,6,6,5,7,1,3,1,3,1,1,3,1,1,1,1,3,3,1,1,3,3])
        ao_loc = np.append(0, np.cumsum(dims)).astype(np.int32)
        nbas = len(dims)
        nao = ao_loc[-1]
        nrow = (ngrids+BLKSIZE-1)//BLKSIZE
        bra = np.exp(-12 - 20 * np.random.rand(nao, ngrids)).T
        mask = np.asarray(np.random.rand(nrow, nbas) < .1).astype(np.uint8)
        nbins = 20
        s_index = _make_screen_index(bra, ao_loc, mask, nbins)
        bra = _fill_zero_blocks(bra, ao_loc, mask)
        ket = bra
        ref = np.einsum('gi,gi->g', bra, ket)

        rho = np.ones(ngrids)
        libdft.VXCdcontract_rho_sparse(
            rho.ctypes.data_as(ctypes.c_void_p),
            bra.ctypes.data_as(ctypes.c_void_p),
            ket.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(ngrids), ctypes.c_int(nbas),
            s_index.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p))
        self.assertAlmostEqual(abs(ref - rho).max(), 0, 24)

    def test_contract_rho_case2(self):
        np.random.seed(1)
        ngrids = BLKSIZE*10 + 8
        dims = np.array([1,3,1,1,1,3,3,6,6,5,7,1,3,1,3,1,1,3,1,1,1,1,3,3,1,1,3,3]*8)
        ao_loc = np.append(0, np.cumsum(dims)).astype(np.int32)
        nbas = len(dims)
        nao = ao_loc[-1]
        nrow = (ngrids+BLKSIZE-1)//BLKSIZE
        bra = np.exp(-12 - 20 * np.random.rand(nao, ngrids)).T
        mask = np.asarray(np.random.rand(nrow, nbas) < .3).astype(np.uint8)
        nbins = 20
        s_index = _make_screen_index(bra, ao_loc, mask, nbins)
        bra = _fill_zero_blocks(bra, ao_loc, mask)
        ket = bra
        ref = np.einsum('gi,gi->g', bra, ket)

        rho = np.ones(ngrids)
        libdft.VXCdcontract_rho_sparse(
            rho.ctypes.data_as(ctypes.c_void_p),
            bra.ctypes.data_as(ctypes.c_void_p),
            ket.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(ngrids), ctypes.c_int(nbas),
            s_index.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p))
        self.assertAlmostEqual(abs(ref - rho).max(), 0, 24)

    def test_dot_ao_ao_case1(self):
        np.random.seed(1)
        ngrids = BLKSIZE*128 + 75
        dims = np.array([1,3,1,1,1,3,3,6,6,5,7,1,3,1,3,1,1,3,1,1,1,1,3,3,1,1,3,3])
        ao_loc = np.append(0, np.cumsum(dims)).astype(np.int32)
        nbas = len(dims)
        nao = ao_loc[-1]
        nrow = (ngrids+BLKSIZE-1)//BLKSIZE
        bra = np.exp(-12 - 20 * np.random.rand(nao, ngrids)).T
        wv = np.random.rand(ngrids)
        mask = np.asarray(np.random.rand(nrow, nbas) < .2).astype(np.uint8)
        mask[-40:,:75] = 0
        mask[-80:,:40] = 0
        mask[:40 ,10:] = 0
        mask[:80 ,40:] = 0
        nbins = 20
        s_index = _make_screen_index(bra, ao_loc, mask, nbins)
        pair_mask = np.ones((nbas, nbas), dtype=np.uint8)
        bra = _fill_zero_blocks(bra, ao_loc, mask)
        ket = bra
        ref = bra.T.dot(ket)

        out = np.zeros(ref.shape)
        libdft.VXCdot_ao_ao_sparse(
            out.ctypes.data_as(ctypes.c_void_p),
            bra.ctypes.data_as(ctypes.c_void_p),
            ket.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(ngrids),
            ctypes.c_int(nbas), ctypes.c_int(0),
            ctypes.c_int(nbins), s_index.ctypes.data_as(ctypes.c_void_p),
            pair_mask.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p))
        self.assertAlmostEqual(abs(ref - out).max(), 0, 24)

    def test_dot_ao_ao_case2(self):
        np.random.seed(1)
        ngrids = BLKSIZE*10 + 8
        dims = np.array([1,3,1,1,1,3,3,6,6,5,7,1,3,1,3,1,1,3,1,1,1,1,3,3,1,1,3,3]*8)
        ao_loc = np.append(0, np.cumsum(dims)).astype(np.int32)
        nbas = len(dims)
        nao = ao_loc[-1]
        nrow = (ngrids+BLKSIZE-1)//BLKSIZE
        bra = np.exp(-5 - 20 * np.random.rand(nao, ngrids)).T
        wv = np.random.rand(ngrids)
        mask = np.asarray(np.random.rand(nrow, nbas) < .5).astype(np.uint8)
        nbins = 20
        s_index = _make_screen_index(bra, ao_loc, mask, nbins)
        pair_mask = np.ones((nbas, nbas), dtype=np.uint8)
        bra = _fill_zero_blocks(bra, ao_loc, mask)
        ket = bra
        ref = bra.T.dot(ket)

        out = np.zeros(ref.shape)
        libdft.VXCdot_ao_ao_sparse(
            out.ctypes.data_as(ctypes.c_void_p),
            bra.ctypes.data_as(ctypes.c_void_p),
            ket.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(ngrids),
            ctypes.c_int(nbas), ctypes.c_int(0),
            ctypes.c_int(nbins), s_index.ctypes.data_as(ctypes.c_void_p),
            pair_mask.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p))
        self.assertAlmostEqual(abs(ref - out).max(), 0, 15)

    def test_dot_aow_ao_sparse(self):
        np.random.seed(1)
        ngrids = BLKSIZE*128 + 75
        dims = np.array([1,3,1,1,1,3,3,6,6,5,7,1,3,1,3,1,1,3,1,1,1,1,3,3,1,1,3,3])
        ao_loc = np.append(0, np.cumsum(dims)).astype(np.int32)
        nbas = len(dims)
        nao = ao_loc[-1]
        nrow = (ngrids+BLKSIZE-1)//BLKSIZE
        bra = np.exp(-12 - 20 * np.random.rand(nao, ngrids)).T
        wv = np.random.rand(ngrids)
        mask = np.asarray(np.random.rand(nrow, nbas) < .2).astype(np.uint8)
        mask[-40:,:75] = 0
        mask[-80:,:40] = 0
        mask[:40 ,10:] = 0
        mask[:80 ,40:] = 0
        nbins = 20
        s_index = _make_screen_index(bra, ao_loc, mask, nbins)
        pair_mask = np.ones((nbas, nbas), dtype=np.uint8)
        bra = _fill_zero_blocks(bra, ao_loc, mask)
        ket = bra
        ref = (bra.T*wv).dot(ket)

        out = np.zeros(ref.shape)
        libdft.VXCdot_aow_ao_sparse(
            out.ctypes.data_as(ctypes.c_void_p),
            bra.ctypes.data_as(ctypes.c_void_p),
            ket.ctypes.data_as(ctypes.c_void_p),
            wv.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(ngrids),
            ctypes.c_int(nbas), ctypes.c_int(0),
            ctypes.c_int(nbins), s_index.ctypes.data_as(ctypes.c_void_p),
            pair_mask.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p))
        self.assertAlmostEqual(abs(ref - out).max(), 0, 24)

    def test_dot_aow_ao_dense(self):
        np.random.seed(1)
        ngrids = BLKSIZE*10 + 8
        dims = np.array([1,3,1,1,1,3,3,6,6,5,7,1,3,1,3,1,1,3,1,1,1,1,3,3,1,1,3,3]*8)
        ao_loc = np.append(0, np.cumsum(dims)).astype(np.int32)
        nbas = len(dims)
        nao = ao_loc[-1]
        nrow = (ngrids+BLKSIZE-1)//BLKSIZE
        bra = np.exp(-5 - 20 * np.random.rand(nao, ngrids)).T
        wv = np.random.rand(ngrids)
        mask = np.asarray(np.random.rand(nrow, nbas) < .5).astype(np.uint8)
        nbins = 20
        s_index = _make_screen_index(bra, ao_loc, mask, nbins)
        bra = _fill_zero_blocks(bra, ao_loc, mask)
        ket = bra
        ref = (bra.T*wv).dot(ket)

        out = np.zeros(ref.shape)
        libdft.VXCdot_aow_ao_dense(
            out.ctypes.data_as(ctypes.c_void_p),
            bra.ctypes.data_as(ctypes.c_void_p),
            ket.ctypes.data_as(ctypes.c_void_p),
            wv.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao), ctypes.c_int(ngrids))
        self.assertAlmostEqual(abs(ref - out).max(), 0, 15)

    def test_scale_ao(self):
        np.random.seed(1)
        ngrids = BLKSIZE*24 + 75
        dims = np.array([1,3,1,1,1,3,3,6,6,5,7,1,3,1,3,1,1,3,1,1,1,1,3,3,1,1,3,3])
        ao_loc = np.append(0, np.cumsum(dims)).astype(np.int32)
        nbas = len(dims)
        nao = ao_loc[-1]
        nrow = (ngrids+BLKSIZE-1)//BLKSIZE
        ao = np.exp(-12 - 20 * np.random.rand(4, nao, ngrids)).transpose(0,2,1)
        wv = np.random.rand(4, ngrids)
        mask = np.asarray(np.random.rand(nrow, nbas) < .2).astype(np.uint8)
        nbins = 20
        s_index = _make_screen_index(ao[0], ao_loc, mask, nbins)
        for i in range(4):
            _fill_zero_blocks(ao[i], ao_loc, mask)
        ref = np.einsum('xgi,xg->gi', ao, wv)

        out = np.ones((nao, ngrids)).T
        libdft.VXCdscale_ao_sparse(
            out.ctypes.data_as(ctypes.c_void_p),
            ao.ctypes.data_as(ctypes.c_void_p),
            wv.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(4), ctypes.c_int(nao),
            ctypes.c_int(ngrids), ctypes.c_int(nbas),
            s_index.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p))
        out = _fill_zero_blocks(out, ao_loc, mask)
        self.assertAlmostEqual(abs(ref - out).max(), 0, 15)

def _fill_zero_blocks(mat, ao_loc, mask):
    nrow, nbas = mask.shape
    for ib in range(nbas):
        for ig in range(nrow):
            if not mask[ig,ib]:
                mat[ig*BLKSIZE:(ig+1)*BLKSIZE,ao_loc[ib]:ao_loc[ib+1]] = 0
    return mat

def _make_screen_index(ao, ao_loc, mask, nbins=20, cutoff=1e-14):
    assert nbins < 120
    cutoff = min(cutoff, .1)
    scale = -nbins / np.log(cutoff)
    ngrids, nao = ao.shape
    nbas = len(ao_loc) - 1
    nrow = (ngrids + BLKSIZE - 1) // BLKSIZE
    ao_val = np.empty((nrow, nbas))
    for ib in range(nbas):
        i0 = ao_loc[ib]
        i1 = ao_loc[ib+1]
        for ig0, ig1 in lib.prange(0, ngrids, BLKSIZE):
            ao_val[ig0//BLKSIZE,ib] = ao[ig0:ig1,i0:i1].max()
    ao_val[mask==0] = 0
    mask[ao_val < cutoff] = 0

    ao_val[ao_val==0] = 1e-300
    log_ao = nbins + np.log(ao_val) * scale
    log_ao[log_ao < 0] = 0
    sindex = np.ceil(log_ao).astype(np.int8)
    return np.asarray(sindex, order='C')

if __name__ == "__main__":
    unittest.main()
