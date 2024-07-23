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

############ sys module ############

import copy
import numpy as np
import ctypes

############ pyscf module ############

from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf.gto.mole import *
from pyscf.pbc.df.isdf.isdf_jk import _benchmark_time
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0, _format_dms, _format_kpts_band, _format_jks
libpbc = lib.load_library('libpbc')

from  pyscf.pbc.df.isdf.isdf_tools_densitymatrix import pack_JK, pack_JK_in_FFT_space
from  pyscf.pbc.df.isdf.isdf_tools_local import aoR_Holder

############ subroutines ############

######### kernel functions #########

fn_dcwisemul_dense_sparse_kernel = getattr(libpbc, "dcwisemul_dense_sparse_kernel", None)
assert fn_dcwisemul_dense_sparse_kernel is not None

### kernel func always return dense matrices ###

class _ewise_mul_calculator:
    
    def __init__(self, l_op, r_op, out=None,
                 row_begin_id=None,
                 row_end_id  =None,
                 col_begin_id=None,
                 col_end_id  =None):
        
        self._l   = l_op
        self._r   = r_op
        self._out = out
        
        self._row_begin_id = row_begin_id
        self._row_end_id   = row_end_id
        self._col_begin_id = col_begin_id
        self._col_end_id   = col_end_id
        
        self._calculated = False
        self._built      = False
        
        ### type: 0: dense x dense ; 1: dense x sparse ; 2: sparse x sparse (wrong!) ###
        
        self._type = None
        
    ### subroutine to build, canonicalize the operation ###
    
    def _canonicalize_l_r_operand(self):
        
        if isinstance(self._l, np.ndarray) and isinstance(self._r, np.ndarray):
            self._type = 0
        else:
            if isinstance(self._r, np.ndarray):
                self._l, self._r = self._r, self._l

            if isinstance(self._l, np.ndarray):
                self._type = 1
            else:
                self._type = 2
        
        if self._type != 1:
            raise NotImplementedError("sparse x sparse or dense x dense is not supported")
        
    def _build_id(self):
        
        assert self._type == 1
        
        if self._row_begin_id is None:
            self._row_begin_id = 0
            self._row_end_id   = self._l.shape[0]
        
        if self._col_begin_id is None:
            self._col_begin_id = self._r.global_gridID_begin
            self._col_end_id   = self._r.global_gridID_end
    
    def _build_l_property(self):
        
        assert self._type == 1
        
        ### determine the shift of the row indices ###
        
        self._l_stride = self._l.shape[1]
        
        if self._l.shape[0] == self._row_end_id - self._row_begin_id:
            self._l_row_shift = 0
        else:
            self._l_row_shift = self._row_begin_id
        
        ### determine the shift of the col indices ###
        
        if self._l.shape[1] == self._col_end_id - self._col_begin_id:
            self._l_col_shift = 0
        else:
            self._l_col_shift = self._col_begin_id
    
    def _build_out(self):
        
        if self._out is None:
            self._out = np.zeros((self._row_end_id - self._row_begin_id, self._col_end_id - self._col_begin_id))
        else:
            assert self._out.shape == (self._row_end_id - self._row_begin_id, self._col_end_id - self._col_begin_id)
    
    def build(self):
        
        self._canonicalize_l_r_operand()
        self._build_id()
        self._build_l_property()
        self._build_out()
        
        self._built = True
    
    ### perform the calculation ###
    
    def calculate(self):
        
        fn_dcwisemul_dense_sparse_kernel(
            self._out.data.ctypes.data_as(ctypes.c_void_p),
            self._l.data.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self._l.shape[0]),
            ctypes.c_int(self._l.shape[1]),
            ctypes.c_int(self._l_row_shift),
            ctypes.c_int(self._l_col_shift),
            self._r.aoR.data.ctypes.data_as(ctypes.c_void_p),
            self._r.ao_involved.data.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self._r.aoR.shape[0]),
            ctypes.c_int(self._r.aoR.shape[1]),
            ctypes.c_int(self._row_begin_id),
            ctypes.c_int(self._row_end_id),
            ctypes.c_int(self._col_begin_id),
            ctypes.c_int(self._col_end_id)
        )
        
        self._calculated = True
    
    @property
    def result(self):
        if not self._built:
            self.build()
        if not self._calculated:
            self.calculate()
        return self._out


def _ewise_mul(a, b, 
               c=None,
               row_begin_id=None,
               row_end_id  =None,
               col_begin_id=None,
               col_end_id=None):

    calculator = _ewise_mul_calculator(a, b, c,
                                       row_begin_id, row_end_id,
                                       col_begin_id, col_end_id)

    return calculator.result

class _matmul_calculator:
    
    def __init__(self, 
                 l_op   = None, 
                 tran_l = 'N', 
                 r_op   = None, 
                 tran_r = 'N',
                 _l_row_begin_id=None,
                 _l_row_end_id  =None,
                 _l_col_begin_id=None,
                 _l_col_end_id  =None,
                 _r_row_begin_id=None,
                 _r_row_end_id  =None,
                 _r_col_begin_id=None,
                 _r_col_end_id  =None,
                 buf=None,
                 out=None):
        
        self._l   = l_op
        self._r   = r_op
        self._out = out
        
        self._tran_l = tran_l
        self._tran_r = tran_r
        
        if self._tran_l is None:
            self._tran_l = 'N'
        if self._tran_r is None:
            self._tran_r = 'N'
        
        self._l_row_begin_id = _l_row_begin_id
        self._l_row_end_id   = _l_row_end_id
        self._l_col_begin_id = _l_col_begin_id
        self._l_col_end_id   = _l_col_end_id
        self._r_row_begin_id = _r_row_begin_id
        self._r_row_end_id   = _r_row_end_id
        self._r_col_begin_id = _r_col_begin_id
        self._r_col_end_id   = _r_col_end_id
        
        self._calculated = False
        self._built      = False
        
        ### type: 0: dense x dense ; 1: dense x sparse ; 2: sparse x dense; 3: sparse x sparse (wrong!) ###
        
        self._type = None
        self._buf = buf

    ###### subroutine to build, canonicalize the operation ######
    
    def _canonicalize_l_r_operand(self):
        
        if isinstance(self._l, np.ndarray) and isinstance(self._r, np.ndarray):
            self._type = 0
        else:
            if isinstance(self._l, np.ndarray):
                self._type = 1
            elif isinstance(self._r, np.ndarray):
                self._type = 2
            else:
                self._type = 3
    
    def _build_id(self):
        
        ############# consider match #############
        
        if self._tran_l == 'N':
            if self._tran_r == 'N':
                if self._l_col_begin_id is None:
                    if self._r_row_begin_id is not None:
                        self._l_col_begin_id = self._r_row_begin_id
                else:
                    if self._r_row_begin_id is None:
                        self._r_row_begin_id = self._l_col_begin_id
                    else:
                        assert self._l_col_begin_id == self._r_row_begin_id
            else:
                if self._l_col_begin_id is None:
                    if self._r_col_begin_id is not None:
                        self._l_col_begin_id = self._r_col_begin_id
                else:
                    if self._r_col_begin_id is None:
                        self._r_col_begin_id = self._l_col_begin_id
                    else:
                        assert self._l_col_begin_id == self._r_col_begin_id
        else:
            if self._tran_r == 'N':
                if self._l_row_begin_id is None:
                    if self._r_row_begin_id is not None:
                        self._l_row_begin_id = self._r_row_begin_id
                else:
                    if self._r_row_begin_id is None:
                        self._r_row_begin_id = self._l_row_begin_id
                    else:
                        assert self._l_row_begin_id == self._r_row_begin_id
            else:
                if self._l_row_begin_id is None:
                    if self._r_col_begin_id is not None:
                        self._l_row_begin_id = self._r_col_begin_id
                else:
                    if self._r_col_begin_id is None:
                        self._r_col_begin_id = self._l_row_begin_id
                    else:
                        assert self._l_row_begin_id == self._r_col_begin_id
        
        ######## l_op ########
        
        if isinstance(self._l, aoR_Holder):
            if self._l_row_begin_id is None:
                self._l_row_begin_id = np.min(self._l.ao_involved)
            if self._l_row_end_id is None:
                self._l_row_end_id   = np.max(self._l.ao_involved) + 1
            if self._l_col_begin_id is None:
                self._l_col_begin_id = self._l.global_gridID_begin
            if self._l_col_end_id is None:
                self._l_col_end_id   = self._l.global_gridID_end
        
        ######## r_op ########
        
        if isinstance(self._r, aoR_Holder):
            if self._r_row_begin_id is None:
                self._r_row_begin_id = np.min(self._r.ao_involved)
            if self._r_row_end_id is None:
                self._r_row_end_id   = np.max(self._r.ao_involved) + 1
            if self._r_col_begin_id is None:
                self._r_col_begin_id = self._r.global_gridID_begin
            if self._r_col_end_id is None:
                self._r_col_end_id   = self._r.global_gridID_end
    
    def _build_out(self):
        
        if self._tran_l == 'N':
            res_nrow = self._l_row_end_id - self._l_row_begin_id
        else:
            res_nrow = self._l_col_end_id - self._l_col_begin_id
        
        if self._tran_r == 'N':
            res_ncol = self._r_col_end_id - self._r_col_begin_id
        else:
            res_ncol = self._r_row_end_id - self._r_row_begin_id    
        
        if self._out is None:
            self._out = np.zeros((res_nrow, res_ncol))
        else:
            assert self._out.shape == (res_nrow, res_ncol)
        
    def build(self):
        
        self._canonicalize_l_r_operand()
        self._build_id()
        self._build_out()
        
        self._built = True
    
    ###### calculation ######

    def _calculate_dd(self):
        
        if self._tran_l == 'N':
            if self._tran_r == 'N':
                lib.ddot(self._l, self._r, c=self._out)
            else:
                lib.ddot(self._l, self._r.T, c=self._out)
        else:
            if self._tran_r == 'N':
                lib.ddot(self._l.T, self._r, c=self._out)
            else:
                lib.ddot(self._l.T, self._r.T, c=self._out)
    
    def _calculate_ds(self):
        pass

    def _calculate_sd(self):
        pass
    
    def _calculate_ss(self):
        pass
    
    def calculate(self):
        
        if self._type == 0:
            self._calculate_dd()
        elif self._type == 1:
            self._calculate_ds()
        elif self._type == 2:
            self._calculate_sd()
        elif self._type == 3:
            self._calculate_ss()
        
        self._calculated = True
    
    @property
    def result(self):
        if not self._built:
            self.build()
        if not self._calculated:
            self.calculate()
        return self._out

def _matmul(a, b, 
            c=None,
            l_tran='N',
            r_tran='N',
            _l_row_begin_id=None,
            _l_row_end_id  =None,
            _l_col_begin_id=None,
            _l_col_end_id  =None,
            _r_row_begin_id=None,
            _r_row_end_id  =None,
            _r_col_begin_id=None,
            _r_col_end_id  =None,
            buf=None):
    
    calculator = _matmul_calculator(a, l_tran, b, r_tran,
                                    _l_row_begin_id, _l_row_end_id,
                                    _l_col_begin_id, _l_col_end_id,
                                    _r_row_begin_id, _r_row_end_id,
                                    _r_col_begin_id, _r_col_end_id,
                                    buf, c)
    
    return calculator.result

### matrix product or dcwise_mul with translation symmetry ###

### perform get_JK in real space! ###



if __name__ == "__main__":
    
    pass

    ## test kernel ##