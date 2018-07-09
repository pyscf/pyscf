! Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
!
! Licensed under the Apache License, Version 2.0 (the "License");
!  you may not use this file except in compliance with the License.
!  You may obtain a copy of the License at
!
!      http://www.apache.org/licenses/LICENSE-2.0
!
!  Unless required by applicable law or agreed to in writing, software
!  distributed under the License is distributed on an "AS IS" BASIS,
!  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!  See the License for the specific language governing permissions and
!  limitations under the License.

module m_prod_basis_param
!
! The purpose of the module is to define auxiliary type to store parameters
! for a product basis generation
!
#include "m_define_macro.F90"
  use m_die, only : die
  use m_warn, only : warn
  use m_param_arr_type, only : param_arr_t
  
  implicit none
    
  type prod_basis_param_t
    type(param_arr_t), pointer :: p => null()
    
    character(100) :: prod_basis_type = ""
    integer :: check_basis            =-999  ! usual test to perform or skip (overlaps and dipoles comparison)
    integer :: check_basis_rspace     =-999  ! usual test to perform or skip (overlaps and dipoles comparison) / moments 
    integer :: check_basis_bloch      =-999  ! recompute k-dependent overlaps out of real-space overlaps.
    integer :: check_basis_write_report =-999! whether to write a file with a report on usual test of dominant products
    integer :: cross_check_global_vertex = -999 ! computes the global vertex coefficients and compares
    integer :: check_zcmmm               = -999 ! check matmuls with the conversion matrix (CM) ac<->dp (CM is stored block-sparse)
    integer :: check_dfmcm_lap           = -999 ! check a list-based matmul with the conversion matrix (CM) 
    integer :: check_zspcm_lap           = -999 ! check a list-based matmul of symmetric,packed matrix with the conversion matrix (CM) 
    integer :: check_exmtrx_vt1_bt2      = -999 ! check an implementation of exchange matrix with atom-centered kernel and dominant product vertex
    
    integer :: stop_check_basis_bloch   =-999! whether to stop after an eventual checking of bloch's overlaps
    integer :: do_shifting_test =-999! additional test to perform or skip 
    integer :: metric_type      =-999! 1-- Cartesian metric, 2 -- Coulomb metric
    integer :: normalize_dp     =-999! whether we want to normalize dominant functions after determining them
    real(8) :: eigmin_local     =-999! eigenvalue threshold to generate local functions
    real(8) :: eigmin_bilocal   =-999! eigenvalue threshold to generate bilocal functions
    integer :: optimize_centers =-999! whether we try to optimize centers of bilocal functions
    integer :: jcutoff          =-999! angular momentum cutoff (desired) for expansion of bilocal dominant products
    integer :: gl_ord_bilocal   =-999! parameters for numerical integration in construction of bilocal dominant products
    integer :: gl_pcs_bilocal   =-999! parameters for numerical integration in construction of bilocal dominant products
    real(8) :: ac_rcut          =-999! radius of a sphere to determine which (atom centered) functions contribute or belong to a given center.
    character(99) :: ac_method  = "" ! Method to choose the contributing centers: LENS or (SPHERE).
    character(99) :: cc_method  = "" ! Method to compute conversion coefficients: (DSYEV), DSYEVR, DSYTR
    character(99) :: cc_omp_para= "" ! Parallelization approach for computation of conversion coefficients PAIRS, ATOMS, (AUTO)
    real(8) :: cc_inv_eigmin = -999  ! Threshold for dropping small eigenvalues in comput. of convers. coeffs.
    integer :: check_hkernel_blocks =-999! whether we will (cross) check blocks of Hartree kernel in mixed basis construction
    integer :: check_hkernel_dp_pb  =-999! generates pb1 = (dp->pb) and pb2, computes kernels and compares the kernels
    integer :: reexpr_further =-999! tells whether the further computation should be done with quantities that are tensors in dp basis set,
                              ! but computed indirectly, by converting from a (smaller) reexpressing basis set.
    integer :: check_hkernel_reexpr =-999 ! test reexpression of dominant products by comparing hkernel_reexpr with hkernel_orig (in dp basis)
    integer :: report_pair2clist  =-999! flag to cause a report on pair2clist structure (can be a lengthy file)
    integer :: report_ls_blocks   =-999! flag to cause a report on ls_blocks structure (can be a lengthy file)
    integer :: bulk_cross_check_hkernel =-999! flag to cause a rather extensive cross check against 
                                        ! a finite-size calculation which includes all super cell
    integer :: report_domiprod       =-999! report main metrics of the initialized type domiprod_t 
    integer :: report_prod_basis_mix =-999! report prod basis with reexpressing coefficients
    integer :: report_prod_basis_dp  =-999! report prod basis when it is still dominant products only
    integer :: report_prod_basis     =-999! report prod basis just before testing
    integer :: gen_aug_lmult         =-999! generate augmented l-multipletts 
    character(100) :: bilocal_type   = "" ! ATOM or MULT
    character(100) :: bilocal_center = "" ! algorithm to determine centers for expansion of bilocal product
    real(8)        :: bilocal_center_pow = -999 ! weight = (coeff * sqrt(<r^2>)**pow)^-1
    character(100) :: bilocal_center_coeff = "" ! 2*L+1, for example    
    character(100) :: local_type     = "" ! ATOM or SHELL
    real(8)        :: core_split_ratio = -999 ! for decision whether an orbital belongs to core
    integer        :: stop_after_dp = -999
    integer        :: pb_funct_xsf = -999 ! whether to plot product functions to xsf files...
    character(100) :: biloc_interm_storage = "" ! type of intermediate storage for generation of bilocal dominant products
    character(100) :: mixed_interm_storage = "" ! type of intermediate storage for generation of reexpression coefficients
  end type ! prod_basis_aux_t
  
  private warn, die
  
contains

!
!
!
subroutine dealloc(v)
  implicit none
  type(prod_basis_param_t), intent(inout) :: v

  v%p=>null()
  v%check_basis = -999
  v%check_basis_rspace = -999
  v%check_basis_bloch = -999
  v%check_basis_write_report = -999
  v%cross_check_global_vertex = -999
  v%check_zcmmm = -999
  v%check_dfmcm_lap = -999
  v%check_zspcm_lap = -999
  v%check_exmtrx_vt1_bt2 = -999
  v%stop_check_basis_bloch = -999
  v%do_shifting_test = -999
  v%metric_type = -999
  v%normalize_dp = -999
  v%eigmin_local = -999
  v%eigmin_bilocal = -999
  v%optimize_centers = -999
  v%jcutoff = -999
  v%gl_ord_bilocal = -999
  v%gl_pcs_bilocal = -999
  v%ac_rcut = -999
  v%cc_inv_eigmin = -999
  v%check_hkernel_blocks = -999
  v%check_hkernel_dp_pb = -999
  v%reexpr_further = -999
  v%check_hkernel_reexpr = -999
  v%report_pair2clist = -999
  v%report_ls_blocks = -999
  v%bulk_cross_check_hkernel = -999
  v%report_domiprod = -999
  v%report_prod_basis_mix = -999
  v%report_prod_basis_dp = -999
  v%report_prod_basis = -999
  v%gen_aug_lmult = -999
  v%bilocal_center_pow = -999
  v%core_split_ratio = -999
  v%stop_after_dp = -999
  v%pb_funct_xsf = -999
    
  v%prod_basis_type = ''
  v%ac_method = ''
  v%cc_method = ''
  v%cc_omp_para = ''
  v%bilocal_type = ''
  v%bilocal_center = ''
  v%bilocal_center_coeff = ''
  v%local_type = ''
  v%biloc_interm_storage = ''
  v%mixed_interm_storage = ''

end subroutine ! dealloc 


!
!
!
subroutine init_prod_basis_param(inp, sv, p, iv, param)
  use m_input, only : input_t, init_parameter
  use m_upper, only : upper
  use m_system_vars, only : system_vars_t, get_jmx, get_basis_type, get_natoms, get_rcut_max
  implicit none
  !! external
  type(input_t), intent(in) :: inp
  type(system_vars_t), intent(in) :: sv
  type(prod_basis_param_t), intent(inout) :: p
  integer, intent(in) :: iv
  type(param_arr_t), intent(in), target, optional :: param
  !! internal
  integer :: jmx, idef
  real(8) :: rcut_prd_def
  
  if(present(param)) then
    p%p=>param
  else  
    p%p => null()
  endif
   
  call init_parameter('prod_basis_type', inp, 'DOMIPROD', p%prod_basis_type, iv)
  p%prod_basis_type = upper(p%prod_basis_type)
  
  call init_parameter('check_basis ', inp, 1, p%check_basis, iv)
  call init_parameter('check_basis_rspace ', inp, 1, p%check_basis_rspace, iv)
  idef=0; if(get_basis_type(sv)==2) idef=1;
  call init_parameter('check_basis_bloch  ', inp, idef, p%check_basis_bloch, iv)
  call init_parameter('check_basis_write_report ', inp, 0, p%check_basis_write_report, iv)
  call init_parameter('cross_check_global_vertex', inp, 0, p%cross_check_global_vertex, iv)
  call init_parameter('check_zcmmm', inp, 0, p%check_zcmmm, iv)
  call init_parameter('check_dfmcm_lap', inp, 0, p%check_dfmcm_lap, iv)
  call init_parameter('check_zspcm_lap', inp, 0, p%check_zspcm_lap, iv)
  call init_parameter('check_exmtrx_vt1_bt2', inp, 0, p%check_exmtrx_vt1_bt2, iv)
  
  call init_parameter('stop_check_basis_bloch ', inp, 0, p%stop_check_basis_bloch, iv)
  call init_parameter('do_shifting_test', inp, 0, p%do_shifting_test, iv)
  call init_parameter('metric_type', inp, 2, p%metric_type, iv)
  call init_parameter('normalize_dp', inp, 0, p%normalize_dp, iv)
  call init_parameter('eigmin_local', inp, 1D-4, p%eigmin_local, iv)
  call init_parameter('eigmin_bilocal', inp, 1D-5, p%eigmin_bilocal, iv)
  jmx = get_jmx(sv)
  call init_parameter('jcutoff', inp, max(5+jmx,2*jmx), p%jcutoff, iv)
  call init_parameter('optimize_centers', inp, 0, p%optimize_centers, iv)
    
  call init_parameter('gl_ord_bilocal', inp, 96, p%gl_ord_bilocal, iv);
  call init_parameter('gl_pcs_bilocal', inp, 1, p%gl_pcs_bilocal, iv);

  rcut_prd_def = get_rcut_max(sv)
  call init_parameter('ac_rcut', inp, rcut_prd_def, p%ac_rcut, iv)
  call init_parameter('ac_method', inp, "SPHERE", p%ac_method, iv)
  p%ac_method = upper(p%ac_method)
  call init_parameter('cc_method', inp, "DSYEV", p%cc_method, iv)
  p%cc_method = upper(p%cc_method)
  call init_parameter('cc_omp_para', inp, 'AUTO', p%cc_omp_para, iv)
  p%cc_omp_para = upper(p%cc_omp_para)
  call init_parameter('cc_inv_eigmin', inp, 1d-9, p%cc_inv_eigmin, iv)
  
  call init_parameter('report_pair2clist', inp, 0, p%report_pair2clist, iv)
  call init_parameter('report_ls_blocks', inp, 0, p%report_ls_blocks, iv)
  call init_parameter('check_hkernel_blocks', inp, 0, p%check_hkernel_blocks, iv);
  call init_parameter('check_hkernel_dp_pb', inp, 0, p%check_hkernel_dp_pb, iv);
  idef = 0; if(upper(p%prod_basis_type)=='MIXED') idef = 1;
  call init_parameter('reexpr_further', inp, 0, p%reexpr_further, iv);
  call init_parameter('check_hkernel_reexpr', inp, 0, p%check_hkernel_reexpr, iv);
  call init_parameter('bulk_cross_check_hkernel', inp, 0, p%bulk_cross_check_hkernel, iv);
  call init_parameter('report_domiprod', inp, 1, p%report_domiprod, iv);
  call init_parameter('report_prod_basis', inp, 1, p%report_prod_basis, iv);
  call init_parameter('report_prod_basis_dp', inp, 0, p%report_prod_basis_dp, iv);
  call init_parameter('report_prod_basis_mix', inp, 0, p%report_prod_basis_mix, iv);
  !! Augment or not augment ?
  idef = 0; 
  if(jmx<2 .and. get_natoms(sv)>1 .and. get_basis_type(sv)==1) idef = 1;
  if(jmx<2 .and. get_basis_type(sv)==2) idef = 1;
  if(upper(p%prod_basis_type)=='DOMIPROD') idef = 0;
  call init_parameter('gen_aug_lmult', inp, idef, p%gen_aug_lmult, iv);
  if(p%gen_aug_lmult>0 .and. upper(p%prod_basis_type)=='DOMIPROD') then
    _warn("gen_aug_lmult>0 .and. prod_basis_type=='DOMIPROD'")
    _warn("==> gen_aug_lmult will be set to 0 (no augmentation is necessary)")
    p%gen_aug_lmult = 0
  endif 
  call init_parameter('bilocal_type', inp, 'ATOM', p%bilocal_type, iv)
  call init_parameter('bilocal_center', inp, 'POW&COEFF', p%bilocal_center, iv)
  call init_parameter('bilocal_center_pow', inp, 1D0, p%bilocal_center_pow, iv)
  call init_parameter('bilocal_center_coeff', inp, '2*L+1', p%bilocal_center_coeff, iv)  

  call init_parameter('local_type', inp, 'ATOM', p%local_type, iv)
  call init_parameter('core_split_ratio', inp, 10D0, p%core_split_ratio, iv)
  call init_parameter('stop_after_dp', inp, 0, p%stop_after_dp, iv)
  call init_parameter('pb_funct_xsf', inp, 0, p%pb_funct_xsf, iv)
  call init_parameter('biloc_interm_storage', inp, "RD", p%biloc_interm_storage, iv) !DA or RD
  call init_parameter('mixed_interm_storage', inp, "RD", p%mixed_interm_storage, iv) !DA or RD
  !! END of Augment or not augment ?
    
end subroutine ! init_aux_prod_basis

!
!
!
character(100) function get_prod_basis_type(p)
  use m_upper, only : upper
  ! external
  implicit none
  type(prod_basis_param_t) :: p
  if(len_trim(p%prod_basis_type)<1) _die('!%prod_basis_type')
  get_prod_basis_type = upper(p%prod_basis_type)
end function ! get_prod_basis_type

!
!
!
character(100) function get_bilocal_center(p)
  use m_upper, only : upper
  ! external
  implicit none
  type(prod_basis_param_t) :: p
  if(len_trim(p%bilocal_center)<1) _die('!%bilocal_center')
  get_bilocal_center = upper(p%bilocal_center)
end function ! get_bilocal_center

!
!
!
character(100) function get_bilocal_center_coeff(p)
  use m_upper, only : upper
  ! external
  implicit none
  type(prod_basis_param_t) :: p
  if(len_trim(p%bilocal_center_coeff)<1) _die('!%bilocal_center_coeff')
  get_bilocal_center_coeff = upper(p%bilocal_center_coeff)
end function ! get_bilocal_center_coeff

!
!
!
real(8) function get_bilocal_center_pow(p) 
  implicit none
  type(prod_basis_param_t), intent(in) :: p
  if(p%bilocal_center_pow==-999) _die('bilocal_center_pow==-999 !')
  get_bilocal_center_pow = p%bilocal_center_pow
end function ! get_bilocal_center_pow

!
!
!
character(100) function get_bilocal_type(p)
  use m_upper, only : upper
  ! external
  implicit none
  type(prod_basis_param_t) :: p
  if(len_trim(p%bilocal_type)<1) _die('!%bilocal_type')
  get_bilocal_type = upper(p%bilocal_type)
end function ! get_bilocal_type

!
!
!
character(100) function get_local_type(p)
  use m_upper, only : upper
  ! external
  implicit none
  type(prod_basis_param_t) :: p
  if(len_trim(p%local_type)<1) _die('!%local_type')
  get_local_type = upper(p%local_type)
end function ! get_local_type

!
!
!
real(8) function get_core_split_ratio(p) 
  implicit none
  type(prod_basis_param_t), intent(in) :: p
  if(p%core_split_ratio<=0) _die('core_split_ratio<=0 !')
  get_core_split_ratio = p%core_split_ratio
end function !  get_core_split_ratio

!
!
!
real(8) function get_eigmin_bilocal(p) 
  implicit none
  type(prod_basis_param_t), intent(in) :: p
  if(p%eigmin_bilocal==-999) _die('eigmin_bilocal=-999 !')
  get_eigmin_bilocal = p%eigmin_bilocal
end function !  get_eigmin_bilocal

!
!
!
real(8) function get_eigmin_local(p) 
  implicit none
  type(prod_basis_param_t), intent(in) :: p
  if(p%eigmin_local==-999) _die('eigmin_local=-999 !')
  get_eigmin_local = p%eigmin_local
end function !  get_eigmin_local

!
!
!
integer function get_report_domiprod(pb_p)
  implicit none
  type(prod_basis_param_t), intent(in) :: pb_p
  if(pb_p%report_domiprod<0) _die('%report_domiprod<0')
  get_report_domiprod = pb_p%report_domiprod
end function !  get_report_domiprod 

!
!
!
integer function get_pb_funct_xsf(pb_p)
  implicit none
  type(prod_basis_param_t), intent(in) :: pb_p
  if(pb_p%pb_funct_xsf<0) _die('%pb_funct_xsf<0')
  get_pb_funct_xsf = pb_p%pb_funct_xsf
end function !  get_pb_funct_xsf
 
!
!
!
integer function get_optimize_centers(pb_p)
  implicit none
  type(prod_basis_param_t), intent(in) :: pb_p
  if(pb_p%optimize_centers<0) _die('%optimize_centers<0')
  get_optimize_centers = pb_p%optimize_centers
end function !  get_optimize_centers 

!
!
!
integer function get_stop_after_dp(pb_p)
  implicit none
  type(prod_basis_param_t), intent(in) :: pb_p
  if(pb_p%stop_after_dp<0) _die('%stop_after_dp<0')
  get_stop_after_dp = pb_p%stop_after_dp
end function ! get_stop_after_dp

!
!
!
integer function get_check_basis(pb_p)
  implicit none
  type(prod_basis_param_t), intent(in) :: pb_p
  if(pb_p%check_basis<0) _die('%check_basis<0')
  get_check_basis = pb_p%check_basis
end function !  get_check_basis 

!
!
!
integer function get_do_shifting_test(pb_p)
  implicit none
  type(prod_basis_param_t), intent(in) :: pb_p
  if(pb_p%do_shifting_test<0) _die('%do_shifting_test<0')
  get_do_shifting_test = pb_p%do_shifting_test
end function !  get_do_shifting_test

!
!
!
integer function get_jcutoff(pb_p)
  implicit none
  type(prod_basis_param_t), intent(in) :: pb_p
  if(pb_p%jcutoff<0) _die('%jcutoff<0')
  get_jcutoff = pb_p%jcutoff
end function ! get_jcutoff

!
!
!
integer function get_metric_type(pb_p)
  implicit none
  type(prod_basis_param_t), intent(in) :: pb_p
  if(pb_p%metric_type<1) _die('%metric_type<1')
  get_metric_type = pb_p%metric_type
end function ! get_metric_type

!
!
!
integer function get_GL_ord_bilocal(pb_p)
  implicit none
  type(prod_basis_param_t), intent(in) :: pb_p
  if(pb_p%GL_ord_bilocal<1) _die('%GL_ord_bilocal<1')
  get_GL_ord_bilocal = pb_p%GL_ord_bilocal
end function ! get_GL_ord_bilocal

!
!
!
integer function get_GL_pcs_bilocal(pb_p)
  implicit none
  type(prod_basis_param_t), intent(in) :: pb_p
  if(pb_p%GL_pcs_bilocal<1) _die('%GL_ord_bilocal<1')
  get_GL_pcs_bilocal = pb_p%GL_pcs_bilocal
end function ! get_GL_pcs_bilocal

end module !m_prod_basis_param
