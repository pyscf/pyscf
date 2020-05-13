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

module m_init_vrtx_cc_apair

#include "m_define_macro.F90" 
  use m_precision, only : blas_int
  use m_die, only : die
  use m_warn, only : warn
  use m_pair_info, only : pair_info_t
  use iso_c_binding, only: c_double, c_double_complex, c_int64_t
 
  !use m_timing, only : get_cdatetime
  
  implicit none
  private die
  private warn
  !private get_cdatetime
    
  real(c_double), allocatable :: ff2(:,:,:,:,:), evals(:,:,:), vertex_real2(:,:,:,:,:)
  real(c_double), allocatable :: rhotb(:,:), sp2rcut(:)
  integer, allocatable :: oo2num(:,:), m2nf(:), p2n(:)
  complex(c_double_complex), allocatable :: vertex_cmplx2(:,:,:,:,:)
  real(c_double), allocatable :: bessel_pp(:,:), f1f2_mom(:)
  real(c_double), allocatable :: roverlap(:,:), S_comp(:),r_scalar_pow_jp1(:), tmp(:)
  complex(8), allocatable :: ylm(:)

  
  contains

!
! 
!
subroutine init_vrtx_cc_apair(dinp,ninp) bind(c, name='init_vrtx_cc_apair')

  use m_fact, only : init_fact
  use m_sv_libnao_prds, only : sv=>sv_prds
  use m_pb_libnao, only : pb
  use m_para_libnao, only : para
  use m_orb_rspace_aux_libnao, only : orb_a
  use m_biloc_aux_libnao, only : a  
  use m_dp_aux_libnao, only : dp_a
  
  use m_sv_prod_log_get, only : sv_prod_log_get
  use m_system_vars, only : get_nr, get_jmx, get_norbs_max, get_natoms, get_sp2rcut
  use m_biloc_aux, only : init_biloc_aux
  use m_orb_rspace_type, only : init_orb_rspace_aux
  use m_prod_basis_param, only : get_jcutoff
  use m_parallel, only : init_parallel  
  use m_dp_aux, only: preinit_dp_aux, deallocate_dp_aux
  use m_init_book_dp_apair, only : init_book_dp_apair  
  use m_hkernel_pb_bcrs8, only : hkernel_pb_bcrs
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: ninp
  real(c_double), intent(in) :: dinp(ninp)
  !! internal

  integer :: ul, natoms, nr, jcutoff, jmx, norbs_max, nf_max, nterm_max, pair
  real(c_double) :: ac_rcut
  real(c_double), allocatable :: sp2rcut(:)
  
  !! executable statements
  call init_parallel(para, 0)
  call init_fact()  !! Initializations for product reduction/spherical harmonics/wigner3j in Talman's way
  call sv_prod_log_get(dinp,ninp, sv, pb)
  call preinit_dp_aux(sv, dp_a)
  call init_book_dp_apair(pb)   
  call hkernel_pb_bcrs(pb, dp_a%hk)
  
  call init_orb_rspace_aux(sv, orb_a, ul)
  call init_biloc_aux(sv, pb%pb_p, para, orb_a, a)

  natoms = get_natoms(a%sv)
  nr = get_nr(a%sv)
  jcutoff = get_jcutoff(a%pb_p)
  jmx = get_jmx(a%sv)
  norbs_max = get_norbs_max(a%sv)
  nf_max = a%nf_max
  nterm_max = a%nterm_max
  ac_rcut   = a%pb_p%ac_rcut
  call get_sp2rcut(sv, sp2rcut)

  _dealloc(ff2)
  _dealloc(vertex_cmplx2)
  _dealloc(vertex_real2)
  _dealloc(evals)
  _dealloc(rhotb)
  _dealloc(bessel_pp)
  _dealloc(f1f2_mom)
  _dealloc(roverlap)
  _dealloc(ylm)
  _dealloc(S_comp)
  _dealloc(r_scalar_pow_jp1)
  _dealloc(tmp)
  _dealloc(p2n)
  
  allocate(ff2(nr,0:jcutoff,nf_max,-jmx*2:jmx*2,2))
  allocate(vertex_cmplx2(-jmx*2:jmx*2,nf_max,norbs_max,norbs_max,2))
  allocate(vertex_real2(-jmx*2:jmx*2,nf_max,norbs_max,norbs_max,2))
  allocate(evals(nf_max,nf_max+1,-jmx*2:jmx*2))
  allocate(rhotb(nr,nterm_max))

  !! Comput of Coulomb matrix elements
  allocate(bessel_pp(a%nr, 0:2*a%jcutoff));
  allocate(f1f2_mom(a%nr));
  allocate(roverlap(-a%jcutoff:a%jcutoff,-a%jcutoff:a%jcutoff));
  allocate(ylm(0:(2*a%jcutoff+1)**2));
  allocate(S_comp(0:2*a%jcutoff));
  allocate(r_scalar_pow_jp1(0:2*a%jcutoff));
  allocate(tmp(-a%jcutoff:a%jcutoff))
  !! END of Comput of Coulomb matrix elements

  allocate(p2n(1+natoms))
  p2n = -999  
  do pair=1,natoms
    p2n(pair) = pb%book_dp(pair)%fi(3)-pb%book_dp(pair)%si(3)+1
  enddo


end subroutine ! init_vrtx_cc_apair


end module !m_init_vrtx_cc_apair
