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

module m_pb_coul_11
!
! initialization of dominant products
!
#include "m_define_macro.F90"
  use m_die, only : die
  use m_pb_coul_aux, only : pb_coul_aux_t

  implicit none


contains

!
!
!
subroutine pb_coul_11(aux, spp1, spp2, &
  is_overlap, array, f1f2_mom, S, real_overlap, bessel_pp, r_scalar_pow_jp1, ylm_thrpriv)
  use m_functs_l_mult_type, only : get_nmult, get_j_si_fi
  use m_coul_comm, only : comp_overlap_v6_arg  
  implicit none
  !!! external
  type(pb_coul_aux_t), intent(in) :: aux
  integer, intent(in) :: spp1, spp2
  logical, intent(in) :: is_overlap
  real(8), intent(inout) :: array(:,:)
  real(8), intent(inout), allocatable :: f1f2_mom(:), S(:), real_overlap(:,:)
  real(8), intent(in), allocatable :: bessel_pp(:,:), r_scalar_pow_jp1(:)
  complex(8), intent(in), allocatable :: ylm_thrpriv(:)
  !! internal
  integer :: j,j1,j2
  integer :: nmu1,mu1,nmu2,mu2,si1,si2,fi1,fi2
  real(8) :: f1j1f2j2
  
  nmu2 = get_nmult(aux%sp_local2functs_mom(spp2))
  nmu1 = get_nmult(aux%sp_local2functs_mom(spp1))
  do mu2=1,nmu2;
    call get_j_si_fi(aux%sp_local2functs_mom(spp2), mu2, j2, si2, fi2)

    do mu1=1,nmu1; 
      call get_j_si_fi(aux%sp_local2functs_mom(spp1), mu1, j1, si1, fi1)
      !print*, si1, fi1

      S = 0;
      !! overlapping or not overlapping orbitals
      if(is_overlap) then 
        f1f2_mom = aux%sp_local2functs_mom(spp1)%ir_mu2v(:,mu1) * &
                   aux%sp_local2functs_mom(spp2)%ir_mu2v(:,mu2)
        do j=abs(j1-j2),j1+j2,2; 
          S(j)=sum(f1f2_mom*bessel_pp(:,j));
          S(j)=S(j)+f1f2_mom(1)*bessel_pp(1,j)/aux%dkappa
        enddo

      else if ((.not. is_overlap) .and. (.not. aux%logical_overlap)) then
        !! Multipoles
        f1j1f2j2 = aux%sp_local2moms(spp1)%ir_mu2v(1,mu1) * &
                   aux%sp_local2moms(spp2)%ir_mu2v(1,mu2)
        j=j1+j2
        S(j)= aux%GGG(j,j2,j1)*f1j1f2j2*r_scalar_pow_jp1(j)

      endif 
      !! END of overlapping or not overlapping orbitals

      call comp_overlap_v6_arg(aux%jcutoff, j1,j2, aux%j1_j2_to_gaunt1, &
        aux%tr_c2r_diag1, aux%tr_c2r_diag2, aux%conjg_c2r_diag1, aux%conjg_c2r_diag2, & 
        S, ylm_thrpriv, real_overlap)

      array(si1:fi1,si2:fi2) = real_overlap(-j1:j1,-j2:j2)
    enddo ! mu1
  enddo ! mu2 
  
end subroutine ! pb_coul_11


end module !m_pb_coul_11
