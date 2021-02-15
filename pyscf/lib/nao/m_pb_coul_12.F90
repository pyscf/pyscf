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

module m_pb_coul_12
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
subroutine pb_coul_12(aux, spp1, spp2, is_overlap, wm2, fmom2, mmom2, &
  array, &
  f1f2_mom, S, real_overlap, tmp, bessel_pp, r_scalar_pow_jp1, ylm_thrpriv)
 
  use m_functs_m_mult_type, only : get_nfunct_mmult,get_m,functs_m_mult_t
  use m_functs_l_mult_type, only : get_nmult, get_j_si_fi
  use m_coul_comm, only : comp_overlap_v6_arg
  
  implicit none
  !!! external
  type(pb_coul_aux_t), intent(in) :: aux
  integer, intent(in) :: spp1
  integer, intent(in) :: spp2(:) ! specie/start/finish (3 numbers)
  logical, intent(in) :: is_overlap
  real(8), intent(in) :: wm2(:,:,:) ! wigner matrices
  type(functs_m_mult_t), intent(in) :: fmom2, mmom2
  real(8), intent(inout) :: array(:,:)
  real(8), intent(inout), allocatable :: f1f2_mom(:), S(:), real_overlap(:,:), tmp(:)
  real(8), intent(in), allocatable :: bessel_pp(:,:), r_scalar_pow_jp1(:)
  complex(8), intent(in), allocatable :: ylm_thrpriv(:)
  !! internal
  integer :: i2,J1,J2,M1,M2,j,nmu1,mu1,si1,fi1, nw(3)
  real(8) :: f1j1f2j2

  if(.not. associated(aux%sp_local2functs_mom)) then
    write(6,*) spp1, spp2(:)
    _die('!aux%sp_local2functs_mom')
  endif

  nw = shape(wm2)/2 + 1

  nmu1 = get_nmult(aux%sp_local2functs_mom(spp1))
  if(size(spp2)/=3) _die('size spp2/=3')

  do mu1=1,nmu1
    call get_j_si_fi(aux%sp_local2functs_mom(spp1), mu1, j1, si1 ,fi1)

    do i2=spp2(2), spp2(3)
      m2 = get_m(fmom2, i2)

      tmp = 0      
      do j2=abs(M2), aux%jcutoff;
        S = 0
        if(is_overlap) then !! overlapping or not overlapping orbitals
          f1f2_mom = aux%sp_local2functs_mom(spp1)%ir_mu2v(:,mu1) * &
                     fmom2%ir_j_prd2v(:,j2,i2)
                 
          do j=abs(j1-j2),j1+j2,2; 
            S(j)=sum(f1f2_mom*bessel_pp(:,j));
            S(j)=S(j)+f1f2_mom(1)*bessel_pp(1,j)/aux%dkappa
          enddo;
                   
        else if ((.not. is_overlap) .and. (.not. aux%logical_overlap)) then
          f1j1f2j2 = aux%sp_local2moms(spp1)%ir_mu2v(1,mu1) * mmom2%ir_j_prd2v(1,j2,i2)

          do j=abs(j1-j2),j1+j2,2;
            if(j/=(j1+j2)) then
              S(j)=0
              cycle
            endif
            S(j)= aux%GGG(j,j2,j1)*f1j1f2j2*r_scalar_pow_jp1(j);
          enddo
        else
          S = 0;
        endif !! endif overlapping or not overlapping orbitals

        call comp_overlap_v6_arg(aux%jcutoff, j1,j2, aux%j1_j2_to_gaunt1, &
          aux%tr_c2r_diag1, aux%tr_c2r_diag2, aux%conjg_c2r_diag1, aux%conjg_c2r_diag2, & 
          S, ylm_thrpriv, real_overlap)

        !! Rotate the spherical harmonics in real_overlap
        do m1=-j1,j1
          tmp(m1) = tmp(m1) + &
            sum(real_overlap(m1,-j2:j2)*wm2(m2+nw(1),-j2+nw(2):j2+nw(2),j2+1))
        enddo ! m1  
        !! END of Rotate the spherical harmonics in real_overlap 

       end do ! l2

       array(si1:fi1,i2) = tmp(-j1:j1)
     end do ! i2
   !enddo ! m1
   enddo ! mu1
     
end subroutine ! pb_coul_12

!
!
!
subroutine comp_kernel_block_12(aux, spp1, spp2, &
  is_overlap, wigner_matrices2, array, f1f2_mom, S, real_overlap, tmp, &
  bessel_pp, r_scalar_pow_jp1, ylm_thrpriv)
 
  use m_functs_m_mult_type, only : get_nfunct_mmult,get_m 
  use m_functs_l_mult_type, only : get_nmult, get_j_si_fi
  use m_coul_comm, only : comp_overlap_v6_arg
  
  implicit none
  !!! external
  type(pb_coul_aux_t), intent(in) :: aux
  integer, intent(in) :: spp1
  integer, intent(in) :: spp2(:) ! specie/start/finish (3 numbers)
  logical, intent(in) :: is_overlap
  real(8), intent(in), allocatable :: wigner_matrices2(:,:,:)
  real(8), intent(inout) :: array(:,:)
  real(8), intent(inout), allocatable :: f1f2_mom(:), S(:), real_overlap(:,:), tmp(:)
  real(8), intent(in), allocatable :: bessel_pp(:,:), r_scalar_pow_jp1(:)
  complex(8), intent(in), allocatable :: ylm_thrpriv(:)
  !! internal
  integer :: i2,J1,J2,M1,M2,j,nmu1,mu1,si1,fi1
  real(8) :: f1j1f2j2
  
  nmu1 = get_nmult(aux%sp_local2functs_mom(spp1))
  if(size(spp2)/=3) _die('size spp2/=3')

  do mu1=1,nmu1
    call get_j_si_fi(aux%sp_local2functs_mom(spp1), mu1, j1, si1 ,fi1)

    do i2=spp2(2), spp2(3)
      m2 = get_m(aux%sp_biloc2functs_mom(spp2(1)), i2)

      tmp = 0      
      do j2=abs(M2), aux%jcutoff;
        S = 0
        if(is_overlap) then !! overlapping or not overlapping orbitals
          f1f2_mom = aux%sp_local2functs_mom(spp1)%ir_mu2v(:,mu1) * &
                     aux%sp_biloc2functs_mom(spp2(1))%ir_j_prd2v(:,j2,i2)
                 
          do j=abs(j1-j2),j1+j2,2; 
            S(j)=sum(f1f2_mom*bessel_pp(:,j));
            S(j)=S(j)+f1f2_mom(1)*bessel_pp(1,j)/aux%dkappa
          enddo;
                   
        else if ((.not. is_overlap) .and. (.not. aux%logical_overlap)) then
          f1j1f2j2 = aux%sp_local2moms(spp1)%ir_mu2v(1,mu1) * &
                     aux%sp_biloc2moms(spp2(1))%ir_j_prd2v(1,j2,i2)

          do j=abs(j1-j2),j1+j2,2;
            if(j/=(j1+j2)) then
              S(j)=0
              cycle
            endif
            S(j)= aux%GGG(j,j2,j1)*f1j1f2j2*r_scalar_pow_jp1(j);
          enddo
        else
          S = 0;
        endif !! endif overlapping or not overlapping orbitals

        call comp_overlap_v6_arg(aux%jcutoff, j1,j2, aux%j1_j2_to_gaunt1, &
          aux%tr_c2r_diag1, aux%tr_c2r_diag2, aux%conjg_c2r_diag1, aux%conjg_c2r_diag2, & 
          S, ylm_thrpriv, real_overlap)

        !! Rotate the spherical harmonics in real_overlap
        do m1=-j1,j1
          tmp(m1) = tmp(m1) + &
            sum(real_overlap(m1,-j2:j2)*wigner_matrices2(m2,-j2:j2,j2));
        enddo ! m1  
        !! END of Rotate the spherical harmonics in real_overlap 

       end do ! l2

       array(si1:fi1,i2) = tmp(-j1:j1)
     end do ! i2
   !enddo ! m1
   enddo ! mu1
     
end subroutine ! comp_kernel_block_12


!
!
!
subroutine comp_kernel_block_21_tr(aux, spp1, spp2, &
  is_overlap, wigner_matrices1, array, f1f2_mom, S, real_overlap, tmp, &
  bessel_pp, r_scalar_pow_jp1, ylm_thrpriv, array_aux)

  implicit none
  !!! external
  type(pb_coul_aux_t), intent(in) :: aux
  integer, intent(in) :: spp1, spp2
  logical, intent(in) :: is_overlap
  real(8), intent(in), allocatable :: wigner_matrices1(:,:,:)
  real(8), intent(inout) :: array(:,:)
  real(8), intent(inout), allocatable :: f1f2_mom(:), S(:), real_overlap(:,:), tmp(:)
  real(8), intent(in), allocatable :: bessel_pp(:,:), r_scalar_pow_jp1(:)
  complex(8), intent(in), allocatable :: ylm_thrpriv(:)
  real(8), intent(inout) :: array_aux(:,:)
  !! internal
  integer :: n1, n2
  integer :: spp1a(3)

  spp1a(1:3) = [spp1, 1, size(aux%pb%sp_biloc2functs(spp1)%prd2m)]
  
  call comp_kernel_block_12(aux, spp2, spp1a, &
    is_overlap, wigner_matrices1, array_aux, f1f2_mom, S, real_overlap, tmp, &
    bessel_pp, r_scalar_pow_jp1, ylm_thrpriv);

  n1 = size(array,1)
  n2 = size(array,2)
  array(1:n1, 1:n2) = transpose(array_aux(1:n2,1:n1))

end subroutine ! comp_kernel_block_21_tr


end module !m_pb_coul_12
