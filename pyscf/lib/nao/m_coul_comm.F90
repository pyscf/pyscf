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

module m_coul_comm
!
! initialization of dominant products
!
#include "m_define_macro.F90"
  use m_log, only : die
  
contains

!
! Computes a customised Gaunt coefficient
!
subroutine comp_gaunt_coeff_kernel(jmx1, jmx2, itype, dkappa, j1_j2_to_gaunt)
#define _sname 'comp_gaunt_coeff_kernel'
  use m_arrays, only : d_array3_t
  use m_wigner2, only : Wigner
  implicit none
  !! external
  integer, intent(in) :: jmx1, jmx2, itype
  real(8), intent(in) :: dkappa
  type(d_array3_t), intent(inout), allocatable :: j1_j2_to_gaunt(:,:)
  !! internal
  integer :: j1,j2,j,m1,m2,expo,m
  real(8) :: factor
  real(8) :: pi
  pi = 4D0*atan(1D0)
  
  !! Compute Gaunt coefficients
  allocate(j1_j2_to_gaunt(0:jmx1,0:jmx2))
  do j2=0, jmx2
    do j1=0, jmx1
      allocate(j1_j2_to_gaunt(j1,j2)%array(-j1:j1,-j2:j2,abs(j1-j2):j1+j2))
      j1_j2_to_gaunt(j1,j2)%array = 0
      
      do j=abs(j1-j2),j1+j2,2;
    
      factor=sqrt( (2D0*j1+1)*(2D0*j2+1)*(2D0*j+1)/(4*pi) )
      expo=(j+j2-j1)/2;
      do m1=-j1,j1
        do m2=-j2,j2; m=m2-m1
          if (abs(m) <= j) then; 

            if(itype==2) then ! Gaunt for overlap
              j1_j2_to_gaunt(j1,j2)%array(m1,m2,j)=4*pi*((-1D0)**expo)*dkappa*((-1D0)**m2)*&
                factor*Wigner(j1,0,j2,0,j,0)*Wigner(j1,m1,j2,-m2,j,m);
            else if(itype==1) then  ! Gaunt for coulomb 
              j1_j2_to_gaunt(j1,j2)%array(m1,m2,j)=16*pi**2*((-1D0)**expo)*dkappa*((-1D0)**m2)*&
                factor*Wigner(j1,0,j2,0,j,0)*Wigner(j1,m1,j2,-m2,j,m);
            else if(itype==3) then ! Gaunt for nabla
              j1_j2_to_gaunt(j1,j2)%array(m1,m2,j)=16*pi**2*((-1D0)**expo)*dkappa*((-1D0)**m2)*&
                factor*Wigner(j1,0,j2,0,j,0)*Wigner(j1,m1,j2,-m2,j,m);
            else 
              write(0,*) _sname//': itype unknown', itype, '==>STOP'
              _die('itype unknown')
            endif ! logical_sigma_overlap
          endif ! (abs(m) <= j)
        enddo ! m2
      enddo ! m1
      enddo ! j
    enddo ! j1
  enddo ! j2
  !! END of Compute Gaunt coefficients

#undef _sname
  
end subroutine ! comp_gaunt_coeff_kernel

!
! Calculates the Coulomb bilocal or local integral
!  \int \int  dr dr'  f_1(r)  |r - r'|^(-1)  f_2(r').
!
subroutine comp_overlap_v6_arg(jcutoff, j1,j2, j1_j2_to_gaunt, &
  tr_c2r_diag1, tr_c2r_diag2, conjg_c2r_diag1, conjg_c2r_diag2, & 
  S, ylm_thrpriv, real_overlap)
  use m_arrays, only : d_array3_t
  
  implicit none
  ! Aeussere Variable: 
  integer, intent(in):: j1,j2,jcutoff
  type(d_array3_t), intent(in), allocatable :: j1_j2_to_gaunt(:,:)
  complex(8), intent(in), allocatable :: tr_c2r_diag1(:), tr_c2r_diag2(:)
  complex(8), intent(in), allocatable :: conjg_c2r_diag1(:), conjg_c2r_diag2(:)
  real(8), intent(in) :: S(0:)
  complex(8), intent(in) :: ylm_thrpriv(0:)
  real(8), intent(inout) :: real_overlap(-jcutoff:jcutoff,-jcutoff:jcutoff)

  ! Innere Variable:
  integer::j,m1,m2,m, jjp1
  integer :: mm1, mm2
  complex(8) :: overlap_thrpriv(-j1:j1,-j2:j2)
  complex(8) :: mat_thrpriv(-j1:j1,-j2:j2)
  
  overlap_thrpriv(-j1:j1,-j2:j2)=0;
  
  do j=abs(j1-j2),j1+j2,2
    jjp1 = j*(j+1)
    do m2=-j2,j2;
    do m1=-j1,j1; m = m2 - m1;
      if (abs(m)>j) cycle;
      overlap_thrpriv(m1,m2) = overlap_thrpriv(m1,m2) + &
        S(j)*j1_j2_to_gaunt(j1,j2)%array(m1,m2,j)*ylm_thrpriv(jjp1+m);

    enddo ! m1
    enddo ! m2
  enddo ! j

  !! convert complex overlap to real_overlap
  do mm2=-j2,j2
   mat_thrpriv(-j1:j1, mm2) = &
     overlap_thrpriv(-j1:j1,mm2)*tr_c2r_diag1(mm2)+overlap_thrpriv(-j1:j1,-mm2)*tr_c2r_diag2(-mm2)
  end do

  do mm1=-j1,j1
    real_overlap(mm1,-j2:j2) = &
      real(conjg_c2r_diag1(mm1)*mat_thrpriv(mm1,-j2:j2)+conjg_c2r_diag2(-mm1)*mat_thrpriv(-mm1,-j2:j2),8)
  end do

end subroutine !comp_overlap_v6_arg


end module !m_coul_comm
