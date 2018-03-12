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

module m_wigner_rotation
  use m_log, only : die

#include "m_define_macro.F90"

  contains

!
! a wrapper around make_standard_rotation
!
subroutine coords2rotation_n_to_z(coord1, coord2, rotation_n_to_z)
  !! external
  implicit none
  real(8), intent(in) :: coord1(3), coord2(3)
  real(8), intent(out) :: rotation_n_to_z(3,3)

  !! internal
  real(8) :: translation_vector(3), phi, theta, d12
  real(8) :: rotation_z_to_n(3,3)

  translation_vector = coord2-coord1;
  d12=sqrt(sum(translation_vector*translation_vector));
  if(d12==0) then
    phi   = 0;
    theta = 0;
  else
    call translation2theta_phi(translation_vector, theta, phi)
  endif;
  call make_standard_rotation(theta,phi,rotation_z_to_n,rotation_n_to_z)

end subroutine ! comp_rotation_n_to_z

!
!
!
subroutine translation2theta_phi(translation, theta, phi)
  implicit none
  ! external
  real(8), intent(in) :: translation(3)
  real(8), intent(out) :: theta, phi
  ! internal
  real(8) :: d12
  
  d12 = sqrt(sum(translation*translation))
  if(d12<1D-12) then
    theta = 0;
    phi = 0
  else
    theta=acos( translation(3)/d12 )
    if(sum(abs(translation(1:2)))/d12<1D-13) then
      phi = 0
    else  
      phi = atan2( translation(2), translation(1) )
    endif
  endif   

end subroutine ! translation2theta_phi

!
!
!
subroutine make_complex_J(j,Jx,Jy,Jz)
  implicit none
  ! Extern: 
  integer, intent(in) :: j
  complex(8), dimension(-j:j,-j:j), intent(out) :: Jx, Jy, Jz
  ! Intern
  integer :: m
  complex(8), dimension(-j:j,-j:j) :: Jplus,Jminus
  complex(8) :: imag_unit
  imag_unit=cmplx(0.d0,1.d0,8)
  
  !write(*,*) 'Enter make_complex_J'

  Jplus=0.d0
  Jminus=0.d0
  do m=-j,j-1; 
    Jplus(m+1,m)=sqrt(j*(j+1.d0)-m*(m+1.d0))
    Jminus(m,m+1)= Jplus(m+1,m)
  enddo
  Jx= 0.5d0*          (Jplus+Jminus)
  Jy=-0.5d0*imag_unit*(Jplus-Jminus)
  Jz=0.d0
  do m=-j,j; Jz(m,m)=m; enddo
  !write(*,*) 'Exit make_complex_J'

end subroutine !make_complex_J
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! conjg()!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!subroutine wigner_matrix(j,theta,phi,d_Wigner)
!
!  use m_algebra, only : complex_eigen
!  
!  implicit none
!  ! Extern
!  integer,intent(in) :: j
!  complex(8), intent(out) :: d_Wigner(-j:j,-j:j)
!  real(8), intent(in) :: theta,phi
!
!  ! Intern
!  integer::k
!  complex(8), dimension(-j:j,-j:j) :: Jx, Jy, Jz, X, diagonal_theta,dy,dz
!  real(8) :: eigenvalues(-j:j)
!  logical :: success
!  call make_complex_J(j,Jx,Jy,Jz)
!  call complex_eigen(2*j+1,Jy,X,eigenvalues,success)
!  !error=Jy- matmul(  transpose(X), matmul(Jz, conjg(X))  )
!  diagonal_theta=0.d0;
!  dz=0.d0 
!  do k=-j,j
!    diagonal_theta(k,k)= cmplx(cos(theta*k),-sin(theta*k),8)
!    dz(k,k)  = cmplx(cos(  phi*k),-sin(  phi*k),8)
!  enddo 
!  ! Vorzeichen entspricht exp - i theta *n * J
!  ! Jy(mu,nu) = X(E,mu) * diagonal(E) * X*(E,nu) : 
!  dy=matmul(   transpose(X),matmul(diagonal_theta,conjg(X))    )  
!  d_Wigner=matmul(dz,dy)
!
!end subroutine !wigner_matrix

!
!
!
subroutine wigner_matrix(theta,phi,d_Wigner)
  use m_algebra, only : complex_eigen
  
  implicit none
  ! Extern
  complex(8), intent(inout), allocatable :: d_Wigner(:, :)!d_Wigner(-j:j,-j:j)
  real(8), intent(in) :: theta,phi

  ! Intern
  integer::j, k
  complex(8), allocatable :: Jx(:, :), Jy(:, :), Jz(:, :), X(:, :)
  complex(8), allocatable :: diagonal_theta(:, :), dy(:, :), dz(:, :)
  real(8), allocatable :: eigenvalues(:)
  integer :: dim4
  logical :: success

  j = ubound(d_Wigner, 1)
  allocate(Jx(-j:j,-j:j))
  allocate(Jy(-j:j,-j:j))
  allocate(Jz(-j:j,-j:j))
  allocate(diagonal_theta(-j:j,-j:j))
  allocate(dy(-j:j,-j:j))
  allocate(dz(-j:j,-j:j))

  dim4 = size(Jy, 1)
  allocate(eigenvalues(1:dim4))
  allocate(X(1:dim4, 1:dim4))
  call make_complex_J(j,Jx,Jy,Jz)
  call complex_eigen(Jy,X,eigenvalues,success,dim4)
  !error=Jy- matmul(  transpose(X), matmul(Jz, conjg(X))  )
  diagonal_theta=0.d0;
  dz=0.d0 
  do k=-j,j
    diagonal_theta(k,k)= cmplx(cos(theta*k),-sin(theta*k),8)
    dz(k,k)  = cmplx(cos(  phi*k),-sin(  phi*k),8)
  enddo 
  ! Vorzeichen entspricht exp - i theta *n * J
  ! Jy(mu,nu) = X(E,mu) * diagonal(E) * X*(E,nu) : 
  dy=matmul(   transpose(X),matmul(diagonal_theta,conjg(X))    )  
  d_Wigner=matmul(dz,dy)

end subroutine !wigner_matrix


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


subroutine wigner_matrix_inv(j,theta,phi,d_Wigner_inv)

  use m_algebra, only : complex_inverse
  implicit none
  ! Extern
  Integer,intent(in)::j
  complex(8), intent(out)::d_Wigner_inv(-j:j,-j:j)
  real(8), intent(in)::theta,phi

  ! Intern
  complex(8), allocatable :: d_Wigner(:, :)
  complex(8), Dimension(1:2*j+1,1:2*j+1):: aux, inverse_aux
  logical::success

  allocate(d_Wigner(-j:j, -j:j))
  d_Wigner = 0

  call wigner_matrix(theta,phi,d_Wigner)
  aux(1:2*j+1,1:2*j+1)=d_wigner!(-j:j,-j:j)

  call complex_inverse( 2*j+1, aux, inverse_aux, success )
  d_wigner_inv(-j:j,-j:j)=inverse_aux(1:2*j+1,1:2*j+1)
  if (.not. success) then
    _die(' Wigner_matrix_inv: sucess=false')
  endif
  
end subroutine wigner_matrix_inv
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Subroutine Real_Wigner_matrix(j,theta,phi,d_r)
  Implicit None
  ! Extern: 
  Integer::j
  real(8):: theta,phi 
  real(8), Dimension(-j:j,-j:j)::d_r
  ! Intern: 
  complex(8), Dimension(-j:j,-j:j)::real_from_complex
  complex(8), Dimension(-j:j,-j:j)::complex_from_real
  complex(8), allocatable :: d_c(:, :)
  Integer::i

  allocate(d_c(-j:j, -j:j))
  d_c = 0
  real_from_complex=0.d0
  real_from_complex(0,0)=1.d0
  do i=1,j
     real_from_complex( i, i)= ((-1D0)**i)* sqrt(2.d0)/2; 
     real_from_complex( i,-i)= sqrt(2.d0)/2; 
     real_from_complex(-i,-i)= cmplx(0.d0,  sqrt(2.d0)/2,8)
     real_from_complex(-i, i)= ((-1D0)**i) *cmplx(0.d0, -sqrt(2.d0)/2,8)
  enddo

   complex_from_real=transpose(conjg(real_from_complex))

  Call  Wigner_matrix(theta,phi,d_c)
  d_r= real(    matmul(real_from_complex,     matmul(conjg(d_c),complex_from_real)     ))

  ! ist hier die falsche Phase ? 
End Subroutine Real_Wigner_matrix


!
! For a given translation vector (R2-R1) and angular momentum j, 
! returns rotation matrices for real spherical harmonics
!
subroutine simplified_wigner(translation, j, wigner, inverse_wigner) 
  use m_algebra, only : real_inverse
  implicit none
  ! Extern: 
  real(8), intent(in) :: translation(3)
  integer, intent(in) :: j
  real(8), intent(out) :: wigner(-j:,-j:), inverse_wigner(-j:,-j:)

  ! Intern 
  real(8) :: norm, phi, theta
  real(8), dimension(2*j+1,2*j+1) :: aux, inverse_aux
  real(8), dimension(-j:j,-j:j) :: dd
  logical::success

  ! write(*,*) 'Enter simplified_Wigner'
  if (j==0) then
    wigner=1;
    inverse_wigner=1;
  
  else if (j>0) then 
    norm=sqrt(sum(translation*translation)) 
    if (norm==0) then
      write(0,*) 'simplified_wigner: norm==0', norm 
      stop
    endif
    call translation2theta_phi(translation, theta, phi)
    call real_wigner_matrix(j,theta,phi,dd)
    aux(1:2*j+1,1:2*j+1)=dd(-j:j,-j:j)
    call real_inverse(2*j+1,aux,inverse_aux,success)
    wigner=dd
    inverse_wigner(-j:j,-j:j)=inverse_aux(1:2*j+1,1:2*j+1)
    if (.not. success) then
      _die('simplified_wigner: sucess==false')
    endif
  else 
     print*, 'j<0? ', j
     _die('simplified_wigner: j<0 ?')
  endif ! j>0	
  ! write(*,*) 'Exit simplified_Wigner'

end subroutine !simplified_wigner

!
! Make rotation matrices out of a translation vector
! translation vector will be collinear to Z' (Z-axis in rotated system of coordinates)
!
subroutine translation_to_rotation(translation, rotation, inverse_rotation)
  implicit none
  real(8) :: translation(3)
  real(8), dimension(1:3,1:3), intent(out) :: rotation, inverse_rotation

  !! internal
  real(8) :: norm, phi, theta

  norm=sqrt(sum(translation*translation))
  if (norm==0) then
    write(0,*) 'simplified_wigner: norm==0', norm
    stop
  endif
  call translation2theta_phi(translation, theta, phi)
  call make_standard_rotation(theta,phi, rotation, inverse_rotation);

end subroutine ! translation_to_rotation

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Conventional rotation matrix
!   rotation   gives  the vector in coord system Z
! Axis z of coord system Z points always along direction (theta,phi)
! theta, phi
!
subroutine make_standard_rotation(theta,phi,rotation, inverse_rotation)
  implicit none
  real(8), intent(in) :: theta,phi
  real(8), dimension(1:3,1:3), intent(out) :: rotation, inverse_rotation

  !! internal
  real(8), dimension(1:3,1:3) :: R_y, Inverse_R_y, R_z, Inverse_R_z
!   integer :: j

        R_y=0;         R_z=0
Inverse_R_y=0; Inverse_R_z=0


Inverse_R_y(1,1)= cos(theta); Inverse_R_y(1,3)=  -sin(theta);
            Inverse_R_y(2,2)=1
Inverse_R_y(3,1)= sin(theta); Inverse_R_y(3,3)=   cos(theta);

        R_y(1,1) = cos(theta);        R_y(1,3) = sin(theta);
                      R_y(2,2) = 1
        R_y(3,1) = -sin(theta);       R_y(3,3) = cos(theta);


Inverse_R_z(1,1)= cos(phi); Inverse_R_z(1,2)=   sin(phi); 
Inverse_R_z(2,1)= -sin(phi);  Inverse_R_z(2,2)=   cos(phi);
Inverse_R_z(3,3)=1

        R_z(1,1)= cos(phi);         R_z(1,2)= - sin(phi); 
        R_z(2,1)=  sin(phi);          R_z(2,2)=   cos(phi);
        R_z(3,3)=1

  rotation=matmul(R_z,R_y)

!   write(6,*)'Inverse_R_y', cos(theta), theta
!   do j=1,3
!   write(6,*) Inverse_R_y(j,:)
!   enddo

!   write(6,*)'Inverse_R_z'
!   do j=1,3
!   write(6,*) Inverse_R_z(j,:)
!   enddo

  inverse_rotation=matmul(Inverse_R_y,Inverse_R_z)
end subroutine !make_standard_rotation


end module !m_wigner_rotation
