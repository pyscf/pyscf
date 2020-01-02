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

module m_spin_dens_fini8

#include "m_define_macro.F90"

  implicit none

contains

!
!
!
subroutine spin_dens(sda, br, dens)
  use m_harmonics, only : rsphar
  use m_spin_dens_aux, only : spin_dens_aux_t, comp_coeff, slm_loc, inz2vo, dm_f2
  
  implicit none
  type(spin_dens_aux_t), intent(in) :: sda
  real(8), intent(in) :: br(3)
  real(8), intent(out) :: dens(:)

  ! Interne Variable:
  real(8) :: br0(3), rho
  real(8) :: fr_val
  integer  :: atm, spa, mu, nnzo, j, m, jjp1,k,so,start_ao
  real(8)  :: coeff(-2:3)

  nnzo  = 0; !! index to write non-zero orbitals to f(:)
           !! latter -- the maximal index in f(:) to treat

  !! values of localized orbitals
  do atm=1,sda%natoms;
    spa  = sda%atom2sp(atm); 
    br0  = br - sda%coord(:,atm);  !!print *, 'br01',br,br01,br1;
    rho = sqrt(sum(br0**2));
    if (rho>sda%atom2rcut(atm)) cycle
    call comp_coeff(sda, coeff, k, rho)
    call rsphar(br0, slm_loc(0:), sda%sp2jmx(spa));
    so = sda%atom2start_orb(atm)
    
    do mu=1,sda%sp2nmult(spa); 
      if(rho>sda%mu_sp2rcut(mu,spa)) cycle;
      start_ao = sda%mu_sp2start_ao(mu, spa)
      fr_val = sum(coeff*sda%psi_log(k-2:k+3,mu,spa));
      j = sda%mu_sp2j(mu,spa);
      jjp1 = j*(j+1);
      inz2vo(nnzo+1:nnzo+2*j+1,1)= fr_val*slm_loc(jjp1-j:jjp1+j)
      do m =-j,j;
        nnzo = nnzo + 1
        inz2vo(nnzo,2) = start_ao + j + m + so - 1
      end do ! m
    enddo; ! mu
  enddo; ! atom
  inz2vo(0,1:2) = nnzo

  ! write(6,*) __FILE__, __LINE__, sum(inz2vo)  
  call spin_dens_orb_sprs(inz2vo, sda%nspin, sda%DM, dens, dm_f2)
  !if (sum(abs(br))<0.3d0) write(6,*) __FILE__, __LINE__, sum(br), sum(dens)
end subroutine ! spin_dens

!
!
!
subroutine spin_dens_orb_sprs(inz2vo, nspin,DM, dens, dm_f2)
  use m_sparse_vec, only : get_nnz
  implicit none
  !! external
  real(8), intent(in), allocatable  :: inz2vo(:,:)
  integer, intent(in)  :: nspin
  real(8), intent(in)  :: DM(:,:,:)
  real(8), intent(out) :: dens(:)
  real(8), intent(inout) :: dm_f2(:) ! auxiliary
  !! internal
  integer :: ispin, inzo1, inzo2, nnzo, o1, o2
  nnzo = get_nnz(inz2vo)
  
  if(nnzo==0) then; dens = 0; return; endif;

  !!-------------------------------------------------------------------
  !! f(:) contains the values of atomic orbitals, which contribute to the density
  !! (at a given point). 
  !!-------------------------------------------------------------------
  dens = 0
  do ispin=1, nspin
    do inzo2=1,nnzo
      o2 = int(inz2vo(inzo2,2))
      do inzo1=1,nnzo
        o1 = int(inz2vo(inzo1,2))
        dm_f2(inzo1) = DM(o1,o2,ispin)*inz2vo(inzo2,1)
      enddo
        
      dens(ispin) = dens(ispin)+sum( dm_f2(1:nnzo) * inz2vo(1:nnzo,1) )
      
    end do ! inzo2
  enddo ! ispin

end subroutine ! spin_dens

end module !m_spin_dens_fini8
