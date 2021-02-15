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

module m_pb_kernel_rows8
!
! initialization of dominant products
!
#include "m_define_macro.F90"
  use m_die, only : die
  use m_log, only : log_size_note, log_memory_note

  implicit none

contains

!
! Evaluates two-center coulomb integrals between products
!
subroutine pb_kernel_rows8(hxc, ps,pf, rows)
  use m_block_crs8, only : is_init, block_crs8_t
  use m_parallel, only : para_t
  use m_pb_coul_aux, only : pb_coul_aux_t
  use m_hxc8, only : hxc8_t
  use m_csphar, only : csphar_rev_m
  use m_functs_l_mult_type, only : get_nmult, get_j_si_fi
    
  implicit  none
  ! external
  type(hxc8_t), intent(in), target :: hxc
  integer, intent(in) :: ps, pf
  real(8), intent(inout) :: rows(:,:)

  ! internal
  type(pb_coul_aux_t), pointer :: a
  type(block_crs8_t), pointer :: sm
  integer, allocatable :: blk2cc(:,:)
  logical, allocatable :: blk2ov(:) ! block --> is overlap
  integer :: spp1, spp2, cc1, cc2,nblk,fi1,fi2,ni,nj,jcl2
  integer :: prow,pcol,blk,nblk_no,sis,j1,j2,m1,m2,jjp1,jjp1mm2
  integer :: sil,sjg,j, fil,fjg, nmu1, nmu2, si1, si2, mu1, mu2
  integer(8) :: bs,bf
  real(8) :: rcut1, rcut2, r_scal, r_vec(3), dSj
  real(8), allocatable :: r_scalar_pow_jp1(:), GGY(:,:,:,:)
  complex(8), allocatable :: covr(:,:), cmat(:,:), ylm_rev(:)

  if(pf<ps) _die('pf<ps')
  if(.not. is_init(hxc%bcrs,__FILE__,__LINE__)) _die('?')
  sm=>hxc%bcrs
  a =>hxc%ca  
  if(a%book_type .ne. 2) _die('a%book_type .ne. 2')
  
  nblk=a%ncenters*(pf-ps+1)
  !! Init local counting 
  sis = int(a%p2sRrsfn(6,ps))
  !! END of Init local counting 

  allocate(blk2ov(nblk))
  blk2ov = .false.
  do prow=ps,pf
    sil = int(a%p2sRrsfn(6,prow)) - sis + 1
    fil = int(a%p2sRrsfn(7,prow)) - sis + 1
    ni  = int(a%p2sRrsfn(8,prow))
 
    do blk=sm%row2ptr(prow),sm%row2ptr(prow+1)-1
      pcol = sm%blk2col(blk)
      blk2ov(pcol+(prow-ps)*a%ncenters) = .true.
      
      sjg = int(a%p2sRrsfn(6,pcol))
      fjg = int(a%p2sRrsfn(7,pcol))
      nj = int(a%p2sRrsfn(8,pcol))

      bs = sm%blk2sfn(1,blk)
      bf = sm%blk2sfn(2,blk)
      rows(sil:fil,sjg:fjg) = reshape(sm%d(bs:bf), [ni,nj])
    enddo ! blk
  enddo ! row
 
!  write(6,*) ps, pf
!  write(6,*) blk2ov  
 
  nblk_no = count(.not. blk2ov) ! number of non-overlapping blocks
  if(nblk_no<1) return

  !!  
  !! Now I will compute the non-overlapping blocks (i.e. add Hartree kernel)
  !!
  
  !! Form a list of non-overlapping centers
  allocate(blk2cc(2,nblk_no))
  blk=0
  do prow=ps,pf
    do pcol=1,a%ncenters
      if(blk2ov(pcol+(prow-ps)*a%ncenters)) cycle
      blk = blk + 1
      blk2cc(1:2,blk) = [prow,pcol]
    enddo ! blk
  enddo ! row  
  !! END of Form a list of non-overlapping centers
  
!  write(6,*) nblk_no
!  write(6,'(2i6)') blk2cc

!! Loop over atomic quadruplets
  jcl2 = 2*a%jcl
!$OMP PARALLEL DEFAULT(NONE) &
!$OMP PRIVATE(r_scalar_pow_jp1,si1,si2,ylm_rev,rcut1,rcut2,fi1,fi2) &
!$OMP PRIVATE(spp1,spp2,cc1,cc2,covr,cmat,m1,m2,jjp1,jjp1mm2,dSj) &
!$OMP PRIVATE(sil,sjg,j,r_scal,r_vec,nmu1,nmu2,j1,j2,GGY) &
!$OMP SHARED (a,rows,sis,nblk_no,blk2cc,jcl2)
  allocate(covr(-a%jcl:a%jcl,-a%jcl:a%jcl))
  allocate(cmat(-a%jcl:a%jcl,-a%jcl:a%jcl))
  allocate(ylm_rev(0:(jcl2+1)**2))
  allocate(r_scalar_pow_jp1(0:jcl2))
  allocate(GGY(-a%jcl:a%jcl,-a%jcl:a%jcl,0:a%jcl,0:a%jcl))
!$OMP DO
  do blk=1,nblk_no
    cc1 = blk2cc(1,blk)
    cc2 = blk2cc(2,blk)

    spp2 = int(a%p2sRrsfn(1,cc2))
    spp1 = int(a%p2sRrsfn(1,cc1))

    rcut2 = a%p2sRrsfn(5,cc2)
    rcut1 = a%p2sRrsfn(5,cc1)
    r_vec = a%p2sRrsfn(2:4,cc2) - a%p2sRrsfn(2:4,cc1)

    call csphar_rev_m(r_vec, ylm_rev, jcl2)

    r_scal = sqrt(sum(r_vec*r_vec))
    
    r_scalar_pow_jp1(0) = 1.0D0/r_scal
    do j=1,jcl2; r_scalar_pow_jp1(j) = r_scalar_pow_jp1(j-1)*r_scalar_pow_jp1(0); enddo
    !do j=0,2*a%jcl; r_scalar_pow_jp1(j) = 1.0D0/(r_scal**(j+1)); enddo

    ! Redefine the Harmonic again
    do j=0,jcl2
      jjp1 = j*(j+1)
      ylm_rev(jjp1-j:jjp1+j) = ylm_rev(jjp1-j:jjp1+j)*r_scalar_pow_jp1(j)
    enddo  
    ! END of Redefine the Harmonic again

    !!
    do j2=0,a%jcl
      do j1=0,a%jcl
        j=j1+j2
        jjp1 = j*(j+1)
        do m2=-j2,j2
          jjp1mm2 = jjp1-m2
          covr(-j1:j1,m2) = a%Gamma_Gaunt(-j1:j1,m2,j1,j2)*ylm_rev(jjp1mm2-j1:jjp1mm2+j1)
        enddo ! m2
        !! convert complex angular momentum coeffs to real coeffs (zaxpy slower here!)
        do m2=-j2,j2
          cmat(-j1:j1,m2) = covr(-j1:j1,m2)*a%tr_c2r_diag1(m2)+covr(-j1:j1,-m2)*a%tr_c2r_diag2(-m2)
        enddo ! m2

        do m1=-j1,j1
          GGY(m1,-j2:j2,j1,j2) = &
            real(a%conjg_c2r_diag1(m1)*cmat(m1,-j2:j2)+a%conjg_c2r_diag2(-m1)*cmat(-m1,-j2:j2),8)
        enddo ! m1
        !! END of convert complex angular momentum coeffs to real coeffs
      enddo ! j1
    enddo ! j2
    !!

    sil = int(a%p2sRrsfn(6,cc1)) - sis + 1
    sjg = int(a%p2sRrsfn(6,cc2))
    
    nmu2 = a%sp2nmu(spp2)
    nmu1 = a%sp2nmu(spp1)    
    do mu2=1,nmu2
      j2  =     a%mu_sp2jsfn(1,mu2,spp2)
      si2 = sjg+a%mu_sp2jsfn(2,mu2,spp2)-1
      fi2 = sjg+a%mu_sp2jsfn(3,mu2,spp2)-1

      do mu1=1,nmu1
        j1  =     a%mu_sp2jsfn(1,mu1,spp1)
        si1 = sil+a%mu_sp2jsfn(2,mu1,spp1)-1
        fi1 = sil+a%mu_sp2jsfn(3,mu1,spp1)-1

        dSj = a%sp_local2moms(spp1)%ir_mu2v(1,mu1)*a%sp_local2moms(spp2)%ir_mu2v(1,mu2)
        rows(si1:fi1,si2:fi2) = GGY(-j1:j1,-j2:j2,j1,j2)*dSj

      enddo ! mu1
    enddo ! mu2
  enddo ! blk
  !$OMP END DO
  _dealloc(ylm_rev)
  _dealloc(covr)
  _dealloc(cmat)
  _dealloc(r_scalar_pow_jp1)
  _dealloc(GGY)
  !$OMP END PARALLEL

  _dealloc(blk2ov)
  _dealloc(blk2cc)

end subroutine !pb_kernel_rows8


end module !m_pb_kernel_rows8
