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

module m_pb_ls_blocks_ovrlp_all
!
! initialization of dominant products
!
#include "m_define_macro.F90"
  use m_die, only : die
  use m_log, only : log_size_note, log_memory_note
  use m_prod_basis_type, only : prod_basis_t

  implicit none

contains

!
! List of blocks for overlapping part of interaction kernels
!
subroutine pb_ls_blocks_ovrlp_all(pb, blk2cc)
  use m_prod_basis_type, only : prod_basis_t
  use m_prod_basis_gen, only : get_nfunct_pbook, get_nbook, get_book
  use m_prod_basis_type, only : get_nc_book, get_coord_center
  use m_prod_basis_type, only : get_rcut_per_center, get_nfunct_per_book
  use m_book_pb, only : book_pb_t
  
  implicit none
  !! external
  type(prod_basis_t), intent(in) :: pb
  integer, intent(inout), allocatable :: blk2cc(:,:)
  !! internal
  integer :: natom_pairs, step, iblock, ip1, ncpb1,icpb1, ip2, ncpb2, icpb2
  logical :: is_overlap
  type(book_pb_t) :: book1, book2
  real(8) :: Rvec1(3), Rvec2(3), r_vec(3), r_scalar, rcut1, rcut2
  
  _dealloc(blk2cc)
  natom_pairs = get_nbook(pb)

  do step=1,2
    iblock = 0
    do ip1=1,natom_pairs
      book1 = get_book(pb, ip1)
      ncpb1 = get_nc_book(pb, book1)

      do ip2=1,natom_pairs
        book2 = get_book(pb, ip2)
        ncpb2 = get_nc_book(pb, book2)

        is_overlap = .false.
        do icpb2=1,ncpb2
          Rvec2 = get_coord_center(pb, book2, icpb2)
          rcut2 = get_rcut_per_center(pb, book2, icpb2)
          
          do icpb1=1,ncpb1
            Rvec1 = get_coord_center(pb, book1, icpb1)
            rcut1 = get_rcut_per_center(pb, book1, icpb1)
            r_vec = Rvec2 - Rvec1

            r_scalar = sqrt(sum(r_vec*r_vec));
            is_overlap = (r_scalar<(rcut1+rcut2))
            if(is_overlap) exit
          enddo ! 
          if(is_overlap) exit
        enddo ! 
        
        if(.not. is_overlap) cycle
        
        iblock = iblock + 1
        if(step==2) blk2cc(1:2,iblock) = [ip1, ip2]
      enddo ! ip1
    enddo ! ip2
    if(step==1) allocate( blk2cc(2,iblock) )
  enddo ! step  

  
end subroutine ! pb_ls_blocks_all

end module !m_pb_ls_blocks_all
