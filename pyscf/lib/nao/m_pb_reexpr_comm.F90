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

module m_pb_reexpr_comm
!
! initialization of dominant products
!
#include "m_define_macro.F90"
  use m_die, only : die
  use m_timing, only : get_cdatetime
  implicit none
  private die

contains


!
!
!
subroutine init_counting_fini(i2b, coeffs)
  use m_book_pb, only : book_pb_t
  use m_coeffs_type, only : d_coeffs_ac_dp_t, alloc_coeffs
  implicit none 
  !! external
  type(book_pb_t), intent(in) :: i2b(:)
  type(d_coeffs_ac_dp_t), intent(inout) :: coeffs
  !! internal
  integer :: i, fi_loc, si_loc, npf_loc, ni

  ni = size(i2b)
  if(ni<1) _die('!ni<1')
  
  call alloc_coeffs(ni, coeffs)
  
  fi_loc = 0
  do i=1,ni 
    coeffs%ind2book_dp_re(i) = -999
    coeffs%ind2book_re(i) = i2b(i)%ic
      
    npf_loc = i2b(i)%fi(3) - i2b(i)%si(3) + 1
    si_loc = fi_loc + 1; 
    fi_loc = si_loc + npf_loc - 1
    coeffs%ind2sfp_loc(1:2,i) = [si_loc, fi_loc]
  enddo !! Cross check here maybe 
  !! ? Does pair2clist(pair)%a(ic)%ic corresponds to atom pair of book(:) array ?

end subroutine  ! init_counting_fini
 
!
!
!
subroutine init_coeffs_counting(nc, book_adj, clist_mix, clist_adj, coeffs)
  use m_book_pb, only : book_pb_t
  use m_coeffs_type, only : d_coeffs_ac_dp_t, alloc_coeffs
  implicit none 
  !! external
  integer, intent(in) :: nc 
  type(book_pb_t), intent(in) :: book_adj(:), clist_mix(:), clist_adj(:)
  type(d_coeffs_ac_dp_t), intent(inout) :: coeffs
  !! internal
  integer :: ic, fi_loc, si_loc, npf_loc, ib

  call alloc_coeffs(nc, coeffs)
  
  fi_loc = 0
  do ic=1,nc 
    ib = clist_adj(ic)%ic
    coeffs%ind2book_dp_re(ic) = ib
    coeffs%ind2book_re(ic) = clist_mix(ic)%ic
      
    npf_loc = book_adj(ib)%fi(3) - book_adj(ib)%si(3) + 1
    si_loc = fi_loc + 1; fi_loc = si_loc + npf_loc - 1
    coeffs%ind2sfp_loc(1:2,ic) = [si_loc, fi_loc]
  enddo !! Cross check here maybe 
  !! ? Does pair2clist(pair)%a(ic)%ic corresponds to atom pair of book(:) array ?

end subroutine  ! init_coeffs_counting


!
!
!
function get_nfunct_per_pair_reexpr(pair, pb, book_adj, pair2clist) result(nfa)
  use m_book_pb, only : book_array1_t, book_pb_t
  use m_prod_basis_type, only : prod_basis_t, get_nfunct_per_book
  implicit none
  !! external
  integer, intent(in) :: pair
  type(prod_basis_t), intent(in) :: pb
  type(book_pb_t), intent(in), allocatable :: book_adj(:)
  type(book_array1_t), intent(in), allocatable :: pair2clist(:)
  integer :: nfa
  !! internal
  integer :: ind, nind, ic
  if(.not. allocated(book_adj)) &
    _die('.not. allocated(book_adj)')
  if(.not. allocated(pair2clist)) &
    _die('.not. allocated(pair2clist)')
  if(pair>size(pair2clist)) &
    _die('pair>size(pair2clist)')
    
  nind = 0
  if(allocated(pair2clist(pair)%a)) nind = size(pair2clist(pair)%a)
  if(nind<1) _die('nind<1')  

  !! Find out total dimension of functions used to reexpress domi prods in the cur pair
  nfa = 0
  do ind=1,nind
    ic = pair2clist(pair)%a(ind)%ic
    if(ic>size(book_adj)) &
      _die('ic>size(book_adj)')
    nfa = nfa + get_nfunct_per_book(pb, book_adj(ic))
  enddo ! ic_indx1
  !! END of Find out total dimension of functions used to reexpress domi prods in the cur pair

end function ! get_nfunct_total

!
!
!
function get_nfunct_pair_max_reexpr(pb, book_adj, pair2clist) result(nf_pair_mx)
  use m_book_pb, only : book_array1_t, book_pb_t
  use m_prod_basis_type, only : prod_basis_t, get_nfunct_per_book, get_npairs
  implicit none
  type(prod_basis_t), intent(in) :: pb
  type(book_pb_t), intent(in), allocatable :: book_adj(:)
  type(book_array1_t), intent(in), allocatable :: pair2clist(:)
  integer :: nf_pair_mx
  !! internal
  integer :: npairs, pair, nind, nf_pair, ic, ind, nf
  
  if(.not. allocated(book_adj)) &
    _die('.not. allocated(book_adj)')
  if(.not. allocated(pair2clist)) &
    _die('.not. allocated(pair2clist)')
  npairs = size(pair2clist)
  if(npairs/=get_npairs(pb)) &
  _die('npairs/=get_npairs(pb)')
  
  nf_pair_mx = 0
  do pair=1,npairs
    nind = 0
    if(allocated(pair2clist(pair)%a)) nind = size(pair2clist(pair)%a)
    if(nind<1) _die('nind<1')
    nf_pair = 0
    if(nind==1 .and. book_adj(pair)%ic==pair2clist(pair)%a(1)%ic) then
      ! write(6,*) 'Pair ', pair, 'is not necessary to reexpress --> will be not treated.'
    else
      ! write(6,*) pair, pb_dp%book_dp(pair)%ic, pb_dp%book_dp(pair)%top, nc
      do ind=1,nind
        ic = pair2clist(pair)%a(ind)%ic
        nf = get_nfunct_per_book(pb, book_adj(ic))
        nf_pair = nf_pair + nf
        !write(6,*) ic, ic_prime, iblocks_pack2block(_pack_indx(ic_prime, ic)), nf
      enddo ! ic_indx
    endif
    nf_pair_mx = max(nf_pair_mx, nf_pair)
  enddo ! pair
  
end function ! get_nfunct_pair_max_reexpr


!
! This subroutine looks into pair2clist entries, finds corresponding entry in the bookkeping book(:)
! and initializes fields ic, si and fi according to the found entry in book.
!
subroutine init_pair2clist_ic_si_fi(book, pair2clist)
  use m_book_pb, only : book_pb_t, book_array1_t, print_info_book_elem
  use m_book_pb, only : get_list_essent_metrics, get_geom_metric_size, get_geom_metric

  implicit none
  ! external
  type(book_pb_t), intent(in), allocatable :: book(:)
  type(book_array1_t), allocatable, intent(inout) :: pair2clist(:)
  ! internal
  integer :: npairs, pair, ic, nind, ind, ms
  type(book_pb_t) :: book_entry
  real(8), allocatable :: metric(:)
  real(8), allocatable :: ls_metrics(:,:)

  if(.not. allocated(book)) &
    _die('.not. allocated(book)')

  if(.not. allocated(pair2clist) ) &
    _die('.not. allocated(pair2clist)')
  
  npairs = size(pair2clist)

  call get_list_essent_metrics(book, ls_metrics)
  ms = get_geom_metric_size()
  allocate(metric(ms))

  do pair=1, npairs
    if(.not. allocated(pair2clist(pair)%a)) &
      _die('not allocated pair2clist(pair)%a')
    nind = size(pair2clist(pair)%a)
    do ind=1, nind
      book_entry = pair2clist(pair)%a(ind)
      metric = get_geom_metric(book_entry)
      ic = get_list_index(ls_metrics, metric)
      if(ic<0) then
        write(6,'(a30, 3i6)') 'pair, ind, ic: ', pair, ind, ic
        call print_info_book_elem(6, book_entry)
        _die('not found.')
      endif

      pair2clist(pair)%a(ind)%ic = ic
      pair2clist(pair)%a(ind)%si = book(ic)%si
      pair2clist(pair)%a(ind)%fi = book(ic)%fi
    enddo ! ind
  enddo ! pair

end subroutine ! init_pair2clist_ic_si_fi

!
!
!
function get_list_index(ls, elem) result(ind)
  implicit none
  !! external
  real(8), intent(in), allocatable :: ls(:,:)
  real(8), intent(in) :: elem(:)
  integer :: ind

  !! internal
  integer :: i, ndim, nelem
  
  if(.not. allocated(ls)) &
   _die('.not. allocated(book)')
  nelem = size(ls,2)
  ndim = size(ls,1)
  if(ndim/=size(elem)) &
    _die('ndim/=size(elem)')

  ind = -1

  do i=1,nelem
    if( all(ls(:, i)==elem) ) then
      ind = i
      exit
    endif
  enddo ! i

end function ! get_index_by_geom_sense

!
! Joins book2 with book1. Elements from book1 will appear first in book12,
! elements of book2 will appear next. Coinciding elements will be removed from book12
! i.e. book12 will be a SET.
!
subroutine adjoin_book_set(book1, book2, book12)
  use m_book_pb, only : book_pb_t, get_list_essent_metrics
  use m_sets, only : dlist2set
  implicit none
  ! external
  type(book_pb_t), intent(in), allocatable :: book1(:), book2(:)
  type(book_pb_t), intent(inout), allocatable :: book12(:)
  ! internal
  integer :: i, j, n1, n2, nset
  real(8), allocatable :: ls(:,:), set(:,:)
  type(book_pb_t), allocatable :: book_concat(:)
  integer, allocatable :: l2s(:)

  if(.not. allocated(book1)) &
   _die('.not. allocated(book1)')

  if(.not. allocated(book2)) &
   _die('.not. allocated(book2)')
 
  n1 = size(book1)
  n2 = size(book2)
  
  allocate(book_concat(n1+n2))
  j = 0
  do i=1, n1; j = j + 1; book_concat(j) = book1(i); enddo ! i
  do i=1, n2; j = j + 1; book_concat(j) = book2(i); enddo ! i

  call get_list_essent_metrics(book_concat, ls)
  call dlist2set(ls, nset, set, l2s)

  !write(6,*) 'nset', nset
  !write(6,*) l2s

  _dealloc(book12)
  allocate(book12(nset))
  do i=1, n1+n2; book12(l2s(i)) = book_concat(i); enddo ! i

  !_die('adjoin_book_set works ?')

end subroutine ! adjoin_book_set

!
!
!
subroutine constr_book_for_pair2clist(pb, pair2clist, book_mix, iv, ilog)
  use m_book_pb, only : book_pb_t, book_array1_t, print_info_book
  use m_prod_basis_type, only : prod_basis_t, init_global_counting3
  ! external
  type(prod_basis_t), intent(inout) :: pb
  type(book_array1_t), allocatable, intent(in) :: pair2clist(:)
  type(book_pb_t), allocatable, intent(inout) :: book_mix(:)
  integer, intent(in) :: iv, ilog
  ! internal
  integer :: pair, npairs, nc, i, nbook_ac_est, ic, tofc, cell(3), spp, ibook
  integer :: j, id_found, nbook_mix, step
  real(8), allocatable :: ind2id(:,:)
  real(8) :: coord(3), id_cur(9)

  !!
  !! Here we have to go over pairs, then over centres inside a pair and 
  !! create a global bookkeeping
  !!
  if(.not. allocated(pair2clist)) &
    _die('.not. allocated(pair2clist)')
  npairs = size(pair2clist)

  !! Count a rough (over)estimation of number of entries in new mixed bookkeeping
  nbook_ac_est = 0
  do pair=1,npairs
    if(.not. allocated(pair2clist(pair)%a)) &
      _die('not allocated pair2clist(pair)%a')
    nbook_ac_est = nbook_ac_est + size(pair2clist(pair)%a)
  enddo ! pair
  !! END of Count a rough (over)estimation of number of entries in new mixed bookkeeping
  allocate(ind2id(9,nbook_ac_est))
!  write(6,*)  'nbook_ac_est>>>>', nbook_ac_est
  !! Construct book_mix
  nbook_mix = 0
  do step=1,2
    ibook = 0
    do pair=1,npairs
      nc = size(pair2clist(pair)%a)
      do i=1,nc
        ic = pair2clist(pair)%a(i)%ic
        cell = pair2clist(pair)%a(i)%cells(:,3)
        tofc = pair2clist(pair)%a(i)%top
        spp = pair2clist(pair)%a(i)%spp
        coord = pair2clist(pair)%a(i)%coord
        id_cur = (/1D0*ic, 1D0*spp, 1D0*tofc, 1d0*cell, coord /)
        !! Search in already found ids
        id_found = -1
        do j=1,ibook
          if(any(abs(id_cur-ind2id(:,j))>1d-14)) cycle
          id_found = j; exit;
        enddo
        !! END of Search in already found ids
        if(id_found>0) cycle
        ibook = ibook + 1
        ind2id(:,ibook) = id_cur
        if(step==2) then
          book_mix(ibook) = pair2clist(pair)%a(i)
          book_mix(ibook)%atoms = 0
        endif
    
        if(iv>0)write(ilog,'(i5,i3,2x,i5,2x,3i4,3x,1i2,1i4,2x,i6,2x,i6)') &
          pair, i, ic, cell, tofc, spp, ibook, id_found
      enddo ! i
    enddo ! pair
    if(step==1) then
      nbook_mix = ibook
      allocate(book_mix(nbook_mix))
    endif
  enddo ! step
  !! END of Construct book_mix

!  write(6,*) "nbook_mix>>>", nbook_mix
  call init_global_counting3(pb, book_mix)

  if(iv>0)call print_info_book(ilog, book_mix)
  !  _die("check book_mix")

end subroutine ! constr_book_for_pair2clist


!
!
!
subroutine init_inverse_ls_blocks(ls_blocks, iblocks_pack2block)
  use m_book_pb, only : book_pb_t
  implicit none
  type(book_pb_t), intent(in), allocatable :: ls_blocks(:,:)
  integer, intent(inout), allocatable :: iblocks_pack2block(:)
  !! internal
  integer :: i, nblocks, ic1, ic2, indx, nsize, icmx

  icmx = maxval(ls_blocks(:,:)%ic)
  
  ! Initialize iblocks_pack2block -- indexing array
  _dealloc(iblocks_pack2block)
  nsize = _pack_size(icmx)
  allocate(iblocks_pack2block(nsize))
  iblocks_pack2block = -999
  nblocks = size(ls_blocks,2)
  do i=1,nblocks
    ic1 = ls_blocks(1,i)%ic; ic2 = ls_blocks(2,i)%ic
    if(ic1>ic2) _die('ic1>ic2')
    indx = _pack_indx(ic1, ic2)
    if (indx>nsize) then
      write(0,*) ic1, ic2, nsize
      write(0,*) 'maybe I need large integers by default... (-i8 during compilation)'
      _die('indx>nsize')
    endif  
    iblocks_pack2block(indx) = i
  enddo ! iblock
  ! END of Initialize iblocks_pack2block -- indexing array

end subroutine ! init_inverse_ls_blocks

!
!
!
subroutine cond_check_hkernel_blocks(aux, pb, ls_blocks, block2hkernel, para, iv, ilog)
#define _sname 'cond_check_hkernel_blocks'
  use m_pack_matrix, only : get_block
  use m_arrays, only : d_array2_t
  use m_book_pb, only : book_pb_t
  use m_prod_basis_param, only : prod_basis_param_t
  use m_prod_basis_type, only : prod_basis_t
  use m_parallel, only : para_t
  use m_pb_hkernel_pack8, only : alloc_comp_kernel_pack_pb
  use m_log, only : log_size_note
  implicit none
  !! external
  type(prod_basis_param_t), intent(in) :: aux  
  type(prod_basis_t), intent(in) :: pb
  type(book_pb_t), allocatable, intent(in) :: ls_blocks(:,:)
  type(d_array2_t), allocatable, intent(in) :: block2hkernel(:)
  type(para_t), intent(in) :: para
  integer, intent(in) :: iv, ilog
  !! internal
  real(8), allocatable :: vC_pack(:), vC_block(:,:)
  integer :: s1,f1,s2,f2, nblocks,block
  real(8) :: diff
  if(aux%check_hkernel_blocks<1) return
  if(.not. allocated(ls_blocks)) _die('.not. allocated(ls_blocks)')
  if(.not. allocated(block2hkernel)) _die('.not. allocated(block2hkernel)')
  
  call alloc_comp_kernel_pack_pb(pb, "HARTREE", 0D0, 1, vC_pack, para, iv, ilog)
  
  nblocks = size(ls_blocks,2)
  if(iv>0)write(ilog,'(a8,2a20,a12,2x,a12)') 'block', 'abs(diff(block))', 'sum(abs(block))', 's1 f1','s2 f2'
  do block=1,nblocks
    s1 = ls_blocks(1,block)%si(3); f1 = ls_blocks(1,block)%fi(3)
    s2 = ls_blocks(2,block)%si(3); f2 = ls_blocks(2,block)%fi(3)
    _dealloc(vC_block)
    allocate(vC_block(f1-s1+1,f2-s2+1))
    call get_block(vC_pack, s1, f1, s2, f2, vC_block, f1-s1+1)
    
    diff = sum(abs(block2hkernel(block)%array - vC_block))
    if(iv>0)write(ilog,'(i8,2e20.12,2i6,2x,2i6)') block, diff, sum(abs(vC_block)), s1,f1,s2,f2
    if(diff>1d-10) call die(_sname//': diff > 1D-10')
  enddo ! block
  
  call log_size_note(_sname, ': passed!', iv)
  
#undef _sname  
end subroutine ! cond_check_hkernel_blocks    
  
  
!
! This constructs a list of blocks for computing Hartree kernel
!   -- all local-local blocks are included;
!   -- all bilocal-bilocal blocks are excluded;
!   -- some necessary local-bilocal blocks are included.
!
subroutine ls_blocks_from_clist(book, pair2clist, ls_blocks)
  use m_book_pb, only : book_pb_t, book_array1_t, print_info_book
  use m_book_pb, only : get_pair_type_book
  use m_arrays, only : i_array1_t
  implicit none 
  !! external
  type(book_pb_t), allocatable, intent(in) :: book(:)
  type(book_array1_t), allocatable, intent(in) :: pair2clist(:)
  type(book_pb_t), allocatable, intent(inout) :: ls_blocks(:,:) ! 1:2, index_of_block
  !! internal
  integer :: npairs, ind, nind, pair, nbook, nblocks, s
  integer :: ind1, ind2, ic1, ic2, pt, nmax_ls_aux, nloc, nbiloc, ml
  !type(book_pb_t), allocatable :: ls_aux(:,:)
  type(i_array1_t), allocatable :: ic2ls_conn_ic(:)
  integer, allocatable :: itmp(:)
  integer :: ncmx_pp     ! number of centres maximal per pair

  if(.not. allocated(pair2clist)) &
   _die('.not. allocated(pair2clist)')

  if(.not. allocated(book)) &
   _die('.not. allocated(book)')

  npairs = size(pair2clist)
  nbook = size(book)

  !! Determine maximal number of centres per pair in pair2clist, also check fields of pair2clist
  ncmx_pp = 0
  do pair=1,npairs
    if(.not. allocated(pair2clist(pair)%a)) then
      write(6,'(a20,i8)') 'pair', pair
      _die('.not. allocated(pair2clist(pair)%a)')
    endif
    nind = size(pair2clist(pair)%a)
    ncmx_pp = max(ncmx_pp, nind)
    if(book(pair)%ic/=pair) _die('consistency check fails')
  enddo ! pair
  !! END of Determine maximal number of centres per pair in pair2clist, also check fields of pair2clist

  !! Determine maximal number of entries for ls_aux
  nloc = 0; 
  nbiloc = 0
  do ind=1, nbook
    pt = get_pair_type_book(book(ind))
    if(pt==1) then
      nloc = nloc + 1
    else if(pt==2) then
      nbiloc = nbiloc + 1
    else
      _die('unknown pair_type')
    endif
  enddo 
  nmax_ls_aux = nloc*(nloc+1)/2 + nbiloc*nloc
  !! END of Determine maximal number of entries for ls_aux

  !write(6,*) nbook, nbook*(nbook+1)/2, nmax_ls_aux
  !! Because bilocal-bilocal will be excluded, nblocks will be less than maximally possible nblocksmx
  !allocate(ls_aux(2,nmax_ls_aux)) not necessary

  call get_nblocks(book, pair2clist, npairs, ncmx_pp, nbook, nblocks)

  !! Allocate and initialize the fast-search index array ic2ls_conn_ic()
  allocate(ic2ls_conn_ic(nbook)) !! <<<<== I don't know first dimension, => check
  !! END of Allocate and initialize the fast-search index array ic2ls_conn_ic()
  
  allocate(itmp(nbook))
  
  _dealloc(ls_blocks)
  allocate(ls_blocks(2,nblocks))
  nblocks = 0
  !! Add reexpressing-reexpressing blocks
  do pair=1,npairs
    nind = size(pair2clist(pair)%a)
    do ind2=1,nind; ic2 = pair2clist(pair)%a(ind2)%ic
      if(ic2>nbook .or. ic2<1) _die('ic2>nbook .or. ic2<1')
      if( .not. allocated(ic2ls_conn_ic(ic2)%array) ) then
        allocate(ic2ls_conn_ic(ic2)%array(ncmx_pp))
        ic2ls_conn_ic(ic2)%array = 0
      endif
      
      do ind1=1,nind; ic1 = pair2clist(pair)%a(ind1)%ic
        if(ic1>ic2) cycle

        if( all(ic2ls_conn_ic(ic2)%array-ic1/=0) ) then ! i.e. if not found such pair
          nblocks = nblocks + 1
          if(ic1 < ic2 ) then
            ls_blocks(1,nblocks) = pair2clist(pair)%a(ind1)
            ls_blocks(2,nblocks) = pair2clist(pair)%a(ind2)
          else
            ls_blocks(1,nblocks) = pair2clist(pair)%a(ind2)
            ls_blocks(2,nblocks) = pair2clist(pair)%a(ind1)
          endif
          ml = minloc(ic2ls_conn_ic(ic2)%array,1)
          if (ic2ls_conn_ic(ic2)%array(ml)/=0) then
            s = size(ic2ls_conn_ic(ic2)%array)
            if(s>nbook) _die('s>nbook')
            itmp(1:s) = ic2ls_conn_ic(ic2)%array
            deallocate(ic2ls_conn_ic(ic2)%array)
            allocate(ic2ls_conn_ic(ic2)%array(2*s))
            ic2ls_conn_ic(ic2)%array = 0
            ic2ls_conn_ic(ic2)%array(1:s) = itmp(1:s)
            ml = minloc(ic2ls_conn_ic(ic2)%array,1)
            !write(6,*) 'reallocation happens', s,ic2, __LINE__
          endif !
          ic2ls_conn_ic(ic2)%array(ml) =  ic1
        endif ! all(ic2ls_conn_ic(:,ic2)-ic1/=0)
      enddo ! ind1
    enddo !ind2
  enddo ! pair
  !! END of Add reexpressing--reexpressing blocks

  !! Add necessary reexpressing-reexpressed-to-be blocks
  do pair=1,npairs 
    nind = size(pair2clist(pair)%a)
    ic2 = book(pair)%ic
    if(ic2>nbook .or. ic2<1) _die('ic2>nbook .or. ic2<1')
    if( .not. allocated(ic2ls_conn_ic(ic2)%array) ) then
      allocate(ic2ls_conn_ic(ic2)%array(ncmx_pp))
      ic2ls_conn_ic(ic2)%array = 0
    endif
    
    do ind=1,nind
      ic1 = pair2clist(pair)%a(ind)%ic
      
      if( all(ic2ls_conn_ic(ic2)%array-ic1/=0) ) then ! i.e. if not found such pair 
        nblocks = nblocks + 1
        if(ic1 < ic2 ) then
          ls_blocks(1,nblocks) = pair2clist(pair)%a(ind)
          ls_blocks(2,nblocks) = book(pair)
        else
          ls_blocks(1,nblocks) = book(pair)
          ls_blocks(2,nblocks) = pair2clist(pair)%a(ind)
        endif
        ml = minloc(ic2ls_conn_ic(ic2)%array,1)
        if (ic2ls_conn_ic(ic2)%array(ml)/=0) then
          s = size(ic2ls_conn_ic(ic2)%array)
          if(s>nbook) _die('s>nbook')
          itmp(1:s) = ic2ls_conn_ic(ic2)%array
          deallocate(ic2ls_conn_ic(ic2)%array)
          allocate(ic2ls_conn_ic(ic2)%array(2*s))
          ic2ls_conn_ic(ic2)%array = 0
          ic2ls_conn_ic(ic2)%array(1:s) = itmp(1:s)
          ml = minloc(ic2ls_conn_ic(ic2)%array,1)
          !write(6,*) 'reallocation happens', s, __LINE__
        endif
        ic2ls_conn_ic(ic2)%array(ml) =  ic1
      endif  ! all(ic2ls_conn_ic(:,ic2)-ic1/=0)
    enddo ! ic_loc
  enddo ! pair
  !! END of Add necessary reexpressing-reexpressed-to-be blocks
  
!  _dealloc(ls_blocks)
!  allocate(ls_blocks(2,nblocks))
!  ls_blocks(:,1:nblocks) = ls_aux(:,1:nblocks)

end subroutine !ls_blocks_from_clist

!
!
!
subroutine get_nblocks(book, pair2clist, npairs, ncmx_pp, nbook, nblocks)
  use m_book_pb, only : book_pb_t, book_array1_t, print_info_book
  use m_book_pb, only : get_pair_type_book
  use m_arrays, only : i_array1_t
  implicit none 
  !! external
  type(book_pb_t), allocatable, intent(in) :: book(:)
  type(book_array1_t), allocatable, intent(in) :: pair2clist(:)
  integer, intent(in) :: ncmx_pp, nbook, npairs
  integer, intent(inout) :: nblocks
  !! internal
  integer :: ind, nind, pair, s
  integer :: ind1, ind2, ic1, ic2, ml
  !type(book_pb_t), allocatable :: ls_aux(:,:)
  type(i_array1_t), allocatable :: ic2ls_conn_ic(:)
  integer, allocatable :: itmp(:)


  !! Allocate and initialize the fast-search index array ic2ls_conn_ic()
  allocate(ic2ls_conn_ic(nbook)) !! <<<<== I don't know first dimension, => check
  !! END of Allocate and initialize the fast-search index array ic2ls_conn_ic()
  
  allocate(itmp(nbook))
  
  nblocks = 0
  !count nblocks
  !! Add reexpressing-reexpressing blocks
  do pair=1,npairs
    nind = size(pair2clist(pair)%a)
    do ind2=1,nind; ic2 = pair2clist(pair)%a(ind2)%ic
      if(ic2>nbook .or. ic2<1) _die('ic2>nbook .or. ic2<1')
      if( .not. allocated(ic2ls_conn_ic(ic2)%array) ) then
        allocate(ic2ls_conn_ic(ic2)%array(ncmx_pp))
        ic2ls_conn_ic(ic2)%array = 0
      endif
      
      do ind1=1,nind; ic1 = pair2clist(pair)%a(ind1)%ic
        if(ic1>ic2) cycle

        if( all(ic2ls_conn_ic(ic2)%array-ic1/=0) ) then ! i.e. if not found such pair
          nblocks = nblocks + 1
          !if(ic1 < ic2 ) then
          !  ls_aux(1,nblocks) = pair2clist(pair)%a(ind1)
          !  ls_aux(2,nblocks) = pair2clist(pair)%a(ind2)
          !else
          !  ls_aux(1,nblocks) = pair2clist(pair)%a(ind2)
          !  ls_aux(2,nblocks) = pair2clist(pair)%a(ind1)
          !endif
          ml = minloc(ic2ls_conn_ic(ic2)%array,1)
          if (ic2ls_conn_ic(ic2)%array(ml)/=0) then
            s = size(ic2ls_conn_ic(ic2)%array)
            if(s>nbook) _die('s>nbook')
            itmp(1:s) = ic2ls_conn_ic(ic2)%array
            deallocate(ic2ls_conn_ic(ic2)%array)
            allocate(ic2ls_conn_ic(ic2)%array(2*s))
            ic2ls_conn_ic(ic2)%array = 0
            ic2ls_conn_ic(ic2)%array(1:s) = itmp(1:s)
            ml = minloc(ic2ls_conn_ic(ic2)%array,1)
            !write(6,*) 'reallocation happens', s,ic2, __LINE__
          endif !
          ic2ls_conn_ic(ic2)%array(ml) =  ic1
        endif ! all(ic2ls_conn_ic(:,ic2)-ic1/=0)
      enddo ! ind1
    enddo !ind2
  enddo ! pair
  !! END of Add reexpressing--reexpressing blocks

  !! Add necessary reexpressing-reexpressed-to-be blocks
  do pair=1,npairs 
    nind = size(pair2clist(pair)%a)
    ic2 = book(pair)%ic
    if(ic2>nbook .or. ic2<1) _die('ic2>nbook .or. ic2<1')
    if( .not. allocated(ic2ls_conn_ic(ic2)%array) ) then
      allocate(ic2ls_conn_ic(ic2)%array(ncmx_pp))
      ic2ls_conn_ic(ic2)%array = 0
    endif
    
    do ind=1,nind
      ic1 = pair2clist(pair)%a(ind)%ic
      
      if( all(ic2ls_conn_ic(ic2)%array-ic1/=0) ) then ! i.e. if not found such pair 
        nblocks = nblocks + 1
        !if(ic1 < ic2 ) then
        !  ls_aux(1,nblocks) = pair2clist(pair)%a(ind)
        !  ls_aux(2,nblocks) = book(pair)
        !else
        !  ls_aux(1,nblocks) = book(pair)
        !  ls_aux(2,nblocks) = pair2clist(pair)%a(ind)
        !endif
        ml = minloc(ic2ls_conn_ic(ic2)%array,1)
        if (ic2ls_conn_ic(ic2)%array(ml)/=0) then
          s = size(ic2ls_conn_ic(ic2)%array)
          if(s>nbook) _die('s>nbook')
          itmp(1:s) = ic2ls_conn_ic(ic2)%array
          deallocate(ic2ls_conn_ic(ic2)%array)
          allocate(ic2ls_conn_ic(ic2)%array(2*s))
          ic2ls_conn_ic(ic2)%array = 0
          ic2ls_conn_ic(ic2)%array(1:s) = itmp(1:s)
          ml = minloc(ic2ls_conn_ic(ic2)%array,1)
          !write(6,*) 'reallocation happens', s, __LINE__
        endif
        ic2ls_conn_ic(ic2)%array(ml) =  ic1
      endif  ! all(ic2ls_conn_ic(:,ic2)-ic1/=0)
    enddo ! ic_loc
  enddo ! pair
  !! END of Add necessary reexpressing-reexpressed-to-be blocks
  !end count nblocks
 
end subroutine ! get_nblocks

end module !m_pb_reexpr_comm
