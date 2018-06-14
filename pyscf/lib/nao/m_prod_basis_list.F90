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

module m_prod_basis_list
! The purpose of the module is to store and deal with a real space information of a product basis
#include "m_define_macro.F90"
  use m_die, only : die
  
  implicit none
  private die

contains

!
! Construct pair2c[enter]_list structure
!
subroutine constr_pair2clist(sv, pb_p, book, pair2clist)

  use m_prod_basis_param, only : prod_basis_param_t
  use m_system_vars, only: system_vars_t,get_basis_type,get_sp2rcut
  use m_log, only: log_size_note
  use m_book_pb, only : book_pb_t, book_array1_t

  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  type(prod_basis_param_t), intent(in) :: pb_p
  type(book_pb_t), intent(in), allocatable :: book(:)
  type(book_array1_t), intent(inout), allocatable :: pair2clist(:)
  !! internal
  integer :: bt, pair, npairs, imm(2,3)=-999
  real(8) :: ac_rcut
  real(8), allocatable :: sp2rcut(:)
  
  if(.not. allocated(book)) _die('.not. allocated(book)')
  npairs = size(book)
  
  _dealloc(pair2clist)
  allocate(pair2clist(npairs))
  
  ac_rcut = pb_p%ac_rcut
  call get_sp2rcut(sv, sp2rcut)
  
  bt = get_basis_type(sv)
  select case(bt)
  case(1)
    imm = 0
    do pair=1,npairs    
      call constr_clist_fini(sv, book(pair), pb_p, sp2rcut, pair2clist(pair)%a)
    enddo
   
  case(2)
      
    imm(1,1:3) = -7;
    imm(2,1:3) = +7;
    
    do pair=1,npairs
      call constr_clist_bulk(sv, imm, book(pair), ac_rcut, pair2clist(pair)%a)  
    enddo ! pair
    
  case default
    _die('unknown basis type')
  end select !
  
  _dealloc(sp2rcut)
  
  
end subroutine ! constr_pair2clist

!
! Construct a list of participating centers for a single atom pair given by   bk 
!
subroutine constr_clist_fini(sv, book, pb_p, sp2rcut, a)
  use m_system_vars, only : system_vars_t, get_atom2sp_ptr
  use m_book_pb, only : book_pb_t
  use m_prod_basis_param, only : prod_basis_param_t
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  type(book_pb_t), intent(in) :: book
  type(prod_basis_param_t), intent(in) :: pb_p
  real(8), intent(in) :: sp2rcut(:)
  type(book_pb_t), intent(inout), allocatable :: a(:)
  !! internal
  real(8) :: dist_min, coord3(3), dist, dist12
  real(8) :: coord1(3), coord2(3), coord(3), ac_rcut
  real(8) :: rcuts(2), dists(2)
  integer :: icenter_loc, step, atom, natoms, sps(2)
  integer, pointer :: atom2sp(:) => null()
  
  _dealloc(a)
  if(book%top<1 .or. book%top>2) return
  
  atom2sp => get_atom2sp_ptr(sv)
  natoms = sv%natoms
  coord1 = sv%atom_sc2coord(:,book%atoms(1))
  coord2 = sv%atom_sc2coord(:,book%atoms(2))
  dist12 = sqrt(sum((coord2-coord1)**2))
  coord  = book%coord
  ac_rcut = pb_p%ac_rcut
 
  sps = atom2sp(book%atoms)
  rcuts = sp2rcut(sps)
     
  do step=1,2
    icenter_loc = 0
    dist_min = 1.0D200
    do atom=1,natoms
      coord3 = sv%atom_sc2coord(:,atom)
      dist = sqrt(sum((coord3-coord)**2))
      dist_min = min(dist_min, dist)
      ! must be an option, but which would be the name? 
      !! if the atom pair is local, then no other centers are necessary...
      if(dist12<1D-5 .and. dist>1D-5) cycle
      ! END of must be an option, but which would be the name?
      select case(pb_p%ac_method)
      case('SPHERE')
        if(dist>ac_rcut) cycle
      case('LENS')
        dists(1) = sqrt(sum((coord3-coord1)**2))
        dists(2) = sqrt(sum((coord3-coord2)**2))
        if(all(dists>rcuts)) cycle
      case default 
        write(6,*) __FILE__, __LINE__, trim(pb_p%ac_method)
        _die('!ac_method')
      end select      
      icenter_loc = icenter_loc + 1
      if(step==2)then
        a(icenter_loc)            = book
        a(icenter_loc)%cells      = 0
        a(icenter_loc)%coord      = coord3
        a(icenter_loc)%spp        = sv%uc%atom2sp(atom)
        a(icenter_loc)%top        = 1
        a(icenter_loc)%ic         = atom ! This must be a local pair index, but atom should be ok with current conventions
        a(icenter_loc)%si(3)      = -998 !
        a(icenter_loc)%fi(3)      = -999 !
      endif ! step 
    enddo ! atom
      
    if(icenter_loc<1) then
        
      write(0,*) __FILE__, __LINE__, ': no centres found for:'
      write(0,'(a20,3i5)') 'atoms(1:2)', book%atoms
      write(0,'(a20,3i4,2x,3i4,2x,3i4)') 'cells', book%cells
      write(0,'(a20,3f11.6)') 'coord1', coord1
      write(0,'(a20,3f11.6)') 'coord2', coord2
      write(0,'(a20,3f11.6)') 'coord ', coord
      write(0,'(a20,3f11.6)') 'distance(2-1)', sqrt(sum((coord2-coord1)**2))
      write(0,'(a20,f11.6)') 'ac_rcut', ac_rcut
      write(0,'(a20,g14.6)') 'min acv_rcut_prd ', dist_min
        
      write(0,*) ': icenter_loc ', icenter_loc
      _die(': icenter_loc<1')
    endif

    if(step==1) then
      allocate(a(icenter_loc))
!#ifdef DEBUG
!      write(0,'(a,2i5,3x,i6)') 'atom1, atom2, icenter_loc', book%atoms, icenter_loc
!#endif
    endif
    
  enddo ! step
  
end subroutine ! constr_clist_fini

!
! Construct a list of participating centers for a single atom pair given by   bk 
!
subroutine constr_clist_bulk(sv, imm, book, ac_rcut, a)
  use m_system_vars, only : system_vars_t, get_uc_vecs_ptr
  use m_book_pb, only : book_pb_t
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  integer, intent(in) :: imm(:,:)
  type(book_pb_t), intent(in) :: book
  real(8), intent(in) :: ac_rcut
  type(book_pb_t), intent(inout), allocatable :: a(:)
  !! internal
  real(8) :: dist_min, coord3_uc(3), coord3(3), dist, dist12
  real(8) :: svec(3), svecs(3,3), coord1(3), coord2(3), coord(3)  
  integer :: i1, i2, i3, icenter_loc, step, cell(3), atom, natoms
  real(8), pointer :: uc_vecs(:,:)
  
  uc_vecs => get_uc_vecs_ptr(sv)
  natoms  = sv%natoms
  
  svecs = matmul(uc_vecs, book%cells)
  coord1 = sv%atom_sc2coord(:,book%atoms(1)) + svecs(:,1)
  coord2 = sv%atom_sc2coord(:,book%atoms(2)) + svecs(:,2)
  dist12 = sqrt(sum((coord2-coord1)**2))
  coord  = book%coord + svecs(:,3)
     
  do step=1,2
    icenter_loc = 0
    dist_min = 1.0D200
    do i1=imm(1,1),imm(2,1)
      do i2=imm(1,2),imm(2,2)
        do i3=imm(1,3),imm(2,3)
          cell = (/i1,i2,i3/)
          svec = matmul(uc_vecs, cell)
          do atom=1,natoms
            coord3_uc = sv%atom_sc2coord(:,atom)
            coord3 = coord3_uc + svec
            dist = sqrt(sum((coord3-coord)**2))
            dist_min = min(dist_min, dist)
            ! must be an option, but which would be the name? 
            !! if the atom pair is local, then no other centers are necessary...
            if(dist12<1D-5 .and. dist>1D-5) cycle
            ! END of must be an option, but which would be the name?
            if(dist>ac_rcut) cycle
            icenter_loc = icenter_loc + 1
            if(step==2)then
              a(icenter_loc)            = book
              a(icenter_loc)%cells(:,1) = cell
              a(icenter_loc)%cells(:,2) = cell 
              a(icenter_loc)%cells(:,3) = cell
              a(icenter_loc)%coord      = coord3_uc
              a(icenter_loc)%spp        = sv%uc%atom2sp(atom)
              a(icenter_loc)%top        = 1
              a(icenter_loc)%ic         = atom ! This must be a local pair index, but atom should be ok with current conventions
              a(icenter_loc)%si(3)      = -999
              a(icenter_loc)%fi(3)      = -999
            endif ! step 
          enddo ! atom
        enddo !i3
      enddo ! i2
    enddo; ! i1
      
    if(icenter_loc<1) then
        
      write(0,*) __FILE__, __LINE__, ': no centres found for:'
      write(0,'(a20,3i5)') 'atoms(1:2)', book%atoms, natoms
      write(0,'(a20,3i4,2x,3i4,2x,3i4)') 'cells', book%cells
      write(0,'(a20,3f11.6)') 'coord1', coord1
      write(0,'(a20,3f11.6)') 'coord2', coord2
      write(0,'(a20,3f11.6)') 'distance(2-1)', sqrt(sum((coord2-coord1)**2))
      write(0,'(a20,f11.6)') 'ac_rcut', ac_rcut
      write(0,'(a20,g14.6)') 'min acv_rcut_prd ', dist_min
        
      write(0,*) ': icenter_loc ', icenter_loc
      _die(': icenter_loc<1')
    endif

    if(step==1) then
      allocate(a(icenter_loc))
#ifdef DEBUG
      write(0,'(a,2i5,3x,i6)') 'atom1, atom2, icenter_loc', book%atoms, icenter_loc
#endif
    endif
    
  enddo ! step
  
end subroutine ! constr_clist_bulk


!
!
!
subroutine gen_list_of_pairs_dp(sv, pb_p, pair2info, iv_in, ilog)
#define _sname 'gen_list_of_pairs_dp'
  use m_pair_info, only : pair_info_t, is_bilocal, is_local, cp_pair_info
  use m_sc_dmatrix, only : get_stored_npairs,sc_dmatrix_t,get_stored_atoms,get_stored_cells
  use m_system_vars, only : get_natoms, system_vars_t, get_rcuts, get_overlap_sc_ptr
  use m_prod_basis_param, only : prod_basis_param_t, get_bilocal_type
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  type(prod_basis_param_t), intent(in) :: pb_p
  type(pair_info_t), allocatable, intent(inout) ::  pair2info(:)
  integer, intent(in) :: iv_in, ilog
  !! internal
  character(100) :: ctype
  integer :: npairs, pair, iv, p
  type(pair_info_t), allocatable ::  pair2info_aux(:)

  iv = iv_in - 1

  ctype = get_bilocal_type(pb_p)
  select case(ctype)
  case('ATOM')
    call get_pair2info_atom(sv, pair2info_aux)
  case('MULT')
    call get_pair2info_mult(sv, pair2info_aux)
  case default
    write(6,'(a,2x,a)') 'bilocal_type: ', trim(ctype)
    _die('bilocal_type unknown')  
  endselect

  if(.not. allocated(pair2info_aux)) _die('!pair2info_aux')
  npairs = size(pair2info_aux)
 
  !! Sort the pairs that local pairs appear first and bilocals are sorted in atom2-first way 
  allocate(pair2info(npairs))
  pair = 0
  do p=1,npairs
    if(is_bilocal(pair2info_aux(p))) cycle
    pair = pair + 1
    call cp_pair_info( pair2info_aux(p), pair2info(pair) )
  enddo

  do p=1,npairs
    if(is_local(pair2info_aux(p))) cycle
    pair = pair + 1
    call cp_pair_info( pair2info_aux(p), pair2info(pair) )
  enddo
  !! END of Sort the pairs that local pairs appear first

  if(iv>0) then
    write(ilog,*) _sname//': print pair2info'
    do pair=1, size(pair2info)
      write(ilog,'(i5,3x,2i5,3x,2i5,3x,3f10.5,3x,3f10.5)') &
        pair, pair2info(pair)%atoms, pair2info(pair)%species, pair2info(pair)%coords
    end do
    write(ilog,*) _sname//': END of print pair2info'
  endif ! iv>0
  
  _dealloc(pair2info_aux)
  
#undef _sname 
end subroutine ! gen_list_of_pairs_dp

!
!
!
subroutine get_pair2info_atom(sv, p2i)
  use m_bilocal_vertex, only : pair_info_t
  use m_sc_dmatrix, only : get_stored_npairs,sc_dmatrix_t,get_stored_atoms,get_stored_cells
  use m_system_vars, only : get_uc_vecs_ptr, system_vars_t, get_rcuts, get_overlap_sc_ptr
  use m_system_vars, only : get_sp2nmult_ptr, get_atom2coord_ptr, get_atom2sp_ptr
  use m_prod_basis_param, only : prod_basis_param_t   
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  type(pair_info_t), intent(inout), allocatable :: p2i(:)
  !! internal
  integer :: npairs, ip, atoms(2), cells(3,2), rf, ls, nmu(2), turn, nstored_pairs, pair
  real(8) :: shifts(3,2), coords(3,2), dist, rcuts(2)
  type(sc_dmatrix_t), pointer :: sc_s
  integer, pointer :: sp2nmult(:), atom2sp(:)
  real(8), pointer :: atom2coord(:,:), uc_vecs(:,:)
  
  sp2nmult => get_sp2nmult_ptr(sv)
  atom2sp => get_atom2sp_ptr(sv)  
  uc_vecs => get_uc_vecs_ptr(sv)
  sc_s => get_overlap_sc_ptr(sv)
  nstored_pairs = get_stored_npairs(sc_s)
  atom2coord => get_atom2coord_ptr(sv)

  npairs = -999
  do turn=1,2
    pair = 0
    do ip=1,nstored_pairs
      atoms = get_stored_atoms(sc_s, ip)
      cells = int(get_stored_cells(sc_s, ip))
      shifts = matmul(uc_vecs(1:3,1:3), cells(1:3,1:2))
      coords = shifts + atom2coord(:,atoms)
      dist = sqrt(sum((coords(1:3,1)-coords(1:3,2))**2))
      rcuts = get_rcuts(sv, atoms)
      if(dist>sum(rcuts)) cycle
      pair = pair + 1
      if(turn==1) cycle
      
      p2i(pair)%atoms = atoms
      p2i(pair)%species = atom2sp(atoms)
      p2i(pair)%cells = cells
      p2i(pair)%coords = coords
      nmu(1:2) = sp2nmult(p2i(pair)%species(1:2))
      p2i(pair)%ls2nrf(1:2) = nmu(1:2)
      allocate(p2i(pair)%rf_ls2mu(maxval(nmu),2))
      p2i(pair)%rf_ls2mu = -1
      do ls=1,2
        do rf=1,nmu(ls); p2i(pair)%rf_ls2mu(rf, ls) = rf; enddo ! rf
      enddo ! ls
    enddo ! ip=1,nstored_pairs
    if(turn>1) cycle
    npairs = pair
    _dealloc(p2i)
    allocate(p2i(npairs))
  enddo ! turn=1,2
  
end subroutine ! get_pair2info_atom

!
!
!
subroutine get_pair2info_mult(sv, p2i)
  use m_bilocal_vertex, only : pair_info_t
  use m_sc_dmatrix, only : get_stored_npairs,sc_dmatrix_t,get_stored_atoms,get_stored_cells
  use m_system_vars, only : get_uc_vecs_ptr, system_vars_t, get_rcuts, get_overlap_sc_ptr
  use m_system_vars, only : get_sp2nmult_ptr, get_atom2coord_ptr, get_atom2sp_ptr
  use m_system_vars, only : get_mu_sp2rcut_ptr
  use m_prod_basis_param, only : prod_basis_param_t   
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  type(pair_info_t), intent(inout), allocatable :: p2i(:)
  !! internal
  integer :: npairs, ip, atoms(2), cells(3,2), rf, ls, nmu(2)
  integer :: turn, nstored_pairs, pair, mub, mua, sp(2)
  real(8) :: shifts(3,2), coords(3,2), dist, rcuts(2), dist_mult
  type(sc_dmatrix_t), pointer :: sc_s
  integer, pointer :: sp2nmult(:), atom2sp(:)
  real(8), pointer :: atom2coord(:,:), mu_sp2rcut(:,:), uc_vecs(:,:)
  logical :: lbilocal
  
  sp2nmult => get_sp2nmult_ptr(sv)
  uc_vecs => get_uc_vecs_ptr(sv)
  sc_s => get_overlap_sc_ptr(sv)
  nstored_pairs = get_stored_npairs(sc_s)
  atom2coord => get_atom2coord_ptr(sv)
  atom2sp => get_atom2sp_ptr(sv)
  mu_sp2rcut => get_mu_sp2rcut_ptr(sv)

  npairs = -999
  do turn=1,2
    pair = 0
    do ip=1,nstored_pairs
      atoms = get_stored_atoms(sc_s, ip)
      cells = int(get_stored_cells(sc_s, ip))
      sp(1:2) = atom2sp(atoms(1:2))
      shifts = matmul(uc_vecs(1:3,1:3), cells(1:3,1:2))
      coords = shifts + atom2coord(:,atoms)
      dist = sqrt(sum((coords(1:3,1)-coords(1:3,2))**2))
      rcuts = get_rcuts(sv, atoms)
      if(dist>sum(rcuts)) cycle
      
      lbilocal = (dist>1d-6 .or. atoms(1)/=atoms(2))
      
      if(lbilocal) then ! bilocal pairs must be generated for each couple of multipletts
        nmu(1:2) = sp2nmult(sp(1:2))
        do mub=1, nmu(2)
          do mua=1, nmu(1)
            dist_mult = mu_sp2rcut(mua,sp(1)) + mu_sp2rcut(mub,sp(2))
            if(dist_mult<=dist) cycle
          
            pair = pair + 1
            if(turn==1) cycle
            p2i(pair)%atoms = atoms
            p2i(pair)%species = sp(1:2)
            p2i(pair)%cells = cells
            p2i(pair)%coords = coords
            p2i(pair)%ls2nrf(1:2) = 1
            allocate(p2i(pair)%rf_ls2mu(1,2))
            p2i(pair)%rf_ls2mu(1,1:2) = [mua, mub]
          enddo 
        enddo !    
      else ! local pairs must be generated for each atom
        pair = pair + 1
        if(turn==1) cycle
        p2i(pair)%atoms = atoms
        p2i(pair)%species =  sp(1:2)
        p2i(pair)%cells = cells
        p2i(pair)%coords = coords
        p2i(pair)%ls2nrf(1:2) = nmu(1:2)
        allocate(p2i(pair)%rf_ls2mu( maxval(nmu), 2) )
        p2i(pair)%rf_ls2mu = -1
        do ls=1,2
          do rf=1,nmu(ls); p2i(pair)%rf_ls2mu(rf,ls) = rf; enddo ! rf
        enddo ! ls
      endif

    enddo ! ip=1,nstored_pairs
    if(turn>1) cycle
    npairs = pair
    _dealloc(p2i)
    allocate(p2i(npairs))
  enddo ! turn=1,2
  
end subroutine ! get_pair2info_mult

end module !m_prod_basis_list
