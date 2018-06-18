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

module m_constr_clist_sprc
! The purpose of the module is to store and deal with a real space information of a product basis
#include "m_define_macro.F90"
  use m_die, only : die
  
  implicit none
  private die

contains

!
! Construct a list of participating centers for a single atom pair given by specie pair and coordinate pair 
!
subroutine constr_clist_sprc(sv, sp12, rc12, pb_p, sp2rcut, a)
  use m_system_vars, only : system_vars_t, get_atom2sp_ptr, get_atom2coord_ptr
  use m_book_pb, only : book_pb_t
  use m_prod_basis_param, only : prod_basis_param_t
  use iso_c_binding, only : c_double, c_int64_t
  implicit none
  !! external
  type(system_vars_t), intent(in) :: sv
  integer, intent(in) :: sp12(:)
  real(c_double), intent(in) :: rc12(:,:)
  type(prod_basis_param_t), intent(in) :: pb_p
  real(8), intent(in) :: sp2rcut(:)
  type(book_pb_t), intent(inout), allocatable :: a(:)
  !! internal
  real(8) :: dist_min, coord3(3), dist, dist12
  real(8) :: coord1(3), coord2(3), coord(3), ac_rcut
  real(8) :: rcuts(2), dists(2)
  integer :: icenter_loc, step, atom, natoms
  integer, pointer :: atom2sp(:) => null()
  real(8), pointer :: atom2coord(:,:) => null()

  _dealloc(a)
  atom2sp => get_atom2sp_ptr(sv)
  atom2coord => get_atom2coord_ptr(sv)
  natoms = sv%natoms
  coord1 = rc12(1:3,1)
  coord2 = rc12(1:3,2)
  dist12 = sqrt(sum((coord2-coord1)**2))
  coord  = (coord2+coord1)/2.0d0
  ac_rcut = pb_p%ac_rcut
 
  !write(6,*) size(sp2rcut), sp12, pb_p%ac_method, natoms
    
  rcuts = sp2rcut(sp12)

  do step=1,2
    icenter_loc = 0
    dist_min = 1.0D200
    do atom=1,natoms
      coord3 = atom2coord(:,atom)
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
        a(icenter_loc)%atoms      = -999
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
      write(0,'(a20,3i5)') 'sp12(1:2)', sp12
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
  
end subroutine ! constr_clist_sprc

end module !m_constr_clist_sprc
