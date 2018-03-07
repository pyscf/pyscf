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

module m_functs_m_mult_type
!
! The purpose of the module is to store and deal with a functions in m-multipletts
!
#include "m_define_macro.F90"
  use m_die, only : die

  implicit none
  private die

  !!
  !! Descr of real space information of the bilocal products for an **atom** pair
  !! 
  type functs_m_mult_t
    real(8), allocatable :: ir_j_prd2v(:,:,:) ! collection of radial functions
    integer, allocatable :: prd2m(:) ! local product -> m (allocated for m-multipletts i.e. bilocal dp)
    real(8), allocatable :: crc(:,:) ! center within the atom pair -> coordinates/cutoff/start/finish/size counting (7 values)

    real(8) :: coords(1:3,1:2) =-999 ! two centers: function is given in a rotated frame (don't sum with shifts from cells, just use)
    real(8) :: rcuts(1:2) =-999      ! orbital cutoffs 
    integer :: species(2) =-999      ! Species of atoms
    integer :: atoms(2) =-999        ! Atoms (in the unit cell)
    integer :: cells(3,2) =-999      ! Only applies to field %atoms(1:2) (just above)
    integer :: nfunct =-999          ! number of functions in this specie
    integer :: nr =-999          ! size(ir_j_prd2v, 1)
    integer :: jmax =-999          ! ubound(ir_j_prd2v, 2)

  end type ! functs_m_mult_t
  !! END of Descr of real space information

  contains


!
!
!
subroutine gather(ff, mem)
  implicit none
  ! external
  type(functs_m_mult_t), intent(in) :: ff
  real(8), intent(inout), allocatable :: mem(:)
  ! internal
  integer :: f, p, n, lb(3), ub(3), i,j,k

  n = 3
  if(allocated(ff%ir_j_prd2v)) n = n + _size_mem(ff%ir_j_prd2v)
  if(allocated(ff%prd2m)) n = n + _size_mem(ff%prd2m)
  if(allocated(ff%crc)) n = n + _size_mem(ff%crc)
  n = n + _size_mem(ff%coords)
  n = n + _size_mem(ff%rcuts)
  n = n + _size_mem(ff%species)
  n = n + _size_mem(ff%atoms)
  n = n + _size_mem(ff%cells)
  n = n + 3

  if(allocated(mem)) then
    if(size(mem)<n) deallocate(mem)
  endif
  if(.not. allocated(mem)) allocate(mem(n))

  p = 1
  mem(p) = ff%nfunct
  if(ff%nfunct<1) return
  
  p = p + 1
  if(p>size(mem)) _die('!p>size(mem)')
  if(allocated(ff%ir_j_prd2v)) then
    mem(p) = 1
    _incr_sf(p,ff%ir_j_prd2v,f)
    if(f>size(mem)) _die('!f>size(mem)')
    _poke_lub(p,ff%ir_j_prd2v,mem)
!    _poke_dat(p,f,ff%ir_j_prd2v,mem)
    ub = ubound(ff%ir_j_prd2v)
    lb = lbound(ff%ir_j_prd2v)
    do k=lb(3),ub(3)
      do j=lb(2),ub(2)
        do i=lb(1),ub(1)
          mem(p) = ff%ir_j_prd2v(i,j,k)
          p = p + 1
        enddo
      enddo
    enddo
    p = f
  else
    mem(p) = 0;
  endif    
 
  p = p + 1
  if(p>size(mem)) _die('!p>size(mem)')
  if(allocated(ff%prd2m)) then
    mem(p) = 1
    _incr_sf(p,ff%prd2m,f)
    if(f>size(mem)) _die('!f>size(mem)')
    _poke_lub(p,ff%prd2m,mem)
!    _poke_dat(p,f,ff%prd2m,mem)
    ub(1:1) = ubound(ff%prd2m)
    lb(1:1) = lbound(ff%prd2m)
    do i=lb(1),ub(1)
      mem(p) = ff%prd2m(i)
      p = p + 1
    enddo
    p = f
  else
    mem(p) = 0;
  endif
 
  p = p + 1
  if(p>size(mem)) _die('!p>size(mem)')
  if(allocated(ff%crc)) then
    mem(p) = 1
    _incr_sf(p,ff%crc,f)
    if(f>size(mem)) _die('!f>size(mem)')
    _poke_lub(p,ff%crc,mem)
!    _poke_dat(p,f,ff%crc,mem)
    ub(1:2) = ubound(ff%crc)
    lb(1:2) = lbound(ff%crc)
    do j=lb(2),ub(2)
      do i=lb(1),ub(1)
        mem(p) = ff%crc(i,j)
        p = p + 1
      enddo
    enddo
    p = f

  else
    mem(p) = 0
  endif
   
  _incr_sf(p,ff%coords,f)
  if(f>size(mem)) _die('!f>size(mem)')
  _poke_lub(p,ff%coords,mem)
  _poke_dat(p,f,ff%coords,mem)
 
  _incr_sf(p,ff%rcuts,f)
  if(f>size(mem)) _die('!f>size(mem)')
  _poke_lub(p,ff%rcuts,mem)
  _poke_dat(p,f,ff%rcuts,mem)
 
  _incr_sf(p,ff%species,f)
  if(f>size(mem)) _die('!f>size(mem)')
  _poke_lub(p,ff%species,mem)
  _poke_dat(p,f,ff%species,mem)
 
  _incr_sf(p,ff%atoms,f)
  if(f>size(mem)) _die('!f>size(mem)')
  _poke_lub(p,ff%atoms,mem)
  _poke_dat(p,f,ff%atoms,mem)
 
  _incr_sf(p,ff%cells,f)
  if(f>size(mem)) _die('!f>size(mem)')
  _poke_lub(p,ff%cells,mem)
  _poke_dat(p,f,ff%cells,mem)
   
  p = p + 1
  if(p+1>size(mem)) _die('!p>size(mem)')
  mem(p:p+1) = [ff%nr, ff%jmax]
 

end subroutine ! gather  

!
!
!
subroutine scatter(mem, ff)
  implicit none
  ! external
  real(8), intent(in) :: mem(:)
  type(functs_m_mult_t), intent(inout) :: ff
  ! internal
  integer :: f, p, d, nn(3), lb(3), ub(3), i,j,k
  
  p = 1
  if(p>size(mem)) _die('!p>size(mem)')
  if(mem(p)<1) then
    call dealloc(ff)
    ff%nfunct = mem(p)
    return
  else
    ff%nfunct = mem(p)  
  endif
  
  p = p + 1
  if(p>size(mem)) _die('!p>size(mem)')
  if(mem(p)==1) then
    d = 3
    p = p + 1
    f = p + 2 * d - 1
    if(f>size(mem)) _die('!f>size(mem)')
    _get_lbubnn(p,d,mem,lb,ub,nn)
    if(any(nn(1:d)<1)) _die('!nn<1')
    if(lb(1)/=1)_die('!lb1')
    if(lb(2)/=0)_die('!lb2')
    if(lb(3)/=1)_die('!lb3')
    _dealloc(ff%ir_j_prd2v)
    allocate(ff%ir_j_prd2v(1:ub(1),0:ub(2),1:ub(3)))
    _incr_sf_nn(p,d,nn,f)
    if(f>size(mem)) _die('!f>size(mem)')
!    ff%ir_j_prd2v = reshape(mem(p:f), nn(1:3))
    do k=lb(3),ub(3)
      do j=lb(2),ub(2)
        do i=lb(1),ub(1)
          ff%ir_j_prd2v(i,j,k) = mem(p)
          p = p + 1
        enddo
      enddo
    enddo
    p = f
  endif  

  p = p + 1
  if(p>size(mem)) _die('!p>size(mem)')
  if(mem(p)==1) then
    d = 1
    p = p + 1
    f = p + 2 * d - 1
    if(f>size(mem)) _die('!f>size(mem)')
    _get_lbubnn(p,d,mem,lb,ub,nn)
    if(any(nn(1:d)<1)) _die('!nn<1')
    if(lb(1)/=1)_die('!lb1')
    _dealloc(ff%prd2m)
    allocate(ff%prd2m(1:ub(1)))
    _incr_sf_nn(p,d,nn,f)
    if(f>size(mem)) _die('!f>size(mem)')
!    ff%prd2m = int(reshape(mem(p:f), nn(1:1)))
    do i=lb(1),ub(1)
      ff%prd2m(i) = mem(p)
      p = p + 1
    enddo
    p = f
  endif  

  p = p + 1
  if(p>size(mem)) _die('!p>size(mem)')
  if(mem(p)==1) then
    d = 2
    p = p + 1
    f = p + 2 * d - 1
    if(f>size(mem)) _die('!f>size(mem)')
    _get_lbubnn(p,d,mem,lb,ub,nn)
    if(any(nn(1:d)<1)) _die('!nn<1')
    if(any(lb(1:d)/=1)) _die('!lb/=1')
    if(any(ub(1:d)<1)) _die('!ub<1')
    _dealloc(ff%crc)
    allocate(ff%crc(1:ub(1), 1:ub(2)))
    _incr_sf_nn(p,d,nn,f)
    if(f>size(mem)) _die('!f>size(mem)')
    !ff%crc = reshape(mem(p:f), nn(1:2))
    do j=lb(2),ub(2)
      do i=lb(1),ub(1)
        ff%crc(i,j) = mem(p)
        p = p + 1
      enddo
    enddo
    p = f
  endif  

  d = 2
  p = p + 1
  f = p + 2 * d - 1
  if(f>size(mem)) _die('!f>size(mem)')
  _get_lbubnn(p,d,mem,lb,ub,nn)
  if(any(nn(1:d)<1)) _die('!nn<1')
  if(any(lb(1:d)/=1)) _die('!lb/=1')
  if(any(ub(1:d)/=[3,2])) _die('!ub/=3,2')
  _incr_sf_nn(p,d,nn,f)
  if(f>size(mem)) _die('!f>size(mem)')
  ff%coords = reshape(mem(p:f), nn(1:2))
  p = f

  d = 1
  p = p + 1
  f = p + 2 * d - 1
  if(f>size(mem)) _die('!f>size(mem)')
  _get_lbubnn(p,d,mem,lb,ub,nn)
  if(any(nn(1:d)<1)) _die('!nn<1')
  if(any(lb(1:d)/=1)) _die('!lb/=1')
  if(any(ub(1:d)/=2)) _die('!ub/=2')
  _incr_sf_nn(p,d,nn,f)
  if(f>size(mem)) _die('!f>size(mem)')
  ff%rcuts = reshape(mem(p:f), nn(1:1))
  p = f

  d = 1; p = p + 1; f = p + 2 * d - 1
  if(f>size(mem)) _die('!f>size(mem)')
  _get_lbubnn(p,d,mem,lb,ub,nn)
  if(any(nn(1:d)<1)) _die('!nn<1')
  if(any(lb(1:d)/=1)) _die('!lb/=1')
  if(any(ub(1:d)/=2)) _die('!ub/=2')
  _incr_sf_nn(p,d,nn,f)
  if(f>size(mem)) _die('!f>size(mem)')
  ff%species = int(reshape(mem(p:f), nn(1:1)))
  p = f

  d = 1; p = p + 1; f = p + 2 * d - 1
  if(f>size(mem)) _die('!f>size(mem)')
  _get_lbubnn(p,d,mem,lb,ub,nn)
  if(any(nn(1:d)<1)) _die('!nn<1')
  if(any(lb(1:d)/=1)) _die('!lb/=1')
  if(any(ub(1:d)/=2)) _die('!ub/=2')
  _incr_sf_nn(p,d,nn,f)
  if(f>size(mem)) _die('!f>size(mem)')
  ff%atoms = int(reshape(mem(p:f), nn(1:1)))
  p = f

  d = 2; p = p + 1; f = p + 2 * d - 1
  if(f>size(mem)) _die('!f>size(mem)')
  _get_lbubnn(p,d,mem,lb,ub,nn)
  if(any(nn(1:d)<1)) _die('!nn<1')
  if(any(lb(1:d)/=1)) _die('!lb/=1')
  if(any(ub(1:d)/=[3,2])) _die('!ub/=2')
  _incr_sf_nn(p,d,nn,f)
  if(f>size(mem)) _die('!p>size(mem)')
  ff%cells = int(reshape(mem(p:f), nn(1:2)))
  p = f
  
  p = p + 1
  if(p>size(mem)) _die('!p>size(mem)')
  ff%nr = int(mem(p))

  p = p + 1
  if(p>size(mem)) _die('!p>size(mem)')
  ff%jmax = int(mem(p))

end subroutine ! scatter  


!
! Get a radial function from the m-miltiplett
!
subroutine dealloc(f)
  implicit none
  !! external
  type(functs_m_mult_t), intent(inout) :: f
  !! internal
  _dealloc(f%ir_j_prd2v)
  _dealloc(f%prd2m)
  _dealloc(f%crc)
  f%coords = -999
  f%rcuts = -999
  f%species = -999
  f%atoms = -999
  f%cells = -999
  f%nfunct = -999
  f%nr = -999
  f%jmax = -999
  
end subroutine ! dealloc


!
!
!
function get_diff_sp(f1, f2) result(d)
  implicit none
  !! external
  type(functs_m_mult_t), intent(in) :: f1, f2
  real(8) :: d
  !! internal
  real(8) :: sa
  integer :: n1, n2
  
  d = 0
  _size_alloc(f1%ir_j_prd2v,n1)
  _size_alloc(f2%ir_j_prd2v,n2)
  if(n1/=n2) then
    d = 9999
    if(d>1d-14) then
      write(6,*) n1, n2
      _die('!sa>1d-14')
    endif  
    return
  else if (n1>0) then
    sa = sum(abs(f1%ir_j_prd2v-f2%ir_j_prd2v))/n1
    d = d + sa 
    if(d>1d-14) _die('!d>1d-14')
  endif


  _size_alloc(f1%prd2m,n1)
  _size_alloc(f2%prd2m,n2)
  if(n1/=n2) then
    d = 9999
    if(d>1d-14) _die('!sa>1d-14')
    return
  else if (n1>0) then
    sa = sum(abs(f1%prd2m-f2%prd2m))/n1
    d = d + sa 
    if(d>1d-14) _die('!d>1d-14')
  endif


  _size_alloc(f1%crc,n1)
  _size_alloc(f2%crc,n2)
  if(n1/=n2) then
    d = 9999
    if(d>1d-14) then
      write(6,*) n1, n2 
      _die('!sa>1d-14')
    endif  
    return
  else if (n1>0) then
    sa = sum(abs(f1%crc-f2%crc))/n1
    d = d + sa 
    if(d>1d-14) then
      write(6,*) n1, n2, allocated(f1%crc), allocated(f2%crc)
      write(6,*) f1%crc
      write(6,*) f2%crc
      _die('!d>1d-14')
    endif  
  endif
  
  n1 = size(f1%coords)
  n2 = size(f2%coords)
  if(n1/=n2) then
    d = 9999
    if(d>1d-14) _die('!sa>1d-14')
    return
  else if (n1>0) then
    sa = sum(abs(f1%coords-f2%coords))/n1
    d = d + sa 
    if(d>1d-14) _die('!d>1d-14')
  endif
  
  n1 = size(f1%rcuts)
  n2 = size(f2%rcuts)
  if(n1/=n2) then
    d = 9999
    if(d>1d-14) _die('!sa>1d-14')
    return
  else if (n1>0) then
    sa = sum(abs(f1%rcuts-f2%rcuts))/n1
    d = d + sa 
    if(d>1d-14) _die('!d>1d-14')
  endif

  n1 = size(f1%species)
  n2 = size(f2%species)
  if(n1/=n2) then
    d = 9999
    if(d>1d-14) _die('!sa>1d-14')
    return
  else if (n1>0) then
    sa = sum(abs(f1%species-f2%species))/n1
    d = d + sa 
    if(d>1d-14) _die('!d>1d-14')
  endif

  n1 = size(f1%atoms)
  n2 = size(f2%atoms)
  if(n1/=n2) then
    d = 9999
    if(d>1d-14) _die('!sa>1d-14')
    return
  else if (n1>0) then
    sa = sum(abs(f1%atoms-f2%atoms))/n1
    d = d + sa 
    if(d>1d-14) _die('!d>1d-14')
  endif

  n1 = size(f1%cells)
  n2 = size(f2%cells)
  if(n1/=n2) then
    d = 9999
    if(d>1d-14) _die('!sa>1d-14')
    return
  else if (n1>0) then
    sa = sum(abs(f1%cells-f2%cells))/n1
    d = d + sa 
    if(d>1d-14) _die('!d>1d-14')
  endif

  sa = abs(f1%nfunct-f2%nfunct)/n1
  d = d + sa 
  if(d>1d-14) _die('!d>1d-14')

  sa = abs(f1%nr-f2%nr)/n1
  d = d + sa 
  if(d>1d-14) _die('!d>1d-14')

  sa = abs(f1%jmax-f2%jmax)/n1
  d = d + sa 
  if(d>1d-14) _die('!d>1d-14')

end function !  get_diff_sp

!
!
!
function get_diff(p2f1, p2f2) result(d)
  implicit none
  !! external
  type(functs_m_mult_t), intent(in), allocatable :: p2f1(:), p2f2(:)
  real(8) :: d
  !! internal
  integer :: n1, n2, p

  d = 0  
  _size_alloc(p2f1,n1)
  _size_alloc(p2f2,n2)
  if(n1/=n2) then
    d = 9999
    if(d>1d-14) _die('!d>1d-14')
    return
  else if (n1>0) then
    d = 0
    do p=1,n1
      d = d + get_diff_sp(p2f1(p), p2f2(p))
    enddo 
  endif

end function !  get_diff 

!
!
!
function get_spent_ram(bsp) result(ram_bytes)
  use m_get_sizeof
  implicit none
  !! external
  type(functs_m_mult_t), intent(in) :: bsp
  real(8) :: ram_bytes
  !! internal
  integer(8) :: nr, ni
  
  nr = 0 
  _add_size_alloc(nr, bsp%ir_j_prd2v)
  _add_size_alloc(nr, bsp%crc)
  _add_size_array(nr, bsp%coords)
  _add_size_array(nr, bsp%rcuts)
  
  ni = 0
  _add_size_alloc(ni, bsp%prd2m)
  _add_size_array(ni, bsp%species)
  _add_size_array(ni, bsp%atoms)
  _add_size_array(ni, bsp%cells)
  ni = ni + 1

  ram_bytes = nr*get_sizeof(bsp%rcuts(1)) + ni*get_sizeof(bsp%atoms(1))
  
end function ! get_spent_ram
  

!
! Number of centers per bilocal specie (often equiivalent to a bilocal pair of atoms)
!
function get_ncenters(bsp) result(nf)
  implicit none
  type(functs_m_mult_t), intent(in) :: bsp  
  integer :: nf
  if(.not. allocated(bsp%crc)) _die('!bsp%crc')
  nf = size(bsp%crc,2)
  if(nf<1) _die('nf<1')
end function ! get_ncenters
  

!
! Computes Fourier transform of m-multipletts for a set of momenta 
! pvec(1:3,1:nmomenta) (given by their Cartesian coordinates)
!
subroutine comp_ft_mmult(pp, sp2functs_mom, nmomenta, pvecs, sp2ft_functs)
  use m_algebra, only : matinv_d
  use m_arrays, only : z_array2_t
  use m_interpolation, only : grid_2_dr_and_rho_min, get_fval
  use m_harmonics, only : rsphar
  implicit none
  !! external
  real(8), intent(in) :: pp(:)
  type(functs_m_mult_t), intent(in), allocatable :: sp2functs_mom(:)
  integer, intent(in) :: nmomenta
  real(8), intent(in) :: pvecs(:,:)
  type(z_array2_t), intent(inout), allocatable :: sp2ft_functs(:)
  !! internal
  integer :: ip, nsp, sp, nr, nprd, m, jmax, prd, j, jcutoff
  logical :: is_true
  real(8) :: pvec(3), pvec_rot(3), psca, dp_jt, kmin_jt, fp_mom, pi
  real(8) :: rotation(3,3)
  real(8), allocatable :: ff(:), slm(:), sp2rot_mat(:,:,:)
  complex(8) :: zi
  pi = 4D0*atan(1D0);
  zi = cmplx(0D0, 1D0,8);
  
  _dealloc(sp2ft_functs)
  nsp = 0
  if(allocated(sp2functs_mom)) nsp = size(sp2functs_mom)
  if (nsp<1) return;

  if(3>size(pvecs,1)) _die('3>size(pvecs,1)')
  if(nmomenta>size(pvecs,2)) _die('nmomenta>size(pvecs,2)')
  if(.not. allocated(sp2functs_mom)) _die('.not. allocated(sp2functs_mom)') 
  sp = 1
  is_true = .True.
  do while (is_true)
    if(.not. allocated(sp2functs_mom(sp)%ir_j_prd2v) .and. sp <= nsp) then
      sp = sp + 1
    else
      is_true = .False.
    endif
  enddo

  if (sp == nsp .and. .not. allocated(sp2functs_mom(sp)%ir_j_prd2v)) then
    _die('ir_j_prd2v not allocated for any spp')
  endif


  nr = get_nr_mmult(sp2functs_mom(sp))
  
  jcutoff = get_jcutoff_mmult(sp2functs_mom)
  
  allocate(sp2ft_functs(nsp))
  
  call grid_2_dr_and_rho_min(nr, pp, dp_jt, kmin_jt)

  allocate(sp2rot_mat(3,3,nsp))
  do sp=1,nsp
    rotation = get_rotation_n_to_z(sp2functs_mom(sp))
    call matinv_d(rotation)
    sp2rot_mat(1:3,1:3,sp) = rotation
  enddo ! sp  

  do sp=1,nsp
    nprd = get_nfunct_mmult(sp2functs_mom(sp))
    allocate(sp2ft_functs(sp)%array(nmomenta,nprd))
    sp2ft_functs(sp)%array = 0
  enddo ! sp

  !$OMP PARALLEL DEFAULT(NONE) &
  !$OMP SHARED(nmomenta,pvecs,nsp,sp2functs_mom,kmin_jt,dp_jt,nr,sp2rot_mat) &
  !$OMP SHARED(sp2ft_functs,jcutoff,zi) &
  !$OMP PRIVATE(ip,pvec,psca,sp,jmax,slm,pvec_rot,nprd,prd,m,j,fp_mom,ff)
  allocate(slm(0:(jcutoff+1)**2))
  allocate(ff(nr))
  !! Loop over momenta for which we should compute Fourier transform
  !$OMP DO
  do ip=1, nmomenta
    pvec = pvecs(:,ip)
    psca = sqrt(sum(pvec**2))

    !! Loop over (m-multiplett) species
    do sp=1,nsp
      call DGEMM('N','N',1,3,3, 1D0,pvec,1,sp2rot_mat(1,1,sp),3,0D0,pvec_rot,1)
      jmax = get_jmax_mmult(sp2functs_mom(sp))
      call rsphar(pvec_rot, slm(0:), jmax)
      nprd = get_nfunct_mmult(sp2functs_mom(sp))
     
      do prd=1,nprd
        m = get_m(sp2functs_mom(sp), prd)
        
        do j=abs(m),jmax
          call get_ff_mmult(sp2functs_mom(sp), j, prd, ff)
          fp_mom = get_fval(ff, psca, kmin_jt, dp_jt, nr)
           sp2ft_functs(sp)%array(ip,prd) = sp2ft_functs(sp)%array(ip,prd) + &
             conjg((zi**j)) * fp_mom * slm( j*(j+1)+m)
        enddo ! j
      enddo ! p=1,nf
    enddo ! sp=1,nsp
    !! END of Loop over (m-multiplett) species
  enddo
  !$OMP ENDDO
  _dealloc(slm)
  _dealloc(ff)
  !! END of Loop over momenta for which we should compute Fourier transform
  !$OMP END PARALLEL

  do sp=1,nsp
    sp2ft_functs(sp)%array = sp2ft_functs(sp)%array * sqrt(pi/2)*(4*pi)
  enddo ! sp

end subroutine ! comp_FT_mmult

!
!
!
function get_rotation_n_to_z(functs) result(rotation_n_to_z)
  use m_wigner_rotation, only :  translation2theta_phi, make_standard_rotation
  implicit none
  !! external
  type(functs_m_mult_t), intent(in)  :: functs
  real(8) :: rotation_n_to_z(3,3)
  !! internal
  integer :: xyz
  real(8) :: translat_vec(3), theta, phi, rotation_z_to_n(3,3)
  
  translat_vec=functs%coords(:,2)-functs%coords(:,1)
  if (sum(abs(translat_vec))>0D0) then
    call translation2theta_phi(translat_vec, theta, phi)
    call make_standard_rotation(theta,phi,rotation_z_to_n,rotation_n_to_z)
  else
    rotation_n_to_z = 0;
    do xyz=1,3; rotation_n_to_z(xyz,xyz)=1; enddo;
  endif
end function !get_rotation_n_to_z  



!
! Computes dipole moments of the functions stored in m-multipletts
!
subroutine comp_dip_mom_mmult(rr, sp2functs, sp2dip_mom)
  use m_arrays, only : d_array2_t
  use m_wigner_rotation, only : make_standard_rotation, translation2theta_phi

  implicit none
  real(8), intent(in) :: rr(:)
  type(functs_m_mult_t), intent(in), allocatable :: sp2functs(:)
  type(d_array2_t), intent(inout), allocatable :: sp2dip_mom(:)! sp%array(funct,xyz)
  !! internal
  integer :: nsp, sp, m, nr, nfunct, f
  logical :: is_true
  real(8) :: pi, d_lambda, frr, dip_mom(3,-1:1), cart_vec_rotated(3)
  real(8) :: theta, phi, translat_vec(3), rotation_z_to_n(3,3), rotation_n_to_z(3,3)
  real(8), allocatable :: ff(:)
  pi = 4D0*atan(1D0)
  nsp = 0
  if(allocated(sp2functs)) nsp = size(sp2functs)
  if (nsp<1) then; _dealloc(sp2dip_mom); return; endif

  allocate(sp2dip_mom(nsp))
  sp = 1
  is_true = .True.
  do while (is_true)
    if(.not. allocated(sp2functs(sp)%ir_j_prd2v) .and. sp <= nsp) then
      sp = sp + 1
    else
      is_true = .False.
    endif
  enddo

  if (sp == nsp .and. .not. allocated(sp2functs(sp)%ir_j_prd2v)) then
    _die('ir_j_prd2v not allocated for any spp')
  endif


  nr = get_nr_mmult(sp2functs(sp))
  allocate(ff(nr))
  _zero(ff)
  d_lambda=log(rr(nr)/rr(1))/(nr-1)

  !! Loop over (m-multiplett) species
  do sp=1,nsp
    nfunct = get_nfunct_mmult(sp2functs(sp))
    allocate(sp2dip_mom(sp)%array(nfunct,3))
    sp2dip_mom(sp)%array = 0

    translat_vec=sp2functs(sp)%coords(:,2)-sp2functs(sp)%coords(:,1)
    if (sum(abs(translat_vec))>0D0) then
      call translation2theta_phi(translat_vec, theta, phi)
      call make_standard_rotation(theta,phi,rotation_z_to_n,rotation_n_to_z)
    else
      rotation_z_to_n = 0;
      do m=1,3; rotation_z_to_n(m,m)=1; end do;
    endif

    do f=1,nfunct
      m = get_m(sp2functs(sp), f)
      if(abs(m)>1) cycle ! if |m|>1, then l>1 and there is no dipole moment...
      call get_ff_mmult(sp2functs(sp), 1, f, ff)
      frr = sqrt(4*pi/3D0)*d_lambda*sum(ff*rr**4)
      dip_mom = 0
      dip_mom(1, 1) = frr
      dip_mom(2,-1) = frr
      dip_mom(3, 0) = frr
      cart_vec_rotated=matmul(rotation_z_to_n, dip_mom(1:3,m))
      sp2dip_mom(sp)%array(f,1:3) = cart_vec_rotated
    enddo ! mu=1,nmult

  enddo ! sp=1,nsp
  !! END of Loop over (m-multiplett) species

end subroutine ! comp_dip_mom_mmult
   
!
! Computes scalar moments of the functions stored in m-multipletts
!
subroutine comp_sca_mom_mmult(rr, sp_biloc2functs, sp_biloc2sca_mom)
  use m_arrays, only : d_array1_t
  implicit none
  real(8), intent(in) :: rr(:)
  type(functs_m_mult_t), intent(in), allocatable :: sp_biloc2functs(:)
  type(d_array1_t), intent(inout), allocatable :: sp_biloc2sca_mom(:)
  !! internal
  integer :: nsp, sp, m, nr, nfunct, f, spp
  logical :: is_true
  real(8) :: pi, d_lambda
  real(8), allocatable :: ff(:)
  pi = 4D0*atan(1D0)
  nsp = 0
  if(allocated(sp_biloc2functs)) nsp = size(sp_biloc2functs)
  if (nsp<1) then; _dealloc(sp_biloc2sca_mom); return; endif

  allocate(sp_biloc2sca_mom(nsp))

  spp = 1
  is_true = .True.
  do while (is_true)
    if(.not. allocated(sp_biloc2functs(spp)%ir_j_prd2v) .and. spp <= nsp) then
      spp = spp + 1
    else
      is_true = .False.
    endif
  enddo

  if (spp == nsp .and. .not. allocated(sp_biloc2functs(spp)%ir_j_prd2v)) then
    _die('ir_j_prd2v not allocated for any spp')
  endif


  nr = get_nr_mmult(sp_biloc2functs(spp))
  allocate(ff(nr))
  _zero(ff)
  d_lambda=log(rr(nr)/rr(1))/(nr-1)

  !! Loop over (m-multiplett) species
  do sp=1,nsp
    nfunct = get_nfunct_mmult(sp_biloc2functs(sp))
    allocate(sp_biloc2sca_mom(sp)%array(nfunct))
    sp_biloc2sca_mom(sp)%array = 0
    do f=1,nfunct
      m = get_m(sp_biloc2functs(sp), f)
      if(abs(m)>0) cycle
      call get_ff_mmult(sp_biloc2functs(sp), 0, f, ff)
      sp_biloc2sca_mom(sp)%array(f) = sqrt(4*pi)*d_lambda*sum(ff*rr**3)
    enddo ! mu=1,nmult
  enddo ! sp=1,nsp
  !! END of Loop over (m-multiplett) species

end subroutine ! comp_sca_mom_mmult

!
!
!
subroutine init_moms_mmult(sp2functs_rea, rr, sp2moms)
  use m_interpolation, only : get_dr_jt
  implicit none
  !! external
  type(functs_m_mult_t), intent(in), allocatable :: sp2functs_rea(:)
  real(8), intent(in) :: rr(:)
  type(functs_m_mult_t), intent(inout), allocatable :: sp2moms(:)
  !! internal
  integer :: j,spp,nprd,p,m,jcutoff,jmax
  real(8) :: dlambda
  !! Dimensions
  integer :: nspp, nr
  logical :: is_true
  !! END of Dimensions

  _dealloc(sp2moms)
  if(.not. allocated(sp2functs_rea)) return
  nspp = size(sp2functs_rea)
  allocate(sp2moms(nspp))

  spp = 1
  is_true = .True.
  do while (is_true)
    if(.not. allocated(sp2functs_rea(spp)%ir_j_prd2v) .and. spp <= nspp) then
      spp = spp + 1
    else
      is_true = .False.
    endif
  enddo

  if (spp == nspp .and. .not. allocated(sp2functs_rea(spp)%ir_j_prd2v)) then
    _die('ir_j_prd2v not allocated for any spp')
  endif

  nr = get_nr_mmult(sp2functs_rea(spp))
  dlambda = get_dr_jt(rr)
  jcutoff = get_jcutoff_mmult(sp2functs_rea)

  do spp=1,nspp
    sp2moms(spp) = sp2functs_rea(spp)
    nprd = get_nfunct_mmult(sp2functs_rea(spp))
    jmax = get_jmax_mmult(sp2functs_rea(spp))
    _dealloc(sp2moms(spp)%ir_j_prd2v)
    allocate(sp2moms(spp)%ir_j_prd2v(1,0:jcutoff,nprd))
    do p=1,nprd
      m = get_m(sp2functs_rea(spp), p)
      do j=abs(m),jmax
        sp2moms(spp)%ir_j_prd2v(1,j,p) = dlambda * &
          sum(sp2functs_rea(spp)%ir_j_prd2v(1:nr,j,p)*rr**(j+3))
      enddo ! j
    enddo ! p
  enddo ! spp

end subroutine ! init_moms_mmult

!
!
!
subroutine multipole_moms(frea, rr, moms)
  use m_interpolation, only : get_dr_jt
  implicit none
  !! external
  type(functs_m_mult_t), intent(in) :: frea
  real(8), intent(in) :: rr(:)
  type(functs_m_mult_t), intent(inout) :: moms
  !! internal
  integer :: j,nprd,p,m,jcutoff,jmax
  real(8) :: dlambda
  !! Dimensions
  integer :: nr
  !! END of Dimensions

  nr = get_nr_mmult(frea)
  dlambda = get_dr_jt(rr)
  jcutoff = get_jmax_mmult(frea)

  moms = frea
  nprd = get_nfunct_mmult(frea)
  jmax = get_jmax_mmult(frea)
  _dealloc(moms%ir_j_prd2v)
  allocate(moms%ir_j_prd2v(1,0:jcutoff,nprd))
  do p=1,nprd
    m = get_m(frea, p)
    do j=abs(m),jmax
      moms%ir_j_prd2v(1,j,p) = dlambda * &
        sum(frea%ir_j_prd2v(1:nr,j,p)*rr**(j+3))
    enddo ! j
  enddo ! p

end subroutine ! multipole_moms

!
!
!
subroutine init_functs_mmult_mom_space(Talman_plan, sp2functs_rea, sp2functs_mom)
  use m_sph_bes_trans, only : sbt_execute, Talman_plan_t, get_nr
  implicit none
  !! external
  type(Talman_plan_t), intent(in) :: Talman_plan
  type(functs_m_mult_t), intent(in), allocatable :: sp2functs_rea(:)
  type(functs_m_mult_t), intent(inout), allocatable :: sp2functs_mom(:)
  !! internal
  real(8), allocatable :: ff(:)  
  integer :: j,spp,m,prd,nprd, jmax,nspp, nr

  nr = get_nr(Talman_plan)
  allocate(ff(nr))
  
  _dealloc(sp2functs_mom)
  if(.not. allocated(sp2functs_rea)) return
  nspp = size(sp2functs_rea)
  allocate(sp2functs_mom(nspp))

  do spp=1,nspp
    sp2functs_mom(spp) = sp2functs_rea(spp)
    nprd = get_nfunct_mmult(sp2functs_rea(spp))
    jmax = get_jmax_mmult(sp2functs_rea(spp))

    do prd = 1, nprd
      m = get_m(sp2functs_rea(spp), prd)
      do j=abs(m), jmax
        call get_ff_mmult(sp2functs_rea(spp), j, prd, ff)
        call sbt_execute(Talman_plan, ff, sp2functs_mom(spp)%ir_j_prd2v(:,j,prd), j, 1)
      enddo ! j
    end do ! prd
  enddo ! spp
  
  _dealloc(ff)

end subroutine ! init_functs_mmult_mom_space


!
!
!
subroutine funct_sbt(Talman_plan, frea, fmom)
  use m_sph_bes_trans, only : sbt_execute, Talman_plan_t, get_nr
  implicit none
  !! external
  type(Talman_plan_t), intent(in) :: Talman_plan
  type(functs_m_mult_t), intent(in) :: frea
  type(functs_m_mult_t), intent(inout) :: fmom
  !! internal
  real(8), allocatable :: ff(:)  
  integer :: j,m,prd,nprd, jmax, nr

  nr = get_nr(Talman_plan)
  allocate(ff(nr))
  
  fmom = frea
  nprd = get_nfunct_mmult(frea)
  jmax = get_jmax_mmult(frea)

  do prd=1, nprd
    m = get_m(frea, prd)
    do j=abs(m), jmax
      call get_ff_mmult(frea, j, prd, ff)
      call sbt_execute(Talman_plan, ff, fmom%ir_j_prd2v(:,j,prd), j, 1)
    enddo ! j
  end do ! prd
  
  _dealloc(ff)

end subroutine ! funct_mom_space

!
! Get a radial function from the m-miltiplett
!
subroutine get_ff_mmult(functs, j, f, ff)
  implicit none
  !! external
  type(functs_m_mult_t), intent(in) :: functs
  integer, intent(in) :: j, f
  real(8), intent(out) :: ff(:)
  !! internal
  integer :: m

  if(.not. allocated(functs%ir_j_prd2v)) _die('ir_j_prd2v ?')
  if(.not. allocated(functs%prd2m)) _die('prd2m ?')

  if(j>ubound(functs%ir_j_prd2v,2)) _die('j>ubound(functs%ir_j_prd2v,2)')
  if(f>ubound(functs%ir_j_prd2v,3)) _die('f>ubound(functs%ir_j_prd2v,3)')
  if(f>ubound(functs%prd2m,1)) _die('f>ubound(functs%prd2m)')
  if(j<0) _die('j<0')
  if(f<1) _die('f<1')

  m = functs%prd2m(f)
  if( j>=abs(m) ) then
    ff = functs%ir_j_prd2v(:,j,f)
  else
    ff = 0
  endif

end subroutine ! get_ff_mmult

!
! Get a radial function from the m-miltiplett
!
function get_ff_ptr(functs, j, f) result(ff)
  implicit none
  !! external
  type(functs_m_mult_t), intent(in), target :: functs
  integer, intent(in) :: j, f
  real(8), pointer :: ff(:)
  !! internal
  integer :: m

  if(j<0) _die('j<0')
  if(f<1) _die('f<1')

  if(.not. allocated(functs%ir_j_prd2v)) _die('ir_j_prd2v ?')
  if(.not. allocated(functs%prd2m)) _die('prd2m ?')

  if(j>ubound(functs%ir_j_prd2v,2)) _die('j>ubound(functs%ir_j_prd2v,2)')
  if(f>ubound(functs%ir_j_prd2v,3)) _die('f>ubound(functs%ir_j_prd2v,3)')
  if(f>ubound(functs%prd2m,1)) _die('f>ubound(functs%prd2m)')

  m = functs%prd2m(f)
  if(j<abs(m)) _die('j<abs(m)')
 
  ff => functs%ir_j_prd2v(:,j,f)
end function ! get_ff_ptr

!
!
!
function get_jmax_mmult(functs) result(nf)
  implicit none
  type(functs_m_mult_t), intent(in) :: functs
  integer :: nf
  !! internal

  if (functs%jmax == -999) then
    nf = -1;
    if(.not. allocated(functs%ir_j_prd2v)) then
      nf = 0
      return
    endif
      !_die('.not. allocated(functs%ir_j_prd2v)')
    nf = ubound(functs%ir_j_prd2v,2)
  else
    nf = functs%jmax
  endif

end function ! get_jmax_mmult

!
!
!
function get_jcutoff_mmult(functs) result(nf)
  implicit none
  type(functs_m_mult_t), intent(in), allocatable :: functs(:)
  integer :: nf
  !! internal
  integer :: nspp, spp
  nf = -1;
  if(.not. allocated(functs)) _die('.not. allocated(functs)')
  nspp = size(functs)

  do spp=1,nspp  
    nf = max(nf, get_jmax_mmult(functs(spp)))
  enddo  

end function ! get_jcutoff_mmult


!
! Counts number of functions in a m-multiplett specie
!
function get_nfunct_mmult(functs) result (nf)
  implicit none
  !! external
  type(functs_m_mult_t), intent(in) :: functs
  integer :: nf
  !! internal
  integer :: nf1, nf2

  if (functs%nfunct ==-999) then

    nf = 0
    if(.not. allocated(functs%ir_j_prd2v)) &
      _die('ir_j_prd2v ?')
    if(.not. allocated(functs%prd2m)) &
      _die('prd2m ?')

    nf1 = size(functs%ir_j_prd2v,3)
    nf2 = size(functs%prd2m)

    nf = nf1

    if(nf1/=nf2) then
      write(0,'(a,2i8)') 'nf1  nf2', nf1, nf2
      _die('nf1/=nf2')
    endif

  else
    nf = functs%nfunct
  endif

end function ! get_nfunct_mmult

!
! Determine magnetic number of a function in m-specie
!
function get_m(functs, f) result (m)
  implicit none
  !! external
  type(functs_m_mult_t), intent(in) :: functs
  integer, intent(in) :: f
  integer :: m
  !! internal
  if(.not. allocated(functs%prd2m)) _die('prd2m ?')

  if(f>size(functs%prd2m)) _die('f>size(functs%prd2m)')
  m =  functs%prd2m(f)
end function ! get_m


!
! Counts number of radial points in radial functions of an m-multiplett
!
function get_nr_mmult(functs) result (nf)
  implicit none
  !! external
  type(functs_m_mult_t), intent(in) :: functs
  integer :: nf
  !! internal

  nf = 0
  if (functs%nr == -999) then
    if(.not. allocated(functs%ir_j_prd2v)) _die('ir_j_prd2v ?')

    nf = size(functs%ir_j_prd2v,1)
  else
    nf = functs%nr
  endif

end function ! get_nr_mmult

!
! Computes scalar moments of the functions stored in m-multipletts
!
subroutine comp_norm_mmult(rr, functs, norms)
  implicit none
  real(8), intent(in) :: rr(:)
  type(functs_m_mult_t), intent(in) :: functs
  real(8), intent(inout), allocatable :: norms(:)
  !! internal
  integer :: m, nr, nf, f, j, jmax
  real(8) :: pi, d_lambda
  real(8), pointer :: ff(:)
  pi = 4D0*atan(1D0)

  nr = get_nr_mmult(functs)
  d_lambda=log(rr(nr)/rr(1))/(nr-1)

  nf = get_nfunct_mmult(functs)
  if(allocated(norms)) then
    if(size(norms)<nf) deallocate(norms)
  endif
  if(.not. allocated(norms)) allocate(norms(nf))

  jmax = get_jmax_mmult(functs)
  
  norms = 0
  do f=1,nf
    m = get_m(functs, f)
    do j=abs(m),jmax
      ff =>get_ff_ptr(functs, j, f)
      norms(f) = norms(f) + sqrt(4*pi*d_lambda**2*sum(ff**2*rr**3))
    enddo ! j  
  enddo ! mu=1,nmult

end subroutine ! comp_norm_mmult

end module ! modul_funct_m_mult
