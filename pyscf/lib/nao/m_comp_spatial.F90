module m_comp_spatial

#include "m_define_macro.F90" 
  use m_precision, only : blas_int
  use m_die, only : die
  use m_warn, only : warn
  use iso_c_binding
 
  implicit none
  private die
  private warn
  
  contains

!
!
!
subroutine test_index_2D(arr, Nx, Ny) bind(c, name="test_index_2D")

  implicit none
  integer(c_int), intent(in), value :: Nx, Ny
  real(c_float), intent(in) :: arr(Nx*Ny)

  integer :: i, j

  print*, "dim: ", Nx, Ny

  do j = 1, Ny
  do i = 1, Nx
    print*, i, j, arr(i + (j-1)*Nx)
  enddo
  enddo
end subroutine !test_index_2D

!
!
!
subroutine test_index_3D(arr, Nx, Ny, Nz) bind(c, name="test_index_3D")

  implicit none
  integer(c_int), intent(in), value :: Nx, Ny, Nz
  real(c_double), intent(in) :: arr(Nx*Ny*Nz)

  integer :: i, j, k, ind

  print*, "dim: ", Nx, Ny, Nz

  do k = 1, Nz
  do j = 1, Ny
  do i = 1, Nx
    ind = i + (j-1)*Nx + (k-1)*Nx*Ny
    print*, i, j, k, ind, arr(ind)
  enddo
  enddo
  enddo
end subroutine !test_index_2D


!
! 
!
subroutine get_spatial_density_parallel(dn_spatial_re, dn_spatial_im, mu2dn_re, mu2dn_im, &
            meshx, meshy, meshz, atom2sp, atom2s, gammin_jt, &
            dg_jt, Nx, Ny, Nz, nprod, natoms, nspecies) bind(c, name='get_spatial_density_parallel')
  use m_interp, only : interp_t, init_interp 
  use m_pb_libnao, only : pb

  !
  ! Using 1D arrays to avoid colum-row major issues
  !

  implicit none
  !! external
  integer(c_int), intent(in), value :: Nx, Ny, Nz, nprod, natoms, nspecies
  real(c_double), intent(in), value :: gammin_jt, dg_jt

  ! complex variable
  real(c_float), intent(inout) :: dn_spatial_re(Nx*Ny*Nz), dn_spatial_im(Nx*Ny*Nz)
  real(c_double), intent(in) :: mu2dn_re(nprod), mu2dn_im(nprod)

  ! real variables
  real(c_double), intent(in) :: meshx(Nx), meshy(Ny), meshz(Nz)
  
  ! integer
  integer(c_int64_t), intent(in) :: atom2sp(natoms), atom2s(natoms+1)
  
  !! internal
  type(interp_t) :: a
  real(8), allocatable :: res(:)
  real(8) :: br(3)
  integer :: ix, iy, iz, ind
  real(8) :: t1,t2,t=0

  call init_interp(pb%rr, a)

  
!_t1
!  !$OMP PARALLEL DEFAULT(NONE) &
!  !$OMP PRIVATE (ix, iy, iz, br, res, ind) &
!  !$OMP SHARED(dn_spatial_re, mu2dn_re, dn_spatial_im, mu2dn_im, Nx, Ny, Nz) &
!  !$OMP SHARED(meshx, meshy, meshz, a, atom2sp, atom2s) &
!  !$OMP SHARED(natoms, nspecies, pb)
  allocate(res(nprod))
  res = 0.0

  !$OMP DO
  do iz = 1, Nz
  do iy = 1, Ny
  do ix = 1, Nx
    br(1) = meshx(ix); br(2) = meshy(iy); br(3) = meshz(iz)
    call comp_dn_xyz(a, pb, atom2sp, atom2s, &
      br, res, natoms, nspecies)
    
    ind = iz + (iy-1)*Nz + (ix-1)*Nz*Ny
    dn_spatial_re(ind) = sum(res*mu2dn_re)
    dn_spatial_im(ind) = sum(res*mu2dn_im)

  enddo
  enddo
  enddo
!  !$OMP END DO
  _dealloc(res)
!  !$OMP END PARALLEL

!_t2(t)
!  print*, "timing loop fortran: ", t


end subroutine ! dens_libnao

!
! compute density change at the point x, y, z
!
subroutine comp_dn_xyz(a, pb, atom2sp, atom2s, br, res, natoms, nspecies)
  
  use m_prod_basis_type, only : prod_basis_t
  use m_functs_l_mult_type, only : get_nmult
  use m_rsphar, only : rsphar, init_rsphar
  use m_interp, only : interp_t, comp_coeff_m2p3_k, comp_coeff_m2p3

  implicit none
  type(interp_t), intent(in) :: a
  type(prod_basis_t), intent(in) :: pb
  real(8), intent(in) :: br(3)
  integer(c_int64_t), intent(in) :: atom2sp(natoms), atom2s(natoms+1)

  integer, intent(in) :: natoms, nspecies

  real(8), allocatable, intent(inout) :: res(:)


  integer :: atm, sp, si, fi, sp_prev, k, j, nmu, mu
  integer(8) :: jmx_sp, s, f
  real(8) :: brp(3), r, rcut, coeffs(-2:3), fval
  real(8), allocatable :: rsh(:)

  sp_prev = -1
  do atm = 1, natoms
    sp = atom2sp(atm) + 1 ! +1 for fortran count
    brp = br - pb%atom2coord(:, atm)

    r = sqrt(sum(brp**2))
    rcut = pb%sp_local2functs(sp)%rcut

    if (r > rcut) cycle

    si = atom2s(atm)
    fi = atom2s(atm+1)
    if (sp_prev /= sp) then
      _dealloc(rsh)
      
      jmx_sp = maxval(pb%sp_local2functs(sp)%mu2j)
      call init_rsphar(jmx_sp)
      allocate(rsh((jmx_sp+1)**2))
      rsh = 0.0
    endif
    call rsphar(brp, jmx_sp, rsh)

    call comp_coeff_m2p3_k(r**2, a, coeffs, k)
    nmu = get_nmult(pb%sp_local2functs(sp))
    f = 0
    do mu=1, nmu
      j = pb%sp_local2functs(sp)%mu2j(mu)
      s = f + 1; f = s + (2*j+1) - 1;
      fval = sum(coeffs*pb%sp_local2functs(sp)%ir_mu2v(k-2:k+3,mu))
      
      print*, si, s, f, size(res)
      res(si+s: si+f) = rsh(j*(j+1)-j+1:j*(j+1)+j+1)*fval
    enddo

  enddo

end subroutine !comp_dn_xyz


end module !m_comp_spatial
