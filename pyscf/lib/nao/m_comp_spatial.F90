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
            meshx, meshy, meshz, atom2sp, atom2coord, &
            atom2s, sp_mu2j, psi_log_rl, sp_mu2s, sp2rcut, rr, gammin_jt, &
            dg_jt, Nx, Ny, Nz, nprod, natoms, nr, nj, nspecies) bind(c, name='get_spatial_density_parallel')

  use m_interp, only : interp_t, init_interp 

  !
  ! Using 1D arrays to avoid colum-row major issues
  !

  implicit none
  !! external
  integer(c_int), intent(in), value :: Nx, Ny, Nz, nprod, natoms, nr, nj, nspecies
  real(c_double), intent(in), value :: gammin_jt, dg_jt

  ! complex variable
  real(c_float), intent(inout) :: dn_spatial_re(Nx*Ny*Nz), dn_spatial_im(Nx*Ny*Nz)
  real(c_double), intent(in) :: mu2dn_re(nprod), mu2dn_im(nprod)

  ! real variables
  real(c_double), intent(in) :: meshx(Nx), meshy(Ny), meshz(Nz), atom2coord(3*natoms)
  real(c_double), intent(in) :: sp2rcut(nspecies), rr(nr), psi_log_rl(nspecies*nj*nr)
  
  ! integer
  integer(c_int64_t), intent(in) :: atom2sp(natoms), atom2s(natoms+1), sp_mu2s(nspecies*(nj+1))
  integer(c_int32_t), intent(in) :: sp_mu2j(nspecies*nj)
  
  !! internal
  type(interp_t) :: a
  real(8), allocatable :: res(:)
  real(8) :: br(3)
  integer :: ix, iy, iz, ind
  real(8) :: t1,t2,t=0



  call init_interp(rr, a)
  
_t1
  !$OMP PARALLEL DEFAULT(NONE) &
  !$OMP PRIVATE (ix, iy, iz, br, res, ind) &
  !$OMP SHARED(dn_spatial_re, mu2dn_re, dn_spatial_im, mu2dn_im, Nx, Ny, Nz) &
  !$OMP SHARED(meshx, meshy, meshz, a, atom2sp, atom2s, atom2coord, sp2rcut) &
  !$OMP SHARED(sp_mu2j, sp_mu2s, psi_log_rl, natoms, nspecies, nr, nj)
  allocate(res(nprod))
  res = 0.0

  !$OMP DO
  do ix = 1, Nx
  do iz = 1, Nz
  do iy = 1, Ny
    br(1) = meshx(ix); br(2) = meshy(iy); br(3) = meshz(iz)
    call comp_dn_xyz(a, atom2sp, atom2s, atom2coord, sp2rcut, sp_mu2j, sp_mu2s, psi_log_rl, &
      br, res, natoms, nspecies, nr, nj)
    
    ind = iz + (iy-1)*Nz + (ix-1)*Nz*Ny
    dn_spatial_re(ind) = sum(res*mu2dn_re)
    dn_spatial_im(ind) = sum(res*mu2dn_im)

  enddo
  enddo
  enddo
  !$OMP END DO
  _dealloc(res)
  !$OMP END PARALLEL

_t2(t)
  print*, "timing loop fortran: ", t


end subroutine ! dens_libnao

!
! compute density change at the point x, y, z
!
subroutine comp_dn_xyz(a, atom2sp, atom2s, atom2coord, sp2rcut, sp_mu2j, sp_mu2s, psi_log_rl, &
    br, res, natoms, nspecies, nr, nj)
  
  use m_rsphar, only : rsphar, init_rsphar
  use m_interp, only : interp_t, comp_coeff_m2p3_k, comp_coeff_m2p3

  implicit none
  type(interp_t), intent(in) :: a
  real(8), intent(in) :: br(3)
  real(c_double), intent(in) :: atom2coord(natoms*3), sp2rcut(nspecies), psi_log_rl(nspecies*nj*nr)
  integer(c_int64_t), intent(in) :: atom2sp(natoms), atom2s(natoms+1), sp_mu2s(nspecies*(nj+1))
  integer(c_int32_t), intent(in) :: sp_mu2j(nspecies*nj)

  integer, intent(in) :: natoms, nspecies, nj, nr

  real(8), allocatable, intent(inout) :: res(:)


  integer :: atm, sp, si, fi, sp_prev, k, ij, ik, j, ind, coeff_ind, ixyz
  integer(8) :: jmx_sp, s, f
  real(8) :: brp(3), r, rcut, coeffs(-2:3), fval, mu2j(nj), psi_log(-2:3)
  real(8), allocatable :: rsh(:)

  sp_prev = -1
  do atm = 1, natoms
    sp = atom2sp(atm) + 1 ! +1 for fortran count
    do ixyz = 1, 3
      ind = ixyz + (atm-1)*3
      brp(ixyz) = br(ixyz) - atom2coord(ind)
    enddo

    r = sqrt(sum(brp**2))
    rcut = sp2rcut(sp)

    if (r > rcut) cycle

    si = atom2s(atm) + 1
    fi = atom2s(atm+1)
    if (sp_prev /= sp) then
      _dealloc(rsh)

      do ij=1, nj
        mu2j(ij) = sp_mu2j(ij + (sp-1)*nj)
      enddo

      jmx_sp = maxval(mu2j)
      call init_rsphar(jmx_sp)
      allocate(rsh((jmx_sp+1)**2))
      rsh = 0.0
    endif
    call rsphar(brp, jmx_sp, rsh)

    call comp_coeff_m2p3(a, r, coeffs, k)
    k = k-2

    do ij =1, nj
      j = sp_mu2j(ij + (sp-1)*nj)
      s = sp_mu2s(ij + (sp-1)*nj)
      f = sp_mu2s(ij+1 + (sp-1)*nj)-1
      
      coeff_ind = -2
      fval = 0.0
      do ik = k, k+5
        !ind = sp + (ij-1)*nspecies + (ik-1)*nspecies*nj
        ind = ik + (ij-1)*nr + (sp-1)*nr*nj
        psi_log(coeff_ind) =  psi_log_rl(ind)
        fval = fval + psi_log_rl(ind)*coeffs(coeff_ind)
        coeff_ind = coeff_ind + 1
      enddo
      if (j /= 0) fval = fval*r**j


      res(si+s: si+f) = rsh(j*(j+1)-j+1:j*(j+1)+j+1)*fval

    enddo
  enddo

end subroutine !comp_dn_xyz


end module !m_comp_spatial
