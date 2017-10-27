module m_tddft_tem

!
! module m_tddft_tem
! Main module to the calculation of the energy loss of an electron 
! along a line and related quantities.
!
!

#include "m_define_macro.F90"
  use m_die, only : die
  use m_warn, only : warn
  use m_timing, only : get_cdatetime
  implicit none

  private warn, die

  real(4), allocatable :: V_mu_center(:)
  complex(8), allocatable :: clm_tem(:)
  complex(4), allocatable :: f(:), FT(:)
  !$OMP THREADPRIVATE(V_mu_center, clm_tem, f, FT)

  contains

!
!
!
subroutine calculate_potential_pb_test(rr, rr_size) bind(c, name='calculate_potential_pb_test')
  use iso_c_binding

  implicit none
  integer(C_INT), intent(in), value :: rr_size
  real(C_DOUBLE), intent(in) :: rr(rr_size)

  print*, "Hola!! ", rr_size
end subroutine

!
!
!
subroutine calculate_potential_pb(rr, time, freq, freq_symm, velec, b, &
  V_freq_real, V_freq_imag, nfmx, jmx_pb, jcut_lmult, rr_size, time_size, freq_size, &
  nprod, nc) bind(c, name='calculate_potential_pb')
  
  use iso_c_binding
  use m_harmonics, only : init_c2r_hc_c2r
  
  implicit none
  
  integer(C_INT), intent(in), value :: nfmx, jmx_pb, jcut_lmult, rr_size, time_size, freq_size, nprod, nc
  real(C_DOUBLE), intent(in) :: rr(rr_size), time(time_size), freq(freq_size), freq_symm(time_size)
  real(C_DOUBLE), intent(in) :: b(3), velec(3)
  real(C_FLOAT), intent(inout) :: V_freq_real( nprod, freq_size), V_freq_imag(nprod, freq_size)


  !intern
  integer :: ic
#include <fftw3.f>
  integer(8) :: plan
  integer :: N, ub, spp
  !complex(4), allocatable :: f(:), FT(:)
  real(8) :: tmin, wmin, dt, R0(3), vnorm, vdir(3)
  complex(8), allocatable :: c2r(:, :), hc_c2r(:, :)
  real(8) :: t1, t2

  print*, "hola tddft_tem"
  
  ub = time_size/2 -1
  dt = time(2)-time(1)
  tmin = time(1)
  wmin = freq_symm(1)

  call init_c2r_hc_c2r(jcut_lmult, c2r, hc_c2r)
  _dealloc(hc_c2r)

  call cputime(t1)
  R0 = 0.0
  vnorm = sqrt(sum(velec**2))
  vdir = velec/vnorm
  R0 = vnorm*tmin*vdir + b

!  !$OMP PARALLEL DEFAULT(NONE) &
!  !$OMP PRIVATE(ic, book, N, plan) &
!  !$OMP SHARED(nc, pb, tem, aux_prod, rr, t, b, nfmx, R0, V_freq) &
!  !$OMP SHARED(wmin, tmin, dt, ub, w, c2r, freq_size, vnorm)
! 
  allocate(V_mu_center(nfmx))
  allocate(clm_tem(0:(jcut_lmult+1)**2))
  allocate(f(1:time_size))
  allocate(FT(1:time_size))

  f=0.0
  FT = 0.0
  V_mu_center = 0.0
  clm_tem = 0.0
  N = size(f, 1)

!  !$OMP CRITICAL
  call sfftw_plan_dft_1d ( plan, N, f, FT, FFTW_FORWARD, FFTW_ESTIMATE )
!  !$OMP END CRITICAL
!
!  !$OMP DO SCHEDULE(DYNAMIC,1)
  do ic=1, nc
      spp = 1 !book%spp
      call get_potential_lmult(rr, time, freq_symm, c2r, f, &
              plan, dt, wmin, tmin, ub, freq_size, spp, R0, &
              V_freq_real, V_freq_imag, vnorm)
  enddo
!  !$OMP END DO
!
!  !$OMP CRITICAL
  call sfftw_destroy_plan ( plan )
!  !$OMP END CRITICAL
!
  _dealloc(V_mu_center)
  _dealloc(clm_tem)
  _dealloc(f)
  _dealloc(FT)

!  !$OMP END PARALLEL

  print*, "End tddft_tem"
  call cputime(t2)
  print*, 'tddft_tem calc potential', t2-t1

end subroutine !calculate_potential_pb_fast

!
!
!
subroutine get_potential_lmult(rr, t, w, c2r, Vt_mu_center, &
          plan, dt, wmin, tmin, ub, nff, spp, R0, V_freq_real, V_freq_imag, vnorm)
!  use m_tddft_tem_inp, only : tddft_tem_inp_t 
!  use m_prod_basis_type, only : prod_basis_t
!  use m_prod_basis_coord, only : prod_basis_coord_t
!  use m_book_pb, only : book_pb_t
!  use m_functs_l_mult_type, only : get_nmult
!
  implicit none
  real(8), intent(in) :: t(*), w(*)
  real(8), intent(in) :: rr(*)
  real(8), intent(in) :: R0(3), dt, wmin, tmin, vnorm
  integer, intent(in) :: ub, nff, spp
#include <fftw3.f>
  integer(8), intent(in) :: plan
  
  complex(4), allocatable, intent(inout) :: Vt_mu_center(:)
  complex(8), allocatable, intent(inout) :: c2r(:, :)
  real(4), intent(inout) :: V_freq_real(*), V_freq_imag(*)

!  !intern
  integer :: it, si, fi, mu, l_mu, n_mu, k
  real(8) :: r_cut, inte1
  complex(4) :: j = cmplx (0.0, 1.0)

  clm_tem = 0.0
!
!  r_cut = pb%sp_local2functs(spp)%rcut
!  si = 0
!  fi = 0
!  n_mu = get_nmult(pb%sp_local2functs(spp))
!
!  do mu = 1, n_mu
!    l_mu = pb%sp_local2functs(spp)%mu2j(mu)
!    si = pb%sp_local2functs(spp)%mu2si(mu)
!    fi = si + (2*l_mu+1) - 1
!
!    inte1 = 0
!    call comp_integral_out(pb, a, mu, l_mu, spp, rr, r_cut, inte1)
!    
!    do k = si, fi
!      Vt_mu_center = 0
!      do it = lbound(t, 1), ubound(t, 1)
!        call get_potential_lmult_t(pb, tem, a, t, it, mu, l_mu, si, fi, spp,&
!                    &r_cut, rr, book%coord, inte1, c2r, Vt_mu_center, k, R0, vnorm)
!      enddo !it
!
!      Vt_mu_center = dt*Vt_mu_center*exp(-j*wmin*(t-tmin))
!#ifndef nofftw3f
!      call sfftw_execute ( plan, Vt_mu_center, FT )
!#endif
!      V_freq(book%si(3) + k-1, :) = FT(ub+1:ub+nff)*exp(-j*(wmin*tmin + &
!                 (w(ub+1:ub+nff)-wmin)*tmin))
!    enddo
!  enddo ! mu
!
end subroutine !get_potential_lmult

!
!
!
!subroutine get_potential_lmult_t(pb, tem, a, t, it, mu, l_mu, si, fi, spp,&
!  r_cut, rr, center, I1, c2r, Vt_mu_center, kn, R0, vnorm)
!  use m_tddft_tem_inp, only : tddft_tem_inp_t 
!  use m_prod_basis_type, only : prod_basis_t
!  use m_prod_basis_coord, only : prod_basis_coord_t
!  use m_csphar, only : csphar
!  use m_interpolation, only : get_fval
!  use m_interp, only : comp_coeff_m2p3_k
!  use m_constants, only: const
!
!  implicit none
!  type(prod_basis_t), intent(in) :: pb
!  type(tddft_tem_inp_t), intent(in) :: tem
!  type(prod_basis_coord_t), intent(in) :: a
!  real(8), allocatable, intent(in) :: t(:)
!  real(8), intent(in), pointer :: rr(:)
!  integer, intent(in) :: it, spp, mu, l_mu, si, fi, kn
!  real(8), intent(in) :: r_cut, I1, center(3), R0(3), vnorm
!  
!  complex(4), allocatable, intent(inout) :: Vt_mu_center(:)
!  complex(8), allocatable, intent(inout) :: c2r(:, :)
!
!  !intern
!  integer :: ir, k
!  real(8) :: R_sub(3), inte1, inte2, norm_R_sub, coeff(-2:3)
!  real(8) :: fr_val
!
!  R_sub = R0 + vnorm*tem%vdir*(t(it)-t(1)) - center !in this version we assume a velocity of 1.0, 
!                                                  !it is the time range that change
!  norm_R_sub = sqrt(sum(R_sub**2))
!  V_mu_center = 0
!  
!  if (norm_R_sub>r_cut) then
!
!      inte1 = I1/(norm_R_sub**(l_mu+1))
!      inte2 = 0
!  else
!    inte1 = 0
!    inte2 = 0
!    ir = 1
!    do while (rr(ir) <r_cut)
!      call comp_coeff_m2p3_k(rr(ir)**2, a%interp_a, coeff,k)
!      fr_val = sum(coeff*pb%sp_local2functs(spp)%ir_mu2v(k-2:k+3,mu))
!      if (rr(ir)<norm_R_sub) then
!        inte1 = inte1 + fr_val*(rr(ir)**(l_mu+2))*rr(ir)
!      else
!        inte2 = inte2 + fr_val*rr(ir)/(rr(ir)**(l_mu-1))
!      endif
!      ir = ir + 1
!    enddo
!    inte1 = inte1*a%interp_a%dr/(norm_R_sub**(l_mu+1))
!    inte2 = inte2*(norm_R_sub**l_mu)*a%interp_a%dr
!  endif
!  clm_tem = 0
!  call csphar(R_sub, clm_tem(0:), a%jcut_lmult)
!  clm_tem(l_mu*(l_mu+1)-l_mu:l_mu*(l_mu+1)+l_mu) = (4*const%pi/(2*l_mu+1))* &
!    clm_tem(l_mu*(l_mu+1)-l_mu:l_mu*(l_mu+1)+l_mu)*(inte1+inte2)
!  call sph_cplx2real(clm_tem(l_mu*(l_mu+1)-l_mu:l_mu*(l_mu+1)+l_mu), V_mu_center(si:fi), &
!    c2r, l_mu, si, fi)
!
!  Vt_mu_center(it) = V_mu_center(kn)
!
!end subroutine !get_potential_lmult_t
!
!!
!!
!!
!subroutine comp_integral_out(pb, a, mu, l_mu, spp, rr, r_cut, inte1)
!  use m_prod_basis_type, only : prod_basis_t
!  use m_prod_basis_coord, only : prod_basis_coord_t
!  use m_interp, only : comp_coeff_m2p3_k
!
!  implicit none
!  type(prod_basis_t), intent(in) :: pb
!  type(prod_basis_coord_t), intent(in) :: a
!  real(8), intent(in), pointer :: rr(:)
!  integer, intent(in) :: spp, mu, l_mu
!  real(8), intent(in) :: r_cut
!  real(8), intent(inout) :: inte1
!  
!  !intern
!  integer :: ir, k
!  real(8) :: coeff(-2:3)
!  real(8) :: fr_val
!
!  ir = 1
!  do while (rr(ir) <= r_cut)
!    call comp_coeff_m2p3_k(rr(ir)**2, a%interp_a, coeff,k)
!    fr_val = sum(coeff*pb%sp_local2functs(spp)%ir_mu2v(k-2:k+3,mu))
!    inte1 = inte1 + fr_val*(rr(ir)**(l_mu+2))*rr(ir)*a%interp_a%dr
!    ir = ir+1
!  enddo
!   
!end subroutine !comp_integral_out
!
!!
!!
!!
!subroutine init_new_freq(w_ev, freq)
!  use m_freq, only : freq_t
!
!  implicit none
!  real(8), allocatable, intent(in) :: w_ev(:)
!  type(freq_t), intent(inout) :: freq
!
!  real(8), allocatable :: w(:)
!  real(8) ::dw
!  real(8) :: omega_max_tddft, omega_min_tddft
!
!  allocate(w(lbound(w_ev, 1):ubound(w_ev, 1)))
!
!  w = w_ev*0.0367493081366D0
!  freq%nff = size(w, 1)! - ind_pos + 1
!
!  dw = w(2)-w(1)
!
!  freq%d_omega1 = dw
!  freq%d_omega2 = dw
!  _dealloc(freq%grid1)
!  allocate(freq%grid1(freq%nff))
!  _dealloc(freq%grid2)
!  allocate(freq%grid2(freq%nff))
!
!  freq%grid1 = w
!  freq%grid2 = w
!
!  omega_min_tddft = w(1)
!  omega_max_tddft = w(ubound(w, 1))
!
!
!  freq%fmin = 1!min(max(nint(omega_min_tddft/freq%d_omega1),1),freq%nff)
!  freq%fmax = freq%nff!min(max(nint((omega_max_tddft-omega_min_tddft)/freq%d_omega1),1),freq%nff)
!
!
!end subroutine !init_freq
!
!!
!!
!!
!!
!subroutine sph_cplx2real(clm, rsh, c2r, l, si, fi)
!
!  implicit none
!  integer, intent(in) :: l, si, fi
!  complex(8), intent(in) :: clm(si:fi)
!  complex(8), allocatable, intent(in) :: c2r(:, :)
!  real(4), intent(inout) :: rsh(si:fi) !rsh(-l:l)
!
!  !intern
!  integer :: m
!
!  do m = -l, l
!    if (m == 0) then
!      rsh(si +m+l) = c2r(m, m)*clm(si +m+l)
!    else
!      rsh(si+m+l) = c2r(m, m)*clm(si+m+l) + c2r(m, -m)*clm(fi-m-l)
!    endif
!  enddo
!
!end subroutine !sph_cplx2real
!
!!
!!
!!
!!
!subroutine sph_real2cplx(rsph, clm, l, si, fi)
!  use m_harmonics, only : init_c2r_hc_c2r
!  use m_algebra, only : matinv_z
!
!  implicit none
!  integer, intent(in) :: l, si, fi
!  real(4), intent(in) :: rsph(si:fi) !rsh(-l:l)
!  complex(4), intent(inout) :: clm(si:fi)
!
!  !intern
!  integer :: m
!  complex(8), allocatable :: c2r(:, :), hc_c2r(:, :), r2c(:, :)
!
!  call init_c2r_hc_c2r(l, c2r, hc_c2r)
!
!  allocate(r2c(lbound(c2r, 1):ubound(c2r, 1), lbound(c2r, 2):ubound(c2r, 2)))
!  r2c = c2r
!  call matinv_z(r2c)
!
!  do m = -l, l
!    if (m == 0) then
!      clm(si +m+l) = r2c(m, m)*rsph(si +m+l)
!    else
!      clm(si+m+l) = r2c(m, m)*rsph(si+m+l) + r2c(m, -m)*rsph(fi-m-l)
!    endif
!  enddo
!
!end subroutine !sph_cplx2real
!
!!
!!
!!
!subroutine get_index(NV, ivx, ivy, ivz)
!
!  implicit none
!  integer, intent(in) :: NV(3)
!
!  integer, intent(inout) :: ivx, ivy, ivz
!
!  if (ivz < NV(3)) then
!    ivz = ivz + 1
!  elseif (ivz == NV(3)) then
!    if (ivy < NV(2)) then
!      ivy = ivy + 1
!      ivz = 1
!    elseif (ivy == NV(2)) then
!      ivx = ivx + 1
!      ivy = 1
!      ivz = 1
!    endif
!  endif
!
!end subroutine !get_index

end module m_tddft_tem
