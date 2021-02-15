module m_comp_vext_tem

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
subroutine comp_vext_tem(time, freq_sym, ntime, nff, ub, nprod, vnorm, &
          vdir, beam_offset, Vfreq_real, Vfreq_imag) bind(c, name="comp_vext_tem")

  !
  ! Compute the external potential from a moving charge in frequency domain
  !

  use m_pb_libnao, only : pb
  use m_system_vars, only : get_atom2sp_ptr
  use m_book_pb, only : book_pb_t
  use m_prod_basis_gen, only : get_nbook, get_book
  use m_functs_l_mult_type, only : get_nmult
  use m_interp, only : interp_t, init_interp
  use m_prod_basis_type, only : get_jcutoff, get_jcutoff_lmult
  use m_rsphar, only : init_rsphar, rsphar
  use m_fact, only : pi

  implicit none

  integer(c_int), intent(in), value :: ntime, nff, nprod, ub
  real(c_double), intent(in), value :: vnorm
  real(c_double), intent(in) :: time(ntime), freq_sym(ntime), vdir(3), beam_offset(3)
  real(c_double), intent(inout) :: Vfreq_real(nff*nprod), Vfreq_imag(nff*nprod)

#include <fftw3.f>
  type(book_pb_t) :: book
  type(interp_t) :: a
  integer :: natoms, iatm, rmax, sp, si, fi, nmu, s, f
  integer :: l, m, mu, k, jcutoff, jcut_lmult
  integer :: ind_lm, nr, it, rsub_max, iw, ind 
  integer(8) :: jmx_pb, plan
  integer, pointer :: atom2sp(:) => null()
  real(8) :: R0(3), center(3), rcut, inte1, I1, I2, norm, R_sub(3), rlm
  real(8) :: tmin, wmin, dt, dw, dr
  complex(8), allocatable :: Vtime(:), FT(:), pre_fact(:), post_fact(:)
  real(8), allocatable :: rsh(:)


  nr = size(pb%rr)
  R0 = vnorm*time(1)*vdir + beam_offset

 
  atom2sp => get_atom2sp_ptr(pb%sv)
  natoms = size(atom2sp)

  call init_interp(pb%rr, a)

  jcutoff = get_jcutoff(pb)
  jcut_lmult = get_jcutoff_lmult(pb)
  jmx_pb = max(jcutoff, jcut_lmult)
  call init_rsphar(jmx_pb)
  
  allocate(pre_fact(ntime))
  allocate(post_fact(nff))
  pre_fact = 0.0
  post_fact = 0.0

  dr = (log(pb%rr(nr))-log(pb%rr(1)))/(nr-1)
  tmin = time(1)
  dt = time(2)-time(1)
  wmin = freq_sym(1)
  dw = freq_sym(2)-freq_sym(1)
  !ub = int(ntime/2)

  pre_fact = dt*exp(-cmplx(0.0, 1.0, 8)*wmin*(time-tmin))
  post_fact = exp(-cmplx(0.0, 1.0, 8)*freq_sym(ub:ub+nff-1)*tmin)

  !$OMP PARALLEL DEFAULT(NONE) &
  !$OMP PRIVATE(Vtime, FT, rsh, iatm, center, sp, rcut, rmax, book, si, s, fi, f) &
  !$OMP PRIVATE(nmu, mu, l, inte1, k, m, ind_lm, R_sub, norm, I1, I2, rsub_max, rlm, it) &
  !$OMP PRIVATE(plan, iw, ind) &
  !$OMP SHARED(Vfreq_real, Vfreq_imag, ntime, natoms, pb, atom2sp, R0, vnorm, vdir) &
  !$OMP SHARED(a, time, pre_fact, post_fact, nff, nprod, jmx_pb, pi, ub, nr)

  allocate(Vtime(ntime))
  allocate(FT(ntime))
  allocate(rsh(1:(jmx_pb+1)**2))
  Vtime = 0.0
  rsh = 0.0
  FT = 0.0

  !$OMP CRITICAL
  call dfftw_plan_dft_1d ( plan, ntime, Vtime, FT, FFTW_FORWARD, FFTW_ESTIMATE )
  !$OMP END CRITICAL

  !$OMP DO SCHEDULE(DYNAMIC,1)
  do iatm = 1, natoms
    center = pb%atom2coord(:, iatm)
    sp = atom2sp(iatm)
    rcut = pb%sp_local2functs(sp)%rcut
    rmax = find_nearest_index(pb%rr, rcut) -1

    book = get_book(pb, iatm)
    si = book%si(3)
    fi = book%fi(3)

    nmu = get_nmult(pb%sp_local2functs(sp))

    do mu = 1, nmu
      
      l = pb%sp_local2functs(sp)%mu2j(mu)
      s = pb%sp_local2functs(sp)%mu2si(mu)
      f = s + (2*l+1) - 1

      inte1 = 0.0

      inte1 = sum(pb%sp_local2functs(sp)%ir_mu2v(1:rmax, mu)*pb%rr(1:rmax)**(l+2)* &
                  pb%rr(1:rmax)*a%dr)

      do k = s, f

        m = mfroml(l, k-s+1)
        ind_lm = (l+1)**2 -l + m

        Vtime = 0.0
        do it = 1, ntime
          R_sub = R0 + vnorm*vdir*(time(it) - time(1)) - center
          norm = sqrt(sum(R_sub**2))

          if (norm > rcut) then
              I1 = inte1/(norm**(l+1))
              I2 = 0.0
          else
              rsub_max = find_nearest_index(pb%rr, norm)

              I1 = sum(pb%sp_local2functs(sp)%ir_mu2v(1:rsub_max, mu)* &
                      pb%rr(1:rsub_max)**(l+2)*pb%rr(1:rsub_max))
              I2 = sum(pb%sp_local2functs(sp)%ir_mu2v(rsub_max:nr, mu)* &
                      pb%rr(rsub_max:nr)/(pb%rr(rsub_max:nr)**(l-1)))
          
              I1 = I1*a%dr/(norm**(l+1))
              I2 = I2*(norm**l)*a%dr
          endif
          
          call rsphar(R_sub, jmx_pb, rsh)
          rlm = rsh(ind_lm)*(4*pi/(2*l+1))*(I1+I2)
          Vtime(it) = cmplx(rlm, 0.0, 8)

        enddo ! it

        Vtime = Vtime*pre_fact

        call dfftw_execute ( plan, Vtime, FT )
        FT(ub:ub+nff-1) = FT(ub:ub+nff-1)*post_fact
        do iw = 1, nff
          ind = si + k-1 + (iw-1)*nprod
          Vfreq_real(ind) = real(FT(ub+iw-1))
          Vfreq_imag(ind) = aimag(FT(ub+iw-1))
        enddo

      enddo ! k

    enddo ! mu
  enddo !iatm
  !$OMP END DO

  !$OMP CRITICAL
  call dfftw_destroy_plan ( plan )
  !$OMP END CRITICAL

  _dealloc(Vtime)
  _dealloc(FT)
  _dealloc(rsh)
  _dealloc(pre_fact)
  _dealloc(post_fact)
  !$OMP END PARALLEL

end subroutine ! comp_vext_tem


!
!
!
function find_nearest_index(arr, val) result(idx)

  implicit none

  real(8), intent(in) :: arr(:), val
  integer :: idx

  idx = 1 

  do while (arr(idx) < val)
    idx = idx + 1
  enddo

end function !find_nearest_index

!
!
!
function mfroml(l, idx) result(m)

  ! idx from 1 to 2l+1 (fortran counting)

  implicit none

  integer, intent(in) :: l, idx
  integer :: m

  m = -l-1 + idx

end function !mfroml

end module !m_comp_vext_tem
