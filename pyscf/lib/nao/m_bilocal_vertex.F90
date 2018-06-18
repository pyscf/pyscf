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

module m_bilocal_vertex

#include "m_define_macro.F90"
  use m_die, only : die
  use iso_c_binding, only: c_double, c_int64_t, c_int, c_double_complex
  use m_precision, only : blas_int
  use m_warn, only : warn
  use m_sph_bes_trans, only : Talman_plan_t
  use m_pair_info, only : pair_info_t
  use m_interp, only : interp_t
  use m_log, only : log_memory_note
  
  !use m_timing, only : get_cdatetime
  
  implicit none
  private die
  private warn
  !private get_cdatetime
  
  type bilocal_dominant_t
     integer               :: atoms(2)=-999
     real(8)               :: coords(3,2)=-999
     integer               :: cells(3,2)=-999
     integer               :: species(2)=-999
     integer               :: npmax=-999 ! if npmax<1 then arrays are not allocated at all
     real(8), allocatable  :: vertices(:,:,:,:)
     real(8), allocatable  :: products(:,:,:,:)
     real(8), allocatable  :: eigenvalues(:,:)
     integer, allocatable  :: m2np(:)
     real(8)               :: center(3)=-999
     real(8)               :: rcut=-999
     integer               :: ls2nrf(2)=-999 ! a correspondence : local specie (1 or 2) --> number of radial functions 
     integer, allocatable  :: rf_ls2mu(:,:) ! a correspondence : radial function, local specie (1 or 2) --> "multiplett" in system_vars_t
  end type !bilocal_dominant_t


  contains


!
!
!
subroutine dealloc_bd(bd)
  implicit none
  !! external
  type(bilocal_dominant_t), intent(inout) :: bd
  !! internal

  _dealloc(bd%vertices)
  if(allocated(bd%products))then
    !write(6,*) 'dealloc %products, ', size(bd%products)*8
    deallocate(bd%products)
  endif
  _dealloc(bd%eigenvalues)
  _dealloc(bd%m2np)
  _dealloc(bd%rf_ls2mu)
    
end subroutine ! dealloc_bd

!
!
!
subroutine dealloc(bdp)
  implicit none
  !! external
  type(bilocal_dominant_t), intent(inout), allocatable :: bdp(:)
  !! internal
  integer :: n, p
    
  _size_alloc(bdp,n)
  
  do p=1,n; call dealloc_bd(bdp(p)); enddo ! p
  
  _dealloc(bdp)
end subroutine ! dealloc

!
!
!
subroutine make_bilocal_vertex_rf(a, pair_info, &
  ff2, evals, vertex_real2, lready, rcut, center, oo2num, m2nf, &
  vertex_cmplx2, rhotb, tthr) !,tt1)
  use m_pair_info, only : pair_info_t
  use m_biloc_aux, only : biloc_aux_t  
  use m_prod_basis_type, only : prod_basis_t
  implicit none
  !! external
  type(biloc_aux_t), intent(in) :: a
  type(pair_info_t), intent(in) :: pair_info
  real(8), allocatable, intent(inout) :: evals(:,:,:), ff2(:,:,:,:,:)  ! output
  real(8), allocatable, intent(inout) :: vertex_real2(:,:,:,:,:)       ! output
  logical, intent(inout) :: lready                                     ! output
  real(8), intent(inout) :: center(:), rcut                            ! output
  integer, allocatable :: oo2num(:,:)                                  ! output
  integer, allocatable :: m2nf(:)                                      ! output
  complex(8), allocatable, intent(inout) :: vertex_cmplx2(:,:,:,:,:)
  real(8), intent(inout) :: rhotb(:,:)  
  real(8), intent(inout) :: tthr(:)!, tt1(:)
  !! internal
  real(8) :: t1, t2
  real(8), allocatable :: real_wigner(:,:)
  integer, allocatable :: rf_ls2so(:,:)

_t1
  rhotb = 0
  call comp_expansion(a, pair_info, lready, center, rcut, oo2num, m2nf, rf_ls2so, ff2,rhotb) !, tt1)
_t2(tthr(1))  
  if(lready) return;  
!  write(6,*) __FILE__, __LINE__, sum(rhotb)
 
  call comp_sph_bes_trans_expansions(a, m2nf, ff2)
_t2(tthr(2)) 
  call diag_metric(a, m2nf, ff2, evals)
_t2(tthr(3))  

  call comp_domiprod_expansions_blas(a, m2nf, evals, ff2)
_t2(tthr(4))

  call comp_vertex_rot_coord_sys(a, pair_info, oo2num, m2nf, rf_ls2so, evals, vertex_real2)
_t2(tthr(5))  

  call transform_vertex_cmplx2real(a, pair_info, m2nf, rf_ls2so, vertex_real2, vertex_cmplx2)
_t2(tthr(6))  

  vertex_real2(:,:,:,:,1) = real(vertex_cmplx2(:,:,:,:,2),8)
_t2(tthr(7))  

  call rotate_real_vertex(a, pair_info, rf_ls2so, vertex_real2, real_wigner )
_t2(tthr(8))
  
  _dealloc(real_wigner)
  _dealloc(rf_ls2so)
    
end subroutine !make_bilocal_vertex_rf

!
!
!
subroutine get_dims(a, m2nf, oo2num, evals, npmax,j12_mx,jcutoff, no)
  use m_prod_basis_param, only : get_eigmin_bilocal, get_jcutoff
  use m_pair_info, only : pair_info_t
  use m_biloc_aux, only : biloc_aux_t
  implicit none
  !! external
  type(biloc_aux_t), intent(in) :: a
  integer, allocatable, intent(in) :: m2nf(:)
  integer, allocatable, intent(in) :: oo2num(:,:)
  real(8), intent(in), allocatable :: evals(:,:,:)
  integer, intent(inout) :: npmax,j12_mx,jcutoff, no(:)
  !! internal
  real(8) :: bilocal_eigmin
  integer :: m, n, np, k, nev

  nev = size(evals,2)
  jcutoff = get_jcutoff(a%pb_p)

  !!!!! Fill in the structure !!!!
  j12_mx = ubound(m2nf,1)
  bilocal_eigmin = get_eigmin_bilocal(a%pb_p)  
  npmax = 0;
  do m=-j12_mx, j12_mx
    n = m2nf(m)
    if(n<1) cycle
    np=0
    do k=1,n
      if(evals(k,nev,m)<=bilocal_eigmin) cycle
      np=np+1
    enddo ! k
    npmax = max(np, npmax);
  enddo ! m

  !write(ilog,*) 'make_bilocal_vertex_for_atom_pair:', npmax, nf_max, atom_pair
  if(npmax<1) return
  if(.not. allocated(oo2num)) _die('!oo2num')
  no = ubound(oo2num)
  
end subroutine ! get_dims


!
!
!
subroutine init_bilocal_dominant(a, inf, oo2num, m2nf, evals, ff2, vertex_real2, lready, rcut, center, d)
  use m_prod_basis_param, only : get_eigmin_bilocal, get_jcutoff
  use m_pair_info, only : pair_info_t
  use m_biloc_aux, only : biloc_aux_t
    
  implicit none
  !! external
  type(biloc_aux_t), intent(in) :: a
  type(pair_info_t), intent(in) :: inf
  integer, allocatable, intent(in) :: oo2num(:,:)  
  integer, allocatable, intent(in) :: m2nf(:)      
  real(8), intent(in), allocatable :: evals(:,:,:)
  real(8), intent(in), allocatable :: ff2(:,:,:,:,:)
  real(8), intent(in), allocatable :: vertex_real2(:,:,:,:,:)
  logical, intent(in) :: lready
  real(8), intent(in) :: rcut, center(:)
  type(bilocal_dominant_t), intent(inout) :: d
  !! internal
  real(8) :: bilocal_eigmin
  integer :: npmax, m, n, np, k, nev, jcutoff, nn(2), j12_mx, no(2)

  d%rcut = rcut
  d%center = center
  if(lready) then;  d%npmax = -1;   return;  endif

  nev = size(evals,2)
  jcutoff = get_jcutoff(a%pb_p)

  !!!!! Fill in the structure !!!!
  j12_mx = ubound(m2nf,1)
  bilocal_eigmin = get_eigmin_bilocal(a%pb_p)  
  npmax = 0;
  do m=-j12_mx, j12_mx
    n = m2nf(m);
    if(n<1) cycle
    np=0
    do k=1,n
      if(evals(k,nev,m)<=bilocal_eigmin) cycle
      np=np+1
    enddo ! k
    npmax = max(np, npmax);
  enddo ! m

  !write(ilog,*) 'make_bilocal_vertex_for_atom_pair:', npmax, nf_max, atom_pair
  d%npmax=npmax;
  if(npmax<1) return
  if(.not. allocated(oo2num)) _die('! %oo2num')
  no = ubound(oo2num)
  
  allocate(d%eigenvalues(npmax,-j12_mx:j12_mx));
  allocate(d%products(a%nr,0:jcutoff,npmax,-j12_mx:j12_mx));
  allocate(d%vertices(no(1),no(2),npmax,-j12_mx:j12_mx));
  allocate(d%m2np(-j12_mx:j12_mx));

  d%vertices = 0
  d%products = 0
  d%eigenvalues = 0
  d%coords(1:3,1:2) = inf%coords(1:3,1:2)
  d%species(1:2) = inf%species(1:2)
  d%atoms(1:2) = inf%atoms(1:2)
  d%cells(1:3,1:2) = inf%cells(1:3,1:2)
  d%ls2nrf(1:2) = inf%ls2nrf(1:2)

  if( allocated(inf%rf_ls2mu) ) then
    nn = ubound(inf%rf_ls2mu)
_dealloc_u(d%rf_ls2mu, nn)
    if(.not. allocated(d%rf_ls2mu)) allocate(d%rf_ls2mu(nn(1),nn(2)))
    d%rf_ls2mu = inf%rf_ls2mu
  else
    _dealloc(d%rf_ls2mu)
  endif

  do m=-j12_mx,j12_mx
    n= m2nf(m)
    if(n<1) cycle
    np=0
    do k=1,n
      if(evals(k,nev,m)<=bilocal_eigmin) cycle
      np=np+1
      d%eigenvalues(np,m)               = evals(k,nev,m);
      d%products(1:a%nr,0:jcutoff,np,m) = ff2(1:a%nr,0:jcutoff,k,m,2);
      d%vertices(1:no(1),1:no(2),np,m)  = vertex_real2(m,k,1:no(1),1:no(2),2);
    enddo ! mprod
    d%m2np(m) = np;
  enddo ! m

end subroutine ! init_bilocal_dominant

!
!
!
subroutine rotate_real_vertex(a, inf, rf_ls2so, vertex_real2, real_wigner )
  use m_pair_info, only : pair_info_t
  use m_biloc_aux, only : biloc_aux_t
  implicit none
  !! external
  type(biloc_aux_t), intent(in) :: a
  type(pair_info_t), intent(in) :: inf
  integer, allocatable, intent(in) :: rf_ls2so(:,:)
  real(8), intent(inout), allocatable :: vertex_real2(:,:,:,:,:)
  real(8), intent(inout), allocatable :: real_wigner(:,:)
  !! internal
  real(8) :: trans_vec(3)
  integer :: sp(2), nrf(2), rf1, rf2, mu1, mu2, j1, j2, c1, c2
  integer :: jj1, jj2, m1, m2, k1, k2

  nrf(1:2) = inf%ls2nrf(1:2)
  sp(1:2) = inf%species(1:2)
  
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!   Jetzt Vertex selbst veraendern, so dass die Faktoren   !!!!!!!!
  !!!!!!!!!!!   des Produkts in einem festen Koordinatensystem bleiben !!!!!!!!
  trans_vec = inf%coords(:,2)-inf%coords(:,1)
  call init_real_wigner(trans_vec, a%jmx, real_wigner)

  vertex_real2(:,:,:,:,2) = 0 ! initialisieren
  do rf2=1,nrf(2)
    mu2 = inf%rf_ls2mu(rf2, 2)
    j2  = a%mu_sp2j(mu2,sp(2))
    jj2 = j2*(j2+1)
    c2  = rf_ls2so(rf2,2)+j2
    do rf1=1,nrf(1)
      mu1 = inf%rf_ls2mu(rf1, 1) 
      j1 = a%mu_sp2j(mu1,sp(1))
      jj1 = j1*(j1+1)
      c1 = rf_ls2so(rf1,1)+j1
      do m2=-j2,j2
        do m1=-j1,j1
          do k1=-j1,j1   ! faster in this order.
            do k2=-j2,j2

              vertex_real2(:,:,c1+m1,c2+m2,2) = vertex_real2(:,:,c1+m1,c2+m2,2)+  &
                real_wigner(k1,jj1+m1) * real_wigner(k2,jj2+m2) * &
                vertex_real2(:,:,c1+k1,c2+k2,1)
                
            enddo
          enddo ! k1,k2
        enddo
      enddo ! m1,m2
    enddo ! mu2
  enddo ! mu1

end subroutine ! rotate_real_vertex

!
!
!
subroutine transform_vertex_cmplx2real(a, inf, m2nf, rf_ls2so, vertex_real2, vertex_cmplx2)
  use m_pair_info, only : pair_info_t
  use m_biloc_aux, only : biloc_aux_t  
  implicit none 
  !! external
  type(biloc_aux_t), intent(in) :: a
  type(pair_info_t), intent(in) :: inf
  integer, allocatable, intent(in) :: m2nf(:)
  integer, allocatable, intent(in) :: rf_ls2so(:,:)
  real(8), intent(in), allocatable :: vertex_real2(:,:,:,:,:)
  complex(8), intent(inout), allocatable :: vertex_cmplx2(:,:,:,:,:)
  !! internal
  integer :: mu1, mu2, j1, j2, m1, m2, k2, k1, c1, c2, nrf(2), sp(2), m, rf1, rf2
  integer :: j12_mx
  
  nrf(1:2) = inf%ls2nrf(1:2)
  sp(1:2) = inf%species(1:2)
 
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Transformation Complex Y ->  Real Y fuer den komplexen Vertex  ausfuehren  !!!
  ! erste Etappe : Faktoren fuer reelle Y !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  vertex_cmplx2(:,:,:,:,1)=0
  do rf2=1,nrf(2)
    mu2 = inf%rf_ls2mu(rf2, 2)
    j2=a%mu_sp2j(mu2,sp(2))
    c2=rf_ls2so(rf2,2)+j2
    
  do rf1=1,nrf(1)
    mu1 = inf%rf_ls2mu(rf1, 1)
    j1=a%mu_sp2j(mu1,sp(1));
    c1=rf_ls2so(rf1,1)+j1
    
    do m1=-j1,j1
      do k1=-j1,j1
        if(abs(m1)/=abs(k1)) cycle

        do m2=-j2,j2
        do k2=-j2,j2
          if(abs(m2)/=abs(k2)) cycle

          vertex_cmplx2(:,:,c1+m1,c2+m2,1)= vertex_cmplx2(:,:,c1+m1,c2+m2,1)+  &
            a%c2r(m1,k1)*a%c2r(m2,k2) * vertex_real2(:,:,c1+k1,c2+k2,2);

        enddo; enddo ! k1,k2
      enddo;
     enddo ! m1,m2
  enddo ! mu2
  enddo ! mu1

  vertex_cmplx2(:,:,:,:,2) = 0
  vertex_cmplx2(0,:,:,:,2) = vertex_cmplx2(0,:,:,:,1);
  j12_mx = ubound(m2nf,1)
  do m=-j12_mx, j12_mx
    if (m==0) cycle
    do m2=-m,m,2*abs(m)
      do m1=-m,m,2*abs(m)
        vertex_cmplx2(m1,:,:,:,2) = vertex_cmplx2(m1,:,:,:,2) + &
          conjg( a%c2r(m1,m2) )*vertex_cmplx2(m2,:,:,:,1)
      enddo
    enddo ! m1,m2
  enddo ! m

end subroutine ! !!! transform_vertex_cmplx2real


!
!
!
subroutine comp_vertex_rot_coord_sys(a, inf, oo2num, m2nf, rf_ls2so, evecs, vertex_real2)
  use m_pair_info, only : pair_info_t
  use m_biloc_aux, only : biloc_aux_t  
  implicit none
  !! external
  type(biloc_aux_t), intent(in) :: a
  type(pair_info_t), intent(in) :: inf
  integer, intent(in), allocatable :: oo2num(:,:)
  integer, intent(in), allocatable :: m2nf(:)
  integer, intent(in), allocatable :: rf_ls2so(:,:)
  real(8), intent(in), allocatable :: evecs(:,:,:)
  real(8), intent(inout), allocatable :: vertex_real2(:,:,:,:,:)
  !! internal
  integer :: rf1, rf2
  integer :: mu1, mu2, j1, j2, m1, m2, m, o1, o2, num, n, c1, c2, nrf(2), sp(2)

  nrf(1:2) = inf%ls2nrf(1:2)
  sp(1:2) = inf%species(1:2)
     
  vertex_real2(:,:,:,:,1)  = 0
  do rf2=1,nrf(2)
    mu2 = inf%rf_ls2mu(rf2, 2)
    j2 = a%mu_sp2j(mu2,sp(2))
    do m2=-j2,j2
      o2 = rf_ls2so(rf2,2)+j2+m2
      
      do rf1=1,nrf(1)
        mu1 = inf%rf_ls2mu(rf1, 1)
        j1 = a%mu_sp2j(mu1,sp(1));
        do m1=-j1,j1
          o1 = rf_ls2so(rf1,1)+j1+m1
          
          m = m1+m2
          n = m2nf(m)
          num = oo2num(o1,o2)
          vertex_real2(m,1:n,o1,o2,1)=evecs(num,1:n,m)
        enddo ! m1
      enddo ! mu1 
    enddo ! m2
  enddo ! mu2

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!!!!!!!!!!!!!!!!! 'Korrekt konjugierte' komplexe Vertizes definieren               !!!!!!!!!!!
  vertex_real2(:,:,:,:,2)=0
  do rf2=1,nrf(2)
    mu2 = inf%rf_ls2mu(rf2, 2)
    j2 = a%mu_sp2j(mu2,sp(2))
    c2 = rf_ls2so(rf2,2)+j2
    
    do rf1=1,nrf(1)
      mu1 = inf%rf_ls2mu(rf1, 1)
      j1 = a%mu_sp2j(mu1,sp(1))
      c1 = rf_ls2so(rf1,1)+j1
      
      do m2=-j2,j2
        do m1=-j1,j1
          m=m1+m2

          if (m >=0) then
            vertex_real2(m,:,c1+m1,c2+m2,2) = vertex_real2( m,:,c1+m1,c2+m2,1)
          else  !  m< 0
            vertex_real2(m,:,c1+m1,c2+m2,2) = vertex_real2(-m,:,c1-m1,c2-m2,1)
          endif

        enddo ! m2
      enddo ! m1
    enddo ! mu2
  enddo ! mu1

end subroutine !comp_vertex_rot_coord_sys



!
!
!
subroutine comp_domiprod_expansions(au, m2nf, evecs, ff2)
  use m_biloc_aux, only : biloc_aux_t
  implicit none
  !! external
  type(biloc_aux_t), intent(in) :: au
  integer, intent(in), allocatable :: m2nf(:)
  real(8), intent(in), allocatable :: evecs(:,:,:)
  real(8), intent(inout), allocatable :: ff2(:,:,:,:,:)
  !! internal
  integer :: m, lbd, m_mi, m_mx, jcutoff, nr, n, a
  nr = au%nr
  jcutoff = ubound(ff2,2)
  m_mi = lbound(m2nf,1)
  m_mx = ubound(m2nf,1)

  ff2(:,:,:,:,2) = 0
  do m=m_mi,m_mx
    n = m2nf(m)
    do lbd=1,n
      do a=1,n
        ff2(:,0:jcutoff,lbd,m,2)= ff2(:,0:jcutoff,lbd,m,2)+ &
          & evecs(a,lbd,m)*ff2(:,0:jcutoff,a,m,1)
      enddo ! nummer
    enddo ! dominant

    if(m>0)ff2(:,:,1:n,-m,2) = ff2(:,:,1:n,m,2)
  enddo ! m

end subroutine ! comp_domiprod_expansions

!
!
!
subroutine comp_domiprod_expansions_blas(au, m2nf, evecs, ff2)
  use m_biloc_aux, only : biloc_aux_t
  implicit none
  !! external
  type(biloc_aux_t), intent(in) :: au
  integer, intent(in), allocatable :: m2nf(:)
  real(8), intent(in), allocatable :: evecs(:,:,:)
  real(8), intent(inout), allocatable :: ff2(:,:,:,:,:)
  !! internal
  integer :: m_mi, m_mx, jcutoff, nr, mm
  integer(blas_int) :: M,N,K, lde
  nr = au%nr
  jcutoff = ubound(ff2,2)
  m_mi = lbound(m2nf,1)
  m_mx = ubound(m2nf,1)

  !ff2(:,:,:,:,2) = 0
  lde = size(evecs,1)
  do mm=m_mi,m_mx
    M = nr*(jcutoff+1)
    N = m2nf(mm)
    K = N
    call DGEMM('N', 'N', M,N,K, 1d0, ff2(:,:,:,mm,1),M, evecs(:,:,mm),lde, 0d0, ff2(:,:,:,mm,2), M)
    if(mm>0)ff2(:,:,1:n,-mm,2) = ff2(:,:,1:n,mm,2)
  enddo ! m

end subroutine ! comp_domiprod_expansions_blas


!
!
!
subroutine diag_metric(au, m2nf,ff2, evecs)
  use m_biloc_aux, only : biloc_aux_t
  implicit none
  !! external
  type(biloc_aux_t), intent(in) :: au
  integer, intent(in), allocatable :: m2nf(:)
  real(8), intent(in), allocatable :: ff2(:,:,:,:,:)
  real(8), intent(inout), allocatable :: evecs(:,:,:)
  !! internal
  integer :: m, j, a,b, nf_max, nev, m_mi, m_mx, jcutoff, nr
  real(8), allocatable :: WORK(:)
  integer(blas_int) :: LWORK, INFO, n
  real(8) :: pi, aux
  pi = 4D0*atan(1D0)
  nr = au%nr
  jcutoff = ubound(ff2,2)
  m_mi = lbound(m2nf,1)
  m_mx = ubound(m2nf,1)
  nf_max = size(evecs,1)
  nev = size(evecs,2)
  if(nev/=nf_max+1) _die('!nev')

  lwork = au%lwork
  allocate(WORK(lwork))
    
  !!
  !! Bestimmung der Matrix der Skalarprodukte aus den Produkten !!!!!
  !! Determine the metric using expansions / diagonalize the metric 
  !!
  evecs=0
  do m = m_mi,m_mx
    n = m2nf(m)
    if (n==0) cycle
    do b=1,n
      do a=1,b
        aux=0
        do j=0,jcutoff
          aux = aux + (4*pi/(2*j+1D0)) * &
            sum(ff2(1:nr,j,a,m,2) * ff2(1:nr,j,b,m,2) * au%dkappa_pp);
        enddo
        evecs(a,b,m) = aux
      enddo ! a
    enddo ! b
    INFO = 0
    call DSYEV("V", "U", n, evecs(:,:,m), nf_max, evecs(:,nev,m), WORK, LWORK, INFO )
    if(info/=0) then
       write(0,'(a,i9)') 'diag_metric: info', INFO
       _die('!diag_metric')
    endif
  enddo ! mz
  
  _dealloc(work)
  
end subroutine !diag_metric

!
!
!
subroutine comp_sph_bes_trans_expansions(a, m2nf, ff2)
  use m_sph_bes_trans, only : sbt_execute
  use m_biloc_aux, only : biloc_aux_t  
  implicit none
  !! external
  type(biloc_aux_t), intent(in) :: a
  integer, intent(in), allocatable :: m2nf(:)
  real(8), intent(inout), allocatable :: ff2(:,:,:,:,:)
  !! internal
  integer :: nr, m, m_mx, m_mi, n, jcutoff, j, k
  nr = a%nr
  jcutoff = ubound(ff2,2)
  m_mi = lbound(m2nf,1)
  m_mx = ubound(m2nf,1)
  !! Convert from coordinate to momentum space
  do m=m_mi,m_mx
    n = m2nf(m)
    do k=1,n
      do j=0,jcutoff
        call sbt_execute(a%Talman_plan, ff2(1:nr,j,k,m,1), ff2(1:nr,j,k,m,2), j,1)
      enddo ! j
    enddo ! k
  enddo ! m
  !! END of Convert from coordinate to momentum space 

end subroutine ! comp_sph_bes_trans_expansions

!
!
!
subroutine comp_expansion(a,inf, lready,center,rcut, oo2num,m2nf,rf_ls2so,ff2, rhotb)
  use m_prod_basis_param, only : get_jcutoff, get_GL_ord_bilocal
  use m_orb_rspace_type, only : get_rho_min_jt, get_dr_jt
  use m_interpolation, only : find
  use m_prod_talman, only : prdred, all_interp_coeffs, prdred_all_interp_coeffs, all_interp_values, prdred_all_interp_values, all_interp_values1, prdred_all_interp_values1 
  use m_thrj_nobuf, only : thrj
  use m_pair_info, only : pair_info_t, get_rf_ls2so
  use m_biloc_aux, only : biloc_aux_t  

  implicit none
  !! external
  type(biloc_aux_t), intent(in) :: a
  type(pair_info_t), intent(in) :: inf
  logical, intent(inout) :: lready
  real(8), intent(inout) :: center(3), rcut
  integer, allocatable, intent(inout) :: oo2num(:,:)           ! output
  integer, allocatable, intent(inout) :: m2nf(:)               ! output
  integer, allocatable, intent(inout) :: rf_ls2so(:,:)         ! output  
  real(8), intent(inout), allocatable, target :: ff2(:,:,:,:,:)
  !! auxiliary
  real(8), intent(inout) :: rhotb(:,:)
  !real(8), intent(inout) :: tt(:)
  
  !! internal
  integer :: sp(2), mu1, mu2, j1, j2, jcutoff, nr, ix, j, c1, c2, jmx12, no(2), mu!, k
  integer :: clbd, lbd, nterm, o1, o2, m1, m2, m, num, irc, nrf(2), jmax(2)
  integer :: rf1, rf2, ir, nrmx
  integer :: ls ! L[ocal] S[pecie] : i.e. 1 or 2
  integer :: rf ! R[adial] F[unction] : i.e. a pointer in the list of radial functions (multipletts)
  real(8) :: rc2_new, trans_vec(3), d12, rcuts(2), wghts(2)
!  real(8) :: t1, t2
  real(8) :: Ra(3), Rb(3), rho_min_jt, dr_jt, tj1, tj2, pi, fact_z1
  real(8), allocatable :: FFr(:,:), ixrj2ck(:,:,:,:), fval(:,:), yz(:)
  real(8), allocatable :: xrjm2f1(:,:,:,:), xrjm2f2(:,:,:,:)
  real(8), allocatable :: xrm2f1(:,:,:), xrm2f2(:,:,:)
  integer, allocatable :: jtb(:), clbdtb(:), lbdtb(:), m2nrmx1(:), m2nrmx2(:)
  real(8), parameter :: zerovec(3) = (/0.0D0, 0.0D0, 0.0D0/);
  
  if(.not. allocated(inf%rf_ls2mu))_die('!rf_ls2mu')
  if(any(inf%ls2nrf<1))_die('ls2nrf<1')
   
!  _t1 
  pi = 4D0*atan(1d0)
  lready = .false.
  jcutoff = get_jcutoff(a%pb_p)
  trans_vec = inf%coords(1:3,2)-inf%coords(1:3,1)
  d12 = sqrt(sum(trans_vec**2));

  sp(1:2) = inf%species(1:2)
  wghts = get_inv_wghts( a, inf )
  fact_z1 = wghts(2) / ( wghts(1)+wghts(2) );
  !write(6,*) 'fact_z1', fact_z1
  !write(6,'(2es20.12,3x,2es20.12)') radii, a%sp2radii(sp(1:2))
  
  center = inf%coords(1:3,1) + (1-fact_z1)* trans_vec

  nr = a%nr
  rho_min_jt = get_rho_min_jt(a%orb_a)
  dr_jt = get_dr_jt(a%orb_a)
  nrf(1:2) = inf%ls2nrf(1:2)

  !! figure out rcut(ls)
  rcuts(1:2) = 0
  do ls=1,2
    do rf=1,nrf(ls)
      mu = inf%rf_ls2mu(rf,ls)
      rcuts(ls) = max(rcuts(ls), a%mu_sp2rcut(mu, sp(ls)))
    enddo ! rf
  enddo ! ls
  !! END of figure out rcut(ls)

  !! Compute the cutoff according to radii, check, if not overlapping return
  rc2_new = maxval(rcuts)**2 - d12**2/4
  if(rc2_new<0) then
    !write(6,'(a35,4g20.10)') 'rcuts, d12, rc2_new: ', rcuts, d12, rc2_new
    !_warn('what a case, rc2_new<0!')
    rcut = -999
    lready = .true.
    return
  endif
  rcut = sqrt(rc2_new)
  !! END of Compute the cutoff according to radii, check, if not overlapping return
  
  !! figure out jmx1+jmx2
  do ls=1,2
    jmax(ls) = maxval(a%mu_sp2j(inf%rf_ls2mu(1:nrf(ls), ls), sp(ls)))
  enddo
  jmx12 = sum(jmax)
  !! END of figure out jmx1+jmx2

  !! figure out no(ls) number of orbitals for the first and second specie
  no(1:2) = 0
  do ls=1,2
    do rf=1,nrf(ls)
      mu = inf%rf_ls2mu(rf,ls)
      j = a%mu_sp2j(mu, sp(ls))
      no(ls) = no(ls) + 2*j+1
    enddo ! rf
  enddo ! ls
  !! END of figure out no(ls) number of orbitals for the first and second specie

  !! Allocate external variables (belong to output)
  _dealloc(oo2num)
  _dealloc(m2nf)
  _dealloc(rf_ls2so)
  
  allocate(oo2num(no(1), no(2)))
  allocate(m2nf(-jmx12:jmx12));
  
  !! figure out rf_ls2so, i.e. correspondence between radial function number, local specie --> start orbital
  allocate(rf_ls2so(maxval(nrf), 2))
  call get_rf_ls2so(sp, nrf, inf%rf_ls2mu, a%mu_sp2j, rf_ls2so)
  !! END of !! figure out rf_ls2so, i.e. correspondence between radial function number, local specie --> start orbital
  
  !! Internal allocatations
  allocate(FFr(nr,0:jcutoff))
  allocate(jtb(a%nterm_max))
  allocate(clbdtb(a%nterm_max))
  allocate(lbdtb(a%nterm_max))

  Ra = [ 0.0D0, 0.0D0, -d12*(1D0-fact_z1) ]
  Rb = [ 0.0D0, 0.0D0,  d12*(fact_z1)     ]

  allocate(ixrj2ck(7,a%ord,a%nr,2))
  call all_interp_coeffs(Ra,Rb,zerovec,a%rr,nr,a%xgla,a%sqrt_wgla,a%ord,ixrj2ck)
  
  !allocate(xrjm2f1(a%ord,a%nr,2,nrf(1)))
  !allocate(xrjm2f2(a%ord,a%nr,2,nrf(2)))
  !call all_interp_values(a%psi_log_rl(:,1:nrf(1),sp(1)),nr,nrf(1), ixrj2ck, a%ord, xrjm2f1)
  !call all_interp_values(a%psi_log_rl(:,1:nrf(2),sp(2)),nr,nrf(2), ixrj2ck, a%ord, xrjm2f2)

  allocate(xrm2f1(a%ord,a%nr,nrf(1)))
  allocate(xrm2f2(a%ord,a%nr,nrf(2)))
  call all_interp_values1(a%psi_log_rl(:,1:nrf(1),sp(1)),nr,nrf(1), ixrj2ck(:,:,:,2), a%ord, xrm2f1)
  call all_interp_values1(a%psi_log_rl(:,1:nrf(2),sp(2)),nr,nrf(2), ixrj2ck(:,:,:,1), a%ord, xrm2f2)

  allocate(m2nrmx1(nrf(1)))
  m2nrmx1 = a%nr
  do mu=1,nrf(1)
    do ir=a%nr,1,-1; 
      if(any(xrm2f1(:,ir,mu)/=0)) then
        m2nrmx1(mu)=ir
        exit
      endif
    enddo
  enddo
  
  allocate(m2nrmx2(nrf(2)))
  m2nrmx2 = a%nr
  do mu=1,nrf(2)
    do ir=a%nr,1,-1; 
      if(any(xrm2f2(:,ir,mu)/=0)) then
        m2nrmx2(mu)=ir
        exit 
      endif
    enddo
  enddo

!  write(6,*) m2nrmx1
!  write(6,*) m2nrmx2
  
  allocate(fval(a%nr,0:2*(a%jcutoff+jmx12)))
  allocate(yz(a%ord))
  
!  _t2(tt(1))
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !! Orbitale fuer Atom_1  und  Atom_2 - um r=0 entwickeln. Konvention:  1: (0,0,-d12/2) 2: (0,0,d12/2) !!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ff2(:,:,:,:,1) = 0

  m2nf = 0;
  oo2num = -999
  do rf1=1,nrf(1)
    mu1 = inf%rf_ls2mu(rf1,1)
    j1 = a%mu_sp2j(mu1,sp(1))
    c1 = rf_ls2so(rf1,1)+j1
    
    do rf2=1,nrf(2)
      mu2 = inf%rf_ls2mu(rf2,2)
      j2 = a%mu_sp2j(mu2,sp(2))
      c2 = rf_ls2so(rf2,2)+j2
  
      !! Compute the X_{j Lambda Lambda'}(r, R, alpha)
      call init_jt_aux(j1,j2,jcutoff, nterm,jtb,clbdtb,lbdtb)

!      write(6,*) __FILE__, __LINE__, rf1, rf2, &
!        sum(a%psi_log(1:nr,mu2,sp(2))), sum(a%psi_log(1:nr,mu1,sp(1)))

!      call prdred( &
!        a%psi_log(1:nr,mu2,sp(2)),j2,Rb,  a%psi_log(1:nr,mu1,sp(1)),j1,Ra, &
!        zerovec, jcutoff, rhotb, a%rr, nr, jtb, clbdtb, lbdtb, nterm, &
!        a%xgla, a%wgla, a%ord, rho_min_jt, dr_jt);

!      call prdred_all_interp_coeffs( &
!        a%psi_log_rl(1:nr,mu2,sp(2)),j2,Rb,  a%psi_log_rl(1:nr,mu1,sp(1)),j1,Ra, &
!        zerovec, jcutoff, rhotb, a%rr, nr, jtb, clbdtb, lbdtb, nterm, &
!        a%ord, a%plval, a%jmax_pl, ijxr2ck)
      
!      _t1
!      call prdred_all_interp_values(xrjm2f2(:,:,:,mu2),j2,Rb, xrjm2f1(:,:,:,mu1),j1,Ra, &
!        zerovec, jcutoff, rhotb, a%rr, nr, jtb, clbdtb, lbdtb, nterm, &
!        a%ord, a%plval, a%jmax_pl, fval, yz)
!      _t2(tt(2))  

      nrmx = min(m2nrmx1(mu1),m2nrmx2(mu2))
      call prdred_all_interp_values1(xrm2f2(:,:,mu2),j2,Rb,nrmx, xrm2f1(:,:,mu1),j1,Ra, &
        zerovec, jcutoff, rhotb, a%rr, nr, jtb, clbdtb, lbdtb, nterm, &
        a%ord, a%plval, a%jmax_pl, fval, yz)

      !! END of Compute the X_{j Lambda Lambda'}(r, R, alpha)
      
!      write(6,*) __FILE__, __LINE__
!      write(6,*) sum(a%psi_log)
!      write(6,*) sum(Rb), j2
!      write(6,*) sum(Ra), j1
!      write(6,*) sum(zerovec), jcutoff
!      write(6,*) ord, pcs, rho_min_jt, dr_jt
!      write(6,*) sum(a%rr), sum(a%pp)
      do m1=-j1,j1
        o1 = c1 + m1
        
        do m2=-j2,j2
          o2 = c2 + m2
          
          m = m1 + m2
          ! Anzahl der Multipletts fuer diesen Wert von m vergroessern
          m2nf(m)=m2nf(m)+1;
          num=m2nf(m);
          oo2num(o1,o2) = num

          !! Reduce to the FFr
          FFr = 0;
          !! cure -- after irc point the product-function is zero
          irc = find(nr, a%rr, sqrt(rc2_new)); ! maybe chosen better!!
          do ix=1, nterm
            j    = jtb(ix);
            clbd = clbdtb(ix);
            lbd  = lbdtb(ix);
            tj1  = thrj(j1,j2,j,m1,m2,-m);
            tj2  = thrj(j,clbd,lbd,-m,m,0);
            FFr(1:irc,clbd)=FFr(1:irc,clbd) + tj1*tj2*rhotb(1:irc, ix)*dble(a%ylm_cr(lbd*(lbd+1)))
          end do;
          FFr(1:irc,0:jcutoff) = FFr(1:irc,0:jcutoff)*sqrt((2*j1+1.0D0)*(2*j2+1.0D0))/(4*pi)
          
!          if(m1==0 .and. m2==0 .and. j1==0 .and. j2==0 .and. mu2==1 .and. mu1==1) then
            
!            do k=0,jcutoff
!              write(6,*) k, sum(abs(FFr(:,k)))!, Rb, Ra, 'only k==0?'
!            enddo

!  write(6,*) sum(a%psi_log(1:nr,mu2,sp(2)))
!  write(6,*) j2
!  write(6,*) rb
!  write(6,*) sum(a%psi_log(1:nr,mu1,sp(1)))
!  write(6,*) j1
!  write(6,*) ra
!  write(6,*) zerovec
!  write(6,*) jcutoff
!  write(6,*) sum(a%rr)
!  write(6,*) nr
!  write(6,*) jtb(1:nterm)
!  write(6,*) clbdtb(1:nterm)
!  write(6,*) lbdtb(1:nterm)
!  write(6,*) nterm
!  write(6,*) ord
!  write(6,*) pcs
!  write(6,*) rho_min_jt
!  write(6,*) dr_jt
!  write(6,*) sum(rhotb(:,1:nterm))
!  write(6,*) dble(a%ylm_cr(:))
 

!          !! Reduce to the FFr
!          FFr = 0;
!          !! cure -- after irc point the product-function is zero
!          irc = find(nr, a%rr, sqrt(rc2_new)); ! maybe chosen better!!
!          do ix=1, nterm
!            j    = jtb(ix);
!            clbd = clbdtb(ix);
!            lbd  = lbdtb(ix);
!            tj1  = thrj(j1,j2,j,m1,m2,-m);
!            tj2  = thrj(j,clbd,lbd,-m,m,0);
!            FFr(1:irc,clbd)=FFr(1:irc,clbd) + tj1*tj2*rhotb(1:irc, ix)*dble(a%ylm_cr(lbd*(lbd+1)))
!          end do;

!            do k=0,jcutoff
!              write(6,*) k, sum(abs(FFr(:,k)))!, Rb, Ra, 'only k==0?'
!            enddo

!          endif 
          
          ff2(1:irc,0:jcutoff,num,m,1)=FFr(1:irc,0:jcutoff)
        enddo ! m2
      enddo ! m1
!      _t2(tt(3))
    enddo ! mu2
  enddo! mu1

  _dealloc(FFr)
  _dealloc(jtb)
  _dealloc(clbdtb)
  _dealloc(lbdtb)
  _dealloc(ixrj2ck)
  _dealloc(xrjm2f1)
  _dealloc(xrjm2f2)
  _dealloc(fval)
  _dealloc(yz)
  _dealloc(m2nrmx1)
  _dealloc(m2nrmx2)

  
end subroutine !comp_expansion 

!
!
!
subroutine init_jt_aux(j1,j2,jcutoff,nterm,jtb,clbdtb,lbdtb)
  implicit none
  !! external
  integer, intent(in) :: j1,j2,jcutoff
  integer, intent(inout) :: nterm
  integer, intent(inout) :: jtb(:), clbdtb(:), lbdtb(:)
  !! internal
  integer :: ij, lbd, clbd, ijmx
  
  !! Compute the X_{j Lambda Lambda'}(r, R, alpha)
  nterm=0
  ijmx = j1+j2
  do ij=abs(j1-j2),ijmx
    do clbd=0,jcutoff
      do lbd=abs(clbd-ij),clbd+ij
        nterm         = nterm+1
        jtb(nterm)    = ij
        clbdtb(nterm) = clbd
        lbdtb(nterm)  = lbd
      enddo
    enddo
  enddo

end subroutine !   init_jt_aux


!
!
!
subroutine init_real_wigner(trans_vec, jmx, real_wigner)
  use m_wigner_rotation, only : simplified_wigner
    
  implicit none
  !! external
  real(8), intent(in) :: trans_vec(3)
  integer, intent(in) :: jmx
  real(8), intent(inout), allocatable :: real_wigner(:,:)
  real(8) :: dR(3)
  !! internal
  integer :: j, m
  real(8), allocatable :: rwigner_aux(:,:,:)
  
  !! Initialization of real_wigner
_dealloc(real_wigner)
  allocate(real_wigner(-jmx:jmx, 0:(jmx+1)**2))
  real_wigner = 0;
  
  allocate(rwigner_aux(-jmx:jmx,-jmx:jmx,2))
    
  dR = trans_vec
  if(sum(dR**2)==0D0) dR = (/0D0, 0D0, 1.0D0/);
  
  do j=0,jmx;
    call simplified_wigner(dR, j, rwigner_aux(-j:j,-j:j,1),rwigner_aux(-j:j,-j:j,2));
    do m=-j,j; real_wigner(-j:j,j*(j+1)+m) = rwigner_aux(m,-j:j,1); enddo ! m
  enddo ! j
  !! END of Initialization of real_wigner_fast

  _dealloc(rwigner_aux)
  
end subroutine ! init_real_wigner


!
! This delivers a weight to be used in calculating center of expansion of bilocals 
! center will be determined by 
! R_c = c1 R1_vec + c2 R2_vec
! where c1 and c2 are coefficients 
! c1 = a / weight1, c2 = a/weight2 and c1 + c2 = 1 (because R_c should be on the 
! line connecting R1_vec and R2_vec)
! 
function get_inv_wghts( a, inf ) result(inv_wght)
  use m_prod_basis_param, only : get_optimize_centers, get_bilocal_center
  use m_interp, only : comp_coeff_m2p3_k
  use m_system_vars, only : get_psi_log_ptr, get_mu_sp2j_ptr, get_jmx
  use m_harmonics, only : get_lu_slm, rsphar
  use m_biloc_aux, only : biloc_aux_t
  implicit none
  !! external
  type(biloc_aux_t), intent(in), target :: a
  type(pair_info_t), intent(in) :: inf
  real(8) :: inv_wght(2)
  !! internal
  integer :: iopt, ls, sp, nrf, rf, mu, ir, rf1, rf2
  integer :: npnt, k1, k2, ssp(2), mmu(2), nnrf(2), ir_loc, jmx, jj(2), lu(2)
  character(100) :: ccenter
  real(8), allocatable :: r2ff(:), rmm2ff(:,:,:), slm1(:), slm2(:)
  real(8) :: R1(3), R2(3), D12, r, r1_sqr, r2_sqr, coeff1(-2:3), coeff2(-2:3)
  real(8) :: f1, f2, r_mean, dr_lin, rvec(3), rvec1(3), rvec2(3)
  real(8), pointer :: psi_log(:,:,:)
  integer, pointer :: mu_sp2j(:,:)
  
  iopt = get_optimize_centers(a%pb_p)
  if(iopt<1) then
    inv_wght = 1D0
    return
  endif
  
  ccenter = get_bilocal_center(a%pb_p)
  if(ccenter=="MAXLOC_BILOCAL") then
    R1 = inf%coords(1:3,1)
    R2 = inf%coords(1:3,2)
    D12 = sqrt(sum((R2 - R1)**2))
    psi_log => get_psi_log_ptr(a%sv)
    mu_sp2j => get_mu_sp2j_ptr(a%sv)
    
    jmx = get_jmx(a%sv)
    npnt = 1024
    allocate(r2ff(npnt))
    allocate(rmm2ff(npnt,-jmx:jmx,-jmx:jmx))
    lu = get_lu_slm(jmx)
    allocate(slm1(lu(1):lu(2)))
    allocate(slm2(lu(1):lu(2)))
    dr_lin = D12 / (npnt - 1)
    
    ssp = inf%species(1:2)
    nnrf = inf%ls2nrf(1:2)
    r_mean = 0
    do rf2=1,nnrf(2)
      mmu(2) = inf%rf_ls2mu(rf2, 2)
      jj(2) = mu_sp2j(mmu(2), ssp(2))
      do rf1=1,nnrf(1)
        mmu(1) = inf%rf_ls2mu(rf1, 1)
        jj(1) = mu_sp2j(mmu(1), ssp(1))
        rmm2ff = 0
        do ir=1,npnt
          r = (ir - 1) * dr_lin
          if(D12>0) then
            rvec = R1 + (R2 - R1) * (r / D12)
          else 
            rvec = R1
          endif
          rvec1 = rvec - R1
          rvec2 = rvec - R2
          call rsphar(rvec1, slm1, jj(1))
          call rsphar(rvec2, slm2, jj(2))

          r1_sqr = r**2
          r2_sqr = (D12 - r)**2
          call comp_coeff_m2p3_k(r1_sqr, a%interp_log, coeff1, k1)
          call comp_coeff_m2p3_k(r2_sqr, a%interp_log, coeff2, k2)
          f1 = sum(psi_log(k1-2:k1+3, mmu(1), ssp(1))*coeff1(-2:3))*slm1(jj(1)*(jj(1)+1))
          f2 = sum(psi_log(k2-2:k2+3, mmu(2), ssp(2))*coeff2(-2:3))*slm2(jj(2)*(jj(2)+1))
          r2ff(ir) = (f1 * f2)**2
        enddo ! ir

        ir_loc = maxloc(r2ff,1)
        r_mean = r_mean + (ir_loc-1)*dr_lin

      enddo ! rf1
    enddo ! rf2
    r_mean = r_mean / product(nnrf)
    
    if(D12>0) then
      inv_wght(1) = r_mean / D12
      inv_wght(2) = 1d0 - inv_wght(1)
    else
      inv_wght(1:2) = 1
    endif  
    
  else  

    do ls=1,2
      sp = inf%species(ls)
      nrf = inf%ls2nrf(ls)
      inv_wght(ls) = 0
      do rf=1,nrf
        mu = inf%rf_ls2mu(rf, ls)
        inv_wght(ls) = inv_wght(ls) + a%mu_sp2inv_wght(mu,sp)
      enddo ! rf
    enddo !ls
    
  endif  
  
  _dealloc(r2ff)
  _dealloc(rmm2ff)

end function ! get_inv_wghts 

end module !m_bilocal_vertex
