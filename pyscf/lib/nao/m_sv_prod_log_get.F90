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

module m_sv_prod_log_get

  use iso_c_binding, only: c_double, c_int64_t
#include "m_define_macro.F90"

  implicit none

  contains

!
!
!
subroutine sv_prod_log_get(d,n, sv, pb)
  use m_system_vars, only : system_vars_t, dealloc_sv=>dealloc
  use m_prod_basis_type, only : prod_basis_t, dealloc_pb=>dealloc
  implicit none
  !! external
  integer(c_int64_t), intent(in) :: n
  real(c_double), intent(in) :: d(n)
  type(system_vars_t), intent(inout), target :: sv
  type(prod_basis_t), intent(inout) :: pb
  !! internal
  integer(c_int64_t) :: nsp, nr, s, f, i, jmx, nmumx, mu, sp, natoms, norbs, norbs_sc
  integer(c_int64_t) :: atom, nspin, mup, nmup, np, no, p, o2, j, s1, f1, ott
  integer(c_int64_t) :: ac_npc_max ! maximal number of participating centers
  integer(c_int64_t) :: jcutoff ! maximal ang mom for expanding bilocals
  real(c_double) :: rmin, rmax, kmax, psi_log_sum, prod_log_sum, tol_loc, tol_biloc, ac_rcut_ratio
  real(c_double) :: ac_rcut_max, psum
  real(c_double), allocatable :: sp2rcut(:), sp2rcutp(:)
  integer(c_int64_t), allocatable :: mu_sp2s(:,:), atom2s(:), atom2mu_s(:)
  integer(c_int64_t), allocatable :: sp2norbsp(:), sp2chrgp(:), sp2nmultp(:)
  integer, allocatable :: mu2s(:)
  integer :: metric_type, ngl, optimize_centers

  call dealloc_sv(sv)
  call dealloc_pb(pb)

  sv%uc%systemlabel = "libnao"
  i = 2
  nsp  = int( d(i), c_int64_t); i=i+1;
  nr   = int( d(i), c_int64_t); i=i+1;
  rmin = d(i); i=i+1;
  rmax = d(i); i=i+1;
  kmax = d(i); i=i+1;
  jmx  = int( d(i), c_int64_t); i=i+1;
  psi_log_sum = d(i); i=i+1;
  prod_log_sum = d(i); i=i+1;
  natoms  = int( d(i), c_int64_t); i=i+1;
  norbs  = int( d(i), c_int64_t); i=i+1;
  norbs_sc  = int( d(i), c_int64_t); i=i+1;
  nspin  = int( d(i), c_int64_t); i=i+1;
  tol_loc  = d(i); i=i+1;
  tol_biloc  = d(i); i=i+1;
  ac_rcut_ratio  = d(i); i=i+1;
  ac_npc_max  = int( d(i), c_int64_t); i=i+1;
  jcutoff = int( d(i), c_int64_t); i=i+1;
  metric_type = int(d(i)); i=i+1;
  optimize_centers = int(d(i)); i=i+1;
  ngl = int(d(i)); i=i+1;

  if (nsp<1) then; write(6,*) __FILE__, __LINE__, nsp; stop '!nsp'; endif
  if (nr<1) then; write(6,*) __FILE__, __LINE__, nr; stop '!nr'; endif
  if (jcutoff<jmx) then; write(6,*) __FILE__, __LINE__, jcutoff, jmx; stop '!jcutoff'; endif
  if (metric_type<1) then; write(6,*) __FILE__, __LINE__, metric_type; stop '!metric_type'; endif
    
  allocate(sv%uc%sp2label(nsp))
  allocate(sv%uc%sp2nmult(nsp))
  allocate(sv%uc%sp2norbs(nsp))
  allocate(sp2rcut(nsp))
  allocate(sv%uc%sp2element(nsp))

  allocate(sv%rr(nr))
  allocate(sv%pp(nr))
  
  i = 100
  s = int( d(i), c_int64_t); f=s+nr-1; sv%rr(:) = d(s:f); i=i+1
  s = int( d(i), c_int64_t); f=s+nr-1; sv%pp(:) = d(s:f); i=i+1
  s = int( d(i), c_int64_t); f=s+nsp-1; sv%uc%sp2nmult(:) = int(d(s:f));   i=i+1
  s = int( d(i), c_int64_t); f=s+nsp-1; sp2rcut(:) = d(s:f);    i=i+1
  s = int( d(i), c_int64_t); f=s+nsp-1; sv%uc%sp2norbs(:) = int(d(s:f));   i=i+1
  s = int( d(i), c_int64_t); f=s+nsp-1; sv%uc%sp2element(:) = int(d(s:f)); i=i+1

  !write(6,*) '  sv%uc%sp2nmult ', sv%uc%sp2nmult
  !write(6,*) '  sv%uc%sp2norbs ', sv%uc%sp2norbs
  !write(6,*) '         sp2rcut ', sp2rcut
  !write(6,*) 'sv%uc%sp2element ', sv%uc%sp2element

  nmumx = maxval(sv%uc%sp2nmult)
  if (nmumx<1) then; write(6,*) __FILE__, __LINE__, nmumx; stop '!nmumx'; endif
  allocate(sv%uc%mu_sp2j(nmumx,nsp))
  allocate(sv%uc%mu_sp2n(nmumx,nsp))
  allocate(sv%uc%mu_sp2rcut(nmumx,nsp))
  allocate(sv%uc%mu_sp2start_ao(nmumx,nsp))
  sv%uc%mu_sp2j = -999
  sv%uc%mu_sp2n = -999
  sv%uc%mu_sp2rcut = -999
  sv%uc%mu_sp2start_ao = -999
  s = int( d(i), c_int64_t);
  do sp=1,nsp
    f=s+sv%uc%sp2nmult(sp)-1; sv%uc%mu_sp2j(1:f-s+1,sp) = int(d(s:f)); s=f+1;
  enddo
  if(jmx/=maxval(sv%uc%mu_sp2j)) then; 
    write(6,*) __FILE__, __LINE__, jmx, maxval(sv%uc%mu_sp2j); 
    write(6,*) sv%uc%mu_sp2j
    stop '!jmx';
  endif

  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  do sp=1,nsp
    f=s+sv%uc%sp2nmult(sp)-1; sv%uc%mu_sp2rcut(1:f-s+1,sp) = d(s:f); s=f+1;
  enddo

  !write(6,*) 'sv%uc%mu_sp2rcut ', sv%uc%mu_sp2rcut

  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  allocate(sv%psi_log(nr,nmumx,nsp))
  sv%psi_log = 0
  do sp=1,nsp
    do mu=1,sv%uc%sp2nmult(sp); f=s+nr-1; sv%psi_log(1:nr,mu,sp) = d(s:f); s=f+1; enddo
  enddo
  
  if( abs(psi_log_sum-sum(sv%psi_log))/abs(psi_log_sum)>5d-13 ) then; 
    write(6,*) __FILE__, __LINE__, psi_log_sum, sum(sv%psi_log);
    stop '!psi_log_sum';
  endif

  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
      
  allocate(mu_sp2s(nmumx+1,nsp))
  mu_sp2s = -999
  do sp=1,nsp; f=s+sv%uc%sp2nmult(sp); mu_sp2s(1:f-s+1,sp) = int(d(s:f), c_int64_t)+1; s=f+1; enddo

  !do sp=1,nsp; write(6,*) 'mu_sp2s ', mu_sp2s(:,sp); enddo
 
  !! Finishing with sp2*
  allocate(sv%psi_log_rl(nr,nmumx,nsp))
  do sp=1,nsp
    do mu=1,sv%uc%sp2nmult(sp)
      sv%psi_log_rl(1:nr,mu,sp) = sv%psi_log(1:nr,mu,sp)/(sv%rr**sv%uc%mu_sp2j(mu,sp))
    enddo
  enddo

  sv%jmx = int(jmx)
  sv%norb_max = maxval(sv%uc%sp2norbs)

  sv%uc%mu_sp2start_ao = 0
  do sp=1,nsp
    f1 = 0
    do mu=1,sv%uc%sp2nmult(sp)
      j = sv%uc%mu_sp2j(mu,sp)
      s1 = f1+1; sv%uc%mu_sp2start_ao(mu,sp)=int(s1); f1 = s1 + 2*j+1 - 1; 
    enddo
  enddo 
  !! END of Finishing with sp2*
  
  if (natoms<1) then
    write(6,*) __FILE__, __LINE__, natoms;
    stop '!natoms<1';
  endif
  
  allocate(sv%atom_sc2coord(3,natoms))
  allocate(sv%atom_sc2start_orb(natoms))
  allocate(sv%atom_sc2atom_uc(natoms))
  allocate(sv%uc%atom2coord(3,natoms))
  allocate(sv%uc%atom2sp(natoms))
  allocate(atom2s(natoms+1))
  allocate(atom2mu_s(natoms+1))
  sv%atom_sc2coord = -999.0D0
  sv%atom_sc2start_orb = -999
  sv%atom_sc2atom_uc = -999
  sv%uc%atom2coord = -999.0D0
  sv%uc%atom2sp = -999
  atom2s = -999

  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  do ott=1,3; f=s+3-1; sv%uc%uc_vecs(1:3,ott) = d(s:f); s=f+1; enddo

  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  f=s+natoms-1; sv%uc%atom2sp(1:natoms) = int(d(s:f))+1; s=f+1;
  
  !write(6,*) __FILE__, __LINE__, ' sv%uc%atom2sp ', sv%uc%atom2sp
  !write(6,*) __FILE__, __LINE__, ' sv%uc%sp2element ', sv%uc%sp2element

  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  f=s+natoms+1-1; atom2s(1:natoms+1) = int(d(s:f), c_int64_t)+1; s=f+1;

  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  f=s+natoms+1-1; atom2mu_s(1:natoms+1) = int(d(s:f), c_int64_t)+1; s=f+1;

  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  do atom=1,natoms; f=s+3-1; sv%uc%atom2coord(1:3,atom) = d(s:f); s=f+1; enddo;

  sv%natoms = int(natoms)
  sv%norbs = int(atom2s(natoms+1)-1)
  if(sv%norbs<nsp) then
    write(6,*) __FILE__, __LINE__, sv%norbs, nsp; stop '!norbs';
  endif  
  no = 0
  do atom=1,natoms
    sp = sv%uc%atom2sp(atom)
    do mu=1,sv%uc%sp2nmult(sp)
      no = no + 2*sv%uc%mu_sp2j(mu,sp)+1
    enddo 
  enddo
  if(no /= sv%norbs) then
    write(6,*) __FILE__, __LINE__, no, sv%norbs; stop '!norbs';
  endif
  
  _dealloc(sv%uc%atom2sfo)
  allocate(sv%uc%atom2sfo(2,natoms))
  do atom=1,natoms
    sp = sv%uc%atom2sp(atom)
    sv%uc%atom2sfo(1,atom) = int(atom2s(atom))
    sv%uc%atom2sfo(2,atom) = int(atom2s(atom+1))
  enddo

  !! Product basis now................ PPPPPPPPPPPPPPPPPPPP
  pb%sv => sv
  pb%pb_p%metric_type = metric_type
  pb%pb_p%eigmin_local = tol_loc
  pb%pb_p%eigmin_bilocal = tol_biloc
  pb%pb_p%jcutoff = int(jcutoff)
  pb%pb_p%optimize_centers = optimize_centers
  pb%pb_p%gl_ord_bilocal = ngl
  pb%pb_p%gl_pcs_bilocal = 1
  pb%pb_p%bilocal_type = "ATOM"
  pb%pb_p%bilocal_center = "POW&COEFF"
  pb%pb_p%bilocal_center_pow = 1D0
  pb%pb_p%bilocal_center_coeff = "2*L+1"
  ac_rcut_max = maxval(sv%uc%mu_sp2rcut)
  pb%pb_p%ac_rcut = ac_rcut_ratio * ac_rcut_max
  pb%pb_p%ac_method = "LENS"

  allocate(pb%sp_local2vertex(nsp))
  allocate(pb%sp_local2functs(nsp))
  allocate(sp2nmultp(nsp))
  allocate(sp2rcutp(nsp))
  allocate(sp2norbsp(nsp))
  allocate(sp2chrgp(nsp))
  sp2nmultp = -999
  sp2rcutp = -999
  sp2norbsp = -999
  sp2chrgp = -999
  
  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  f=s+nsp-1; sp2nmultp(1:nsp) = int(d(s:f), c_int64_t); s=f+1;
  if(any(sp2nmultp<1)) then
    write(6,*) __FILE__, __LINE__, sp2nmultp; stop '!sp2nmultp';
  endif

  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  f=s+nsp-1; sp2rcutp(1:nsp) = d(s:f); s=f+1;
  if(any(sp2rcutp<0)) then
    write(6,*) __FILE__, __LINE__, sp2rcutp; stop '!sp2rcutp';
  endif
  
  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  f=s+nsp-1; sp2norbsp(:) = int(d(s:f), c_int64_t); s=f+1;
  if(any(sp2norbsp<1)) then
    write(6,*) __FILE__, __LINE__, sp2norbsp; stop '!sp2norbsp';
  endif

  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  f=s+nsp-1; sp2chrgp(:) = int(d(s:f), c_int64_t); s=f+1;
  if(sum(abs(sp2chrgp-sv%uc%sp2element))/=0) then
    write(6,*) __FILE__, __LINE__, sv%uc%sp2element, sp2chrgp; stop '!sp2element';
  endif
  
  do sp=1,nsp
    nmup = sp2nmultp(sp)
    allocate(pb%sp_local2functs(sp)%mu2j(nmup))
    allocate(pb%sp_local2functs(sp)%ir_mu2v(nr,nmup))
    allocate(pb%sp_local2functs(sp)%mu2si(nmup))
    allocate(pb%sp_local2functs(sp)%mu2rcut(nmup))
    no = sv%uc%sp2norbs(sp)
    np = sp2norbsp(sp)
    allocate(pb%sp_local2vertex(sp)%vertex(no,no,np))
    pb%sp_local2functs(sp)%sp = int(sp)
    pb%sp_local2functs(sp)%rcut = sp2rcutp(sp)
    pb%sp_local2functs(sp)%nfunct = int(sp2norbsp(sp))
  enddo
  
  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  do sp=1,nsp
    nmup = sp2nmultp(sp)
    f=s+nmup-1; pb%sp_local2functs(sp)%mu2j(1:nmup) = int(d(s:f)); s=f+1;
  enddo

  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  do sp=1,nsp
    nmup = sp2nmultp(sp)
    f=s+nmup-1; pb%sp_local2functs(sp)%mu2rcut(1:nmup) = d(s:f); s=f+1;
  enddo

  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  do sp=1,nsp
    nmup = sp2nmultp(sp)
    do mup=1,nmup
      f=s+nr-1; pb%sp_local2functs(sp)%ir_mu2v(1:nr,mup) = d(s:f); s=f+1;
    enddo ! mup
  enddo ! sp

  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  _dealloc(mu2s)
  allocate(mu2s(maxval(sp2nmultp)+1))
  do sp=1,nsp
    nmup = sp2nmultp(sp)
    f=s+nmup+1-1; mu2s(1:nmup+1) = int( d(s:f) ); s=f+1;
    pb%sp_local2functs(sp)%mu2si(1:nmup) = mu2s(1:nmup)+1
  enddo

  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif
  do sp=1,nsp
    no = sv%uc%sp2norbs(sp)
    np = sp2norbsp(sp)
    do p=1,np
      do o2=1,no
        f=s+no-1; pb%sp_local2vertex(sp)%vertex(1:no,o2,p) = d(s:f); s=f+1;
      enddo ! o2
    enddo ! p
  enddo ! sp

  i = i + 1
  if(s/=d(i)) then; write(6,*) __FILE__, __LINE__, s, d(i); stop '!incr'; endif

  psum = 0
  do sp=1,nsp; psum = psum + sum(pb%sp_local2functs(sp)%ir_mu2v); enddo ! sp

  if( abs(prod_log_sum-psum)/abs(prod_log_sum)>5d-13 ) then; 
    write(6,*) __FILE__, __LINE__, prod_log_sum, psum;
    stop '!prod_log_sum';
  endif
  
  _dealloc(pb%pp)
  _dealloc(pb%rr)
  _dealloc(pb%atom2coord)
  allocate(pb%rr(nr))
  allocate(pb%pp(nr))
  allocate(pb%atom2coord(3,sv%natoms))
  pb%atom2coord = sv%uc%atom2coord
  pb%rr = sv%rr
  pb%pp = sv%pp
  pb%uc_vecs = sv%uc%uc_vecs

end subroutine ! m_sv_prod_log_get

end module ! m_sv_prod_log_get
