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

module m_siesta_ion

#include "m_define_macro.F90"  
  use m_die, only : die
  use m_precision, only : siesta_int
  use m_timing, only : get_cdatetime
  use m_warn, only : warn
  implicit none
  private die

  !!
  !! type to hold the information from an .ion file
  !! should be enough for writting this data back to a readable .ion file
  !!
  type siesta_ion_t
    !! preamble
    character(20) :: pp_fname = ''
    integer       :: atomic_number = -999
    real(8)       :: atomic_mass = -999
    real(8)       :: charge = -999
    integer       :: Lmxo = -999
    integer       :: Lmxkb = -999
    character(10) :: BasisType = ''
    logical       :: SemiC = .false. !  Semicore ?
    integer, allocatable :: ilo2l(:)
    integer, allocatable :: ilo2Nsemic(:)
    integer, allocatable :: ilo2cfigmx(:)
    integer, allocatable :: ilo2n(:)
    integer, allocatable :: ilo2nzeta(:)
    integer, allocatable :: ilo2polorb(:)
    real(8), allocatable :: ilo2splitnorm(:)
    real(8), allocatable :: ilo2vcte(:)
    real(8), allocatable :: ilo2rinn(:)
    real(8), allocatable :: ilo2qcoe(:)
    real(8), allocatable :: ilo2qyuk(:)
    real(8), allocatable :: ilo2qwid(:)
    real(8), allocatable :: ilo2rcs1(:,:)
    real(8), allocatable :: ilo2lambdas1(:,:)
    !! 
    integer, allocatable :: il2nkbl(:)
    real(8), allocatable :: il2erefs(:)
    !! END preamble

    !! pp header
    character(len=2)        :: name = ''
    logical                 :: relativistic = .false.
    character(len=2)        :: icorr = ''
    character(len=3)        :: irel = ''
    character(len=4)        :: nicore = ''
    character(len=10)       :: method(6)
    character(len=70)       :: text = ''
    !! END pp header

    character(2)            :: symbol = ''    
    character(20)           :: label = ''
    real(8)                 :: valence_charge = -999
    real(8)                 :: self_energy = -999

    !! PAOs 
    integer                 :: jmx = -999
    integer                 :: npao = -999
    integer, allocatable    :: pao2l(:)
    integer, allocatable    :: pao2n(:)
    integer, allocatable    :: pao2z(:)
    integer, allocatable    :: pao2is_polarized(:)
    real(8), allocatable    :: pao2population(:)
    real(8), allocatable    :: pao2delta(:)
    real(8), allocatable    :: pao2cutoff(:)
    integer                 :: pao_npts = -999
    real(8), allocatable    :: ir_pao2ff(:,:)

    !! KBs 
    integer                 :: jmx_kb = -999
    integer                 :: nkb = -999
    integer, allocatable    :: kb2l(:)
    integer, allocatable    :: kb2n(:)
    real(8), allocatable    :: kb2energy(:)
    real(8), allocatable    :: kb2delta(:)
    real(8), allocatable    :: kb2cutoff(:)
    integer                 :: kb_npts = -999
    real(8), allocatable    :: ir_kb2ff(:,:)

    !! vna
    integer                 :: vna_npts = -999
    real(8)                 :: vna_delta = -999
    real(8)                 :: vna_cutoff = -999
    real(8), allocatable    :: ir2vna(:)

    !! chlocal
    integer                 :: chlocal_npts = -999
    real(8)                 :: chlocal_delta = -999
    real(8)                 :: chlocal_cutoff =-999
    real(8), allocatable    :: ir2chlocal(:)

    !! core
    integer                 :: core_npts = -999
    real(8)                 :: core_delta = -999
    real(8)                 :: core_cutoff = -999
    real(8), allocatable    :: ir2core(:)
    
  end type ! siesta_ion_t

  contains  

!!
!!
!!
subroutine gnuplot_sp2ion(sp2ion, iv, ilog)
  use m_io, only : get_free_handle
  implicit none
  type(siesta_ion_t), intent(in) :: sp2ion(:)
  integer, intent(in) :: iv, ilog

  !! internal
  integer :: isp, l, irwf, ifile, ir, nspecies
  real(8) :: r, rpowl
  character(1000) :: fname

  nspecies = size(sp2ion)

  !! Write for gnuplot
  do isp=1, nspecies
    do irwf=1, sp2ion(isp)%npao
      write(fname,'(a,i0.2,a,i0.2,a)') './rwf-ion-', irwf, '-', isp, '.txt';
      ifile = get_free_handle();
      l = sp2ion(isp)%pao2l(irwf)
      open(ifile, file=trim(fname), action='write');
      do ir=1,sp2ion(isp)%pao_npts
        r = sp2ion(isp)%pao2delta(irwf)*(ir-1)
        if(r==0) then
          if (l==0) then
            rpowl = 1 
          else
            rpowl = 0
          endif     
        else
          rpowl = (r**l)
        endif
        write(ifile,*) r, sp2ion(isp)%ir_pao2ff(ir,irwf)* rpowl;
      end do
      close(ifile)
      if(iv>0)write(ilog,*) 'gnuplot_sp2ion: ', trim(fname)
    end do
  end do
  !! END of Write for gnuplot

  _T 
end subroutine !gnuplot_sp2ion


!
!
!
subroutine sp2ion_to_j_rcut(sp2ion, mu_sp2j, mu_sp2rcut)
  implicit none
  type(siesta_ion_t), intent(in), allocatable :: sp2ion(:)
  integer, intent(inout), allocatable :: mu_sp2j(:,:)
  real(8), intent(inout), allocatable :: mu_sp2rcut(:,:)
  ! internal
  integer :: isp, nspecies, nmu_mx, npao
  _dealloc(mu_sp2j)
  _dealloc(mu_sp2rcut)
  
  nspecies = 0
  if(allocated(sp2ion)) nspecies = size(sp2ion);
  if(nspecies<1) return
  
  nmu_mx = maxval(sp2ion(:)%npao)
 
  !! Allocate and init of internal arrays of mu_sp2* type
  allocate(mu_sp2j(nmu_mx, nspecies));
  allocate(mu_sp2rcut(nmu_mx, nspecies));
  mu_sp2j = -999
  mu_sp2rcut = -999
  do isp=1, nspecies
    npao = sp2ion(isp)%npao
    mu_sp2j(1:npao, isp)  = sp2ion(isp)%pao2l(1:npao)
    mu_sp2rcut(1:npao, isp)  = sp2ion(isp)%pao2cutoff(1:npao)
  end do
  !! END of Allocate and init of internal arrays of mu_sp2* type

end subroutine ! 

!
!
!
subroutine sp2ion_to_nmult_Z(sp2ion, sp2nmult, sp2Z)
  implicit none
  type(siesta_ion_t), intent(in), allocatable :: sp2ion(:)
  integer, intent(inout), allocatable :: sp2nmult(:), sp2Z(:)
  ! internal
  integer :: isp, nspecies
  _dealloc(sp2nmult)
  _dealloc(sp2Z)
  
  nspecies = 0
  if(allocated(sp2ion)) nspecies = size(sp2ion);
  if(nspecies<1) return 

  !! Allocate and init internal data arrays of sp2* type
  allocate(sp2Z(nspecies));
  allocate(sp2nmult(nspecies));
  do isp=1,nspecies
    sp2Z(isp)   = sp2ion(isp)%atomic_number
    sp2nmult(isp) = sp2ion(isp)%npao
  end do
  !! END of Allocate and init internal data arrays of sp2* type

end subroutine ! sp2ion_to_element_nmult

!
! 
!
subroutine siesta_save_ion(fname, ion, iv, ilog)
  use m_upper, only : upper
  implicit none
  integer, intent(in) :: iv, ilog
  character(*), intent(in) :: fname
  type(siesta_ion_t), intent(in) :: ion
  
  !! internal
  integer :: io, ios, ilo, il
  !! for reading the pseudopotential header
  integer :: i
  !! reading of pseudo atomic orbitals
  integer :: ir, ipao, npts
  real(8) :: r
  !! reading of KBs
  integer :: ikb

  

  io = 190
  open(io, file=fname, action='write')
  write(io,'(a)') '<preamble>'
  write(io,'(a)')''
  write(io,'(a)') '<basis_specs>'
  write(io,'(a)') '==============================================================================='
  write(io,'(a20,1x,a2,i4,4x,a5,g12.5,4x,a7,g12.5)', iostat=ios) &
    ion%pp_fname, 'Z=', ion%atomic_number, 'Mass=', ion%atomic_mass,'Charge=', ion%charge
  
  !! Next line 
  !Lmxo=1 Lmxkb=3     BasisType=split      Semic=F
  write(io,'(a5,i1,1x,a6,i1,5x,a10,a10,1x,a6,l1)') 'Lmxo=',ion%Lmxo, &
    'Lmxkb=',ion%Lmxkb,'BasisType=',ion%basistype,'Semic=',ion%SemiC
  !! Read line 
  do ilo=1, ion%Lmxo+1
    !L=0  Nsemic=0  Cnfigmx=2
    write(io,'(a2,i1,2x,a7,i1,2x,a8,i1)')'L=',ion%ilo2l(ilo),'Nsemic=',ion%ilo2NsemiC(ilo),'Cnfigmx=',ion%ilo2cfigmx(ilo)
    !          n=1  nzeta=2  polorb=0
    write(io,'(10x,a2,i1,2x,a6,i1,2x,a7,i1)')'n=',ion%ilo2n(ilo),'nzeta=',ion%ilo2nzeta(ilo),'polorb=',ion%ilo2polorb(ilo)
    !            splnorm:   0.15000 
    write(io,'(10x,a10,2x,g12.5)')'splnorm:',ion%ilo2splitnorm(ilo)
    !               vcte:    0.0000 
    write(io,'(10x,a10,2x,g12.5)')'vcte:',ion%ilo2vcte(ilo)
    !               rinn:    0.0000 
    write(io,'(10x,a10,2x,g12.5)')'rinn:',ion%ilo2rinn(ilo)
    !                rcs:    0.0000      0.0000
    write(io,'(10x,a10,2x,4g12.5)')'rcs:',ion%ilo2rcs1(:,ilo)
    !            lambdas:    1.0000      1.0000
    write(io,'(10x,a10,2x,4g12.5)')'lambdas:',ion%ilo2lambdas1(:,ilo)
  end do
  write(io,'(a)') '-------------------------------------------------------------------------------'
  do il=1,ion%Lmxkb+1
    !L=0  Nkbl=1  erefs: 0.17977+309
    write(io,'(a2,i1,2x,a5,i1,2x,a6,4g12.5)',iostat=ios)'L=',il-1,'Nkbl=',ion%il2nkbl(il),'erefs:',ion%il2erefs(il)
  end do
  write(io,'(a)') '==============================================================================='
  write(io,'(a)') '</basis_specs>'
  write(io,'(a)') ''
  write(io,'(a)') '<pseudopotential_header>'
  write(io,"(1x,a2,1x,a2,1x,a3,1x,a4)") ion%name, ion%icorr, ion%irel, ion%nicore
  write(io,"(1x,6a10,/,1x,a70)") (ion%method(i),i=1,6), ion%text
  write(io,'(a)') '</pseudopotential_header>'
  write(io,'(a)') '</preamble>'
  write(io,'(a2,28x,a)') ion%symbol, "# Symbol"
  write(io,'(a20,10x,a)') ion%label, "# Label"
  write(io,'(i5,25x,a)') ion%atomic_number, "# Atomic number"
  write(io,'(g22.12,25x,a)') ion%valence_charge, "# Valence charge"
  write(io,'(g22.12,4x,a)') ion%atomic_mass, "# Mass"
  write(io,'(g22.12,4x,a)') ion%self_energy, "# Self energy"
  write(io,'(2i4,22x,a)') ion%jmx, ion%npao, "# Lmax for basis, no. of nl orbitals "
  write(io,'(2i4,22x,a)') ion%jmx_kb, ion%nkb, "# Lmax for projectors, no. of nl KB projectors "
  write(io,'(a)') "# PAOs:__________________________"

  !! writing pseudo atomic orbitals
  npts = ion%pao_npts
  do ipao=1, ion%npao
    !  0  2  1  0  2.000000 #orbital l, n, z, is_polarized, population
    write(io,'(4i3,f10.6,2x,a)')ion%pao2l(ipao),ion%pao2n(ipao),ion%pao2z(ipao),ion%pao2is_polarized(ipao),&
      ion%pao2population(ipao), " #orbital l, n, z, is_polarized, population"
    ! 500    0.996588569579E-02     4.97297696220     # npts, delta, cutoff
    write(io,'(i4,2g22.12,a)') npts,ion%pao2delta(ipao),ion%pao2cutoff(ipao),' # npts, delta, cutoff'
    do ir=1,npts
      r = ion%pao2delta(ipao)*(ir-1)
      write(io,'(2g22.12)') r, ion%ir_pao2ff(ir,ipao)
    end do
  end do
  !! END of writing pseudo atomic orbitals

  write(io,'(a)') '# KBs:__________________________'
  !! reading Kleiman-Bulander projectors
  npts = ion% kb_npts
  do ikb=1, ion%nkb
    write(io,'(2i3,f22.16,2x,a)') ion%kb2l(ikb),ion%kb2n(ikb),ion%kb2energy(ikb),' #kb l, n (sequence number), Reference energy'
    write(io,'(i4,2g22.12,a)') npts,ion%kb2delta(ikb),ion%kb2cutoff(ikb), ' # npts, delta, cutoff'
    do ir=1,npts
      write(io,'(2g22.12)') ion%kb2delta(ikb)*(ir-1), ion%ir_kb2ff(ir,ikb)
    end do
  end do
  !! END of reading Kleiman-Bulander projectors

  !! Write Vna 
  write(io,'(a)') '# Vna:__________________________'
  !! Write Vna
  write(io,'(i4,2g22.12,a)') ion%vna_npts,ion%vna_delta,ion%vna_cutoff, ' # npts, delta, cutoff'
  do ir=1,ion%vna_npts
    write(io,'(2g22.12)') ion%vna_delta*(ir-1), ion%ir2vna(ir)
  end do
  !! END Write Vna 

  !! Write chlocal
  write(io,'(a)') '# Chlocal:__________________________'
  write(io,'(i4,2g22.12,a)') ion%chlocal_npts,ion%chlocal_delta,ion%chlocal_cutoff, ' # npts, delta, cutoff'
  do ir=1,ion%chlocal_npts
    write(io,'(2g22.12)') ion%chlocal_delta*(ir-1), ion%ir2chlocal(ir)
  end do
  !! END Write Vna 

  if(upper(ion%nicore)/='NC') then
    !! Write core
    write(io,'(a)') '# Core:__________________________'
    write(io,'(i4,2g22.12,a)') ion%core_npts,ion%core_delta,ion%core_cutoff, ' # npts, delta, cutoff'
    if(.not. allocated(ion%ir2core)) _die('!ir2core???')
    do ir=1,ion%core_npts
      write(io,'(2g22.12)') ion%core_delta*(ir-1), ion%ir2core(ir)
    end do
    !! END Write core 
  endif
  
  close(io)
  if(iv>0) write(ilog,*)'siesta_save_ion: ', trim(fname)

end subroutine !


!
! 
!
subroutine siesta_gnuplot_ion_struct(prefix, ion, iv, ilog)
  implicit none
  integer, intent(in) :: iv, ilog
  character(*), intent(in) :: prefix
  type(siesta_ion_t), intent(in) :: ion

  !! internal
  integer :: npt, nmult, j, imult, ir
  character(1000) :: fname
  real(8) :: norm, r, f

  !! Plotting the radial functions from .ion file
  npt = ion%pao_npts
  nmult = ion%npao
  do imult=1, nmult
    write(fname,'(a,i0.1,a)') trim(prefix)//'-pao-',imult,'.txt'
    open(100, file=fname, action='write')
    norm = 0
    do ir=1, npt
      r = ion% pao2delta(imult) *(ir-1)
      j = ion% pao2l(imult)
      f = ion% ir_pao2ff(ir,imult)*r**j
      write(100,'(100e23.14)') r, f
      norm = norm + f**2 * (r**2)
    end do
    close(100)
    if(iv>0)write(ilog,'(a,a,g20.10)')'siesta_gnuplot_ion_struct: ', trim(fname), norm*ion% pao2delta(imult)
  end do
  !! END Plotting the radial functions from .ion file

  !! Plotting the KB projectors from .ion file
  npt = ion%kb_npts
  nmult = ion%nkb
  do imult=1, nmult
    write(fname,'(a,i0.1,a)') trim(prefix)//'-kb-',imult,'.txt'
    open(100, file=fname, action='write')
    norm = 0
    do ir=1, npt
      r = ion% kb2delta(imult) *(ir-1)
      j = ion% kb2l(imult)
      f = ion% ir_kb2ff(ir,imult)*r**j
      write(100,'(100e23.14)') r, f
      norm = norm + f**2 * (r**2)
    end do
    close(100)
    if(iv>0)write(ilog,'(a,a,g20.10)')'siesta_gnuplot_ion_struct: ', trim(fname), norm * ion%kb2delta(imult)
  end do
  !! END Plotting the KB projectors from .ion file

  !! Plotting the Vna from .ion file
  npt = ion%vna_npts
  fname=trim(prefix)//'-vna.txt'
  open(100, file=fname, action='write')
  do ir=1, npt
    r = ion% vna_delta *(ir-1)
    f = ion% ir2vna(ir)
    write(100,'(100e23.14)') r, f
  end do
  close(100)
  if(iv>0)write(ilog,'(a,a)')'siesta_gnuplot_ion_struct: ', trim(fname)
  !! END Plotting the Vna from .ion file

  !! Plotting the chlocal from .ion file
  npt = ion%chlocal_npts
  fname=trim(prefix)//'-chlocal.txt'
  open(100, file=fname, action='write')
  do ir=1, npt
    r = ion% chlocal_delta *(ir-1)
    f = ion% ir2chlocal(ir)
    write(100,'(100e23.14)') r, f
  end do
  close(100)
  if(iv>0)write(ilog,'(a,a)')'siesta_gnuplot_ion_struct: ', trim(fname)
  !! END Plotting the chlocal from .ion file

end subroutine 

!!
!! Get the info from all .ion files
!!
subroutine siesta_get_ion_files(wfsx, sp2ion)
  use m_siesta_wfsx, only : siesta_wfsx_t, siesta_wfsx_to_sp2label_atm2sp
  implicit none
  type(siesta_wfsx_t), intent(in)      :: wfsx
  type(siesta_ion_t), allocatable, intent(inout) :: sp2ion(:)

  !! internal
  integer :: isp, nspecies, norbitals_for_this_specie, ipao
  character(20), allocatable :: sp2label(:)
  integer, allocatable       :: atom2specie(:)
  
!  iv = iv_in - 1
  !! Initialize the arrays sp2label & species
  call siesta_wfsx_to_sp2label_atm2sp(wfsx, sp2label, atom2specie)
  nspecies = size(sp2label);

  !! Read information from all .ion files
  allocate(sp2ion(nspecies))
  do isp=1, nspecies
    call siesta_get_ion('./'//trim(sp2label(isp))//'.ion', sp2ion(isp))
    norbitals_for_this_specie = 0
    do ipao=1,sp2ion(isp)%npao
      norbitals_for_this_specie = norbitals_for_this_specie + (2*sp2ion(isp)%pao2l(ipao)+1)
    end do

    !! Simple consistency check
    if( norbitals_for_this_specie > maxval(wfsx%orb2ao)) then
      write(0,*) 'siesta_get_ion_files: norbitals_for_this_specie > maxval(wfsx%orb2ao)', &
        norbitals_for_this_specie, maxval(wfsx%orb2ao);
      write(0,*) 'Input files .WFSX and .ion do not seem to be consistent.'
      write(0,*) 'It is advisable to copy the whole directory with a SIESTA calculation'
      write(0,*) 'and start your calculation of excited states in this copied directory.'
      stop 'siesta_get_ion_files'
    endif
    !! END of Simple consistency check
  end do

end subroutine ! siesta_get_ion_files

!
! 
!
subroutine siesta_get_ion(fname, ion)
  use m_io, only : read_line, get_free_handle
  use m_upper, only : upper
  implicit none
  character(*), intent(in) :: fname
  type(siesta_ion_t), intent(inout) :: ion
  
  !! internal
  integer :: io, ios, ilo, il, l
  character(100) :: dummy, dummy1, dummy2, dummy3
  character(1000) :: cline
  integer :: atomic_number !! check consistency of atomic number
  real(8) :: atomic_mass !! check consistency of atomic mass
  !! reading of pseudo atomic orbitals
  integer :: i, ir, ipao, npts, ikb,n
  real(8) :: r
  real(8), allocatable :: ilo2rcs(:)
  logical :: data1_missing

!  iv = iv_in - 1
  
  io = get_free_handle()
  open(io, file=fname, action='read')
  read(io,*) dummy !! <preamble>
  if(.not. dummy=="<preamble>") then
    write(0,*)'siesta_get_ion: (dummy=/"<preamble>") ==> stop', trim(dummy)
    stop
  endif
  read(io,*) dummy !! <basis_specs>
  read(io,*) dummy !! ============
  ion%atomic_number = -999
  ion%atomic_mass = -999
  ion%charge = -999
  !! First line with formats
!  read(io,'(a20,1x,a3,i4,3x,a5,f8.3,8x,a7,e20.5)', iostat=ios) ion%pp_fname, dummy1, ion%atomic_number, &
!    dummy2, ion%atomic_mass,dummy3,ion%charge
  read(io,'(a20,1x,a3,i4,3x,a5,e14.8,8x,a7)', iostat=ios) ion%pp_fname, dummy1, ion%atomic_number, &
    dummy2, ion%atomic_mass,dummy3!,ion%charge
  if(ios/=0)ion%charge = 0.17977D+306
  
  !if(iv>0)write(ilog,*)'siesta_get_ion: ion%pp_fname ', ion%pp_fname 
  !if(iv>0)write(ilog,*)'siesta_get_ion01: ', trim(dummy1), ion%atomic_number
  !if(iv>0)write(ilog,*)'siesta_get_ion01: ', trim(dummy2), ion%atomic_mass
  !if(iv>0)write(ilog,*)'siesta_get_ion01: ', trim(dummy3), ion%charge

  !! Next line 
  !Lmxo=1 Lmxkb=3     BasisType=split      Semic=F
  read(io,'(a5,i1,1x,a6,i2,4x,a10,a10,1x,a6,l1)') dummy,ion%Lmxo, dummy1,ion%Lmxkb,dummy2,ion%basistype,dummy3,ion%SemiC
  !if(iv>1)write(ilog,*)'siesta_get_ion02: ', trim(dummy),  ' ', ion%Lmxo
  !if(iv>1)write(ilog,*)'siesta_get_ion02: ', trim(dummy1), ' ', ion%Lmxkb
  !if(iv>1)write(ilog,*)'siesta_get_ion02: ', trim(dummy2), ' ', ion%basistype
  !if(iv>1)write(ilog,*)'siesta_get_ion02: ', trim(dummy3), ' ', ion%SemiC
  allocate(ion%ilo2l( ion%Lmxo+1))
  allocate(ion%ilo2Nsemic( ion%Lmxo+1))
  allocate(ion%ilo2cfigmx( ion%Lmxo+1))
  allocate(ion%ilo2n( ion%Lmxo+1))
  allocate(ion%ilo2nzeta( ion%Lmxo+1))
  allocate(ion%ilo2polorb( ion%Lmxo+1))
  allocate(ion%ilo2splitnorm( ion%Lmxo+1))
  allocate(ion%ilo2vcte( ion%Lmxo+1))
  allocate(ion%ilo2rinn( ion%Lmxo+1))
  allocate(ion%ilo2qcoe( ion%Lmxo+1))
  allocate(ion%ilo2qyuk( ion%Lmxo+1))
  allocate(ion%ilo2qwid( ion%Lmxo+1))
  !! Read line 
  data1_missing = .false.
  do ilo=1, ion%Lmxo+1
    !L=0  Nsemic=0  Cnfigmx=2
    if(data1_missing) then
      read(cline,'(a2,i1,2x,a7,i1,2x,a8,i1)')dummy,ion%ilo2l(ilo),dummy1,ion%ilo2NsemiC(ilo),dummy2,ion%ilo2cfigmx(ilo)
    else
      read(io,'(a2,i1,2x,a7,i1,2x,a8,i1)')dummy,ion%ilo2l(ilo),dummy1,ion%ilo2NsemiC(ilo),dummy2,ion%ilo2cfigmx(ilo)
    endif
    !if(iv>1)write(ilog,'(a,a,a,2i6)')'siesta_get_ion03: ', trim(dummy),  ' ', ion%ilo2l(ilo), ilo
    !if(iv>1)write(ilog,*)'siesta_get_ion03: ', trim(dummy1), ' ', ion%ilo2NsemiC(ilo), ilo
    !if(iv>1)write(ilog,*)'siesta_get_ion03: ', trim(dummy2), ' ', ion%ilo2cfigmx(ilo), ilo
    do n=1, ion%ilo2NsemiC(ilo)+1
      !          n=1  nzeta=2  polorb=0
      call read_line(io, cline)
      read(cline,'(10x,a2,i1,2x,a6,i1,2x,a7,i1)',iostat=ios)dummy1,ion%ilo2n(ilo),&
        dummy2,ion%ilo2nzeta(ilo),dummy3,ion%ilo2polorb(ilo)
      !if(iv>1)write(ilog,*)'siesta_get_ion04: ', trim(dummy1), ' ', ion%ilo2n(ilo), ilo
      !if(iv>1)write(ilog,*)'siesta_get_ion04: ', trim(dummy2), ' ', ion%ilo2nzeta(ilo), ilo
      !if(iv>1)write(ilog,*)'siesta_get_ion04: ', trim(dummy3), ' ', ion%ilo2polorb(ilo), ilo

      if(trim(dummy1)/='n=' .or. trim(dummy2)/='nzeta=' .or. trim(dummy3)/='polorb=') then
        data1_missing = .true.
        _warn('data block is missing ???')
        ion%ilo2nzeta(ilo)=0
        exit
      else 
        data1_missing = .false.
        if(.not. allocated(ion%ilo2rcs1)) then
          allocate(ion%ilo2rcs1( ion%ilo2nzeta(ilo), ion%Lmxo+1))
          ion%ilo2rcs1 = -999
        endif  
        if(.not. allocated(ion% ilo2lambdas1)) then
          allocate(ion%ilo2lambdas1( ion%ilo2nzeta(ilo),ion%Lmxo+1))
          ion%ilo2lambdas1 = -999
        endif
        !            splnorm:   0.15000 
        read(io,'(12x,a9,e20.10)')dummy,ion%ilo2splitnorm(ilo)
        !if(iv>1)write(ilog,*)'siesta_get_ion05: ', trim(dummy),  ' ', ion%ilo2splitnorm(ilo), ilo
        !               vcte:    0.0000 
        read(io,'(15x,a5,e20.10)')dummy,ion%ilo2vcte(ilo)
        !if(iv>1)write(ilog,*)'siesta_get_ion06: ', trim(dummy),  ' ', ion%ilo2vcte(ilo), ilo
        !               rinn:    0.0000 
        read(io,'(15x,a5,e20.10)')dummy,ion%ilo2rinn(ilo)
        !if(iv>1)write(ilog,*)'siesta_get_ion07: ', trim(dummy),  ' ', ion%ilo2rinn(ilo), ilo
        !                rcs:    0.0000      0.0000
        read(io,'(16x,a5,10e9.4)')dummy,ion%ilo2rcs1(:,ilo)
        if(dummy/='rcs:') then ! added fields 
          ion%ilo2qcoe(ilo) = ion%ilo2rcs1(1,ilo)
          read(io,'(16x,a5,10e9.4)')dummy,ion%ilo2qyuk(ilo)
          read(io,'(16x,a5,10e9.4)')dummy,ion%ilo2qwid(ilo)
          read(io,'(16x,a5,10e9.4)')dummy,ion%ilo2rcs1(:,ilo)
        endif
        !if(iv>1)write(ilog,*)'siesta_get_ion08: ', trim(dummy),  ' ', &
        !  real(ion%ilo2rcs1(1:ion%ilo2nzeta(ilo),ilo),4), ilo
        !            lambdas:    1.0000      1.0000
        read(io,'(12x,a8,10e9.4)')dummy,ion%ilo2lambdas1(:,ilo)
        !if(iv>1)write(ilog,*)'siesta_get_ion09: ', trim(dummy),  ' ', &
        !  real(ion%ilo2lambdas1(1:ion%ilo2nzeta(ilo),ilo),4), ilo
      endif 
        
    enddo ! n=1, ion%ilo2NsemiC(ilo)+1 
  end do ! ilo=1, ion%Lmxo+1

  read(io,*) dummy !! ============
  allocate(ion%il2nkbl(ion%Lmxkb+1))
  allocate(ion%il2erefs(ion%Lmxkb+1))
  do il=1,ion%Lmxkb+1
    !L=0  Nkbl=1  erefs: 0.17977+309
    read(io,'(a2,i2,1x,a5,i1,2x,a5,e20.10)',iostat=ios) dummy,l, dummy1,ion%il2nkbl(il), dummy2,ion%il2erefs(il)
    if(ios/=0) ion%il2erefs(il)= 0.17977D+308;
    !if(iv>1)write(ilog,*)'siesta_get_ion10: ', trim(dummy),  ' ', l, il
    !if(iv>1)write(ilog,*)'siesta_get_ion11: ', trim(dummy1), ' ', ion%il2nkbl(il), il
    !if(iv>1)write(ilog,*)'siesta_get_ion12: ', trim(dummy2), ' ', ion%il2erefs(il), il
  end do
  read(io,*) dummy !!===============================================================================
  read(io,*) dummy !!</basis_specs>
  read(io,*) dummy !!<pseudopotential_header>
  if(.not. dummy=="<pseudopotential_header>") then
    write(0,*)'trim(dummy)', trim(dummy)
    _die('(dummy=/"<pseudopotential_header>")')
  endif

  read(io,"(1x,a2,1x,a2,1x,a3,1x,a4)") ion%name, ion%icorr, ion%irel, ion%nicore
  read(io,"(1x,6a10,/,1x,a70)") (ion%method(i),i=1,6), ion%text
  !if(iv>1)write(ilog,*) 'ion%name        ', ion%name
  !if(iv>1)write(ilog,*) 'ion%icorr       ', ion%icorr
  !if(iv>1)write(ilog,*) 'ion%irel        ', ion%irel
  !if(iv>1)write(ilog,*) 'ion%nicore      ', ion%nicore
  !if(iv>1)write(ilog,*) 'ion%method(1:6) ', ion%method(1:6)
  !if(iv>1)write(ilog,*) 'ion%text        ', ion%text
  read(io,*) dummy !!</pseudopotential_header>
  read(io,*) dummy !!</preamble>
  read(io,*) ion%symbol
  read(io,*) ion%label
  read(io,*) atomic_number
  if(atomic_number/=ion%atomic_number) then
    write(0,*)'siesta_get_ion: (atomic_number/=ion%atomic_number) ==> stop', atomic_number, ion%atomic_number
    stop
  endif
  read(io,*) ion%valence_charge
  read(io,*) atomic_mass
  if(atomic_mass/=ion%atomic_mass) then
    write(0,*)'siesta_get_ion: (atomic_mass/=ion%atomic_mass) ==> stop', atomic_mass, ion%atomic_mass
    stop
  endif
  read(io,*) ion%self_energy
  read(io,*) ion%jmx, ion%npao
  read(io,*) ion%jmx_kb, ion%nkb

  !if(iv>0)write(ilog,*) 'siesta_get_ion: ', ion%symbol
  !if(iv>0)write(ilog,*) 'siesta_get_ion: ', ion%label
  !if(iv>0)write(ilog,*) 'siesta_get_ion: ', atomic_number
  !if(iv>0)write(ilog,*) 'siesta_get_ion: ', ion%valence_charge
  !if(iv>0)write(ilog,*) 'siesta_get_ion: ', atomic_mass
  !if(iv>1)write(ilog,*) 'siesta_get_ion: ', ion%self_energy
  !if(iv>0)write(ilog,*) 'siesta_get_ion: ', ion%jmx, ion%npao
  !if(iv>1)write(ilog,*) 'siesta_get_ion: ', ion%jmx_kb, ion%nkb
  read(io,'(a30)') dummy !!# PAOs:__________________________
  if(dummy(3:6)/='PAOs') then
    write(6,*) 'trim(dummy) |', trim(dummy), '|'
    _warn('.ion conventions ?')
  endif
  !if(iv>1)write(ilog,*) 'siesta_get_ion: ', trim(dummy)

  allocate(ion%pao2l(ion%npao))
  allocate(ion%pao2n(ion%npao))
  allocate(ion%pao2z(ion%npao))
  allocate(ion%pao2is_polarized(ion%npao))
  allocate(ion%pao2population(ion%npao))
  allocate(ion%pao2delta(ion%npao))
  allocate(ion%pao2cutoff(ion%npao))

  !! reading pseudo atomic orbitals
  do ipao=1, ion%npao
    !  0  2  1  0  2.000000 #orbital l, n, z, is_polarized, population
    read(io,*) ion%pao2l(ipao),ion%pao2n(ipao),ion%pao2z(ipao),ion%pao2is_polarized(ipao),ion%pao2population(ipao)
    ! 500    0.996588569579E-02     4.97297696220     # npts, delta, cutoff
    read(io,*) npts,ion%pao2delta(ipao),ion%pao2cutoff(ipao)
    if(ipao==1) then;  !! allocation is necessary
      allocate(ion%ir_pao2ff(npts,ion%npao)); 
      ion%pao_npts = npts;
    endif
    if(ion%pao_npts<npts) then !! cross check is necessary
      write(0,*)'siesta_get_ion: (ion%pao_npts<npts) ==> stop', ion%pao_npts, npts
      stop
    endif
    do ir=1,npts
      read(io,*) r, ion%ir_pao2ff(ir,ipao)
    end do
    !if(iv>1)write(ilog,*) 'siesta_get_ion: ', ion%pao2l(ipao), ion%pao2n(ipao), ion%pao2z(ipao), &
    !   ion%pao2is_polarized(ipao), real(ion%pao2population(ipao),4)
    !if(iv>1)write(ilog,*) 'siesta_get_ion: ', npts, ion%pao2delta(ipao), ion%pao2cutoff(ipao) 
  end do
  !! END of reading pseudo atomic orbitals

  read(io,'(a30)') dummy !!# KBs:__________________________
  if(dummy(3:5)/='KBs') then
    write(6,*) 'trim(dummy) |', trim(dummy), '|'
    _warn('.ion conventions ?')
  endif
  !if(iv>1)write(ilog,*) 'siesta_get_ion: ', trim(dummy)
  allocate(ion%kb2l(ion%nkb))
  allocate(ion%kb2n(ion%nkb))
  allocate(ion%kb2energy(ion%nkb))
  allocate(ion%kb2delta(ion%nkb))
  allocate(ion%kb2cutoff(ion%nkb))
  !! reading Kleiman-Bulander projectors
  do ikb=1, ion%nkb
    read(io,*) ion%kb2l(ikb),ion%kb2n(ikb),ion%kb2energy(ikb)
    !if(iv>1)write(ilog,*) 'siesta_get_ion: ', ion%kb2l(ikb),ion%kb2n(ikb),ion%kb2energy(ikb)
    read(io,*) npts,ion%kb2delta(ikb),ion%kb2cutoff(ikb)
    !if(iv>1)write(ilog,*) 'siesta_get_ion: ', npts,ion%kb2delta(ikb),ion%kb2cutoff(ikb)
    if(ikb==1) then;  !! allocation is necessary
      allocate(ion%ir_kb2ff(npts,ion%nkb)); 
      ion%kb_npts = npts;
    endif
    if(ion%kb_npts<npts) then !! cross check is necessary
      write(0,*)'siesta_get_ion: (ion%kb_npts<npts) ==> stop', ion%kb_npts, npts
      stop
    endif
    do ir=1,npts
      read(io,*) r, ion%ir_kb2ff(ir,ikb)
    end do
  end do
  !! END of reading Kleiman-Bulander projectors

  !! Read Vna 
  read(io,'(a30)') dummy !!# Vna:__________________________
  if(dummy(3:5)/='Vna') then
    write(6,*) 'trim(dummy) |', trim(dummy), '|'
    _warn('.ion conventions ?')
  endif  
  !if(iv>1)write(ilog,*) 'siesta_get_ion: ', trim(dummy)
  !! Read Vna
  read(io,*) ion%vna_npts,ion%vna_delta,ion%vna_cutoff
  !if(iv>1)write(ilog,*) 'siesta_get_ion: ', ion%vna_npts,ion%vna_delta,ion%vna_cutoff
  allocate(ion%ir2vna(ion%vna_npts))
  do ir=1,ion%vna_npts
    read(io,*) r, ion%ir2vna(ir)
  end do
  !! END Read Vna 

  !! Read chlocal
  read(io,'(a30)') dummy !!# Chlocal:______________________
  if(dummy(3:9)/='Chlocal') then
    write(6,*) 'trim(dummy) |', trim(dummy), '|'
    _warn('.ion conventions ?')
  endif  
  !if(iv>1)write(ilog,*) 'siesta_get_ion: ', trim(dummy)
  !! Read Vna
  read(io,*) ion%chlocal_npts,ion%chlocal_delta,ion%chlocal_cutoff
  !if(iv>1)write(ilog,*) 'siesta_get_ion: ', ion%chlocal_npts,ion%chlocal_delta,ion%chlocal_cutoff
  allocate(ion%ir2chlocal(ion%chlocal_npts))
  do ir=1,ion%chlocal_npts
    read(io,*) r, ion%ir2chlocal(ir)
  end do
  !! END Read Vna 

  ! Read core eventually
  ion%core_npts=0
  ion%core_delta=0
  ion%core_cutoff=0
_dealloc(ion%ir2core)
  if(upper(ion%nicore)/='NC' .and. ion%atomic_number>0) then
    ! Read core
    read(io,'(a30)',IOSTAT=ios) dummy !# Core:__________________________
    if(ios/=0) then
      write(6,*) __FILE__, __LINE__
      stop 'no core ?'
    endif 
    
    !if(iv>1)write(ilog,*) 'siesta_get_ion: ', trim(dummy)
    if(dummy(3:6)/='Core') then
      write(6,*) 'trim(dummy) |', trim(dummy), '|'
      _warn('.ion conventions ?')
    endif  
    !500    0.425450228585E-02     2.12299664064     # npts, delta, cutoff
    !! Read Core
    read(io,*) ion%core_npts,ion%core_delta,ion%core_cutoff
    !if(iv>1)write(ilog,*) 'siesta_get_ion: ', ion%core_npts,ion%core_delta,ion%core_cutoff
    allocate(ion%ir2core(ion%core_npts))
    do ir=1,ion%core_npts
      read(io,*) r, ion%ir2core(ir)
    end do
    !! END of Read core
  endif
  ! END of Read core eventually
  close(io)
  !if(iv>0)write(ilog,*)'siesta_get_ion: ', trim(fname)
  _dealloc(ilo2rcs)
  
end subroutine !

!
!
!
subroutine siesta_plot_ion(prefix, pt_mu_sp2psi, isp, mu_sp2delta, mu_sp2j, iv, ilog)
  implicit none
  character(*), intent(in) :: prefix
  integer, intent(in) :: iv, ilog, isp
  real(8), intent(in) :: pt_mu_sp2psi(:,:,:)
  integer, intent(in) :: mu_sp2j(:,:)
  real(8), intent(in) :: mu_sp2delta(:,:)

  !! internal
  integer :: npt, nmult, j, imult, ir
  character(1000) :: fname
  real(8) :: norm, r

  npt = size(pt_mu_sp2psi,1)
  nmult = size(pt_mu_sp2psi,2) 
  !! Plotting the radial functions from .ion file
  do imult=1, nmult
    write(fname,'(a,a1,i0.1,a,i0.1,a)') trim(prefix),'-',isp,'-',imult,'.txt'
    open(100, file=fname, action='write')
    norm = 0
    do ir=1, npt
      r = mu_sp2delta(imult,1)*(ir-1)
      j = mu_sp2j(imult,1)
      write(100,'(100e23.14)') r, pt_mu_sp2psi(ir,imult,1)*r**j
      norm = norm + (pt_mu_sp2psi(ir,imult,1)*r**j)**2 * (r**2)
    end do
    close(100)
    if(iv>0)write(ilog,*)'siesta_plot_ion: ', trim(fname), norm*mu_sp2delta(imult,1)
  end do
  !! END Plotting the radial functions from .ion file

end subroutine ! siesta_plot_ion

!
! Read Siesta's .ion files in order to get dimensions
!
subroutine import_ion_get_dimensions(fname, nrwf, maxnpts, iv, ilog)
  use m_upper, only : upper
  !! external
  character(len=*), intent(in) :: fname
  integer, intent(in)          :: iv, ilog
  integer, intent(out)         :: nrwf, maxnpts

  !! internal
  integer :: ifile, iostat, lmax, npts, irwf, ipt
  character(len=200) :: line
  real(8) :: delta, rcut

  ifile = 123
  open(ifile,file=trim(fname),form='formatted',action='read',status='old',iostat=iostat);
  if(iostat/=0) then; write(ilog,*)'import_ion_get_dimensions: error: file ', trim(fname), " ?"; stop; endif;
  rewind(ifile);
  read(ifile,'(a)') line
  if (upper(trim(line)) .eq. '<PREAMBLE>') then
    do while(upper(trim(line)) .ne. '</PREAMBLE>')
      read(ifile,'(a)') line
    end do
  endif
  read(ifile,'(a2)') line
  read(ifile,'(a20)') line
  read(ifile,*) line
  read(ifile,*) line
  read(ifile,*) line
  read(ifile,*) line
  read(ifile,*) lmax, nrwf
  read(ifile,*) line
  read(ifile,*) line
  !! Compute maxnpts
  maxnpts = 0
  do irwf=1, nrwf
    read(ifile,*) line
    read(ifile,*) npts, delta, rcut
    if(maxnpts<npts) maxnpts = npts;
    do ipt=1,npts
      read(ifile,*)line
    end do
  end do

  if(iv>1) then
    write(ilog,*) 'import_ion_get_dimensions: nrwf, maxnpts: ', trim(fname), nrwf, maxnpts
  end if
  close(ifile)

end subroutine ! import_ion_get_dimensions

!
! Read Siesta's .ion files in order to get radial wavefunctions
!
subroutine import_ion_get_rwfs(fname, pt_mu2psi, mu2j, mu2npts, mu2rcut, mu2delta, maxpts, max_multipletts, element, iv, ilog)
  use m_upper, only : upper
  !! external
  character(len=*), intent(in) :: fname
  integer, intent(in)          :: iv, ilog, maxpts, max_multipletts
  real(8), intent(inout)       :: pt_mu2psi(maxpts, max_multipletts), mu2rcut(max_multipletts),mu2delta(max_multipletts)
  integer, intent(inout)       :: mu2j(max_multipletts), mu2npts(max_multipletts), element

  !! internal
  integer :: ifile, iostat, lmax, irwf, nrwf, ipt, n, z, is_polarized
  character(len=200) :: line
  real(8) :: population, rcoord

  ifile = 124
  open(ifile,file=trim(fname),form='formatted',action='read',status='old',iostat=iostat);
  if(iostat/=0) then; write(ilog,*)'import_ion_get_dimensions: error: file ', trim(fname), " ?"; stop; endif;
  rewind(ifile);
  !! Skip preamble
  read(ifile,'(a)') line
  if (upper(trim(line)) .eq. '<PREAMBLE>') then
    do while(upper(trim(line)) .ne. '</PREAMBLE>')
      read(ifile,'(a)') line
    end do
  endif
  read(ifile,'(a2)') line
  read(ifile,'(a20)') line
  read(ifile,*) element
  read(ifile,*) line
  read(ifile,*) line
  read(ifile,*) line
  read(ifile,*) lmax, nrwf
  read(ifile,*) line
  read(ifile,*) line
  !! Load the radial wavefunctions

  mu2j = -999; mu2rcut = -999; mu2delta = -999; mu2npts  = -999;
  pt_mu2psi = 0
  do irwf=1, nrwf
    read(ifile,*) mu2j(irwf), n, z, is_polarized, population
    read(ifile,*) mu2npts(irwf), mu2delta(irwf), mu2rcut(irwf)
    do ipt=1,mu2npts(irwf)
      read(ifile,*)rcoord, pt_mu2psi(ipt, irwf)
    end do

    if(iv<=1) cycle
    write(ilog,'(a,a,i5,i5,f14.10,e25.12)') 'import_ion_get_rwf: ',trim(fname),irwf,mu2j(irwf),mu2rcut(irwf),mu2delta(irwf)
  end do
  close(ifile);

end subroutine ! import_ion_get_rwfs

end module !m_siesta_ion
