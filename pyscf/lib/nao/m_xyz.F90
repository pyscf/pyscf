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

module m_xyz

  implicit none

  contains

!
!
!
subroutine xyz_sort(iv, ilog)
  implicit none
  integer, intent(in) :: iv, ilog

  integer :: narg, natoms, atom_s, atom_o, min_dist(1), atom_min_dist, atom_ss
  integer, parameter :: lenchr=1000
  real(8), allocatable :: coord(:,:)
  real(8), allocatable :: coord_out(:,:)
  integer, allocatable :: atom_sorted2atom_orig(:)
  character(2), allocatable :: atm2symbol_out(:)
  character(2), allocatable :: atm2symbol(:)
  character(lenchr) :: filename, filename_output, line 
  real(8) :: dist
  logical :: already_sorted
  real(8), allocatable :: atom_orig2inv_distance(:)

  narg = command_argument_count();
  if(narg<1) then; write(ilog,*)'xyz_sort: SORT: narg<1 ', narg; stop; endif;
  
  call get_command_argument(1, filename);
  call read_xyz_file(filename, natoms, atm2symbol, coord, line)

  allocate(coord_out(3,natoms))
  coord_out = 0
  allocate(atm2symbol_out(natoms))
  atm2symbol_out = atm2symbol
  allocate(atom_sorted2atom_orig(natoms))
  atom_sorted2atom_orig = 0
  allocate(atom_orig2inv_distance(natoms))
  atom_orig2inv_distance = 0
  
  coord_out(:,1) = coord(:,1)
  atm2symbol_out(1) = atm2symbol(1)
  atom_sorted2atom_orig = -1
  atom_sorted2atom_orig(1) = 1
  do atom_s=1,natoms-1

    atom_orig2inv_distance = -1
    do atom_o=1,natoms
      already_sorted = .false.
      do atom_ss=1,atom_s
        if(atom_o==atom_sorted2atom_orig(atom_ss)) already_sorted = .true.
      enddo
      if(already_sorted) cycle
      dist = sqrt(sum((coord_out(:,atom_s)-coord(:,atom_o))**2));
      if(dist==0) atom_orig2inv_distance(atom_o) = -1;
      if(dist>0)  atom_orig2inv_distance(atom_o) = 1D0/dist;
    enddo
    !if(iv>0)write(ilog,*) 'xyz_sort: atom_orig2inv_distance '
    !if(iv>0)write(ilog,*) atom_orig2inv_distance
    min_dist = maxloc(atom_orig2inv_distance);
    atom_min_dist = min_dist(1)
    dist = sqrt(sum((coord_out(:,atom_sorted2atom_orig(atom_s))-coord(:,atom_min_dist))**2))
    if(iv>0)write(ilog,'(a,2i7,1g20.12)') 'xyz_sort: atom1 atom2 dist ', &
      atom_sorted2atom_orig(atom_s), atom_min_dist, dist
    coord_out(:,atom_s+1) = coord(:,atom_min_dist)
    atm2symbol_out(atom_s+1) = atm2symbol(atom_min_dist)
    atom_sorted2atom_orig(atom_s+1) = atom_min_dist
    
    !do atom_ss=1,atom_s+1
    !  write(ilog,*) coord(:,atom_sorted2atom_orig(atom_ss))
    !enddo

    !do atom_ss=1,atom_s+1
    !  write(ilog,*) coord_out(:,atom_ss)
    !enddo
    
  enddo

  filename_output=trim(filename)//".sort.xyz";
  line = trim(line)//'. Sorted.'
  call write_xyz_file(filename_output, natoms, atm2symbol_out, coord_out/0.529177249D0, line, iv, ilog);

end subroutine ! xyz_sort


!
! Generates graphene sheet
!
subroutine generate_graphene(iv, ilog)
  implicit none
  integer, intent(in) :: iv, ilog

  integer :: natoms
  integer, parameter :: lenchr=1000
  character(LENCHR) :: argument
  real(8), allocatable :: coord(:,:)
  character(2), allocatable :: atm2symbol(:)
  character(lenchr) :: filename, line
  integer :: ix, iy, ny, nx, a
  real(8) :: bond_length, dx, dy, x, y, yyy, pi
  pi = 4D0*atan(1D0)

  !narg = command_argument_count();
  call get_command_argument(1, filename);
  call get_command_argument(3, argument); 

  nx = 5
  ny = 7
  natoms = nx*ny
  allocate(coord(3,natoms))
  allocate(atm2symbol(natoms))
  coord = 0
  bond_length = 1.42D0
  dx = bond_length*2*sin(pi/3);
  dy = bond_length

  a = 0
  y = 0
  do iy = 1, ny
    dy = bond_length*cos(pi/3)*mod(iy,2)+bond_length*abs(1-mod(iy,2))
    y = y + dy
    write(6,*) 'iy, dy, y ', iy, dy, y
  x = 0
  do ix = 1, nx
    a = a + 1
    dx = bond_length*2*sin(pi/3)
    yyy = 0
    if(mod(iy,4)<2) yyy = bond_length*sin(pi/3)
    x = dx*ix + yyy
    write(6,*) 'ix, x', ix, x 
    coord(1, a) = x
    coord(2, a) = y
    atm2symbol(a) = "C"
  end do
  end do
  
  line = "graphene"

  call write_xyz_file(filename, natoms, atm2symbol, coord/0.529177249D0, line, iv, ilog);

end subroutine !generate_graphene

!
!
!
subroutine xyz_atom2rotation(iv, ilog)
  implicit none
  integer, intent(in) :: iv, ilog

  integer :: narg, natoms
  integer, parameter :: lenchr=1000
  character(LENCHR) :: argument
  real(8), allocatable :: coord(:,:)
  character(2), allocatable :: atm2symbol(:)
  character(lenchr) :: filename, line
  integer :: i, atom
  real(8) :: theta, phi
  real(8) :: center(3)
  real(8) :: translation(3), norm

  narg = command_argument_count();
  if(narg<3) then; write(ilog,*)'xyz_atom2rotation: narg<3 ', narg; stop; endif;

  call get_command_argument(1, filename);
  call read_xyz_file(filename, natoms, atm2symbol, coord, line)
  call get_command_argument(3, argument); read(argument,*) atom

  do i=1,3; center(i) = sum(coord(i,:))/natoms; enddo;
  do i=1,natoms; coord(:,i) = coord(:,i) - center(:); enddo

  
  translation = coord(:, atom)
  norm=sqrt(sum(translation*translation))
  if (norm==0) then
    write(0,*) 'xyz_atom2rotation: norm==0', norm
    stop
  endif
  phi=aimag(log(cmplx(translation(1),translation(2),8)))
  theta=acos(translation(3)/norm)

  write(ilog,*)'xyz_atom2rotation:       atom ', atom
  write(ilog,*)'xyz_atom2rotation: theta, phi ',real(theta), real(phi), ' Rad'

end subroutine ! xyz_shift


!
!
!
subroutine xyz_rotate(iv, ilog)
  implicit none
  integer, intent(in) :: iv, ilog

  integer :: narg, natoms
  integer, parameter :: lenchr=1000
  character(LENCHR) :: argument
  real(8), allocatable :: coord(:,:)
  character(2), allocatable :: atm2symbol(:)
  character(lenchr) :: filename, filename_output, line
  integer :: i
  real(8) :: theta, phi
  real(8) :: center(3)
  real(8) :: rotation(3,3), inverse_rotation(3,3)

  narg = command_argument_count();
  if(narg<4) then; write(ilog,*)'xyz_rotate: narg<4 ', narg; stop; endif;

  call get_command_argument(1, filename);
  call read_xyz_file(filename, natoms, atm2symbol, coord, line)
  call get_command_argument(3, argument); read(argument,*) theta
  call get_command_argument(4, argument); read(argument,*) phi

  do i=1,3; center(i) = sum(coord(i,:))/natoms; enddo;
  do i=1,natoms; coord(:,i) = coord(:,i) - center(:); enddo

  call make_standard_rotation(theta, phi, rotation, inverse_rotation )
  do i=1,natoms; coord(:,i) = matmul(rotation, coord(:,i)); enddo
  
  write(ilog,*)'xyz_rotate by ',real(theta), real(phi),' Rad done'
  filename_output=trim(filename)//".rotate.xyz";
  call write_xyz_file(filename_output, natoms, atm2symbol, coord/0.529177249D0, line, iv, ilog);

end subroutine ! xyz_shift


!
!
!
subroutine xyz_shift_shift_merge(iv, ilog)
  implicit none
  integer, intent(in) :: iv, ilog

  integer :: narg, n
  integer, parameter :: lenchr=1000
  character(LENCHR) :: argument
  real(8) :: shift_x, shift_y, shift_z, shift(3)
  integer :: natoms
  real(8), allocatable :: coord(:,:)
  character(2), allocatable :: atm2symbol(:)
  character(lenchr)  :: line

  real(8), allocatable :: coord1(:,:)
  real(8), allocatable :: coord2(:,:)
  real(8), allocatable :: coord3(:,:)
  
  character(2), allocatable :: atm2symbol3(:)

  character(lenchr) :: filename, filename_output
  character(2*lenchr) :: line3

  narg = command_argument_count();
  if(narg<5) then; write(ilog,*)'xyz_shift_shift_merge: SHIFT: narg<5 ', narg; stop; endif;

  call get_command_argument(1, filename);
  call read_xyz_file(filename, natoms, atm2symbol, coord, line)
  call get_command_argument(3, argument); read(argument,*) shift_x
  call get_command_argument(4, argument); read(argument,*) shift_y
  call get_command_argument(5, argument); read(argument,*) shift_z
  shift = (/shift_x, shift_y, shift_z /);

  allocate(coord1(3,natoms))
  allocate(coord2(3,natoms))
  do n=1, natoms; coord1(:,n)  = coord(:,n) + shift; enddo;
  do n=1, natoms; coord2(:,n)  = coord(:,n) - shift; enddo;

  allocate(coord3(3,2*natoms))
  coord3(:,        1:natoms)   = coord1(:,1:natoms)
  coord3(:, natoms+1:2*natoms) = coord2(:,1:natoms)

  allocate(atm2symbol3(2*natoms));
  atm2symbol3(1:natoms)          = atm2symbol(1:natoms)
  atm2symbol3(natoms+1:2*natoms) = atm2symbol(1:natoms) 

  line3 = trim(line) //' & '// trim(line)
  write(ilog,*)'xyz_shift: shift, -shift & merge done'
  filename_output=trim(filename)//".edit.xyz";
  call write_xyz_file(filename_output, natoms*2, atm2symbol3, coord3/0.529177249D0, line3, iv, ilog);

end subroutine ! xyz_shift


!
!
!
subroutine compute_meancoord(iv, ilog)
  implicit none
  integer, intent(in) :: iv, ilog

  !! internal  
  integer, parameter  :: lenchr=1000
  character(lenchr)   :: argument
  integer :: i,narg, atom_start, atom_finish
  real(8) :: center(3)
  integer :: natoms
  real(8), allocatable :: coord(:,:)
  character(2), allocatable :: atm2symbol(:)
  character(lenchr) :: line


  narg = command_argument_count();
  if(narg<4) then; write(6,*)'compute_meancoord: narg<4 ', narg; stop; endif;
  call get_command_argument(1, argument);
  call read_xyz_file(argument, natoms, atm2symbol, coord, line)
  call get_command_argument(3, argument); read(argument,*) atom_start
  call get_command_argument(4, argument); read(argument,*) atom_finish

  do i=1,3
    center(i)  = sum(coord(i,atom_start:atom_finish))/(atom_finish-atom_start+1)
  end do
  write(ilog,*)'compute_meancoord: center '
  write(ilog,*)center
  if(iv>0)write(ilog,*)'compute_meancoord: done';

end subroutine ! compute_mean_coord

!
!
!
subroutine compute_meandist(iv, ilog)
  implicit none
  integer, intent(in) :: iv, ilog

  !! internal  
  integer, parameter  :: lenchr=1000
  character(lenchr)   :: argument
  integer :: i,narg, group1_start, group1_finish, group2_start, group2_finish
  real(8) :: center1(3), center2(3)
  integer :: natoms
  real(8), allocatable :: coord(:,:)
  character(2), allocatable :: atm2symbol(:)
  character(lenchr) :: line


  narg = command_argument_count();
  if(narg<6) then; write(6,*)'compute_meandist: narg<4 ', narg; stop; endif;

  call get_command_argument(1, argument);
  call read_xyz_file(argument, natoms, atm2symbol, coord, line)
  call get_command_argument(3, argument); read(argument,*) group1_start
  call get_command_argument(4, argument); read(argument,*) group1_finish
  call get_command_argument(5, argument); read(argument,*) group2_start
  call get_command_argument(6, argument); read(argument,*) group2_finish

  write(ilog,*) 'compute_meandist: group1 from atom ', group1_start, ' to atom ', group1_finish
  write(ilog,*) 'compute_meandist: group2 from atom ', group2_start, ' to atom ', group2_finish

  do i=1,3
    center1(i)  = sum(coord(i,group1_start:group1_finish))/(group1_finish-group1_start+1)
  end do

  do i=1,3
    center2(i)  = sum(coord(i,group2_start:group2_finish))/(group2_finish-group2_start+1)
  end do

  write(ilog,*)'compute_meandist:  sqrt(sum((center1-center2)**2))'
  write(ilog,*)sqrt(sum((center1-center2)**2)), ' Angstrem'
  write(ilog,*)'compute_meandist: done';

end subroutine ! compute_mean_coord

!
!
!
subroutine xyz_shift(iv, ilog)
  implicit none
  integer, intent(in) :: iv, ilog

  integer :: narg, n, natoms
  integer, parameter :: lenchr=1000
  character(LENCHR) :: argument
  real(8) :: shift_x, shift_y, shift_z, shift(3)
  real(8), allocatable :: coord(:,:)
  character(2), allocatable :: atm2symbol(:)
  character(lenchr) :: filename, filename_output, line 

  narg = command_argument_count();
  if(narg<5) then; write(ilog,*)'xyz_shift: SHIFT: narg<5 ', narg; stop; endif;
  
  call get_command_argument(1, filename);
  call read_xyz_file(filename, natoms, atm2symbol, coord, line)
  call get_command_argument(3, argument); read(argument,*) shift_x
  call get_command_argument(4, argument); read(argument,*) shift_y
  call get_command_argument(5, argument); read(argument,*) shift_z
  shift = (/shift_x, shift_y, shift_z /);
  do n=1, natoms; coord(:,n) = coord(:,n) + shift; enddo
  write(ilog,*)'xyz_shift: shift by ',real(shift),' Ang, done.'
  filename_output=trim(filename)//".edit.xyz";
  call write_xyz_file(filename_output, natoms, atm2symbol, coord/0.529177249D0, line, iv, ilog);

end subroutine ! xyz_shift

!
!
!
subroutine xyz_shiftcenter(iv, ilog)
  implicit none
  integer, intent(in) :: iv, ilog

  integer :: narg, n, natoms, i
  integer, parameter :: lenchr=1000
  real(8), allocatable :: coord(:,:)
  character(2), allocatable :: atm2symbol(:)
  real(8) :: center(3), shift(3)
  character(lenchr) :: filename, filename_output, line 

  narg = command_argument_count();
  if(narg<1) then; write(ilog,*)'xyz_shiftcenter: SHIFT: narg<1 ', narg; stop; endif;
  call get_command_argument(1, filename);
  call read_xyz_file(filename, natoms, atm2symbol, coord, line)

  do i=1,3
    center(i)  = sum(coord(i,1:natoms))/natoms
  end do
  
  shift = -center;
  do n=1, natoms; coord(:,n) = coord(:,n) + shift; enddo
  write(ilog,*)'xyz_shiftcenter: shift by ',real(shift),' Ang, done.'
  filename_output=trim(filename)//".shiftcenter.xyz";
  call write_xyz_file(filename_output, natoms, atm2symbol, coord/0.529177249D0, line, iv, ilog);

end subroutine ! xyz_shift


!
! Read coordinates and info from xyz file
!
subroutine read_xyz_file(filename, natoms, atm2symbol, coord, line)
  implicit none
  character(*), intent(in) :: filename
  integer, intent(inout)   :: natoms
  character(2), allocatable, intent(inout) :: atm2symbol(:)
  real(8), allocatable, intent(inout) :: coord(:,:)
  character(*), intent(inout)         :: line

  !! internal
  integer, parameter  :: M=1000
  integer :: iostat, j, n
  character :: line_char(M)

  !! XYZ file reading
  open(15, file=filename, action='read', status='old', iostat=iostat);
  if(iostat/=0) then; write(0,*)'read_xyz_file: ', trim(filename), '?'; stop; endif;
  read(15,*)natoms
  !! Read line
  line_char = '';
  read(15, '(1000A1)', iostat=iostat) (line_char(j),j=1,M); 
  do j=1,M
     line(j:j)=line_char(j)
  enddo;

  allocate(atm2symbol(natoms), coord(3,natoms))
  n = 1
  do n=1,natoms
    read(15,*,iostat=iostat) atm2symbol(n), coord(1:3,n)
    if(iostat/=0) exit
  end do
  !! Stream is over -- we know number of atoms and every thing what was in the stream
  if(natoms/=n-1) then; write(0,*)'read_xyz_file: (natoms/=n-1)', natoms,n-1; stop; endif
  close(15);
!  if(iv>0) write(6,*)'read_xyz_file: filename ', trim(filename)
!  !! END of XYZ file reading
!  if(iv>0) write(6,*)'read_xyz_file:   natoms ', natoms
!  if(iv>0) write(6,*)'read_xyz_file:     line ', trim(line)

end subroutine ! read_xyz_file

!!
!!
!!
subroutine create_xyz(filename, atom_uc2specie, specie2element, &
  atom_sc2atom_uc, natoms_xyz, coord_bohr, line, iv, ilog);
  use m_z2sym, only : get_sym
  implicit none
  character(*), intent(in) :: filename
  integer, intent(in)      :: atom_uc2specie(:)
  integer, intent(in)      :: specie2element(:)
  integer, intent(in)      :: atom_sc2atom_uc(:)
  integer, intent(in)      :: natoms_xyz
  real(8), intent(in)      :: coord_bohr(3,*) ! xyz, atom
  character(*), intent(in) :: line
  integer, intent(in)      :: iv, ilog
  
  integer :: Z, sp, atom_xyz
  real(8), allocatable :: coord_ang(:,:)
  character(2), allocatable :: atm2symbol(:)
  
  !! Dimensions
  integer :: natoms
  real(8), parameter :: Ang2Bohr = 1.889725989D0
  natoms = size(atom_uc2specie)
  !! END of Dimensions
  
  allocate(atm2symbol(natoms_xyz))
  allocate(coord_ang(3,natoms_xyz))
  
  do atom_xyz=1,natoms_xyz
    if(atom_xyz>natoms)sp = atom_uc2specie(atom_sc2atom_uc(atom_xyz))
    if(atom_xyz<=natoms)sp = atom_uc2specie(atom_xyz)
    Z =  specie2element(sp) 
    atm2symbol(atom_xyz) = get_sym(Z)
    coord_ang(:,atom_xyz) = coord_bohr(:,atom_xyz)/Ang2Bohr
  enddo

  call write_xyz_file(filename, natoms_xyz, atm2symbol, coord_ang/0.529177249D0, line, iv, ilog)

end subroutine ! create_xyz

!!
! Write an xyz file
!!
subroutine write_xyz_file(filename, natoms, atm2symbol, coord, line, iv, ilog)
  implicit none
  character(*), intent(in) :: filename
  integer, intent(in)      :: natoms, iv, ilog
  character(2), intent(in) :: atm2symbol(:)
  real(8), intent(in)      :: coord(:,:)
  character(*), intent(in) :: line

  integer :: iostat, n

  !! XYZ file writing
  open(15, file=trim(filename), action='write', iostat=iostat);
  if(iostat/=0) then; write(0,*)'write_xyz_file: ', trim(filename), '?'; stop; endif;
  write(15,'(i6)')natoms
  write(15, '(A)') trim(line); 
  do n=1,natoms
    write(15,'(a2,3f15.8)') atm2symbol(n), coord(1:3,n)*0.529177249D0
  end do
  close(15)
  if(iv>0)write(ilog,*) 'write_xyz_file: ', trim(filename)
  !! END of XYZ file writing
  

end subroutine ! write_xyz_file

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Conventional rotation matrix
!   rotation   gives  the vector in coord system Z
! Axis z of coord system Z points always along direction (theta,phi)
! theta, phi
!
subroutine make_standard_rotation(theta,phi,rotation, inverse_rotation)
  implicit none
  real(8), intent(in) :: theta,phi
  real(8), dimension(3,3), intent(out) :: rotation, inverse_rotation

  !! internal
  real(8), dimension(3,3) :: R_y, Inverse_R_y, R_z, Inverse_R_z
!   integer :: j

        R_y=0;         R_z=0
Inverse_R_y=0; Inverse_R_z=0


Inverse_R_y(1,1)= cos(theta); Inverse_R_y(1,3)=  -sin(theta);
            Inverse_R_y(2,2)=1
Inverse_R_y(3,1)= sin(theta); Inverse_R_y(3,3)=   cos(theta);

        R_y(1,1) = cos(theta);        R_y(1,3) = sin(theta);
                      R_y(2,2) = 1
        R_y(3,1) = -sin(theta);       R_y(3,3) = cos(theta);


Inverse_R_z(1,1)= cos(phi); Inverse_R_z(1,2)=   sin(phi); 
Inverse_R_z(2,1)= -sin(phi);  Inverse_R_z(2,2)=   cos(phi);
Inverse_R_z(3,3)=1

        R_z(1,1)= cos(phi);         R_z(1,2)= - sin(phi); 
        R_z(2,1)=  sin(phi);          R_z(2,2)=   cos(phi);
        R_z(3,3)=1

  rotation=matmul(R_z,R_y)

!   write(6,*)'Inverse_R_y', cos(theta), theta
!   do j=1,3
!   write(6,*) Inverse_R_y(j,:)
!   enddo

!   write(6,*)'Inverse_R_z'
!   do j=1,3
!   write(6,*) Inverse_R_z(j,:)
!   enddo

  inverse_rotation=matmul(Inverse_R_y,Inverse_R_z)
end subroutine !make_standard_rotation


!
! Gets XYZ file
!
subroutine get_geometry( fname, coord_bohr, atom2nuc_chrg, comment, iv, ilog );
  use m_io, only : get_free_handle
  use m_z2sym, only : z2sym
    
  implicit none
  !! external
  character(*), intent(in) :: fname
  real(8), intent(inout), allocatable :: coord_bohr(:,:)
  integer, intent(inout), allocatable :: atom2nuc_chrg(:)
  character(*), intent(inout) :: comment
  integer, intent(in) :: iv, ilog
  !! internal
  integer :: ifile, ios, natoms, nuc_charge, i, n
  character(2), allocatable :: atom2symbol(:)
  integer, parameter :: M=1000
  character(M) :: line
  character :: line_char(M)
  real(8), parameter :: Ang2Bohr = 1.889725989D0
    
  ifile = get_free_handle()
  open(ifile, file=fname, action='read', status='old', iostat=ios);
  if(ios/=0) then; write(0,*)'get_geometry: ', trim(fname), '?'; stop; endif;
  read(ifile,*,iostat=ios)natoms
  if(iv>0)write(ilog,*) 'get_geometry: natoms ', natoms
  if(ios/=0 .or. natoms<1) then
    write(0,*)'First line must be number of atoms', __LINE__, __FILE__
    stop 'get_geometry'
  endif
  
  
  !! Read line
  line_char = '';
  read(ifile, '(1000A1)', iostat=ios) (line_char(i),i=1,M); 
  do i=1,M; line(i:i)=line_char(i); enddo;
  !! END of Read line
  comment = line;

  allocate(atom2symbol(natoms),coord_bohr(3,natoms))
  n = 1
  do n=1,natoms
    read(ifile,*,iostat=ios) atom2symbol(n), coord_bohr(1:3,n)
    if(ios/=0) exit
  end do
  close(ifile);
  
  coord_bohr = coord_bohr * Ang2Bohr; ! convert from Angstrom to Bohr units
  
  !! Stream is over -- we know number of atoms and every thing what was in the stream
  if(natoms/=n-1) then; write(0,*)'get_geometry: (natoms/=n-1)', natoms,n-1; stop; endif
  
  allocate(atom2nuc_chrg(natoms));
  do n=1,natoms
    nuc_charge = 0
    do i=1, size(z2sym); if(z2sym(i)==atom2symbol(n))nuc_charge=i; enddo
    if(nuc_charge==0) then; write(0,*)'get_geometry: (nuc_charge==0)', nuc_charge, atom2symbol(n); stop; endif
    atom2nuc_chrg(n) = nuc_charge
  enddo
  
  if(iv>0)write(ilog,*)'get_geometry: ', trim(fname), ' is read';
  
  if(iv>0) then
    write(ilog,*) int(natoms,2), '! natoms'
    write(ilog,*) trim(comment)
    do n=1,natoms
      write(ilog,*) int(atom2nuc_chrg(n),2),real(coord_bohr(:,n)), ' ! Bohr'
    end do
  endif

end subroutine ! get_geometry


end module ! m_xyz
