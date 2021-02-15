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

module m_z2sym

  implicit none

  integer, parameter :: z2sym_size = 109
  character(2), dimension(z2sym_size) :: z2sym = &
  (/ "H ","He","Li","Be","B ","C ","N ","O ","F ","Ne","Na","Mg","Al","Si","P ","S ","Cl","Ar","K ","Ca","Sc","Ti","V ","Cr",    &
     "Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y ","Zr","Nb","Mo", "Tc","Ru","Rh","Pd", "Ag","Cd", &
     "In","Sn","Sb","Te","I ","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf", &
     "Ta","W ","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U ","Np","Pu","Am","Cm", &
     "Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt" /);

contains

!
! Finds the nuclear charge for the given symbol
!
integer function get_z(csym)
  implicit none
  !! external
  character(2), intent(in) :: csym
  !! internal 
  integer :: z
  get_z = -1
  do z=1,z2sym_size;
    if(csym==z2sym(z)) then
      get_z = z
      exit
    endif
  enddo ! z
  if(get_z<1) then
    write(6,*) csym, ' ', __LINE__, ' ', __FILE__
    stop 'element not found'
  endif
end function ! get_z

!
!
!
character(2) function get_sym(z)
  implicit none
  integer, intent(in) :: z

  get_sym = "UN"
  
  select case (z)
  case(:0)
    get_sym = "GH"
  case(1:z2sym_size) 
    get_sym = z2sym(z)
  end select
    
end function ! get_sym    


end module ! m_z2sym
