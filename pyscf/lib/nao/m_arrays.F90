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

module m_arrays

  implicit none

  type d_array1_t; real(8), allocatable :: array(:);      end type ! d_array1_t
  type d_array2_t; real(8), allocatable :: array(:,:);    end type ! d_array2_t
  type d_array3_t; real(8), allocatable :: array(:,:,:);  end type ! d_array3_t
  type d_array4_t; real(8), allocatable :: array(:,:,:,:);end type ! d_array4_t

  type i_array1_t; integer, allocatable :: array(:);      end type ! i_array1_t
  type i_array2_t; integer, allocatable :: array(:,:);    end type ! i_array2_t
    
  type s_array1_t; real(4), allocatable :: array(:);      end type ! s_array1_t
  type s_array2_t; real(4), allocatable :: array(:,:);    end type ! s_array2_t
  type s_array3_t; real(4), allocatable :: array(:,:,:);  end type ! s_array3_t

  type z_array1_t; complex(8), allocatable :: array(:);     end type ! z_array1_t
  type z_array2_t; complex(8), allocatable :: array(:,:);   end type ! z_array2_t
  type z_array3_t; complex(8), allocatable :: array(:,:,:); end type ! z_array3_t

  type c_array1_t; complex(4), allocatable :: array(:);     end type ! c_array1_t
  type c_array2_t; complex(4), allocatable :: array(:,:);   end type ! c_array2_t
  type c_array3_t; complex(4), allocatable :: array(:,:,:); end type ! c_array3_t

  type l_array1_t; logical, allocatable :: array(:);      end type ! l_array1_t
  type i1_array1_t; integer(1), allocatable :: array(:);  end type ! i1_array1_t

  type d_tensor_sym_freq_t
    real(8), allocatable :: sf(:,:,:)
    real(8), allocatable :: sf_zip(:,:,:)
    complex(8), allocatable :: funct(:,:,:)
    complex(8), allocatable :: funct_zip(:,:,:)
    real(8), allocatable :: X_up(:,:)
    real(8), allocatable :: funct_static(:)
  end type ! d_tensor_sym_freq_t

  interface chk_sum
    module procedure chk_sum_3_d_array2_t
  end interface ! chk_sum   
  
  interface sum_abs
    module procedure sum_abs_3_d_array2_t
  end interface  ! sum_abs

  contains

!
!
!
function chk_sum_3_d_array2_t(array) result(chk_sum)
  implicit none
  !! external
  type(d_array2_t), intent(in) :: array(:,:,:)
  real(8) :: chk_sum
  !! internal
  integer :: i1,i2,i3
  
  chk_sum = 0
  do i3=1,size(array,3)
    do i2=1,size(array,2)
      do i1=1,size(array,1)
        chk_sum = chk_sum + sum(array(i1,i2,i3)%array)
      enddo
    enddo
  enddo  
end function ! chk_sum_3_d_array2_t

!
!
!
function sum_abs_3_d_array2_t(array1, array2) result(res)
  implicit none
  !! external
  type(d_array2_t), intent(in) :: array1(:,:,:), array2(:,:,:)
  real(8) :: res
  !! internal
  integer :: i1,i2,i3
  
  res = 0
  do i3=1,size(array1,3)
    do i2=1,size(array1,2)
      do i1=1,size(array1,1)
        res = res + sum(abs(array1(i1,i2,i3)%array-array2(i1,i2,i3)%array))
      enddo
    enddo
  enddo
  
end function ! chk_sum_3_d_array2_t

end module !m_arrays

