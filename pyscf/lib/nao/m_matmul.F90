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

module m_matmul

  implicit none

  contains

!
!
!
subroutine matmul_Ax_d(A, x, y)
  implicit none
  real(8), intent(in):: A(:,:), x(:)
  real(8), intent(inout):: y(:)
  
  !write(6,*) "in matmul_AB_d"
  ! SUBROUTINE DGEMV(TRANS, M, N, ALPHA, A, LDA, X,
  ! INCX, BETA, Y, INCY)
  
  call DGEMV('N', size(A,1), size(A,2), 1.0d0, A, size(A,1), x, &
       1, 0.0d0, Y, 1)
  
end subroutine matmul_Ax_d

!
!
!
subroutine matmul_Ax_z(A, x, y)
  implicit none
  complex(8), intent(in):: A(:,:), x(:)
  complex(8), intent(inout):: y(:)

  complex(8):: alpha, beta

  alpha=1.0d0
  beta=0.0d0
    
  call ZGEMV('N', size(A,1), size(A,2), alpha, A, size(A,1), x, &
       1, beta, Y, 1)
  
end subroutine matmul_Ax_z

!
!
!
subroutine matmul_Atx_d(A, x, y)
  implicit none
  real(8), intent(in):: A(:,:), x(:)
  real(8), intent(inout):: y(:)
  
  !write(6,*) "in matmul_AB_d"
  ! SUBROUTINE DGEMV(TRANS, M, N, ALPHA, A, LDA, X,
  ! INCX, BETA, Y, INCY)
  
  call DGEMV('T', size(A,1), size(A,2), 1.0d0, A, size(A,1), x, &
       1, 0.0d0, Y, 1)
  
end subroutine 


!
!
!
subroutine matmul_AB_d(A, B, D)
  implicit none
  real(8), intent(in):: A(:,:), B(:,:)
  real(8), intent(inout):: D(:,:)
  
  !write(6,*) "in matmul_AB_d"
  ! SUBROUTINE xGEMM ( TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )
  call DGEMM('N', 'N', size(A,1), size(B,2), size(A,2), 1.0D0, &
       A, size(A,1), B, size(B,1), 0.0D0, D, size(D,1))
  !write(6,*) "in matmul_AB_d"
  
end subroutine matmul_AB_d

!
!
!
subroutine matmul_AB_z(A, B, D)
  implicit none
  complex(8), intent(in):: A(:,:), B(:,:)
  complex(8), intent(inout):: D(:,:)

  complex(8):: alpha, beta

  alpha=1.0d0
  beta=0.0d0
  
  ! SUBROUTINE xGEMM ( TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )
  call ZGEMM('N', 'N', size(A,1), size(B,2), size(A,2), alpha, &
       A, size(A,1), B, size(B,1), beta, D, size(D,1))
  
end subroutine matmul_AB_z

!
!
!
subroutine matmul_AtB_d(A, B, D)
  implicit none
  real(8), intent(in):: A(:,:), B(:,:)
  real(8), intent(inout):: D(:,:)

  !write(6,*) "in matmul_AtB_d"
  ! SUBROUTINE xGEMM ( TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )
  call DGEMM('T', 'N', size(A,2), size(B,2), size(A,1), 1.0D0, &
       A, size(A,1), B, size(B,1), 0.0D0, D, size(D,1))
  !write(6,*) "in matmul_AtB_d"

end subroutine matmul_AtB_d

!
!
!
subroutine matmul_AhB_z(A, B, D)
  implicit none
  complex(8), intent(in):: A(:,:), B(:,:)
  complex(8), intent(inout):: D(:,:)

  complex(8):: alpha, beta

  alpha=1.0d0
  beta=0.0d0
    
  ! SUBROUTINE xGEMM ( TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )
  call ZGEMM('C', 'N', size(A,2), size(B,2), size(A,1), alpha, &
       A, size(A,1), B, size(B,1), beta, D, size(D,1))

end subroutine matmul_AhB_z

!
!
!
subroutine matmul_ABt_d(A, B, D)
  implicit none
  real(8), intent(in):: A(:,:), B(:,:)
  real(8), intent(inout):: D(:,:)

  !write(6,*) "in matmul_ABt_d"
  ! SUBROUTINE xGEMM ( TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )
  call DGEMM('N', 'T', size(A,1), size(B,1), size(A,2), 1.0D0, &
       A, size(A,1), B, size(B,1), 0.0D0, D, size(D,1))
  !write(6,*) "in matmul_ABt_d*"

end subroutine matmul_ABt_d

!
!
!
subroutine matmul_ABt_z(A, B, D)
  implicit none
  complex(8), intent(in):: A(:,:), B(:,:)
  complex(8), intent(inout):: D(:,:)

  complex(8):: alpha, beta

  alpha=1.0d0
  beta=0.0d0
    
  ! SUBROUTINE xGEMM ( TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )
  call ZGEMM('N', 'T', size(A,1), size(B,2), size(A,2), alpha, &
       A, size(A,1), B, size(B,1), beta, D, size(D,1))
  
end subroutine matmul_ABt_z


end module !m_matmul
