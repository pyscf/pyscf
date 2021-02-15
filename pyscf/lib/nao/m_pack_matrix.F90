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

module m_pack_matrix

  use m_precision, only : blas_int
#include "m_define_macro.F90"

  implicit none
  private blas_int

  interface get_block
    module procedure get_block4
    module procedure get_block8
  end interface !get_block
  
  interface pack2unpack;  
    module procedure dpack2unpack;
    module procedure spack2unpack;
  end interface ! pack2unpack
  
  interface  matmul_pack_full
    module procedure matmul_pack_full_4
    module procedure matmul_pack_full_8
  end interface ! matmul_pack_full

  interface matmul_full_pack
    module procedure matmul_full_pack_4
    module procedure matmul_full_pack_8
  end interface ! matmul_full_pack

  contains

!
!
!
function pack_size2unpack_dim(pack_size) result(ndim)
  implicit none
  !! external
  integer(8), intent(in) :: pack_size
  integer :: ndim
  !! internal
  real(8) :: rdim
  rdim = (sqrt(8.0D0*pack_size+1.0d0) - 1.0D0)/2.0D0
  ndim = int(rdim)
  if(ndim/=rdim) stop '!ndim/=rdim'
  if(ndim*(ndim+1)/2 /= pack_size) stop '!pack_size'
end function ! pack_size2unpack_dim 
  
!
! pack index -> i,j
!
pure subroutine get_row_col(p, i,j)
  implicit none
  !! external
  integer(8), intent(in) :: p
  integer, intent(inout) :: i,j
  !! internal
  integer(8) :: j8
  
  j8 = int((sqrt(1d0+8*(p-1))+1d0)/2.0D0)
  i = int(p - j8*(j8-1)/2)
  j = int(j8)

end subroutine ! get_row_col 

!
! i,j -> pack index
!
integer pure function get_pack_ind(i,j)
  implicit none
  integer, intent(in) :: i,j
  if(j>i) then ; get_pack_ind = j*(j-1)/2+i; else; get_pack_ind=i*(i-1)/2+j; endif
end function !  get_pack_ind 

!
!
!
subroutine put_block_pack_mat(A_block, si1, fi1, si2, fi2, A_pack)
  implicit none
  ! external
  real(8), intent(in)    :: A_block(:,:) 
  integer, intent(in)    :: si1, fi1, si2, fi2
  real(8), intent(inout) :: A_pack(:)

  ! internal
  integer :: j, i, local_i, local_j, ind

! Loop with second index in the upper level (without any block operation)
  local_j = 0
  do j=si2,fi2;
    local_j = local_j + 1
    local_i = 0
    do i=si1, fi1
      local_i = local_i + 1;
      if(j>i) then ; ind = j*(j-1)/2+i; else; ind=i*(i-1)/2+j; endif
      A_pack(ind) = A_block(local_i,local_j);
    end do
  end do
! END of Loop with second index in the upper level (without any block operation)

end subroutine ! put_block_pack_mat

!
!
!
subroutine put_block_pack_mat84(A_block, si,fi, sj,fj, AP)
  implicit none
  ! external
  real(8), intent(in)    :: A_block(:,:) 
  integer, intent(in)    :: si,fi, sj,fj
  real(4), intent(inout) :: AP(:)
  ! internal
  integer(8) :: j, ind1, ind2, fij, jj

! Loop with second index in the upper level (without any block operation)
  do j=sj,fj
    fij = min(int(fi,8),j)
    jj = j*(j-1)/2

    ind1 = jj+si
    ind2 = jj+fij
    if(ind2>size(AP)) then
      write(6,*) ind2, size(AP), __FILE__, __LINE__
      stop 'ind2>size(AP)'
    endif
 
    AP(ind1:ind2) = real(A_block(1:fij-si+1,j-sj+1),4);
  end do
! END of Loop with second index in the upper level (without any block operation)

end subroutine ! put_block_pack_mat84

!
!
!
subroutine put_block_pack_mat88(A_block, si,fi, sj,fj, AP)
  implicit none
  ! external
  real(8), intent(in)    :: A_block(:,:) 
  integer, intent(in)    :: si,fi, sj,fj
  real(8), intent(inout) :: AP(:)
  ! internal
  integer(8) :: j, ind1, ind2, fij, jj

! Loop with second index in the upper level (without any block operation)
  do j=sj,fj
    fij = min(int(fi,8),j)
    jj = j*(j-1)/2

    ind1 = jj+si
    ind2 = jj+fij
    if(ind2>size(AP)) then
      write(6,*) ind2, size(AP), __FILE__, __LINE__
      stop 'ind2>size(AP)'
    endif
 
    AP(ind1:ind2) = A_block(1:fij-si+1,j-sj+1);
  end do
! END of Loop with second index in the upper level (without any block operation)

end subroutine ! put_block_pack_mat88



!
!
!
subroutine spack2unpack(mat_pack, mat)
  implicit none
  real(4), intent(in), allocatable :: mat_pack(:)
  real(4), intent(inout), allocatable :: mat(:,:)
  !! internal
  integer :: b
  !! Dimensions
  integer :: ndim
  real(8) :: rdim
  
  if(.not. allocated(mat_pack)) then
    write(0,*) 'spack2unpack: .not. allocated(mat_pack) ==> return'
    return
  endif
  
  rdim = (sqrt(8.0D0*size(mat_pack)+1.0d0) - 1.0D0)/2.0D0
  ndim = int(rdim)
  if(ndim/=rdim) then
    write(0,*) 'spack2unpack: (ndim/=rdim) ==> return'
    return
  endif
  !! END Dimensions

  if(.not. allocated(mat)) allocate(mat(ndim, ndim))
  if(lbound(mat,1)/=1 .or. lbound(mat,2)/=1 .or. ubound(mat,1)/=ndim .or. ubound(mat,2)/=ndim ) deallocate(mat);
  if(.not. allocated(mat)) allocate(mat(ndim, ndim))
    
  do b=1,ndim
    mat(1:b,b) = mat_pack(b*(b-1)/2+1:b*(b-1)/2+b)
    mat(b,1:b) = mat_pack(b*(b-1)/2+1:b*(b-1)/2+b)
  end do
  
end subroutine ! spack2unpack

!
!
!
subroutine dpack2unpack(mat_pack, mat)
  implicit none
  real(8), intent(in), allocatable :: mat_pack(:)
  real(8), intent(inout), allocatable :: mat(:,:)
  !! internal
  integer :: b
  !! Dimensions
  integer :: ndim
  
  if(.not. allocated(mat_pack)) then
    write(0,*) 'dpack2unpack: .not. allocated(mat_pack) ==> return'
    return
  endif

  ndim = pack_size2unpack_dim(int(size(mat_pack),8))  
  !! END Dimensions

  if(.not. allocated(mat)) allocate(mat(ndim, ndim))
  if(lbound(mat,1)/=1 .or. lbound(mat,2)/=1 .or. ubound(mat,1)/=ndim .or. ubound(mat,2)/=ndim ) deallocate(mat);
  if(.not. allocated(mat)) allocate(mat(ndim, ndim))
    
  do b=1,ndim
    mat(1:b,b) = mat_pack(b*(b-1)/2+1:b*(b-1)/2+b)
    mat(b,1:b) = mat_pack(b*(b-1)/2+1:b*(b-1)/2+b)
  end do
  
end subroutine ! dpack2unpack

!
!
!
subroutine matmul_pack_full_4(ma,nb, A_pack, B,ldb, C,ldc)
  implicit none
  integer, intent(in) :: ma,nb,ldb,ldc
  real(4), intent(in) :: A_pack(*) ! upper, packed
  real(4), intent(in) :: B(ldb,*)
  real(4), intent(out) :: C(ldc,*)

  !! internal
  integer :: block_size, block, nblocks !! blocks
  real(4), allocatable :: aux(:,:)
  integer :: istart, ifinish, remainder_size

  block_size = 256
  allocate(aux(block_size, ma))
  nblocks = int(ma/block_size)

  !! Loop over blocks
  do block=1,nblocks
    istart = (block-1)*block_size+1
    ifinish = istart + block_size-1
    call get_block(A_pack, istart, ifinish, 1, ma, aux, block_size)
    call SGEMM('N', 'N', block_size, nb, ma, 1.0, aux, block_size, B, ldb, 0.0, C(istart,1), ldc);
  enddo
  !! END of loop over blocks

  !! Remainder
  remainder_size = modulo(ma, block_size)
  if(remainder_size==0) return
  istart = nblocks * block_size + 1
  ifinish = istart + remainder_size - 1
  call get_block(A_pack, istart, ifinish, 1, ma, aux, block_size)
  call SGEMM('N', 'N', remainder_size, nb, ma, 1.0, aux, block_size, B, ldb, 0.0, C(istart,1), ldc);
  !! END of Remainder

end subroutine ! matmul_pack_full_4

!
!
!
subroutine matmul_full_pack_4(ma,nb, A,lda, B_pack, C,ldc)
  implicit none
  integer, intent(in) :: ma,nb,lda,ldc
  real(4), intent(in) :: A(lda,*)  ! general, full
  real(4), intent(in) :: B_pack(*) ! upper, packed
  real(4), intent(out) :: C(ldc,*)

  !! internal
  integer :: block_size, block, nblocks !! blocks
  real(4), allocatable :: aux(:,:)
  integer :: kstart, kfinish, remainder_size

  block_size = 256
  allocate(aux(nb,block_size))
  nblocks = int(nb/block_size)

  !! Loop over blocks
  do block=1,nblocks
    kstart = (block-1)*block_size+1
    kfinish = kstart + block_size-1
    call get_block(B_pack, 1, nb, kstart, kfinish, aux, nb )
    call SGEMM('N', 'N', ma, block_size, nb, 1.0, A, lda, aux, nb, 0.0, C(1,kstart),ldc);
  enddo
  !! END of loop over blocks

  !! Remainder
  remainder_size = modulo(nb, block_size)
  if(remainder_size==0) return
  kstart = nblocks * block_size + 1
  kfinish = kstart + remainder_size - 1
  call get_block(B_pack, 1, nb, kstart, kfinish, aux, nb)
  call SGEMM('N', 'N', ma, remainder_size, nb, 1.0, A, lda, aux, nb, 0.0, C(1,kstart),ldc);
  !! END of Remainder

end subroutine ! matmul_pack_full_4

!
!
!
subroutine matmul_full_pack_8(ma,nb, A,lda, B_pack, C,ldc)
  implicit none
  integer, intent(in) :: ma,nb,lda,ldc
  real(8), intent(in) :: A(lda,*)  ! general, full
  real(8), intent(in) :: B_pack(*) ! upper, packed
  real(8), intent(out) :: C(ldc,*)

  !! internal
  integer :: kstart, kfinish, rs, bs, iblock, nblocks !! blocks
  real(8), allocatable :: aux(:,:)

  bs = 256
  allocate(aux(nb,bs))
  nblocks = int(nb/bs)

  !! Loop over blocks
  do iblock=1,nblocks
    kstart = (iblock-1)*bs+1
    kfinish = kstart + bs-1
    call get_block(B_pack, 1, nb, kstart, kfinish, aux, nb )
    call DGEMM('N', 'N', ma, bs, nb, 1D0, A, lda, aux, nb, 0D0, C(1,kstart),ldc)
  enddo
  !! END of loop over blocks

  !! Remainder
  rs = modulo(nb, bs)
  if(rs==0) return
  kstart = nblocks * bs + 1
  kfinish = kstart + rs - 1
  call get_block(B_pack, 1, nb, kstart, kfinish, aux, nb)
  call DGEMM('N', 'N', ma, rs, nb, 1D0, A, lda, aux, nb, 0D0, C(1,kstart),ldc)
  !! END of Remainder

end subroutine ! matmul_full_pack_8

!
!
!
subroutine get_block8(A_pack, istart, ifinish, jstart, jfinish, A_block, lda)
  implicit none
  ! external
  integer, intent(in)  :: istart, ifinish, jstart, jfinish, lda
  real(8), intent(in)  :: A_pack(*)
  real(8), intent(out) :: A_block(lda,*) 

  ! internal
  integer :: j, i, ind

!  do j=jstart, jfinish
!    do i=istart, min(j,ifinish)
!      ind = j*(j-1)/2+i
!      A_block(i-istart+1,j-jstart+1) = A_pack(ind)
!    end do
!  end do

!  do j=jstart, jfinish
!    do i=j+1, ifinish
!      ind=i*(i-1)/2+j
!      A_block(i-istart+1,j-jstart+1) = A_pack(ind)
!    end do
!  end do


  do j=jstart,jfinish;
    do i=istart, ifinish
      if(j>i) then ; ind = j*(j-1)/2+i; else; ind=i*(i-1)/2+j; endif
      A_block(i-istart+1,j-jstart+1) = A_pack(ind);
    end do
  end do

end subroutine ! get_block8

!
!
!
subroutine get_block4(A_pack, istart, ifinish, jstart, jfinish, A_block, lda)
  implicit none
  ! external
  integer, intent(in)  :: istart, ifinish, jstart, jfinish, lda
  real(4), intent(in)  :: A_pack(*)
  real(4), intent(out) :: A_block(lda,*) 

  ! internal
  integer :: j, i, ind

! Loop with second index in the upper level (without any block operation)
  do j=jstart,jfinish;
    do i=istart, ifinish
      if(j>i) then ; ind = j*(j-1)/2+i; else; ind=i*(i-1)/2+j; endif
      A_block(i-istart+1,j-jstart+1) = A_pack(ind);
    end do
  end do
! END of Loop with second index in the upper level (without any block operation)

end subroutine ! get_block4


!
!
!
subroutine matmul_pack_full_8(ma,nb, A_pack, B,ldb, C,ldc)
  implicit none
  integer, intent(in) :: ma,nb,ldb,ldc
  real(8), intent(in) :: A_pack(*) ! upper, packed
  real(8), intent(in) :: B(ldb,*)
  real(8), intent(out) :: C(ldc,*)

  !! internal
  integer :: block_size, block, nblocks !! blocks
  real(8), allocatable :: aux(:,:)
  integer :: istart, ifinish, remainder_size

  block_size = 256
  allocate(aux(block_size, ma))
  nblocks = int(ma/block_size)

  !! Loop over blocks
  do block=1,nblocks
    istart = (block-1)*block_size+1
    ifinish = istart + block_size-1
    call get_block(A_pack, istart, ifinish, 1, ma, aux, block_size)
    call DGEMM('N', 'N', block_size, nb, ma, 1.0D0, aux, block_size, B, ldb, 0.0D0, C(istart,1), ldc);
  enddo
  !! END of loop over blocks

  !! Remainder
  remainder_size = modulo(ma, block_size)
  if(remainder_size==0) return
  istart = nblocks * block_size + 1
  ifinish = istart + remainder_size - 1
  call get_block(A_pack, istart, ifinish, 1, ma, aux, block_size)
  call DGEMM('N', 'N', remainder_size, nb, ma, 1.0D0, aux, block_size, B, ldb, 0.0D0, C(istart,1), ldc);
  !! END of Remainder

  _dealloc(aux)
  
end subroutine ! matmul_pack_full_d

!
!
!
subroutine dunpack2pack(mat, mat_pack)
  implicit none
  real(8), intent(in) :: mat(:,:)
  real(8), intent(inout), allocatable :: mat_pack(:)
  
  integer :: ndim, b !i,j, ii
    
  ndim = size(mat,1)

  if(.not. allocated(mat_pack)) allocate(mat_pack(ndim*(ndim+1)/2))
    
  do b=1, ndim
    call DCOPY(b, mat(1,b),1, mat_pack(b*(b-1)/2+1), 1)
  end do

end subroutine dunpack2pack

!
!
!
subroutine spack(UPLO, n, A, AP)
  !use m_upper, only : upper
  implicit none
  !! external
  character, intent(in) :: UPLO
  integer, intent(in) :: n
  real(4), intent(in) :: A(:,:)
  real(4), intent(inout) :: AP(:)
  !! internal  
  integer :: lda, nn(2), b

  if(n<1) then
    write(6,*) __FILE__, __LINE__
    stop 'n<1'
  endif  
  nn = ubound(A)
  if(any(nn<n)) then
    write(0,*) __FILE__, __LINE__
    stop 'any(nn<n)'
  endif
  lda = size(A,1)
  if(size(AP)<n*(n+1)/2) then
    write(0,*) __FILE__, __LINE__
    stop '!size(AP)'
  endif  

  select case(UPLO)
  case("u", "U")
    do b=1, n
      call SCOPY(b, A(1,b),1, AP(b*(b-1)/2+1), 1)
    end do
  case("l", "L")
    do b=1, n
      AP(b*(b-1)/2+1:b*(b-1)/2+b) = A(b,1:b)
    end do
  case default
    write(0,*) __FILE__, __LINE__
    stop '!UPLO'
  end select    

end subroutine !dpack


!
!
!
subroutine dpack(UPLO, n, A, AP)
  !use m_upper, only : upper
  implicit none
  !! external
  character, intent(in) :: UPLO
  integer, intent(in) :: n
  real(8), intent(in) :: A(:,:)
  real(8), intent(inout) :: AP(:)
  !! internal  
  integer :: lda, nn(2), b

  if(n<1) then
    write(6,*) __FILE__, __LINE__
    stop 'n<1'
  endif  
  nn = ubound(A)
  if(any(nn<n)) then
    write(0,*) __FILE__, __LINE__
    stop 'any(nn<n)'
  endif
  lda = size(A,1)
  if(size(AP)<n*(n+1)/2) then
    write(0,*) __FILE__, __LINE__
    stop '!size(AP)'
  endif  

  select case(UPLO)
  case("u", "U")
    do b=1, n
      call DCOPY(b, A(1,b),1, AP(b*(b-1)/2+1), 1)
    end do
  case("l", "L")
    do b=1, n
      AP(b*(b-1)/2+1:b*(b-1)/2+b) = A(b,1:b)
    end do
  case default
    write(0,*) __FILE__, __LINE__
    stop '!UPLO'
  end select    

end subroutine !dpack

!
!
!
subroutine cpack(UPLO, n, A, AP)
  !use m_upper, only : upper
  implicit none
  !! external
  character, intent(in) :: UPLO
  integer, intent(in) :: n
  complex(4), intent(in) :: A(:,:)
  complex(4), intent(inout) :: AP(:)
  !! internal  
  integer :: lda, nn(2), b

  if(n<1) then
    write(6,*) __FILE__, __LINE__
    stop 'n<1'
  endif  
  nn = ubound(A)
  if(any(nn<n)) then
    write(0,*) __FILE__, __LINE__
    stop 'any(nn<n)'
  endif
  lda = size(A,1)
  if(size(AP)<n*(n+1)/2) then
    write(0,*) __FILE__, __LINE__
    stop '!size(AP)'
  endif  

  select case(UPLO)
  case("u", "U")
    do b=1, n
      call CCOPY(b, A(1,b),1, AP(b*(b-1)/2+1), 1)
    end do
  case("l", "L")
    do b=1, n
      AP(b*(b-1)/2+1:b*(b-1)/2+b) = A(b,1:b)
    end do
  case default
    write(0,*) __FILE__, __LINE__
    stop '!UPLO'
  end select    

end subroutine !zpack


!
!
!
subroutine zpack(UPLO, n, A, AP)
  !use m_upper, only : upper
  implicit none
  !! external
  character, intent(in) :: UPLO
  integer, intent(in) :: n
  complex(8), intent(in) :: A(:,:)
  complex(8), intent(inout) :: AP(:)
  !! internal  
  integer :: lda, nn(2), b

  if(n<1) then
    write(6,*) __FILE__, __LINE__
    stop 'n<1'
  endif  
  nn = ubound(A)
  if(any(nn<n)) then
    write(0,*) __FILE__, __LINE__
    stop 'any(nn<n)'
  endif
  lda = size(A,1)
  if(size(AP)<n*(n+1)/2) then
    write(0,*) __FILE__, __LINE__
    stop '!size(AP)'
  endif  

  select case(UPLO)
  case("u", "U")
    do b=1, n
      call ZCOPY(b, A(1,b),1, AP(b*(b-1)/2+1), 1)
    end do
  case("l", "L")
    do b=1, n
      AP(b*(b-1)/2+1:b*(b-1)/2+b) = A(b,1:b)
    end do
  case default
    write(0,*) __FILE__, __LINE__
    stop '!UPLO'
  end select    

end subroutine !zpack

!
!  B := alpha*A + beta*B
!
subroutine zunpackh(n, alpha, AP, beta, B)
  implicit none
  !! external
  integer, intent(in) :: n !! dimension of matrix
  complex(8), intent(in) :: alpha
  complex(8), intent(in) :: AP(:) !AP(*)
  complex(8), intent(in) :: beta
  complex(8), intent(inout) :: B(:,:) !B(ldm,*)
  !! internal
  integer :: j

  if(beta==0D0) then

    if(alpha==0D0) then

      B(1:n,1:n) = 0

    else if(alpha==1D0) then

      do j=1,n; B(1:j,j)   = AP(j*(j-1)/2+1:j*(j-1)/2+j); enddo
      !do j=2,n; B(j,1:j-1) = AP(j*(j-1)/2+1:j*(j-1)/2+j); enddo
      !do j=1,n; B(j,1:j) = AP(j*(j-1)/2+1:j*(j-1)/2+j); enddo
      do j=1,n; B(j,1:j) = conjg(B(1:j,j)); enddo
    else 
   
      do j=1,n; B(1:j,j)   = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j); enddo
      !do j=2,n; B(j,1:j-1) = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j); enddo
      !do j=1,n; B(j,1:j) = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j); enddo
      do j=1,n; B(j,1:j) = conjg(B(1:j,j)); enddo
    end if ! alpha  

  else if (beta==1D0) then      

    if(alpha==0D0) then
      continue
    
    else if (alpha==1D0) then

      do j=1,n; B(1:j,j)   = AP(j*(j-1)/2+1:j*(j-1)/2+j)+B(1:j,j); enddo
      !do j=2,n; B(j,1:j-1) = AP(j*(j-1)/2+1:j*(j-1)/2+j)+B(1:j,j); enddo
      !do j=1,n; B(j,1:j) = AP(j*(j-1)/2+1:j*(j-1)/2+j)+B(1:j,j); enddo
      do j=1,n; B(j,1:j) = conjg(B(1:j,j)); enddo
    else 
      do j=1,n; B(1:j,j)   = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j)+B(1:j,j); enddo
      !do j=2,n; B(j,1:j-1) = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j)+B(1:j,j); enddo
      !do j=1,n; B(j,1:j) = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j)+B(1:j,j); enddo
      do j=1,n; B(j,1:j) = conjg(B(1:j,j)); enddo
      endif

  else ! any beta
  
    if(alpha==0D0) then

      do j=1,n; B(1:j,j)   = beta*B(1:j,j); enddo
      !do j=2,n; B(j,1:j-1) = beta*B(1:j,j); enddo
      !do j=1,n; B(j,1:j) = beta*B(1:j,j); enddo
      do j=1,n; B(j,1:j) = conjg(B(1:j,j)); enddo
    else if (alpha==1D0) then

      do j=1,n; B(1:j,j)   = AP(j*(j-1)/2+1:j*(j-1)/2+j)+beta*B(1:j,j); enddo
      !do j=2,n; B(j,1:j-1) = AP(j*(j-1)/2+1:j*(j-1)/2+j)+beta*B(1:j,j); enddo
      !do j=1,n; B(j,1:j) = AP(j*(j-1)/2+1:j*(j-1)/2+j)+beta*B(1:j,j); enddo
      do j=1,n; B(j,1:j) = conjg(B(1:j,j)); enddo
    else
   
      do j=1,n; B(1:j,j)   = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j)+beta*B(1:j,j); enddo
      !do j=2,n; B(j,1:j-1) = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j)+beta*B(1:j,j); enddo
      !do j=1,n; B(j,1:j) = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j)+beta*B(1:j,j); enddo
      do j=1,n; B(j,1:j) = conjg(B(1:j,j)); enddo
    endif 
    
  endif

end subroutine ! zunpackh

!
!  B := alpha*AP + beta*B
!
subroutine zunpacks(n, alpha, AP, beta, B)
  implicit none
  !! external
  integer, intent(in) :: n !! dimension of matrix
  complex(8), intent(in) :: alpha
  complex(8), intent(in) :: AP(:) !AP(*)
  complex(8), intent(in) :: beta
  complex(8), intent(inout) :: B(:,:) !B(ldm,*)
  !! internal
  integer :: j

  if(beta==0D0) then

    if(alpha==0D0) then

      B(1:n,1:n) = 0

    else if(alpha==1D0) then

      do j=1,n; B(1:j,j)   = AP(j*(j-1)/2+1:j*(j-1)/2+j); enddo
      !do j=2,n; B(j,1:j-1) = AP(j*(j-1)/2+1:j*(j-1)/2+j); enddo
      !do j=1,n; B(j,1:j) = AP(j*(j-1)/2+1:j*(j-1)/2+j); enddo
      do j=1,n; B(j,1:j) = (B(1:j,j)); enddo
    else 
   
      do j=1,n; B(1:j,j)   = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j); enddo
      !do j=2,n; B(j,1:j-1) = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j); enddo
      !do j=1,n; B(j,1:j) = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j); enddo
      do j=1,n; B(j,1:j) = (B(1:j,j)); enddo
    end if ! alpha  

  else if (beta==1D0) then      

    if(alpha==0D0) then
      continue
    
    else if (alpha==1D0) then

      do j=1,n; B(1:j,j)   = AP(j*(j-1)/2+1:j*(j-1)/2+j)+B(1:j,j); enddo
      !do j=2,n; B(j,1:j-1) = AP(j*(j-1)/2+1:j*(j-1)/2+j)+B(1:j,j); enddo
      !do j=1,n; B(j,1:j) = AP(j*(j-1)/2+1:j*(j-1)/2+j)+B(1:j,j); enddo
      do j=1,n; B(j,1:j) = (B(1:j,j)); enddo
    else 
      do j=1,n; B(1:j,j)   = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j)+B(1:j,j); enddo
      !do j=2,n; B(j,1:j-1) = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j)+B(1:j,j); enddo
      !do j=1,n; B(j,1:j) = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j)+B(1:j,j); enddo
      do j=1,n; B(j,1:j) = (B(1:j,j)); enddo
      endif

  else ! any beta
  
    if(alpha==0D0) then

      do j=1,n; B(1:j,j)   = beta*B(1:j,j); enddo
      !do j=2,n; B(j,1:j-1) = beta*B(1:j,j); enddo
      !do j=1,n; B(j,1:j) = beta*B(1:j,j); enddo
      do j=1,n; B(j,1:j) = (B(1:j,j)); enddo
    else if (alpha==1D0) then

      do j=1,n; B(1:j,j)   = AP(j*(j-1)/2+1:j*(j-1)/2+j)+beta*B(1:j,j); enddo
      !do j=2,n; B(j,1:j-1) = AP(j*(j-1)/2+1:j*(j-1)/2+j)+beta*B(1:j,j); enddo
      !do j=1,n; B(j,1:j) = AP(j*(j-1)/2+1:j*(j-1)/2+j)+beta*B(1:j,j); enddo
      do j=1,n; B(j,1:j) = (B(1:j,j)); enddo
    else
   
      do j=1,n; B(1:j,j)   = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j)+beta*B(1:j,j); enddo
      !do j=2,n; B(j,1:j-1) = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j)+beta*B(1:j,j); enddo
      !do j=1,n; B(j,1:j) = alpha*AP(j*(j-1)/2+1:j*(j-1)/2+j)+beta*B(1:j,j); enddo
      do j=1,n; B(j,1:j) = (B(1:j,j)); enddo
    endif 
    
  endif

end subroutine ! zunpacks

end module !m_pack_matrix


