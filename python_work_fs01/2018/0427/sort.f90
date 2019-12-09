!
! movie of sort algorithm
! niconico nm5251176
!
recursive subroutine quicksort(a,first,last) 
! slashdot.jp t-nissie no nikki 2008/08/24 "Fortran de kaita quick sort"
  implicit none
  real*8 a(*), x, t
  integer first, last, i, j

  x=a((first+last)/2)
  i=first
  j=last
  do
    do while (a(i)<x)
      i = i + 1
    enddo
    do while (x<a(j))
      j = j - 1
    enddo
    if(i >= j)exit
    t = a(i); a(i) = a(j); a(j) = t
    i = i + 1
    j = j - 1
  enddo
  if(first < i - 1) call quicksort(a,first, i-1 )
  if(j + 1 < last ) call quicksort(a,j+1  , last)
end subroutine quicksort

subroutine selectionsort(a,n) 
! www.geocities.jp/eyeofeconomyandhealth/homepage/mondai/kotae1-5.html#1
  implicit none
  real*8 a(*), t
  integer i, j, k, n

  do i = 1, n - 1
    k = i
    do j = i + 1, n
      if(a(k).gt.a(j)) k = j
    enddo
    t    = a(i)
    a(i) = a(k)
    a(k) = t
  enddo
end subroutine selectionsort

program main
real*8, allocatable :: tc(:), e(:,:), k(:), temp(:)
integer :: ik, ibnd, nk, nbnd
  nk = 318
  allocate(tc(nk))
  do ik = 1, nk
    read(5,*)  tc(ik)
  enddo

  ! call quicksort(temp,1,nbnd)
  call selectionsort(tc,nk)

  do ik = 1, nk
    write(*,'(f14.8)') tc(ik)
  enddo
stop
end program main
