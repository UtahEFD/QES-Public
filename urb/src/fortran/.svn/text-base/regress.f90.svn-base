!                                 Notice
!  This program was prepared by the University of California (University)
!  under Contract W-7405-ENG-36 with the U.S. Department of Energy (DOE).
!  All rights in the program are reserved by DOE on behalf of the Government
!  and the University pursuant to the contract. You are authorized to use
!  this program for Government purposes but it is not to be released or
!  distributed to the public.
!  NEITHER THE UNITED STATES NOR THE UNITED STATES DEPARTMENT OF ENERGY,
!  NOR THE UNIVERSITY OF CALIFORNIA, NOR ANY OF THEIR EMPLOYEES,
!  MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY
!  OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS, OF
!  ANY INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS
!  THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
!
      subroutine regress
!subroutine regress to do a linear regression to determin
!ustar and zocan
!ANU 8/30/2005
	 
         use datamodule
         implicit none

         real sumx
         real sumy
         real sumxy
         real sumxsq
         integer no  !, index,c
         integer k_top
         real xm,ym,y
         real ao
         real local_mag

!         sumx = 0
!         sumy = 0
!         sumxy = 0
!         sumxsq = 0

!         do index = 1,inumcanopy
!            do c = nint(cH(index)/dz)+2,nz-2
!     
!               y = log((real(c) - 1.5)*dz)
!               sumx = sumx + canopy_uin_prof(index,c)
!!	print *,"u1=",canopy_uin_prof(index,c)
!               sumy = sumy + y
!               sumxy = sumxy + canopy_uin_prof(index,c) * y
!               sumxsq = sumxsq + canopy_uin_prof(index,c)**2
!            enddo
!
!            no = nint((nz-1)-((cH(index)/dz)+2))
!            custar(index) = vk * (((no* sumxsq) - (sumx**2)) / ((no * sumxy) - (sumx * sumy)))
!            !     print *,"ustar=",ustar
!            xm = sumx / no
!            ym = sumy / no
!            ao = ym - ((vk / custar(index)) * xm)
!            czo(index) = exp(ao)
!!	print *,"zocan=",zocan
!
!
!         enddo
         
         do j=1,ny-1
            do i=1,nx-1
               if(canopy_top(i,j) .gt. 0.)then
                  do k=2,nz-1
                     canopy_ktop(i,j)=k
                     if(canopy_top(i,j) .lt. zm(k+1))exit
                  enddo
                  do k=canopy_ktop(i,j),nz-1
                     k_top=k
                     if(2.*canopy_top(i,j) .lt. zm(k+1))exit
                  enddo
                  if(k_top .eq. canopy_ktop(i,j))k_top=canopy_ktop(i,j)+1
                  if(k_top .gt. nz-1)k_top=nz-1
                  sumx = 0
                  sumy = 0
                  sumxy = 0
                  sumxsq = 0
                  no = 0
                  do k = canopy_ktop(i,j),k_top
                     no = no + 1
                     local_mag=sqrt((uo(i,j,k)**2.)+(vo(i,j,k)**2.))
                     ! MAN 07/25/2008 stretched vertical grid
                     y = log(zm(k))
                     sumx = sumx + local_mag
                     sumy = sumy + y
                     sumxy = sumxy + local_mag * y
                     sumxsq = sumxsq + local_mag**2
                  enddo
                  canopy_ustar(i,j) = vk * (((no* sumxsq) - (sumx**2))/((no * sumxy) - (sumx * sumy)))
                  xm = sumx / no
                  ym = sumy / no
                  ao = ym - ((vk / canopy_ustar(i,j)) * xm)
                  canopy_zo(i,j) = exp(ao)
               endif
            enddo
         enddo
      end