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
       subroutine euler
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! subroutine to calculate mass consistent velocity field	
! subroutine called by main.f90
! subroutine calls outfile.f90 to print out results
! alpha1 = horizontal gaussian precision moduli			
! alpha2 = vertical gaussian precision moduli			
! Eric Pardyjak							
! December 2000							
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

          use datamodule ! make data from module "datamodule" visible
          implicit none
          real ovdx,ovdy,ovdz
          real ovalph1,ovalph2
          ovalph1=1./(2.*alpha1**2)
          ovalph2=1./(2.*alpha2**2)
          ovdx=1./dx
          ovdy=1./dy
! make sure outside cells still have uo value
          do k=1,nz
             do j=1,ny
                do i=1,nx
                   u(i,j,k)=uo(i,j,k)
                   v(i,j,k)=vo(i,j,k)
                   w(i,j,k)=wo(i,j,k)
                enddo
             enddo
          enddo
! using staggard grid arrangement calculate new mass consistent
! velocities. Note that we go from 2,nx-1 because we are using
! the p2's to the left and right. This is kind of a bruut force
! method.
          do k=2,nz-1
             ! MAN 07/25/2008 stretched vertical grid
             ovdz=1./(0.5*(dz_array(k)+dz_array(k-1)))
             do j=2,ny-1
                do i=2,nx-1
                   u(i,j,k)=uo(i,j,k)+ovalph1*ovdx*(p2(i,j,k)-p2(i-1,j,k))
                   v(i,j,k)=vo(i,j,k)+ovalph1*ovdy*(p2(i,j,k)-p2(i,j-1,k))
                   w(i,j,k)=wo(i,j,k)+ovalph2*ovdz*(p2(i,j,k)-p2(i,j,k-1))
                enddo
             enddo
          enddo
! all cells that are buildings have a zero velocity within them
          do k=1,nz-1
             do j=1,ny-1
                do i=1,nx-1
                   if(icellflag(i,j,k).eq.0.)then ! MAN 7/8/2005 celltype definition change
                      u(i,j,k)  =0.
                      u(i+1,j,k)=0.
                      v(i,j,k)  =0.
                      v(i,j+1,k)=0.
                      w(i,j,k)  =0.
                      w(i,j,k+1)=0.
                   endif
                enddo
             enddo
          enddo
          return
       end
