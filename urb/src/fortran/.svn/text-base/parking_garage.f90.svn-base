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
      subroutine parking_garage
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! This subroutine set the velocities within the parking garage
! ERP/AAG 8/2007
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

         use datamodule ! make data from module "datamodule" visible
         implicit none
		   real K_garage
		   K_garage = 0.5	!multiplictive factor to reduce velocity in garage
         do k=kstart(ibuild),kend(ibuild)
            do j=jstart(ibuild),jend(ibuild)
               do i=istart(ibuild),iend(ibuild)
                  uo(i,j,k)  =uo(i,j,k)*K_garage
                  uo(i+1,j,k)=uo(i+1,j,k)*K_garage
                  vo(i,j,k)  =vo(i,j,k)*K_garage
                  vo(i,j+1,k)=vo(i,j+1,k)*K_garage
                  wo(i,j,k)  =wo(i,j,k)*K_garage
                  wo(i,j,k+1)=wo(i,j,k+1)*K_garage
			         icellflag(i,j,k) = 10
               enddo
            enddo
         enddo
         return
      end
