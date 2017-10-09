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
       subroutine divergence
!cccccccccccccccccccccccccccccccccccccccccccccccc
! This subroutine calculates the divergence of  
! the initial velocity field        
! Eric Pardyjak               
! December 2000               
! r(i,j,k)= (-2*alpha1**2)*divergence     
! where r is the right hand side of the functional being
! minimized.
!cccccccccccccccccccccccccccccccccccccccccccccccc
          use datamodule ! make data from module "datamodule" visible
          implicit none
          real ovdx,ovdy,ovdz
          ovdx=1./dx
          ovdy=1./dy
          do k=1,nz-1
             ! MAN 07/25/2008 stretched vertical grid
             ovdz=1./(dz_array(k))
             do j=1,ny-1
                do i=1,nx-1
                   r(i,j,k)=(-2.*alpha1**2)*(ovdx*(uo(i+1,j,k)-uo(i,j,k)) +   &
                                             ovdy*(vo(i,j+1,k)-vo(i,j,k)) +   &
                                             ovdz*(wo(i,j,k+1)-wo(i,j,k)))
                enddo
             enddo
          enddo
          return
       end

