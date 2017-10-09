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
       subroutine sor3d
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! SOR3D is a 3 dimensional successive overrelaxation method solver.
! err is an absolute error measurement
! iter is the number of iterations spent in the solver
! p1 & p2 are Lagrange multipliers
! omegarelax is overelaxation coefficient specified in main.f
! ERP
! inlcudes PKKs f90 modifications 10/01/2002
! includes twh's deallocation procedure 1/8 and 1/9 2003
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

          use datamodule ! make data from module "datamodule" visible
! operations  done a priori to speed up the code AAG & IS  07/03/06
          implicit none
          integer iter
          real one_minus_omegarelax, one_over_nxnynz,res_red_factor
          
          one_minus_omegarelax= (1-omegarelax)
          one_over_nxnynz=1/real((nx-1)*(ny-1)*(nz-1))
          p1(1:nx-1,1:ny-1,1:nz-1)=0.0
          res_red_factor=1/(10**residual_reduction)
          if(num_sites .eq. 1 .and. inumbuild .eq. 0 .and. inumcanopy .eq. 0)itermax=0
!---begin iteration loop   
! loops change for 1, imax-1 to 2,imax-2, AAG & IS 07/03/06
          do iter=1,itermax
             abse=0.0
             p2(1:nx-1,1:ny-1,1:nz-1)=p1(1:nx-1,1:ny-1,1:nz-1)
             do k=2,nz-2
                do j=2,ny-2
                   do i=2,nx-2
                      p1(i,j,k)=denoms(i,j,k)*((e(i,j,k)*p1(i+1,j,k)+f(i,j,k)*p1(i-1,j,k)) &
                                              +(g(i,j,k)*p1(i,j+1,k)+h(i,j,k)*p1(i,j-1,k)) &
                                              +(m(i,j,k)*p1(i,j,k+1)+n(i,j,k)*p1(i,j,k-1)) &
                                              -r(i,j,k))  + one_minus_omegarelax*p1(i,j,k) 
                   enddo
                enddo
             enddo
! implementing the 'k' boundary condition, AAG & IS 07/03/06
             p1(1:nx-1,1:ny-1,1)=p1(1:nx-1,1:ny-1,2)
! calculating residual, AAG 07/03/06
             abse=sum(abs(p1(1:nx-1,1:ny-1,1:nz-1)-p2(1:nx-1,1:ny-1,1:nz-1)))*one_over_nxnynz
! MAN 09/26/2006 added residual reduction check instead of max error            
             if(iter .eq. 1)then
                eps = abse*res_red_factor
             endif
! checking convergence, AAG 07/03/06
             if (abse<eps .or. abse .le. 1.e-9 .or. iter .eq. itermax) exit
             !if(iter .eq. itermax) exit
          enddo
! MAN 06/04/2007 keeps the correct number of iterations for cases where iter equals itermax
          !print*, '# Iterations = ',iter
          !print*, 'Average Residual = ',abse
          p2(1:nx-1,1:ny-1,1:nz-1)=p1(1:nx-1,1:ny-1,1:nz-1)
          sor_iter = iter
          return
       end
