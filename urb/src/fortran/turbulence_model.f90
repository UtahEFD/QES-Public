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
       subroutine turbulence_model
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! subroutine to turbulent viscosity (use smogorinsky model) 
! subroutine called by diffusion.f90      
! Akshay A. Gowardhan                  
! August 2006                    
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
          use datamodule
          implicit none
          real dxi, dyi, dzi, shear, cs_les, delta, mol_visc
          cs_les= 0.2
          mol_visc= 1.8e-6
          delta= (dx*dy*dz)**(1./3.)
          dxi=1./dx
          dyi=1./dy
          visc(:,:,:)=0.
!  Turbulence model (smagorinsky)
          shear=0.0
          do k=2,nz-1
             ! MAN 07/25/2008 stretched vertical grid
             dzi=1./(dz_array(k))
             do j=2,ny-1
                do i=2,nx-1
                   shear= 2.0*( &
                         (dxi*(u(i,j,k)-u(i-1,j,k)))**2+ &
                         (dyi*(v(i,j,k)-v(i,j-1,k)))**2+ &
                         (dzi*(w(i,j,k)-w(i,j,k-1)))**2)
                   shear= shear +0.25*( &
                         ((dyi*(u(i,j+1,k)-u(i,j,k)))+ (dxi*(v(i+1,j,k)-v(i,j,k))))**2+ &
                         ((dyi*(u(i,j,k)-u(i,j-1,k)))+ (dxi*(v(i+1,j-1,k)-v(i,j-1,k))))**2+ &
                         ((dyi*(u(i-1,j+1,k)-u(i-1,j,k)))+ (dxi*(v(i,j,k)-v(i-1,j,k))))**2+ &
                         ((dyi*(u(i-1,j,k)-u(i-1,j-1,k)))+ (dxi*(v(i,j-1,k)-v(i-1,j-1,k))))**2)
                   shear= shear +0.25*( &
                         ((dzi*(u(i,j,k+1)-u(i,j,k)))+ (dxi*(w(i+1,j,k)-w(i,j,k))))**2+ &
                         ((dzi*(u(i,j,k)-u(i,j,k-1)))+ (dxi*(w(i+1,j,k-1)-w(i,j,k-1))))**2+ &
                         ((dzi*(u(i-1,j,k+1)-u(i-1,j,k)))+ (dxi*(w(i,j,k)-w(i-1,j,k))))**2+ &
                         ((dzi*(u(i-1,j,k)-u(i-1,j,k-1)))+ (dxi*(w(i,j,k-1)-w(i-1,j,k-1))))**2)
                   shear= shear +0.25*( &
                         ((dzi*(v(i,j,k+1)-v(i,j,k)))+ (dyi*(w(i,j+1,k)-w(i,j,k))))**2+ &
                         ((dzi*(v(i,j,k)-v(i,j,k-1)))+ (dyi*(w(i,j+1,k-1)-w(i,j,k-1))))**2+ &
                         ((dzi*(v(i,j-1,k+1)-v(i,j-1,k)))+ (dyi*(w(i,j,k)-w(i,j-1,k))))**2+ &
                         ((dzi*(v(i,j-1,k)-v(i,j-1,k-1)))+ (dyi*(w(i,j,k-1)-w(i,j-1,k-1))))**2)
                   visc(i,j,k)= (cs_les*delta)**2* sqrt(abs(shear)) + mol_visc
                enddo
             enddo
          enddo
          visc(1,:,:)=visc(nx-1,:,:)
          visc(nx,:,:)=visc(2,:,:)
          visc(:,:,1)=0.0
          visc(:,:,nz)=visc(:,:,2)
          visc(:,1,:)=visc(:,ny-1,:)
          visc(:,ny,:)=visc(:,2,:)
          do k=2,nz-2
             do j=2,ny-2
                do i=2,nx-2
                   if(icellflag(i+1,j,k)==0 .or. icellflag(i-1,j,k)==0 .or. icellflag(i,j+1,k)==0 .or. icellflag(i,j-1,k)==0 &
                         .or. icellflag(i,j,k+1)==0 .or. icellflag(i,j,k-1)==0 )visc(i,j,k)=0.0
                enddo
             enddo
          enddo
          return
       end