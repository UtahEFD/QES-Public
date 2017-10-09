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
       subroutine diffusion
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! subroutine to calculate diffusive flux and update the velocity field with this flux  
! subroutine called by main.f90
! subroutine calls outfile.f90 to print out results   
! subroutine calls turbulence_model.f90 to calculate the turbulent viscosity        
! Akshay A. Gowardhan                  
! August 2006                    
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
          use datamodule ! make data from module "datamodule" visible
          implicit none
          integer ip,im,jp,jm,kp,km
          real dxi, dyi, dzi !,vxp,vxm,vyp,vym,vzp,vzm
          real Tuuip, Tuuim, Tuvjp,Tuvjm, Tuwkp,Tuwkm
          real Tvuip,Tvuim, Tvvjp, Tvvjm, Tvwkp, Tvwkm
          real Twuip, Twuim, Twvjp, Twvjm, Twwkp, Twwkm, mzp, mz, mzm
          dxi=1./dx
          dyi=1./dy
          do k=1,nz
             do j=1,ny
                do i=1,nx
                   uo(i,j,k)=u(i,j,k)
                   vo(i,j,k)=v(i,j,k)
                   wo(i,j,k)=w(i,j,k)
                enddo
             enddo
          enddo
          call turbulence_model
          do k=2,nz-1
             ! MAN 07/25/2008 stretched vertical grid
             mzp= dz_array(k+1)/2.
		     mz = dz_array(k)/2.
		     mzm= dz_array(k-1)/2.
             do j=2,ny-1
                do i=2,nx-1
                   ip = i + 1
                   jp = j + 1
                   kp = k + 1
                   im = i - 1
                   jm = j - 1
                   km = k - 1
! X momentum      
                   Tuuip = visc(i,j,k)*(2.*( dxi * (uo(ip,j,k)-uo(i ,j,k) )))
                   Tuuim = visc(i,j,k)*(2.*( dxi * (uo(i ,j,k)-uo(im,j,k) )))
                   Tuvjp = visc(i,j,k)*(dyi*(uo(i,jp,k)-uo(i, j,k))+dxi*(vo(ip,j, k)-vo(i ,j ,k)))
                   Tuvjm = visc(i,j,k)*(dyi*(uo(i,j ,k)-uo(i,jm,k))+dxi*(vo(ip,jm,k)-vo(i ,jm,k)))
                   Tuwkp = visc(i,j,k)*((1./(mzp+mz))*(uo(i,j,kp)-uo(i ,j,k))+dxi*(wo(ip,j, k)-wo(i ,j ,k)))
                   Tuwkm = visc(i,j,k)*((1./(mzm+mz))*(uo(i,j ,k)-uo(i,j,km))+dxi*(wo(ip,j,km)-wo(i ,j,km)))
                   Fxd(i,j,k) = dxi*(Tuuip  - Tuuim )+ dyi*( Tuvjp - Tuvjm )+ (0.5/mz)*( Tuwkp - Tuwkm )
! Y momentum
                   Tvuip = visc(i,j,k)*(dxi*(vo(ip,j,k)-vo(i ,j,k))+dyi*(uo(i,jp ,k)-uo(i ,j,k)))
                   Tvuim = visc(i,j,k)*(dxi*(vo(i ,j,k)-vo(im,j,k))+dyi*(uo(im,jp,k)-uo(im,j,k)))
                   Tvvjp = visc(i,j,k)*(2.*dyi*(vo(i,jp,k)-vo(i,j ,k)))
                   Tvvjm = visc(i,j,k)*(2.*dyi*(vo(i,j ,k)-vo(i,jm,k)))
                   Tvwkp = visc(i,j,k)*((1./(mzp+mz))*(vo(i,j,kp)-vo(i, j,k))+dyi*(wo(i,jp,k )-wo(i ,j,k)))
                   Tvwkm = visc(i,j,k)*((1./(mzm+mz))*(vo(i,j,k )-vo(i,j,km))+dyi*(wo(i,jp,km)-wo(i,j,km)))
                   Fyd(i,j,k)= dxi*( Tvuip - Tvuim)+  dyi*(  Tvvjp - Tvvjm)+ (0.5/mz)*( Tvwkp - Tvwkm)
! Z momentum
                   Twuip = visc(i,j,k)*(dxi*(wo(ip,j,k)-wo(i ,j,k))+(1./(mzp+mz))*(uo(i ,j,kp)-uo(i ,j,k)))
                   Twuim = visc(i,j,k)*(dxi*(wo(i ,j,k)-wo(im,j,k))+(1./(mzp+mz))*(uo(im,j,kp)-uo(im,j,k)))
                   Twvjp = visc(i,j,k)*(dyi*(wo(i,jp,k)-wo(i, j,k))+(1./(mzp+mz))*(vo(i ,j,kp)-vo(i ,j,k)))
                   Twvjm = visc(i,j,k)*(dyi*(wo(i,j ,k)-wo(i,jm,k))+(1./(mzp+mz))*(vo(i,jm,kp)-vo(i,jm,k)))
                   Twwkp = visc(i,j,k)*(2*(0.5/mzp)*(wo(i,j,kp)-wo(i,j ,k)))
                   Twwkm = visc(i,j,k)*(2*(0.5/mzm)*(wo(i,j,k )-wo(i,j,km)))
                   Fzd(i,j,k) = dxi*( Twuip  - Twuim)+ dyi*(Twvjp  - Twvjm)+ (1./(mzp+mz))*( Twwkp - Twwkm)
                   enddo
                enddo
             enddo
             dt= 0.25*(min(dx,dy,dz))**2/maxval(visc)
!Update velocity with diffusive fluxes  
             do k=2,nz-1
                do j=2,ny-1
                   do i=2,nx-1
                      uo(i,j,k)= uo(i,j,k)+ (dt*(Fxd(i,j,k)))
                      vo(i,j,k)= vo(i,j,k)+ (dt*(Fyd(i,j,k)))
                      wo(i,j,k)= wo(i,j,k)+ (dt*(Fzd(i,j,k)))
                   enddo
                enddo
             enddo
             return
          end
