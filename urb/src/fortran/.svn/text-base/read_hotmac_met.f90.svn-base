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
      subroutine read_hotmac_met

         use datamodule ! make data from module "datamodule" visible
         implicit none
         integer jj,kk,tt,t_idx_hotmac,skip_idx_hotmac
         character*128 f_name    !filename for a data profile at one of the sites.
         integer nx_hotmac,ny_hotmac,nz_hotmac
         integer nx_hotmacp1,ny_hotmacp1,nz_hotmacp1,nzs_hotmac
         real dx_hotmac,dy_hotmac
         real utmx_hotmac,utmy_hotmac,rad2deg
         real dumx,dumy,dumu,dumv,costheta,sintheta
         real dist1,dist2,dcx,dcy,gmt,delgmt,gmtday,hotmac_time
         real, allocatable :: x_hotmac(:,:),y_hotmac(:,:),z_hotmac(:,:,:)
         real, allocatable :: u_hotmac(:,:,:),v_hotmac(:,:,:)
         real, allocatable :: zo_hotmac(:,:,:),ws_hotmac(:,:,:,:),wd_hotmac(:,:,:,:)
         integer, allocatable :: x_idx(:),y_idx(:)
         character*1 adum
         real, allocatable :: zgrnd(:,:),ztop(:,:),ustar_hotmac(:,:)
         real, allocatable :: hmz(:),hmzm(:)
         
         rad2deg=180./pi
         
         read(36,*) !Description line
         read(36,*)f_name				!File name of the individual file
         read(36,*)skip_idx_hotmac  !Number of HOTMAC time steps to skip
         !Reading each profile/sensor file
         open(unit=52,file=f_name,form='unformatted',status='old')
         rewind(52)
         do i=1,10
            read(52)adum
         enddo
         read(52) gmt,delgmt,gmtday,hotmac_time,utmx_hotmac,utmy_hotmac,&
            nx_hotmac,ny_hotmac,nz_hotmac,nzs_hotmac,dx_hotmac,dy_hotmac
         nx_hotmacp1=nx_hotmac+1
         ny_hotmacp1=ny_hotmac+1
         nz_hotmacp1=nz_hotmac+1
!         print*,'HOTMAC UTMX =',utmx_hotmac*1000,utmx_hotmac*1000+nx_hotmac*dx_hotmac
!         print*,'HOTMAC UTMY =',utmy_hotmac*1000,utmy_hotmac*1000+ny_hotmac*dy_hotmac
!         print*,'QUIC UTMX =',utmx,utmx+nx*dx
!         print*,'QUIC UTMY =',utmy,utmy+ny*dy
         allocate(hmz(nz_hotmacp1),hmzm(nz_hotmacp1))
         allocate(zgrnd(nx_hotmacp1,ny_hotmacp1),ztop(nx_hotmacp1,ny_hotmacp1))
         allocate(ustar_hotmac(nx_hotmacp1,ny_hotmacp1))
         allocate(u_hotmac(nx_hotmacp1,ny_hotmacp1,nz_hotmacp1),&
                  v_hotmac(nx_hotmacp1,ny_hotmacp1,nz_hotmacp1))
         allocate(x_hotmac(nx_hotmac,ny_hotmac),&
                  y_hotmac(nx_hotmac,ny_hotmac),&
                  z_hotmac(nx_hotmac,ny_hotmac,nz_hotmac))
         allocate(zo_hotmac(nx_hotmac,ny_hotmac,num_time_steps),&
                  ws_hotmac(nx_hotmac,ny_hotmac,nz_hotmac,num_time_steps),&
                  wd_hotmac(nx_hotmac,ny_hotmac,nz_hotmac,num_time_steps))
         zo_hotmac(:,:,:)=0.
         u_hotmac(:,:,:)=0.
         v_hotmac(:,:,:)=0.
         ws_hotmac(:,:,:,:)=0.
         wd_hotmac(:,:,:,:)=0.
         x_hotmac(:,:)=0.
         y_hotmac(:,:)=0.
         z_hotmac(:,:,:)=0.
         rewind(52)
         read(52) (hmz(k),k=1,nz_hotmacp1)
         read(52) (hmzm(k),k=1,nz_hotmacp1)
         read(52) adum !(zsoil(k),k=1,nzs_hotmac) !
         read(52) adum !(zmsoil(k),k=1,nzs_hotmac)
         read(52) adum !((iwater(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
         read(52) adum !((ftree(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
         read(52) ((zgrnd(i,j),i=1,nx_hotmacp1),j=1,ny_hotmacp1)
         read(52) adum !((dzgdx(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
         read(52) adum !((dzgdy(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
         read(52) ((ztop(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
         do k=2,nz_hotmacp1
            do j=1,ny_hotmac
               do i=1,nx_hotmac
                  z_hotmac(i,j,k-1)=(ztop(i,j)-zgrnd(i,j))*hmzm(k)/hmz(nz_hotmacp1)
               enddo
            enddo
         enddo
         costheta=cos(-domain_rotation*pi/180.)
         sintheta=sin(-domain_rotation*pi/180.)
         do j=1,ny_hotmac
            do i=1,nx_hotmac
               x_hotmac(i,j)=(real(i-1)+0.5)*dx_hotmac+utmx_hotmac*1000.-utmx
               y_hotmac(i,j)=(real(j-1)+0.5)*dy_hotmac+utmy_hotmac*1000.-utmy
               dumx=costheta*x_hotmac(i,j)+sintheta*y_hotmac(i,j)
               dumy=-sintheta*x_hotmac(i,j)+costheta*y_hotmac(i,j)
               x_hotmac(i,j)=dumx
               y_hotmac(i,j)=dumy
            enddo
         enddo
         t_idx_hotmac=0
         tt=0
         do while(tt .lt. num_time_steps)
            t_idx_hotmac=t_idx_hotmac+1
            if(t_idx_hotmac .gt. skip_idx_hotmac)then
               read(52,end=41111) adum !gmt,delgmt,gmtday
               read(52) ((ustar_hotmac(i,j),i=1,nx_hotmacp1),j=1,ny_hotmacp1)
               read(52) adum !((theta_star(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((qstar(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((bowen(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((solar(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((shortw(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((uplongw(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((dlongw(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((sensib(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((latent(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((soilfl(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((dfxtree(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((ufxtree(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((wopt(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((preg(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) (((u_hotmac(i,j,k),i=1,nx_hotmacp1),j=1,ny_hotmacp1),k=1,nz_hotmacp1)
               read(52) (((v_hotmac(i,j,k),i=1,nx_hotmacp1),j=1,ny_hotmacp1),k=1,nz_hotmacp1)
               tt=tt+1
               do k=2,nz_hotmacp1
                  do j=1,ny_hotmac
                     do i=1,nx_hotmac
                        dumu=(0.5*(u_hotmac(i,j,k)+u_hotmac(i+1,j,k)))
                        dumv=(0.5*(v_hotmac(i,j,k)+v_hotmac(i,j+1,k)))
                        ws_hotmac(i,j,k-1,tt)=sqrt((dumu**2.)+(dumv**2.))
                        wd_hotmac(i,j,k-1,tt)=270.-rad2deg*atan2(dumv,dumu)
                     enddo
                  enddo
               enddo
               do j=1,ny_hotmac
                  do i=1,nx_hotmac
                     zo_hotmac(i,j,tt)=z_hotmac(i,j,1)*exp(-0.4*ws_hotmac(i,j,1,tt)/ustar_hotmac(i,j))
                     if(zo_hotmac(i,j,tt) .lt. 1e-5)then
                        zo_hotmac(i,j,tt)=1e-5
                     elseif(zo_hotmac(i,j,tt) .gt. 0.45*dz )then
                        zo_hotmac(i,j,tt)=0.45*dz
                     endif
                  enddo
               enddo
            else
               read(52,end=41111) adum !gmt,delgmt,gmtday
               read(52) adum !((ustar_hotmac(i,j),i=1,nx_hotmacp1),j=1,ny_hotmacp1)
               read(52) adum !((theta_star(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((qstar(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((bowen(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((solar(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((shortw(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((uplongw(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((dlongw(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((sensib(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((latent(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((soilfl(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((dfxtree(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((ufxtree(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((wopt(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !((preg(i,j),i=1,nx_hotmac),j=1,ny_hotmac)
               read(52) adum !(((u_hotmac(i,j,k),i=1,nx_hotmacp1),j=1,ny_hotmacp1),k=1,nz_hotmacp1)
               read(52) adum !(((v_hotmac(i,j,k),i=1,nx_hotmacp1),j=1,ny_hotmacp1),k=1,nz_hotmacp1)
            endif
            read(52) adum !(((w_hotmac(i,j,k),i=1,nx_hotmac),j=1,ny_hotmac),k=1,nz_hotmac)
            read(52) adum !(((thet(i,j,k),i=1,nx_hotmac),j=1,ny_hotmac),k=1,nz_hotmac)
            read(52) adum !(((wvap(i,j,k),i=1,nx_hotmac),j=1,ny_hotmac),k=1,nz_hotmac)
            read(52) adum !(((qsq(i,j,k),i=1,nx_hotmac),j=1,ny_hotmac),k=1,nz_hotmac)
            read(52) adum !(((q2l(i,j,k),i=1,nx_hotmac),j=1,ny_hotmac),k=1,nz_hotmac)
            read(52) adum !(((tsoil(i,j,k),i=1,nx_hotmac),j=1,ny_hotmac),k=1,ksmax)
            read(52) adum !(((avthet(i,j,k),i=1,nx_hotmac),j=1,ny_hotmac),k=1,nz_hotmac)
            read(52) adum !(((qliq(i,j,k),i=1,nx_hotmac),j=1,ny_hotmac),k=1,nz_hotmac)
            read(52) adum !(((eddy(i,j,k),i=1,nx_hotmac),j=1,ny_hotmac),k=1,nz_hotmac)
            read(52) adum !(((black(i,j,k),i=1,nx_hotmac),j=1,ny_hotmac),k=1,nz_hotmac)
            read(52) adum !(((edkh(i,j,k),i=1,nx_hotmac),j=1,ny_hotmac),k=1,nz_hotmac)
            read(52) adum !(((hedyx(i,j,k),i=1,nx_hotmac),j=1,ny_hotmac),k=1,nz_hotmac)
            read(52) adum !(((hedyy(i,j,k),i=1,nx_hotmac),j=1,ny_hotmac),k=1,nz_hotmac)
            read(52) adum !(((ratio(i,j,k),i=1,nx_hotmac),j=1,ny_hotmac),k=1,nz_hotmac)
            read(52) adum !(((radcl(i,j,k),i=1,nx_hotmac),j=1,ny_hotmac),k=1,nz_hotmac)
         enddo
41111    continue
         close(52)	!closing the profile/sensor data file
         if(tt .eq. 0)then
            print*,'Error: too many HOTMAC time steps were skipped.  The HOTMAC simulation only had',&
               t_idx_hotmac,'time steps.'
         elseif(tt .lt. num_time_steps)then
            print*,'Error: there were an insufficient number of HOTMAC time steps to yield all',num_time_steps,&
               'QUIC-URB time steps.  The last HOTMAC time step will be used for all subsequent QUIC-URB time steps.'
            t_idx_hotmac=tt
            do tt=t_idx_hotmac+1,num_time_steps
               ws_hotmac(:,:,:,tt)=ws_hotmac(:,:,:,t_idx_hotmac)
               wd_hotmac(:,:,:,tt)=wd_hotmac(:,:,:,t_idx_hotmac)
               zo_hotmac(:,:,tt)=zo_hotmac(:,:,t_idx_hotmac)
            enddo
         endif
         num_sites=0
         allocate(x_idx(nx_hotmac*ny_hotmac),y_idx(nx_hotmac*ny_hotmac))
         do jj=1,ny_hotmac
            do ii=1,nx_hotmac
               if(x_hotmac(ii,jj) .ge. 0. .and. x_hotmac(ii,jj) .le. (nx-1)*dx .and.&
                     y_hotmac(ii,jj) .ge. 0. .and. y_hotmac(ii,jj) .le. (ny-1)*dy)then
                  num_sites=num_sites+1
                  x_idx(num_sites)=ii
                  y_idx(num_sites)=jj
               endif
            enddo
         enddo
         if(num_sites .eq. 0)then
            num_sites=1
            dist1=0.5*((x_hotmac(nx_hotmac,1)-x_hotmac(1,1))+(x_hotmac(1,ny_hotmac)-x_hotmac(1,1)))
            dcx=0.5*dx*(nx-1)
            dcy=0.5*dy*(ny-1)
            do jj=1,ny_hotmac
               do ii=1,nx_hotmac
                  dist2=sqrt(((x_hotmac(ii,jj)-dcx)**2.)+((y_hotmac(ii,jj)-dcy)**2.))
                  if(dist2 .le. dist1)then
                     dist1=dist2
                     x_idx(num_sites)=ii
                     y_idx(num_sites)=jj
                  endif
               enddo
            enddo
            x_hotmac(x_idx(1),y_idx(1))=dcx
            y_hotmac(x_idx(1),y_idx(1))=dcy
         endif
         num_vert_points=nz_hotmac
         !allocate arrays
         allocate(site_blayer_flag(num_sites,num_time_steps))
         allocate(site_pp(num_sites,num_time_steps))
         allocate(site_H(num_sites,num_time_steps))
         allocate(site_rL(num_sites,num_time_steps))
         allocate(site_ac(num_sites,num_time_steps))
         allocate(site_xcoord(num_sites))
         allocate(site_ycoord(num_sites))
         allocate(u_prof(num_sites,nz),v_prof(num_sites,nz))
         allocate(uoint(num_sites),voint(num_sites))
         allocate(wm(num_sites,nx,ny),wms(num_sites,nx,ny))
         allocate(site_nz_data(num_sites,num_time_steps))
         allocate(site_z_data(num_sites,num_time_steps,num_vert_points))
         allocate(site_ws_data(num_sites,num_time_steps,num_vert_points))
         allocate(site_wd_data(num_sites,num_time_steps,num_vert_points))
         site_blayer_flag(:,:)=4
         u_prof(1:num_sites,1:nz) = 0
         v_prof(1:num_sites,1:nz) = 0
         site_xcoord(1:num_sites) = 0
         site_ycoord(1:num_sites) = 0
         site_nz_data(:,:)=nz_hotmac

         do kk=1,num_sites
            site_xcoord(kk)=x_hotmac(x_idx(kk),y_idx(kk)) !x coordinate of site location (meters)
            site_ycoord(kk)=y_hotmac(x_idx(kk),y_idx(kk)) !y coordinate of site location (meters)
            do tt=1,num_time_steps
               site_pp(kk,tt)=min(zo_hotmac(x_idx(kk),y_idx(kk),tt),0.45*dz)		!if blayer = 2 site_pp = exp else site_pp = zo
               do ii=1,site_nz_data(kk,tt)
                  site_z_data(kk,tt,ii)=z_hotmac(x_idx(kk),y_idx(kk),ii)
                  site_ws_data(kk,tt,ii)=ws_hotmac(x_idx(kk),y_idx(kk),ii,tt)
! MAN 02/05/2007 Domain Rotation
                  site_wd_data(kk,tt,ii)=wd_hotmac(x_idx(kk),y_idx(kk),ii,tt)-domain_rotation
                  if(site_wd_data(kk,tt,ii).lt. 0.)then
                       site_wd_data(kk,tt,ii)=site_wd_data(kk,tt,ii)+360.
                  elseif(site_wd_data(kk,tt,ii).ge. 360.)then
                       site_wd_data(kk,tt,ii)=site_wd_data(kk,tt,ii)-360.
                  endif
! end MAN 02/05/2007
               enddo
            enddo
         enddo    !kk = num_sites
         deallocate(hmz,hmzm,ustar_hotmac,zgrnd,ztop)
         deallocate(x_hotmac,y_hotmac,z_hotmac)
         deallocate(zo_hotmac,ws_hotmac,wd_hotmac)
         deallocate(x_idx,y_idx)
         return
      end
