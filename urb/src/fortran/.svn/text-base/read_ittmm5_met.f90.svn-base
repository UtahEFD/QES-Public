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
      subroutine read_ittmm5_met

         use datamodule ! make data from module "datamodule" visible
! operations  done a priori to speed up the code AAG & IS  07/03/06
         implicit none
         integer jj,kk,tt
         character*128 f_name,dum    !filename for a data profile at one of the sites.
         integer nittx,nitty,nittz
         real ittutmx,ittutmy,ittutmzone,rad2deg
         real dumx,dumy,dumz,dumu,dumv,costheta,sintheta
         real dist1,dist2,dcx,dcy
         real, allocatable :: ittx(:,:,:),itty(:,:,:),ittz(:,:,:)
         real, allocatable :: ittzo(:,:,:),ittws(:,:,:,:),ittwd(:,:,:,:)
         integer, allocatable :: x_idx(:),y_idx(:)
         
         rad2deg=180./pi
         read(36,*) !Description line
         do tt=1,num_time_steps
            read(36,*)f_name				!File name of the individual file
            !Reading each profile/sensor file
            open(unit=52,file=f_name,status='old')
            read(52,*) !description line
            read(52,*)dum !lat long of sw grid cell center
            read(52,*)ittutmx,ittutmy,ittutmzone !utmx of sw grid cell center
            read(52,*)dum !time stamp
            read(52,*)nittx !num cells in east-west direction for MM5 data
            read(52,*)nitty !num cells in north-south direction for MM5 data
            read(52,*)nittz !num cells in vertical direction for MM5 data
            read(52,*) !description line
            read(52,*) !description line
            if(tt .eq. 1)then
               if(ittutmzone .ne. utmzone)then
                  print*,'Project UTM Zone and Met data UTM Zone are not equal'
               endif
               allocate(ittx(nittx,nitty,nittz),&
                        itty(nittx,nitty,nittz),&
                        ittz(nittx,nitty,nittz))
               allocate(ittzo(nittx,nitty,num_time_steps),&
                        ittws(nittx,nitty,nittz,num_time_steps),&
                        ittwd(nittx,nitty,nittz,num_time_steps))
               ittzo(:,:,:)=0.
               ittws(:,:,:,:)=0.
               ittwd(:,:,:,:)=0.
            endif
            ittx(:,:,:)=0.
            itty(:,:,:)=0.
            ittz(:,:,:)=0.
            do ii=1,nittx
               do jj=1,nitty
                  read(52,*)dumx,dumy,dumz,ittzo(ii,jj,tt)
               enddo
            enddo
            read(52,*) !description line
            read(52,*) !description line
            read(52,*) !description line
            do kk=1,nittz
               do jj=1,nitty
                  do ii=1,nittx
                     read(52,*)ittx(ii,jj,kk),itty(ii,jj,kk),ittz(ii,jj,kk),&
                               dumu,dumv
                     ittws(ii,jj,kk,tt)=sqrt((dumu**2.)+(dumv**2.))
                     ittwd(ii,jj,kk,tt)=270.-rad2deg*atan2(dumv,dumu)
                  enddo
               enddo
            enddo
            close(52)	!closing the profile/sensor data file
         enddo
         costheta=cos(-domain_rotation*pi/180.)
         sintheta=sin(-domain_rotation*pi/180.)
         do kk=1,nittz
            do jj=1,nitty
               do ii=1,nittx
                  ittx(ii,jj,kk)=ittx(ii,jj,kk)-utmx
                  itty(ii,jj,kk)=itty(ii,jj,kk)-utmy
                  dumx=costheta*ittx(ii,jj,kk)+sintheta*itty(ii,jj,kk)
                  dumy=-sintheta*ittx(ii,jj,kk)+costheta*itty(ii,jj,kk)
                  ittx(ii,jj,kk)=dumx
                  itty(ii,jj,kk)=dumy
               enddo
            enddo
         enddo
         num_sites=0
         allocate(x_idx(nittx*nitty),y_idx(nittx*nitty))
         do jj=1,nitty
            do ii=1,nittx
               if(ittx(ii,jj,1) .ge. 0. .and. ittx(ii,jj,1) .le. (nx-1)*dx .and.&
                     itty(ii,jj,1) .ge. 0. .and. itty(ii,jj,1) .le. (ny-1)*dy)then
                  num_sites=num_sites+1
                  x_idx(num_sites)=ii
                  y_idx(num_sites)=jj
               endif
            enddo
         enddo
         if(num_sites .eq. 0)then
            num_sites=1
            dist1=0.5*((ittx(nittx,1,1)-ittx(1,1,1))+(ittx(1,nitty,1)-ittx(1,1,1)))
            dcx=0.5*dx*(nx-1)
            dcy=0.5*dy*(ny-1)
            do jj=1,nitty
               do ii=1,nittx
                  dist2=sqrt(((ittx(ii,jj,1)-dcx)**2.)+((itty(ii,jj,1)-dcy)**2.))
                  if(dist2 .le. dist1)then
                     dist1=dist2
                     x_idx(num_sites)=ii
                     y_idx(num_sites)=jj
                  endif
               enddo
            enddo
            ittx(x_idx(1),y_idx(1),1)=dcx
            itty(x_idx(1),y_idx(1),1)=dcy
         endif
         num_vert_points=nittz
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
         site_nz_data(:,:)=nittz

lp001:   do kk=1,num_sites
            site_xcoord(kk)=ittx(x_idx(kk),y_idx(kk),1) !x coordinate of site location (meters)
            site_ycoord(kk)=itty(x_idx(kk),y_idx(kk),1) !y coordinate of site location (meters)
            do tt=1,num_time_steps
               site_pp(kk,tt)=ittzo(x_idx(kk),y_idx(kk),tt)		!if blayer = 2 site_pp = exp else site_pp = zo
               do ii=1,site_nz_data(kk,tt)
                  site_z_data(kk,tt,ii)=ittz(x_idx(kk),y_idx(kk),ii)
                  site_ws_data(kk,tt,ii)=ittws(x_idx(kk),y_idx(kk),ii,tt)
! MAN 02/05/2007 Domain Rotation
                  site_wd_data(kk,tt,ii)=ittwd(x_idx(kk),y_idx(kk),ii,tt)-domain_rotation
                  if(site_wd_data(kk,tt,ii).lt. 0.)then
                       site_wd_data(kk,tt,ii)=site_wd_data(kk,tt,ii)+360.
                  elseif(site_wd_data(kk,tt,ii).ge. 360.)then
                       site_wd_data(kk,tt,ii)=site_wd_data(kk,tt,ii)-360.
                  endif
! end MAN 02/05/2007
               enddo
            enddo
         enddo   lp001       !kk = num_sites
         deallocate(ittx,itty,ittz)
         deallocate(ittzo,ittws,ittwd)
         deallocate(x_idx,y_idx)
         return
      end
