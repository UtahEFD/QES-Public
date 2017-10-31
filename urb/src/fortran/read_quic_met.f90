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
      subroutine read_quic_met

         use datamodule ! make data from module "datamodule" visible
! operations  done a priori to speed up the code AAG & IS  07/03/06
         implicit none
         integer kk,tt
         character*128 f_name    !filename for a data profile at one of the sites.
         
         !blayer_flag has already been read in
         read(36,*)num_sites			!total number of sites
         read(36,*)num_vert_points
         !allocate arrays
         allocate(site_blayer_flag(num_sites,num_time_steps))
         allocate(site_pp(num_sites,num_time_steps))
         allocate(site_H(num_sites,num_time_steps))
         allocate(site_rL(num_sites,num_time_steps))
         allocate(site_ac(num_sites,num_time_steps))
         allocate(site_xcoord(num_sites))
         allocate(site_ycoord(num_sites))
!         allocate(site_zcoord(num_sites))
!         allocate(t_site(num_sites,num_time_steps))
!         allocate(dir_site(num_sites,num_time_steps))
!         allocate(vel_site(num_sites,num_time_steps))
         allocate(u_prof(num_sites,nz),v_prof(num_sites,nz))
         allocate(uoint(num_sites),voint(num_sites))
         allocate(wm(num_sites,nx,ny),wms(num_sites,nx,ny))
         allocate(site_nz_data(num_sites,num_time_steps))
         !MAN 4/5/2007 number of data points to allocate in data points profiles
         allocate(site_z_data(num_sites,num_time_steps,num_vert_points))
         allocate(site_ws_data(num_sites,num_time_steps,num_vert_points))
         allocate(site_wd_data(num_sites,num_time_steps,num_vert_points))
         
!         allocate(site_ustar(num_sites),site_ac(num_sites))
!         allocate(site_lc(num_sites),site_d(num_sites))


         !initialize velocity profiles at each site
         u_prof(1:num_sites,1:nz) = 0
         v_prof(1:num_sites,1:nz) = 0


         !initialize the vectors
         site_xcoord(1:num_sites) = 0
         site_ycoord(1:num_sites) = 0
         site_nz_data(:,:)=1
!         site_zcoord(1:num_sites) = 0
!         t_site(1:num_sites,1:num_time_steps) = -99
!         dir_site(1:num_sites,1:num_time_steps) = 0
!         vel_site(1:num_sites,1:num_time_steps) = 0

lp001:   do kk=1,num_sites
            read(36,*)						!The site name
            read(36,*)						!Description line
            read(36,*)f_name				!File name of the individual file
            !Reading each profile/sensor file
            open(unit=52,file=f_name,status='old')
            read(52,*)						!The site name
            read(52,*)site_xcoord(kk)		!x coordinate of site location (meters)
            read(52,*)site_ycoord(kk)		!y coordinate of site location (meters)
            do tt=1,num_time_steps
               read(52,*) !time stamp
               read(52,*)site_blayer_flag(kk,tt) !boundary layer flag for each site (1=log,2=exp,3=canopy,4=data)e
	            read(52,*)site_pp(kk,tt)			!if blayer = 2 site_pp = exp else site_pp = zo
               select case(site_blayer_flag(kk,tt))
                  case(1)!logarithmic profile
                     read(52,*)site_rL(kk,tt)			!reciprocal Monin-Obukhov length
                  case(3)!urban canopy
                     read(52,*)site_rL(kk,tt)			!reciprocal Monin-Obukhov length
                     read(52,*)site_H(kk,tt)			   !canopy height
                     read(52,*)site_ac(kk,tt)			!atenuation coefficient
                  case(4)!data points
                     read(52,*)site_nz_data(kk,tt)		!number of data points in vertical wind profile
               endselect
               read(52,*)!skip line			!"height  direction   magnitude"  Label
               do ii=1,site_nz_data(kk,tt)
                  read(52,*)site_z_data(kk,tt,ii),site_ws_data(kk,tt,ii),site_wd_data(kk,tt,ii)
! MAN 02/05/2007 Domain Rotation
                  site_wd_data(kk,tt,ii)=site_wd_data(kk,tt,ii)-domain_rotation
                  if(site_wd_data(kk,tt,ii).lt. 0.)then
                       site_wd_data(kk,tt,ii)=site_wd_data(kk,tt,ii)+360.
                  elseif(site_wd_data(kk,tt,ii).ge. 360.)then
                       site_wd_data(kk,tt,ii)=site_wd_data(kk,tt,ii)-360.
                  endif
! end MAN 02/05/2007
               enddo
            enddo
            close(52)	!closing the profile/sensor data file
         enddo   lp001       !kk = num_sites
! MAN 10/10/2007 if there is only one measurement make sure it is in the domain
         if(num_sites .eq. 1)then
            site_xcoord(1)=0.5*real(nx-1)*dx
            site_ycoord(1)=0.5*real(ny-1)*dy
         endif
! end MAN 10/10/2007
         return
      end
