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
      program main
!		subroutine main
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! QWIC-URB is a multi-building flow solver using 3 dimensional 	
! successive overrelaxation method solver and is based on the	
! work of Rockle (1990) and Kaplan and Dinar (1996).		
!								
! p1 and p2 are cell centered quantites	and called Lagrange	
! Multipliers.							
! u,v,w and uo,vo,wo are cell face quantites			
! uo,vo, and wo are the inital non-mass conserved velocities	
! u,v,w is the final mass conserved velocity field	
! 								
! bcsetup.f does much of the work in the code, here the ground	
! and buildings are defined, the empircal flow parameterizations
! input and the boundary conditions for the SOR solver set up.	
!								
!  - the final velocity field is written out in euler.f	
!  - the Lagrangde multiplier field is writtein out in sor3d.f	
!								
! UPDATES							
! Eric Pardyjak							
! QWIC-URBv1.00Beta September 2002			
!  September 2002 Update:				
!  This version has an updated input file. The height of zref	
!  for the power inlet velocity profile has been added.		
! QWIC-URBv1.00 October 2002				
! This version has multiple updates that include		
!  1. new output format of output files & new output files	
!  2. input.dat has a new line for a rooftop recirculation 	
!  3. new coordinate system					
!  4. fixes in array out of bounds in sor3d.f			
!
! QWIC-URBv1.00f90 January 2003
!	1. Winds out of the North and South can be handle (ERP 12/17)
!	2. A bug in the street canyon routine was fixed (MDW 1/8)
!	3. Allocatable arrays are now deallocated allowing the qwicurb
!        to be run multiple time in the GUI. (TWH 1/8)
!
! Note if this version of the code is being used with a Matlab GUI
! and Visual Compaq Fortran Compiler 6.6, it will be necessary to do the
! following:
!	1. Download the 6.6B update (VFRUN66BI.exe). It can be found at 
!      the following location: 
!			http://h18009.www1.hp.com/fortran/visual/redist.html
!	2. Once patch has been installed move the file dformd.dll from
!	   the C:\Windows\system32 on XP machines or 
!		   C:\WINNT\system32 on Windows2000 machines folder to 
!		   C:\MATLAB6p5\bin\win32
!	3. Be sure to run mex -setup so that the new df66opts.bat is
!      updated.
!
! erp 1/29/04
! This verson of QUICURB has Petra Kastner-Klein's street canyon
! models available. See subroutine init.f90 and bcsetup.f90
!
! ERP 10/04/04
! This version of QUICURB allows for multiple runs to simulate
! "quasi-steady" time dependent flows
! ERP 3/8/05
! - This version of the code contains the new building sorting logic to sort
! building from small to tall so that the algorithms are applied in that order
! - THis version also uses a local velocity to determine the orientation of a
! street canyon vortex
! ERP 6/17/05
! This version of QUICURB has the basic data assimilation algorithms based
! on Barnes objective mapping as implemented by Tom Booth
! new subroutines include: sensorinit
!
!The following file unit numbers are currently in use in QUIC4.0 
! 8/8/2005 ERP
! File numbering usage:
! unit	name				open location	close location
! 28		uofield.dat			outfile.f90			main.f90
! 33		QU_celltype.dat		bcsetup.f90			main.f90
! 34		QU_velocity.dat		outfile.f90			main.f90
! 35		QU_simparams.inp	init.f90			init.f90
! 36		QU_metparams.inp	init.f90			main.f90
! 37		QU_buildings.inp	init.f90			init.f90
! 38		QU_velocity.bin		outfile.f90			main.f90
! 39		QU_celltype.bin		bcsetup.f90			main.f90
! 44		QU_screenout.dat	init.f90			main.f90
! 46		QP_buildout.inp		init.f90			main.f90
! 47		QU_fileoptions.inp	init.f90			init.f90
! 52		f_name				sensorinit.f90		sensorinit.f90	f_name is character string variable
! 55		QU_veg.inp			sensorinit.f90		sensorinit.f90
! 60		uoutmatu.dat		outfile.f90
! 61		
! 62
! 74        QU_intersect.dat	street_intersect.f90
! 75        QU_icellflagtest.datstreet_intersect.f90
!
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         use datamodule ! make data from module "datamodule" visible
         implicit none

!         real t1

!	start timing sequence
!	t1=secnds(0.0)

				 ! ADL - Moved to init.f90 - main isn't called for C++ interface
				 ! 7/1/2009
         !alpha1=1.	!correlation of wind components term
         !alpha2=1.		!correlation of wind components term
         !eta=(alpha1/alpha2)**2	!ratio of gaussian precision moduli

         ! since omega is also used for an angle in bcsetup

         !omegarelax=1.78		!acceleration term in SOR

!   iteration parameters
!         itermax=3000		!max number of iterations
!         eps=1.e-8		!max error

! read input file
         call init
! sort buildings by height small to tall	
         call sort
         if(diffusion_flag == 1)then
         	  itermax=itermax/diffstep
         endif
!         time=start_time
!     begin multirun pseudo time loop
         do i_time = 1,num_time_steps
!     read met input file for each time step	
            call sensorinit
            
! ADL - Moved to sensorinit so that the max_velmag actually gets set.
! With C++/Fort main isn't actually called.
! 7/1/2009
!            max_velmag=0.
!            do i=1,nx
!               do j=1,ny
!                  max_velmag=max(max_velmag,sqrt((uo(i,j,nz-1)**2.)+(vo(i,j,nz-1)**2.)))
!               enddo
!            enddo
!     call boundary condition set up routine
            call building_parameterizations
! MAN 06/04/2007 check to see if there are buildings, vegetation, or multiple profiles
            if(inumbuild .eq. 0 .and. num_sites .eq. 1 .and. inumcanopy .eq. 0)then
               diffusion_flag=0
               itermax=1
            endif
! end MAN 06/04/2007
            if(diffusion_flag == 1)then
               do diffiter= 1,diffstep
			            call divergence
! call denominators
			            call denominators;
!	call sor3d routine
                  call sor3d
!	call Euler equations to get new updated velocities
                  call euler
!   call Diffusion operator
			            call diffusion
		           enddo
		        else
! 	call divergence routine to calculate divergence of uo field
               call divergence
! call denominators
	             call denominators;               
!	call sor3d routine
               call sor3d

!	call Euler equations to get new updated velocities
!	note that Euler call outfile.f
               call euler
            endif
            call outfile
!            time=time + time_incr*i_time
         enddo

! deallocate allocatable arrays - f90 specific
         deallocate(z,zm,dz_array) !MAN 7/21/2008 stretched grid
         deallocate(icellflag,ibldflag, bldnum, bldtype)         ! twh - added this line
         deallocate(uo, vo, wo)                         ! twh - added this line
         deallocate(p1, p2, r)                          ! twh - added this line
         deallocate(e, f, g, h, m, n, o, p, q, denoms, u, v, w) ! twh - added this line
         deallocate(Ht, Wti, Lti, aa, bb)               ! erp 1/31/2003
         deallocate(xfo, yfo, zfo, gamma)                      ! twh - added this line
         if(inumcanopy .gt. 0)then
            deallocate(cXfo,cYfo,cZfo,cgroup,cH,cL,cW,cgamma)
            deallocate(canopy_ktop,canopy_top,canopy_atten)
            deallocate(canopy_zo,canopy_ustar,canopy_d)
         endif
         !MAN 8/30/2005 stacked building fix
         deallocate(zfo_actual)
         if(diffusion_flag .gt. 0)deallocate(Fxd,Fyd,Fzd,visc)
         deallocate(u_prof,v_prof)						!TMB 2/25/05
         deallocate(uoint,voint)							!TMB 2/25/05
         deallocate(wm,wms)									!TMB 2/25/05
!ERP 8/17/05
         deallocate(group_id)

!erp 6/08/04 deallocate new variables
         deallocate(uo_bl,vo_bl)
	
! erp 2/03/04 deallocate PKKs new variables
         deallocate(Havx,Havy)     !average building height of a street can.
         deallocate(Hlowx,Hlowy)     !lower building height of a street can.
         deallocate(Hdifx,Hdify)    !lower building height of a street can.
         deallocate(kref)    !reference level
         deallocate(kcantop)    ! level of can. top
         
         deallocate(time)

         if(uofield_flag.eq.1) close(28)
         if(format_flag.eq.1 .or. format_flag.eq.3)close(33) !erp 3/02/05
         if(format_flag.eq.1 .or. format_flag.eq.3)close(34) !erp 3/02/05
         if(format_flag.eq.2 .or. format_flag.eq.3)close(38) !erp 3/02/05
         if(format_flag.eq.2 .or. format_flag.eq.3)close(39) !erp 3/02/05
!	close(28)		!close uofield.dat

         close(36)		!close QU_metparams.inp
         close(44)		!close QU_screenout.dat
         close(46)		!close QP_buildout.inp
         if(staggered_flag .eq. 1)close(99)
         if(inumcanopy .gt. 0)close(100)
         !close(48)		!close QP_buildorder.inp

         !TMB 3/11/05 I open/close these files in sensorinit, but I want to have a complete list of all
         !the files opened/closed listed here so there would be no confusion later on

         !close(51)		!close uosensorfield.dat   this is now done in sensorinit
         !close(52)		!close the individual profile files with this is also done in sensorinit
	
         !TMB end

!	end timing sequence
!	print*,secnds(t1)
         stop
!	return
      end
