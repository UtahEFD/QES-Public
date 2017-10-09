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
      subroutine outfile
!********************************************************
!							
! this subroutine writes out the data files and closes all
! necessary files.
! ASCII files currently written:
!	1. uofield.dat - the initial non-mass consistent velocity field
!	2. uoutmat.dat - the final velocity field (Matlab Format)
!	
! ASCII files that can be uncommented to be written:
!	1. uoutfield.dat - the final velocity field (TECHPLOT format)
!	2. uoutmat.dat - final u-velocity field before cell cent. averaging
!	3. voutmat.dat - final u-velocity field before cell cent. averaging
!	4. woutmat.dat - final u-velocity field before cell cent. averaging
!
! 1/15/04 erp Unformatted binary writes have been added
! ERP
!********************************************************
         use datamodule ! make data from module "datamodule" visible

         implicit none
         real x,y

!--AAG & IS 06/29/06 : 
! Because quicPlume.f90 uses double precision the velocity 
! data is written out in double precision. This is needed to eliminate the error 
! due to reading a single precision binary file into a double precision code.
!
! ***NOTE***: if quicplume.f90 is compiled as "single precision" then you need to use
!
! ----> real(selected_real_kind(p=4)) <----
! end akshay & inanc       

!allocate(x_write(nx-1,ny-1,nz-1),y_write(nx-1,ny-1,nz-1),z_write(nx-1,ny-1,nz-1))

! open the data files for output to MATLAB (unite=34)
! and TECPLOT(unit=27)
!        open(unit=27,file="uoutfield.dat",status="unknown")
! multi-time step output
 912     format(i9,  '       !Time Increment')
 913     format(i9,  '       !Number of time steps')
 914     format(f9.4,'       !Time')
 916     format(3i6,1x,3(f17.5,1x))

         if(i_time.eq.1)then
            if(uofield_flag.eq.1)	& !erp 3/2/05   
                  open(unit=28,file="uofield.dat",status="unknown")
            if(format_flag.eq.1 .or. format_flag.eq.3)	& !erp 3/2/05   
                  open(unit=34,file="QU_velocity.dat",status="unknown")
            if(format_flag.eq.2 .or. format_flag.eq.3) & !erp 3/2/05   
                  open(unit=38,file="QU_velocity.bin",form='unformatted',status="unknown")
            if(staggered_flag .eq. 1) &
                  open(unit=99,file="QU_staggered_velocity.bin",form='unformatted',status="unknown")
            if(inumcanopy .gt. 0) &
                  open(unit=100,file="QU_vegetation.bin",form='unformatted',status="unknown")
         endif
         !MAN 07/28/2008 
         if(inumcanopy .gt. 0)then
            write(100)((canopy_ustar(i,j),i=1,nx-1),j=1,ny-1)
            write(100)(((canopy_atten(i,j,k),i=1,nx-1),j=1,ny-1),k=1,nz-1)
         endif
         if(uofield_flag.eq.1) then !erp 3/02/05
            if(i_time.eq.1)then
               write(28,*)'% Inital velocity field i,j,k,uo,vo,wo'
               write(28,913)num_time_steps
            endif
            write(28,*)'% Begin Output for new time step'
            write(28,912)i_time
            write(28,914)time(i_time)
            do k=1,nz
               do j=1,ny
                  do i=1,nx
                     write(28,916)i,j,k,uo(i,j,k),vo(i,j,k),wo(i,j,k)
                  enddo
               enddo
            enddo
         endif !endif for uofield_flag
	
         if(format_flag.eq.1 .or. format_flag.eq.3)  then
            if(i_time.eq.1)then
               write(34,*)'%matlab velocity output file'
               write(34,913)num_time_steps
            endif
            write(34,*)'% Begin Output for new time step'
            write(34,912)i_time
            write(34,914)time(i_time)
         endif
         if(staggered_flag .eq. 1)then
            write(99)(((u(i,j,k),i=2,nx),j=1,ny-1),k=1,nz-1), &
               	   (((v(i,j,k),i=1,nx-1),j=2,ny),k=1,nz-1), &
               	   (((w(i,j,k),i=1,nx-1),j=1,ny-1),k=2,nz)
         endif
!erp modified to appropriately reflect physical locations
!erp 10/8/2003
!erp 1/18/2005 lines added to write out unformatted for binary read into Matlab
         do k=1,nz-1
            do j=1,ny-1
               do i=1,nx-1
                  u(i,j,k)=(0.5*(u(i,j,k)+u(i+1,j,k)))
                  v(i,j,k)=(0.5*(v(i,j,k)+v(i,j+1,k)))
                  w(i,j,k)=(0.5*(w(i,j,k)+w(i,j,k+1)))
               enddo
            enddo
         enddo
         select case (format_flag)
		      case (1)
			      do k=1,nz-1
                  do j=1,ny-1
                     do i=1,nx-1
                        x=(0.5*(real(i+1)+real(i))-1.)*dx
                        y=(0.5*(real(j+1)+real(j))-1.)*dy
                        write(34,101)x,y,zm(k),u(i,j,k),v(i,j,k),w(i,j,k)
                     enddo
                  enddo
               enddo
			   case (2)
               write(38)(((u(i,j,k),i=1,nx-1),j=1,ny-1),k=1,nz-1),   &
               	      (((v(i,j,k),i=1,nx-1),j=1,ny-1),k=1,nz-1),   &
               	      (((w(i,j,k),i=1,nx-1),j=1,ny-1),k=1,nz-1)
			   case (3)
			      do k=1,nz-1
                  do j=1,ny-1
                     do i=1,nx-1
                        x=(0.5*(real(i+1)+real(i))-1.)*dx
                        y=(0.5*(real(j+1)+real(j))-1.)*dy
                        write(34,101)x,y,zm(k),u(i,j,k),v(i,j,k),w(i,j,k)
                     enddo
                  enddo
               enddo
               write(38)(((u(i,j,k),i=1,nx-1),j=1,ny-1),k=1,nz-1),   &
               	      (((v(i,j,k),i=1,nx-1),j=1,ny-1),k=1,nz-1),   &
               	      (((w(i,j,k),i=1,nx-1),j=1,ny-1),k=1,nz-1)
         end select
 101     format(6(f11.5,1x))
         return
      end
