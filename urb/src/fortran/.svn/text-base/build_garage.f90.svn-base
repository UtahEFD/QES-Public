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
      subroutine build_garage
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! This subroutine set the velocities within the parking garage
! ERP/AAG 8/2007
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

         use datamodule ! make data from module "datamodule" visible
         implicit none

         integer inumbuild_temp, Number_stories, i_floors, inumbuild_old
         
         integer, allocatable :: bldnum_pg(:),bldtype_pg(:),group_id_pg(:)
         real, allocatable :: Ht_pg(:),Wti_pg(:),Lti_pg(:),atten_pg(:)
         real, allocatable :: xfo_pg(:),yfo_pg(:),zfo_actual_pg(:),gamma_pg(:)
         real, allocatable :: Weff_pg(:),Leff_pg(:),Lf_pg(:),Lr_pg(:)
         real, allocatable :: Sx_east_pg(:),Sy_north_pg(:),phiprime_pg(:),phi_pg(:),theta_pg(:)

         inumbuild_old = inumbuild
         inumbuild = inumbuild_old + num_new_builds

         allocate(xfo_pg(inumbuild),yfo_pg(inumbuild),zfo_actual_pg(inumbuild))  
         allocate(gamma_pg(inumbuild),Ht_pg(inumbuild),Wti_pg(inumbuild),Lti_pg(inumbuild))
         allocate(bldnum_pg(inumbuild),bldtype_pg(inumbuild),group_id_pg(inumbuild),atten_pg(inumbuild))
         allocate(Weff_pg(inumbuild),Leff_pg(inumbuild),Lf_pg(inumbuild),Lr_pg(inumbuild))
         allocate(Sx_east_pg(inumbuild),Sy_north_pg(inumbuild),phiprime_pg(inumbuild))
         allocate(phi_pg(inumbuild),theta_pg(inumbuild))
         
         bldnum_pg(1:inumbuild_old)  = bldnum(:)
         bldtype_pg(1:inumbuild_old) = bldtype(:)
         group_id_pg(1:inumbuild_old)=group_id(:)
         xfo_pg(1:inumbuild_old)=xfo(:)
         yfo_pg(1:inumbuild_old)=yfo(:)
         zfo_actual_pg(1:inumbuild_old)=zfo_actual(:)
         gamma_pg(1:inumbuild_old)=gamma(:)
         Ht_pg(1:inumbuild_old)=Ht(:)
         Wti_pg(1:inumbuild_old)=Wti(:)
         Lti_pg(1:inumbuild_old)=Lti(:)
         atten_pg(1:inumbuild_old)=atten(:)
         Weff_pg(1:inumbuild_old)=Weff(:)
         Leff_pg(1:inumbuild_old)=Leff(:)
         Lf_pg(1:inumbuild_old)=Lf(:)
!         Lfy_pg(1:inumbuild_old)=Lfy(:)
         Lr_pg(1:inumbuild_old)=Lr(:)
         Sx_east_pg(1:inumbuild_old) = Sx_east(:)
         Sy_north_pg(1:inumbuild_old) = Sy_north(:)
         phiprime_pg(1:inumbuild_old)=phiprime(:)
         phi_pg(1:inumbuild_old)=phi(:)
         theta_pg(1:inumbuild_old)=theta(:)
         deallocate(xfo,yfo,zfo_actual,gamma,Ht,Wti,Lti,bldnum,bldtype,group_id,atten,phiprime)
         deallocate(Sx_east,Sy_north,Lf,Lr,Weff,Leff,phi,theta)

         inumbuild_temp=inumbuild_old

         do ibuild=1,inumbuild_old
            if(bldtype_pg(ibuild) .eq. 10)then
               Number_stories = atten_pg(ibuild)
               do i_floors = 1,Number_stories
                  inumbuild_temp = inumbuild_temp + 1
                  k=kstart(ibuild)-1+2*i_floors
                  bldnum_pg(inumbuild_temp)= inumbuild_temp
                  group_id_pg(inumbuild_temp) = group_id_pg(ibuild)
                  bldtype_pg(inumbuild_temp) = 1
                  Wti_pg(inumbuild_temp)=Wti_pg(ibuild)
                  Lti_pg(inumbuild_temp)=Lti_pg(ibuild)
                  gamma_pg(inumbuild_temp)=gamma_pg(ibuild)
                  atten_pg(inumbuild_temp)=0
                  xfo_pg(inumbuild_temp)=xfo_pg(ibuild)
                  yfo_pg(inumbuild_temp)=yfo_pg(ibuild)
                  ! MAN 07/25/2008 stretched vertical grid
                  zfo_actual_pg(inumbuild_temp) = z(k-1)
                  Ht_pg(inumbuild_temp) = z(k)
               
                  phi_pg(inumbuild_temp) = phi_pg(ibuild)
                  theta_pg(inumbuild_temp) = theta_pg(ibuild)
                  Weff_pg(inumbuild_temp)=Weff_pg(ibuild)
                  Leff_pg(inumbuild_temp)= Leff_Pg(ibuild)
               
                  Lf_pg(inumbuild_temp)=0.
!                  Lfy_pg(inumbuild_temp)=0.
                  Lr_pg(inumbuild_temp)=0.
                  Sx_east_pg(inumbuild_temp)=9999.
                  Sy_north_pg(inumbuild_temp)=9999.
!                  kend_pg=nint(Ht_pg(inumbuild_temp)/dz)+1
!                  do k=nint(zfo_actual_pg(inumbuild_temp)/dz)+2,kend_pg
                  do j=jstart(ibuild),jend(ibuild)
                     do i=istart(ibuild),iend(ibuild)
                        icellflag(i,j,k)=0
                     enddo
                  enddo
!                  enddo
               enddo     !END LOOP THROUGH stories
            endif
         enddo
         allocate(xfo(inumbuild),yfo(inumbuild),zfo_actual(inumbuild))  
         allocate(gamma(inumbuild))
         allocate(Ht(inumbuild),Wti(inumbuild),Lti(inumbuild))
         allocate(bldnum(inumbuild),bldtype(inumbuild),group_id(inumbuild))
         allocate(atten(inumbuild))
         allocate(Weff(inumbuild),Leff(inumbuild),Lf(inumbuild),Lr(inumbuild))
         allocate(Sx_east(inumbuild),Sy_north(inumbuild),phiprime(inumbuild),phi(inumbuild),theta(inumbuild))
         bldnum(:) =  bldnum_pg(:)
         bldtype(:) = bldtype_pg(:)
         group_id(:)= group_id_pg(:)
         xfo(:) = xfo_pg(:)
         yfo(:)=yfo_pg(:)
         zfo_actual(:)=zfo_actual_pg(:)
         gamma(:)=gamma_pg(:)
         Ht(:)=Ht_pg(:)
         Wti(:)=Wti_pg(:)
         Lti(:)=Lti_pg(:)
         atten(:)=atten_pg(:)
         Weff(:)=Weff_pg(:)
         Leff(:)=Leff_pg(:)
         Lf(:)=Lf_pg(:)
!         Lfy(:)=Lfy_pg(:)
         Lr(:)=Lr_pg(:)
         Sx_east(:) = Sx_east_pg(:)
         Sy_north(:) = Sy_north_pg(:)
         phiprime(:)=phiprime_pg(:)
         phi(:)=phi_pg(:)
         theta(:)=theta_pg(:)
         deallocate(xfo_pg,yfo_pg,zfo_actual_pg,gamma_pg,Ht_pg,Wti_pg,Lti_pg,bldnum_pg,bldtype_pg)
         deallocate(group_id_pg,atten_pg,phiprime_pg,phi_pg,theta_pg,Lf_pg,Lr_pg)
         deallocate(Sx_east_pg,Sy_north_pg,Weff_pg,Leff_pg)
         return
      end
