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
      subroutine unbuild_garage
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! This subroutine set the velocities within the parking garage
! ERP/AAG 8/2007
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

         use datamodule ! make data from module "datamodule" visible
         implicit none

         integer inumbuild_old
       
         integer, allocatable :: bldnum_pg(:),bldtype_pg(:),group_id_pg(:)
         real, allocatable :: Ht_pg(:),Wti_pg(:),Lti_pg(:),atten_pg(:)
         real, allocatable :: xfo_pg(:),yfo_pg(:),zfo_actual_pg(:),gamma_pg(:)
         real, allocatable :: Weff_pg(:),Leff_pg(:),Lf_pg(:),Lr_pg(:)
         real, allocatable :: Sx_east_pg(:),Sy_north_pg(:),phiprime_pg(:),phi_pg(:),theta_pg(:)

         inumbuild_old = inumbuild
         inumbuild = inumbuild_old - num_new_builds

         allocate(xfo_pg(inumbuild),yfo_pg(inumbuild),zfo_actual_pg(inumbuild))  
         allocate(gamma_pg(inumbuild),Ht_pg(inumbuild),Wti_pg(inumbuild),Lti_pg(inumbuild))
         allocate(bldnum_pg(inumbuild),bldtype_pg(inumbuild),group_id_pg(inumbuild),atten_pg(inumbuild))
         allocate(Weff_pg(inumbuild),Leff_pg(inumbuild),Lf_pg(inumbuild),Lr_pg(inumbuild))
         allocate(Sx_east_pg(inumbuild),Sy_north_pg(inumbuild),phiprime_pg(inumbuild))
         allocate(phi_pg(inumbuild),theta_pg(inumbuild))
         
         bldnum_pg(:)  = bldnum(1:inumbuild)
         bldtype_pg(:) = bldtype(1:inumbuild)
         group_id_pg(:)=group_id(1:inumbuild)
         xfo_pg(:)=xfo(1:inumbuild)
         yfo_pg(:)=yfo(1:inumbuild)
         zfo_actual_pg(:)=zfo_actual(1:inumbuild)
         gamma_pg(:)=gamma(1:inumbuild)
         Ht_pg(:)=Ht(1:inumbuild)
         Wti_pg(:)=Wti(1:inumbuild)
         Lti_pg(:)=Lti(1:inumbuild)
         atten_pg(:)=atten(1:inumbuild)
         Weff_pg(:)=Weff(1:inumbuild)
         Leff_pg(:)=Leff(1:inumbuild)
         Lf_pg(:)=Lf(1:inumbuild)
         Lr_pg(:)=Lr(1:inumbuild)
         Sx_east_pg(:) = Sx_east(1:inumbuild)
         Sy_north_pg(:) = Sy_north(1:inumbuild)
         phiprime_pg(:)=phiprime(1:inumbuild)
         phi_pg(:)=phi(1:inumbuild)
         theta_pg(:)=theta(1:inumbuild)
         deallocate(xfo,yfo,zfo_actual,gamma,Ht,Wti,Lti,bldnum,bldtype,group_id,atten,phiprime)
         deallocate(Sx_east,Sy_north,Lf,Lr,Weff,Leff,phi,theta)
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
