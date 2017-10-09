      subroutine poisson
         use datamodule ! make data from module "datamodule" visible
         implicit none
         integer iter
        real coeff
         real, allocatable:: ep(:,:,:),fp(:,:,:),gp(:,:,:),hp(:,:,:),mp(:,:,:),np(:,:,:)
         allocate(ep(nx-1,ny-1,nz-1),fp(nx-1,ny-1,nz-1),gp(nx-1,ny-1,nz-1),hp(nx-1,ny-1,nz-1))
         allocate(mp(nx-1,ny-1,nz-1),np(nx-1,ny-1,nz-1))
! A and B are coefficients for sor solver and are defined in init.f90
         !do k=1,nz-1
         !   ! MAN 07/25/2008 stretched vertical grid
         !   B=dx**2./(dz_array(k)**2.) ! MAN 07/25/2008 stretched vertical grid
         !   ep(:,:,k)=  e(:,:,k)/ ( 2.* ( o(:,:,k) + A*p(:,:,k) + B*q(:,:,k) ) )
         !   fp(:,:,k)=  f(:,:,k)/ ( 2.* ( o(:,:,k) + A*p(:,:,k) + B*q(:,:,k) ) )
         !   gp(:,:,k)=A*g(:,:,k)/ ( 2.* ( o(:,:,k) + A*p(:,:,k) + B*q(:,:,k) ) )
         !   hp(:,:,k)=A*h(:,:,k)/ ( 2.* ( o(:,:,k) + A*p(:,:,k) + B*q(:,:,k) ) )
         !   mp(:,:,k)=B*m(:,:,k)/ ( 2.* ( o(:,:,k) + A*p(:,:,k) + B*q(:,:,k) ) )
         !   np(:,:,k)=B*n(:,:,k)/ ( 2.* ( o(:,:,k) + A*p(:,:,k) + B*q(:,:,k) ) )
         !enddo

          do k=2,nz-1
!            uncomment this when stretched gridding in x and y direction are implemented
!            e(:,:,k)=denom(i,j,k)*e(i,j,k)/(dx_array(i)*0.5*(dx_array(i)+dx_array(i+1)))
!            f(:,:,k)=denom(i,j,k)*f(i,j,k)/(dx_array(i)*0.5*(dx_array(i)+dx_array(i-1)))
!            g(:,:,k)=denom(i,j,k)*g(i,j,k)/(dy_array(j)*0.5*(dy_array(j)+dy_array(j+1)))
!            h(:,:,k)=denom(i,j,k)*h(i,j,k)/(dy_array(j)*0.5*(dy_array(j)+dy_array(j-1)))
             ep(:,:,k)=e(:,:,k)/(dx*dx)
             fp(:,:,k)=f(:,:,k)/(dx*dx)
             gp(:,:,k)=g(:,:,k)/(dy*dy)
             hp(:,:,k)=h(:,:,k)/(dy*dy)
             mp(:,:,k)=m(:,:,k)/(dz_array(k)*0.5*(dz_array(k)+dz_array(k+1)))
             np(:,:,k)=n(:,:,k)/(dz_array(k)*0.5*(dz_array(k)+dz_array(k-1)))
          enddo

					! This causes a bad bug when street_intersection is turned on.
					!ep=ep/(ep+fp+gp+hp+mp+np)
					!fp=fp/(ep+fp+gp+hp+mp+np)
					!gp=gp/(ep+fp+gp+hp+mp+np)
					!hp=hp/(ep+fp+gp+hp+mp+np)
					!mp=mp/(ep+fp+gp+hp+mp+np)
					!np=np/(ep+fp+gp+hp+mp+np)
				  
				  do k=1,nz-1
				  	do j=1,ny-1
				  		do i=1,nx-1
				  			coeff=ep(i,j,k)+fp(i,j,k)+gp(i,j,k)+hp(i,j,k)+mp(i,j,k)+np(i,j,k)
				  			
				  			ep(i,j,k)=ep(i,j,k)/coeff
				  			fp(i,j,k)=fp(i,j,k)/coeff
				  			gp(i,j,k)=gp(i,j,k)/coeff
				  			hp(i,j,k)=hp(i,j,k)/coeff
				  			mp(i,j,k)=mp(i,j,k)/coeff
				  			np(i,j,k)=np(i,j,k)/coeff
				  			
				  		enddo
				  	enddo
				  enddo

         i=1
         j=1
         k=1
         do iter=1,10
            do k= 2,nz-1
               do j=2,ny-1
                  do i= 2,nx-1
                     if(icellflag(i,j,k) .eq. 9 .and. icellflag(i-1,j,k) .eq. 9)then
                        uo(i,j,k)=((ep(i,j,k)*uo(i+1,j,k)+fp(i,j,k)*uo(i-1,j,k)) &
                                  +(gp(i,j,k)*uo(i,j+1,k)+hp(i,j,k)*uo(i,j-1,k)) &
                                  +(mp(i,j,k)*uo(i,j,k+1)+np(i,j,k)*uo(i,j,k-1)))
                     endif
                     if(icellflag(i,j,k) .eq. 9 .and. icellflag(i,j-1,k) .eq. 9)then
                        vo(i,j,k)=((ep(i,j,k)*vo(i+1,j,k)+fp(i,j,k)*vo(i-1,j,k)) &
                                    +(gp(i,j,k)*vo(i,j+1,k)+hp(i,j,k)*vo(i,j-1,k)) &
                                    +(mp(i,j,k)*vo(i,j,k+1)+np(i,j,k)*vo(i,j,k-1)))
                     endif
                     if(icellflag(i,j,k) .eq. 9 .and. icellflag(i,j,k-1) .eq. 9)then
                        wo(i,j,k)=((ep(i,j,k)*wo(i+1,j,k)+fp(i,j,k)*wo(i-1,j,k)) &
                                    + (gp(i,j,k)*wo(i,j+1,k)+hp(i,j,k)*wo(i,j-1,k)) &
                                    + (mp(i,j,k)*wo(i,j,k+1)+np(i,j,k)*wo(i,j,k-1)))
                     endif
                  enddo
               enddo
            enddo
         enddo
         return
      end
