	subroutine denominators

		use datamodule ! make data from module "datamodule" visible
		implicit none

		do k=2,nz-1

			e(:,:,k)=e(:,:,k)/(dx*dx)
			f(:,:,k)=f(:,:,k)/(dx*dx)
			g(:,:,k)=g(:,:,k)/(dy*dy)
			h(:,:,k)=h(:,:,k)/(dy*dy)
			m(:,:,k)=m(:,:,k)/(dz_array(k)*0.5*(dz_array(k)+dz_array(k+1)))
			n(:,:,k)=n(:,:,k)/(dz_array(k)*0.5*(dz_array(k)+dz_array(k-1)))
		
			r(:,:,k)=r(:,:,k)
		
			denoms(:,:,k)=omegarelax/(e(:,:,k)+f(:,:,k)+g(:,:,k)+h(:,:,k)+m(:,:,k)+n(:,:,k))
		
		enddo
		
		return
	end

