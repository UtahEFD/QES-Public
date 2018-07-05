#include "Solver.h"


Solver::Solver(URBInputData* UID)
{

	rooftopFlag = UID->simParams->rooftopFlag;
	upwindCavityFlag = UID->simParams->upwindCavityFlag;
	streetCanyonFlag = UID->simParams->streetCanyonFlag;
	streetIntersectionFlag = UID->simParams->streetIntersectionFlag;
	wakeFlag = UID->simParams->wakeFlag;
	sidewallFlag = UID->simParams->sidewallFlag;

	Vector3<int> v;
	v = *(UID->simParams->domain);
	nx = v[0];
	ny = v[1];
	nz = v[2];

	nx += 1;        /// +1 for Staggered grid
	ny += 1;        /// +1 for Staggered grid
	nz += 2;        /// +2 for staggered grid and ghost cell


	Vector3<float> w;
	w = *(UID->simParams->grid);
	dx = w[0];
	dy = w[1];
	dz = w[2];
	itermax = UID->simParams->maxIterations;
	dxy = MIN(dx, dy);

	z_ref = UID->metParams->sensor->height;
	U_ref = UID->metParams->sensor->speed;
	z0 = UID->buildings->wallRoughness;

	for (int i = 0; i < UID->buildings->buildings.size(); i++)
	{
		if (UID->buildings->buildings[i]->buildingGeometry == 1)
		{
			buildings.push_back(UID->buildings->buildings[i]);
		}
	}

	int j = 0;
	for (int i = 0; i < nz; i++)
	{
		if (UID->simParams->verticalStretching == 0)
			dzArray.push_back(dz);
		else
			dzArray.push_back(UID->simParams->dzArray[j]);

		if (i != 0 && i != nz - 2)
			j++;
	}

	zm.push_back(-0.5*dzArray[0]);
	z.push_back(0.0f);
	for (int i = 1; i < nz; i++)
	{
		z.push_back(z[i - 1] + dzArray[i]);
		zm.push_back(z[i] - 0.5f * dzArray[i]);
	} 

	mesh = 0;
	if (UID->terrain)
		mesh = new Mesh(UID->terrain->tris);

}

void Solver::defineWalls(int* iCellFlag, float* n, float* m, float* f, float* e, float* h, float* g)
{
	for (int k = 1; k < nz-2; k++){
		for (int j = 1; j < ny-2; j++){
			for (int i = 1; i < nx-2; i++){
				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
				if (iCellFlag[icell_cent] != 0) {
					
					/// Wall bellow
					if (iCellFlag[icell_cent-(nx-1)*(ny-1)]==0) {
						n[icell_cent] = 0.0f; 

					}
					/// Wall above
					if (iCellFlag[icell_cent+(nx-1)*(ny-1)]==0) {
						m[icell_cent] = 0.0f;
					}
					/// Wall in back
					if (iCellFlag[icell_cent-1]==0){
						f[icell_cent] = 0.0f; 
					}
					/// Wall in front
					if (iCellFlag[icell_cent+1]==0){
						e[icell_cent] = 0.0f; 
					}
					/// Wall on right
					if (iCellFlag[icell_cent-(nx-1)]==0){
						h[icell_cent] = 0.0f;
					}
					/// Wall on left
					if (iCellFlag[icell_cent+(nx-1)]==0){
						g[icell_cent] = 0.0f; 
					}
				}
			}
		}
	}

		/// New boundary condition implementation
	for (int k = 1; k < nz-1; k++){
		for (int j = 0; j < ny-1; j++){
			for (int i = 0; i < nx-1; i++){
				int icell_cent = i + j*(nx-1) + k*(nx-1)*(ny-1);   /// Lineralized index for cell centered values
				e[icell_cent] /= (dx*dx);
				f[icell_cent] /= (dx*dx);
				g[icell_cent] /= (dy*dy);
				h[icell_cent] /= (dy*dy);
				m[icell_cent] /= (dz*dz);
				n[icell_cent] /= (dz*dz);
				//denom(:,:,k)=omegarelax/(e(:,:,k)+f(:,:,k)+g(:,:,k)+h(:,:,k)+m(:,:,k)+n(:,:,k))
			}
		}
	}
}

void Solver::upWind(Building* build, int* iCellFlag, double* u0, double* v0, double* w0, float* z, float* zm)
{

		 int perpendicular_flag,ns_flag;
		 int upIstart,upIstop,upJstart,upJstop;
		 float uo_h,vo_h,upwind_dir,upwind_rel,xco,yco;
		 std::vector<float> x,y;
		 float xf1,yf1,xf2,yf2,tol,ynorm,lfcoeff;
		 float zf,x_u,y_u,x_v,y_v,x_w,y_w;
		 float xs_u,xs_v,xs_w,xv_u,xv_v,xv_w,xrz_u,xrz_v;
		 float urot,vrot,uhrot,vhrot,vel_mag;
		 float vortex_height,build_width,retarding_factor;
		 float length_factor,height_factor,rz_end,retarding_height,eff_height;
		 float totalLength,perpendicularDir,gamma_eff;
		 int ktop,kbottom,iface,ivert,x_idx,y_idx;

		 if(build->buildingGeometry == 4)
		 {
			eff_height=0.8*(build->height - build->baseHeightActual ) + build->baseHeightActual;
			/*  This building geometry doesn't exist yet. Update this when it does.
			 xco = (RectangularBuilding*)(build)->xfo + (RectangularBuilding*)(build)->length*cos((RectangularBuilding*)(build)->rotation) //!CENTER of building in QUIC domain coordinates
			 yco = (RectangularBuilding*)(build)->yfo + (RectangularBuilding*)(build)->length*sin((RectangularBuilding*)(build)->rotation)
			*/
		 }
		 else if (build->buildingGeometry == 6)
		 {
			eff_height = build->height;
			xco = build->centroidX;
			yco = build->centroidY;
		 }
		 else //must be 1 which is rectangular building
		 {
			 eff_height = build->height;
			 xco = ((RectangularBuilding*)(build))->xFo + ((RectangularBuilding*)(build))->length*cos(((RectangularBuilding*)(build))->rotation); //!CENTER of building in QUIC domain coordinates
			 yco = ((RectangularBuilding*)(build))->yFo + ((RectangularBuilding*)(build))->length*sin(((RectangularBuilding*)(build))->rotation);
		 }
		 
		 // find upwind direction and deterMINe the type of flow regime
		 uo_h = u0[ CELL((int)(xco/dx), (int)(yco/dy), build->kEnd+1, 0)];
		 vo_h = v0[ CELL((int)(xco/dx), (int)(yco/dy), build->kEnd+1, 0)];
		 upwind_dir = atan2(vo_h,uo_h);
		 upwind_rel = upwind_dir - build->rotation;
		 uhrot = uo_h * cos(build->rotation) + vo_h * sin(build->rotation);
		 vhrot = -uo_h * sin(build->rotation) + vo_h * cos(build->rotation);
		 vel_mag = sqrt( (uo_h * uo_h) + ( vo_h * vo_h) );
		 tol = 10 * pi / 180.0f;
		 retarding_factor = 0.4f;
		 length_factor = 0.4f;
		 height_factor = 0.6;

		 if(upwindCavityFlag == 1)
			lfcoeff=2;
		 else
			lfcoeff=1.5;

		 if( upwind_rel > pi) upwind_rel = upwind_rel - 2 * pi;
		 if(upwind_rel < -pi) upwind_rel = upwind_rel + 2 * pi;


		 if(build->buildingGeometry == 6)
		 {
			/*
				NOTE::buildingGeo 6 isn't being implemented right now, so just leave this blank.

			allocate(LfFace(bldstopidx(ibuild)-bldstartidx(ibuild)),LengthFace(bldstopidx(ibuild)-bldstartidx(ibuild)))
			iface=0
			do ivert=bldstartidx(ibuild),bldstopidx(ibuild)
			   x1=0.5*(bldx(ivert)+bldx(ivert+1))
			   y1=0.5*(bldy(ivert)+bldy(ivert+1))
			   xf1=(bldx(ivert)-x1)*cos(upwind_dir)+(bldy(ivert)-y1)*sin(upwind_dir)
			   yf1=-(bldx(ivert)-x1)*sin(upwind_dir)+(bldy(ivert)-y1)*cos(upwind_dir)
			   xf2=(bldx(ivert+1)-x1)*cos(upwind_dir)+(bldy(ivert+1)-y1)*sin(upwind_dir)
			   yf2=-(bldx(ivert+1)-x1)*sin(upwind_dir)+(bldy(ivert+1)-y1)*cos(upwind_dir)
			   upwind_rel=atan2(yf2-yf1,xf2-xf1)+0.5*pi
			   if(upwind_rel .gt. pi)upwind_rel=upwind_rel-2*pi
			   if(abs(upwind_rel) .gt. pi-tol)then
				  perpendicularDir=atan2(bldy(ivert+1)-bldy(ivert),bldx(ivert+1)-bldx(ivert))+0.5*pi
				  if(perpendicularDir.le.-pi)perpendicularDir=perpendicularDir+2*pi
				  if(abs(perpendicularDir) .ge. 0.25*pi .and. abs(perpendicularDir) .le. 0.75*pi)then
					 ns_flag=1
				  else
					 ns_flag=0
				  endif
				  gamma_eff=perpendicularDir
				  if(gamma_eff .ge. 0.75*pi)then
					 gamma_eff=gamma_eff-pi
				  elseif(gamma_eff .ge. 0.25*pi)then
					 gamma_eff=gamma_eff-0.5*pi
				  elseif(gamma_eff .lt. -0.75*pi)then
					 gamma_eff=gamma_eff+pi
				  elseif(gamma_eff .lt. -0.25*pi)then
					 gamma_eff=gamma_eff+0.5*pi
				  endif
				  uhrot=uo_h*cos(gamma_eff)+vo_h*sin(gamma_eff)
				  vhrot=-uo_h*sin(gamma_eff)+vo_h*cos(gamma_eff)
				  iface=iface+1
				  LengthFace(iface)=sqrt(((xf2-xf1)**2.)+((yf2-yf1)**2.))
				  LfFace(iface)=abs(lfcoeff*LengthFace(iface)*cos(upwind_rel)/(1+0.8*LengthFace(iface)/eff_height))
				  if(upwindflag .eq. 3)theniCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
iCellFlag
					 vortex_height=MIN(LengthFace(iface),eff_height)
					 retarding_height=eff_height
				  else
					 vortex_height=eff_height
					 retarding_height=eff_height
				  endif
				  ! MAN 07/25/2008 stretched vertical grid
				  do k=2,kstart(ibuild)
					 kbottom=k
					 if(zfo(ibuild) .le. zm(k))exit
				  enddo
				  do k=kstart(ibuild),nz-1
					 ktop=k
					 if(height_factor*retarding_height+zfo_actual(ibuild) .le. z(k))exit
				  enddo
				  upIstart=MAX(nint(MIN(bldx(ivert),bldx(ivert+1))/dx)-nint(1.5*LfFace(iface)/dx),2)
				  upIstop=MIN(nint(MAX(bldx(ivert),bldx(ivert+1))/dx)+nint(1.5*LfFace(iface)/dx),nx-1)
				  upJstart=MAX(nint(MIN(bldy(ivert),bldy(ivert+1))/dy)-nint(1.5*LfFace(iface)/dy),2)
				  upJstop=MIN(nint(MAX(bldy(ivert),bldy(ivert+1))/dy)+nint(1.5*LfFace(iface)/dy),ny-1)
				  ynorm=abs(yf2)
				  do k=kbottom,ktop
					 zf=zm(k)-zfo(ibuild)
					 do j=upJstart,upJstop
						do i=upIstart,upIstop
						   x_u=((real(i)-1)*dx-x1)*cos(upwind_dir)+ &
										((real(j)-0.5)*dy-y1)*sin(upwind_dir)
						   y_u=-((real(i)-1)*dx-x1)*sin(upwind_dir)+ &
										((real(j)-0.5)*dy-y1)*cos(upwind_dir)
						   x_v=((real(i)-0.5)*dx-x1)*cos(upwind_dir)+ &
										((real(j)-1)*dy-y1)*sin(upwind_dir)
						   y_v=-((real(i)-0.5)*dx-x1)*sin(upwind_dir)+	&
										((real(j)-1)*dy-y1)*cos(upwind_dir)
						   x_w=((real(i)-0.5)*dx-x1)*cos(upwind_dir)+ &
										((real(j)-0.5)*dy-y1)*sin(upwind_dir)
						   y_w=-((real(i)-0.5)*dx-x1)*sin(upwind_dir)+	&
										((real(j)-0.5)*dy-y1)*cos(upwind_dir)
!u values
						   if(abs(y_u) .le. ynorm .and. height_factor*vortex_height .gt. zf)then
							  xs_u=((xf2-xf1)/(yf2-yf1))*(y_u-yf1)+xf1
							  xv_u=-LfFace(iface)*sqrt((1-((y_u/ynorm)**2.))*(1-((zf/(height_factor*vortex_height))**2.)))
							  xrz_u=-LfFace(iface)*sqrt((1-((y_u/ynorm)**2.))*(1-((zf/(height_factor*retarding_height))**2.)))
							  if(zf .gt. height_factor*vortex_height)then
								 rz_end=0.
							  else
								 rz_end=length_factor*xv_u
							  endif
							  if(upwindflag .eq. 1)then
								 if(x_u-xs_u .ge. xv_u .and. x_u-xs_u .le. 0.1*dxy .and. iCellFlag(i,j,k) .ne. 0)then
									uo(i,j,k)=0.
								 endif
							  else
								 if(x_u-xs_u .ge. xrz_u .and. x_u-xs_u .lt. rz_end &
									   .and. iCellFlag(i,j,k) .ne. 0)then
									if(upwindflag .eq. 3)then
									   uo(i,j,k)=((x_u-xs_u-xrz_u)*(retarding_factor-1.)/(rz_end-xrz_u)+1.)*uo(i,j,k)
									else
									   uo(i,j,k)=retarding_factor*uo(i,j,k)
									endif
									if(abs(uo(i,j,k)) .gt. max_velmag)then
									   print*,'Parameterized U exceeds MAX in upwind',&
										  uo(i,j,k),max_velmag,i,j,k
									endif
								 endif
								 if(x_u-xs_u .ge. length_factor*xv_u .and. x_u-xs_u .le. 0.1*dxy &
									   .and. iCellFlag(i,j,k) .ne. 0)then
									urot=uo(i,j,k)*cos(gamma_eff)
									vrot=-uo(i,j,k)*sin(gamma_eff)
									if(ns_flag .eq. 1)then
									   vrot=-vhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05)   &
											*(-height_factor*sin(((pi*abs(x_u-xs_u))/(length_factor*LfFace(iface)))+0))
									else
									   urot=-uhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05)   &
											*(-height_factor*sin(((pi*abs(x_u-xs_u))/(length_factor*LfFace(iface)))+0))
									endif
									uo(i,j,k)=urot*cos(-gamma_eff)+vrot*sin(-gamma_eff)
									if(abs(uo(i,j,k)) .gt. max_velmag)then
									   print*,'Parameterized U exceeds MAX in upwind',&
										  uo(i,j,k),max_velmag,i,j,k
									endif
								 endif
							  endif
						   endif
!v values
						   if(abs(y_v) .le. ynorm .and. height_factor*vortex_height .gt. zf)then
							  xs_v=((xf2-xf1)/(yf2-yf1))*(y_v-yf1)+xf1
							  xv_v=-LfFace(iface)*sqrt((1-((y_v/ynorm)**2.))*(1-((zf/(height_factor*vortex_height))**2.)))
							  xrz_v=-LfFace(iface)*sqrt((1-((y_v/ynorm)**2.))*(1-((zf/(height_factor*retarding_height))**2.)))
							  if(zf .ge. height_factor*vortex_height)then
								 rz_end=0.
							  else
								 rz_end=length_factor*xv_v
							  endif
							  if(upwindflag .eq. 1)then
								 if(x_v-xs_v .ge. xv_v .and. x_v-xs_v .le. 0.1*dxy .and. iCellFlag(i,j,k) .ne. 0)then
									vo(i,j,k)=0.
								 endif
							  else
								 if(x_v-xs_v .ge. xrz_v .and. x_v-xs_v .lt. rz_end &
									   .and. iCellFlag(i,j,k) .ne. 0)then
									if(upwindflag .eq. 3)then
									   vo(i,j,k)=((x_v-xs_v-xrz_v)*(retarding_factor-1.)/(rz_end-xrz_v)+1.)*vo(i,j,k)
									else
									   vo(i,j,k)=retarding_factor*vo(i,j,k)
									endif
									if(abs(vo(i,j,k)) .gt. max_velmag)then
									   print*,'Parameterized V exceeds MAX in upwind',&
										  vo(i,j,k),max_velmag,i,j,k
									endif
								 endif
								 if(x_v-xs_v .ge. length_factor*xv_v .and. x_v-xs_v .le. 0.1*dxy &
									   .and. iCellFlag(i,j,k) .ne. 0)then
									urot=vo(i,j,k)*sin(gamma_eff)
									vrot=vo(i,j,k)*cos(gamma_eff)
									if(ns_flag .eq. 1)then
									   vrot=-vhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05)   &
											*(-height_factor*sin(((pi*abs(x_v-xs_v))/(length_factor*LfFace(iface)))+0))
									else
									   urot=-uhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05)   &
											*(-height_factor*sin(((pi*abs(x_v-xs_v))/(length_factor*LfFace(iface)))+0))
									endif
									vo(i,j,k)=-urot*sin(-gamma_eff)+vrot*cos(-gamma_eff)
									if(abs(vo(i,j,k)) .gt. max_velmag)then
									   print*,'Parameterized V exceeds MAX in upwind',&
										  vo(i,j,k),max_velmag,i,j,k
									endif
								 endif
							  endif
						   endif
!w values
						   if(abs(y_w) .le. ynorm .and. height_factor*vortex_height .gt. zf)then
							  xs_w=((xf2-xf1)/(yf2-yf1))*(y_w-yf1)+xf1
							  xv_w=-LfFace(iface)*sqrt((1-((y_w/ynorm)**2.))*(1-((zf/(height_factor*vortex_height))**2.)))
							  if(upwindflag .eq. 1)then
								 if(x_w-xs_w .ge. xv_w .and. x_w-xs_w .le. 0.1*dxy .and. iCellFlag(i,j,k) .ne. 0)then
									wo(i,j,k)=0.
									if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
									   iCellFlag(i,j,k)=2
									endif
								 endif
							  else
								 if(x_w-xs_w .ge. xv_w .and. x_w-xs_w .lt. length_factor*xv_w &
									   .and. iCellFlag(i,j,k) .ne. 0)then
									wo(i,j,k)=retarding_factor*wo(i,j,k)
									if(abs(wo(i,j,k)) .gt. max_velmag)then
									   print*,'Parameterized W exceeds MAX in upwind',&
										  wo(i,j,k),max_velmag,i,j,k
									endif
									if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
									   iCellFlag(i,j,k)=2
									endif
								 endif
								 if(x_w-xs_w .ge. length_factor*xv_w .and. x_w-xs_w .le. 0.1*dxy &
									   .and. iCellFlag(i,j,k) .ne. 0)then
									wo(i,j,k)=-vel_mag*(0.1*cos(((pi*abs(x_w-xs_w))/(length_factor*LfFace(iface))))-0.05)
									if(abs(wo(i,j,k)) .gt. max_velmag)then
									   print*,'Parameterized W exceeds MAX in upwind',&
										  wo(i,j,k),max_velmag,i,j,k
									endif
									if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
									   iCellFlag(i,j,k)=2
									endif
								 endif
							  endif
						   endif
						enddo
					 enddo
				  enddo
			   endif
			   if(bldx(ivert+1) .eq. bldx(bldstartidx(ibuild)) &
					 .and. bldy(ivert+1) .eq. bldy(bldstartidx(ibuild)))exit
			enddo
			if(iface .gt. 0)then
			   totalLength=0.
			   Lf(ibuild)=0.
			   do ivert=1,iface
				  Lf(ibuild)=Lf(ibuild)+LfFace(ivert)*LengthFace(ivert)
				  totalLength=totalLength+LengthFace(ivert)
			   enddo
			   Lf(ibuild)=Lf(ibuild)/totalLength
			else
			   Lf(ibuild)=-999.0
			endif
			deallocate(LfFace,LengthFace)
			*/
			build->iStart += 1; //dummy code for build, throw out on implementation
		 }
		 else
			//Location of corners relative to the center of the building
			x.push_back(((NonPolyBuilding*)build)->xFo + ((NonPolyBuilding*)build)->width * sin((build)->rotation) - xco);
			y.push_back(((NonPolyBuilding*)build)->yFo  - ((NonPolyBuilding*)build)->width * cos(build->rotation) - yco);
			x.push_back(x[0] + ((NonPolyBuilding*)build)->length * cos(build->rotation));
			y.push_back(y[0] + ((NonPolyBuilding*)build)->length * sin(build->rotation));
			x.push_back(((NonPolyBuilding*)build)->xFo - ((NonPolyBuilding*)build)->width * sin(build->rotation) - xco);
			y.push_back(((NonPolyBuilding*)build)->yFo + ((NonPolyBuilding*)build)->width * cos(build->rotation) - yco);
			x.push_back(x[2] + ((NonPolyBuilding*)build)->length * cos(build->rotation));
			y.push_back(y[2] + ((NonPolyBuilding*)build)->length * sin(build->rotation));


			//flip the last two values to maintain order
			float tempx, tempy;
			tempx = x[3];
			tempy = y[3];
			x[3] = x[2];
			y[3] = y[2];
			x[2] = tempx;
			y[2] = tempy;



			perpendicular_flag = 0;

			int num = -1;
			if(upwind_rel > 0.5 * pi - tol && upwind_rel < 0.5 * pi + tol )
			{ 
			  num = 2;
			   perpendicular_flag=1;
			   ns_flag=1;
			   ((NonPolyBuilding*)build)->length = abs(lfcoeff * ((NonPolyBuilding*)build)->length * sin(upwind_rel) / ( 1 + 0.8 * ((NonPolyBuilding*)build)->length / eff_height));
			   build_width=((NonPolyBuilding*)build)->length;
			}
			else if(upwind_rel > -tol && upwind_rel > tol)
			{
			   num = 1;
			   perpendicular_flag=1;
			   ns_flag=0;
			   ((NonPolyBuilding*)build)->length = abs(lfcoeff * ((NonPolyBuilding*)build)->width * cos(upwind_rel) / ( 1 + 0.8 * ((NonPolyBuilding*)build)->width / eff_height));
			   build_width=((NonPolyBuilding*)build)->width;
		   }
			else if(upwind_rel > -0.5 * pi - tol && upwind_rel < -0.5 * pi + tol)
			{
			   num = 4;
			   perpendicular_flag=1;
			   ns_flag=1;
			   ((NonPolyBuilding*)build)->length = abs(lfcoeff * ((NonPolyBuilding*)build)->length * sin(upwind_rel) / ( 1 + 0.8 * ((NonPolyBuilding*)build)->length / eff_height));
			   build_width=((NonPolyBuilding*)build)->length;
			}
			else if(upwind_rel > pi - tol || upwind_rel < -pi + tol)
			{
			   num = 3;
			   perpendicular_flag=1;
			   ns_flag=0;
			   ((NonPolyBuilding*)build)->length = abs(lfcoeff * ((NonPolyBuilding*)build)->width * cos(upwind_rel) / ( 1 + 0.8 * ((NonPolyBuilding*)build)->width / eff_height));
			   build_width=((NonPolyBuilding*)build)->width;
			}

			if (num > -1)
			{
			   xf1=x[ (num % 4) + 1 ]*cos(upwind_dir)+y[ (num % 4) + 1 ]*sin(upwind_dir);
			   yf1=-x[ (num % 4) + 1 ]*sin(upwind_dir)+y[ (num % 4) + 1 ]*cos(upwind_dir);
			   xf2=x[ ( (num - 1) % 4) + 1]*cos(upwind_dir)+y[((num - 1) % 4) + 1]*sin(upwind_dir);
			   yf2=-x[ ((num - 1) % 4) + 1 ]*sin(upwind_dir)+y[((num - 1) % 4) + 1]*cos(upwind_dir);
			}

			ynorm = abs(yf1);
			if(perpendicular_flag == 1)
			{
			   if(upwindCavityFlag == 3)
			   {
				  vortex_height = MIN(build_width , eff_height);
				  retarding_height = eff_height;
			  }
			   else
			   {
				  vortex_height = eff_height;
				  retarding_height = eff_height;
			   }
			   // MAN 07/25/2008 stretched vertical grid
			   for (int k = 1; k <= build->kStart; k++)
			   {
				  kbottom = k;
				  if( build->baseHeight <= zm[k]) break;
			   }
			   for ( int k = build->kStart; k <= nz-1; k++)
				{
				  ktop = k;
				  if(height_factor * retarding_height + build->baseHeightActual <= z[k] ) break;
			   }
			   upIstart = MAX(build->iStart - (int)(1.5 * build->Lf / dx), 2);
			   upIstop = MIN(build->iEnd + (int)(1.5 * build->Lf / dx), nx - 1);
			   upJstart = MAX(build->jStart - (int)(1.5 * build->Lf / dy), 2);
			   upJstop = MIN(build->jEnd + (int)(1.5 * build->Lf / dy), ny-1);
				for ( int k = kbottom; k <= ktop; k++)
				{
				  zf = zm[k]- build->baseHeight;
					for (int j = upJstart; j <= upJstop; j++)
					{
					   for ( int i = upIstart; i <= upIstop; i++)
					   {
						x_u = (((float)(i) - 1) * dx - xco) * cos(upwind_dir) + 
									 (((float)(j) - 0.5 ) * dy - yco) * sin(upwind_dir);
						y_u = -(((float)(i)-1)*dx-xco)*sin(upwind_dir)+ 
									 (((float)(j)-0.5)*dy-yco)*cos(upwind_dir);
						x_v = (((float)(i)-0.5)*dx-xco)*cos(upwind_dir)+ 
									 (((float)(j)-1)*dy-yco)*sin(upwind_dir);
						y_v = -(((float)(i)-0.5)*dx-xco)*sin(upwind_dir)+	
									 (((float)(j)-1)*dy-yco)*cos(upwind_dir);
						x_w = (((float)(i)-0.5)*dx-xco)*cos(upwind_dir)+ 
									 (((float)(j)-0.5)*dy-yco)*sin(upwind_dir);
						y_w= -(((float)(i)-0.5)*dx-xco)*sin(upwind_dir)+	
									 (((float)(j)-0.5)*dy-yco)*cos(upwind_dir);
// u values
						if(y_u >= -ynorm && y_u <= ynorm)
						{
						   xs_u =((xf2-xf1)/(yf2-yf1))*(y_u-yf1)+xf1;
						   
						   if(zf > height_factor * vortex_height )
						   {
							  rz_end = 0.0f;
							  xv_u = 0.0f;
							  xrz_u = 0.0f;
						   }
						   else
						   {
							  xv_u = -build->Lf * sqrt( (1 - (pow(y_u/ynorm,2))) * (1 - pow((zf/(height_factor*vortex_height)),2)));
							  xrz_u = -build->Lf * sqrt( (1 - (pow(y_u/ynorm,2))) * (1 - pow((zf/(height_factor*retarding_height)),2)));
							  rz_end = length_factor * xv_u;
						   }
						   if(upwindCavityFlag == 1)
						   {
							  if(x_u-xs_u >= xv_u && x_u - xs_u <= 0.1 * dxy && iCellFlag[CELL(i,j,k,1)] != 0)
								 u0[CELL(i,j,k,0)] = 0.0f;
						   }
						   else
						   {
							  if(x_u - xs_u >= xrz_u && x_u - xs_u < rz_end &&
									iCellFlag[CELL(i,j,k,1)] != 0)
							  {
								 if(upwindCavityFlag == 3)
									u0[CELL(i,j,k,0)] *= ((x_u - xs_u - xrz_u) * (retarding_factor - 1.0f) / (rz_end - xrz_u) + 1.0f);
								 else
									u0[CELL(i,j,k,0)] *= retarding_factor;
								 if( abs(u0[CELL(i,j,k,0)]) > max_velmag)
									printf("Parameterized U exceeds MAX in upwind: %lf : %lf i:%d j:%d k:%d\n",u0[CELL(i,j,k,0)],max_velmag,i,j,k);
							  }
							  if(x_u - xs_u >= length_factor * xv_u && x_u - xs_u <= 0.1f * dxy &&
									iCellFlag[CELL(i,j,k,1)] != 0)
							  {
								 urot = u0[CELL(i,j,k,0)]*cos(build->rotation);
								 vrot = -u0[CELL(i,j,k,0)]*sin(build->rotation);
								 if(ns_flag == 1)
								 {
									vrot=-vhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05) * (-height_factor*sin(((pi*abs(x_u-xs_u))/(length_factor*build->Lf))+0));
								 }
								 else
								 {
									urot=-uhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05) * (-height_factor*sin(((pi*abs(x_u-xs_u))/(length_factor*build->Lf))+0));
								 }
								 u0[CELL(i,j,k,0)]=urot*cos(-build->rotation)+vrot*sin(-build->rotation);
								 if(abs(u0[CELL(i,j,k,0)]) > max_velmag)
									printf("Parameterized U exceeds MAX in upwind: %lf : %lf i:%d j:%d k:%d\n",u0[CELL(i,j,k,0)],max_velmag,i,j,k);
							  }
						   }
						}
//v values
						if(y_v >= -ynorm && y_v <= ynorm)
						{
						   xs_v =((xf2-xf1)/(yf2-yf1))*(y_v-yf1)+xf1;
						   
						   if(zf >= height_factor * vortex_height )
						   {
							  rz_end = 0.0f;
							  xv_v = 0.0f;
							  xrz_v = 0.0f;
						   }
						   else
						   {
							  xv_v = -build->Lf * sqrt( (1 - (pow(y_v/ynorm,2))) * (1 - pow((zf/(height_factor*vortex_height)),2)));
							  xrz_v = -build->Lf * sqrt( (1 - (pow(y_v/ynorm,2))) * (1 - pow((zf/(height_factor*retarding_height)),2)));
							  rz_end = length_factor * xv_v;
						   }
						   if(upwindCavityFlag == 1)
						   {
							  if(x_v-xs_v >= xv_v && x_v - xs_v <= 0.1 * dxy && iCellFlag[CELL(i,j,k,1)] != 0)
								 v0[CELL(i,j,k,0)] = 0.0f;
						   }
						   else
						   {
							  if(x_v - xs_v >= xrz_v && x_v - xs_v < rz_end &&
									iCellFlag[CELL(i,j,k,1)] != 0)
							  {
								 if(upwindCavityFlag == 3)
									v0[CELL(i,j,k,0)] *= ((x_v - xs_v - xrz_v) * (retarding_factor - 1.0f) / (rz_end - xrz_v) + 1.0f);
								 else
									v0[CELL(i,j,k,0)] *= retarding_factor;
								 if( abs(v0[CELL(i,j,k,0)]) > max_velmag)
									printf("Parameterized V exceeds MAX in upwind: %lf : %lf i:%d j:%d k:%d\n",v0[CELL(i,j,k,0)],max_velmag,i,j,k);
							  }
							  if(x_v - xs_v >= length_factor * xv_v && x_v - xs_v <= 0.1f * dxy &&
									iCellFlag[CELL(i,j,k,1)] != 0)
							  {
								 urot = v0[CELL(i,j,k,0)]*sin(build->rotation);
								 vrot = -v0[CELL(i,j,k,0)]*cos(build->rotation);
								 if(ns_flag == 1)
								 {
									vrot=-vhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05) * (-height_factor*sin(((pi*abs(x_v-xs_v))/(length_factor*build->Lf))+0));
								 }
								 else
								 {
									vrot=-vhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05) * (-height_factor*sin(((pi*abs(x_v-xs_v))/(length_factor*build->Lf))+0));
								 }
								 v0[CELL(i,j,k,0)]=-urot*sin(-build->rotation)+vrot*cos(-build->rotation);
								 if(abs(v0[CELL(i,j,k,0)]) > max_velmag)
									printf("Parameterized V exceeds MAX in upwind: %lf : %lf i:%d j:%d k:%d\n",v0[CELL(i,j,k,0)],max_velmag,i,j,k);
							  }
						   }
						}
//w values
						if(y_w >= -ynorm && y_w <= ynorm)
						{
						   xs_w =((xf2-xf1)/(yf2-yf1))*(y_w-yf1)+xf1;
						   
						   if(zf >= height_factor * vortex_height )
						   {
							  xv_w = 0.0f;
						   }
						   else
						   {
							  xv_w = -build->Lf * sqrt( (1 - (pow(y_w/ynorm,2))) * (1 - pow((zf/(height_factor*vortex_height)),2)));
						   }
						   if(upwindCavityFlag == 1)
						   {
							  if(x_w-xs_w >= xv_w && x_w-xs_w <= 0.1*dxy && iCellFlag[CELL(i,j,k,1)] != 0)
							  {
								 w0[CELL(i,j,k,0)] = 0.0f;
								 if(i < nx && j < ny && k < nz)
								 {
									iCellFlag[CELL(i,j,k,1)]=2;
								 }
							  }
						   }
						   else
						   {
							  if(x_w - xs_w >= xv_w && x_w - xs_w  < length_factor * xv_w &&
									iCellFlag[CELL(i,j,k,1)] != 0)
							  {
								 w0[CELL(i,j,k,0)] *= retarding_factor;
								 if(abs(w0[CELL(i,j,k,0)]) > max_velmag)
									printf("Parameterized W exceeds MAX in upwind: %lf : %lf i:%d j:%d k:%d\n",w0[CELL(i,j,k,0)],max_velmag,i,j,k);
								 if(i < nx - 1 && j < ny - 1 && k < nz - 1)
									iCellFlag[CELL(i,j,k,1)] = 2;
							  }
							  if(x_w - xs_w >= length_factor * xv_w && x_w - xs_w <= 0.0f && iCellFlag[CELL(i,j,k,1)] != 0)
							  {
								 //w0[CELL(i,j,k,0)] = -vel_mag*(0.1*cos(((pi*abs(x_w-xs_w))/(length_factor*build->Lf)))-0.05);
								 if(abs(w0[CELL(i,j,k,0)]) > max_velmag)
								 {
									printf("Parameterized W exceeds MAX in upwind: %lf : %lf i:%d j:%d k:%d\n",w0[CELL(i,j,k,0)],max_velmag,i,j,k);
								 }
								 if(i < nx - 1 && j < ny - 1 && k < nz - 1)
								 {
									iCellFlag[CELL(i,j,k,1)] = 2;
								 }
							  }
						   }
						}
					 } 
				  } 
			   }
		   }
			else
			   build->Lf = -999.0f;

}



void Solver::reliefWake(NonPolyBuilding* build, float* u0, float* v0)
{
	 int perpendicular_flag, uwakeflag, vwakeflag, wwakeflag;
	 float uo_h, vo_h, upwind_dir, upwind_rel, xco, yco;
	 float x1, y1, x2, y2, x3, y3, x4, y4;
	 float xw1, yw1, xw2, yw2, xw3, yw3, xf2, yf2, tol, zb, ynorm;
	 float farwake_exponent, farwake_factor, farwake_velocity;
	 float cav_fac, wake_fac, beta, LoverH, WoverH, upwind_rel_norm, eff_height;
	 float canyon_factor, xc, yc, dNu, dNv, xwall, xu, yu, xv, yv, xp, yp, xwallu, xwallv, xwallw;
	 int x_idx, y_idx, x_idx_min, iu, ju, iv, jv, kk, iw, jw;
	 float vd, hd, Bs, BL, shell_height, xw, yw, dNw;
	 int roof_perpendicular_flag, ns_flag;
	 int ktop, kbottom, nupwind;
	 float LrRect[3], LrLocal, LrLocalu, LrLocalv, LrLocalw;
	 float epsilon;
	 
	 epsilon = 10e-10;
	 
	 if(build->buildingGeometry == 4 && build->buildingRoof > 0)  //no current data param for buildingRoof
		eff_height = 0.8 * (build->height - build->baseHeightActual) + build->baseHeightActual;
	 else
		eff_height = build->height;

	 xco = build->xFo + build->Lt * cos(build->rotation); //!CENTER of building in QUIC domain coordinates
	 yco = build->yFo + build->Lt * sin(build->rotation);

	 //! find upwind direction and determine the type of flow regime
	 uo_h = u0[CELL( (int)(xco/dx), (int)(yco/dy), build->kEnd + 1, 0)];
	 vo_h = v0[CELL( (int)(xco/dx), (int)(yco/dy), build->kEnd + 1, 0)];
	 upwind_dir = atan2(vo_h,uo_h);
	 upwind_rel = upwind_dir - build->rotation;

	 if(upwind_rel > pi) upwind_rel = upwind_rel - 2 * pi;

	 if(upwind_rel <= -pi) upwind_rel = upwind_rel + 2 * pi;

	 upwind_rel_norm = upwind_rel + 0.5 * pi;

	 if(upwind_rel_norm > pi) upwind_rel_norm = upwind_rel_norm - 2 * pi;
	 tol = 0.01f * pi / 180.0f;

	 //!Location of corners relative to the center of the building

	 x1 = build->xFo + build->Wt * sin(build->rotation) - xco;
	 y1 = build->yFo - build->Wt * cos(build->rotation) - yco;
	 x2 = x1 + build->length * cos(build->rotation);
	 y2 = y1 + build->length * sin(build->rotation);
	 x4 = build->xFo - build->Wt * sin(build->rotation) - xco;
	 y4 = build->yFo + build->Wt * cos(build->rotation) - yco;
	 x3 = x4 + build->length * cos(build->rotation);
	 y3 = y4 + build->length * sin(build->rotation);
	 if(upwind_rel > 0.5f * pi + tol && upwind_rel < pi - tol)
	 {
		xw1=x1*cos(upwind_dir)+y1*sin(upwind_dir);
		yw1=-x1*sin(upwind_dir)+y1*cos(upwind_dir);
		xw2=x4*cos(upwind_dir)+y4*sin(upwind_dir);
		yw2=-x4*sin(upwind_dir)+y4*cos(upwind_dir);
		xf2=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		yf2=-x2*sin(upwind_dir)+y2*cos(upwind_dir);
		xw3=x3*cos(upwind_dir)+y3*sin(upwind_dir);
		yw3=-x3*sin(upwind_dir)+y3*cos(upwind_dir);
		perpendicular_flag=0;
	}
	 else if (upwind_rel >= 0.5f * pi - tol && upwind_rel <= 0.5f * pi + tol)
	 {
		xw1=x4*cos(upwind_dir)+y4*sin(upwind_dir);
		yw1=-x4*sin(upwind_dir)+y4*cos(upwind_dir);
		xw3=x3*cos(upwind_dir)+y3*sin(upwind_dir);
		yw3=-x3*sin(upwind_dir)+y3*cos(upwind_dir);
		xf2=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		perpendicular_flag=1;
	}
	 else if(upwind_rel > tol && upwind_rel < 0.5f * pi - tol)
	 {
		xw1=x4*cos(upwind_dir)+y4*sin(upwind_dir);
		yw1=-x4*sin(upwind_dir)+y4*cos(upwind_dir);
		xw2=x3*cos(upwind_dir)+y3*sin(upwind_dir);
		yw2=-x3*sin(upwind_dir)+y3*cos(upwind_dir);
		xf2=x1*cos(upwind_dir)+y1*sin(upwind_dir);
		yf2=-x1*sin(upwind_dir)+y1*cos(upwind_dir);
		xw3=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		yw3=-x2*sin(upwind_dir)+y2*cos(upwind_dir);
		perpendicular_flag=0;
	}
	 else if( abs(upwind_rel) <= tol)
	 {
		xw1=x3*cos(upwind_dir)+y3*sin(upwind_dir);
		yw1=-x3*sin(upwind_dir)+y3*cos(upwind_dir);
		xw3=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		yw3=-x2*sin(upwind_dir)+y2*cos(upwind_dir);
		xf2=x1*cos(upwind_dir)+y1*sin(upwind_dir);
		perpendicular_flag=1;
	}
	 else if(upwind_rel < -tol && upwind_rel > -0.5f * pi + tol)
	 {
		xw1=x3*cos(upwind_dir)+y3*sin(upwind_dir);
		yw1=-x3*sin(upwind_dir)+y3*cos(upwind_dir);
		xw2=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		yw2=-x2*sin(upwind_dir)+y2*cos(upwind_dir);
		xf2=x4*cos(upwind_dir)+y4*sin(upwind_dir);
		yf2=-x4*sin(upwind_dir)+y4*cos(upwind_dir);
		xw3=x1*cos(upwind_dir)+y1*sin(upwind_dir);
		yw3=-x1*sin(upwind_dir)+y1*cos(upwind_dir);
		perpendicular_flag=0;
	}
	 else if(upwind_rel < -0.5f * pi + tol && upwind_rel > -0.5f * pi - tol)
	 {
		xw1=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		yw1=-x2*sin(upwind_dir)+y2*cos(upwind_dir);
		xw3=x1*cos(upwind_dir)+y1*sin(upwind_dir);
		yw3=-x1*sin(upwind_dir)+y1*cos(upwind_dir);
		xf2=x3*cos(upwind_dir)+y3*sin(upwind_dir);
		perpendicular_flag=1;
	}
	 else if(upwind_rel < -0.5f * pi - tol && upwind_rel > -pi + tol)
	 {
		xw1=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		yw1=-x2*sin(upwind_dir)+y2*cos(upwind_dir);
		xw2=x1*cos(upwind_dir)+y1*sin(upwind_dir);
		yw2=-x1*sin(upwind_dir)+y1*cos(upwind_dir);
		xf2=x3*cos(upwind_dir)+y3*sin(upwind_dir);
		yf2=-x3*sin(upwind_dir)+y3*cos(upwind_dir);
		xw3=x4*cos(upwind_dir)+y4*sin(upwind_dir);
		yw3=-x4*sin(upwind_dir)+y4*cos(upwind_dir);
		perpendicular_flag=0;
	}
	 else
	 {
		xw1=x1*cos(upwind_dir)+y1*sin(upwind_dir);
		yw1=-x1*sin(upwind_dir)+y1*cos(upwind_dir);
		xw3=x4*cos(upwind_dir)+y4*sin(upwind_dir);
		yw3=-x4*sin(upwind_dir)+y4*cos(upwind_dir);
		xf2=x2*cos(upwind_dir)+y2*sin(upwind_dir);
		perpendicular_flag=1;
	 }
	 build->Leff = build->width * build->length / abs(yw3-yw1);
	 if(perpendicular_flag == 1)
		build->Weff = build->width * build->length / abs(xf2-xw1);
	 else
		build->Weff = build->width * build->length / abs(xf2-xw2);
	 return;
}