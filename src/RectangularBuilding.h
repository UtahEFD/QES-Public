#pragma once

#include "ParseInterface.h"
#include "NonPolyBuilding.h"

#define CELL(i,j,k,sub) ((i) + (j) * ((nx) - (sub)) + (k) * ((nx) - (sub)) * ((ny) - (sub)))

class RectangularBuilding : public NonPolyBuilding
{
private:


public:

	RectangularBuilding()
	{
		buildingGeometry = 1;
		Lf = -999;
		Leff = 0;
		Weff = 0;
	}

	RectangularBuilding(float xfo, float yfo, float bh, float dx, float dy, float dz)
	{
		buildingGeometry = 1;
		Lf = -999;
		Leff = 0;
		Weff = 0;

		groupID = 999;
		buildingType = 5;
		height = dz;
		baseHeight = bh;
		centroidX = xfo + 0.5f * dx;
		centroidX = yfo + 0.5f * dy;
		xFo = xfo;
		yFo = yfo;
		length = dx;
		width = dy;
		rotation = 0.0f;
		
		Wt = 0.5 * width;
		Lt = 0.5 * length;
	}

	virtual void parseValues()
	{
		parsePrimitive<int>(true, groupID, "groupID");
		parsePrimitive<int>(true, buildingType, "buildingType");
		parsePrimitive<float>(true, height, "height");
		parsePrimitive<float>(true, baseHeight, "baseHeight");
		parsePrimitive<float>(true, centroidX, "centroidX");
		parsePrimitive<float>(true, centroidY, "centroidY");
		parsePrimitive<float>(true, xFo, "xFo");
		parsePrimitive<float>(true, yFo, "yFo");
		parsePrimitive<float>(true, length, "length");
		parsePrimitive<float>(true, width, "width");
		parsePrimitive<float>(true, rotation, "rotation");
		parsePrimitive<int>(false, buildingDamage, "buildingDamage");
		parsePrimitive<float>(false, atten, "atten");
		Wt = 0.5 * width;
		Lt = 0.5 * length;
	}

	void setBoundaries(float dx, float dy, float dz, int nz, float *zm)
	{
		iStart = xFo / dx + 1;  
		iEnd = (xFo + length) / dx;  
		jEnd = (yFo + width) / dy;  
		jStart = (yFo - width) / dy + 1;
/*		if (buildingDamage != 2)
		{
			for (int i = 1; i < nz - 1; i++)
			{
				kStart = i;
				if (baseHeightActual <= zm[i])
					break; 
			}
			for (int i = kStart; i < nz - 1; i++)
			{
				kEnd =i;
				if (height < zm[i + 1])
					break;
			}
		}*/
		kStart = baseHeight / dz;
		kEnd = kStart + (height / dz);
	}





/*
Note: The select case portion is almost identical across geometry types 1, 2, and 6.
Difference is that 6 has one more case, type 5. similar to default except sets to 1
instead of 0. Also same as 
*/

	void setCells(int nx, int ny, int nz, int *icellflag, int *ibldflag, int ibuild) 
	{
		if(!rotation)
		{

		 for (int j = jStart; j <= jEnd; j++)
		 {
		    for (int i = iStart; i <= iEnd; i++)
		     {
		       switch (buildingType)
		       {
		          case 0:
		         	for ( int k = kStart; k <= kEnd; k++)
		         	{
		             icellflag[CELL(i,j,k,1)] = 1;
		             ibldflag[CELL(i,j,k,1)] = ibuild; //take in as parameter maybe?
		         	}
		         	break;
		          case 2:
		             for( int k = kStart; k <= kEnd; k++)
		             /*{
		                if( icellflag[i][j][k] != 0)
		                {
		                   if(lu_canopy_flag > 0)
		                   {
		                      if(canopy_top[i][j] == landuse_height[i][j])
		                      {
		                         if(k == 1) 
		                         	for (int c = 0; c < nz; c++)
		                         		canopy_atten[i][j][c] = 0.0f;
		                         if( height < 0.5f * dz_array[0])
		                            canopy_top[i][j] = 0.0f;
		                         else
		                         {
		                            canopy_top(i,j)=Ht(ibuild)
		                            canopy_atten(i,j,k)=atten(ibuild)
		                         }
		                      }
		                      else if( height > canopy_top[i][j])
		                         canopy_top[i][j] = height;
		                   }
		                   else
		                   {
		                      if(height > canopy_top[i][j])
		                         canopy_top[i][j] = height;
		                      canopy_atten[i][j][k] = atten;
		                   }
		                }
		             }*/
		             break;
		          case 3:
		             /* I don't really understand this part right now. ilevel and ceiling haven't been brought up before.
		             ilevel=0
		             do k=kstart(ibuild),kend(ibuild)
		                ilevel=ilevel+1
		                if(ilevel/2 .ne. ceiling(0.5*real(ilevel)))cycle
		                icellflag(i,j,k)=0
		                ibldflag(i,j,k)=ibuild
		             enddo*/
		             break;                             
		          default:
		         	for ( int k = kStart; k <= kEnd; k++)
		         	{
		             icellflag[CELL(i,j,k,1)] = 0;
		             ibldflag[CELL(i,j,k,1)] = ibuild; //take in as parameter maybe?
		         	}
		       }
		    }
		 }
		}
	}
};
/*                else
                {

//! calculate corner coordinates of the building
                     float x1 = xfo + width * sin(gamma);
                     float y1 = yfo(ibuild)-Wt(ibuild)*cos(gamma(ibuild))
                     float x2 = x1+Lti(ibuild)*cos(gamma(ibuild))
                     float y2 = y1+Lti(ibuild)*sin(gamma(ibuild))
                     float x4 = xfo(ibuild)-Wt(ibuild)*sin(gamma(ibuild))
                     float y4 = yfo(ibuild)+Wt(ibuild)*cos(gamma(ibuild))
                     float x3 = x4+Lti(ibuild)*cos(gamma(ibuild))
                     float y3 = y4+Lti(ibuild)*sin(gamma(ibuild))
 271                 format(8f8.3)
                     if(gamma(ibuild).gt.0)then
                        xmin=x4
                        xmax=x2
                        ymin=y1
                        ymax=y3
                     endif
                     if(gamma(ibuild).lt.0)then
                        xmin=x1
                        xmax=x3
                        ymin=y2
                        ymax=y4
                     endif
                     istart(ibuild)=nint(xmin/dx)
                     iend(ibuild)=nint(xmax/dx)
                     jstart(ibuild)=nint(ymin/dy)
                     jend(ibuild)=nint(ymax/dy)
!erp  do k=int(zfo(ibuild)),kend(ibuild)  
!erp        do j=int(ymin),int(ymax)
!erp     do i=int(xmin),int(xmax)
!erp     x_c=real(i) !x coordinate to be checked
!erp     y_c=real(j) !y coordinate to be checked
! changed int to nint in next three lines 8-14-06
                     do j=nint(ymin/dy)+1,nint(ymax/dy)+1   !convert back to real world unit, TZ 10/29/04
                        do i=nint(xmin/dx)+1,nint(xmax/dx)+1   !convert back to real world unit, TZ 10/29/04
                           x_c=(real(i)-0.5)*dx !x coordinate to be checked   !convert back to real world unit, TZ 10/29/04
                           y_c=(real(j)-0.5)*dy !y coordinate to be checked   !convert back to real world unit, TZ 10/29/04
!calculate the equations of the lines making up the 4 walls of the
!building
						   if( x4 .eq. x1)x4=x4+.0001
                           slope = (y4-y1)/(x4-x1) !slope of L1
                           xL1 = x4 + (y_c-y4)/slope
                           if( x3 .eq. x2)x3=x3+.0001
                           slope = (y3-y2)/(x3-x2) !slope of L2
                           xL2 = x3 + (y_c-y3)/slope
                           if( x2 .eq. x1)x2=x2+.0001
                           slope = (y2-y1)/(x2-x1) !slope of L3
                           yL3 = y1 + slope*(x_c-x1)
                           if( x3 .eq. x4)x3=x3+.0001
                           slope = (y3-y4)/(x3-x4) !slope of L4
                           yL4 = y4 + slope*(x_c-x4)
                           if(x_c.gt.xL1.and.x_c.lt.xL2.and.y_c.gt.yL3.and.y_c.lt.yL4)then
                              select case(bldtype(ibuild))
                                 case(0)
                                    icellflag(i,j,kstart(ibuild):kend(ibuild))=1
                                    ibldflag(i,j,kstart(ibuild):kend(ibuild))=ibuild
                                 case(2)
                                    do k=kstart(ibuild),kend(ibuild)
                                       if(icellflag(i,j,k) .ne. 0)then
                                          if(lu_canopy_flag .gt. 0)then
                                             if(canopy_top(i,j) .eq. landuse_height(i,j))then
                                                if(k .eq. 2)canopy_atten(i,j,:)=0.
                                                if(Ht(ibuild) .lt. 0.5*dz_array(1))then
                                                   canopy_top(i,j)=0.
                                                else
                                                   canopy_top(i,j)=Ht(ibuild)
                                                   canopy_atten(i,j,k)=atten(ibuild)
                                                endif
                                             elseif(Ht(ibuild) .gt. canopy_top(i,j))then
                                                canopy_top(i,j)=Ht(ibuild)
                                             endif
                                          else
                                             if(Ht(ibuild) .gt. canopy_top(i,j))then
                                                canopy_top(i,j)=Ht(ibuild)
                                             endif
                                             canopy_atten(i,j,k)=atten(ibuild)
                                          endif
                                       endif
                                    enddo
                                 case(3)
                                    ilevel=0
                                    do k=kstart(ibuild),kend(ibuild)
                                       ilevel=ilevel+1
                                       if(ilevel/2 .ne. ceiling(0.5*real(ilevel)))cycle
                                       icellflag(i,j,k)=0
                                       ibldflag(i,j,k)=ibuild
                                    enddo                                 
                                 case default
                                    icellflag(i,j,kstart(ibuild):kend(ibuild))=0
                                    ibldflag(i,j,kstart(ibuild):kend(ibuild))=ibuild
                              endselect
                           endif
                        enddo
                     enddo
                  endif
! generate cylindrical buildings
! need to specify a and b as the major and minor axis of
! the ellipse
! xco and yco are the coordinates of the center of the ellipse
               }*/
