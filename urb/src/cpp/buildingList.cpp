#include "buildingList.h"

namespace QUIC
{
	void urbBuildingList::sort()
	{
	  int veggies = 0;
		for(unsigned int j = 0; j < vector<urbBuilding>::size(); j++)
		for(unsigned int i = 0; i < vector<urbBuilding>::size() - j; i++)
		{
                    // No good enums in the header so will use the
                    // hard-coded values for now
                    // 
                    // if(vector<urbBuilding>::operator[](i).type == quBuildings::VEGETATION)
                    if(vector<urbBuilding>::operator[](i).type == 9)
			{
				swapBuildings
				(
				  vector<urbBuilding>::operator[](i), 
				  vector<urbBuilding>::operator[](vector<urbBuilding>::size() - j - 1)
				);
				veggies++;				
			}
			else if(vector<urbBuilding>::operator[](i).height > vector<urbBuilding>::operator[](j).height)
			{
				swapBuildings(vector<urbBuilding>::operator[](i), vector<urbBuilding>::operator[](j));
			}
		}
		// Vegetation at the back.		
	}

  urbBuilding& urbBuildingList::getBuildingAt(int const& i, int const& j, int const& k)
  {
    index3D ndx3d = {i, j, k};
    return this->getBuildingAt(ndx3d);
  }
  
  urbBuilding& urbBuildingList::getBuildingAt(index3D const& ndx3d)
  {
    for(unsigned int n = 0; n < vector<urbBuilding>::size(); n++)
		{
			if(vector<urbBuilding>::operator[](n).inBuildingQ(ndx3d))
			{
				return vector<urbBuilding>::operator[](n);
			}
		}
		
		return vector<urbBuilding>::operator[](0);
  }

  float urbBuildingList::averageHeight() const
  {
    if(vector<urbBuilding>::size() <= 0) {return 0.;}
  
    float sum = 0.;
    for(unsigned i = 0; i < vector<urbBuilding>::size(); i++)
    {
      sum += vector<urbBuilding>::operator[](i).height + vector<urbBuilding>::operator[](i).zfo;
    }
    return sum / vector<urbBuilding>::size(); // assumes there is no vegetation...
  }

  float3 urbBuildingList::getBuildingLocation(int const& buildingIndex) const
  {
    unsigned bndx = buildingIndex % vector<urbBuilding>::size();

		float3 lctn = {-1., -1., -1.};
    if(0 <= bndx && bndx < vector<urbBuilding>::size())
		{
			lctn.x = vector<urbBuilding>::operator[](bndx).xfo;
			lctn.y = vector<urbBuilding>::operator[](bndx).yfo;
			lctn.z = vector<urbBuilding>::operator[](bndx).zfo;
		}
		return lctn;
  }
 	float3 urbBuildingList::getBuildingDimensions(int const& buildingIndex) const
 	{
		unsigned bndx = buildingIndex % vector<urbBuilding>::size();

		float3 dmnsns = {0., 0., 0.};
 	  if(0 <= bndx && bndx < vector<urbBuilding>::size())
		{
			dmnsns.x = vector<urbBuilding>::operator[](bndx).width;
			dmnsns.y = vector<urbBuilding>::operator[](bndx).length;
			dmnsns.z = vector<urbBuilding>::operator[](bndx).height;
		}
		return dmnsns;
 	}
	
 	double urbBuildingList::closestBuilding(index3D const& loc) const
 	{
 	  if(vector<urbBuilding>::size() <= 0)
	  {
	    return (double) FLT_MAX;
	  }
	
	  int i = loc.i;
	  int j = loc.j;
	  int k = loc.k;
	
		// Stores minimum distance of (i,j,k) to each building.
		vector<double> distance;

		// Adding 0.5*gridResolution as the QUIC-URB grid is shifted by 0.5*gridResolution 
		double iCell = i + .5; // converting units in meters (original position of the cell in meters)
		double jCell = j + .5;
		double kCell = k + .5;//TEMPORARY SUBTRACT 1---******************************CHECK THIS*****

		double x, y, l, w, h, minDisFaces, minDisTop;
		double actualDis[4]; // absolute value of the perpendDis or the actual distance, we have 4 faces

		double iedge;  // edges of the suface, declared as doubles as ...
		double jedge;  // one of the edge value for cells perpendicular...
		double kedge;

		int edge1; //right edge of the front plane
		int edge2; //left edge of the front plane

		int edgeX1;
		int edgeX2;
		int edgeY1;
		int edgeY2;

		for(unsigned bndx = 0; bndx < vector<urbBuilding>::size(); ++bndx)
		{
		  urbBuilding bld = vector<urbBuilding>::operator[](bndx);
			x = bld.xfo; //storing the building parameters 
			y = bld.yfo;
			l = bld.length;
			w = bld.width;
			h = bld.height;

			minDisFaces = 0.; //minimum distance to 4 faces(sides) from a cell
			minDisTop   = 0.;//minimum distance to the top(roof) of the building from a cell 

			if(kCell < h) //For this condition we have only 4 planes for each building
			{
				for(int i = 0; i < 4; ++i)
				{
					// i=0 is front face 
					// i=1, back face
					// i=2, right side face (facing towards front face of the building)
					// i=3, left side face:

					if(i == 0) //front face
					{
						edge1 = (int)(y - (w/2)); 
						edge2 = (int)(y + (w/2)); 

						jedge = x; // to get the edge in X-Direction
						if(jCell <= edge1 || jCell >= edge2) //for cells (i,j,qk) off the plane
						{
							iedge = (fabs(edge2 - jCell) < fabs(edge1 - jCell)) ? edge2 : edge1;
						}
						else //for cells perpendicular to the faces
						{
							iedge = jCell;
						}
						actualDis[i] = pow( (pow((iCell - jedge), 2.)) + (pow((jCell - iedge), 2.)), .5 );
					}

					if(i == 1) //back face
					{
						edge1 = (int)(y - (w/2));
						edge2 = (int)(y + (w/2));
						jedge = x + l; //back face
						if(jCell<edge1 || jCell>edge2)
						{ 
							iedge = (fabs(edge2 - jCell) < fabs(edge1 - jCell)) ? edge2 : edge1;
						}
						else 
						{
							iedge = jCell;
						}
						actualDis[i] = pow( (pow((iCell - jedge), 2.)) + (pow((jCell - iedge), 2.)), .5 );
					}

					if(i==2) //right side face
					{
						edge1 = (int)(x);
						edge2 = (int)(x + l);
						iedge = y - (w/2);
						if(iCell>edge2 || iCell<edge1)
						{
							jedge = (fabs(edge1 - iCell) < fabs(edge2 - iCell)) ? edge1 : edge2;
						}
						else
						{
							jedge = iCell;
						}
						actualDis[i] = pow( (pow((iCell - jedge), 2.)) + (pow((jCell - iedge), 2.)), .5 );
					}
					if(i==3) // left side face
					{
						edge1 = (int)(x);
						edge2 = (int)(x + l);
						iedge = y + (w/2);
						if(iCell > edge2 || iCell < edge1)
						{
							jedge = (fabs(edge1-iCell) < fabs(edge2-iCell)) ? edge1 : edge2;
						}
						else
						{
							jedge = iCell;
						}
						actualDis[i]=pow( (pow((iCell-jedge),2.)) + (pow((jCell-iedge),2.)) , .5 );
					}
				} // End: for Loop for number of faces

				minDisFaces = actualDis[1]; //assuming one is minimum

				for(int i = 0; i < (int)(sizeof(actualDis) / sizeof(*actualDis)); ++i)   //sizeof() provide number of bytes
				{
					if(minDisFaces > actualDis[i]) {minDisFaces = actualDis[i];}
				}

				// checking if ground is closer than any of the faces
				if(minDisFaces > kCell) {minDisFaces = kCell;}

				distance.push_back(minDisFaces);
			}
			else //if k >= h
			{
				edgeX1 = (int)(x);
				edgeX2 = (int)(x + l);
				edgeY1 = (int)(y - (w/2));
				edgeY2 = (int)(y + (w/2));

				kedge = h;

				if(iCell < edgeX1 || iCell > edgeX2 || jCell < edgeY1 || jCell > edgeY2) // for all the off plane cells (areas B0 and B1 in the PPT)
				{
					iedge = jCell;

					if(iCell <= edgeX1) // cells in front of front face
					{
						jedge = edgeX1;
						if(jCell < edgeY1) iedge = edgeY1;
						if(jCell > edgeY2) iedge = edgeY2;   
					}
					if(iCell >= edgeX2) //cells behind the back face
					{
						jedge = edgeX2;
						if(jCell <= edgeY1) iedge = edgeY1;
						if(jCell >  edgeY2) iedge = edgeY2;    
					}
					if(iCell > edgeX1 && iCell < edgeX2) //cells  on either side of side faces
					{
						jedge = iCell;
						if(jCell <= edgeY1) iedge = edgeY1;
						if(jCell >  edgeY2) iedge = edgeY2;
					}	
				}
				else //if the prependicular from the cell lies on the roof.
				{
					iedge = jCell;
					jedge = iCell;
				}	  

				minDisTop = pow( (pow((iCell - jedge), 2.)) + (pow((jCell - iedge), 2.)) + (pow((kCell - kedge), 2.)), .5 );	

				// checking if ground is closer than the distance to the roof.
				if(minDisTop > kCell) {minDisTop = kCell;}

				distance.push_back(minDisTop);
			} // End: if else of k > h or k < h
		} // End: for loop for buildings

		std::sort(distance.begin(),distance.end());

		return distance[0];// returning smallest distance
 	}

//Protected or Private  
 	void urbBuildingList::swapBuildings(urbBuilding& b1, urbBuilding& b2)
	{
		urbBuilding tmp = b1;
		b1 = b2;
		b2 = tmp;
		// Swapped.
	}
}
