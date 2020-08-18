#pragma once

/*
 * This class represents a building that is a block with a length width height
 * and origin position. Rectangular buildigns may also have a rotation.
 */

#include "util/ParseInterface.h"
#include "PolyBuilding.h"

#define _USE_MATH_DEFINES
#include <cmath>

using namespace std;
using std::cerr;
using std::endl;
using std::vector;
using std::cout;

class RectangularBuilding : public PolyBuilding
{
protected:
	//int icell_cent, icell_cut;

public:

	RectangularBuilding()
	{

	}

	virtual void parseValues()
	{
            // Extract important XML stuff
            parsePrimitive<float>(true, H, "height");
            parsePrimitive<float>(true, base_height, "baseHeight");
            parsePrimitive<float>(true, x_start, "xStart");
            parsePrimitive<float>(true, y_start, "yStart");
            parsePrimitive<float>(true, L, "length");
            parsePrimitive<float>(true, W, "width");
            parsePrimitive<float>(true, building_rotation, "buildingRotation");

            // We cannot call the constructor since the default had
            // already been called.  So, use other functions in
            // PolyBuilding to create the correct PolyBuilding from
            // height, x_start, y_start, etc...

            // We will now initialize the polybuilding using the
            // protected variables accessible to RectangularBuilding
            // from the PolyBuilding class:
            height_eff = H+base_height;

            // ?????  maybe ignore or hard code here??? until we get a
            // reference back to the Root XML
            //x_start += WID->simParams->halo_x;
            //y_start += WID->simParams->halo_y;

            building_rotation *= M_PI/180.0;
            polygonVertices.resize (5);
            polygonVertices[0].x_poly = polygonVertices[4].x_poly = x_start;
            polygonVertices[0].y_poly = polygonVertices[4].y_poly = y_start;
            polygonVertices[1].x_poly = x_start-W*sin(building_rotation);
            polygonVertices[1].y_poly = y_start+W*cos(building_rotation);
            polygonVertices[2].x_poly = polygonVertices[1].x_poly+L*cos(building_rotation);
            polygonVertices[2].y_poly = polygonVertices[1].y_poly+L*sin(building_rotation);
            polygonVertices[3].x_poly = x_start+L*cos(building_rotation);
            polygonVertices[3].y_poly = y_start+L*sin(building_rotation);

            // This will now be process for ALL buildings...
            // extract the vertices from this definition here and make the
            // poly building...
            // setPolybuilding(WGD->nx, WGD->ny, WGD->dx, WGD->dy, WGD->u0, WGD->v0, WGD->z);
	}

};
