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

class PolygonQUICBuilding : public PolyBuilding
{
protected:
  // int icell_cent, icell_cut;

public:
  PolygonQUICBuilding()
  {
  }

  virtual void parseValues()
  {
    std::vector<float> xVertex, yVertex;

    // Extract important XML stuff
    parsePrimitive<float>(true, H, "height");
    parsePrimitive<float>(true, base_height, "baseHeight");
    parseMultiPrimitives<float>(true, xVertex, "xVertex");
    parseMultiPrimitives<float>(true, yVertex, "yVertex");

    // parsePrimitive<float>(true, x_start, "xStart");
    // parsePrimitive<float>(true, y_start, "yStart");
    // parsePrimitive<float>(true, L, "length");
    // parsePrimitive<float>(true, W, "width");
    // parsePrimitive<float>(true, building_rotation, "buildingRotation");

    // We cannot call the constructor since the default had
    // already been called.  So, use other functions in
    // PolyBuilding to create the correct PolyBuilding from
    // height, x_start, y_start, etc...

    x_start = 0;
    y_start = 0;
    L = 0;
    W = 0;
    building_rotation = 0;

    // We will now initialize the polybuilding using the
    // protected variables accessible to RectangularBuilding
    // from the PolyBuilding class:
    height_eff = H + base_height;

    // ?????  maybe ignore or hard code here??? until we get a
    // reference back to the Root XML
    // x_start += WID->simParams->halo_x;
    // y_start += WID->simParams->halo_y;

    int nNodes = xVertex.size();
    polygonVertices.resize(nNodes + 1);
    for (auto k = 0u; k < nNodes; k++) {
      polygonVertices[k].x_poly = xVertex[k];
      polygonVertices[k].y_poly = yVertex[k];
    }
    polygonVertices[nNodes].x_poly = xVertex[0];
    polygonVertices[nNodes].y_poly = yVertex[0];

    x_start = 0;
    y_start = 0;
    L = 0;
    W = 0;

    // This will now be process for ALL buildings...
    // extract the vertices from this definition here and make the
    // poly building...
    // setPolybuilding(WGD->nx, WGD->ny, WGD->dx, WGD->dy, WGD->u0, WGD->v0, WGD->z);
  }
};
