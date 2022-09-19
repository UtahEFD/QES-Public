/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

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
    for (int k = 0; k < nNodes; k++) {
      polygonVertices[k].x_poly = xVertex[k];
      polygonVertices[k].y_poly = yVertex[k];
    }
    polygonVertices[nNodes].x_poly = xVertex[0];
    polygonVertices[nNodes].y_poly = yVertex[0];

    // This will now be process for ALL buildings...
    // extract the vertices from this definition here and make the
    // poly building...
    // setPolybuilding(WGD->nx, WGD->ny, WGD->dx, WGD->dy, WGD->u0, WGD->v0, WGD->z);
  }
};
