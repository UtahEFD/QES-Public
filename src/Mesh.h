/*
 * QES-Winds
 *
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
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
 *
 */


#pragma once

/*
 * This class serves as a container for a BVH of triangles that
 * represents a connected collection of Triangles
 */

#include "Triangle.h"
#include "BVH.h"

#include <limits>
#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <iostream>

using std::vector;

class Mesh
{
public:
	BVH* tris;

    int mlSampleRate;

        //temp var for Optix
        //OptixRayTrace *optixRayTracer;
        vector<Triangle*> optixTris;

	/*
	 * Creates a BVH out of a list of Triangles
	 *
	 * @param tris -list of triangles.
	 */
	Mesh(vector<Triangle*> tris)
            : mlSampleRate( 100 )
	{
		this->tris = BVH::createBVH(tris);

                //temp var for Optix
                // this->optixRayTracer = new OptixRayTrace(tris);
                       //optixTris = tris;
                //temp var for getting the list of triangles
                       trisList = tris;
        }

	/*
	 * Gets the height from a location on the xy plane
	 * to a triangle in the BVH
	 *
	 * @param x -x position
	 * @param y -y position
	 * @return distance to the triangle directly above the point.
	 */
	float getHeight(float x, float y);

  private:
        //temporary variable for getting the list of Triangles though the mesh
        std::vector<Triangle *> trisList;

};
