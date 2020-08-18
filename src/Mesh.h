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
