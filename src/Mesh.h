#pragma once

/*
 * This class serves as a container for a BVH of triangles that 
 * represents a connected collection of Triangles
 */

#include "Triangle.h"
#include "BVH.h"
#include "SphereDirections.h"
#include "Ray.h"
#include "HitRecord.h"
#include <limits>

using std::vector;

class Mesh
{
public:
	BVH* tris;

	/*
	 * Creates a BVH out of a list of Triangles
	 *
	 * @param tris -list of triangles.
	 */
	Mesh(vector<Triangle*> tris)
	{

		this->tris = BVH::createBVH(tris);
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


        /*
         *Caculates the mixing length for all fluid objects 
         */
        std::vector<float> calculateMixingLength(int dimX, int dimY, int dimZ, float dx, float dy, float dz, const std::vector<int> &icellflag);
        
};
