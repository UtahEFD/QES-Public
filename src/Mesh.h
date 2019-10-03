#pragma once

/*
 * This class serves as a container for a BVH of triangles that 
 * represents a connected collection of Triangles
 */

#include "Triangle.h"
#include "BVH.h"

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

};
