#pragma once

#include "Triangle.h"
#include "BVH.h"

using std::vector;

class Mesh
{
public:
	BVH* tris;

	Mesh(vector<Triangle*> tris)
	{

		this->tris = BVH::createBVH(tris);
	}

	float getHeight(float x, float y);

};