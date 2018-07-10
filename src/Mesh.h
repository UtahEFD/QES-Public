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
		float min = 999999999.0f;
		for (int i = 0; i < tris.size(); i++)
		{
			if ( (*(tris[i]->a))[2] > 0 && (*(tris[i]->a))[2] < min )
				min = (*(tris[i]->a))[2];
			if ( (*(tris[i]->b))[2] > 0 && (*(tris[i]->b))[2] < min )
				min = (*(tris[i]->b))[2];
			if ( (*(tris[i]->c))[2] > 0 && (*(tris[i]->c))[2] < min )
				min = (*(tris[i]->c))[2];
		}
		for (int i = 0; i < tris.size(); i++)
		{
			(*(tris[i]->a))[2] -= min;
			(*(tris[i]->b))[2] -= min;
			(*(tris[i]->c))[2] -= min;
		}

		this->tris = BVH::createBVH(tris);
	}

	float getHeight(float x, float y);

};