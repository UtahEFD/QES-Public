#pragma once

#include "Vector3.h"

#define G 0.0f
#define H 0.0f
#define I 1.0f

#define LOWEST_OF_THREE(x,y,z) ( (x) <= (y) && (x) <= (z) ? (x) : ( (y) <= (x) && (y) <= (z) ? (y) : (z) ) )
#define HIGHEST_OF_THREE(x,y,z) ( (x) >= (y) && (x) >= (z) ? (x) : ( (y) >= (x) && (y) >= (z) ? (y) : (z) ) )

class Triangle
{
public:
	Vector3<float> a,b,c;
	Traingle(Vector3<float> aN, Vector3<float> bN, Vector3<float> cN);
	{
		a = aN;
		b = bN;
		c = cN;
	}

	float getHeightTo(float x, float y);

	void getBoundaries(float& xmin, float& xmax, float& ymin, float& ymax, float& zmin, float& zmax);
};
