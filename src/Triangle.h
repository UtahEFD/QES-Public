#pragma once

#include "util/ParseInterface.h"
#include "Vector3.h"



#define LOWEST_OF_THREE(x,y,z) ( (x) <= (y) && (x) <= (z) ? (x) : ( (y) <= (x) && (y) <= (z) ? (y) : (z) ) )
#define HIGHEST_OF_THREE(x,y,z) ( (x) >= (y) && (x) >= (z) ? (x) : ( (y) >= (x) && (y) >= (z) ? (y) : (z) ) )

class Triangle : public ParseInterface
{
public:
	Vector3<float> *a, *b, *c;

	Triangle()
	{
		a = b = c = 0;
	}

	Triangle(Vector3<float> aN, Vector3<float> bN, Vector3<float> cN)
	{
		a = new Vector3<float>(aN);
		b = new Vector3<float>(bN);
		c = new Vector3<float>(cN);
	}

	float getHeightTo(float x, float y);

	void getBoundaries(float& xmin, float& xmax, float& ymin, float& ymax, float& zmin, float& zmax);

	virtual void parseValues();
};
