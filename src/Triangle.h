#pragma once

/*
 * This clas represents a triangle being made up of 3 points each
 * with an x,y,z location
 */

#include "util/ParseInterface.h"
#include "Vector3.h"
#include <cmath>


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

	/*
	 * uses a vertical ray cast from point x y at height 0 with barycentric interpolation to
	 * determine if the ray hits inside this triangle.
	 *
	 * @param x -x location
	 * @param y -y location
	 * @return the length of the ray before intersection, if no intersection, -1 is returned
	 */
	float getHeightTo(float x, float y);


	/*
	 * gets the minimum and maximum values in the x y and z dimensions
	 *
	 * @param xmin -lowest value in the x dimension
	 * @param xmax -highest value in the x dimension
	 * @param ymin -lowest value in the y dimension
	 * @param ymax -highest value in the y dimension
	 * @param zmin -lowest value in the z dimension
	 * @param zmax -highest value in the z dimension
	 */
	void getBoundaries(float& xmin, float& xmax, float& ymin, float& ymax, float& zmin, float& zmax);


	virtual void parseValues();
};
