#pragma once

#include "Triangle.h"
#include <vector>

#define GETMIN(x,y) ( (x) < (y) ? (x) : (y))
#define GETMAX(x,y) ( (x) > (y) ? (x) : (y))

using std::vector;

class BVH
{
private:
	BVH* leftBox;
	BVH* rightBox;

	bool isLeaf;
	Triangle tri;
	static void mergeSort(std::vector<BVH *>& list, const int type);

public:
	float xmin, xmax, ymin, ymax, zmin, zmax;

	BVA(BVA* l, BVA* r);
	BVA(Triangle t);
	BVH(std::vector<BVH *> m, int height);

	float heightToTri(float x, float y);

	/*
	method that creates a BVA structure from a vector of models
	*/
	static BVH* createBVH(const std::vector<Triangle> tris);
};
