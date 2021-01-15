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
 * This class is a Bounding Volume Hierarchy data structure. This
 * organizes Triangles spacially allowing for fast access based on location.
 */

#include <vector>

#include "Triangle.h"

#define GETMIN(x,y) ( (x) < (y) ? (x) : (y))
#define GETMAX(x,y) ( (x) > (y) ? (x) : (y))



using std::vector;

class BVH
{
private:
	BVH* leftBox;
	BVH* rightBox;

	bool isLeaf;
	Triangle* tri;


	/*
	 * This function sorts Bounding Boxes by recursively dividing them
	 * apart, and then putting them in order and then merging the lists.
	 *
	 * @param list -the list of bounding boxes that should be sorted
	 * @param type -identifier for what dimension the boxes are being sorted by
	 */
	static void mergeSort(std::vector<BVH *>& list, const int type);

public:
	float xmin, xmax, ymin, ymax, zmin, zmax;

	/*
	 * Creates a bounding box encasing two child bounding boxes
	 *
	 * @param l -left child box
	 * @param r -right child box
	 */
	BVH(BVH* l, BVH* r);

	/*
	 * Creates a bounding box encasing a triangle, this marks the box
	 * as a leaf meaning it is at the bottom of the tree.
	 *
	 * @param t -The triangle to be put in the heirarchy
	 */
	BVH(Triangle* t);

	/*
	 * Creates a bounding volume heirarchy from a list of bounding boxes.
	 * height is used to determine the spacial ordering.
	 *
	 * @param m -list of bounding boxes
	 * @param height -current depth in the tree
	 */
	BVH(std::vector<BVH *> m, int height);

	/*
	 * Takes a point in the x y plane and finds what triangle is directly above
	 * it. It returns how many meters exist between the point and the ground
	 *
	 * @param x -x position
	 * @param y -y position
	 * @return distance from the point to the triangle directly above it
	 */
	float heightToTri(float x, float y);

	/*
	 *method that creates a BVA structure from a vector of models
	 *
	 * @param tris -list of triangles that will be placed in the structure
	 */
	static BVH* createBVH(const std::vector<Triangle*> tris);

};
