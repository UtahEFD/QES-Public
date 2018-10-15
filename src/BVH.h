#pragma once

#include "Triangle.h"
#include <vector>

#define GETMIN(x,y) ( (x) < (y) ? (x) : (y))
#define GETMAX(x,y) ( (x) > (y) ? (x) : (y))

using std::vector;
<<<<<<< HEAD

=======
/*
 *This class is a bounding volume hierarchy for triangles. BVHs are
 *data structures where objects are stored spacially, this allows for 
 *fast access (log2(n)) time to any individual triangle. Objects are stored
 *in a tree structure where intermediate nodes bound the area covered by the
 *leaves.
 */
>>>>>>> origin/doxygenAdd
class BVH
{
private:
	BVH* leftBox;
	BVH* rightBox;

	bool isLeaf;
	Triangle* tri;
<<<<<<< HEAD
=======

	/*
	 *Sorts a BVH based on the criteria given in type. This lets us
	 *sort by x dimension on one tier, and y on the next which leads to
	 *better seperation and less search time.
	 *@param list -a list of all BVH that are being sorted
	 *@param type -identifier for what dimension the list is being sorted
	 */
>>>>>>> origin/doxygenAdd
	static void mergeSort(std::vector<BVH *>& list, const int type);

public:
	float xmin, xmax, ymin, ymax, zmin, zmax;

<<<<<<< HEAD
	BVH(BVH* l, BVH* r);
	BVH(Triangle* t);
	BVH(std::vector<BVH *> m, int height);

	float heightToTri(float x, float y);

	/*
	method that creates a BVA structure from a vector of models
	*/
=======
	/*
	 *creates a bounding volume hierarchy by taking in two
	 *bounding volumes and assigning them to the left and right
	 *child of this box. The region this contains is the union
	 *of the child boxes.
	 *@param l -BVH to be assigned to the left child
	 *@param r -BVH to be assigned to the right child
	 */
	BVH(BVH* l, BVH* r);

	/*
	 *Creates a BVH leaf node.
	 *@param t -triangle to be encased by the bounding box
	 */
	BVH(Triangle* t);

	/*
	 *Creates a BVH by sorting the a list of BVH leaves, assigning them
	 *to the left and right children. This runs recursively until the leaves
	 *have been hit. Height indicates what level of the tree is being created
	 *which is used for sorting dimensionally.
	 *@param m -list of BVH leaves
	 *@param height -level of the BVH currently being created (root is 0)
	 */
	BVH(std::vector<BVH *> m, int height);

	/*
	 *Queries the BVH with a point in the x y plane and returns the distance
	 *to the farthest triangle that exists in the positive z direction. If the
	 *current BVH is a leaf, it returns the height, if it bounds sub-boxes it
	 *recursively calls this operation on it's children and compares the return values.
	 *@param x -query location in the x direction
	 *@param y -query location in the y direction
	 */
	float heightToTri(float x, float y);

	/*
	 *method that creates a BVH structure from a list of triangles.
	 *@param tris -list of triangles to be enclosed by the BVH
	 *@return -the root of the BVH
	 */
>>>>>>> origin/doxygenAdd
	static BVH* createBVH(const std::vector<Triangle*> tris);
};
