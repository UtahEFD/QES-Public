#pragma once

#include "Vector3.h"
#include <vector>

/* 
 * The Edge class contains 2 Vector3<float>s. This represents
 * a connection between these two points.
 */
template <class T>
class Edge
{
private:
	std::vector< T > values;
public:
	
	/*
	 *Default constructor, both items initialized as 0. 
	 */
	Edge();

	/*
	 *This constructor takes in two values and sets them
	 *as values 0 and 1 respectively.
	 *@param a -value to be assigned to point 0
	 *@param b -value to be assigned to point 1
	 */
	Edge(const T a, const T b);

	/*
	 *Operator overload of the [] operator. This
	 *is how data members are accessed. Value of the
	 *input is sanitized for array out of bounds errors.
	 *@param i -index indicating which point should be returned
	 *@return -a reference to the value denoted by i.
	 */
	T& operator[](const int i);

	/*
	 *== comparative operator overload.
	 *this returns true if the values in each edge match.
	 *note: if the calues on the edges are swapped this is still true.
	 *@param e -edge to be compared to this edge.
	 *@return -true if edges are equal, false otherwise. 
	 */
	bool operator==(Edge< T > e);

	/*
	 *Checks to see if value v exists in this edge. If it does,
	 *then it returns the index of that value. If it doesn't this
	 *returns -1
	 *@param v -value to query for index.
	 *@return -the index of the given value, -1 if not found.
	 */
	int getIndex(T v);

};

//this is because this is a template class
#include "Edge.cpp"