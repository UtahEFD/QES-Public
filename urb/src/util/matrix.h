/**
*	Author: Andrew Larson <lars2865@d.umn.edu>
* Reason: Do rotations for the viewer class to rotate in a 3D space.
*/

#ifndef INC_MATRIX_H
#define INC_MATRIX_H 1

/*--------------------------------------------------------------*/
//
//	This file and matrix.cpp represent a matrix. This class is 
//	not finished, but provides a little functionality for a 
//	specific purpose.
//
/*--------------------------------------------------------------*/

#include "util/point.h" // Get the typedef for Vector3D

namespace sivelab
{
  class normal;
}

using namespace sivelab;

class matrix 
{
  public:
		matrix();
		matrix(unsigned int rows, unsigned int columns);
		matrix(float vec[], unsigned int vec_len);
		matrix(const matrix& m);
		~matrix();

		int getRows();
		int getCols();
		float* getMtrx();

		float operator() (int index);
		float& operator() (int rIndex, int cIndex) const;
		float& operator() (int rIndex, int cIndex);
		matrix& operator= (const matrix& m);
		
		point operator* (point v);
		normal operator* (normal n);

	private:
		unsigned int rows;
		unsigned int cols;
		float * mtrx;
};

matrix makeRotation(normal n, float a);
matrix makeRotation(float a, normal n);

matrix makeRotation(float a, point v);
matrix makeRotation(point v, float a);

#endif
