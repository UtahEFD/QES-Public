/* File: Matrix3.h
 * Author: Matthew Overby
 */

#ifndef SLUI_MATRIX3_H
#define SLUI_MATRIX3_H

#include "vector3D.h"
#include "vector4D.h"

namespace SLUI {

class Matrix3 {

	public:
		/** @brief Constructor
		* 
		* Creates a 3x3 matrix with the structure:
		* 	[ a.x, a.y, a.z ]
		* 	[ b.x, b.y, b.z ]
		* 	[ c.x, c.y, c.z ]
		*/
		Matrix3(vector3D, vector3D, vector3D);

		/** @brief Get determinant
		* @return the float value of the determinant
		*/
		float getDeter();

		/** @brief Get determinant
		* @return the float value of the determinant for a 2x2 (vector4D) matrix
		*/
		float getDeter2x2(vector4D);

		/** @brief Get transpose
		* @return the 3x3 transpose of the matrix
		*/
		Matrix3 getTranspose();

		/** @brief Get identity
		* @return 3x3 identity matrix
		*/
		Matrix3 getIdentity();

		/** @brief Get inverse
		* @return the 3x3 inverse of the matrix
		*/
		Matrix3 getInverse();

		/** @brief Sets all elements to zero
		*/
		void setZero();

		/** @brief Overloaded operator for equal
		* @return address of copied matrix
		*/
		Matrix3& operator=  (Matrix3&);

		/** @brief Overloaded operator for plus
		* @return 3x3 matrix that is the sum of two matrices
		*/
		Matrix3 operator+ (Matrix3&);

		/** @brief Overloaded operator for minus
		* @return 3x3 matrix that is the result of subtracting two matrices
		*/
		Matrix3 operator- (Matrix3&);

		/** @brief Overloaded operator for multiply
		* @return 3x3 matrix that is the result of the product
		*
		* Multiplies all matrix elements by the specified float
		*/
		Matrix3 operator* (float);

		/** @brief Overloaded operator for multiply
		* @return 3x3 matrix that is the result of the product
		*
		* Multiplies all matrix elements by another matrix
		*/
		Matrix3 operator* (Matrix3&);

		/** @brief Prints the current matrix
		*/
		void printMatrix();

	private:
		vector3D a;
		vector3D b;
		vector3D c;
};

}

#endif

