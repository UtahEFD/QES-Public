/* File: Matrix4.h
 * Author: Matthew Overby
 */

#ifndef SLUI_MATRIX4_H
#define SLUI_MATRIX4_H

#include "Matrix3.h"

namespace SLUI {

class Matrix4 {

	public:
		/** @brief Default Constructor
		*/
		Matrix4();

		/** @brief Constructor
		* 
		* Creates a 4x4 matrix with the structure:
		* 	[ a.x, a.y, a.z, a.w ]
		* 	[ b.x, b.y, b.z, b.w ]
		* 	[ c.x, c.y, c.z, c.w ]
		* 	[ d.x, d.y, d.z, d.w ]
		*/
		Matrix4(vector4D, vector4D, vector4D, vector4D);

		/** @brief Get determinant
		* @return the float value of the determinant
		*/
		float getDeter();

		/** @brief Get inverse of a 2x2 matrix (vector4D)
		* @return the vector4D inverse
		*/
		vector4D getInverse2x2(vector4D);

		/** @brief Get transpose
		* @return the 4x4 transpose of the matrix
		*/
		Matrix4 getTranspose();

		/** @brief Get identity
		* @return 4x4 identity matrix
		*/
		Matrix4 getIdentity();

		/** @brief Get inverse
		* @return the 4x4 inverse of the matrix
		*/
		Matrix4 getInverse();

		/** @brief Sets all elements to zero
		*/
		void setZero();

		/** @brief Overloaded operator for equal
		* @return address of copied matrix
		*/
		Matrix4& operator=  (Matrix4&);

		/** @brief Overloaded operator for plus
		* @return 4x4 matrix that is the sum of two matrices
		*/
		Matrix4 operator+ (Matrix4&);

		/** @brief Overloaded operator for minus
		* @return 4x4 matrix that is the result of subtracting two matrices
		*/
		Matrix4 operator- (Matrix4&);

		/** @brief Overloaded operator for multiply
		* @return 4x4 matrix that is the result of the product
		*
		* Multiplies all matrix elements by the specified float
		*/
		Matrix4 operator* (float);

		/** @brief Overloaded operator for multiply
		* @return 4x4 matrix that is the result of the product
		*
		* Multiplies all matrix elements by another matrix
		*/
		Matrix4 operator* (Matrix4&);

		/** @brief Prints the current matrix
		*/
		void printMatrix();

		vector4D a;
		vector4D b;
		vector4D c;
		vector4D d;

	private:
};

}

#endif

