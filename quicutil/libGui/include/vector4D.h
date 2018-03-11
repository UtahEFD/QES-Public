/* File: vector4D.h
 * Author: Matthew Overby
 */

#ifndef SLUI_VECTOR4D_H
#define SLUI_VECTOR4D_H

#include <cstdio>
#include <cmath>

namespace SLUI {

class vector4D {

	public:
		/** @brief Default Constructor
		*/
		vector4D();

		/** @brief Creates a new vector with float values (x, y, z, w)
		*/
		vector4D(float, float, float, float);

		/** @brief Prints out the vector
		*/
		void printVec();

		/** @brief Overloaded == operator
		* @return true if vectors are equal, false otherwise
		*
		* Compares two vectors, and if their values are
		* within 0.0001 away from eachother, returns true. 
		*/
		bool operator== (vector4D&);

		/** @brief Overloaded != operator
		* @return true if vectors are not equal, false otherwise
		*
		* Compares two vectors, and if their values are
		* within 0.0001 away from eachother, returns false. 
		*/
		bool operator!= (vector4D&);

		/** @brief Overloaded + operator
		* @return vector that is the sum of two vectors
		*/
		vector4D operator+ (vector4D&);

		/** @brief Overloaded += operator
		* @return vector that is the sum of itself and another vector
		*/
		vector4D& operator+= (vector4D&);

		/** @brief Overloaded - operator
		* @return vector that is the difference of two vectors
		*/
		vector4D operator- (vector4D&);

		/** @brief Overloaded -= operator
		* @return vector that is the difference of itself and another vector
		*/
		vector4D& operator-= (vector4D&);

		/** @brief Overloaded * operator
		* @return vector that is the product of a vector and a float value
		*/
		vector4D operator* (float);

		/** @brief Overloaded *= operator
		* @return vector that is the product of itself and a float value
		*/
		vector4D& operator*= (float);

		/** @brief Overloaded * operator
		* @return Dot product (float) of two vectors
		*/
		float operator* (vector4D&);

		/** @brief Overloaded % operator
		* @return Cross product (vector) of two vectors
		*/
		vector4D operator% (vector4D&);

		/** @brief Normalize the vector
		* @return Vector that is the normal of the current vector
		*/
		vector4D normalize() const;

		/** @brief Get zero vector
		* @return Vector whose x, y, z, and w are zero
		*/
		vector4D zeroVec();

		/** @brief Normalize the vector
		* Normalizes the current vector instead of returning a copy
		*/
		void normalizeThis();

		/** @brief Homogenize the vector
		* Homogenizes the current vector instead of returning a copy
		*/
		void homogenizeThis();

		float x;
		float y;
		float z;
		float w;

	private:

};

}

#endif
