/* File: vector3D.h
 * Author: Matthew Overby
 */

#ifndef SLUI_VECTOR3D_H
#define SLUI_VECTOR3D_H
#define vzero vector3D(0.0f, 0.0f, 0.0f);

#include <cstdio>
#include <iostream>
#include <cmath>

namespace SLUI {

class vector3D {

	public:
		/** @brief Default Constructor
		*/
		vector3D();

		/** @brief Creates a new vector with float values (x, y, z)
		*/
		vector3D(float, float, float);

		/** @brief Default Destructor
		*/
		~vector3D();

		/** @brief Prints out the vector
		*/
		void printVec();

		/** @brief Overloaded == operator
		* @return true if vectors are equal, false otherwise
		*
		* Compares two vectors, and if their values are
		* within 0.0001 away from eachother, returns true. 
		*/
		bool operator== (vector3D&);

		/** @brief Overloaded != operator
		* @return true if vectors are not equal, false otherwise
		*
		* Compares two vectors, and if their values are
		* within 0.0001 away from eachother, returns false. 
		*/
		bool operator!= (vector3D&);

		/** @brief Overloaded + operator
		* @return vector that is the sum of two vectors
		*/
		vector3D operator+ (vector3D&);

		/** @brief Overloaded += operator
		* @return vector that is the sum of itself and another vector
		*/
		vector3D& operator+= (vector3D&);

		/** @brief Overloaded - operator
		* @return vector that is the difference of two vectors
		*/
		const vector3D operator- (vector3D&);

		/** @brief Overloaded -= operator
		* @return vector that is the difference of itself and another vector
		*/
		vector3D& operator-= (vector3D&);

		/** @brief Overloaded * operator
		* @return vector that is the product of a vector and a float value
		*/
		vector3D operator* (float);

		/** @brief Overloaded *= operator
		* @return vector that is the product of itself and a float value
		*/
		vector3D& operator*= (float);

		/** @brief Overloaded * operator
		* @return Dot product (float) of two vectors
		*/
		float operator* (vector3D&);

		/** @brief Overloaded *= operator
		* @return vector that is the product of itself and another vector
		*
		* e.i. (x,y,z)*=(a,b,c) := (a*x, b*y, c*z)
		*/
		vector3D operator*= (vector3D&);

		/** @brief Overloaded % operator
		* @return Cross product (vector) of two vectors
		*/
		vector3D operator% (vector3D&);

		/** @brief Get distance between two vectors
		* @return float value of the distance between this vector and another
		*/
		float distance(vector3D&);

		/** @brief Normalize the vector
		* @return Vector that is the normal of the current vector
		*/
		vector3D normalize() const;

		/** @brief Normalize the vector
		* Normalizes the current vector instead of returning a copy
		*/
		void normalizeThis();

		float x;
		float y;
		float z;

	private:
};

}

#endif

