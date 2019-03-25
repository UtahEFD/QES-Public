/* File: Matrix3.cpp
 * Author: Matthew Overby
 */

#include "Matrix3.h"

using namespace SLUI;

Matrix3::Matrix3(vector3D a1, vector3D b1, vector3D c1){

	a = a1;
	b = b1;
	c = c1;
}

float Matrix3::getDeter(){

	float j, k, l;

	j = a.x*(b.y*c.z - b.z*c.y);
	k = a.y*(b.x*c.z - b.z*c.x);
	l = a.z*(b.x*c.y - b.y*c.x);

	return (j-k+l);
}

float Matrix3::getDeter2x2(vector4D vec){

	float j, k;

	j = vec.w * vec.z;
	k = vec.x * vec.y;

	return (j-k);
}

Matrix3 Matrix3::getTranspose(){

	Matrix3 result = *this;

	result.a = vector3D(a.x, b.x, c.x);
	result.b = vector3D(a.y, b.y, c.y);
	result.c = vector3D(a.z, b.z, c.z);

	return result;
}

Matrix3 Matrix3::getIdentity(){

	Matrix3 result = *this;

	result.a = vector3D(1.0f, 0.0f, 0.0f);
	result.b = vector3D(0.0f, 1.0f, 0.0f);
	result.c = vector3D(0.0f, 0.0f, 1.0f);

	return result;
}

Matrix3 Matrix3::getInverse(){

	Matrix3 inverse = *this;
	if(inverse.getDeter() == 0.0f)
		return inverse.getIdentity();

	return inverse;
}

void Matrix3::setZero(){

	a = vzero;
	b = vzero;
	c = vzero;
}

Matrix3& Matrix3::operator=  (Matrix3& m3){

	a = m3.a;
	b = m3.b;
	c = m3.c;
	return *this;
}

Matrix3 Matrix3::operator+ (Matrix3& m3){

	Matrix3 result = *this;
	result.a = result.a + m3.a;
	result.b = result.b + m3.b;
	result.c = result.c + m3.c;
	return result; 
}

Matrix3 Matrix3::operator- (Matrix3& m3){

	Matrix3 result = *this;
	result.a = result.a - m3.a;
	result.b = result.b - m3.b;
	result.c = result.c - m3.c;
	return result; 
}

Matrix3 Matrix3::operator* (float n){

	Matrix3 result = *this;
	result.a = a * n;
	result.b = b * n;
	result.c = c * n;
	return result; 
}

Matrix3 Matrix3::operator* (Matrix3& m3){

	Matrix3 result = *this;
	Matrix3 temp = result;
	Matrix3 temp2 = m3.getTranspose();

	result.a.x = temp.a * temp2.a; 
	result.a.y = temp.a * temp2.b; 
	result.a.z = temp.a * temp2.c;

	result.b.x = temp.b * temp2.a; 
	result.b.y = temp.b * temp2.b; 
	result.b.z = temp.b * temp2.c;

	result.c.x = temp.c * temp2.a; 
	result.c.y = temp.c * temp2.b; 
	result.c.z = temp.c * temp2.c;

	return result;
}

void Matrix3::printMatrix(){

	printf("\n");
	a.printVec();
	printf("\n");
	b.printVec();
	printf("\n");
	c.printVec();
	printf("\n");
}

