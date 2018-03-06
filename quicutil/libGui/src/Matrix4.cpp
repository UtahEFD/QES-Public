/* File: Matrix4.cpp
 * Author: Matthew Overby
 */

#include "Matrix4.h"

using namespace SLUI;

Matrix4::Matrix4(){

	a = vector4D(0.0, 0.0, 0.0, 0.0);
	b = vector4D(0.0, 0.0, 0.0, 0.0);
	c = vector4D(0.0, 0.0, 0.0, 0.0);
	d = vector4D(0.0, 0.0, 0.0, 0.0);
}

Matrix4::Matrix4(vector4D a1, vector4D b1, vector4D c1, vector4D d1){

	a = a1;
	b = b1;
	c = c1;
	d = d1;
}

float Matrix4::getDeter(){

	float j, k, l, m;

	Matrix3 a11 = Matrix3(vector3D(b.x, b.y, b.z), vector3D(c.x, c.y, c.z), vector3D(d.x, d.y, d.z));
	Matrix3 a12 = Matrix3(vector3D(b.w, b.y, b.z), vector3D(c.w, c.y, c.z), vector3D(d.w, d.y, d.z));
	Matrix3 a13 = Matrix3(vector3D(b.w, b.x, b.z), vector3D(c.w, c.x, c.z), vector3D(d.w, d.x, d.z));
	Matrix3 a14 = Matrix3(vector3D(b.w, b.x, b.y), vector3D(c.w, c.x, c.y), vector3D(d.w, d.x, d.y));

	j = a11.getDeter()*a.w;
	k = a12.getDeter()*a.x;
	l = a13.getDeter()*a.y;
	m = a14.getDeter()*a.z;

	return (j-k+l-m);
}

vector4D Matrix4::getInverse2x2(vector4D m){

	float deter = 1/(m.w*m.z - m.x*m.y);

	if(deter == 0.0f)
		return m.zeroVec();

	float temp = m.w;
	m.w = m.z;
	m.z = temp;
	m.x = -1*m.x;
	m.y = -1*m.y;

	m = m*deter;

	return m;
}

Matrix4 Matrix4::getTranspose(){

	Matrix4 result = *this;

	result.a = vector4D(a.w, b.w, c.w, d.w);
	result.b = vector4D(a.x, b.x, c.x, d.x);
	result.c = vector4D(a.y, b.y, c.y, d.y);
	result.d = vector4D(a.z, b.z, c.z, d.z);

	return result;
}

Matrix4 Matrix4::getIdentity(){

	Matrix4 identity = *this;

	identity.a = vector4D(1.0f, 0.0f, 0.0f, 0.0f);
	identity.b = vector4D(0.0f, 1.0f, 0.0f, 0.0f);
	identity.c = vector4D(0.0f, 0.0f, 1.0f, 0.0f);
	identity.d = vector4D(0.0f, 0.0f, 0.0f, 1.0f);

	return identity;
}


Matrix4 Matrix4::getInverse(){

	Matrix4 inverse = *this;
/*
	vector4D p = vector4D(inverse.a.w, inverse.a.x, inverse.b.w, inverse.b.x);
	vector4D q = vector4D(inverse.a.y, inverse.a.z, inverse.b.y, inverse.b.z);
	vector4D r = vector4D(inverse.c.w, inverse.c.x, inverse.d.w, inverse.b.x);
	vector4D s = vector4D(inverse.c.y, inverse.c.z, inverse.d.y, inverse.d.z);

	vector4D temp = r.mult(getInverse2x2(p));
	temp = temp.mult(q);
	s = getInverse2x2(s -  temp);

	temp = getInverse2x2(p);
	temp = r.mult(temp);
	r = s.mult(temp)*-1;

	temp = getInverse2x2(p).mult(q);
	q = temp.mult(s)*-1;

	p = getInverse2x2(p) - (getInverse2x2(p).mult(q)).mult(r);

*/
	return inverse;
}


void Matrix4::setZero(){

	a = a.zeroVec();
	b = b.zeroVec();
	c = c.zeroVec();
	d = d.zeroVec();
}

Matrix4& Matrix4::operator=  (Matrix4& m4){

	a = m4.a;
	b = m4.b;
	c = m4.c;
	d = m4.d;

	return *this;
}

Matrix4 Matrix4::operator+ (Matrix4 &m4){

	Matrix4 result = *this;

	result.a = result.a + m4.a;
	result.b = result.b + m4.b;
	result.c = result.c + m4.c;
	result.d = result.d + m4.d;

	return result; 
}

Matrix4 Matrix4::operator- (Matrix4 &m4){

	Matrix4 result = *this;

	result.a = result.a - m4.a;
	result.b = result.b - m4.b;
	result.c = result.c - m4.c;
	result.d = result.d - m4.d;

	return result; 
}

Matrix4 Matrix4::operator* (float n){

	Matrix4 result = *this;
	result.a = a * n;
	result.b = b * n;
	result.c = c * n;
	result.d = d * n;
	return result; 
}

Matrix4 Matrix4::operator* (Matrix4& m4){

	Matrix4 result = *this;
	Matrix4 temp = result;
	Matrix4 temp2 = m4.getTranspose();


	result.a.w = temp.a * temp2.a;
	result.a.x = temp.a * temp2.b; 
	result.a.y = temp.a * temp2.c; 
	result.a.z = temp.a * temp2.d;

	result.b.w = temp.b * temp2.a;
	result.b.x = temp.b * temp2.b; 
	result.b.y = temp.b * temp2.c; 
	result.b.z = temp.b * temp2.d;

	result.c.w = temp.c * temp2.a;
	result.c.x = temp.c * temp2.b; 
	result.c.y = temp.c * temp2.c; 
	result.c.z = temp.c * temp2.d;

	return result;
}

void Matrix4::printMatrix(){

	printf("\n");
	a.printVec();
	printf("\n");
	b.printVec();
	printf("\n");
	c.printVec();
	printf("\n");
	d.printVec();
	printf("\n");

}



