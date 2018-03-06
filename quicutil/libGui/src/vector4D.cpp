/* File: vector4D.cpp
 * Author: Matthew Overby
 */

#include "vector4D.h"

using namespace SLUI;

vector4D::vector4D(){
	x = 0.0f;
	y = 0.0f;
	z = 0.0f;
	w = 0.0f;
}

vector4D::vector4D(float a, float b, float c, float d){
	x = a;
	y = b;
	z = c;
	w = d;
}

void vector4D::printVec(){
	printf("(%f, %f, %f, %f)", x, y, z, w);
}


vector4D vector4D::zeroVec(){
	return vector4D(0.0f, 0.0f, 0.0f, 0.0f);
}

bool vector4D::operator== (vector4D& vector){
	if((w-vector.w)<0.0001 && (x-vector.x)<0.0001 && (y-vector.y)<0.0001 && (z-vector.z)<0.0001)
		return true;
	else
		return false;
}

bool vector4D::operator!= (vector4D& vector){
	vector4D temp = *this;
	if(temp == vector)
		return false;
	else
		return true;
}


vector4D vector4D::operator+ (vector4D& vector){
	vector4D result = *this;
	result.x = vector.x + x;
	result.y = vector.y + y;
	result.z = vector.z + z;
	result.w = vector.w + w;
	return result; 
}

vector4D& vector4D::operator+= (vector4D& vector){
	x += vector.x;
	y += vector.y;
	z += vector.z;
	w += vector.w;
	return *this;
}

vector4D vector4D::operator- (vector4D& vector){
	vector4D result = *this;
	result.x = x - vector.x;
	result.y = y - vector.y;
	result.z = z - vector.z;
	result.w = w - vector.w;
	return result; 
}

vector4D& vector4D::operator-= (vector4D& vector){
	x -= vector.x;
	y -= vector.y;
	z -= vector.z;
	w -= vector.w;
	return *this;
}

vector4D vector4D::operator* (float n){

	vector4D result = *this;
	result.x = x * n;
	result.y = y * n;
	result.z = z * n;
	result.w = w * n;
	return result; 
}

vector4D& vector4D::operator*= (float n){

	x *= n;
	y *= n;
	z *= n;
	w *= n;
	return *this;
}

float vector4D::operator* (vector4D& vector){

	float f = x*vector.x + y*vector.y + z*vector.z + w*vector.w ;
	return f; 
}

vector4D vector4D::operator% (vector4D& vector){

	vector4D result = *this;

	return result;
}

vector4D vector4D::normalize() const{

	vector4D result = *this;

	float length = sqrt(result.x*result.x + result.y*result.y + result.z*result.z + result.w*result.w);

	if(length == 0.0f)
		return result;

	result.x = result.x/length;
	result.y = result.y/length;
	result.z = result.z/length;
	result.w = result.w/length;

	return result;
}

void vector4D::normalizeThis() {

	float length = sqrt(x*x + y*y + z*z + w*w);

	if(length == 0.0f)
		return;

	x = x/length;
	y = y/length;
	z = z/length;
	w = w/length;

}

void vector4D::homogenizeThis() {
	x = x/w;
	y = y/w;
	z = z/w;
	w = w/w;
}




