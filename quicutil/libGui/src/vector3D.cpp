/* File: vector3D.cpp
 * Author: Matthew Overby
 */

#include "vector3D.h"

using namespace SLUI;

vector3D::vector3D(){
	x = 0.0f;
	y = 0.0f;
	z = 0.0f;
}

vector3D::vector3D(float a, float b, float c){
	x = a;
	y = b;
	z = c;
}

vector3D::~vector3D(){
}

void vector3D::printVec(){
	std::cout << "(" << x << ", " << y << ", " << z << ")" << std::endl;
}

bool vector3D::operator== (vector3D& vector){
	if((x-vector.x)<0.0001 && (y-vector.y)<0.0001 && (z-vector.z)<0.0001)
		return true;
	else
		return false;
}

bool vector3D::operator!= (vector3D& vector){
	vector3D temp = *this;
	if(temp == vector)
		return false;
	else
		return true;
}


vector3D vector3D::operator+ (vector3D& vector){
	vector3D result = *this;
	result.x = vector.x + x;
	result.y = vector.y + y;
	result.z = vector.z + z;
	return result; 
}

vector3D& vector3D::operator+= (vector3D& vector){
	x += vector.x;
	y += vector.y;
	z += vector.z;
	return *this;
}

const vector3D vector3D::operator- (vector3D& vector){
	vector3D result = *this;
	result.x = x - vector.x;
	result.y = y - vector.y;
	result.z = z - vector.z;
	return result; 
}

vector3D& vector3D::operator-= (vector3D& vector){
	x -= vector.x;
	y -= vector.y;
	z -= vector.z;
	return *this;
}

vector3D vector3D::operator* (float n){

	vector3D result = *this;
	result.x = x * n;
	result.y = y * n;
	result.z = z * n;
	return result; 
}

vector3D& vector3D::operator*= (float n){

	x *= n;
	y *= n;
	z *= n;
	return *this;
}

float vector3D::operator* (vector3D& vector){

	float f = x*vector.x + y*vector.y + z*vector.z;
	return f; 
}


vector3D vector3D::operator*= (vector3D& vec){
	vector3D result = *this;
	result.x = result.x * vec.x;
	result.y = result.y * vec.y;
	result.z = result.z * vec.z;
	return result;
}

vector3D vector3D::operator% (vector3D& vector){

	vector3D result = *this;

	result.x = y*vector.z - z*vector.y;
	result.y = z*vector.x - x*vector.z;
	result.z = x*vector.y - y*vector.x;
	return result;
}

float vector3D::distance(vector3D& v){

	float dist = sqrt(	(x - v.x )*(x - v.x ) +
				(y - v.y )*(y - v.y ) +
				(z - v.z )*(z - v.z ) );

	return dist;
}

vector3D vector3D::normalize() const{

	vector3D result = *this;

	float length = sqrt(result.x*result.x + result.y*result.y + result.z*result.z);

	if(length == 0.0f)
		return result;

	result.x = result.x/length;
	result.y = result.y/length;
	result.z = result.z/length;

	return result;
}

void vector3D::normalizeThis() {

	float length = sqrt(x*x + y*y + z*z);

	if(length == 0.0f)
		return;

	x = x/length;
	y = y/length;
	z = z/length;

}

