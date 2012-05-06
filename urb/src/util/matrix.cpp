#include "matrix.h"

#include <string.h>
#include <iostream>

#include "util/normal.h"

matrix::matrix() 
{
	rows = 2;
	cols = 2;
	mtrx = new float[rows * cols];	
	memset(mtrx,0,sizeof(float) * rows * cols);
}

matrix::matrix(unsigned int rs, unsigned int cs) 
{
	rows = rs;
	cols = cs;
	mtrx = new float[rows * cols];
	memset(mtrx,0,sizeof(float) * rows * cols);
}

matrix::matrix(float vec[], unsigned int vec_len) 
{
	rows = 1;
	cols = vec_len;
	mtrx = new float[rows * cols];
	memcpy(mtrx,vec,vec_len);
}

matrix::matrix(const matrix& m) 
{
	rows = m.rows;
	cols = m.cols;
	mtrx = new float[rows * cols];
	memcpy(mtrx,m.mtrx,rows * cols);
}

int matrix::getRows() {return rows;}
int matrix::getCols() {return cols;}
float* matrix::getMtrx() {return mtrx;}

matrix::~matrix() {delete[] mtrx;}

float matrix::operator() (int index) {return mtrx[index];}

float& matrix::operator() (int rIndex, int cIndex) const 
{
	if(rIndex > (int) rows || cIndex > (int) cols) {std::cerr << "matrix out of bounds error." << std::endl;}
	if(rIndex < 0) {rIndex = rIndex % rows;}
	if(cIndex < 0) {cIndex = cIndex % cols;}
	return mtrx[cIndex * rows + rIndex];
}

float& matrix::operator() (int rIndex, int cIndex) 
{
	if(rIndex > (int) rows || cIndex > (int) cols) {std::cerr << "matrix out of bounds error." << std::endl;}
	if(rIndex < 0) {rIndex = rIndex % rows;}
	if(cIndex < 0) {cIndex = cIndex % cols;}
	return mtrx[cIndex * rows + rIndex];	
}

matrix& matrix::operator= (const matrix& m) 
{
	if(this != &m) 
	{
		rows = m.rows;
		cols = m.cols;
		delete[] mtrx;
		mtrx = new float[rows * cols];
		memcpy(mtrx,m.mtrx,rows * cols);
	}
	return *this;
}


point matrix::operator* (point v) 
{
	//CHEATING!!! For 3x3 matrix and 3x1 vector only.
	point result;
	point row1 = point(mtrx[0], mtrx[1], mtrx[2]);
	point row2 = point(mtrx[3], mtrx[4], mtrx[5]);
	point row3 = point(mtrx[6], mtrx[7], mtrx[8]);

	//result.x = mtrx[0] * v.x + mtrx[1] * v.y + mtrx[2] * v.z;
	//result.y = mtrx[3] * v.x + mtrx[4] * v.y + mtrx[5] * v.z;
	//result.z = mtrx[6] * v.x + mtrx[7] * v.y + mtrx[8] * v.z;
	//cout << "r: " << result.x << " " << result.y << " " << result.z << endl;
	
	return point(row1*v, row2*v, row3*v);
}

normal matrix::operator* (normal n) {return normal(this->operator*((point) n));}

matrix makeRotation(float a, normal n) {return makeRotation((point) n, a);}
matrix makeRotation(normal n, float a) {return makeRotation((point) n, a);}

matrix makeRotation(float a, point v) {return makeRotation(v,a);}
matrix makeRotation(point v, float a) 
{
	//It is assumed that a is in degrees
	float r = a * 3.141592 / 180.;
	//----------------------------------
	matrix rot(3,3);
	point n = normal(v);
	//cout << "n.xyx: " << n.x << " " << n.y << " " << n.z << endl;
	//Rotation Matrix
		//Row 1
	rot(0,0) = n.x * n.x*(1 - cos(r)) + cos(r);			//cout << rot(0,0) << " " << flush;
	rot(0,1) = n.x * n.y*(1 - cos(r)) - n.z * sin(r);	//cout << rot(0,1) << " " << flush;
	rot(0,2) = n.x * n.z*(1 - cos(r)) + n.y * sin(r);	//cout << rot(0,2) << " " << endl;
		//Row 2
	rot(1,0) = n.x * n.y*(1 - cos(r)) + n.z * sin(r);	//cout << rot(1,0) << " " << flush;
	rot(1,1) = n.y * n.y*(1 - cos(r)) + cos(r);			//cout << rot(1,1) << " " << flush;
	rot(1,2) = n.y * n.z*(1 - cos(r)) - n.x * sin(r);	//cout << rot(1,2) << " " << endl;
		//Row 3
	rot(2,0) = n.x * n.z*(1 - cos(r)) - n.y * sin(r);	//cout << rot(2,0) << " " << flush;
	rot(2,1) = n.y * n.z*(1 - cos(r)) + n.x * sin(r);	//cout << rot(2,1) << " " << flush;
	rot(2,2) = n.z * n.z*(1 - cos(r)) + cos(r);			//cout << rot(2,2) << " " << endl;
	return rot;
}
