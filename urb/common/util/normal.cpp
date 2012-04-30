#include "normal.h"

normal::normal(void) : point() {z = 1.f;}

normal::normal(normal const& n) : point(n) {}

normal::normal(point const& p) : point(p)
{
  // Defaulting of normal to UP.
  z = (p.x + p.y + p.z == 0.f) ? 1.f : p.z ;
  normal::normalize(x, y, z);
}

normal::normal(float const& _x, float const& _y, float const& _z) : point(_x, _y, _z) 
{
  // Defaulting of normal to UP.
  z = (_x + _y + _z == 0.f) ? 1.f : _z ;
  normal::normalize(x, y, z);
}

float normal::getX() const {return x;}
float normal::getY() const {return y;}
float normal::getZ() const {return z;}

normal normal::operator -() const {return normal(-x, -y, -z);}

normal& normal::operator =(normal const& n) 
{
  point::operator=(n);
  return *this;
}

normal& normal::operator =(point const& p) 
{
  point::operator=(p);
  z = (p.x + p.y + p.z == 0.f) ? 1. : p.z ;

  normal::normalize(x, y, z);
  return *this;
}

normal normal::operator /(normal const& m) const
{
  return normal(y*m.z - z*m.y, z*m.x - x*m.z, x*m.y - y*m.x);
}

normal normal::operator /(point const& p) const
{
  return normal(y*p.z - z*p.y, z*p.x - x*p.z, x*p.y - y*p.x);
}

void normal::normalize(point& p) 
{
  normal::normalize(p.x, p.y, p.z);
}

void normal::normalize(float& x, float& y, float& z)
{
  float vecLength = sqrt(x*x + y*y + z*z);
  x /= vecLength;
  y /= vecLength;
  z /= vecLength;
}

normal operator /(point const& p, normal const& m)
{
  return m.operator/(p);
}

