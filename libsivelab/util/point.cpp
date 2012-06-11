#include "point.h"

namespace sivelab
{
  /*
  point::point(void) {x = y = z = 0.;}

  point::point(point const& p) 
  {
    x = p.x;
    y = p.y;
    z = p.z;		
  }

  point::point(float const& _x, float const& _y, float const& _z) 
  {
    x = _x;
    y = _y;
    z = _z;
  }   

  point point::operator -() const {return point(-x, -y, -z);}

  point& point::operator +=(point const& p) 
  {
    x += p.x;
    y += p.y;
    z += p.z;
    return *this;
  }

  point& point::operator +=(float const& s) 
  {
    x += s;
    y += s;
    z += s; 
    return *this;
  }

  point& point::operator -=(point const& p) 
  {
    x -= p.x;
    y -= p.y;
    z -= p.z;
    return *this;
  }

  point& point::operator -=(float const& s) 
  {
    x -= s;
    y -= s;
    z -= s; 
    return *this;
  }

  point& point::operator =(point const& p) 
  {
    x = p.x;
    y = p.y;
    z = p.z;
    return *this;
  }

  point point::operator +(point const& q) const {return point(x + q.x, y + q.y, z + q.z);}
  point point::operator +(float const& s) const {return point(x +   s, y +   s, z +   s);}

  point point::operator -(point const& q) const {return point(x - q.x, y - q.y, z - q.z);}
  point point::operator -(float const& s) const {return point(x -   s, y -   s, z -   s);}

  float point::operator *(point const& q) const {return x*q.x + y*q.y + z*q.z;}
  point point::operator *(float const& s) const {return point(x*s, y*s, z*s);}

  point point::operator /(point const& q) const {return point(y*q.z - z*q.x, z*q.x - x*q.z, x*q.y - y*q.x);}
  point point::operator /(float const& s) const {return point(x / s, y / s, z / s);}

  bool  point::operator <(point const& q) const {return mag(*this) < mag(q);}
  bool  point::operator >(point const& q) const {return mag(*this) > mag(q);}

  bool  point::operator==(point const& q) const {return (x == q.x && y == q.y && z == q.z);}
  bool  point::operator!=(point const& q) const {return !(*this == q);}

  point operator +(float const& s, point const& q) {return q.operator+(s);}
  point operator -(float const& s, point const& q) {return q.operator+(s);}
  point operator *(float const& s, point const& q) {return q.operator*(s);}

  float mag(point const& p)
  {
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
  }
  */
}
