/*  
 *  Vector3D.cpp
 *
 *  Created by Pete Willemsen on 01/19/2011.
 *  Copyright 2011 Department of Computer Science, University of Minnesota Duluth. All rights reserved.
 *
 * This file is part of CS5721 Computer Graphics library (cs5721Graphics).
 *
 * cs5721Graphics is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cs5721Graphics is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with cs5721Graphics.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "Vector3D.h"

using namespace std;

namespace sivelab
{
  Vector3D::Vector3D()
  : x(data[0]),
    y(data[1]),
    z(data[2]) 
  { 
    data[0] = 0;
    data[1] = 0;
    data[2] = 0;
  }

  Vector3D::Vector3D(const double x, const double y, const double z) 
  : x(data[0]),
    y(data[1]),
    z(data[2])
  {
    data[0] = x;
    data[1] = y;
    data[2] = z;
  }

  const double Vector3D::operator[](const int i) const
  { 
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 3); 
    return data[i]; 
  }
    
  void Vector3D::set(const double x, const double y, const double z) 
  {
    data[0] = x;
    data[1] = y;
    data[2] = z;
  }

  double Vector3D::normalize(void)
  {
    const double vector_length = sqrt( data[0]*data[0] + 
				       data[1]*data[1] +
				       data[2]*data[2] );
    if (vector_length > 0.0)
      {
        data[0] /= vector_length;
        data[1] /= vector_length;
        data[2] /= vector_length;
      }
    
    return vector_length;
  }

  double Vector3D::mag() const
  {
    return sqrt(data[0]*data[0] + data[1]*data[1] + data[2]*data[2]);
  }

  double Vector3D::dot(const Vector3D &q) const
  {
    return x*q.x + y*q.y + z*q.z;
  }
  
  float Vector3D::operator*(Vector3D const& q) const
  {
    return x*q.x + y*q.y + z*q.z;
  }

  Vector3D Vector3D::cross(const Vector3D &q) const
  {
    return Vector3D(y*q.z - z*q.x, z*q.x - x*q.z, x*q.y - y*q.x);
  }
        
  Vector3D Vector3D::operator/(Vector3D const& q) const
  {
    return Vector3D(y*q.z - z*q.x, z*q.x - x*q.z, x*q.y - y*q.x);
  }

  Vector3D& Vector3D::operator=(const Vector3D &rhs)
  {
    // v1 = v2 --> same as v1.operator=(v2);
    data[0] = rhs.data[0];
    data[1] = rhs.data[1];
    data[2] = rhs.data[2];
    return *this;
  }

  Vector3D const Vector3D::operator-() const
  {
    return Vector3D(-data[0], -data[1], -data[2]);
  }

  //! 
  Vector3D& Vector3D::operator+=(const Vector3D &rhs)
  {
    data[0] += rhs.data[0];
    data[1] += rhs.data[1];
    data[2] += rhs.data[2];
    return *this;
  }

  Vector3D& Vector3D::operator-=(const Vector3D &rhs)
  {
    data[0] -= rhs.data[0];
    data[1] -= rhs.data[1];
    data[2] -= rhs.data[2];
    return *this;
  }

  Vector3D const Vector3D::operator*(float const& s) const
  {
    return Vector3D(x * s, y * s, z * s);
  }
  
  Vector3D const Vector3D::operator/(float const& s) const
  {
    // Trust that s != 0?
    return Vector3D(x / s, y / s, z / s);
  }

  bool const Vector3D::operator==(Vector3D const& q) const
  {
    return x == q.x && y == q.y && z == q.z;
  }
  
  bool const Vector3D::operator!=(Vector3D const& q) const
  {
    return x != q.x || y != q.y || z != q.z;
  }

  ostream &operator<<(ostream& os, const sivelab::Vector3D& q)
  {
    os << '[' << q[0] << ' ' << q[1] << ' ' << q[2] << ']';
    return os;
  }

  istream &operator>>(istream& is, sivelab::Vector3D& q)
  {
    double x=0, y=0, z=0;
    is >> x >> y >> z;
    q.set(x,y,z);
    return is;
  }

  const Vector3D operator+(const sivelab::Vector3D &lhs, const sivelab::Vector3D &rhs)
  {
    return sivelab::Vector3D(lhs) += rhs;
  }

  const Vector3D operator-(const sivelab::Vector3D &lhs, const sivelab::Vector3D &rhs)
  {
    return sivelab::Vector3D(lhs) -= rhs;
  }
  
  const Vector3D operator*(float const& s, Vector3D const& q);
}

