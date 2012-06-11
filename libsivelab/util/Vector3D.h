/*  
 *  Vector3D.h
 *
 *  Created by Pete Willemsen on 01/19/2011.
 *  Copyright 2011 Department of Computer Science, University of Minnesota Duluth. All rights reserved.
 *
 *	Modified by: Andrew Larson <lars2865@d.umn.edu>
 *  Reason: Pushed a point class together with this one, adding more operator
 *          functionality.
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

#ifndef __CS5721_GRAPHICSLIB_VECTOR3__
#define __CS5721_GRAPHICSLIB_VECTOR3__ 1

#include <cassert>
#include <iostream>
#include <cmath>

//! Namespace to hold all classes and functionality for CS 5721.
/** 
 * Namespace will be set to contain classes and functionality related
 * to the projects in CS 5721 Computer Graphics at the University of
 * Minnesota Duluth.
 */
namespace sivelab
{
  //! Class representing a basic 3D vector for use with ray tracers and rasterizers in CS5721
  /**
   *
   */
  class Vector3D {
  public:

    //! Default constructor for the Vector3D class.  Sets all values to 0.
    Vector3D();

    //! Constructor for the Vector3D class that sets values to provided x, y, and z values.
    Vector3D(const double x, const double y, const double z);

    //! Destructor
    ~Vector3D() {}

    //! 
    /**
     */
    const double operator[](const int i) const;
    
    //! 
    /**
     */    
    void set(const double x, const double y, const double z); 
    
    //! Destructively normalize the vector, making it unit length.
    /** @return the length of the vector prior to normalization.
     */
    double normalize(void);                

    double mag() const;

    //! Compute the dot product between two vectors. 
    /**
     */
    double dot(const Vector3D &v) const;
    float operator*(Vector3D const& q) const;

    //! Compute the cross product between two vectors. 
    /**
     */
    Vector3D cross(const Vector3D &v) const;      
    Vector3D operator/(Vector3D const& q) const;

    //! Vector3D assignment operator.  Let's you do v1 = v2;
    /** @param
     * @return a reference to the Vector3D to allow chaining of operations.
     */
    Vector3D& operator=(const Vector3D &rhs);

    Vector3D const operator-() const;

    Vector3D& operator+=(Vector3D const& p);
    Vector3D& operator-=(Vector3D const& p);

    Vector3D const operator*(float const& s) const;
    Vector3D const operator/(float const& s) const;

    bool const operator==(Vector3D const& q) const;
    bool const operator!=(Vector3D const& q) const;

  public:
    double& x;
    double& y;
    double& z;

  private:
    friend std::ostream& operator<<(std::ostream& os, const Vector3D &v);
    friend std::istream& operator>>(std::istream& is, Vector3D &v); 
    
    double data[3];
  };

const Vector3D operator+(const Vector3D& lhs, const Vector3D& rhs);
const Vector3D operator-(const Vector3D& lhs, const Vector3D& rhs);
const Vector3D operator*(float const& s, Vector3D const& q);
}

#endif // __CS5721_GRAPHICSLIB_VECTOR3__
