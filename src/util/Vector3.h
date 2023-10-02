/****************************************************************************
 * Copyright (c) 2022 University of Utah
 * Copyright (c) 2022 University of Minnesota Duluth
 *
 * Copyright (c) 2022 Behnam Bozorgmehr
 * Copyright (c) 2022 Jeremy A. Gibbs
 * Copyright (c) 2022 Fabien Margairaz
 * Copyright (c) 2022 Eric R. Pardyjak
 * Copyright (c) 2022 Zachary Patterson
 * Copyright (c) 2022 Rob Stoll
 * Copyright (c) 2022 Lucas Ulmer
 * Copyright (c) 2022 Pete Willemsen
 *
 * This file is part of QES-Winds
 *
 * GPL-3.0 License
 *
 * QES-Winds is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Winds is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Winds. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file Vector3.h */

#pragma once

#include <type_traits>
#include <iostream>
#include <math.h>

#define FLOATS_ARE_EQUAL(x, y) (((x) - (y)) < 0.000001 && ((x) - (y)) > -0.000001)

/**
 * @class Vector3
 * @brief Template class that holds 3 values.
 *
 * Values can be accessed as if this was an array.
 */

class Vector3
{
protected:
  // std::vector<float> values;
  float values[3];

public:
  Vector3()
    : values{ 0.0, 0.0, 0.0 }
  {
  }

  /*
    template <typename X>
    Vector3(const Vector3<X>& newV)
    {
    values[0] = newV[0];
    values[1] = newV[1];
    values[2] = newV[2];
    }
  */
  /*
  template<typename X>
  Vector3(const X a, const X b, const X c)
    : values{ a, b, c }
  {
  }
  */
  Vector3(float x, float y, float z)
    : values{ x, y, z }
  {
  }

  ~Vector3() {}

  /**
   * Accesses the value at position i.
   *
   * @param i the index of the value to return
   * @return a reference to the value stored at i
   */
  float &operator[](const int i)
  {
    return values[i];
  }

  /**
   * Accesses the value at position i.
   *
   * @param i the index of the value to return
   * @return a copy to the value stored at i
   */
  float operator[](const int i) const
  {
    return values[i];
  }

  /**
   * Returns if two Vector3 values of the same type are equal.
   *
   * @param v the vector3 to compare with this
   * @return if values at index 0,1,2 are all equal with their counterparts
   */
  bool operator==(const Vector3 &v)
  {
    // if (std::is_same<T,float>::value || std::is_same<T, double>::value)
    return FLOATS_ARE_EQUAL(values[0], v.values[0]) && FLOATS_ARE_EQUAL(values[1], v.values[1]) && FLOATS_ARE_EQUAL(values[2], v.values[2]);
    /*else
      return v.values[0] == values[0] && v.values[1] == values[1] && v.values[2] == values[2];*/
  }

  // assignment operator
  Vector3 &operator=(const Vector3 &v)
  {
    values[0] = v.values[0];
    values[1] = v.values[1];
    values[2] = v.values[2];
    return *this;
  }

  // scalar addition (assignment)
  Vector3 &operator+=(const Vector3 &v)
  {
    values[0] += v.values[0];
    values[1] += v.values[1];
    values[2] += v.values[2];
    return *this;
  }

  // scalar subtraction (assignment)
  Vector3 &operator-=(const Vector3 &v)
  {
    values[0] -= v.values[0];
    values[1] -= v.values[1];
    values[2] -= v.values[2];
    return *this;
  }

  // scalar multiplication (assignment)
  Vector3 &operator*=(const float &a)
  {
    values[0] *= a;
    values[1] *= a;
    values[2] *= a;
    return *this;
  }

  // scalar division (assignment)
  Vector3 &operator/=(const float &a)
  {
    values[0] /= a;
    values[1] /= a;
    values[2] /= a;
    return *this;
  }

  // addition operator
  Vector3 operator-(const Vector3 &v1)
  {
    return { values[0] - v1.values[0], values[1] - v1.values[1], values[2] - v1.values[2] };
  }

  // subtraction operator
  Vector3 operator+(const Vector3 &v1)
  {
    return { values[0] + v1.values[0], values[1] + v1.values[1], values[2] + v1.values[2] };
  }

  // scalar product (dot product)
  float dot(const Vector3 &v1) const
  {
    return (values[0] * v1.values[0] + values[1] * v1.values[1] + values[2] * v1.values[2]);
  }

  // vector product
  Vector3 cross(const Vector3 &v1) const
  {
    return { values[1] * v1.values[2] - values[2] * v1.values[1],
             values[2] * v1.values[0] - values[0] * v1.values[2],
             values[0] * v1.values[1] - values[1] * v1.values[0] };
  }

  // scalar product (dot product)
  float operator*(const Vector3 &v1) const
  {
    return (values[0] * v1.values[0] + values[1] * v1.values[1] + values[2] * v1.values[2]);
  }


  // multiplication by scalar
  Vector3 operator*(const float &a)
  {
    return { a * values[0], a * values[1], a * values[2] };
  }

  // return the length of the vector
  float length() const
  {
    return sqrtf(values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
  }

  // multiplication by scalar
  friend Vector3 operator*(const float &a, const Vector3 &v1)
  {
    return { a * v1.values[0], a * v1.values[1], a * v1.values[2] };
  }

  // division by scalar
  Vector3 operator/(const float &a)
  {
    return { values[0] / a, values[1] / a, values[2] / a };
  }

  // reflection v.reflect(n) = v - 2*(v*n)*n
  Vector3 reflect(const Vector3 &n)
  {
    return *this - 2.0f * (*this * n) * n;
  }

  // distance with other vector where this is extremity (ie v1.distance(v2) = |v1 - v2|)
  float distance(Vector3 &v2)
  {
    return (sqrtf((values[0] - v2[0]) * (values[0] - v2[0]) + (values[1] - v2[1]) * (values[1] - v2[1]) + (values[2] - v2[2]) * (values[2] - v2[2])));
  }


  friend std::istream &operator>>(std::istream &is, Vector3 &v)
  {
    is >> v.values[0] >> v.values[1] >> v.values[2];
    return is;
  }

  friend Vector3 operator-(const Vector3 &v1, const Vector3 &v2)
  {
    return { v1.values[0] - v2.values[0], v1.values[1] - v2.values[1], v1.values[2] - v2.values[2] };
  }

  friend std::ostream &operator<<(std::ostream &out, const Vector3 &v)
  {
    out << "[";
    for (int i(0); i < 2; i++)
      out << v.values[i] << ", ";
    out << v.values[2] << "]";
    return out;
  }
};
