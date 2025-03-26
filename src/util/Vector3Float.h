/****************************************************************************
 * Copyright (c) 2024 University of Utah
 * Copyright (c) 2024 University of Minnesota Duluth
 *
 * Copyright (c) 2024 Behnam Bozorgmehr
 * Copyright (c) 2024 Jeremy A. Gibbs
 * Copyright (c) 2024 Fabien Margairaz
 * Copyright (c) 2024 Eric R. Pardyjak
 * Copyright (c) 2024 Zachary Patterson
 * Copyright (c) 2024 Rob Stoll
 * Copyright (c) 2024 Lucas Ulmer
 * Copyright (c) 2024 Pete Willemsen
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

/** @file Vector3Float.h */

#pragma once

#include <type_traits>
#include <iostream>
#include <cmath>

#define FLOATS_ARE_EQUAL(x, y) (((x) - (y)) < 0.000001 && ((x) - (y)) > -0.000001)

/**
 * @class Vector3Float
 * @brief Template class that holds 3 values.
 *
 * Values can be accessed as if this was an array.
 */

class Vector3Float
{
protected:
  // std::vector<float> values;
  float values[3];

public:
  Vector3Float()
    : values{ 0.0, 0.0, 0.0 }
  {
  }

  /*
    template <typename X>
    Vector3Float(const Vector3Float<X>& newV)
    {
    values[0] = newV[0];
    values[1] = newV[1];
    values[2] = newV[2];
    }
  */
  /*
  template<typename X>
  Vector3Float(const X a, const X b, const X c)
    : values{ a, b, c }
  {
  }
  */
  Vector3Float(float x, float y, float z)
    : values{ x, y, z }
  {
  }

  ~Vector3Float() {}

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
   * Returns if two Vector3Float values of the same type are equal.
   *
   * @param v the vector3 to compare with this
   * @return if values at index 0,1,2 are all equal with their counterparts
   */
  bool operator==(const Vector3Float &v)
  {
    // if (std::is_same<T,float>::value || std::is_same<T, double>::value)
    return FLOATS_ARE_EQUAL(values[0], v.values[0]) && FLOATS_ARE_EQUAL(values[1], v.values[1]) && FLOATS_ARE_EQUAL(values[2], v.values[2]);
    /*else
      return v.values[0] == values[0] && v.values[1] == values[1] && v.values[2] == values[2];*/
  }

  // assignment operator
  Vector3Float &operator=(const Vector3Float &v)
  {
    values[0] = v.values[0];
    values[1] = v.values[1];
    values[2] = v.values[2];
    return *this;
  }

  // scalar addition (assignment)
  Vector3Float &operator+=(const Vector3Float &v)
  {
    values[0] += v.values[0];
    values[1] += v.values[1];
    values[2] += v.values[2];
    return *this;
  }

  // scalar subtraction (assignment)
  Vector3Float &operator-=(const Vector3Float &v)
  {
    values[0] -= v.values[0];
    values[1] -= v.values[1];
    values[2] -= v.values[2];
    return *this;
  }

  // scalar multiplication (assignment)
  Vector3Float &operator*=(const float &a)
  {
    values[0] *= a;
    values[1] *= a;
    values[2] *= a;
    return *this;
  }

  // scalar division (assignment)
  Vector3Float &operator/=(const float &a)
  {
    values[0] /= a;
    values[1] /= a;
    values[2] /= a;
    return *this;
  }

  // addition operator
  Vector3Float operator-(const Vector3Float &v1)
  {
    return { values[0] - v1.values[0], values[1] - v1.values[1], values[2] - v1.values[2] };
  }

  // subtraction operator
  Vector3Float operator+(const Vector3Float &v1)
  {
    return { values[0] + v1.values[0], values[1] + v1.values[1], values[2] + v1.values[2] };
  }

  // scalar product (dot product)
  float dot(const Vector3Float &v1) const
  {
    return (values[0] * v1.values[0] + values[1] * v1.values[1] + values[2] * v1.values[2]);
  }

  // vector product
  Vector3Float cross(const Vector3Float &v1) const
  {
    return { values[1] * v1.values[2] - values[2] * v1.values[1],
             values[2] * v1.values[0] - values[0] * v1.values[2],
             values[0] * v1.values[1] - values[1] * v1.values[0] };
  }

  // scalar product (dot product)
  float operator*(const Vector3Float &v1) const
  {
    return (values[0] * v1.values[0] + values[1] * v1.values[1] + values[2] * v1.values[2]);
  }


  // multiplication by scalar
  Vector3Float operator*(const float &a)
  {
    return { a * values[0], a * values[1], a * values[2] };
  }

  // return the length of the vector
  float length() const
  {
    return sqrtf(values[0] * values[0] + values[1] * values[1] + values[2] * values[2]);
  }

  // multiplication by scalar
  friend Vector3Float operator*(const float &a, const Vector3Float &v1)
  {
    return { a * v1.values[0], a * v1.values[1], a * v1.values[2] };
  }

  // division by scalar
  Vector3Float operator/(const float &a)
  {
    return { values[0] / a, values[1] / a, values[2] / a };
  }

  // reflection v.reflect(n) = v - 2*(v*n)*n
  Vector3Float reflect(const Vector3Float &n)
  {
    return *this - 2.0f * (*this * n) * n;
  }

  // distance with other vector where this is extremity (ie v1.distance(v2) = |v1 - v2|)
  float distance(Vector3Float &v2)
  {
    return (sqrtf((values[0] - v2[0]) * (values[0] - v2[0]) + (values[1] - v2[1]) * (values[1] - v2[1]) + (values[2] - v2[2]) * (values[2] - v2[2])));
  }


  friend std::istream &operator>>(std::istream &is, Vector3Float &v)
  {
    is >> v.values[0] >> v.values[1] >> v.values[2];
    return is;
  }

  friend Vector3Float operator-(const Vector3Float &v1, const Vector3Float &v2)
  {
    return { v1.values[0] - v2.values[0], v1.values[1] - v2.values[1], v1.values[2] - v2.values[2] };
  }

  friend std::ostream &operator<<(std::ostream &out, const Vector3Float &v)
  {
    out << "[";
    for (int i(0); i < 2; i++)
      out << v.values[i] << ", ";
    out << v.values[2] << "]";
    return out;
  }
};
