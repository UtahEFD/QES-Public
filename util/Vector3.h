/****************************************************************************
 * Copyright (c) 2021 University of Utah
 * Copyright (c) 2021 University of Minnesota Duluth
 *
 * Copyright (c) 2021 Behnam Bozorgmehr
 * Copyright (c) 2021 Jeremy A. Gibbs
 * Copyright (c) 2021 Fabien Margairaz
 * Copyright (c) 2021 Eric R. Pardyjak
 * Copyright (c) 2021 Zachary Patterson
 * Copyright (c) 2021 Rob Stoll
 * Copyright (c) 2021 Pete Willemsen
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

/*
 * This is a template class that holds 3 values. These values
 * can be accessed as if this was an array.
 */

#include <type_traits>
#include <iostream>
#include "util/ParseInterface.h"


#define FLOATS_ARE_EQUAL(x, y) (((x) - (y)) < 0.000001 && ((x) - (y)) > -0.000001)

template<class T>
class Vector3;

template<typename T>
std::ostream &operator<<(std::ostream &, const Vector3<T> &);

template<class T>
class Vector3 : public ParseInterface
{
  friend std::ostream &operator<<<T>(std::ostream &, const Vector3<T> &);

protected:
  std::vector<T> values;

public:
  Vector3()
  {
    values.clear();
    values.push_back((0));
    values.push_back((0));
    values.push_back((0));
  }

  /*	template <typename X> Vector3(const Vector3<X>& newV)
        {
		for (int i = 0; i < 3; i++)
        values[i] = newV[i];
        }
    */

  template<typename X>
  Vector3(const X a, const X b, const X c)
  {
    values.clear();
    values.push_back(a);
    values.push_back(b);
    values.push_back(c);
  }

  virtual void parseValues()
  {
    values.clear();
    parseTaglessValues<T>(values);
  }

  /*
	 * accesses the value at position i
	 *
	 * @param i -the index of the value to return
	 * @return a reference to the value stored at i
	 */
  T &operator[](const int i) const
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 3);
    return values[i];
  }

  T &operator[](const int i)
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 3);
    return values[i];
  }


  /*
	 * returns if two Vector3 values of the same type are equal
	 *
	 * @param v -the vector3 to compare with this
	 * @return if values at index 0,1,2 are all equal with their counterparts
	 */
  bool operator==(const Vector3<T> &v)
  {
    if (std::is_same<T, float>::value || std::is_same<T, double>::value)
      return FLOATS_ARE_EQUAL(values[0], v.values[0]) && FLOATS_ARE_EQUAL(values[1], v.values[1]) && FLOATS_ARE_EQUAL(values[2], v.values[2]);
    else
      return v.values[0] == values[0] && v.values[1] == values[1] && v.values[2] == values[2];
  }

  // assignment operator
  Vector3<T> &operator=(const Vector3<T> &v)
  {
    values[0] = v.values[0];
    values[1] = v.values[1];
    values[2] = v.values[2];
    return *this;
  }

  // scalar addition (assignment)
  Vector3<T> &operator+=(const Vector3<T> &v)
  {
    values[0] += v.values[0];
    values[1] += v.values[1];
    values[2] += v.values[2];
    return *this;
  }

  // scalor substraction (assignment)
  Vector3<T> &operator-=(const Vector3<T> &v)
  {
    values[0] -= v.values[0];
    values[1] -= v.values[1];
    values[2] -= v.values[2];
    return *this;
  }

  // scalor multiplication (assignment)
  Vector3<T> &operator*=(const T &a)
  {
    values[0] *= a;
    values[1] *= a;
    values[2] *= a;
    return *this;
  }

  // scalor division (assignment)
  Vector3<T> &operator/=(const T &a)
  {
    values[0] /= a;
    values[1] /= a;
    values[2] /= a;
    return *this;
  }

  // addition operator
  Vector3<T> operator-(const Vector3<T> &v1)
  {
    return Vector3<T>(values[0] - v1.values[0], values[1] - v1.values[1], values[2] - v1.values[2]);
  }

  // substraction operator
  Vector3<T> operator+(const Vector3<T> &v1)
  {
    return Vector3<T>(values[0] + v1.values[0], values[1] + v1.values[1], values[2] + v1.values[2]);
  }

  // scalar product (dot product)
  T dot(const Vector3<T> &v1) const
  {
    return (values[0] * v1.values[0] + values[1] * v1.values[1] + values[2] * v1.values[2]);
  }

  // scalar product (dot product)
  T operator*(const Vector3<T> &v1) const
  {
    return (values[0] * v1.values[0] + values[1] * v1.values[1] + values[2] * v1.values[2]);
  }


  // multiplication by scalar
  Vector3<T> operator*(const T &a)
  {
    return Vector3<T>(a * values[0], a * values[1], a * values[2]);
  }

  // return the length of the vector
  T length(void) const
  {
    return sqrt(values[0] * values[0] + values[0] * values[0] + values[0] * values[0]);
  }

  // multiplication by scaler
  friend Vector3<T> operator*(const T &a, const Vector3<T> &v1)
  {
    return Vector3<T>(a * v1.values[0], a * v1.values[1], a * v1.values[2]);
  }

  // division by scalar
  Vector3<T> operator/(const T &a)
  {
    return Vector3<T>(values[0] / a, values[1] / a, values[2] / a);
  }

  // relfection
  Vector3<T> reflect(const Vector3<T> &n)
  {
    return *this - 2.0 * (*this * n) * n;
  }

  // distance with other vector where this is extemity (ie v1.distance(v2) = |v1 - v2|)
  T distance(Vector3<double> &v2)
  {
    return (sqrt((values[0] - v2[0]) * (values[0] - v2[0]) + (values[1] - v2[1]) * (values[1] - v2[1]) + (values[2] - v2[2]) * (values[2] - v2[2])));
  }


  friend std::istream &operator>>(std::istream &is, Vector3<T> &v)
  {
    is >> v.values[0] >> v.values[1] >> v.values[2];
    return is;
  }

  friend Vector3<T> operator-(const Vector3<T> &v1, const Vector3<T> &v2)
  {
    return Vector3<T>(v1.values[0] - v2.values[0], v1.values[1] - v2.values[1], v1.values[2] - v2.values[2]);
  }
};

template<typename T>
std::ostream &operator<<(std::ostream &out, const Vector3<T> &v)
{
  out << "[";
  for (int i(0); i < 2; i++)
    out << v.values[i] << ", ";
  out << v.values[2] << "]";
  return out;
}
