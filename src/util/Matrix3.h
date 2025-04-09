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
 * This file is part of QES-Plume
 *
 * GPL-3.0 License
 *
 * QES-Plume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * QES-Plume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with QES-Plume. If not, see <https://www.gnu.org/licenses/>.
 ****************************************************************************/

/** @file Matrix3
 * @brief This class handles calculation of matrix in R3
 */

#pragma once

#define FLOATS_ARE_EQUAL(x, y) (((x) - (y)) < 0.000001 && ((x) - (y)) > -0.000001)

#include <type_traits>
#include <cmath>
#include <iostream>

/* --------------------------------------------------------------------------------------
 * MATRIX 3
 * -------------------------------------------------------------------------------------- */
template<class T>
class Vector3;

/* --------------------------------------------------------------------------------------
 * MATRIX 3
 * -------------------------------------------------------------------------------------- */
template<class T>
class Matrix3;

/* --------------------------------------------------------------------------------------
 * MATRIX 3 SYMMETRIC
 * -------------------------------------------------------------------------------------- */

template<class T>
class Matrix3sym;

/* --------------------------------------------------------------------------------------
 * MATRIX 3
 * -------------------------------------------------------------------------------------- */

template<class T>
class Vector3
{

protected:
  T m_val[3];

public:
  Vector3()
    : m_val{ 0.0, 0.0, 0.0 }
  {
  }

  template<typename X>
  Vector3(X a1, X a2, X a3)
    : m_val{ a1, a2, a3 }
  {
  }

  T &operator[](const int &i)
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 3);
    return m_val[i];
  }

  bool operator==(const Vector3<T> &v) const
  {
    if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
      return FLOATS_ARE_EQUAL(m_val[0], v.m_val[0])
             && FLOATS_ARE_EQUAL(m_val[1], v.m_val[1])
             && FLOATS_ARE_EQUAL(m_val[2], v.m_val[2]);
    } else {
      return v.m_val[0] == m_val[0] && v.m_val[1] == m_val[1] && v.m_val[2] == m_val[2];
    }
  }

  // assignment operator
  Vector3<T> &operator=(const Vector3<T> &v)
  {
    if (this != &v) {
      m_val[0] = v.m_val[0];
      m_val[1] = v.m_val[1];
      m_val[2] = v.m_val[2];
    }
    return *this;
  }

  // scalar addition (assignment)
  Vector3<T> &operator+=(const Vector3<T> &v)
  {
    m_val[0] += v.m_val[0];
    m_val[1] += v.m_val[1];
    m_val[2] += v.m_val[2];
    return *this;
  }
  // scalar subtraction (assignment)
  Vector3<T> &operator-=(const Vector3<T> &v)
  {
    m_val[0] -= v.m_val[0];
    m_val[1] -= v.m_val[1];
    m_val[2] -= v.m_val[2];
    return *this;
  }
  // scalar multiplication (assignment)
  Vector3<T> &operator*=(const T &a)
  {
    m_val[0] *= a;
    m_val[1] *= a;
    m_val[2] *= a;
    return *this;
  }
  // scalar division (assignment)
  Vector3<T> &operator/=(const T &a)
  {
    m_val[0] /= a;
    m_val[1] /= a;
    m_val[2] /= a;
    return *this;
  }

  // addition operator
  Vector3<T> operator-(const Vector3<T> &v1) const
  {
    return { m_val[0] - v1.m_val[0], m_val[1] - v1.m_val[1], m_val[2] - v1.m_val[2] };
  }
  // subtraction operator
  Vector3<T> operator+(const Vector3<T> &v1) const
  {
    return { m_val[0] + v1.m_val[0], m_val[1] + v1.m_val[1], m_val[2] + v1.m_val[2] };
  }

  // scalar product (dot product)
  T operator*(const Vector3<T> &v1) const
  {
    return (m_val[0] * v1.m_val[0] + m_val[1] * v1.m_val[1] + m_val[2] * v1.m_val[2]);
  }
  // scalar product (dot product)
  T dot(const Vector3<T> &v1) const
  {
    return (m_val[0] * v1.m_val[0] + m_val[1] * v1.m_val[1] + m_val[2] * v1.m_val[2]);
  }

  // multiplication by scalar
  Vector3<T> operator*(const T &a) const
  {
    return { a * m_val[0], a * m_val[1], a * m_val[2] };
  }

  // multiplication by scalar
  friend Vector3<T> operator*(const T &a, const Vector3<T> &v1)
  {
    return { a * v1.m_val[0], a * v1.m_val[1], a * v1.m_val[2] };
  }
  // division by scalar
  Vector3<T> operator/(const T &a) const
  {
    return { m_val[0] / a, m_val[1] / a, m_val[2] / a };
  }
  // return the length of the vector
  T length() const
  {
    return sqrt(m_val[0] * m_val[0] + m_val[1] * m_val[1] + m_val[2] * m_val[2]);
  }

  // reflection v.reflect(n) = v - 2*(v*n)*n
  Vector3<T> reflect(const Vector3<T> &n) const
  {
    return *this - 2.0 * (*this * n) * n;
  }

  // distance with other vector where this is extremity (ie v1.distance(v2) = |v1 - v2|)
  double distance(Vector3<T> &v2) const
  {
    return (sqrt((m_val[0] - v2[0]) * (m_val[0] - v2[0]) + (m_val[1] - v2[1]) * (m_val[1] - v2[1]) + (m_val[2] - v2[2]) * (m_val[2] - v2[2])));
  }

  friend std::istream &operator>>(std::istream &is, Vector3<T> &v)
  {
    is >> v.m_val[0] >> v.m_val[1] >> v.m_val[2];
    return is;
  }
  friend std::ostream &operator<<(std::ostream &out, const Vector3 &v)
  {
    out << "[" << v.m_val[0] << ", " << v.m_val[1] << ", " << v.m_val[2] << "]";
    return out;
  }
};

/* --------------------------------------------------------------------------------------
 * MATRIX 3
 * -------------------------------------------------------------------------------------- */

template<class T>
class Matrix3
{

protected:
  T m_val[9];

public:
  Matrix3()
  {
    for (auto &k : m_val)
      k = (0);
  }

  template<typename X>
  Matrix3(const X a11, const X a12, const X a13, const X a21, const X a22, const X a23, const X a31, const X a32, const X a33)
    : m_val{ a11, a12, a13, a21, a22, a23, a31, a32, a33 }
  {
  }

  // accesses the value at position i,j
  T &operator[](const int i) const
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 9);
    return m_val[i];
  }

  // accesses the value at position i,j
  T &operator[](const int i)
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 9);
    return m_val[i];
  }

  // accesses the value at position i,j
  T &operator()(const int &i, const int &j) const
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 3 && j >= 0 && j < 3);
    return m_val[3 * i + j];
  }

  // returns if two Vector3 values of the same type are equal
  bool operator==(const Matrix3<T> &v) const
  {
    if (std::is_same<T, float>::value || std::is_same<T, double>::value)
      return FLOATS_ARE_EQUAL(m_val[0], v.m_val[0])
             && FLOATS_ARE_EQUAL(m_val[1], v.m_val[1])
             && FLOATS_ARE_EQUAL(m_val[2], v.m_val[2])
             && FLOATS_ARE_EQUAL(m_val[3], v.m_val[3])
             && FLOATS_ARE_EQUAL(m_val[4], v.m_val[4])
             && FLOATS_ARE_EQUAL(m_val[5], v.m_val[5])
             && FLOATS_ARE_EQUAL(m_val[6], v.m_val[6])
             && FLOATS_ARE_EQUAL(m_val[7], v.m_val[7])
             && FLOATS_ARE_EQUAL(m_val[8], v.m_val[8]);
    else
      return v.m_val[0] == m_val[0]
             && v.m_val[1] == m_val[1]
             && v.m_val[2] == m_val[2]
             && v.m_val[3] == m_val[3]
             && v.m_val[4] == m_val[4]
             && v.m_val[5] == m_val[5]
             && v.m_val[6] == m_val[6]
             && v.m_val[7] == m_val[7]
             && v.m_val[8] == m_val[8];
  }

  // assignment operator
  Matrix3<T> &operator=(const Matrix3<T> &M)
  {
    if (this != &M) {
      for (int k = 0; k < 9; k++)
        m_val[k] = M.m_val[k];
    }
    return *this;
  }

  // assignment operator (from a symmetric matrix)
  Matrix3<T> &operator=(Matrix3sym<T> &M)
  {
    m_val[0] = M[0];
    m_val[1] = M[1];
    m_val[2] = M[2];
    m_val[3] = M[1];
    m_val[4] = M[3];
    m_val[5] = M[4];
    m_val[6] = M[2];
    m_val[7] = M[4];
    m_val[8] = M[5];
    return *this;
  }

  // assignment addition operator
  Matrix3<T> &operator+=(const Matrix3<T> &M)
  {
    for (int k = 0; k < 9; k++)
      m_val[k] += M.m_val[k];
    return *this;
  }

  // assignment substation operator
  Matrix3<T> &operator-=(const Matrix3<T> &M)
  {
    for (int k = 0; k < 9; k++)
      m_val[k] -= M.m_val[k];
    return *this;
  }

  // assignment multiplication operator
  Matrix3<T> &operator*=(const T &a)
  {
    for (auto &k : m_val)
      k *= a;
    return *this;
  }

  // assignment division operator
  Matrix3<T> &operator/=(const T &a)
  {
    for (auto &k : m_val)
      k /= a;
    return *this;
  }

  // addition operator
  Matrix3<T> operator+(const Matrix3<T> &M) const
  {
    Matrix3<T> R;
    for (int k = 0; k < 9; k++)
      R.m_val[k] = m_val[k] + M.m_val[k];

    return R;
  }

  // subtraction operator
  Matrix3<T> operator-(const Matrix3<T> &M)
  {
    Matrix3<T> R;
    for (int k = 0; k < 9; k++)
      R.m_val[k] = m_val[k] - M.m_val[k];
    return R;
  }

  // matrix multiplication
  Matrix3<T> operator*(const Matrix3<T> &M) const
  {
    Matrix3<T> R(0);
    for (int i(0); i < 3; i++)// line
      for (int j(0); j < 3; j++)// column
        for (int k(0); k < 3; k++)
          R.m_val[i + 3 * j] += m_val[i + 3 * k] * M.m_val[k + 3 * j];
    return R;
  }

  // matrix multiplication (with symmetric matrix)
  Matrix3<T> operator*(const Matrix3sym<T> &M) const
  {
    Matrix3<T> R(0);
    for (int i(0); i < 3; i++)// line
      for (int j(0); j < 3; j++)// column
        for (int k(0); k < 3; k++)
          R.m_val[i + 3 * j] += m_val[i + 3 * k] * M.m_val[k + 3 * j];
    return R;
  }

  // vector matrix multiplication
  Vector3<T> operator*(const Vector3<T> &V) const
  {
    Vector3<T> R(0);
    for (int i(0); i < 3; i++)// line
      for (int k(0); k < 3; k++)
        R.m_val[i] += m_val[i + 3 * k] * V.m_val[k];
    return R;
  }

  // multiplication by scalar
  friend Matrix3<T> operator*(const T &a, const Matrix3<T> &M)
  {
    Matrix3<T> R;
    for (int k = 0; k < 9; k++)
      R.m_val[k] = a * M.m_val[k];
    return R;
  }

  // division by scalar
  Matrix3<T> operator/(const T &a) const
  {
    Matrix3<T> R;
    for (int k = 0; k < 9; k++)
      R.m_val[k] = m_val[k] / a;
    return R;
  }

  // set identity matrix
  void setIdentity()
  {
    m_val[0] = 1;
    m_val[1] = 0;
    m_val[2] = 0;
    m_val[3] = 0;
    m_val[4] = 1;
    m_val[5] = 0;
    m_val[6] = 0;
    m_val[7] = 0;
    m_val[6] = 1;
  }

  // inplace inversion
  Matrix3<T> invert()
  {
    // now calculate the determinant
    T det = m_val[0] * (m_val[4] * m_val[8] - m_val[5] * m_val[7])
            - m_val[1] * (m_val[3] * m_val[8] - m_val[5] * m_val[6])
            + m_val[2] * (m_val[1] * m_val[7] - m_val[4] * m_val[6]);

    // check for near zero value determinants
    if (std::abs(det) < 1e-10) {
      std::cout << "WARNING (Plume::invert3): matrix nearly singular" << std::endl;
      std::cout << "abs(det) = \"" << std::abs(det) << std::endl;
      det = 10e10;
    }

    // calculate the inverse (inverted matrix depends on other components of the matrix)
    Matrix3<T> Minv;
    Minv[0] = (m_val[4] * m_val[8] - m_val[5] * m_val[7]) / det;
    Minv[1] = -(m_val[1] * m_val[8] - m_val[2] * m_val[7]) / det;
    Minv[2] = (m_val[1] * m_val[5] - m_val[4] * m_val[2]) / det;

    Minv[3] = -(m_val[1] * m_val[8] - m_val[5] * m_val[6]) / det;
    Minv[4] = (m_val[0] * m_val[8] - m_val[2] * m_val[6]) / det;
    Minv[5] = -(m_val[0] * m_val[5] - m_val[2] * m_val[1]) / det;

    Minv[6] = (m_val[1] * m_val[7] - m_val[6] * m_val[4]) / det;
    Minv[7] = -(m_val[0] * m_val[7] - m_val[1] * m_val[6]) / det;
    Minv[8] = (m_val[0] * m_val[4] - m_val[1] * m_val[3]) / det;

    for (int k = 0; k < 9; k++)
      m_val[k] = Minv.m_val[k];

    return *this;
  }

  // read matrix from istream
  friend std::istream &operator>>(std::istream &is, Matrix3<T> &M)
  {
    is >> M.m_val[0] >> M.m_val[1] >> M.m_val[2] >> M.m_val[3] >> M.m_val[4] >> M.m_val[5] >> M.m_val[6] >> M.m_val[7] >> M.m_val[8];
    return is;
  }
  friend std::ostream &operator<<(std::ostream &out, const Matrix3<T> &M)
  {
    out << "[";
    out << "[" << M.m_val[0] << ", " << M.m_val[1] << ", " << M.m_val[2] << "]";
    out << "[" << M.m_val[3] << ", " << M.m_val[4] << ", " << M.m_val[5] << "]";
    out << "[" << M.m_val[6] << ", " << M.m_val[7] << ", " << M.m_val[8] << "]";
    out << "]";
    out << "[";
    return out;
  }
};

// output to ostream


/* --------------------------------------------------------------------------------------
 * MATRIX 3 SYMMETRIC
 * -------------------------------------------------------------------------------------- */

template<class T>
class Matrix3sym
{

protected:
  T m_val[6];

public:
  Matrix3sym()
  {
    for (auto &k : m_val)
      k = (0);
  }

  template<typename X>
  Matrix3sym(const X a11, const X a12, const X a13, const X a22, const X a23, const X a33)
  {
    m_val[0] = a11;
    m_val[1] = a12;
    m_val[2] = a13;
    m_val[3] = a22;
    m_val[4] = a23;
    m_val[5] = a33;
  }

  // accesses the value at position i,j
  T &operator[](const int i) const
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 6);
    return m_val[i];
  }

  // accesses the value at position i,j
  T &operator[](const int i)
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 6);
    return m_val[i];
  }

  // accesses the value at position i,j
  T &operator()(const int i, const int j) const
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 3 && j >= i && j < 3);
    return m_val[i + j + ceil(0.5 * i)];
  }

  // accesses the value at position i,j
  T &operator()(const int i, const int j)
  {
    // do a sanity check to make sure indices are OK!
    assert(i >= 0 && i < 3 && j >= 0 && j < 3);
    return m_val[i + j + ceil(0.5 * i)];
  }

  // returns if two Vector3 values of the same type are equal
  bool operator==(const Matrix3<T> &v)
  {
    if (std::is_same<T, float>::value || std::is_same<T, double>::value)
      return FLOATS_ARE_EQUAL(m_val[0], v.m_val[0])
             && FLOATS_ARE_EQUAL(m_val[1], v.m_val[1])
             && FLOATS_ARE_EQUAL(m_val[2], v.m_val[2])
             && FLOATS_ARE_EQUAL(m_val[3], v.m_val[3])
             && FLOATS_ARE_EQUAL(m_val[4], v.m_val[4])
             && FLOATS_ARE_EQUAL(m_val[5], v.m_val[5]);
    else
      return v.m_val[0] == m_val[0]
             && v.m_val[1] == m_val[1]
             && v.m_val[2] == m_val[2]
             && v.m_val[3] == m_val[3]
             && v.m_val[4] == m_val[4]
             && v.m_val[5] == m_val[5];
  }

  // assignment operator
  Matrix3sym<T> &operator=(const Matrix3sym<T> &M)
  {
    if (this != &M) {
      for (int k = 0; k < 6; k++)
        m_val[k] = M.m_val[k];
    }
    return *this;
  }

  // assignment operator
  Matrix3sym<T> &operator+=(const Matrix3sym<T> &M)
  {
    for (int k = 0; k < 6; k++)
      m_val[k] += M.m_val[k];
    return *this;
  }

  // assignment operator
  Matrix3sym<T> &operator-=(const Matrix3sym<T> &M)
  {
    for (int k = 0; k < 6; k++)
      m_val[k] -= M.m_val[k];
    return *this;
  }

  // assignment operator
  Matrix3sym<T> &operator*=(const T &a)
  {
    for (auto &k : m_val)
      k *= a;
    return *this;
  }

  // assignment operator
  Matrix3sym<T> &operator/=(const T &a)
  {
    for (auto &k : m_val)
      k /= a;
    return *this;
  }

  // addition operator
  Matrix3sym<T> operator+(const Matrix3sym<T> &M) const
  {
    Matrix3sym<T> R;
    for (int k = 0; k < 6; k++)
      R.m_val[k] = m_val[k] + M.m_val[k];
    return R;
  }

  // subtraction operator
  Matrix3sym<T> operator-(const Matrix3sym<T> &M) const
  {
    Matrix3sym<T> R;
    for (int k = 0; k < 6; k++)
      R.m_val[k] = m_val[k] - M.m_val[k];
    return R;
  }

  // multiplication by scalar
  friend Matrix3sym<T> operator*(const T &a, const Matrix3sym<T> &M)
  {
    Matrix3sym<T> R;
    for (int k = 0; k < 6; k++)
      R.m_val[k] = a * M.m_val[k];
    return R;
  }

  // division by scalar
  Matrix3sym<T> operator/(const T &a) const
  {
    Matrix3sym<T> R;
    for (int k = 0; k < 6; k++)
      R.m_val[k] = m_val[k] / a;
    return R;
  }

  void setIdentity()
  {
    m_val[0] = 1;
    m_val[1] = 0;
    m_val[2] = 0;
    m_val[3] = 1;
    m_val[4] = 0;
    m_val[5] = 1;
  }

  // calculate invariants
  Vector3<T> calcInvariants()
  {
    Vector3<T> invar;
    invar[0] = m_val[0] + m_val[3] + m_val[5];
    invar[1] = m_val[0] * m_val[3] + m_val[0] * m_val[5] + m_val[3] * m_val[5] - m_val[1] * m_val[1] - m_val[2] * m_val[2] - m_val[4] * m_val[4];
    invar[3] = m_val[0] * (m_val[3] * m_val[5] - m_val[4] * m_val[4])
               - m_val[1] * (m_val[1] * m_val[5] - m_val[4] * m_val[2])
               + m_val[2] * (m_val[1] * m_val[4] - m_val[3] * m_val[2]);
    return invar;
  }

  // make realizable
  Matrix3sym<T> &makeRealizable(const T &invarianceTol)
  {
    // first calculate the invariants and see if they are already realizable
    Vector3<T> invar = this->calcInvariants();

    if (invar[0] > invarianceTol && invar[1] > invarianceTol && invar[2] > invarianceTol) {
      return *this;// is already realizable -> what should be return here??? -Pete
    }

    // since tau is not already realizable, need to make it realizable
    // start by making a guess of ks, the sub-filter scale tke
    // I keep wondering if we can use the input TURB->tke for this or if we should leave it as is
    T b = 4.0 / 3.0 * invar[0];
    T c = invar[1];
    T ks = 1.01 * (-b + std::sqrt(b * b - 16.0 / 3.0 * c)) / (8.0 / 3.0);

    // if the initial guess is bad, use the straight-up invar_xx value
    if (ks < invarianceTol || isnan(ks)) {
      ks = 0.5 * std::abs(invar[0]);// also 0.5*abs(invar_xx)
    }

    Matrix3sym<T> M;
    M[0] = m_val[0] + 2.0 / 3.0 * ks;
    M[1] = m_val[1];
    M[2] = m_val[2];
    M[3] = m_val[3] + 2.0 / 3.0 * ks;
    M[4] = m_val[4];
    M[5] = m_val[5] + 2.0 / 3.0 * ks;

    invar = M.calcInvariants();

    // now adjust the diagonals by 0.05% of the trace, till M is realizable
    int iter = 0;
    while ((invar[0] < invarianceTol || invar[1] < invarianceTol || invar[2] < invarianceTol) && iter < 1000) {
      iter = iter + 1;

      // increase trace by 5%
      ks = ks * 1.050;

      // note that the right hand side is not M, to force to only increase by increasing ks
      M[0] = m_val[0] + 2.0 / 3.0 * ks;
      M[3] = m_val[3] + 2.0 / 3.0 * ks;
      M[5] = m_val[5] + 2.0 / 3.0 * ks;

      invar = M.calcInvariants();
    }

    if (iter == 999) {
      std::cout << "WARNING (Plume::makeRealizable): unable to make stress tensor realizable.";
    }

    for (int k = 0; k < 6; k++)
      m_val[k] = M.m_val[k];

    return *this;
  }

  // read from istream
  friend std::istream &operator>>(std::istream &is, Matrix3sym<T> &M)
  {
    is >> M.m_val[0] >> M.m_val[1] >> M.m_val[2] >> M.m_val[3] >> M.m_val[4] >> M.m_val[5];
    return is;
  }

  friend std::ostream &operator<<(std::ostream &out, const Matrix3sym<T> &M)
  {
    out << "[";
    out << "[" << M.m_val[0] << ", " << M.m_val[1] << ", " << M.m_val[2] << "]";
    out << "[" << M.m_val[1] << ", " << M.m_val[3] << ", " << M.m_val[4] << "]";
    out << "[" << M.m_val[2] << ", " << M.m_val[4] << ", " << M.m_val[5] << "]";
    out << "]";
    return out;
  }
};
